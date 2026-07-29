"""Step 3 — the mel filterbank and Whisper's frontend. See notes/03-mel-and-frontend.md.

    uv run python scripts/03_frontend.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module

from whispr import audio, mel
from whispr.plotting import ACCENT, ACCENT2, INK, MUTED, plt, save, use_style

synth_utterance = import_module("02_stft").synth_utterance
REFERENCE_NPZ = Path(__file__).resolve().parent.parent / "assets" / "whisper_mel_filters.npz"
SR = 16_000


def fig_mel_scale() -> None:
    hz = np.linspace(0, 8000, 2000)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.2))

    axes[0].plot(hz, mel.hz_to_mel(hz), color=ACCENT, lw=1.8, label="Slaney (Whisper)")
    axes[0].plot(hz, 2595 * np.log10(1 + hz / 700) * (mel.hz_to_mel(8000) / (2595 * np.log10(1 + 8000 / 700))),
                 color=MUTED, lw=1.2, ls="--", label="HTK (rescaled, for shape)")
    axes[0].axvline(1000, color=ACCENT2, lw=0.9, ls=":")
    axes[0].text(1150, 5, "1 kHz: linear → log", fontsize=8, color=ACCENT2)
    axes[0].set_xlabel("frequency (Hz)")
    axes[0].set_ylabel("mel")
    axes[0].set_title("The mel scale: linear below 1 kHz, log above", loc="left")
    axes[0].legend(fontsize=7)
    axes[0].grid(True, lw=0.5)

    # Where equal-width mel bands land in Hz.
    edges = mel.mel_to_hz(np.linspace(0, mel.hz_to_mel(8000), 82))
    axes[1].plot(range(81), np.diff(edges), color=ACCENT, lw=1.8)
    axes[1].axhline(SR / 400, color=ACCENT2, lw=1.2, ls="--",
                    label=f"DFT bin spacing = {SR/400:.0f} Hz")
    axes[1].set_xlabel("mel band index")
    axes[1].set_ylabel("band width (Hz)")
    axes[1].set_title("Bands below index ~20 are narrower than one DFT bin", loc="left")
    axes[1].legend(fontsize=7)
    axes[1].grid(True, lw=0.5)

    fig.suptitle("Why mel: capacity goes where hearing has resolution",
                 x=0.02, ha="left", weight="bold")
    fig.tight_layout()
    save(fig, "03_mel_scale.png")


def fig_filterbank() -> None:
    fb = mel.mel_filterbank()
    freqs = np.linspace(0, SR / 2, fb.shape[1])

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.2))
    for i in range(0, 80, 2):
        axes[0].plot(freqs, fb[i].numpy(), lw=0.8,
                     color=plt.cm.magma(i / 80 * 0.85))
    axes[0].set_xlabel("frequency (Hz)")
    axes[0].set_ylabel("weight")
    axes[0].set_title("80 triangular filters (every 2nd shown)\n"
                      "Slaney-normalised: equal area, not equal peak", loc="left")
    axes[0].set_xlim(0, 8000)

    im = axes[1].imshow(fb.numpy(), origin="lower", aspect="auto", cmap="magma",
                        extent=[0, 8000, 0, 80])
    axes[1].set_xlabel("frequency (Hz)")
    axes[1].set_ylabel("mel channel")
    axes[1].set_title("The same thing as a matrix (80 x 201)", loc="left")
    fig.colorbar(im, ax=axes[1], fraction=0.046)

    fig.suptitle("The mel filterbank is an 80x201 matrix — that's the whole idea",
                 x=0.02, ha="left", weight="bold")
    fig.tight_layout()
    save(fig, "03_filterbank.png")


def fig_verification() -> None:
    """Our filterbank vs the array shipped inside openai/whisper."""
    ours = mel.mel_filterbank()
    ref = mel.load_reference_filters(REFERENCE_NPZ)
    diff = (ours - ref).abs()

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.0))
    for ax, data, title in (
        (axes[0], ours, "ours (whispr/mel.py)"),
        (axes[1], ref, "openai/whisper mel_filters.npz"),
    ):
        ax.imshow(data.numpy(), origin="lower", aspect="auto", cmap="magma",
                  extent=[0, 8000, 0, 80])
        ax.set_title(title, loc="left")
        ax.set_xlabel("frequency (Hz)")
    axes[0].set_ylabel("mel channel")

    im = axes[2].imshow(diff.numpy(), origin="lower", aspect="auto", cmap="viridis",
                        extent=[0, 8000, 0, 80])
    axes[2].set_title(f"|difference| — max {diff.max():.1e}\n(float32 epsilon is 1.2e-7)",
                      loc="left", color=ACCENT2)
    axes[2].set_xlabel("frequency (Hz)")
    fig.colorbar(im, ax=axes[2], fraction=0.046)

    fig.suptitle("Verification: identical to OpenAI's filterbank",
                 x=0.02, ha="left", weight="bold")
    fig.tight_layout()
    save(fig, "03_verification.png")


def fig_pipeline() -> None:
    """Every stage of the frontend on one signal."""
    wav = synth_utterance(3.0)

    stft = torch.stft(wav, 400, 160, window=torch.hann_window(400),
                      center=True, return_complex=True)
    power = stft[..., :-1].abs() ** 2
    melspec = mel._cached_filterbank(SR, 400, 80) @ power
    logspec = torch.clamp(melspec, min=1e-10).log10()
    final = mel.log_mel_spectrogram(wav, pad_to=None)

    fig, axes = plt.subplots(1, 4, figsize=(13.0, 3.2))
    dur = len(wav) / SR

    d = 10 * torch.log10(power.clamp(min=1e-10)).numpy()
    axes[0].imshow(d, origin="lower", aspect="auto", cmap="magma",
                   extent=[0, dur, 0, 8000], vmin=d.max() - 80, vmax=d.max())
    axes[0].set_title("1. power spectrum\n(201 linear bins)", loc="left")
    axes[0].set_ylabel("frequency (Hz)")

    axes[1].imshow(melspec.numpy(), origin="lower", aspect="auto", cmap="magma",
                   extent=[0, dur, 0, 80])
    axes[1].set_title("2. after mel matmul\n(80 channels, linear scale)", loc="left")
    axes[1].set_ylabel("mel channel")

    axes[2].imshow(logspec.numpy(), origin="lower", aspect="auto", cmap="magma",
                   extent=[0, dur, 0, 80])
    axes[2].set_title("3. log10\n(fricatives become visible)", loc="left")

    axes[3].imshow(final.numpy(), origin="lower", aspect="auto", cmap="magma",
                   extent=[0, dur, 0, 80])
    axes[3].set_title(f"4. floor + rescale — THE INPUT\nrange [{final.min():.2f}, {final.max():.2f}]",
                      loc="left", color=ACCENT)
    for ax in axes:
        ax.set_xlabel("time (s)")

    fig.suptitle("The frontend, stage by stage: 480,000 numbers become 80 x 3000",
                 x=0.02, ha="left", weight="bold")
    fig.tight_layout()
    save(fig, "03_frontend_pipeline.png")


def fig_two_floors() -> None:
    """The competing absolute and relative floors (notes §4b)."""
    wav = audio.chirp(100, 6000, 10.0, amplitude=0.5)
    loud = mel.log_mel_spectrogram(wav)
    quiet = mel.log_mel_spectrogram(wav * 0.01)
    diff = loud - quiet

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.0))
    for ax, data, title in (
        (axes[0], loud, "amplitude 0.5\nrelative floor (peak-8) binds"),
        (axes[1], quiet, "amplitude 0.005\nabsolute clamp (1e-10) binds"),
    ):
        ax.imshow(data.numpy()[:, :1200], origin="lower", aspect="auto", cmap="magma",
                  vmin=-1.5, vmax=1.7)
        ax.set_title(title, loc="left")
        ax.set_xlabel("frame")
    axes[0].set_ylabel("mel channel")

    im = axes[2].imshow(diff.numpy()[:, :1200], origin="lower", aspect="auto",
                        cmap="coolwarm")
    axes[2].set_title("difference: 1.00 in signal, 0.9524 in silence\n"
                      "-> NOT a constant offset", loc="left", color=ACCENT)
    axes[2].set_xlabel("frame")
    fig.colorbar(im, ax=axes[2], fraction=0.046)

    fig.suptitle("The frontend is not loudness invariant, and its two floors compete",
                 x=0.02, ha="left", weight="bold")
    fig.tight_layout()
    save(fig, "03_two_floors.png")


def report() -> None:
    ours = mel.mel_filterbank()
    ref = mel.load_reference_filters(REFERENCE_NPZ)
    print("\nVerification against openai/whisper's shipped filterbank:")
    print(f"  shape           : {tuple(ours.shape)} vs {tuple(ref.shape)}")
    print(f"  max abs diff    : {(ours - ref).abs().max().item():.3e}")
    print(f"  allclose(1e-6)  : {torch.allclose(ours, ref, atol=1e-6)}")

    widths = [(row > 0).sum().item() for row in ours]
    print("\nFilter widths in DFT bins (notes 4b — the low bands are undersampled):")
    print(f"  filters  0-19   : {min(widths[:20])}-{max(widths[:20])} bins")
    print(f"  filters 70-79   : {min(widths[-10:])}-{max(widths[-10:])} bins")
    disjoint = [i for i in range(79) if not ((ours[i] > 0) & (ours[i + 1] > 0)).any()]
    print(f"  disjoint pairs  : {disjoint}")

    wav = synth_utterance(3.0)
    spec = mel.log_mel_spectrogram(wav)
    print("\nThe encoder's input tensor:")
    print(f"  shape           : {tuple(spec.shape)}")
    print(f"  range           : [{spec.min():.3f}, {spec.max():.3f}]  span {spec.max()-spec.min():.4f}")
    print("\nWhat actually got compressed (two different numbers — don't conflate them):")
    print(f"  total values    : {audio.N_SAMPLES:,} -> {spec.numel():,} "
          f"(only {audio.N_SAMPLES/spec.numel():.1f}x — we kept most of the information)")
    print(f"  SEQUENCE LENGTH : {audio.N_SAMPLES:,} -> {spec.shape[-1]:,} "
          f"({audio.N_SAMPLES//spec.shape[-1]}x) -> 1,500 after the conv stem")
    print(f"  attention cost  : {audio.N_SAMPLES**2:.1e} -> {1500**2:.1e} scores "
          f"({(audio.N_SAMPLES**2)/(1500**2):,.0f}x cheaper)")
    print("  The frontend is a *sequence-length* reduction. That is the point.")


if __name__ == "__main__":
    use_style()
    print("Step 3 — the mel filterbank and Whisper's frontend\n")
    fig_mel_scale()
    fig_filterbank()
    fig_verification()
    fig_pipeline()
    fig_two_floors()
    report()
