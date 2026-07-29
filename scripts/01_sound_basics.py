"""Step 1 — Sound as numbers. See notes/01-sound-and-sampling.md.

Produces figures/01_*.png and audible demos in assets/.

    uv run python scripts/01_sound_basics.py
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from whispr import audio
from whispr.plotting import ACCENT, ACCENT2, MUTED, plt, save, use_style

ASSETS = Path(__file__).resolve().parent.parent / "assets"


def fig_waveform_zoom() -> None:
    """A vowel-like tone at three zoom levels: the same array, three stories."""
    sr = 16_000
    # A crude synthetic vowel: a 120 Hz glottal pulse train shaped by formants.
    # Real vowels are exactly this — a buzzy source filtered by the vocal tract.
    f0 = 120.0
    wav = sum(
        (1.0 / (i + 1)) * audio.sine(f0 * (i + 1), 0.5, sr, amplitude=0.5)
        for i in range(1, 25)
    )
    # Emphasise the two lowest formants of an /a/-ish sound.
    for formant, gain in ((700.0, 0.9), (1220.0, 0.6)):
        wav = wav + gain * audio.sine(formant, 0.5, sr, amplitude=0.18)
    wav = wav / wav.abs().max()

    fig, axes = plt.subplots(3, 1, figsize=(7.5, 5.6))
    spans = [
        (0.5, "0.5 s — 8,000 samples: you see an envelope, no structure"),
        (0.05, "50 ms — 800 samples: the pitch period appears (~120 Hz)"),
        (0.008, "8 ms — 128 samples: individual oscillations, the actual data"),
    ]
    for ax, (span, title) in zip(axes, spans):
        n = int(span * sr)
        t = np.arange(n) / sr * 1000  # ms
        seg = wav[:n].numpy()
        if span <= 0.008:
            ax.plot(t, seg, color=ACCENT, lw=0.9, marker="o", ms=2.5, mfc="white", mew=0.6)
        else:
            ax.plot(t, seg, color=ACCENT, lw=0.7)
        ax.set_title(title, loc="left")
        ax.set_xlabel("time (ms)")
        ax.set_ylabel("pressure")
        ax.margins(x=0)
    fig.suptitle("One signal, three zoom levels", x=0.02, ha="left", weight="bold")
    fig.tight_layout()
    save(fig, "01_waveform_zoom.png")


def fig_nyquist() -> None:
    """Sampling a 3 kHz tone above and below Nyquist."""
    true_f = 3000.0
    dense_sr = 200_000  # stand-in for the continuous signal
    dur = 0.002

    fig, axes = plt.subplots(1, 3, figsize=(9.5, 2.9), sharey=True)
    for ax, sr in zip(axes, [16_000, 8_000, 4_000]):
        t_dense = np.arange(int(dur * dense_sr)) / dense_sr
        ax.plot(t_dense * 1000, np.sin(2 * math.pi * true_f * t_dense),
                color=MUTED, lw=1.0, label="true 3 kHz")

        n = int(dur * sr)
        t = np.arange(n) / sr
        s = np.sin(2 * math.pi * true_f * t)
        ax.plot(t * 1000, s, color=ACCENT, lw=1.0, marker="o", ms=4,
                mfc="white", mew=1.0, label=f"sampled @ {sr//1000} kHz")

        nyq = sr / 2
        ok = true_f < nyq
        ax.set_title(
            f"sr={sr//1000} kHz · Nyquist {nyq/1000:g} kHz\n"
            + ("faithful" if ok else f"ALIASED → appears as {audio.alias_frequency(true_f, sr):.0f} Hz"),
            loc="left",
            color=ACCENT2 if ok else ACCENT,
        )
        ax.set_xlabel("time (ms)")
        ax.margins(x=0)
    axes[0].set_ylabel("amplitude")
    axes[0].legend(loc="lower right", fontsize=7)
    fig.suptitle("A 3 kHz tone sampled three ways", x=0.02, ha="left", weight="bold")
    fig.tight_layout()
    save(fig, "01_nyquist.png")


def fig_aliasing_foldback() -> None:
    """The fold-back map: what frequency you *think* you're seeing."""
    sr = 16_000
    f = np.linspace(0, 3 * sr, 2000)
    apparent = np.abs(f - sr * np.round(f / sr))

    fig, ax = plt.subplots(figsize=(7.5, 3.0))
    ax.plot(f / 1000, apparent / 1000, color=ACCENT, lw=1.6)
    ax.plot([0, sr / 2 / 1000], [0, sr / 2 / 1000], color=ACCENT2, lw=1.6,
            label="faithful region (f < Nyquist)")
    ax.axvline(sr / 2 / 1000, color=MUTED, ls="--", lw=0.9)
    ax.text(sr / 2 / 1000 + 0.3, 6.6, "Nyquist = 8 kHz", fontsize=8, color=MUTED)
    ax.set_xlabel("true frequency (kHz)")
    ax.set_ylabel("apparent frequency (kHz)")
    ax.set_title("Above Nyquist, frequencies fold back and masquerade as lower ones",
                 loc="left")
    ax.legend(fontsize=8)
    ax.grid(True, lw=0.5)
    ax.margins(x=0)
    fig.tight_layout()
    save(fig, "01_aliasing_foldback.png")


def fig_resample_right_and_wrong() -> None:
    """Naive decimation vs. a proper low-pass resampler, on a sweep."""
    import torchaudio

    sr, target = 16_000, 4_000
    sweep = audio.chirp(100, 7800, 2.0, sr)

    naive = sweep[::4]  # WRONG: no anti-alias filter
    proper = torchaudio.transforms.Resample(sr, target)(sweep)

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.0), sharey=True)
    for ax, sig, title in (
        (axes[0], naive, "Naive `wav[::4]` — energy folds back down"),
        (axes[1], proper, "torchaudio Resample — low-pass first, then decimate"),
    ):
        ax.specgram(sig.numpy(), NFFT=256, Fs=target, noverlap=192, cmap="magma")
        ax.set_title(title, loc="left")
        ax.set_xlabel("time (s)")
    axes[0].set_ylabel("frequency (Hz)")
    fig.suptitle("The same 100→7800 Hz sweep downsampled to 4 kHz two ways",
                 x=0.02, ha="left", weight="bold")
    fig.tight_layout()
    save(fig, "01_resampling.png")


def fig_quantization() -> None:
    sr = 16_000
    wav = audio.sine(440, 0.01, sr, amplitude=0.8)
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 2.7), sharey=True)
    for ax, bits in zip(axes, [3, 5, 16]):
        q = audio.quantize(wav, bits)
        err = (q - wav).abs().mean().item()
        ax.plot(wav.numpy(), color=MUTED, lw=1.2, label="original")
        ax.step(range(len(q)), q.numpy(), color=ACCENT, lw=1.0, where="mid",
                label=f"{bits}-bit")
        ax.set_title(f"{bits}-bit · mean abs error {err:.2e}", loc="left")
        ax.set_xlabel("sample")
    axes[0].set_ylabel("amplitude")
    axes[0].legend(fontsize=7, loc="lower left")
    fig.suptitle("Quantization: 16-bit error is ~1e-5, far below anything that matters",
                 x=0.02, ha="left", weight="bold")
    fig.tight_layout()
    save(fig, "01_quantization.png")


def write_audible_demos() -> None:
    """Files you can actually listen to — aliasing is much clearer by ear."""
    ASSETS.mkdir(parents=True, exist_ok=True)
    sr = 16_000

    # A clean sweep, correctly band-limited: rises and keeps rising.
    clean = audio.chirp(200, 7000, 4.0, sr, amplitude=0.4)
    audio.save_audio(ASSETS / "aliasing_clean_16k.wav", clean, sr)

    # The same sweep sampled at 8 kHz *without* filtering: it rises, hits 4 kHz,
    # then audibly turns around and descends. That U-turn is aliasing.
    t = torch.arange(int(4.0 * 8000), dtype=torch.float32) / 8000
    k = (7000 - 200) / 4.0
    folded = 0.4 * torch.sin(2 * math.pi * (200 * t + 0.5 * k * t**2))
    audio.save_audio(ASSETS / "aliasing_folded_8k.wav", folded, 8000)

    print("  wrote assets/aliasing_clean_16k.wav  (sweep rises smoothly)")
    print("  wrote assets/aliasing_folded_8k.wav  (sweep turns around — aliasing)")


def report() -> None:
    sr = 16_000
    print("\nNumbers that motivate the whole frontend:")
    print(f"  30 s @ {sr} Hz            = {audio.N_SAMPLES:,} samples")
    print(f"  self-attention over that  = {audio.N_SAMPLES**2:.2e} scores per head per layer")
    print(f"  after Whisper's frontend  = 3,000 frames  ({audio.N_SAMPLES // 3000}x reduction)")
    print(f"  after the conv stem       = 1,500 frames  ({audio.N_SAMPLES // 1500}x reduction)")
    print(f"  attention over 1,500      = {1500**2:.2e}  — {(audio.N_SAMPLES**2)/(1500**2):.0f}x cheaper")
    print("\nAliasing check (closed form):")
    for f in (1000, 7000, 9000, 15000, 17000):
        print(f"  {f:>6} Hz sampled at 16 kHz appears at {audio.alias_frequency(f, sr):>6.0f} Hz")


if __name__ == "__main__":
    use_style()
    print("Step 1 — sound as numbers\n")
    fig_waveform_zoom()
    fig_nyquist()
    fig_aliasing_foldback()
    fig_resample_right_and_wrong()
    fig_quantization()
    write_audible_demos()
    report()
