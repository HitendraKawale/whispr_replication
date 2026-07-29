"""Step 2 — the DFT and STFT. See notes/02-dft-and-stft.md.

    uv run python scripts/02_stft.py
"""

from __future__ import annotations

import time

import numpy as np
import torch

from whispr import audio, dft
from whispr.plotting import ACCENT, ACCENT2, INK, MUTED, plt, save, use_style

SR = 16_000


def synth_utterance(duration: float = 2.0, sr: int = SR) -> torch.Tensor:
    """A crude synthetic 'utterance': a pitch-varying buzz through moving formants.

    Not speech, but it has the two features that make speech spectrograms
    interesting — harmonic stacks that drift in pitch, and formant bands that
    move independently of the pitch. Plus two plosive bursts.
    """
    n = int(duration * sr)
    t = torch.arange(n, dtype=torch.float32) / sr

    # Pitch contour: falls from 150 Hz to 100 Hz, the way a statement does.
    f0 = 150 - 50 * (t / duration)
    phase = 2 * np.pi * torch.cumsum(f0, dim=0) / sr
    source = sum((1.0 / h) * torch.sin(h * phase) for h in range(1, 30))

    # Two formants that slide between vowel targets, e.g. /a/ -> /i/.
    out = 0.3 * source
    for start, end, gain in ((700, 300, 1.0), (1200, 2400, 0.7)):
        fmt = start + (end - start) * (t / duration)
        fphase = 2 * np.pi * torch.cumsum(fmt, dim=0) / sr
        out = out + gain * 0.25 * torch.sin(fphase) * (0.5 + 0.5 * source / source.abs().max())

    # Broadband bursts (plosives) — sharp in time, spread in frequency.
    # Placed as fractions of the duration so they land inside any length.
    blen = int(0.012 * sr)
    for frac in (0.28, 0.68):
        i = int(frac * n)
        if i + blen <= n:
            out[i : i + blen] += 1.2 * torch.randn(blen) * torch.hann_window(blen)

    return (out / out.abs().max()).to(torch.float32)


def fig_dft_is_a_matrix() -> None:
    """Show the DFT matrix itself — the probes it correlates against."""
    n = 64
    w = dft.dft_matrix(n)
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.0))
    axes[0].imshow(w.real.numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    axes[0].set_title("Re(W) — cosine probes", loc="left")
    axes[1].imshow(w.imag.numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    axes[1].set_title("Im(W) — sine probes", loc="left")
    for ax in axes[:2]:
        ax.set_xlabel("time index t")
        ax.set_ylabel("frequency index k")

    for k, color in ((1, ACCENT), (4, ACCENT2), (16, MUTED)):
        axes[2].plot(w[k].real.numpy(), color=color, lw=1.2, label=f"row k={k}")
    axes[2].set_title("Individual rows are just sinusoids", loc="left")
    axes[2].set_xlabel("time index t")
    axes[2].legend(fontsize=7)
    fig.suptitle("The DFT is a matrix multiply against sinusoidal probes",
                 x=0.02, ha="left", weight="bold")
    fig.tight_layout()
    save(fig, "02_dft_matrix.png")


def fig_leakage() -> None:
    """Rectangular vs Hann on a tone that sits between bins."""
    n_fft = 400
    x = audio.sine(1020.0, n_fft / SR, SR)  # 1020 Hz: between the 1000/1040 bins
    w = dft.hann_window(n_fft)

    rect = dft.rfft_naive(x).abs()
    hann = dft.rfft_naive(x * w).abs()
    freqs = np.arange(len(rect)) * SR / n_fft

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.1))
    axes[0].plot(x.numpy(), color=MUTED, lw=0.8, label="signal")
    axes[0].plot((x * w).numpy(), color=ACCENT, lw=0.9, label="windowed")
    axes[0].plot(w.numpy(), color=ACCENT2, lw=1.2, ls="--", label="Hann window")
    axes[0].set_title("The window tapers the frame edges to zero", loc="left")
    axes[0].set_xlabel("sample")
    axes[0].legend(fontsize=7)

    to_db = lambda v: 20 * np.log10(np.maximum(v.numpy(), 1e-12) / v.max().item())
    axes[1].plot(freqs, to_db(rect), color=MUTED, lw=1.0, label="rectangular (no window)")
    axes[1].plot(freqs, to_db(hann), color=ACCENT, lw=1.2, label="Hann")
    axes[1].set_xlim(0, 4000)
    axes[1].set_ylim(-90, 5)
    axes[1].set_title("Spectral leakage: a 1020 Hz tone, in dB", loc="left")
    axes[1].set_xlabel("frequency (Hz)")
    axes[1].set_ylabel("dB below peak")
    axes[1].legend(fontsize=7)
    axes[1].grid(True, lw=0.5)
    fig.suptitle("Why we window: without one, one tone contaminates every bin",
                 x=0.02, ha="left", weight="bold")
    fig.tight_layout()
    save(fig, "02_leakage.png")


def fig_resolution_tradeoff() -> None:
    """The uncertainty principle, as four pictures of the same signal."""
    x = synth_utterance(2.0)
    fig, axes = plt.subplots(1, 4, figsize=(12.5, 3.2), sharey=True)
    for ax, n_fft in zip(axes, (128, 400, 1024, 4096)):
        spec = torch.stft(
            x, n_fft=n_fft, hop_length=n_fft // 4,
            window=torch.hann_window(n_fft), center=True, return_complex=True,
        ).abs()
        logspec = 20 * torch.log10(spec.clamp(min=1e-8)).numpy()
        ax.imshow(
            logspec, origin="lower", aspect="auto", cmap="magma",
            extent=[0, 2.0, 0, SR / 2], vmin=logspec.max() - 70, vmax=logspec.max(),
        )
        ax.set_ylim(0, 4000)
        ax.set_xlabel("time (s)")
        label = "  ← Whisper" if n_fft == 400 else ""
        ax.set_title(
            f"n_fft={n_fft}{label}\n{1000*n_fft/SR:.0f} ms window · {SR/n_fft:.0f} Hz bins",
            loc="left", color=ACCENT if n_fft == 400 else INK,
        )
    axes[0].set_ylabel("frequency (Hz)")
    fig.suptitle(
        "Time/frequency uncertainty: short windows resolve the two bursts, "
        "long windows resolve the harmonics. Never both.",
        x=0.02, ha="left", weight="bold",
    )
    fig.tight_layout()
    save(fig, "02_resolution_tradeoff.png")


def fig_ours_vs_torch() -> None:
    """Our from-scratch STFT against torch's, and the difference."""
    x = synth_utterance(1.0)
    ours = dft.stft_naive(x, 400, 160, window=dft.hann_window(400), center=True)
    theirs = torch.stft(x, n_fft=400, hop_length=160, window=torch.hann_window(400),
                        center=True, return_complex=True)
    diff = (ours - theirs).abs()

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.0), sharey=True)
    for ax, data, title in (
        (axes[0], ours.abs(), "ours: pad → frame → window → matmul"),
        (axes[1], theirs.abs(), "torch.stft (FFT)"),
    ):
        d = 20 * torch.log10(data.clamp(min=1e-8)).numpy()
        ax.imshow(d, origin="lower", aspect="auto", cmap="magma",
                  extent=[0, 1.0, 0, SR / 2], vmin=d.max() - 70, vmax=d.max())
        ax.set_title(title, loc="left")
        ax.set_xlabel("time (s)")
    im = axes[2].imshow(diff.numpy(), origin="lower", aspect="auto", cmap="viridis",
                        extent=[0, 1.0, 0, SR / 2])
    axes[2].set_title(f"|difference| — max {diff.max():.2e}", loc="left")
    axes[2].set_xlabel("time (s)")
    fig.colorbar(im, ax=axes[2], fraction=0.046)
    axes[0].set_ylabel("frequency (Hz)")
    fig.suptitle("Same answer, different algorithm — this is what lets us use torch.stft",
                 x=0.02, ha="left", weight="bold")
    fig.tight_layout()
    save(fig, "02_ours_vs_torch.png")


def report() -> None:
    x = synth_utterance(1.0)

    t0 = time.perf_counter()
    dft.stft_naive(x, 400, 160, window=dft.hann_window(400))
    naive_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    torch.stft(x, n_fft=400, hop_length=160, window=torch.hann_window(400),
               center=True, return_complex=True)
    fft_s = time.perf_counter() - t0

    print("\nCost of readability (1 s of audio, n_fft=400, hop=160):")
    print(f"  naive O(N^2) matmul : {naive_s*1000:8.2f} ms")
    print(f"  torch.stft (FFT)    : {fft_s*1000:8.2f} ms")
    print(f"  speedup             : {naive_s/fft_s:8.1f}x")

    print("\nWhisper's frontend arithmetic:")
    print(f"  n_fft=400 @ 16 kHz  -> {400//2+1} frequency bins, {SR/400:.0f} Hz apart")
    print(f"  window              -> {1000*400/SR:.0f} ms")
    print(f"  hop=160             -> {1000*160/SR:.0f} ms")
    spec = torch.stft(torch.zeros(audio.N_SAMPLES), n_fft=400, hop_length=160,
                      window=torch.hann_window(400), center=True, return_complex=True)
    print(f"  480,000 samples     -> {spec.shape[1]} frames, drop last -> {spec.shape[1]-1}")
    print(f"  conv stem stride 2  -> {(spec.shape[1]-1)//2} encoder positions")


if __name__ == "__main__":
    use_style()
    print("Step 2 — the DFT and the STFT\n")
    fig_dft_is_a_matrix()
    fig_leakage()
    fig_resolution_tradeoff()
    fig_ours_vs_torch()
    report()
