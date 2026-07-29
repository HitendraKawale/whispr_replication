"""The DFT and STFT, implemented from first principles.

Nothing here is used in training — `torch.stft` is, because it's an FFT and
ours is an O(N²) matrix multiply. This module exists so that `torch.stft` is
not a black box, and the tests assert our version and PyTorch's agree to
floating-point precision.
"""

from __future__ import annotations

import math

import torch


def dft_matrix(n: int, dtype: torch.dtype = torch.complex64) -> torch.Tensor:
    """The N×N DFT matrix W, where W[k, t] = exp(-2πi·kt/N).

    The DFT is *just a change of basis*. Row k is a complex sinusoid at
    frequency k; multiplying by it asks "how much of my signal looks like this
    sinusoid, and at what phase?" The answer is one complex number: magnitude
    is "how much", angle is "at what phase".

    Built in float64: k*t reaches N² and the angle wraps many times, so single
    precision loses several digits before the cast back down.
    """
    k = torch.arange(n, dtype=torch.float64).unsqueeze(1)  # (n, 1) frequency index
    t = torch.arange(n, dtype=torch.float64).unsqueeze(0)  # (1, n) time index
    angle = -2 * math.pi * (k * t % n) / n  # mod n first: exp is n-periodic
    return torch.complex(torch.cos(angle), torch.sin(angle)).to(dtype)


def dft(x: torch.Tensor) -> torch.Tensor:
    """Naive O(N²) DFT by explicit matrix multiply. Correct, and slow.

    Accumulates in complex128 and returns complex64, so the result matches an
    FFT to ~1e-6 rather than ~1e-3.
    """
    n = x.shape[-1]
    w = dft_matrix(n, dtype=torch.complex128).to(x.device)
    out = torch.einsum("kt,...t->...k", w, x.to(torch.complex128))
    return out.to(torch.complex64)


def rfft_naive(x: torch.Tensor) -> torch.Tensor:
    """Naive DFT keeping only the non-redundant half.

    For real input, X[N-k] = conj(X[k]) — the negative frequencies carry no new
    information. So we keep bins 0..N/2, which is N//2 + 1 of them. Every real
    spectrogram in this repo has that many rows, and n_fft=400 is why the
    Whisper frontend has 201 frequency bins before the mel projection.
    """
    return dft(x)[..., : x.shape[-1] // 2 + 1]


def bin_to_hz(bin_index: torch.Tensor | int, n_fft: int, sample_rate: int) -> float:
    """Which physical frequency does DFT bin k correspond to?

    Bin k sits at k · sr/n_fft Hz. The spacing sr/n_fft is the *frequency
    resolution*: with n_fft=400 at 16 kHz, bins are 40 Hz apart.
    """
    return bin_index * sample_rate / n_fft


def hann_window(n: int, periodic: bool = True) -> torch.Tensor:
    """The Hann window: 0.5·(1 − cos(2πt/N)).

    `periodic=True` (torch's default, and what STFT wants) divides by N;
    `periodic=False` (symmetric, what you want for filter design) divides by
    N−1. Mixing them up is a classic source of tiny mismatches against
    reference implementations.
    """
    denom = n if periodic else n - 1
    t = torch.arange(n, dtype=torch.float32)
    return 0.5 * (1 - torch.cos(2 * math.pi * t / denom))


def frame(x: torch.Tensor, frame_length: int, hop_length: int) -> torch.Tensor:
    """Slice a signal into overlapping frames: (..., num_frames, frame_length).

    Uses `unfold`, which returns a *view* — no copy, no extra memory.
    """
    return x.unfold(dimension=-1, size=frame_length, step=hop_length)


def stft_naive(
    x: torch.Tensor,
    n_fft: int,
    hop_length: int,
    window: torch.Tensor | None = None,
    center: bool = True,
) -> torch.Tensor:
    """STFT from scratch: pad → frame → window → DFT.

    Returns a complex tensor of shape (n_fft//2 + 1, num_frames), matching
    `torch.stft(..., return_complex=True)`.

    `center=True` reflect-pads by n_fft//2 on both sides so that frame `i` is
    *centred* on sample `i·hop`. This is what torch and librosa do by default,
    and it's why a 480,000-sample input yields 3001 frames rather than 2999.
    Whisper then drops the final frame to land on exactly 3000.
    """
    if window is None:
        window = hann_window(n_fft)

    if center:
        # F.pad's reflect mode needs an explicit batch dim for 1-D input.
        squeeze = x.dim() == 1
        if squeeze:
            x = x.unsqueeze(0)
        x = torch.nn.functional.pad(x, (n_fft // 2, n_fft // 2), mode="reflect")
        if squeeze:
            x = x.squeeze(0)

    frames = frame(x, n_fft, hop_length)  # (num_frames, n_fft)
    windowed = frames * window  # taper each frame's edges
    spec = rfft_naive(windowed)  # (num_frames, n_fft//2 + 1)
    return spec.transpose(-2, -1)  # (freq, time), torch's convention


def magnitude(spec: torch.Tensor) -> torch.Tensor:
    """|X| — how much energy at this frequency, discarding phase."""
    return spec.abs()


def power(spec: torch.Tensor) -> torch.Tensor:
    """|X|² — what Whisper's mel filterbank is applied to."""
    return spec.abs() ** 2


def griffin_lim_note() -> str:
    """Why throwing away phase is defensible (referenced from notes/02)."""
    return (
        "Phase is discarded because it is largely unpredictable from, and "
        "perceptually secondary to, the magnitude envelope. Griffin-Lim can "
        "reconstruct intelligible audio from magnitude alone, which is the "
        "empirical argument that magnitude carries the linguistic content."
    )
