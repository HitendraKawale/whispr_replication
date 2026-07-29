"""The mel filterbank and Whisper's exact log-mel frontend.

This is the most precision-critical module in the repo. If the frontend is off
by a scale factor or a mel convention, the model still trains, the loss still
goes down, and everything is quietly worse — with no error to point at. So
every constant here is justified in a comment, and tests/test_mel.py compares
the filterbank element-wise against the array shipped inside `openai/whisper`.

Whisper's frontend (paper §2.2 + the reference implementation):

    16 kHz mono
      -> STFT, n_fft=400 (25 ms), hop=160 (10 ms), Hann, center=True
      -> drop the final frame              (3001 -> 3000)
      -> power spectrum |X|^2              (201 bins)
      -> 80-channel Slaney mel filterbank  (80 x 201 matmul)
      -> log10, clamped at 1e-10
      -> floor at (max - 8): an 80 dB dynamic range
      -> (x + 4) / 4: lands roughly in [-1, 1]
"""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from whispr.audio import N_SAMPLES, SAMPLE_RATE

N_FFT = 400  # 25 ms window at 16 kHz
HOP_LENGTH = 160  # 10 ms stride
N_MELS = 80  # paper: "80-channel log-magnitude Mel spectrogram"
N_FRAMES = N_SAMPLES // HOP_LENGTH  # 3000 frames per 30 s window

# Slaney mel-scale constants. Note this is *not* the HTK formula
# (2595·log10(1 + f/700)) that most tutorials give. Whisper's filterbank comes
# from librosa's default, which is Slaney: linear below 1 kHz, log above.
_F_SP = 200.0 / 3.0  # Hz per mel in the linear region
_MIN_LOG_HZ = 1000.0  # where the scale switches from linear to log
_MIN_LOG_MEL = _MIN_LOG_HZ / _F_SP  # = 15.0 mel
_LOGSTEP = math.log(6.4) / 27.0  # chosen so the two regions meet smoothly


def hz_to_mel(freq: np.ndarray | float) -> np.ndarray:
    """Hz -> mel, Slaney convention (librosa's `htk=False`).

    The mel scale approximates human pitch perception: below ~1 kHz we hear
    pitch roughly linearly in Hz, above it roughly logarithmically. Doubling
    100 Hz to 200 Hz is a big perceptual jump; 5000 to 5100 Hz is inaudible.
    """
    freq = np.asarray(freq, dtype=np.float64)
    mels = freq / _F_SP
    log_region = freq >= _MIN_LOG_HZ
    mels = np.where(
        log_region,
        _MIN_LOG_MEL + np.log(np.maximum(freq, _MIN_LOG_HZ) / _MIN_LOG_HZ) / _LOGSTEP,
        mels,
    )
    return mels


def mel_to_hz(mels: np.ndarray | float) -> np.ndarray:
    """mel -> Hz, the inverse of `hz_to_mel`."""
    mels = np.asarray(mels, dtype=np.float64)
    freqs = mels * _F_SP
    log_region = mels >= _MIN_LOG_MEL
    return np.where(
        log_region,
        _MIN_LOG_HZ * np.exp(_LOGSTEP * (np.maximum(mels, _MIN_LOG_MEL) - _MIN_LOG_MEL)),
        freqs,
    )


def mel_filterbank(
    sample_rate: int = SAMPLE_RATE,
    n_fft: int = N_FFT,
    n_mels: int = N_MELS,
    fmin: float = 0.0,
    fmax: float | None = None,
) -> torch.Tensor:
    """Build the (n_mels, n_fft//2 + 1) mel projection matrix.

    Each row is a triangular window over linear-frequency bins. The triangles
    are narrow at low frequency and wide at high frequency — evenly spaced *on
    the mel axis*, which is what makes the output perceptually uniform.

    Row i spans [mel_f[i], mel_f[i+2]] and peaks at mel_f[i+1], so adjacent
    triangles overlap by half. This reproduces `librosa.filters.mel(...,
    htk=False, norm="slaney")`, which is where Whisper's shipped array came
    from.
    """
    fmax = sample_rate / 2 if fmax is None else fmax

    # Centre frequency of each DFT bin.
    fft_freqs = np.linspace(0, sample_rate / 2, n_fft // 2 + 1)

    # n_mels + 2 band edges, evenly spaced in *mel*, mapped back to Hz.
    mel_edges = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz_edges = mel_to_hz(mel_edges)

    fdiff = np.diff(hz_edges)
    # ramps[i, j] = hz_edges[i] - fft_freqs[j]
    ramps = hz_edges[:, None] - fft_freqs[None, :]

    weights = np.zeros((n_mels, len(fft_freqs)))
    for i in range(n_mels):
        lower = -ramps[i] / fdiff[i]  # rising edge of triangle i
        upper = ramps[i + 2] / fdiff[i + 1]  # falling edge
        weights[i] = np.maximum(0.0, np.minimum(lower, upper))

    # Slaney normalisation: make each filter integrate to a constant area
    # rather than peak at 1. Without this, high-frequency filters — being wider
    # — would dominate simply by covering more bins.
    enorm = 2.0 / (hz_edges[2 : n_mels + 2] - hz_edges[:n_mels])
    weights *= enorm[:, None]

    return torch.from_numpy(weights).to(torch.float32)


@lru_cache(maxsize=4)
def _cached_filterbank(sample_rate: int, n_fft: int, n_mels: int) -> torch.Tensor:
    return mel_filterbank(sample_rate, n_fft, n_mels)


def log_mel_spectrogram(
    wav: torch.Tensor,
    n_mels: int = N_MELS,
    sample_rate: int = SAMPLE_RATE,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    pad_to: int | None = N_SAMPLES,
) -> torch.Tensor:
    """Whisper's frontend, end to end.

    Input:  (num_samples,) or (batch, num_samples) float waveform at 16 kHz.
    Output: (n_mels, 3000) or (batch, n_mels, 3000).

    With `pad_to=N_SAMPLES` (the default) the waveform is forced to exactly
    30 seconds first, so the output is always 3000 frames — Whisper's fixed
    input size. Pass `pad_to=None` to keep the natural length, which is useful
    for plotting short clips.
    """
    if pad_to is not None:
        from whispr.audio import pad_or_trim

        wav = pad_or_trim(wav, pad_to)

    window = torch.hann_window(n_fft, device=wav.device)
    stft = torch.stft(
        wav, n_fft=n_fft, hop_length=hop_length, window=window,
        center=True, return_complex=True,
    )

    # Drop the final frame. With center=True there are 1 + n//hop frames; the
    # last one is centred on the sample *past* the end of the signal, so it
    # carries only reflection padding. 3001 -> 3000.
    power = stft[..., :-1].abs() ** 2

    filters = _cached_filterbank(sample_rate, n_fft, n_mels).to(wav.device)
    mel = filters @ power

    log_spec = torch.clamp(mel, min=1e-10).log10()

    # Floor 80 dB below the peak. This is a *per-utterance* dynamic range clamp:
    # it discards the near-silent noise floor, whose absolute level varies with
    # recording conditions and carries no linguistic information.
    log_spec = torch.maximum(log_spec, log_spec.amax(dim=(-2, -1), keepdim=True) - 8.0)

    # Affine rescale into roughly [-1, 1]. The paper says "we globally scale the
    # input to be between -1 and 1 with approximately zero mean". Note this is a
    # fixed affine map, not a learned or per-batch normalisation — so the same
    # audio always produces the same tensor.
    return (log_spec + 4.0) / 4.0


def load_reference_filters(path: str | Path, n_mels: int = N_MELS) -> torch.Tensor:
    """Load the filterbank array shipped inside `openai/whisper`, for testing."""
    with np.load(str(path)) as z:
        return torch.from_numpy(z[f"mel_{n_mels}"]).to(torch.float32)
