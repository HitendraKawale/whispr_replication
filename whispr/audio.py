"""Waveform-level audio I/O and synthesis.

Everything downstream assumes the invariant established here: **mono, float32,
16 kHz, roughly in [-1, 1]**. `load_audio` is the only place that invariant is
created, so it is the only place it can be violated.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

# Whisper's frontend constants (paper §2.2). Defined here because the sample
# rate is a property of the audio, not of the spectrogram.
SAMPLE_RATE = 16_000
CHUNK_SECONDS = 30
N_SAMPLES = SAMPLE_RATE * CHUNK_SECONDS  # 480_000 samples in one Whisper window


def load_audio(path: str | Path, sample_rate: int = SAMPLE_RATE) -> torch.Tensor:
    """Load any audio file as mono float32 at `sample_rate`.

    Returns a 1-D tensor of shape (num_samples,).

    Resampling goes through torchaudio's Resample, which low-pass filters before
    decimating. Never downsample by slicing — see notes/01, aliasing is
    irreversible.

    I/O is via soundfile (libsndfile) rather than torchaudio.load: as of
    torchaudio 2.11 the load/save paths delegate to the separate torchcodec
    package. libsndfile reads FLAC natively, which is what LibriSpeech ships.
    """
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)  # (samples, channels)
    wav = torch.from_numpy(data.T)  # (channels, samples)

    if wav.shape[0] > 1:  # downmix to mono by averaging channels
        wav = wav.mean(dim=0, keepdim=True)

    if sr != sample_rate:
        wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)(wav)

    return wav.squeeze(0).to(torch.float32)


def save_audio(path: str | Path, wav: torch.Tensor, sample_rate: int = SAMPLE_RATE) -> None:
    """Write a 1-D float waveform to disk as 16-bit PCM."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    arr = wav.detach().cpu().numpy()
    if arr.ndim == 2:  # (channels, samples) -> (samples, channels)
        arr = arr.T
    sf.write(str(path), np.clip(arr, -1.0, 1.0), sample_rate, subtype="PCM_16")


def pad_or_trim(wav: torch.Tensor, length: int = N_SAMPLES) -> torch.Tensor:
    """Force a waveform to exactly `length` samples.

    Whisper's input is a *fixed* 30-second window — always 480,000 samples,
    zero-padded if the utterance is shorter. This is unusual (most ASR models
    use variable-length inputs with masking) and it is the reason the encoder
    can use a fixed positional encoding of exactly 1500 positions.
    """
    if wav.shape[-1] > length:
        return wav[..., :length]
    if wav.shape[-1] < length:
        pad = length - wav.shape[-1]
        return torch.nn.functional.pad(wav, (0, pad))
    return wav


def sine(
    freq: float,
    duration: float,
    sample_rate: int = SAMPLE_RATE,
    amplitude: float = 0.5,
    phase: float = 0.0,
) -> torch.Tensor:
    """A pure tone — our ground truth for testing the DFT in step 2."""
    t = torch.arange(int(duration * sample_rate), dtype=torch.float32) / sample_rate
    return amplitude * torch.sin(2 * math.pi * freq * t + phase)


def chirp(
    f0: float,
    f1: float,
    duration: float,
    sample_rate: int = SAMPLE_RATE,
    amplitude: float = 0.5,
) -> torch.Tensor:
    """A linear frequency sweep from f0 to f1 Hz.

    The instantaneous frequency is f(t) = f0 + (f1-f0)*t/T, and the signal is
    sin of its *integral* — a common off-by-one-integral bug is to use
    sin(2*pi*f(t)*t), which sweeps at twice the intended rate.
    """
    n = int(duration * sample_rate)
    t = torch.arange(n, dtype=torch.float32) / sample_rate
    k = (f1 - f0) / duration
    phase = 2 * math.pi * (f0 * t + 0.5 * k * t**2)
    return amplitude * torch.sin(phase)


def alias_frequency(freq: float, sample_rate: int) -> float:
    """Where a tone of `freq` Hz *appears* after sampling at `sample_rate`.

    Frequencies above Nyquist fold back into [0, sr/2]. This closed form is what
    we assert against in tests.
    """
    return abs(freq - sample_rate * round(freq / sample_rate))


def quantize(wav: torch.Tensor, bits: int) -> torch.Tensor:
    """Round a float waveform to `bits` of resolution, staying in float.

    Used only to *demonstrate* quantization noise; real 16-bit audio is already
    quantized by the time we see it.
    """
    levels = 2 ** (bits - 1)
    return torch.clamp(torch.round(wav * levels) / levels, -1.0, 1.0)


def db(x: torch.Tensor, floor: float = 1e-10) -> torch.Tensor:
    """Amplitude to decibels."""
    return 20 * torch.log10(torch.clamp(x.abs(), min=floor))
