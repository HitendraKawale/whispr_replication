"""Precomputed log-mel spectrograms on disk.

Decoding FLAC and computing an 80xN log-mel for every batch costs about a third
of training step time (notes/07 §7), and it is the *same* computation every
epoch. Doing it once and memory-mapping the result removes that cost.

Layout, one pair of files per (split, window, n_mels):

    data/mel_cache/train-clean-100_w17.0_m80.npy    (N, n_mels, n_frames) float16
    data/mel_cache/train-clean-100_w17.0_m80.json   utterance ids, in row order

float16 because the round-trip error on real log-mels is 4.9e-4 — 0.02% of the
value range — while halving the file from 15.5 GB to 7.75 GB. The array is
memory-mapped, so RAM use is the OS page cache rather than the file size, and
building it never holds more than one utterance in memory.

The cache is keyed by window length because `n_frames` depends on it. Asking for
a window the cache wasn't built for is an error, not a silent mismatch.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from whispr import audio, mel
from whispr.config import AudioConfig
from whispr.data import DEFAULT_ROOT, Utterance

CACHE_DIR = DEFAULT_ROOT.parent / "mel_cache"


def cache_paths(
    split: str, config: AudioConfig, cache_dir: Path | str = CACHE_DIR
) -> tuple[Path, Path]:
    """The .npy/.json pair for this split and frontend configuration."""
    stem = f"{split}_w{config.window_seconds:g}_m{config.n_mels}"
    d = Path(cache_dir)
    return d / f"{stem}.npy", d / f"{stem}.json"


class MelCache:
    """Read-only, memory-mapped access to precomputed mels."""

    def __init__(self, npy_path: Path | str, meta_path: Path | str) -> None:
        self.meta = json.loads(Path(meta_path).read_text())
        self.array = np.load(str(npy_path), mmap_mode="r")
        self.row_of = {utt_id: i for i, utt_id in enumerate(self.meta["utt_ids"])}

        expected = (len(self.row_of), self.meta["n_mels"], self.meta["n_frames"])
        if self.array.shape != tuple(expected):
            raise ValueError(
                f"cache shape {self.array.shape} does not match its index {expected}; "
                f"delete {npy_path} and rebuild"
            )

    def __len__(self) -> int:
        return len(self.row_of)

    def __contains__(self, utt_id: str) -> bool:
        return utt_id in self.row_of

    @property
    def window_seconds(self) -> float:
        return self.meta["window_seconds"]

    @property
    def n_frames(self) -> int:
        return self.meta["n_frames"]

    def get(self, utt_id: str) -> torch.Tensor:
        """Return one (n_mels, n_frames) spectrogram as float32."""
        row = self.array[self.row_of[utt_id]]
        # np.asarray copies the mmap slice; .astype gives us back float32.
        return torch.from_numpy(np.asarray(row, dtype=np.float32))

    def check_matches(self, config: AudioConfig) -> None:
        """Fail loudly if the cache was built for a different frontend."""
        if (
            self.meta["window_seconds"] != config.window_seconds
            or self.meta["n_mels"] != config.n_mels
            or self.meta["n_frames"] != config.n_frames
        ):
            raise ValueError(
                f"cache was built for window={self.meta['window_seconds']}s "
                f"n_mels={self.meta['n_mels']} n_frames={self.meta['n_frames']}, "
                f"but the config asks for window={config.window_seconds}s "
                f"n_mels={config.n_mels} n_frames={config.n_frames}"
            )

    @classmethod
    def load(
        cls,
        split: str,
        config: AudioConfig,
        cache_dir: Path | str = CACHE_DIR,
    ) -> "MelCache | None":
        """Load if present and matching, else None. Never raises on absence."""
        npy, meta = cache_paths(split, config, cache_dir)
        if not (npy.exists() and meta.exists()):
            return None
        cache = cls(npy, meta)
        cache.check_matches(config)
        return cache


def build(
    utterances: list[Utterance],
    config: AudioConfig,
    split: str,
    cache_dir: Path | str = CACHE_DIR,
    dtype: str = "float16",
    overwrite: bool = False,
    on_progress=None,
) -> MelCache:
    """Compute and write the cache for `utterances`.

    Only utterances that fit the window are stored — the same filter
    `LibriSpeechDataset` applies, so the cache and the dataset agree on
    membership.

    Writes through `open_memmap`, so peak memory is one spectrogram rather than
    the whole array.
    """
    npy, meta_path = cache_paths(split, config, cache_dir)
    npy.parent.mkdir(parents=True, exist_ok=True)

    if npy.exists() and not overwrite:
        existing = MelCache(npy, meta_path)
        existing.check_matches(config)
        return existing

    keep = [u for u in utterances if u.duration <= config.window_seconds]
    shape = (len(keep), config.n_mels, config.n_frames)

    array = np.lib.format.open_memmap(
        str(npy), mode="w+", dtype=np.dtype(dtype), shape=shape
    )

    start = time.perf_counter()
    for i, utt in enumerate(keep):
        wav = audio.load_audio(utt.path, config.sample_rate)
        spec = mel.log_mel_spectrogram(
            wav,
            n_mels=config.n_mels,
            sample_rate=config.sample_rate,
            hop_length=config.hop_length,
            pad_to=config.n_samples,
        )
        array[i] = spec.numpy().astype(dtype)
        if on_progress and (i % 200 == 0 or i == len(keep) - 1):
            on_progress(i + 1, len(keep), time.perf_counter() - start)

    array.flush()
    del array  # close the memmap before reopening read-only

    meta_path.write_text(
        json.dumps(
            {
                "split": split,
                "window_seconds": config.window_seconds,
                "n_mels": config.n_mels,
                "n_frames": config.n_frames,
                "sample_rate": config.sample_rate,
                "hop_length": config.hop_length,
                "dtype": dtype,
                "utt_ids": [u.utt_id for u in keep],
            }
        )
    )
    return MelCache(npy, meta_path)


def gain_offset(db: float) -> float:
    """The log-mel offset equivalent to scaling the waveform by `db` decibels.

    Scaling audio by k multiplies mel *power* by k^2, so log10 shifts by
    log10(k^2) = db/10, and the frontend's final `(x + 4) / 4` divides that by
    4 — giving **db / 40**.

    This is exact, not an approximation: the -8 dB floor is relative to the
    utterance's own peak, so it shifts with it. Verified to 2e-5 for gains from
    -20 dB to +12 dB (tests/test_melcache.py). It *breaks* below about -30 dB,
    where the frontend's absolute 1e-10 clamp starts binding instead of the
    relative floor — the two-competing-floors behaviour from notes/03 §4b.

    This is what makes a precomputed cache compatible with gain augmentation:
    we add a scalar instead of re-running the frontend.
    """
    return db / 40.0
