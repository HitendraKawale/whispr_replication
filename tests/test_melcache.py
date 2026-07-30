"""Tests for precomputed mel caching.

The cache is an optimisation, so the bar is that it changes nothing: the same
tensors, and augmentation that is still equivalent to the waveform-domain
version it replaces.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from whispr import audio, mel, melcache
from whispr.config import AudioConfig
from whispr.data import DEFAULT_ROOT, LibriSpeechDataset, cached_index

requires_corpus = pytest.mark.skipif(
    not DEFAULT_ROOT.exists(), reason="LibriSpeech not downloaded"
)


class TestGainOffset:
    """The identity that lets a precomputed cache support gain augmentation."""

    def test_offset_is_db_over_40(self):
        assert melcache.gain_offset(6.0) == pytest.approx(0.15)
        assert melcache.gain_offset(-6.0) == pytest.approx(-0.15)
        assert melcache.gain_offset(0.0) == 0.0

    @pytest.mark.parametrize("db", [-20, -12, -6, -3, -1, 1, 3, 6, 12])
    def test_matches_actual_waveform_gain(self, db):
        """Scaling audio then computing the mel == computing the mel then adding.

        Exact rather than approximate, because the -8 dB floor is relative to the
        utterance's own peak and therefore shifts with it.
        """
        wav = audio.chirp(120, 6500, 6.0, amplitude=0.4)
        cfg = AudioConfig(window_seconds=17.0)

        base = mel.log_mel_spectrogram(wav, pad_to=cfg.n_samples)
        scaled = mel.log_mel_spectrogram(wav * 10 ** (db / 20), pad_to=cfg.n_samples)

        assert torch.allclose(scaled, base + melcache.gain_offset(db), atol=1e-3)

    def test_breaks_down_at_extreme_attenuation(self):
        """Documents the limit rather than pretending it isn't there.

        Below roughly -30 dB the frontend's absolute 1e-10 clamp starts binding
        instead of the relative floor (notes/03 §4b), and the clean offset
        identity stops holding. Our jitter is +/-6 dB, far inside the safe range.
        """
        wav = audio.chirp(120, 6500, 6.0, amplitude=0.4)
        cfg = AudioConfig(window_seconds=17.0)
        base = mel.log_mel_spectrogram(wav, pad_to=cfg.n_samples)
        scaled = mel.log_mel_spectrogram(wav * 10 ** (-40 / 20), pad_to=cfg.n_samples)

        assert not torch.allclose(scaled, base + melcache.gain_offset(-40), atol=1e-2)


class TestCachePaths:
    def test_name_encodes_window_and_mels(self):
        npy, meta = melcache.cache_paths("dev-clean", AudioConfig(window_seconds=17.0))
        assert npy.name == "dev-clean_w17_m80.npy"
        assert meta.name == "dev-clean_w17_m80.json"

    def test_different_windows_get_different_files(self):
        a, _ = melcache.cache_paths("dev-clean", AudioConfig(window_seconds=15.0))
        b, _ = melcache.cache_paths("dev-clean", AudioConfig(window_seconds=17.0))
        assert a != b

    def test_load_returns_none_when_absent(self, tmp_path):
        assert melcache.MelCache.load("nope", AudioConfig(), cache_dir=tmp_path) is None


@requires_corpus
class TestBuildAndRead:
    @pytest.fixture(scope="class")
    def small(self, tmp_path_factory):
        """Build a tiny cache from real audio."""
        cfg = AudioConfig(window_seconds=17.0)
        utts = cached_index(DEFAULT_ROOT, "dev-clean")[:12]
        d = tmp_path_factory.mktemp("cache")
        cache = melcache.build(utts, cfg, "tiny", cache_dir=d)
        return cache, utts, cfg, d

    def test_shape_and_dtype(self, small):
        cache, utts, cfg, _ = small
        keep = [u for u in utts if u.duration <= cfg.window_seconds]
        assert len(cache) == len(keep)
        assert cache.array.shape == (len(keep), cfg.n_mels, cfg.n_frames)
        assert cache.array.dtype == np.float16

    def test_get_matches_on_the_fly_computation(self, small):
        """The whole point: identical output, within float16 resolution."""
        cache, utts, cfg, _ = small
        for utt in utts[:5]:
            if utt.utt_id not in cache:
                continue
            direct = mel.log_mel_spectrogram(
                audio.load_audio(utt.path),
                n_mels=cfg.n_mels,
                hop_length=cfg.hop_length,
                pad_to=cfg.n_samples,
            )
            assert torch.allclose(cache.get(utt.utt_id), direct, atol=1e-3)

    def test_returns_float32(self, small):
        cache, utts, _, _ = small
        assert cache.get(utts[0].utt_id).dtype == torch.float32

    def test_membership(self, small):
        cache, utts, _, _ = small
        assert utts[0].utt_id in cache
        assert "not-a-real-id" not in cache

    def test_rebuild_is_skipped_when_present(self, small):
        cache, utts, cfg, d = small
        again = melcache.build(utts, cfg, "tiny", cache_dir=d)
        assert len(again) == len(cache)

    def test_mismatched_config_is_rejected(self, small):
        """A window change must fail loudly, not silently feed wrong shapes."""
        cache, _, _, _ = small
        with pytest.raises(ValueError, match="built for window"):
            cache.check_matches(AudioConfig(window_seconds=15.0))

    def test_load_roundtrip(self, small):
        cache, utts, cfg, d = small
        loaded = melcache.MelCache.load("tiny", cfg, cache_dir=d)
        assert loaded is not None
        assert torch.equal(loaded.get(utts[0].utt_id), cache.get(utts[0].utt_id))


@requires_corpus
class TestDatasetIntegration:
    @pytest.fixture(scope="class")
    def pair(self, tmp_path_factory):
        cfg = AudioConfig(window_seconds=17.0)
        utts = cached_index(DEFAULT_ROOT, "dev-clean")[:12]
        d = tmp_path_factory.mktemp("cache")
        cache = melcache.build(utts, cfg, "tiny", cache_dir=d)
        return utts, cfg, cache

    def test_cached_dataset_matches_uncached(self, pair):
        utts, cfg, cache = pair
        plain = LibriSpeechDataset(utts, cfg)
        cached = LibriSpeechDataset(utts, cfg, mel_cache=cache)

        assert len(plain) == len(cached)
        for i in range(min(5, len(plain))):
            assert torch.allclose(cached[i]["mel"], plain[i]["mel"], atol=1e-3)
            assert cached[i]["text"] == plain[i]["text"]
            assert cached[i]["utt_id"] == plain[i]["utt_id"]

    def test_missing_utterances_are_rejected(self, pair):
        """Silently returning wrong mels would be far worse than an error."""
        from whispr.data import Utterance

        utts, cfg, cache = pair
        extra = list(utts) + [
            Utterance("ghost-0-0", "/nope.flac", "HELLO", 5.0, "ghost")
        ]
        with pytest.raises(ValueError, match="not in the mel cache"):
            LibriSpeechDataset(extra, cfg, mel_cache=cache)

    def test_augmentation_shifts_but_preserves_shape(self, pair):
        utts, cfg, cache = pair
        ds = LibriSpeechDataset(utts, cfg, mel_cache=cache, augment=True)
        base = LibriSpeechDataset(utts, cfg, mel_cache=cache, augment=False)

        torch.manual_seed(0)
        a = ds[0]["mel"]
        b = base[0]["mel"]
        # A pure offset: the difference is constant, and within the +/-6 dB range.
        diff = a - b
        assert diff.std().item() < 1e-5
        assert abs(diff.mean().item()) <= melcache.gain_offset(6.0) + 1e-6

    def test_augmentation_varies_between_calls(self, pair):
        utts, cfg, cache = pair
        ds = LibriSpeechDataset(utts, cfg, mel_cache=cache, augment=True)
        torch.manual_seed(0)
        a = ds[0]["mel"]
        torch.manual_seed(1)
        b = ds[0]["mel"]
        assert not torch.allclose(a, b)
