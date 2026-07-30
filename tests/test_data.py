"""Step 4 tests: the split must be honest and the teacher-forcing shift correct.

Tests needing the corpus skip when it is absent (data/ is gitignored):

    uv run python scripts/04_data.py --download
"""

from __future__ import annotations

import pytest
import torch

from whispr import audio
from whispr.config import AudioConfig, Config, ModelConfig
from whispr.data import (
    DEFAULT_ROOT,
    LibriSpeechDataset,
    Utterance,
    cached_index,
    collate,
    describe,
    speaker_split,
    standard_split,
)

requires_corpus = pytest.mark.skipif(
    not DEFAULT_ROOT.exists(),
    reason="LibriSpeech not downloaded — run scripts/04_data.py --download",
)


def fake_utterances(n=60, speakers=10, duration=5.0):
    return [
        Utterance(
            utt_id=f"{100+i%speakers}-000-{i:04d}",
            path=f"/nonexistent/{i}.flac",
            text="HELLO WORLD",
            duration=duration + (i % 7),
            speaker=str(100 + i % speakers),
        )
        for i in range(n)
    ]


class TestConfig:
    def test_window_arithmetic(self):
        cfg = AudioConfig(window_seconds=15.0)
        assert cfg.n_samples == 240_000
        assert cfg.n_frames == 1500
        assert cfg.n_audio_ctx == 750

    def test_paper_window_gives_paper_numbers(self):
        cfg = AudioConfig(window_seconds=30.0)
        assert cfg.n_samples == audio.N_SAMPLES == 480_000
        assert cfg.n_frames == 3000
        assert cfg.n_audio_ctx == 1500

    def test_model_and_audio_context_must_agree(self):
        """The guard that stops a window change from silently breaking the model."""
        with pytest.raises(ValueError, match="must agree"):
            Config(audio=AudioConfig(window_seconds=30.0), model=ModelConfig(n_audio_ctx=750))

    def test_default_config_is_consistent(self):
        cfg = Config()
        assert cfg.model.n_audio_ctx == cfg.audio.n_audio_ctx

    def test_model_matches_table_1_tiny(self):
        m = ModelConfig()
        assert m.is_tiny_shaped
        assert (m.n_audio_layer, m.n_audio_state, m.n_audio_head) == (4, 384, 6)


class TestSpeakerSplit:
    def test_split_is_speaker_disjoint(self):
        """The property the whole evaluation rests on."""
        train, val = speaker_split(fake_utterances(), val_speakers=3)
        assert {u.speaker for u in train} & {u.speaker for u in val} == set()

    def test_all_utterances_are_kept(self):
        utts = fake_utterances()
        train, val = speaker_split(utts, val_speakers=3)
        assert len(train) + len(val) == len(utts)

    def test_val_speaker_count_is_respected(self):
        _, val = speaker_split(fake_utterances(speakers=10), val_speakers=4)
        assert len({u.speaker for u in val}) == 4

    def test_split_is_deterministic(self):
        utts = fake_utterances()
        a, _ = speaker_split(utts, seed=7)
        b, _ = speaker_split(utts, seed=7)
        assert [u.utt_id for u in a] == [u.utt_id for u in b]

    def test_different_seeds_give_different_splits(self):
        utts = fake_utterances(speakers=20)
        _, a = speaker_split(utts, val_speakers=4, seed=1)
        _, b = speaker_split(utts, val_speakers=4, seed=2)
        assert {u.speaker for u in a} != {u.speaker for u in b}


class TestLengthFiltering:
    def test_long_utterances_are_dropped_not_truncated(self):
        """Truncating audio but keeping the transcript manufactures hallucination."""
        utts = [
            Utterance("a", "/x.flac", "SHORT", 5.0, "1"),
            Utterance("b", "/y.flac", "LONG", 25.0, "1"),
        ]
        ds = LibriSpeechDataset(utts, AudioConfig(window_seconds=15.0))
        assert len(ds) == 1
        assert ds.dropped == 1
        assert ds.utterances[0].utt_id == "a"

    def test_bigger_window_keeps_more(self):
        utts = fake_utterances(n=100)
        small = LibriSpeechDataset(utts, AudioConfig(window_seconds=6.0))
        big = LibriSpeechDataset(utts, AudioConfig(window_seconds=30.0))
        assert len(big) > len(small)
        assert len(big) == len(utts)


class TestCollate:
    def _batch(self, lengths):
        return [
            {
                "mel": torch.randn(80, 1500),
                "text": "X",
                "utt_id": f"u{i}",
                "speaker": "1",
                "duration": 5.0,
                "tokens": torch.arange(n, dtype=torch.long) + 1,
            }
            for i, n in enumerate(lengths)
        ]

    def test_mels_stack(self):
        out = collate(self._batch([5, 5]))
        assert out["mel"].shape == (2, 80, 1500)

    def test_teacher_forcing_shift(self):
        """labels[i] must be the token that follows tokens[i].

        Off-by-one here is a classic silent bug: the model learns to copy its
        input, training loss looks implausibly good, and generation is garbage.
        The fixture's tokens are 1..n, so labels must be exactly tokens + 1.
        """
        out = collate(self._batch([6]))
        tokens, labels = out["tokens"][0], out["labels"][0]
        assert tokens.tolist() == [1, 2, 3, 4, 5]
        assert labels.tolist() == [2, 3, 4, 5, 6]
        assert torch.equal(labels, tokens + 1)

    def test_shapes_are_one_shorter_than_input(self):
        out = collate(self._batch([10, 10]))
        assert out["tokens"].shape == (2, 9)
        assert out["labels"].shape == (2, 9)

    def test_ragged_sequences_are_padded(self):
        out = collate(self._batch([3, 8, 5]))
        assert out["tokens"].shape == (3, 7)  # longest 8, minus the shift

    def test_padding_is_ignored_by_the_loss(self):
        """-100 is cross_entropy's ignore_index, so padding gives no gradient."""
        out = collate(self._batch([3, 8]))
        assert (out["labels"][0] == -100).any()
        assert (out["labels"][1] != -100).all()

        logits = torch.randn(2, 7, 50, requires_grad=True)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, 50), out["labels"].reshape(-1), ignore_index=-100
        )
        loss.backward()
        # The padded positions of row 0 must receive exactly zero gradient.
        assert torch.all(logits.grad[0, -2:] == 0)

    def test_loss_ignores_pad_value_not_just_masks_it(self):
        out = collate(self._batch([2, 6]))
        n_real = int((out["labels"] != -100).sum())
        assert n_real == 1 + 5


@requires_corpus
class TestRealCorpus:
    @pytest.fixture(scope="class")
    def index(self):
        return cached_index(DEFAULT_ROOT, "dev-clean")

    def test_corpus_size(self, index):
        assert len(index) == 2703
        assert sum(u.duration for u in index) / 3600 == pytest.approx(5.39, abs=0.05)
        assert len({u.speaker for u in index}) == 40

    def test_transcripts_are_uppercase_letters_only(self, index):
        charset = set("".join(u.text for u in index))
        assert charset == set(" 'ABCDEFGHIJKLMNOPQRSTUVWXYZ")

    def test_audio_is_already_16k_mono(self, index):
        wav = audio.load_audio(index[0].path)
        assert wav.dim() == 1
        assert wav.shape[0] == pytest.approx(index[0].duration * 16_000, rel=0.01)

    def test_item_shapes(self, index):
        cfg = AudioConfig()
        ds = LibriSpeechDataset(index, cfg)
        item = ds[0]
        assert item["mel"].shape == (cfg.n_mels, cfg.n_frames)
        assert item["mel"].dtype == torch.float32
        assert isinstance(item["text"], str) and item["text"]

    def test_mel_is_in_the_expected_range(self, index):
        ds = LibriSpeechDataset(index, AudioConfig())
        spec = ds[0]["mel"]
        assert -2.0 < spec.min() < 0.5
        assert 0.0 < spec.max() < 2.0
        assert (spec.max() - spec.min()).item() == pytest.approx(2.0, abs=1e-4)

    def test_real_split_sizes(self, index):
        train, val = speaker_split(index)
        assert len({u.speaker for u in val}) == 6
        assert {u.speaker for u in train} & {u.speaker for u in val} == set()
        assert len(train) == 2280
        assert len(val) == 423

    def test_augmentation_changes_the_input(self, index):
        plain = LibriSpeechDataset(index, AudioConfig(), augment=False)
        noisy = LibriSpeechDataset(index, AudioConfig(), augment=True)
        torch.manual_seed(0)
        a = noisy[0]["mel"]
        torch.manual_seed(1)
        b = noisy[0]["mel"]
        assert not torch.equal(a, b)  # random each call
        assert torch.equal(plain[0]["mel"], plain[0]["mel"])  # deterministic without

    def test_describe_is_informative(self, index):
        s = describe(index)
        assert "2,703 utts" in s and "40 speakers" in s


@pytest.mark.skipif(
    not (DEFAULT_ROOT / "train-clean-100").exists(),
    reason="train-clean-100 not downloaded",
)
class TestStandardSplit:
    """LibriSpeech's own partition, used for the 100 h run."""

    def test_sizes(self):
        train, val = standard_split(DEFAULT_ROOT, "train-clean-100", "dev-clean")
        assert len(train) == 28_539
        assert len(val) == 2_703
        assert sum(u.duration for u in train) / 3600 == pytest.approx(100.6, abs=0.2)

    def test_speakers_are_disjoint_by_corpus_design(self):
        """LibriSpeech guarantees this; we assert it rather than trust it."""
        train, val = standard_split(DEFAULT_ROOT, "train-clean-100", "dev-clean")
        assert {u.speaker for u in train} & {u.speaker for u in val} == set()
        assert len({u.speaker for u in train}) == 251

    def test_a_17s_window_covers_the_whole_train_split(self):
        """Why the 100 h run uses 17 s: the 15 s default would drop 29% of it.

        train-clean-100 has a median utterance of 14.0 s, unlike dev-clean's
        5.9 s, so the window that was right for the baseline is wrong here.
        """
        train, _ = standard_split(DEFAULT_ROOT, "train-clean-100", "dev-clean")
        at_15 = LibriSpeechDataset(train, AudioConfig(window_seconds=15.0))
        at_17 = LibriSpeechDataset(train, AudioConfig(window_seconds=17.0))
        assert at_15.dropped > 8000
        assert at_17.dropped < 100
        assert at_17.total_hours() > 100.0

    def test_17s_window_arithmetic(self):
        cfg = AudioConfig(window_seconds=17.0)
        assert cfg.n_samples == 272_000
        assert cfg.n_frames == 1700
        assert cfg.n_audio_ctx == 850
