"""Step 8 tests: WER arithmetic, normalisation, and decoder mechanics."""

from __future__ import annotations

import pytest
import torch

from whispr.config import ModelConfig
from whispr.decode import (
    Decoder,
    edit_distance,
    normalise,
    word_error_rate,
)
from whispr.model import build_model
from whispr.tokenizer import WhisprTokenizer

CORPUS = ["THE QUICK BROWN FOX", "HELLO WORLD", "MISTER QUILTER IS THE APOSTLE"] * 30


@pytest.fixture(scope="module")
def tokenizer():
    return WhisprTokenizer.train(CORPUS, vocab_size=300)


@pytest.fixture(scope="module")
def setup(tokenizer):
    cfg = ModelConfig(
        n_vocab=tokenizer.n_vocab, n_audio_ctx=50, n_audio_state=64, n_audio_head=4,
        n_audio_layer=2, n_text_ctx=24, n_text_state=64, n_text_head=4, n_text_layer=2,
    )
    model = build_model(cfg)
    return model, tokenizer, cfg


class TestEditDistance:
    def test_identical_is_zero(self):
        assert edit_distance(["A", "B"], ["A", "B"]) == 0

    def test_empty_cases(self):
        assert edit_distance([], []) == 0
        assert edit_distance(["A", "B", "C"], []) == 3

    def test_substitution(self):
        assert edit_distance(["A", "B", "C"], ["A", "X", "C"]) == 1

    def test_deletion(self):
        assert edit_distance(["A", "B", "C"], ["A", "C"]) == 1

    def test_insertion(self):
        assert edit_distance(["A", "C"], ["A", "B", "C"]) == 1

    def test_is_symmetric(self):
        a, b = ["THE", "CAT", "SAT"], ["A", "CAT", "SAT", "DOWN"]
        assert edit_distance(a, b) == edit_distance(b, a)

    def test_all_three_error_types(self):
        ref = ["THE", "QUICK", "BROWN", "FOX"]
        hyp = ["THE", "SLOW", "FOX", "JUMPS"]  # 1 sub, 1 del, 1 ins
        assert edit_distance(ref, hyp) == 3


class TestNormalise:
    def test_uppercases(self):
        assert normalise("hello world") == "HELLO WORLD"

    def test_strips_punctuation(self):
        assert normalise("Hello, world! It's fine.") == "HELLO WORLD IT'S FINE"

    def test_keeps_apostrophes(self):
        """LibriSpeech's alphabet includes them, so they are content."""
        assert normalise("QUILTER'S") == "QUILTER'S"

    def test_collapses_whitespace(self):
        assert normalise("  A   B  \n C ") == "A B C"

    def test_digits_are_removed_not_kept(self):
        """Our labels never contain digits, so a digit is always an error."""
        assert normalise("ROOM 101") == "ROOM"

    def test_is_idempotent(self):
        t = "Hello, World! It's 5 o'clock."
        assert normalise(normalise(t)) == normalise(t)


class TestWordErrorRate:
    def test_perfect_transcription_is_zero(self):
        r = word_error_rate(["THE QUICK BROWN FOX"], ["THE QUICK BROWN FOX"])
        assert r["wer"] == 0.0
        assert r["errors"] == 0
        assert r["words"] == 4

    def test_one_error_in_four_words(self):
        r = word_error_rate(["THE QUICK BROWN FOX"], ["THE SLOW BROWN FOX"])
        assert r["wer"] == pytest.approx(0.25)

    def test_empty_hypothesis_is_one_hundred_percent(self):
        assert word_error_rate(["THE QUICK BROWN FOX"], [""])["wer"] == 1.0

    def test_wer_can_exceed_one(self):
        """Insertions are unbounded, so WER is not a percentage of anything."""
        r = word_error_rate(["HELLO"], ["A B C D E F"])
        assert r["wer"] > 1.0

    def test_normalisation_is_applied(self):
        """Punctuation and casing must not count as recognition errors."""
        assert word_error_rate(["THE QUICK BROWN FOX"], ["the quick brown fox!"])["wer"] == 0.0

    def test_aggregates_over_corpus_not_per_utterance(self):
        """A 1-word utterance wrong is 100% WER and would dominate a mean.

        Corpus-level: 1 error out of 11 words = 9.1%.
        Per-utterance mean would be (100% + 0%) / 2 = 50%.
        """
        refs = ["HELLO", "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG"]
        hyps = ["GOODBYE", "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG"]
        r = word_error_rate(refs, hyps)
        assert r["words"] == 10
        assert r["errors"] == 1
        assert r["wer"] == pytest.approx(0.1)

    def test_matches_jiwer(self):
        """Cross-check against the standard library."""
        jiwer = pytest.importorskip("jiwer")
        refs = ["THE QUICK BROWN FOX", "HELLO WORLD THIS IS A TEST"]
        hyps = ["THE SLOW BROWN FOX JUMPS", "HELLO WORLD IS A TEST"]
        ours = word_error_rate(refs, hyps)["wer"]
        theirs = jiwer.wer([normalise(r) for r in refs], [normalise(h) for h in hyps])
        assert ours == pytest.approx(theirs, abs=1e-9)


class TestDecoder:
    def test_greedy_returns_one_result_per_item(self, setup):
        model, tok, cfg = setup
        dec = Decoder(model, tok, device=torch.device("cpu"))
        mel = torch.randn(3, cfg.n_mels, cfg.n_audio_ctx * 2)
        out = dec.greedy(mel)
        assert len(out) == 3
        assert all(isinstance(r.text, str) for r in out)

    def test_greedy_accepts_unbatched_input(self, setup):
        model, tok, cfg = setup
        dec = Decoder(model, tok, device=torch.device("cpu"))
        out = dec.greedy(torch.randn(cfg.n_mels, cfg.n_audio_ctx * 2))
        assert len(out) == 1

    def test_output_starts_with_the_sot_sequence(self, setup):
        model, tok, cfg = setup
        dec = Decoder(model, tok, device=torch.device("cpu"))
        out = dec.greedy(torch.randn(1, cfg.n_mels, cfg.n_audio_ctx * 2))
        assert out[0].tokens[: tok.prompt_length] == tok.sot_sequence()

    def test_special_tokens_are_stripped_from_text(self, setup):
        model, tok, cfg = setup
        dec = Decoder(model, tok, device=torch.device("cpu"))
        out = dec.greedy(torch.randn(2, cfg.n_mels, cfg.n_audio_ctx * 2))
        for r in out:
            assert "<|" not in r.text

    def test_respects_the_context_limit(self, setup):
        model, tok, cfg = setup
        dec = Decoder(model, tok, device=torch.device("cpu"))
        out = dec.greedy(torch.randn(1, cfg.n_mels, cfg.n_audio_ctx * 2))
        assert len(out[0].tokens) <= cfg.n_text_ctx

    def test_greedy_is_deterministic(self, setup):
        model, tok, cfg = setup
        dec = Decoder(model, tok, device=torch.device("cpu"))
        mel = torch.randn(1, cfg.n_mels, cfg.n_audio_ctx * 2)
        assert dec.greedy(mel)[0].tokens == dec.greedy(mel)[0].tokens

    def test_beam_returns_one_result_per_item(self, setup):
        model, tok, cfg = setup
        dec = Decoder(model, tok, device=torch.device("cpu"))
        mel = torch.randn(2, cfg.n_mels, cfg.n_audio_ctx * 2)
        out = dec.beam(mel, beam_size=3)
        assert len(out) == 2

    def test_beam_scores_at_least_as_well_as_greedy(self, setup):
        """Beam search searches a superset of greedy's single path.

        With length normalisation the comparison isn't a strict guarantee, so
        this allows a small tolerance — but a beam that scores much *worse*
        than greedy indicates a bug in the ranking.
        """
        model, tok, cfg = setup
        dec = Decoder(model, tok, device=torch.device("cpu"))
        mel = torch.randn(1, cfg.n_mels, cfg.n_audio_ctx * 2)
        g = dec.greedy(mel)[0]
        b = dec.beam(mel, beam_size=4)[0]
        assert b.avg_logprob > g.avg_logprob - 0.5

    def test_avg_logprob_is_negative(self, setup):
        model, tok, cfg = setup
        dec = Decoder(model, tok, device=torch.device("cpu"))
        out = dec.greedy(torch.randn(1, cfg.n_mels, cfg.n_audio_ctx * 2))
        assert out[0].avg_logprob < 0

    def test_a_trained_model_reproduces_what_it_memorised(self, setup):
        """End-to-end: overfit one example, then check decoding returns it.

        This is the real integration test of the whole stack — frontend,
        tokenizer, model, and decoder all have to agree for it to pass.
        """
        import torch.nn.functional as F

        model, tok, cfg = setup
        model = build_model(cfg)  # a fresh one, so other tests aren't affected
        target = "THE QUICK BROWN FOX"
        tokens = tok.encode_training(target)
        mel = torch.randn(1, cfg.n_mels, cfg.n_audio_ctx * 2)
        inp = torch.tensor([tokens[:-1]])
        lab = torch.tensor([tokens[1:]])

        opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
        model.train()
        for _ in range(220):
            logits = model(mel, inp)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), lab.reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        assert loss.item() < 0.05, f"failed to memorise, loss {loss.item():.3f}"
        out = Decoder(model, tok, device=torch.device("cpu")).greedy(mel)
        assert out[0].text == target
