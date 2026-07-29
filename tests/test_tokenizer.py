"""Step 5 tests: round-tripping, the multitask prefix, and the id conventions."""

from __future__ import annotations

import pytest

from whispr.tokenizer import (
    EOT,
    NOTIMESTAMPS,
    PAD,
    SOT,
    SPECIAL_TOKENS,
    TRANSCRIBE,
    TRANSLATE,
    WhisprTokenizer,
)

CORPUS = [
    "MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES",
    "AND WE ARE GLAD TO WELCOME HIS GOSPEL",
    "NOR IS MISTER QUILTER'S MANNER LESS INTERESTING THAN HIS MATTER",
    "HE TELLS US THAT AT THIS FESTIVE SEASON OF THE YEAR",
    "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG",
] * 40


@pytest.fixture(scope="module")
def tok():
    return WhisprTokenizer.train(CORPUS, vocab_size=512)


class TestRoundTrip:
    @pytest.mark.parametrize(
        "text",
        [
            "HELLO WORLD",
            "THE QUICK BROWN FOX",
            "QUILTER'S",
            "A",
            "MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES",
        ],
    )
    def test_decode_encode_is_identity(self, tok, text):
        assert tok.decode(tok.encode(text)) == text

    def test_roundtrip_on_the_whole_corpus(self, tok):
        assert all(tok.decode(tok.encode(t)) == t for t in CORPUS)

    def test_unseen_words_still_roundtrip(self, tok):
        """BPE always falls back to characters, so nothing is unrepresentable."""
        for word in ("ZANZIBAR", "XYLOPHONE", "QQQQ"):
            assert tok.decode(tok.encode(word)) == word

    def test_no_leading_space_leaks(self, tok):
        """add_prefix_space=True must not show up in the output."""
        assert not tok.decode(tok.encode("HELLO")).startswith(" ")


class TestSpecialTokens:
    def test_pad_is_zero(self, tok):
        """So that a zero-filled tensor is padding, not an arbitrary word."""
        assert tok.special.pad == 0
        assert tok.token_id(PAD) == 0

    def test_all_special_tokens_exist(self, tok):
        for t in SPECIAL_TOKENS:
            assert isinstance(tok.token_id(t), int)

    def test_special_ids_are_distinct(self, tok):
        ids = [tok.token_id(t) for t in SPECIAL_TOKENS]
        assert len(set(ids)) == len(ids)

    def test_special_ids_are_stable(self, tok):
        """Ids get baked into checkpoints, so this ordering must not drift."""
        assert [tok.token_id(t) for t in SPECIAL_TOKENS] == list(range(len(SPECIAL_TOKENS)))

    def test_is_special_recognises_them(self, tok):
        assert tok.is_special(tok.special.sot)
        assert tok.is_special(tok.special.eot)
        assert not tok.is_special(tok.encode("HELLO")[0])

    def test_unknown_token_raises(self, tok):
        with pytest.raises(KeyError):
            tok.token_id("<|klingon|>")


class TestMultitaskFormat:
    def test_sot_sequence_order(self, tok):
        """Paper §2.3: SOT, language, task, then optionally NOTIMESTAMPS."""
        seq = tok.sot_sequence()
        assert seq == [
            tok.token_id(SOT),
            tok.token_id("<|en|>"),
            tok.token_id(TRANSCRIBE),
            tok.token_id(NOTIMESTAMPS),
        ]

    def test_translate_task_swaps_one_token(self, tok):
        """The whole point: the task is a token, not an architecture."""
        a = tok.sot_sequence(task="transcribe")
        b = tok.sot_sequence(task="translate")
        assert sum(x != y for x, y in zip(a, b)) == 1
        assert tok.token_id(TRANSLATE) in b

    def test_timestamps_flag_drops_the_notimestamps_token(self, tok):
        assert tok.token_id(NOTIMESTAMPS) not in tok.sot_sequence(timestamps=True)
        assert tok.token_id(NOTIMESTAMPS) in tok.sot_sequence(timestamps=False)

    def test_prompt_length(self, tok):
        assert tok.prompt_length == 4
        assert len(tok.sot_sequence()) == tok.prompt_length


class TestTrainingSequences:
    def test_structure_is_prefix_text_eot(self, tok):
        ids = tok.encode_training("HELLO WORLD")
        assert ids[: tok.prompt_length] == tok.sot_sequence()
        assert ids[-1] == tok.special.eot
        assert tok.decode(ids[tok.prompt_length : -1]) == "HELLO WORLD"

    def test_decodes_back_to_the_original_text(self, tok):
        for text in CORPUS[:5]:
            ids = tok.encode_training(text)
            assert tok.decode(ids) == text  # specials skipped by default

    def test_max_len_truncates_but_keeps_eot(self, tok):
        long_text = " ".join(["WORD"] * 200)
        ids = tok.encode_training(long_text, max_len=32)
        assert len(ids) == 32
        assert ids[-1] == tok.special.eot
        assert ids[: tok.prompt_length] == tok.sot_sequence()

    def test_short_sequences_are_untouched_by_max_len(self, tok):
        a = tok.encode_training("HELLO")
        b = tok.encode_training("HELLO", max_len=128)
        assert a == b


class TestVocabulary:
    def test_respects_the_requested_size(self):
        t = WhisprTokenizer.train(CORPUS, vocab_size=300)
        assert t.n_vocab <= 300

    def test_bigger_vocab_compresses_better(self):
        small = WhisprTokenizer.train(CORPUS, vocab_size=300)
        big = WhisprTokenizer.train(CORPUS, vocab_size=1000)
        assert (
            big.compression_stats(CORPUS)["tokens_per_word"]
            < small.compression_stats(CORPUS)["tokens_per_word"]
        )

    def test_common_words_become_single_tokens(self, tok):
        for word in ("THE", "OF", "IS"):
            assert len(tok.encode(f"{word}")) == 1, f"{word} should be one token"

    def test_compression_stats_shape(self, tok):
        s = tok.compression_stats(CORPUS)
        assert set(s) == {"vocab_size", "tokens", "tokens_per_word", "chars_per_token"}
        assert s["tokens_per_word"] > 0


class TestPersistence:
    def test_save_and_load_roundtrip(self, tok, tmp_path):
        path = tmp_path / "tok.json"
        tok.save(path)
        loaded = WhisprTokenizer.load(path)
        assert loaded.n_vocab == tok.n_vocab
        assert loaded.special == tok.special
        assert loaded.encode("THE QUICK BROWN FOX") == tok.encode("THE QUICK BROWN FOX")

    def test_load_or_train_trains_then_loads(self, tmp_path):
        from whispr.tokenizer import load_or_train

        path = tmp_path / "t.json"
        a = load_or_train(path, CORPUS, vocab_size=300)
        assert path.exists()
        b = load_or_train(path)  # no texts needed the second time
        assert a.encode("HELLO") == b.encode("HELLO")

    def test_load_or_train_without_texts_or_file_raises(self, tmp_path):
        from whispr.tokenizer import load_or_train

        with pytest.raises(FileNotFoundError):
            load_or_train(tmp_path / "missing.json")


def test_no_train_val_leak_is_possible_by_construction(tok):
    """The tokenizer only ever sees what it is handed.

    Not a property of the class so much as a reminder of the contract:
    scripts/05_tokenizer.py passes training transcripts only, because the merge
    list would otherwise encode which word pieces occur in the held-out set.
    """
    only_train = WhisprTokenizer.train(["AAAA BBBB"] * 20, vocab_size=300)
    assert only_train.decode(only_train.encode("ZZZZ")) == "ZZZZ"  # still representable
