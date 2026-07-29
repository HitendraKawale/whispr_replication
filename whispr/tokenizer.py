"""Byte-pair encoding with Whisper's multitask special tokens.

Whisper reuses GPT-2's 50,257-token byte-level BPE. We train our own, much
smaller one, for two reasons:

1. **Parameter budget.** The decoder ties input and output embeddings, so the
   vocabulary costs `n_vocab x 384` parameters. At 50,257 that is 19.3M — half
   of Tiny's 39M budget spent on an embedding table for a corpus with a
   28-character alphabet.
2. **Coverage.** GPT-2's vocabulary was fitted to web text: mixed case,
   punctuation, code, markup. LibriSpeech labels are uppercase letters, spaces
   and apostrophes. Almost all of GPT-2's vocabulary would never be emitted,
   and the tokens we do use would be badly fitted to uppercase text.

The *format* — the special-token scheme from paper §2.3 — is reproduced exactly,
because that is the interesting part of Whisper's tokenizer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

# Whisper's special tokens (paper §2.3, Figure 1). We keep the same names and
# the same ordering convention: the sequence always begins with SOT, then a
# language tag, then a task tag, then optionally NOTIMESTAMPS.
SOT = "<|startoftranscript|>"
EOT = "<|endoftext|>"
TRANSCRIBE = "<|transcribe|>"
TRANSLATE = "<|translate|>"
NOTIMESTAMPS = "<|notimestamps|>"
NOSPEECH = "<|nospeech|>"
PREV = "<|startofprev|>"
PAD = "<|pad|>"
LANG_EN = "<|en|>"

# Order matters only in that it must be stable across training runs — the ids
# are baked into checkpoints.
SPECIAL_TOKENS = [
    PAD,  # id 0, so a zero-filled tensor is padding
    EOT,
    SOT,
    LANG_EN,
    TRANSCRIBE,
    TRANSLATE,
    NOTIMESTAMPS,
    NOSPEECH,
    PREV,
]


@dataclass(frozen=True)
class SpecialIds:
    pad: int
    eot: int
    sot: int
    lang_en: int
    transcribe: int
    translate: int
    notimestamps: int
    nospeech: int
    prev: int


class WhisprTokenizer:
    """A small BPE plus Whisper's task-specification token scheme."""

    def __init__(self, tokenizer: Tokenizer) -> None:
        self.tokenizer = tokenizer
        self.special = SpecialIds(
            pad=self.token_id(PAD),
            eot=self.token_id(EOT),
            sot=self.token_id(SOT),
            lang_en=self.token_id(LANG_EN),
            transcribe=self.token_id(TRANSCRIBE),
            translate=self.token_id(TRANSLATE),
            notimestamps=self.token_id(NOTIMESTAMPS),
            nospeech=self.token_id(NOSPEECH),
            prev=self.token_id(PREV),
        )

    # ---------------------------------------------------------------- training

    @classmethod
    def train(
        cls,
        texts: list[str],
        vocab_size: int = 2048,
        min_frequency: int = 2,
    ) -> "WhisprTokenizer":
        """Fit a BPE on transcript text.

        Train on the *training split only*. Fitting the vocabulary on validation
        text is a subtle leak: the merges themselves encode which word pieces
        occur in the held-out set.
        """
        tokenizer = Tokenizer(models.BPE(unk_token=None))
        # Split on whitespace but keep the space attached to the following word,
        # so "THE CAT" -> ["THE", " CAT"] and detokenising is exact.
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
        tokenizer.decoder = decoders.ByteLevel()

        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=SPECIAL_TOKENS,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=False,
        )
        tokenizer.train_from_iterator(texts, trainer)
        return cls(tokenizer)

    # ------------------------------------------------------------ persistence

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save(str(path))

    @classmethod
    def load(cls, path: str | Path) -> "WhisprTokenizer":
        return cls(Tokenizer.from_file(str(path)))

    # --------------------------------------------------------------- encoding

    @property
    def n_vocab(self) -> int:
        return self.tokenizer.get_vocab_size()

    def token_id(self, token: str) -> int:
        tid = self.tokenizer.token_to_id(token)
        if tid is None:
            raise KeyError(f"token {token!r} is not in the vocabulary")
        return tid

    def encode(self, text: str) -> list[int]:
        """Text -> token ids, with no special tokens."""
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        """Token ids -> text.

        `add_prefix_space=True` means the first word is stored as " THE" rather
        than "THE", so that one token serves both sentence-initial and
        mid-sentence positions. That is good for vocabulary efficiency but
        leaves a leading space on decode, which we strip here so that
        `decode(encode(t)) == t`.
        """
        text = self.tokenizer.decode(list(ids), skip_special_tokens=skip_special)
        return text[1:] if text.startswith(" ") else text

    def sot_sequence(self, task: str = "transcribe", timestamps: bool = False) -> list[int]:
        """The prompt prefix that tells the decoder what job to do.

        Paper §2.3: SOT, then the language tag, then the task, then optionally
        NOTIMESTAMPS. This is Whisper's central trick — one model does
        transcription, translation, language ID and voice activity detection,
        selected purely by which tokens you seed the decoder with.
        """
        seq = [self.special.sot, self.special.lang_en]
        seq.append(self.special.translate if task == "translate" else self.special.transcribe)
        if not timestamps:
            seq.append(self.special.notimestamps)
        return seq

    def encode_training(
        self,
        text: str,
        task: str = "transcribe",
        max_len: int | None = None,
    ) -> list[int]:
        """A full training target: prefix + text + EOT.

        The model is trained to predict every token after the SOT sequence. The
        prefix tokens are inputs, not predictions — masking that is the caller's
        job (see `prompt_length`).
        """
        ids = self.sot_sequence(task) + self.encode(text) + [self.special.eot]
        if max_len is not None and len(ids) > max_len:
            # Keep the prefix and the EOT; drop text from the middle-end.
            ids = ids[: max_len - 1] + [self.special.eot]
        return ids

    @property
    def prompt_length(self) -> int:
        """How many leading tokens are task specification rather than content."""
        return len(self.sot_sequence())

    def is_special(self, token_id: int) -> bool:
        return token_id in vars(self.special).values()

    # -------------------------------------------------------------- reporting

    def compression_stats(self, texts: list[str]) -> dict:
        """Tokens per word and per character — how well the vocabulary fits."""
        n_tokens = sum(len(self.encode(t)) for t in texts)
        n_words = sum(len(t.split()) for t in texts)
        n_chars = sum(len(t) for t in texts)
        return {
            "vocab_size": self.n_vocab,
            "tokens": n_tokens,
            "tokens_per_word": n_tokens / max(n_words, 1),
            "chars_per_token": n_chars / max(n_tokens, 1),
        }


def load_or_train(
    path: str | Path,
    texts: list[str] | None = None,
    vocab_size: int = 2048,
) -> WhisprTokenizer:
    """Load a saved tokenizer, or fit and save one."""
    path = Path(path)
    if path.exists():
        return WhisprTokenizer.load(path)
    if texts is None:
        raise FileNotFoundError(f"{path} not found and no texts given to train on")
    tok = WhisprTokenizer.train(texts, vocab_size=vocab_size)
    tok.save(path)
    return tok
