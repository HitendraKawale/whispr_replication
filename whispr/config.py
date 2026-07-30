"""Configuration for the model and data pipeline.

Every field that *deviates* from the paper says so and says why. Everything
else is the paper's value. Keeping this in one file makes the replication's
compromises auditable rather than scattered through the code.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from whispr.audio import SAMPLE_RATE
from whispr.mel import HOP_LENGTH, N_MELS


@dataclass(frozen=True)
class AudioConfig:
    """The frontend. Identical to the paper except for the window length."""

    sample_rate: int = SAMPLE_RATE
    n_mels: int = N_MELS
    hop_length: int = HOP_LENGTH

    # DEVIATION. The paper uses a fixed 30-second window (3000 frames, 1500
    # encoder positions). LibriSpeech dev-clean has a median utterance of 5.9 s,
    # so a 30 s window would spend 76% of the encoder's compute on zero padding,
    # and attention cost is quadratic in that length.
    #
    #   window  utts kept  audio kept  padding waste  encoder positions
    #     30 s     99.7%       5.31 h        76%           1500
    #     15 s     93.0%       4.36 h        58%            750
    #     10 s     79.0%       3.12 h        47%            500
    #
    # 15 s keeps 93% of the corpus at a quarter of the attention cost. The
    # architecture is unchanged — only this constant moves — so scaling back to
    # 30 s is a one-line edit if you have the compute.
    window_seconds: float = 15.0

    @property
    def n_samples(self) -> int:
        return int(self.window_seconds * self.sample_rate)

    @property
    def n_frames(self) -> int:
        """Mel frames per window — the encoder's input length before the stem."""
        return self.n_samples // self.hop_length

    @property
    def n_audio_ctx(self) -> int:
        """Encoder positions, after the conv stem's stride-2 layer halves it."""
        return self.n_frames // 2


@dataclass(frozen=True)
class ModelConfig:
    """Whisper Tiny (paper Table 1: 4 layers, width 384, 6 heads, 39M params).

    `n_vocab` is a deviation: the paper reuses GPT-2's 50,257-token byte-level
    BPE, which for a 39M-parameter model would put ~19M parameters (half the
    model) in the embedding table alone — and 50k tokens is absurd for a corpus
    with a 28-character alphabet. We train our own BPE in step 5.

    2048 was chosen by measuring, not guessing (scripts/05_tokenizer.py):

        vocab   tokens/word   embedding params
          512       2.28          196,608
         1024       1.76          393,216
         2048       1.46          786,432     <- here
         4096       1.25        1,572,864

    Past 2048 the compression gains flatten while the embedding table doubles,
    and with only 46k words of training text the extra tokens would each be
    seen a handful of times.

    `n_text_ctx` is 128 rather than the paper's 448: the longest tokenised
    utterance in our (<=15 s) corpus is 94 tokens.
    """

    n_mels: int = N_MELS
    n_vocab: int = 2048
    n_audio_ctx: int = 750  # set from AudioConfig; 1500 in the paper
    n_audio_state: int = 384
    n_audio_head: int = 6
    n_audio_layer: int = 4
    n_text_ctx: int = 128  # max decoder positions; the paper uses 448
    n_text_state: int = 384
    n_text_head: int = 6
    n_text_layer: int = 4

    @property
    def is_tiny_shaped(self) -> bool:
        """Does this match Table 1's Tiny row?"""
        return (
            self.n_audio_state == 384
            and self.n_audio_head == 6
            and self.n_audio_layer == 4
            and self.n_text_layer == 4
        )


@dataclass(frozen=True)
class TrainConfig:
    """Optimizer settings.

    Betas, epsilon, weight decay, grad clipping and the warmup-then-linear-decay
    schedule are exactly the paper's (Table 17). Only the scale is reduced:
    the paper does 2^20 updates at batch 256 across many GPUs.
    """

    # Paper values, unchanged.
    betas: tuple[float, float] = (0.9, 0.98)
    eps: float = 1e-6
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0

    # DEVIATION: scale. Paper is 1,048,576 updates at batch size 256, which is
    # 2-3 passes over 680,000 hours. We have 2,117 utterances (265 steps per
    # epoch at batch 8), so 8,000 updates is already ~30 epochs. Overfitting is
    # the binding constraint, not underfitting — hence validation-based
    # checkpoint selection rather than simply training longer.
    max_updates: int = 8_000

    # Throughput on MPS is flat at ~16 utts/s from batch 2 to 16 (see
    # notes/06), so batch size costs nothing in throughput and is purely a
    # gradient-noise choice.
    batch_size: int = 8

    # DEVIATION: the paper warms up over 2,048 updates out of 1,048,576 — 0.2%
    # of training. Copying 2,048 here would spend 26% of our run ramping up.
    # Scaled to the same *fraction* would be ~16 steps, which is too few for
    # Adam's second-moment estimate to settle, so we use 500 (6%).
    warmup_updates: int = 500

    # Paper's max LR for Tiny is 1.5e-3 at batch 256. We use batch 8, so a
    # proportionally smaller LR is the right starting point.
    learning_rate: float = 5e-4

    seed: int = 1234


@dataclass(frozen=True)
class Config:
    audio: AudioConfig = field(default_factory=AudioConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def __post_init__(self) -> None:
        # The model's encoder length must match what the frontend produces.
        if self.model.n_audio_ctx != self.audio.n_audio_ctx:
            raise ValueError(
                f"model.n_audio_ctx={self.model.n_audio_ctx} but the frontend "
                f"produces {self.audio.n_audio_ctx} positions "
                f"({self.audio.window_seconds}s window). These must agree."
            )


def default_config() -> Config:
    return Config()
