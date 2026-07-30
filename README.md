# whispr — replicating Whisper from scratch on an M1 MacBook

A step-by-step, **from-scratch** replication of
[*Robust Speech Recognition via Large-Scale Weak Supervision*](paper/whisper_2212.04356.pdf)
(Radford et al., 2022 — the Whisper paper), built small enough to train on a
16 GB Apple M1 while staying architecturally faithful to the paper.

This repo is as much a **course in audio ML** as it is a model. Every step ships
a note explaining the signal-processing or modelling idea, a runnable script that
draws the picture, and a test that pins the behaviour down.

## The premise

Whisper is, stripped of its scale, a surprisingly plain idea:

> Turn 30 seconds of audio into an 80-channel log-mel spectrogram, feed it to a
> vanilla encoder–decoder Transformer, and train it to *predict the next text
> token*. No CTC, no forced alignment, no pronunciation lexicon, no HMM.

The paper's contribution is the **data** (680,000 hours of weakly-supervised
audio) and the **multitask token format**, not exotic architecture. That is
excellent news for a replication: the architecture is reproducible exactly, and
we can study what falls out when the data is 5 hours instead of 680,000.

## What we can and cannot replicate

| Paper | This repo | Why |
|---|---|---|
| 680,000 h weakly-supervised, multilingual | ~5 h LibriSpeech `dev-clean` (then optionally 100 h) | One M1, no data pipeline for web-scale audio |
| 39M–1550M params (Tiny→Large) | ~size of Tiny (39M), configurable smaller | Memory + time |
| 2^20 updates, batch 256, FP16 on many GPUs | ~10^4 updates, batch 8–16, fp32 on MPS | One M1 |
| **Zero-shot** eval on unseen datasets | **In-distribution** eval on a held-out split | Zero-shot is a property of the data scale, not the code |

The honest framing: **we replicate the method exactly and the results
directionally.** A 5-hour model will not be robust — and *that gap is the
paper's actual thesis*, which we get to demonstrate rather than just read about.

## Quickstart

```bash
uv sync                      # creates .venv with Python 3.12 + PyTorch (MPS)
uv run python -c "import torch; print(torch.backends.mps.is_available())"
```

Then walk the steps in order — each is self-contained and writes figures to
`figures/`:

```bash
uv run python scripts/01_sound_basics.py
```

## Roadmap

Each step is one commit, with a note, a runnable script, and tests.
See [`notes/00-roadmap.md`](notes/00-roadmap.md).

- [x] **0** — [Scaffold, paper, environment](notes/00-roadmap.md)
- [x] **1** — [Sound as numbers](notes/01-sound-and-sampling.md): sampling, Nyquist, aliasing
- [x] **2** — [The DFT and STFT](notes/02-dft-and-stft.md), implemented by hand
- [x] **3** — [Whisper's exact log-mel frontend](notes/03-mel-and-frontend.md) — verified to 1.2e-7
- [x] **4** — [LibriSpeech pipeline](notes/04-data.md) with a speaker-disjoint split
- [x] **5** — [BPE tokenizer](notes/05-tokenizer.md) + the multitask token format
- [x] **6** — [The Transformer](notes/06-model.md) — verified **bit-identical** to OpenAI's
- [x] **7** — [Training on MPS](notes/07-training.md)
- [x] **8** — [Decoding and WER](notes/08-decoding-and-wer.md)

## Verification: is this actually Whisper?

The two claims worth checking, both reproducible from the repo:

| Component | Compared against | Max difference |
|---|---|---|
| 80-mel filterbank | the array shipped in `openai/whisper` | **1.9e-9** |
| Full log-mel frontend | `whisper.log_mel_spectrogram` | **1.2e-7** |
| Encoder output | real `whisper-tiny` weights in our class | **0.0** |
| Decoder logits | same | **0.0** |
| Parameter count (tiny dims) | the released checkpoint | **exact** (37.18M) |

Loading OpenAI's `whisper-tiny` checkpoint into `whispr.model.Whispr` with
`strict=True` succeeds and produces **bit-identical** outputs. The
implementation is not merely similar to Whisper's — it is the same computation.

```bash
uv run --with openai-whisper --with "numba>=0.61" pytest tests/ -v
```

(One useful by-product: Table 1's parameter counts are rounded up ~5%. Tiny is
really 37.18M, not 39M.)

## Layout

```
notes/      the course — one markdown note per step
whispr/     the library — importable, tested
scripts/    runnable demos that produce figures/
tests/      pins numerical behaviour against references
paper/      the paper itself
```

## Reference facts from the paper (so we don't drift)

Frontend: 16 kHz mono · 80 mel channels · 25 ms window · 10 ms stride ·
30 s fixed input → 3000 frames · globally scaled to roughly [-1, 1].

Architecture: 2× Conv1d stem (kernel 3, GELU, second has stride 2) →
**sinusoidal** positional encoding → pre-activation residual Transformer blocks
→ final LayerNorm on the encoder. Decoder uses **learned** positional embeddings
and **tied** input/output token embeddings.

Tiny: 4 layers, width 384, 6 heads, 39M params, max LR 1.5e-3.

Training: AdamW (β₁=0.9, β₂=0.98, ε=1e-6), weight decay 0.1, max grad norm 1.0,
linear decay to zero after 2048 warmup updates.

## Licence

MIT. The paper PDF is redistributed under arXiv's non-exclusive licence and is
the property of its authors.
