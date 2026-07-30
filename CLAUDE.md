# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A from-scratch replication of the Whisper paper (arXiv:2212.04356, PDF in
`paper/`) sized to train on an Apple M1. It is deliberately structured as a
**course**: nine steps, each with a note in `notes/`, a figure-producing script
in `scripts/`, and tests. The notes are the primary documentation — read the
relevant one before changing a module.

## Commands

```bash
uv sync                                  # Python 3.12 venv (PyTorch has no 3.14 wheels)
uv run pytest -q                         # 222 tests, ~15 s
uv run pytest tests/test_mel.py -q       # one file
uv run pytest -q -k "kv_cache"           # one test by name
uv run pytest -q -p no:warnings          # suppress a noisy pytest deprecation

# The reference-implementation tests (9 of them) skip without openai-whisper.
# Its own pins resolve to a numba too old for 3.12, hence the explicit constraint:
uv run --with openai-whisper --with "numba>=0.61" --with "llvmlite>=0.44" pytest -q
```

Pipeline, in order (each writes to `figures/`):

```bash
uv run python scripts/04_data.py --download                      # dev-clean, 322 MB
uv run python scripts/04_data.py --download --split train-clean-100  # 6 GB
uv run python scripts/07_train.py --sanity                       # ALWAYS run first
uv run python scripts/07_train.py                                # 3.7 h baseline
uv run python scripts/07_train.py --corpus train-clean-100 --window 17 --steps 25000
uv run python scripts/08_evaluate.py --checkpoint checkpoints/run_100h/best.pt --corpus train-clean-100
uv run python scripts/09_report.py                               # writes RESULTS.md
```

Long runs: use `PYTHONUNBUFFERED=1` or you see nothing until the process exits.
Runs are resumable — `--resume best.pt` restores model, optimizer and step, since
checkpoints carry optimizer state.

## Architecture

Strictly bottom-up; each layer assumes the one below is verified.

```
audio.py    load/resample -> mono float32 16 kHz  (soundfile, NOT torchaudio.load)
  dft.py    DFT/STFT from scratch (pedagogical; torch.stft is used in practice)
  mel.py    Slaney mel filterbank + log_mel_spectrogram -> (80, n_frames)
 data.py    LibriSpeech index, splits, Dataset, collate (teacher-forcing shift)
tokenizer.py BPE + Whisper's SOT/lang/task/notimestamps special tokens
   model.py AudioEncoder + TextDecoder (names match OpenAI's checkpoint keys)
   train.py Trainer, AdamW, warmup+linear decay, overfit_one_batch
  decode.py greedy (KV-cached) + beam, text normaliser, WER
  config.py AudioConfig / ModelConfig / TrainConfig — all deviations documented here
```

**`config.py` is the source of truth for deviations from the paper.** Every field
that differs says so and says why. Add new deviations there with reasoning, not
inline in code.

### Invariants worth not breaking

- **`AudioConfig.window_seconds` drives `n_frames` and `n_audio_ctx`.** Changing
  the window requires a matching `ModelConfig.n_audio_ctx`; `Config.__post_init__`
  raises if they disagree. The encoder also raises with a clear message.
- The conv stem's stride-2 layer means **`n_audio_ctx == n_frames // 2`**. Baseline
  is 15 s → 1500 → 750; the 100 h run is 17 s → 1700 → 850.
- **Module and parameter names in `model.py` match OpenAI's checkpoint exactly.**
  `tests/test_model.py` loads real `whisper-tiny` weights with `strict=True` and
  asserts bit-identical outputs. Renaming anything there breaks that test, which
  is the strongest correctness guarantee in the repo — don't rename casually.
- `collate` produces `tokens = seq[:-1]`, `labels = seq[1:]`, `-100` in padding,
  and (via `mask_prefix`) `-100` over the constant task prefix.
- Checkpoints store their own `config`; `scripts/08_evaluate.py` reads it back
  rather than assuming the current default. Two runs use different windows.

### Things that look like bugs but aren't

- `dft.py` is O(N²) and unused in training. It exists so `torch.stft` isn't a
  black box; a test asserts they agree.
- The mel filterbank's rows do **not** peak at 1.0 — Slaney normalisation gives
  equal area, not equal peak. If you "fix" this you stop matching Whisper.
- Filters 13 and 14 share no DFT bin. Real property of Whisper's own filterbank
  at `n_fft=400`; asserted in a test.
- The frontend is not loudness invariant, and its absolute `1e-10` clamp and
  relative `max-8` floor swap over around 1% amplitude. See `notes/03` §4b.
- `LayerNorm` upcasts to fp32 deliberately (MPS half-precision instability).
- The causal mask is skipped when query length is 1 — required for incremental
  decoding, where `q` is one row but `k`/`v` span the history.

## Conventions

- Every step ends with passing tests and one commit. Tests come with the code,
  not after.
- Numerical claims in notes and commit messages are **measured**, and the
  measurement is reproducible from a script or test. Don't state a figure you
  haven't run.
- `notes/` explains *why*; code comments explain *why here*. Neither restates
  what the code plainly does.
- Verify against something external — a closed form, a reference implementation,
  or a known-good library — before moving up a layer.

## Gotchas

- `torchaudio.load`/`save` need the separate `torchcodec` package as of 2.11.
  Use `soundfile` (already a dependency, reads FLAC).
- Running the test suite while training competes for the GPU and slows both.
- `num_workers=0` by default; data loading is ~1/3 of step time, and MPS with
  forked workers has a history of hangs. It's a config field if you want to try.
- `data/`, `checkpoints/` and `*.pt` are gitignored; `figures/` and
  `assets/whisper_mel_filters.npz` (the golden reference array) are committed.
