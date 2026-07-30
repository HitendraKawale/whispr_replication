# Step 7 — Training on an M1

> Goal: get the loss down, and know *why* every hyperparameter has the value it
> has.

## 1. Before training: two checks that cost minutes and save hours

### Check 1 — is the initial loss what theory says?

A model that knows nothing predicts a uniform distribution over the vocabulary.
Cross-entropy of a uniform distribution over `V` classes is `ln(V)`:

```
ln(2048) = 7.625      measured first loss: 8.08
```

Close enough (initialisation isn't perfectly uniform). If it had come out at 0.5
the labels would be leaking; at 50 the initialisation would be broken; at
`ln(51865)` the vocabulary would be wrong. **One number rules out a whole family
of bugs.**

### Check 2 — can it overfit one batch?

The single most valuable sanity check in deep learning. Take 8 examples and
train on them until the model memorises them. A correct model *must* be able to:

```
step    0  loss 8.0798
step   10  loss 4.2616
step   25  loss 0.0933
step   50  loss 0.0467
step  100  loss 0.0046
step  149  loss 0.0002      PASS
```

If this plateaus, something is structurally broken — a detached gradient, an
inverted mask, a misaligned label — and no hyperparameter will fix it. Because
this passes, every later problem is an *optimisation* problem, not a plumbing
problem.

```bash
uv run python scripts/07_train.py --sanity
```

## 2. The hyperparameters, and which ones we changed

From paper Table 17, **unchanged**:

| Setting | Value | Why it is what it is |
|---|---|---|
| Optimiser | AdamW | Adam + decoupled weight decay |
| β₁, β₂ | 0.9, 0.98 | β₂=0.98 (not 0.999) — shorter memory, standard for Transformers |
| ε | 1e-6 | Larger than Adam's 1e-8 default; guards against tiny second moments |
| Weight decay | 0.1 | |
| Max grad norm | 1.0 | |
| Schedule | linear decay to 0 after warmup | |
| Init | Gaussian fan-in, std = 1/√fan_in | Keeps activation variance constant with depth |

**Changed**, and each one is in `whispr/config.py` with its reasoning:

| Setting | Paper | Ours | Why |
|---|---|---|---|
| Updates | 1,048,576 | 8,000 | 265 steps/epoch here — 8k is already ~30 epochs |
| Batch size | 256 | 8 | Throughput is flat on MPS (notes/06), so batch is free |
| Max LR | 1.5e-3 | 5e-4 | Scaled down with the batch size |
| Warmup | 2,048 | 500 | See below — this one is a trap |

### The warmup trap

The paper warms up over 2,048 updates out of 1,048,576 — **0.2%** of training.
Copy `2048` into an 8,000-step run and you spend **26%** of training ramping up.
Scale by the same *fraction* instead and you get 16 steps, which is too few for
Adam's second-moment estimate to settle.

Neither the absolute value nor the relative one transfers. 500 (6%) is a
judgement call, and the point is that it *has* to be one.

### Why warmup exists at all

Adam divides by a running estimate of gradient magnitude. In the first steps
that estimate is built from almost no data and is unreliable, so a full learning
rate produces enormous, badly-scaled updates that can permanently damage the
model before it has learned anything. Ramping up lets the statistics stabilise
while the steps are small.

## 3. Weight decay, applied selectively

Not in the paper, but decaying everything measurably hurts:

```python
if param.dim() < 2 or "embedding" in name:
    no_decay.append(param)      # biases, LayerNorm gains, embeddings
else:
    decay.append(param)         # weight matrices
```

Weight decay is a prior that says "smaller weights generalise better". That is a
sensible prior for a weight *matrix*. It is not sensible for a LayerNorm gain
(whose job is to be the right scale) or a bias (whose job is to be the right
offset), and pulling embeddings toward zero just makes rare tokens
indistinguishable from each other.

## 4. The loss must not reward predicting a constant

Whisper computes the loss over the task-specification tokens too — predicting
`<|en|>` *is* language identification, so it's a real task there.

In our setup the prefix is `[SOT, EN, TRANSCRIBE, NOTIMESTAMPS]` — **the same
every time**. Training on it hands the model three free correct predictions per
utterance. With a median of 28 tokens, that deflates the reported loss by
roughly 10% while measuring nothing.

So we mask them (`collate(..., mask_prefix=...)`) and report a loss that is
entirely about transcription. This is a deliberate deviation, made for honest
reporting rather than better results — the masked loss is a *worse* number.

## 5. Overfitting is the design constraint, not a risk

2,117 training utterances, 3.69 hours. At batch 8 that's 265 steps per epoch, so
8,000 updates is **~30 passes over the data**. The paper does 2–3 passes over
680,000 hours.

The paper can therefore say (§2.4) "over-fitting is not a large concern, and we
do not use any data augmentation or regularization". We do not have that luxury.
So:

- **±6 dB gain jitter** on training audio (justified in notes/04 — the frontend
  is provably not loudness invariant, so this teaches something real).
- **Validation on 6 unseen speakers**, evaluated every 250 steps.
- **`best.pt` is saved on validation improvement.** The final model is *not* the
  last step; it's the best-generalising one. With this much overfitting, that
  distinction is the difference between a usable model and a memoriser.

## 6. What to watch in the curves

`figures/07_training_curves.png`:

- **The train/val gap** is overfitting, and it should be large. That's the
  paper's thesis showing up as a picture.
- **Gradient norm vs the clip at 1.0.** Early on, norms exceed 1.0 and clipping
  is active — that's what it's for. If clipping is *still* active late in
  training, the learning rate is too high.
- **A loss plateau around 2–3 early on** is normal. The model first learns the
  unigram distribution of English text (which needs no audio at all), and only
  then starts using the encoder. Loss falling to ~3 and sitting there for a
  while is the model being a language model before it becomes a speech
  recogniser.

## 7. What actually limits training here — and a wrong guess, corrected

I originally wrote that data loading was **~35%** of step time, inferred from the
gap between step 6's model benchmark (~0.64 s) and the observed training rate
(~1.16 s). That inference was wrong, and it is worth leaving the correction in
rather than quietly editing it away.

Profiled properly (batch 8, 17 s window, MPS, cool machine):

| Component | Time | Share |
|---|---|---|
| Backward pass | 472 ms | 67% |
| Forward pass | 137 ms | 19% |
| Grad clip + `optimizer.step()` | 68 ms | 10% |
| `.item()` / `float()` GPU sync | 25 ms | 4% |
| **FLAC decode + mel (8 files)** | **33 ms** | **4.5%** |
| Precomputed mel read | 2 ms | 0.3% |

Data loading was never a third of anything. It's **4.5%**, and the backward pass
is two thirds. The lesson is the obvious one I failed to apply: *measure the
parts, don't subtract two totals.* The gap I was attributing to data loading was
mostly not there in the first place.

### Where the missing ~400 ms went

`735 ms` of measured step still doesn't reach the `~1,160 ms` observed during the
long run. The most likely explanation is **sustained-load thermal throttling** —
the profile above is a cool machine doing 12 iterations, while the real run was
five hours in and the laptop was hot enough to be noticeable by hand. I haven't
instrumented clock speeds to prove it, so this stays a hypothesis rather than a
measurement.

Which is itself useful to know: on a fan-limited laptop, **your effective step
time is a function of how long you've been training**, and a short benchmark will
always flatter you.

### What the mel cache actually buys

`whispr/melcache.py` precomputes log-mels into a memory-mapped float16 array
(7.75 GB for train-clean-100 at 17 s, 0.70 GB for dev-clean). Measured:

| | before | after |
|---|---|---|
| Data loading, batch of 8 | 33 ms | **2 ms** (20× faster) |
| Training step | 735 ms | **704 ms** (~4% faster) |
| Full dev-clean validation | 61 s | **51 s** (17% faster) |

So: a 20× speedup on something that was 4.5% of the cost. **Training gets ~4%
faster, not the 1.5× I predicted.** Validation gains more because it has no
backward pass to hide the loading behind.

Whether that justifies 7.9 GB of disk is a real question. Reasons it might:
repeated experiments pay the frontend cost once instead of every epoch; the CPU
work disappears, which on a thermally-limited machine may matter more than the
4% suggests; and it makes `num_workers=0` free rather than a compromise.

One nice property: it stays compatible with gain augmentation. Because scaling
the waveform by `k` decibels is *exactly* a `db/40` offset on the finished
log-mel (verified to 2e-5 for −20…+12 dB — the relative floor shifts with the
peak), augmentation is a scalar add rather than a reason to re-run the frontend.
That identity falls straight out of step 3's two-floors analysis, and it breaks
below about −30 dB where the absolute clamp takes over — well outside our ±6 dB.

### And validation, which was a real problem

A full dev-clean pass is 324 batches; at a 250-step eval interval it cost more
than the training it was measuring. Capping the periodic eval at 40 batches (320
utterances) estimates the loss to within a few hundredths — plenty for choosing a
checkpoint — and the full sweep still runs once at the end. That fix was worth
far more than the cache.

## 8. Notes on MPS

- `num_workers=0` in the DataLoader. MPS plus forked workers is a reliable
  source of hangs.
- LayerNorm forced to fp32 (see notes/06). Half-precision layer norm on MPS
  loses enough resolution to destabilise training.
- No AMP. `torch.autocast` support on MPS is patchier than on CUDA, and at this
  model size we are not memory-bound, so there is little to gain and a class of
  numerical bugs to avoid.
- `PYTORCH_ENABLE_MPS_FALLBACK=1` (set in `whispr/device.py`) so any op without
  a Metal kernel silently falls back to CPU rather than raising.

## Run it

```bash
uv run python scripts/07_train.py --sanity   # do this first, always
uv run python scripts/07_train.py
uv run python scripts/07_train.py --plot
```
