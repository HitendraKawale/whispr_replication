# Step 4 — The data pipeline

> Goal: a clean, honest corpus and an evaluation split that can't flatter us.

## 1. What LibriSpeech dev-clean is

Read speech from LibriVox public-domain audiobooks, segmented and aligned.
"dev-clean" is the development split of the higher-quality half.

```
2,703 utterances · 5.39 hours · 40 speakers · 54,402 words
16 kHz mono FLAC · median 5.9 s · max 32.6 s
alphabet: A-Z, apostrophe, space  (28 characters — no punctuation, no digits)
```

Layout:

```
dev-clean/1272/128104/1272-128104-0000.flac
dev-clean/1272/128104/1272-128104.trans.txt
   └ speaker  └ chapter        └ "1272-128104-0000 MISTER QUILTER IS THE APOSTLE OF..."
```

Already 16 kHz mono, so step 1's resampler has nothing to do here. That's
convenient and also unrepresentative — it's a big part of why LibriSpeech is
an easy benchmark.

### What this corpus is *not*

Worth being blunt, because it sets the ceiling on everything that follows:

- **One domain.** Audiobooks. Read, rehearsed, articulate.
- **No noise, no overlap, no spontaneous speech**, no disfluencies.
- **No punctuation or casing** in the labels, so the model can't learn them.
- **5 hours.** Whisper trained on 680,000 — a factor of **126,000×**.

We are training on 0.0008% of the paper's data. The paper's central claim is
that robustness comes from data scale and diversity. We have neither, so we
should expect to reproduce the *method* and not the *robustness* — and being
able to measure that gap is the point.

## 2. The window-length deviation

This is the one real engineering compromise in the replication, so here it is
in full.

Whisper uses a **fixed 30-second** input: always 3000 mel frames, always 1500
encoder positions, short utterances zero-padded. That's clean, and it's
affordable when you have a GPU fleet and 30-second segments that are actually
full of speech.

LibriSpeech utterances have a median of 5.9 seconds. Padding them to 30 s means
the encoder spends most of its time attending to zeros — and attention is
quadratic in sequence length:

| Window | Utts kept | Audio kept | Padding waste | Encoder positions | Relative attention cost |
|---|---|---|---|---|---|
| 30 s (paper) | 99.7% | 5.31 h | 76% | 1500 | 4.0× |
| **15 s (ours)** | **93.0%** | **4.36 h** | **58%** | **750** | **1.0×** |
| 10 s | 79.0% | 3.12 h | 47% | 500 | 0.44× |

15 s keeps 93% of the corpus for a quarter of the attention cost. Only the
constant moves — `AudioConfig.window_seconds` — the architecture is untouched,
so going back to 30 s is a one-line edit for anyone with the compute.

### Dropped, not truncated

Utterances longer than the window are **discarded**, not cut short. Truncating
audio while keeping the whole transcript would train the model to produce words
it can no longer hear — a direct recipe for the hallucination failure the paper
describes in §2.4. Better to lose 7% of the data than to teach the model to
invent.

After dropping: **2,117 training utterances, 3.69 hours.**

## 3. The split: speaker-disjoint, deliberately

The tempting split is random-by-utterance. It is also close to worthless here.

LibriSpeech has 40 speakers reading long books. A random split puts the *same
speaker reading the same chapter* on both sides. The model memorises the voice
and the vocabulary of that specific book, and validation loss drops for reasons
that have nothing to do with speech recognition.

So we hold out **whole speakers**:

```
train : 2,280 utts · 4.58 h · 34 speakers
val   :   423 utts · 0.81 h ·  6 speakers   (no overlap)
```

Validation now measures "can it transcribe a voice it has never heard", which
is the question worth asking. It will produce a worse number. That's the point
of an honest number.

## 4. Augmentation: a small, justified departure

The paper uses **no augmentation and no regularisation** (§2.4) — with 680k
hours, diversity comes from the data and overfitting isn't the binding
constraint. At 3.69 hours it very much is.

We apply one thing: **gain jitter of ±6 dB**. The choice is directly motivated
by step 3's finding that the frontend is *not* loudness invariant — the same
sentence at two levels produces two genuinely different tensors. So level
robustness is something the model must learn from data, and varying the level
is teaching it something the preprocessing does not hand over.

Deliberately *not* doing SpecAugment yet: the paper only added it for Large-V2,
and it's better to first see how far we get without, so its effect is
measurable rather than assumed.

## 5. Batching

Mels are a fixed `(80, 1500)`, so they stack directly. Token sequences vary, so
`collate` pads them and builds labels with `-100` in the pad positions —
`cross_entropy` ignores that index, so padding contributes no gradient.

The teacher-forcing shift also happens here:

```python
out["tokens"] = tokens[:, :-1]   # decoder input
out["labels"] = labels[:, 1:]    # what it should predict
```

Predict token `t+1` from tokens `0..t`. Getting this off by one is a classic
silent bug: the model learns to copy its input, loss looks implausibly good,
and generation produces garbage.

## What to take to step 5

- 2,117 training utterances, 3.69 h, 34 speakers; 423 held-out utterances from
  6 unseen speakers.
- The label alphabet is 28 characters. A 50,257-token GPT-2 vocabulary would be
  absurd here — which is the whole argument of the next step.

## Run it

```bash
uv run python scripts/04_data.py --download   # fetch the corpus (322 MB)
uv run python scripts/04_data.py              # stats + figures
```
