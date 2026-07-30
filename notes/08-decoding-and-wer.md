# Step 8 — Decoding and WER

> Goal: turn logits into text, and measure the result in a way that is
> comparable to published numbers rather than flattering to us.

## 1. Decoding is a search problem

Training gives us `P(token_t | audio, tokens_<t)`. Getting a *transcript* means
finding the sequence that maximises the product of those probabilities — and
there are `2048^T` candidate sequences. We approximate.

### Greedy

Take the most likely token at each step.

```
step 1:  P(THE)=0.7  P(A)=0.2  ...  -> THE
step 2:  P(CAT)=0.5  P(CAR)=0.3 ...  -> CAT
```

Fast, one forward pass per token, and usually fine. Its failure mode is
specific: **it cannot recover from a bad early choice.** Once a wrong token is
emitted the model conditions on its own mistake, and the rest of the sentence
often has to be wrong to stay grammatical.

### Beam search

Keep the `k` best *partial sequences* rather than the one best token.

```
        ┌─ THE CAT  (-0.9)   ← kept
THE ────┤
        └─ THE CAR  (-1.4)   ← kept
        ┌─ A CAT    (-2.1)   ← kept
A ──────┤
        └─ A CAR    (-3.0)   ← pruned
```

Greedy picks the best token; beam search approximates picking the best
*sequence*. That distinction matters in speech, where the acoustic evidence for
a word often doesn't resolve until a syllable or two later — exactly the case
where a locally-unlikely token leads somewhere much better.

### Length normalisation, or beam search says nothing

Every token adds a negative log-probability, so longer sequences always score
worse. Unnormalised beam search therefore has a systematic preference for
**stopping early**, and its favourite transcript is the empty one.

```python
candidates.sort(key=lambda x: x[1] / (len(x[0]) ** length_penalty), reverse=True)
```

Dividing by length removes the bias. With `length_penalty=1.0` we rank by mean
log-probability per token. This is not a refinement — omit it and beam search is
worse than greedy.

## 2. The KV cache

Naively, generating token `t` re-runs the decoder over all `t` previous tokens:
O(T²) total work for a T-token transcript.

But the keys and values for tokens `0..t-1` don't change when token `t` arrives —
the causal mask guarantees earlier positions can't see later ones. So cache them.

Two kinds of entry, with different lifetimes:

| Cache | Depends on | Grows |
|---|---|---|
| Self-attention K/V | the tokens so far | +1 position per step |
| **Cross-attention K/V** | **the audio only** | **never — computed once** |

The second is the bigger win. The encoder output is fixed for the whole
utterance, so its 850 keys and values are computed **once** and reused for every
decoding step. That is why the encoder/decoder split is efficient at inference
and not just architecturally tidy.

Measured, 4-layer decoder, 120 generated tokens: **136 ms cached vs 283 ms
uncached**, with byte-identical output. `greedy(mel, use_cache=False)` keeps the
naive path available, and a test asserts the two agree — a cache is an
optimisation, so it must change nothing.

### Two things the implementation forces you to get right

**The causal mask must be skipped when the query length is 1.** During
incremental decoding `q` has one row while `k` and `v` span the whole history, so
`mask[:1, :1]` is the wrong shape. It's also unnecessary: a single query's every
visible key is already in its past.

**Position offsets can't be inferred from the cache.** The cache holds growing
self-attention entries *and* fixed-length cross-attention ones, so "look at the
first cached tensor's length" is fragile. Pass the offset explicitly.

Also: the hooks that grow the cache must be **removed** when decoding finishes,
or they keep concatenating into the next utterance and silently corrupt it.
Hence a `model.kv_cache()` context manager rather than a bare install call.

### Beam search deliberately doesn't cache

Beams are re-ranked and pruned at every step, so the cache would have to be
re-indexed by beam ancestry after each prune. That's real bookkeeping, and it
would bury the four lines that make beam search *be* beam search. Since the beam
implementation here exists to explain the idea, it recomputes the prefix.
Cross-attention K/V are still computed once per utterance either way.

## 3. WER, and why the normaliser matters more than you'd think

**Word Error Rate** is edit distance at the word level:

```
WER = (Substitutions + Deletions + Insertions) / Words_in_reference
```

Three things about it that are easy to get wrong:

**It is not a percentage of anything.** Insertions are unbounded, so WER can
exceed 100%. A model that emits a hundred words for a five-word reference has a
WER of about 2000%.

**Aggregate over the corpus, never average per utterance.** A one-word utterance
transcribed wrong is 100% WER. Averaging per-utterance lets short utterances
dominate:

```
refs: "HELLO"  +  a 9-word sentence, transcribed perfectly
corpus-level : 1 error / 10 words          =  10%
per-utterance: (100% + 0%) / 2             =  50%   ← wrong
```

**Normalise first.** The paper spends all of §3.2 and Appendix C on this, and
reports WER drops "of up to 50 percent" from normalisation alone. If the
reference says `MISTER` and the model says `Mr.`, that is not a recognition
error — it's a formatting difference, and counting it measures transcript
convention rather than speech recognition.

Our labels are already uppercase and unpunctuated, so our normaliser is short
(uppercase, strip anything outside `A-Z ' `, collapse whitespace). The principle
still holds, and `word_error_rate` applies it to both sides.

Cross-checked against `jiwer` to within 1e-9.

## 4. Results

*Filled in from the actual run — see `results/wer.json` and
`figures/08_*.png`.*

The number that matters is the **held-out speakers** one. The training-speaker
number is reported alongside it purely to show the size of the generalisation
gap, which at 3.7 hours of data is the interesting quantity.

### What to expect, and why

A useful frame before looking: our model has seen **3.69 hours** of read
audiobook speech from 34 speakers. Whisper-tiny saw **680,000 hours** across
thousands of conditions — 184,000× more. LibriSpeech-trained systems reach
~5% WER on test-clean, but they use 960 hours (260× ours) plus a language model.

So the expected outcome is a model that has clearly learned *something* —
English phonotactics, common words, the shape of the mapping — and produces
fluent-looking output that is frequently wrong. That is what 3.7 hours buys, and
it is the paper's thesis rendered as a measurement rather than a claim.

## 5. Reading the error analysis

`figures/08_wer_distribution.png` has three panels worth reading carefully:

- **WER histogram.** Look at the *shape*, not just the mean. A bimodal
  distribution (some utterances nearly right, others total nonsense) means
  something different from a uniform smear.
- **WER vs reference length.** Longer utterances should be worse: more tokens,
  more chances for greedy decoding to derail irrecoverably.
- **WER vs the model's own average log-probability.** If confidence tracks
  correctness, the model has usable uncertainty and you can threshold on it. If
  it doesn't, the model is confidently wrong — which is the same pathology
  behind Whisper's hallucination problem at scale (§2.4).

## Run it

```bash
uv run python scripts/08_evaluate.py                    # greedy, full held-out set
uv run python scripts/08_evaluate.py --beam 5           # beam search
uv run python scripts/08_evaluate.py --compare --limit 60  # greedy vs beam
```
