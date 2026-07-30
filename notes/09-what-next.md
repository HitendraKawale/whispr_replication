# Step 9 — What this replication shows, and what to do next

## 1. What actually got replicated

Worth separating cleanly, because "replicated the paper" can mean three very
different things.

**The method: exactly.** The frontend is numerically identical to OpenAI's
(1.2e-7). The architecture is *bit-identical* — their `whisper-tiny` checkpoint
loads into our class with `strict=True` and produces the same outputs to 0.0.
The optimiser settings are Table 17's. The multitask token format is §2.3's.
There is no hand-waving in this layer.

**The training recipe: faithfully, at 1/100th scale.** Same optimiser, same
schedule shape, same initialisation. Different in the ways a single M1 forces —
and every one of those is written down in `config.py` with a reason.

**The results: directionally, and that is the interesting part.** We do not get
Whisper's WER, and the reason is not a bug we failed to find. It is 680,000
hours versus 100.

## 2. The one result that matters most

The 3.7-hour model learned to be a **language model**:

```
REF  MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD
HYP  THE HILLS WERE THE OTHER AND THE OTHER AS IF THE OTHERLY AND THE
```

Fluent English shape, near-zero acoustic grounding. This is worth understanding
mechanically rather than filing under "not enough data".

The decoder's job is to model `P(token | audio, previous tokens)`. That
factorises into two learnable things: the **text distribution** and the
**audio→text alignment**. The text distribution is *much* cheaper to learn —
46,000 words of transcripts is plenty to learn that "THE" is common and follows
almost anything, and it requires no encoder at all. The alignment needs the
model to discover that specific spectrogram patterns correspond to specific
subwords, which is a far harder correspondence and needs far more examples.

So gradient descent does the easy thing first. Cross-entropy drops from 7.6 to
~5.5 almost entirely by learning unigrams and bigrams. Then it plateaus, because
the next improvement requires the encoder to become useful — and with 3.7 hours
it never gets there before overfitting sets in.

**This is why Whisper is a data paper.** The architecture was never the hard
part. The 680,000 hours are what let the alignment term win.

## 3. Concrete next steps, in order of value per hour

### Precompute the mel spectrograms — done, and smaller than I expected

Implemented in `whispr/melcache.py`. It makes data loading 20× faster (33 ms →
2 ms per batch) but data loading was only 4.5% of a training step, so training
gets **~4% faster, not the 1.5× originally predicted here**. Validation gains
more (61 s → 51 s over dev-clean). See notes/07 §7 for the corrected profile and
for why the original estimate was wrong.

### Train longer (the real win, and it needs no code)

Our 100 h run does ~7 epochs. Published seq2seq systems on LibriSpeech-100
typically train **50–100+** epochs and reach 15–25% WER. Our validation loss was
still falling when the schedule ended — we stopped because of wall-clock, not
convergence. This is the single highest-value change and it requires only
patience (and the item above).

### Add SpecAugment

The paper added it for Large-V2 (footnote 3). Masking random time and frequency
bands forces the model to use context rather than memorising exact patterns. At
our data scale, where overfitting is the binding constraint, this should help
more than it did for them.

### Use the 960-hour set

`train-clean-360` and `train-other-500` bring LibriSpeech to 960 h — another 9.6×.
`whispr.data.standard_split` already handles multiple splits; the change is one
argument plus disk.

### Then the interesting experiments

Once training is fast enough to iterate, the questions worth asking are the
paper's own:

- **Robustness.** Add noise to dev-clean at various SNRs and plot WER. The paper's
  Figure 2 does this and it's where Whisper's advantage over LibriSpeech-trained
  models is most dramatic. Our model should degrade catastrophically — and
  *measuring* that is a real replication of the paper's central claim.
- **Out-of-domain.** Record yourself and transcribe it. Nothing conveys "trained
  on audiobooks" faster.
- **The scaling curve.** Train on 3.7 h, 10 h, 30 h, 100 h and plot WER against
  hours on log axes. That single figure is the paper's thesis, reproduced.
- **Encoder freezing.** Does an encoder trained on 100 h transfer to a 3.7 h
  fine-tune? This is the pretrain/finetune question the paper's introduction is
  arguing against.

### Things deliberately left undone

- **Timestamps.** LibriSpeech has no word-level alignment, so `<|notimestamps|>`
  is always emitted. Would need a corpus with alignments.
- **Multilingual and translation.** The tokens are reserved and never emitted.
  Common Voice would enable it, and the format already supports it — the point of
  reproducing §2.3 faithfully is that this is a data change, not a code change.
- **Beam search KV caching.** Correct but slow; see notes/08 for why the
  bookkeeping wasn't worth the clarity cost.

## 4. If you only remember three things

1. **A spectrogram is a change of basis, and every constant in it is a modelling
   decision.** 16 kHz throws away what doesn't distinguish words. 25 ms windows
   blur pitch and keep formants. 80 mel channels put capacity where hearing has
   resolution. None of it is arbitrary, and all of it is choosing what to discard.

2. **Verify each layer against something external before building on it.** The
   frontend matching OpenAI to 1e-7 and the model matching to 0.0 is what made
   the training results *interpretable* — a bad number could only mean data or
   optimisation, never a silent bug three layers down. That discipline was worth
   more than any single technique here.

3. **Overfit one batch first.** Loss 8.08 → 0.0002 in 149 steps took two minutes
   and proved gradients, masks and label alignment were all correct. Every
   confusing result afterwards was a real result rather than a plumbing bug.
