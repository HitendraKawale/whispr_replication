# Step 1 — Sound as numbers

> Goal: understand what is actually in the array before we transform it, and
> why Whisper's very first line of preprocessing is "resample to 16 kHz".

## 1. What a waveform is

A microphone measures **air pressure at a point in space, over time**. That's
it. A sound is a scalar function `p(t)`: one number per instant. Everything
else — pitch, timbre, which word was said, who said it — is structure *inside*
that one wiggling number.

This is worth sitting with, because it's the source of the difficulty. The
information "the speaker said *cat*" is not stored anywhere in `p(t)` in a
localized way. It is smeared across tens of thousands of samples as a pattern of
oscillation. Our whole frontend exists to re-express `p(t)` in coordinates where
that pattern becomes *visible*.

## 2. Sampling: from a function to an array

We can't store a continuous function, so we measure it every `1/sr` seconds.
`sr` is the **sample rate** in Hz. One second of 16 kHz audio is an array of
16,000 floats.

```
p(t)  ────────╮   ╭──────╮   ╭────       continuous pressure
              ╰───╯      ╰───╯
   sample:    ●  ●  ●  ●  ●  ●  ●        every 1/sr seconds
              ↓
   array:  [0.3, -0.1, -0.6, 0.2, ...]
```

The question that decides everything: **how fast is fast enough?**

## 3. Nyquist, and why 16 kHz

**Nyquist–Shannon**: a sample rate of `sr` can faithfully represent frequencies
strictly below `sr/2`. That limit, `sr/2`, is the **Nyquist frequency**.

The intuition: to know a sinusoid is oscillating, you need at least two samples
per cycle — one to catch the peak, one the trough. Fewer than that and you
cannot distinguish it from a *slower* wave through the same points.

So:

| Sample rate | Nyquist | Captures |
|---|---|---|
| 8 kHz (telephone) | 4 kHz | Speech, muffled. `s` and `f` become hard to tell apart |
| **16 kHz (Whisper)** | **8 kHz** | **All phonetically relevant speech energy** |
| 44.1 kHz (CD) | 22.05 kHz | Full human hearing range (~20 Hz–20 kHz) |

Whisper resamples everything to 16 kHz because **speech information lives below
8 kHz**. Voiced sounds (vowels) have their energy in the low hundreds of Hz plus
harmonics; the highest-frequency content that matters is the fricative noise in
`s`, `sh`, `f`, which lives around 4–8 kHz. Above 8 kHz there is essentially no
information about *which word was said* — only room tone and recording quality.

Going to 44.1 kHz would nearly triple the sequence length to encode noise. For a
Transformer whose cost is quadratic in sequence length, that is a catastrophic
trade. **16 kHz is the point where you stop paying for information you don't
need.**

## 4. Aliasing: what happens when you break the rule

If a frequency above Nyquist is present when you sample, it does not disappear.
It **folds back** into the representable range, disguised as a *different, lower*
frequency. This is aliasing, and it is destructive: once folded, the original
and the impostor are the same array of numbers, and no amount of later
processing can separate them.

A sinusoid at frequency `f` sampled at `sr` appears at:

```
f_apparent = |f - sr * round(f / sr)|
```

So at `sr = 8000`, a 7000 Hz tone appears as a **1000 Hz** tone. Run
`scripts/01_sound_basics.py` and you can *hear* this in the generated
`assets/aliasing_*.wav` files — a rising sweep that turns around and comes back
down, which is the classic aliasing signature.

This is why "resample to 16 kHz" is never just "throw away every third sample".
A correct resampler **low-pass filters first**, removing everything above the new
Nyquist, and only then decimates. `torchaudio.transforms.Resample` does this;
naive slicing does not. Getting this wrong silently corrupts your dataset in a
way that looks fine in a `.shape` check.

## 5. Quantization and amplitude

Samples are also discretized in *value*. 16-bit PCM stores each sample as an
integer in `[-32768, 32767]`. We immediately convert to float in `[-1, 1]` —
dividing by 32768 — because neural networks want floats.

16 bits gives roughly 96 dB of dynamic range, far more than needed. Quantization
noise is not our problem; it's a solved problem. We convert to float and forget
about it.

One thing that *is* our problem: **amplitude is not information**. The same
sentence recorded loudly and quietly should transcribe identically. Whisper
handles this at the end of the frontend (step 3) by clamping and globally
scaling the log-mel, not by normalizing the waveform. Worth remembering — a
common bug is to per-utterance-normalize the waveform, which destroys the
relationship between speech and silence.

## 6. Why we don't feed the waveform to the Transformer

Tempting question: the waveform is already numbers, so why not skip the DSP?

**Sequence length.** 30 seconds at 16 kHz is 480,000 samples. Self-attention is
O(n²). A 480,000-long sequence is ~2.3 × 10¹¹ attention scores per head per
layer. Utterly infeasible.

Whisper's frontend takes those 480,000 numbers down to **3000 frames** — a 160×
reduction — and the conv stem halves it again to **1500**. That is the number
the Transformer actually sees. The frontend is, first and foremost, an
information-preserving compression from 480,000 to 1500.

(Models like wav2vec 2.0 *do* learn from raw audio, using a strided conv stack to
do the same downsampling. It works, but it costs more compute and Whisper's
authors deliberately chose the boring, well-validated option — see §2.2, "to
avoid confounding our findings with model improvements".)

## What to take to step 2

- A waveform is one number per instant; all structure is temporal.
- 16 kHz because speech dies out below 8 kHz; sampling faster buys noise.
- Resampling must low-pass first or you get irreversible aliasing.
- 480,000 samples is too many. We need frequencies, not samples — that's the DFT.

## Run it

```bash
uv run python scripts/01_sound_basics.py
```

Produces `figures/01_*.png` and audible `assets/*.wav` demos.
