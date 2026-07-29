# Roadmap: how we get from "sound is a wiggle" to a working ASR model

## Why this order

Whisper's pipeline is a chain where each link is meaningless without the one
before it. So we build it strictly bottom-up, and at every step we *verify
against something external* — a closed-form answer, a reference implementation,
or a known-good library — before moving on. Debugging a Transformer while your
spectrogram is silently wrong is a special kind of misery, and we're going to
avoid it by construction.

```
   air pressure          numbers            frequencies         perceptual freq
  ┌──────────┐  sample  ┌────────┐   STFT   ┌──────────┐  mel   ┌───────────┐
  │  speech  │ ───────► │waveform│ ───────► │spectrogram│ ────► │ log-mel   │
  └──────────┘  step 1  └────────┘  step 2  └──────────┘ step 3 └───────────┘
                                                                      │
                                                                      ▼
  ┌──────────┐  decode  ┌────────┐  train   ┌──────────┐ tokens ┌───────────┐
  │   text   │ ◄─────── │ logits │ ◄─────── │Transformer│ ◄──── │  encoder  │
  └──────────┘  step 8  └────────┘  step 7  └──────────┘ step 6 └───────────┘
                                                  ▲
                                            steps 4–5: data + vocabulary
```

## The steps

| # | Step | What you'll understand afterwards | Verified against |
|---|------|-----------------------------------|------------------|
| 1 | Sound as numbers | Why 16 kHz, what aliasing sounds like, why we don't feed raw samples to a Transformer | Closed-form sine, audible artefacts |
| 2 | DFT & STFT by hand | What a "frequency bin" is, why windows exist, the time/frequency resolution tradeoff | `torch.stft` |
| 3 | Mel + Whisper frontend | Why mel and not linear Hz, and Whisper's exact 80×3000 input tensor | `openai-whisper`'s own `log_mel_spectrogram` |
| 4 | Data pipeline | The shape of a real ASR corpus | LibriSpeech official checksums |
| 5 | Tokenizer | Why BPE, and how Whisper crams *tasks* into the token stream | Round-trip encode/decode |
| 6 | Model | Every line of the architecture paragraph in §2.2 | Param count vs. Table 1 |
| 7 | Training | Warmup, clipping, and why loss goes flat before it goes down | Overfit-one-batch → loss ≈ 0 |
| 8 | Decoding & WER | Greedy vs. beam, and why text normalization changes WER by 50% | `jiwer` |

## The discipline

Two rules that make the difference between a replication and a pile of scripts:

1. **Every step ends with a passing test and a commit.** If it isn't tested, it
   isn't done, because the failure will surface six steps later disguised as a
   modelling problem.
2. **Never trust a tensor you haven't looked at.** Every step produces a figure.
   Spectrograms in particular are wrong in ways that are invisible in
   `.shape` and obvious in `imshow`.

## The honest expectation

At 5 hours of single-domain read speech and ~39M parameters, we should expect:

- **Overfit-one-batch**: loss → ~0. If this fails, the model is broken.
- **dev-clean train / held-out eval**: the model will learn English phonotactics
  and produce fluent-looking, frequently-wrong transcripts. WER of 30–60% is a
  *success* at this scale.
- **Robustness**: none. Play it anything that isn't a LibriSpeech audiobook and
  it will fall apart.

That last row is the whole point of the paper. Whisper's robustness came from
680,000 hours of messy, diverse audio — not from the architecture we're about to
write, which is a 2017 Transformer with a spectrogram bolted to the front. Being
able to *feel* that difference is worth more than a good WER number.
