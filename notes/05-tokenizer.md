# Step 5 — Tokenizer and the multitask format

> Goal: turn text into integers, and understand the token-format trick that is
> arguably Whisper's real contribution.

## 1. Why not characters?

We have 28 characters. Why not just predict one at a time?

You can — early ASR did — but it makes the decoder's job unnecessarily long and
hard. "PARTICULARLY" is 12 decoder steps, 12 chances to derail, and each step
carries almost no information. Attention has to hold context across hundreds of
steps to keep a sentence coherent.

## 2. Why not whole words?

The opposite failure. English has hundreds of thousands of word forms; any fixed
word vocabulary meets words it has never seen and has no way to write them down.
For a corpus with 46,060 training words, a word vocabulary would be enormous,
mostly-rare, and still incomplete.

## 3. BPE: the compromise

**Byte-pair encoding** starts from characters and repeatedly merges the most
frequent adjacent pair into a new token. Frequent words end up as single tokens;
rare words decompose into pieces. Nothing is ever unrepresentable, because the
character level is always the fallback.

```
THE QUICK BROWN FOX JUMPS
  -> [' THE'] [' QUICK'] [' BR'] ['OWN'] [' FO'] ['X'] [' J'] ['UM'] ['PS']
      common words: 1 token        rarer words: split into pieces
```

Note the leading spaces. With `add_prefix_space=True` a word is stored as
`" THE"`, so the *same* token serves sentence-initial and mid-sentence
positions instead of needing two. Decoding strips the one leading space so that
`decode(encode(t)) == t` exactly.

## 4. Choosing the vocabulary size, by measurement

Whisper reuses GPT-2's **50,257-token** vocabulary. We should not, for two
reasons.

**Parameter budget.** The decoder ties its input and output embeddings (paper
§2.2), so the vocabulary costs `n_vocab × 384` parameters. At 50,257 that is
**19.3M — half of Tiny's entire 39M budget**, spent on an embedding table for a
28-character alphabet.

**Fit.** GPT-2's BPE was fitted to web text: mixed case, punctuation, code,
markup. LibriSpeech labels are uppercase letters, spaces and apostrophes. Nearly
all of those 50k tokens could never be emitted.

So we fit our own. The size is an empirical question, so it was measured on the
training split (`scripts/05_tokenizer.py`):

| vocab | tokens/word (train) | tokens/word (val) | embedding params |
|---|---|---|---|
| 256 | 5.34 | 5.41 | 102K |
| 512 | 2.28 | 2.35 | 197K |
| 1024 | 1.76 | 1.84 | 393K |
| **2048** | **1.46** | **1.57** | **786K** |
| 4096 | 1.25 | 1.39 | 1.57M |

**2048.** Past that the compression gains flatten while the table doubles, and
with 46k words of training text the extra tokens would each be seen a handful of
times — badly estimated embeddings for tokens that barely occur.

The train/val gap (1.46 vs 1.57) is the vocabulary meeting words it wasn't fitted
on, and it widens as the vocabulary grows — a small, visible instance of
overfitting in a component people rarely think of as fittable.

### Fit on the training split only

The BPE is trained on training transcripts only. Fitting it on all text is a
subtle leak: the merge list itself encodes which word pieces occur in the
held-out set, so validation would be scored with a vocabulary that has already
seen its answers.

## 5. The multitask format — the actually interesting part

This is Whisper's central trick, and it costs almost nothing to implement.

Whisper is one model that does transcription, translation, language
identification and voice-activity detection. Not four heads, not four models —
**one autoregressive decoder**, where the task is selected purely by which
tokens you seed it with.

```
<|startoftranscript|> <|en|> <|transcribe|> <|notimestamps|> THE QUICK ... <|endoftext|>
        SOT           lang     task           format              text          EOT
```

Change `<|transcribe|>` to `<|translate|>` and the same weights translate into
English. Change `<|en|>` to `<|de|>` and it transcribes German. Let the model
*predict* the language token instead of forcing it, and you have language
identification for free. Predict `<|nospeech|>` and you have voice-activity
detection.

Everything the system does is expressed as a **prefix of the sequence it is
already trained to continue.** A conditional language model that happens to be
conditioned on audio.

Our replication has one language and one task, so we emit
`SOT, <|en|>, <|transcribe|>, <|notimestamps|>` every time — a constant prefix.
The format is reproduced faithfully anyway, because the format *is* the idea,
and because it means adding a task later is a data change rather than an
architecture change.

We reserve `<|translate|>`, `<|nospeech|>` and `<|startofprev|>` in the
vocabulary but never emit them.

### Timestamps

Whisper can emit timestamp tokens interleaved with text, quantised to 20 ms.
LibriSpeech gives us utterance-level alignment only — no word timings — so we
always emit `<|notimestamps|>`. That token exists precisely so the model knows
which regime it is in.

## 6. Special token ids

`<|pad|>` is deliberately **id 0**, so a zero-filled tensor is padding rather
than some arbitrary word. Small thing; prevents a whole family of confusing bugs.

```
0 <|pad|>   1 <|endoftext|>   2 <|startoftranscript|>   3 <|en|>
4 <|transcribe|>   5 <|translate|>   6 <|notimestamps|>
7 <|nospeech|>   8 <|startofprev|>
```

## 7. Sequence lengths

With vocab 2048, over the utterances we keep (≤15 s):

```
median 28 tokens · p99 72 · max 94   (including the 4 prefix tokens and EOT)
```

So `n_text_ctx = 128` is comfortable — the paper's 448 exists to accommodate
30 seconds of dense speech plus prior-context conditioning, neither of which we
have.

## What to take to step 6

- Vocabulary is 2048; embeddings cost 786K parameters and are tied.
- Every training target is `[SOT, EN, TRANSCRIBE, NOTIMESTAMPS] + text + [EOT]`.
- The first 4 tokens are *inputs*, not predictions — the loss must not reward
  predicting a constant prefix, or the reported loss will look better than the
  model is.
- Decoder context is 128.

## Run it

```bash
uv run python scripts/05_tokenizer.py
```
