# Step 3 — The mel scale and Whisper's exact frontend

> Goal: produce the precise 80×3000 tensor Whisper's encoder eats, and prove it
> is the same one — element for element — that OpenAI's code produces.

This step is verified harder than any other, because frontend bugs are silent.
A wrong mel convention or a missing normalisation doesn't crash; the model
trains, the loss falls, and the result is just quietly worse forever.

**Result: our filterbank matches the array shipped inside `openai/whisper` to
1.9e-9, and our full frontend matches their `log_mel_spectrogram` to 1.2e-7 —
float32 round-off.** See the table at the bottom.

## 1. Why not just use the 201 linear bins?

Two problems with the linear spectrogram from step 2.

**It's the wrong axis for perception.** Linear Hz spaces bins evenly: 0–40 Hz,
40–80 Hz, … 7960–8000 Hz. But hearing is not linear. The gap from 100→200 Hz is
an octave, an enormous perceptual jump. The gap from 7900→8000 Hz is inaudible.
A linear spectrogram spends the same number of bins — the same *model capacity*
— on both. It over-represents the top of the range, where speech has almost no
information, and under-represents the bottom, where formants live.

**It's bigger than it needs to be.** 201 bins × 3000 frames, when 80 suffice.

## 2. The mel scale

The mel scale is an empirical fit to "when do listeners say two tones are
equally far apart in pitch". It's roughly **linear below 1 kHz, logarithmic
above**.

Whisper uses the **Slaney** formulation (via librosa's default), *not* the HTK
formula `2595·log10(1 + f/700)` that most tutorials show:

```
below 1000 Hz:   mel = f / (200/3)                              ← linear
above 1000 Hz:   mel = 15 + log(f/1000) / (log(6.4)/27)         ← logarithmic
```

The constants look arbitrary and are: `200/3` Hz per mel, the breakpoint at
exactly 1 kHz = 15 mel, and `log(6.4)/27` chosen so the two pieces meet with
matching slope. They're fitted values, not derived ones.

**This matters practically.** HTK and Slaney disagree by up to ~5% in band
placement. Use the wrong one and your filterbank is subtly misaligned against
every pretrained Whisper checkpoint. `whispr/mel.py` implements Slaney and
`tests/test_mel.py` asserts it against the real array.

## 3. The filterbank is a matrix

We want 80 numbers from 201. That's an 80×201 matrix multiply. The only question
is what's in the matrix.

Each row is a **triangle**: rises from `edge[i]`, peaks at `edge[i+1]`, falls to
`edge[i+2]`. The 82 edges are spaced **evenly in mel**, then converted back to
Hz — which is what makes them narrow-and-dense at low frequency, wide-and-sparse
at high frequency. Adjacent triangles overlap by half, so no frequency falls in
a gap.

```
weight
  1 │   ╱╲    ╱╲      ╱──╲        ╱────╲
    │  ╱  ╲  ╱  ╲    ╱    ╲      ╱      ╲
  0 └─╱────╲╱────╲──╱──────╲────╱────────╲──→ Hz
      100  200   400       1k          4k
      narrow at low freq ──────► wide at high freq
```

### Slaney normalisation — the part people get wrong

A high-frequency triangle covers many more DFT bins than a low-frequency one. If
every triangle peaked at 1.0, the wide ones would output larger values purely
because they're summing more bins — a systematic bias toward high frequencies
that has nothing to do with the audio.

Slaney normalisation divides each filter by its bandwidth so all filters have
**equal area**:

```python
enorm = 2.0 / (hz_edges[2:n_mels+2] - hz_edges[:n_mels])
weights *= enorm[:, None]
```

Consequence: the filterbank peaks are *not* 1.0 — low filters peak high, wide
ones peak low. If you build a filterbank and every row peaks at exactly 1.0, you
have skipped this and you do not match Whisper.

## 4. The log, and the two magic constants

```python
log_spec = torch.clamp(mel, min=1e-10).log10()
log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)   # (a)
log_spec = (log_spec + 4.0) / 4.0                          # (b)
```

**The log itself.** Loudness is perceived logarithmically, and speech energy
spans an enormous dynamic range. Without the log, a vowel's energy dwarfs a
fricative's by orders of magnitude, and the fricative — carrying just as much
phonetic information — rounds to nothing. `clamp(1e-10)` keeps `log10(0)` from
producing `-inf`.

**(a) the −8 floor** discards everything more than 80 dB below the loudest point
in *this* clip. That's the recording's noise floor: microphone hiss and room
tone, whose absolute level varies wildly between recordings and carries no
linguistic content. Flooring it stops the model from learning to identify
recording equipment.

Note this is a **per-utterance** operation — the one adaptive step in the whole
frontend.

**(b) the affine rescale** maps the result into roughly [−1, 1] with about zero
mean (paper §2.2). It's a *fixed* map — not batch norm, not a learned scale.
Same audio in, same tensor out, always.

Why `+4)/4`? After the floor, values sit in `[max−8, max]`. For typical speech
`max ≈ 0`, so the range is about `[−8, 0]`; `(x+4)/4` sends that to `[−1, 1]`.

The span is *always* exactly 2.0, but where that window sits depends on the
input level — a full-scale chirp lands at `[−0.40, 1.60]`, the same chirp at 1%
amplitude at `[−1.40, 0.60]`. The paper's "between −1 and 1" describes typical
speech levels, not a guarantee. See §4b.

## 4b. Two things the tests turned up

Neither of these is a bug in our code — we match OpenAI's array and output
exactly. They're properties of Whisper's frontend that are easy to assume away,
and both are pinned by tests in `tests/test_mel.py`.

### The low mel bands are undersampled

At `n_fft=400` the DFT bins are 40 Hz apart. But 80 mel bands spread over
0–8 kHz makes the lowest bands only ~10 Hz wide — **narrower than a single
bin**. Measured widths:

- filters 0–20: **1–2 DFT bins each**
- filters 70–79: 8+ bins each
- filter 13 and filter 14 share *no* bin at all — the only genuinely disjoint
  adjacent pair

So the bottom quarter of the mel channels aren't 80 independent measurements of
the low band; they're a handful of DFT bins re-weighted and spread across many
rows. The mel scale is asking for more low-frequency resolution than a 25 ms
window can actually deliver. It's harmless — the information is still there,
just redundantly encoded — but "80 mel channels" does not mean 80 independent
numbers.

### The frontend is *not* loudness invariant — and has two competing floors

Scaling the waveform by `k` shifts the whole log-mel by `log10(k²)/4`. Nothing
in the pipeline removes it. The same sentence recorded loud and quiet gives the
encoder two different tensors, and robustness to that must be **learned from
data**. (More evidence for the paper's thesis: robustness comes from the 680k
hours, not the preprocessing.)

Worse, the shift isn't even uniform, because there are two floors competing:

| Floor | Where it comes from | Value |
|---|---|---|
| Absolute | `clamp(mel, min=1e-10)` | `log10 = −10`, fixed |
| Relative | `maximum(x, x.max() − 8)` | 80 dB below *this clip's* peak |

At normal levels the relative floor is higher, so it binds and silence maps to
`peak − 8`. Scale the input down ~100× and the peak drops until `peak − 8` falls
*below* −10 — now the absolute clamp binds and the relative floor never fires.

Measured on the same chirp at two levels, the difference between the two
spectrograms is **1.0 in the loud regions but 0.9524 in the silent ones**. The
two are not related by a shift at all. Quiet recordings get a genuinely
different representation, not merely an offset one.

Practical upshot for step 4: don't feed the model wildly varying input levels
and expect the frontend to sort it out. It won't.

## 5. The frontend, end to end

```
480,000 samples (30 s @ 16 kHz)
  │  torch.stft(n_fft=400, hop=160, hann, center=True)
  ▼
(201, 3001) complex
  │  drop the last frame — it's centred past the signal's end, all padding
  ▼
(201, 3000) complex   ──|·|²──►   (201, 3000) power
  │  80×201 mel filterbank matmul
  ▼
(80, 3000)
  │  log10 → floor at max−8 → (x+4)/4
  ▼
(80, 3000) in roughly [-1, 1]   ← the encoder's input
```

**A 160× compression, and the whole reason a Transformer can touch this at all.**

## 6. Verification

Both references live in the repo (`assets/whisper_mel_filters.npz` is the actual
array shipped in `openai/whisper`):

| Check | Reference | Max abs difference |
|---|---|---|
| 80-mel filterbank | `openai/whisper` shipped `mel_filters.npz` | **1.9e-9** |
| 128-mel filterbank | same | **1.9e-9** |
| Full frontend, 30 s noise | `whisper.log_mel_spectrogram` | **1.2e-7** |
| Full frontend, 3 s tone | same | **1.2e-7** |
| Full frontend, 10 s chirp | same | **1.2e-7** |
| Full frontend, very quiet input | same | **2.4e-7** |

Float32 has ~7 decimal digits, so 1e-7 *is* equality. Reproduce with:

```bash
uv run pytest tests/test_mel.py -v
uv run --with openai-whisper --with "numba>=0.61" pytest tests/test_mel.py -v
```

The second command adds the reference-implementation tests, which `skip` when
`openai-whisper` isn't installed. (Its own dependency pins resolve to a numba
too old for Python 3.12, hence the explicit `numba` constraint.)

## 7. What the model actually sees

Look at `figures/03_frontend_pipeline.png`. Things to notice:

- **Horizontal bands that move** — formants. This is where phonetic identity is.
- **The bottom rows are dense with structure, the top rows nearly empty.** That's
  the mel scale doing its job: capacity allocated where information is.
- **Vertical streaks** — plosive bursts, broadband and brief.
- **Padding is not zero.** Silence maps to a constant *negative* value after the
  log, not to 0. The model can trivially learn to ignore it, but it is not
  "empty" in the tensor.

## What to take to step 4

- The encoder's input is `(80, 3000)` in roughly [−1, 1], always exactly that shape.
- Our frontend is numerically identical to OpenAI's, so any later failure is a
  modelling or data problem — never a frontend one. That is the entire value of
  this step.

## Run it

```bash
uv run python scripts/03_frontend.py
```
