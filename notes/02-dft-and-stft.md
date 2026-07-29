# Step 2 — The DFT and the STFT

> Goal: stop treating `torch.stft` as a magic box. By the end you should be able
> to say exactly what each number in a spectrogram means and what it cost.

## 1. The DFT is a change of basis, not a mystery

You have `N` samples. Instead of describing them in the **time basis** ("sample
0 is 0.3, sample 1 is −0.1, …"), describe them in the **frequency basis** ("this
signal is 0.8 units of a 100 Hz wave, plus 0.3 units of a 250 Hz wave, …").

Both descriptions have `N` numbers. Neither loses information — you can go back
and forth exactly. It's a rotation of the same vector.

The DFT does it by asking, for each candidate frequency `k`, *"how much does my
signal look like a sinusoid at frequency k?"*:

```
X[k] = Σ_t  x[t] · exp(-2πi·k·t/N)
       └──┬──┘   └───────┬───────┘
      your signal    a probe sinusoid at frequency k
```

That's a dot product with a sinusoid, one per frequency. Stack the probes into a
matrix and the whole DFT is one matrix multiply — literally `dft_matrix(n) @ x`
in `whispr/dft.py`. **The DFT is a matrix multiply.** The FFT is the same
answer computed in O(N log N) by exploiting redundancy in that matrix; it is an
optimization, not a different idea.

### Why complex numbers

Each `X[k]` is complex because "how much of frequency k" needs **two** numbers:

- **magnitude** `|X[k]|` — how strong that frequency is
- **phase** `∠X[k]` — where in its cycle the wave starts

You need phase because a sine and a cosine at the same frequency are different
signals, and a single real number can't distinguish them. A pure dot-product
with `cos` would report zero for a `sin` input — the complex probe tests both
simultaneously.

We will end up **throwing phase away** (step 3). This is lossy and deliberate:
phase is perceptually secondary and hard to model, whereas the magnitude
envelope is where phonetic identity lives. Griffin-Lim reconstructing
intelligible speech from magnitude alone is the empirical justification.

### Bins: what frequency is bin k?

```
frequency of bin k = k · sample_rate / n_fft
```

For Whisper (`n_fft=400`, `sr=16000`): bins are **40 Hz apart**, and there are
`400/2 + 1 = 201` of them covering 0 Hz to 8 kHz (Nyquist).

Why only half? For real input, `X[N−k] = conj(X[k])` — the upper half is a
mirror image carrying no new information. `rfft` drops it. **201 is where
Whisper's 201-row linear spectrogram comes from**, before the mel filterbank
squashes it to 80.

## 2. Why one DFT over the whole signal is useless

Take the DFT of a 30-second utterance and you get: "there was some energy at
200 Hz, some at 1 kHz…" — averaged over the whole thirty seconds. Completely
useless. Speech is *non-stationary*; the entire signal is the sequence of
changes. A global DFT integrates exactly the information we need away.

## 3. The STFT: chop it up first

**Short-Time Fourier Transform.** Assume the signal is roughly stationary over a
short window (~25 ms — long enough for a few pitch periods, short enough that
the vocal tract hasn't moved much). Then:

```
signal ─┬─ frame 0 (25 ms) ─→ window ─→ DFT ─→ column 0 ┐
        ├─ frame 1 (25 ms) ─→ window ─→ DFT ─→ column 1 │  spectrogram
        ├─ frame 2 (25 ms) ─→ window ─→ DFT ─→ column 2 │  (freq × time)
        └─ ...              hop 10 ms                   ┘
```

Frames **overlap**: 25 ms windows every 10 ms, so each sample appears in ~2.5
frames. Overlap exists because windowing attenuates frame edges to nearly zero
— without overlap, information near frame boundaries would be lost.

Whisper's numbers (paper §2.2): 25 ms window = **400 samples** at 16 kHz,
10 ms hop = **160 samples**.

## 4. Windowing, and why rectangular frames are a disaster

Naively chopping means multiplying by a rectangular window. The discontinuity at
each edge is a *fake* signal event, and the DFT dutifully reports it — as energy
smeared across **every** frequency bin. This is **spectral leakage**, and it can
bury real structure under a 30 dB noise floor.

Fix: taper the frame edges smoothly to zero. The **Hann window**,
`0.5·(1 − cos(2πt/N))`, is the standard choice.

Run `scripts/02_stft.py` and look at `figures/02_leakage.png` — the same pure
tone with and without a window. Rectangular smears; Hann concentrates.

One gotcha that bites everyone: **periodic vs. symmetric**. `periodic=True`
divides by `N`, `symmetric` by `N−1`. STFT wants periodic (torch's default). Mix
them and you get a small mismatch against reference implementations that is
maddening to track down.

## 5. The resolution tradeoff (this is the real content)

You cannot have sharp time resolution and sharp frequency resolution at once.
This is not an engineering limitation — it's the Fourier uncertainty principle.

```
frequency resolution = sr / n_fft        (bin spacing)
time resolution      ≈ n_fft / sr        (window duration)
```

They multiply to 1. Improve one, ruin the other:

| n_fft @ 16 kHz | Window | Freq resolution | Good for |
|---|---|---|---|
| 128 | 8 ms | 125 Hz | Sharp transients — plosives (`t`, `k`), onsets |
| **400 (Whisper)** | **25 ms** | **40 Hz** | **The compromise** |
| 2048 | 128 ms | 7.8 Hz | Resolving individual pitch harmonics |

At `n_fft=400`, 40 Hz spacing does **not** resolve individual harmonics of a
100–200 Hz voice — they blur into a smooth envelope. That's fine, arguably
good: the *envelope* (the formants) carries phonetic identity, while the
harmonic fine structure carries pitch and speaker identity, which for
transcription is nuisance variation. **Whisper's 25 ms window quietly discards
speaker pitch and keeps phonetic content.** That's a modelling decision hiding
in a DSP parameter.

See `figures/02_resolution_tradeoff.png` — same 2 seconds of speech-like signal
at four window sizes. Short windows: crisp vertical edges, blurry horizontal
bands. Long windows: the reverse.

## 6. Centering, and where 3000 comes from

`center=True` reflect-pads the signal by `n_fft//2` on each side, so frame `i`
is *centred* on sample `i·hop` rather than starting there. Consequence:

```
num_frames = 1 + len(x) // hop        (with center=True)
           = 1 + 480000 // 160 = 3001
```

Whisper drops the last frame → **exactly 3000**. The conv stem's stride-2 layer
halves that to **1500**, which is the encoder's sequence length and the size of
its fixed positional encoding. Every one of these numbers is load-bearing.

## 7. Cost

Our `stft_naive` is a 400×400 complex matmul per frame. `torch.stft` uses the
FFT: O(N log N). For n_fft=400 that's roughly a 40× speedup per frame, and
we compute 3000 frames per 30-second clip.

We keep the naive version because it is *readable* and because the test asserting
it matches `torch.stft` to 1e-4 is what earns us the right to use the fast one.

## What to take to step 3

- A spectrogram column is 201 magnitudes, one per 40 Hz bin, from a 25 ms window.
- Phase is discarded; magnitude carries the phonetic content.
- 25 ms is chosen to blur harmonics and keep formants.
- 480,000 samples → 3000 frames → (after conv) 1500 encoder positions.
- 201 linear bins is still more than we want, and linear Hz is the wrong axis
  for perception. Both problems are solved by the mel filterbank.

## Run it

```bash
uv run python scripts/02_stft.py
uv run pytest tests/test_dft.py -v
```
