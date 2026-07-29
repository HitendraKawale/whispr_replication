# Step 6 — The encoder–decoder Transformer

> Goal: implement §2.2 line by line, and prove the implementation is not merely
> similar to Whisper's but functionally identical to it.

**Result: loading the real `whisper-tiny` checkpoint into our class reproduces
OpenAI's encoder outputs and decoder logits with a maximum difference of exactly
`0.0`.** Details in §7.

## 1. The architecture paragraph, annotated

Here is §2.2 with each clause mapped to code.

> "We chose an **encoder–decoder Transformer** as this architecture has been
> well validated to scale reliably."

Deliberately boring. The paper is a claim about *data*, and using a 2017
architecture unchanged is how they avoid confounding that claim with modelling
improvements. Good news for us: the architecture is exactly reproducible.

> "The encoder processes this input representation with a small stem consisting
> of **two convolution layers with a filter width of 3** and the GELU activation
> function, where the **second convolution layer has a stride of two**."

```python
self.conv1 = nn.Conv1d(n_mels, n_state, kernel_size=3, padding=1)
self.conv2 = nn.Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
```

Two jobs. It mixes across the 80 mel channels — otherwise a Transformer would
have to learn cross-frequency structure through a plain linear projection — and
the stride-2 layer **halves the sequence**, 1500 frames → 750 positions. That
halving is a 4× cut in attention cost and is exactly why the frontend's frame
count is twice the encoder's context length.

> "**Sinusoidal** position embeddings are then added to the output of the stem."

Fixed, not learned. The encoder's positions are *physical time* — frame 100 is
always 1 second in — and there is nothing to learn about that mapping. Registered
as a buffer, so it never receives a gradient.

> "The transformer uses **pre-activation residual blocks**."

```python
x = x + self.attn(self.attn_ln(x))     # not  ln(x + attn(x))
```

The LayerNorm lives *inside* the residual branch, leaving an unnormalised
identity path from input to output. Gradients reach early layers without passing
through a normalisation at every step, which is what makes deep Transformers
trainable.

There's a clean way to test this: zero the branch's output weights and a
pre-norm block becomes an *exact identity*, even for badly-scaled input. A
post-norm block would still normalise and could not. That's `test_is_pre_activation_not_post`.

> "and a **final layer normalization** is applied to the encoder output."

Required *because* of pre-norm — nothing else normalises the output, so the
residual stream would otherwise leave the encoder at whatever scale it drifted to.

> "The decoder uses **learned position embeddings** and **tied input-output
> token representations**."

Learned here, sinusoidal in the encoder. Easy to get backwards; there's a test
for each. Decoder positions are token indices, which carry learnable structure
that physical time does not.

Tying means the output projection *is* the embedding matrix, transposed:

```python
return x @ self.token_embedding.weight.transpose(0, 1)
```

Halves the vocabulary's cost and ties "predicting token t" to "having read
token t" — a real regularity, not worth learning twice.

## 2. The shape of the thing

```
mel (B, 80, 1500)
   │ conv1 k=3        → (B, 384, 1500)   GELU
   │ conv2 k=3 s=2    → (B, 384,  750)   GELU
   │ transpose        → (B, 750, 384)
   │ + sinusoids(750, 384)
   │ 4 × [ self-attn → mlp ]        (pre-norm)
   │ LayerNorm
   ▼
audio features (B, 750, 384) ───────────────┐
                                            │ cross-attention
tokens (B, T) ──┐                           │
   │ embed + learned pos                    │
   │ 4 × [ causal self-attn → cross-attn ───┘ → mlp ]
   │ LayerNorm
   │ @ embedding.T          (tied)
   ▼
logits (B, T, 2048)
```

The encoder runs **once**; the decoder runs once per token. That asymmetry is
what makes the cross-attention KV cache worth having in step 8 — the audio keys
and values never change during decoding.

## 3. Small details that matter

**The key projection has no bias.** `nn.Linear(n_state, n_state, bias=False)`.
It's mathematically redundant: a constant added to every key shifts all
attention logits for a given query equally, and softmax is shift-invariant. Free
parameter saving, zero behavioural change. OpenAI does this; matching it is
required for checkpoint compatibility.

**LayerNorm computes in fp32.** On MPS, half-precision layer norm loses enough
resolution to destabilise training. Casting up costs nothing measurable and
removes a class of divergence that is miserable to diagnose.

**Initialisation is Gaussian fan-in** (Table 17): `std = 1/sqrt(fan_in)`, so a
layer's output variance matches its input variance and activations neither
explode nor vanish with depth.

## 4. A surprise: constant shifts of an embedding row are invisible

While testing that the embeddings really are tied, the obvious probe —
perturb `W[7]`, check logit 7 moves — *failed*. The reason is worth knowing.

The decoder ends in a LayerNorm, so its output `x` is **zero-mean**. The logit
for token 7 is `x · W[7]`. Add a constant `c` to every component of `W[7]` and
the logit changes by `c · Σx = 0`.

So through a tied projection preceded by LayerNorm, **the constant component of
every embedding row is unused** — a genuine degree of freedom the model has no
gradient signal about. Perturb with a random vector and the logit moves as
expected. Both directions are pinned by tests.

## 5. Parameter budget

Our replication model, at vocab 2048 and a 15-second window:

| Component | Parameters |
|---|---|
| Encoder | 7,632,384 |
| Decoder | 10,299,648 |
| — of which token embedding | 786,432 |
| — of which positional embedding | 49,152 |
| **Total** | **17,932,032** |

About 18M against Tiny's 37M, and essentially all of the difference is the
vocabulary: GPT-2's 51,865 tokens cost 19.9M in a tied table, ours cost 0.79M.
**The non-embedding model is the same size as Whisper Tiny's.**

## 6. Table 1's parameter counts are rounded

Running our architecture at OpenAI's exact dimensions:

| Model | Layers | Width | Ours | Table 1 | Real checkpoint |
|---|---|---|---|---|---|
| Tiny | 4 | 384 | **37.18M** | 39M | **37.18M** |
| Base | 6 | 512 | **71.83M** | 74M | 71.83M |
| Small | 12 | 768 | **240.58M** | 244M | 240.58M |

We match the **actual checkpoints** exactly. Table 1's figures are rounded up by
about 5%. Worth knowing before spending an afternoon hunting for two million
missing parameters.

## 7. The decisive verification

```python
real  = whisper.load_model("tiny")
ours  = Whispr(ModelConfig(**WHISPER_DIMS["tiny"]))
ours.load_state_dict(real.state_dict(), strict=True)   # no missing, no unexpected
```

`strict=True` succeeding already proves every parameter name and shape matches.
Then, on the same input:

| Comparison | Max difference |
|---|---|
| Encoder output `(1, 1500, 384)` | **0.0** |
| Decoder logits `(1, 7, 51865)` | **0.0** |
| Argmax token agreement | **100%** |

Not "close" — **bit-identical**. Module ordering, mask conventions, the
transposes, the tying, the missing key bias: all of it is right, or these would
differ.

This is worth the effort because it *ends* a whole category of debugging. If
step 7's training fails, the architecture is not the reason.

## 8. Throughput on the M1 — batch size is free

Measured forward+backward on this model, MPS, 15-second window:

| Batch | ms/step | utterances/s | 12k steps |
|---|---|---|---|
| 2 | 137 | 14.6 | 27 min |
| 4 | 259 | 15.5 | 52 min |
| 8 | 502 | 15.9 | 100 min |
| 16 | 997 | 16.0 | 199 min |

**Throughput is flat at ~16 utterances/second regardless of batch size.** The M1
GPU is compute-bound here, not kernel-launch-bound, so a bigger batch costs
proportionally more wall time and buys nothing in throughput.

The useful consequence: batch size is a purely *optimisation* decision — how
noisy do you want the gradient — not a speed one. That is the opposite of the
situation on a large GPU, where small batches waste the device.

At batch 8 there are 265 steps per epoch over our 2,117 training utterances,
so 12,000 updates would be **45 epochs** on 3.7 hours of audio. The paper does
2–3 epochs on 680,000 hours. Overfitting is not a risk here, it's a certainty,
and step 7 has to be built around that rather than hoping.

## What to take to step 7

- ~18M parameters, all receiving gradients (there's a test).
- The encoder runs once per clip, the decoder once per token.
- The architecture is verified bit-identical to OpenAI's, so training failures
  are training problems — optimisation, data, or hyperparameters.
- ~16 utterances/s on MPS; batch size is free; we will be doing tens of epochs
  over a tiny corpus, so validation-based checkpoint selection is mandatory.

## Run it

```bash
uv run python scripts/06_model.py
uv run --with openai-whisper --with "numba>=0.61" pytest tests/test_model.py -v
```
