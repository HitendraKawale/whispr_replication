"""The Whisper encoder-decoder Transformer.

Paper §2.2, in full:

    "We chose an encoder-decoder Transformer as this architecture has been well
    validated to scale reliably. [...] The encoder processes this input
    representation with a small stem consisting of two convolution layers with
    a filter width of 3 and the GELU activation function, where the second
    convolution layer has a stride of two. Sinusoidal position embeddings are
    then added to the output of the stem, after which the encoder Transformer
    blocks are applied. The transformer uses pre-activation residual blocks,
    and a final layer normalization is applied to the encoder output. The
    decoder uses learned position embeddings and tied input-output token
    representations."

Every sentence there is a line of code below. Module and parameter names follow
OpenAI's reference implementation, so a real Whisper checkpoint can be loaded
into this class for cross-checking (see tests/test_model.py).
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from whispr.config import ModelConfig


def sinusoids(length: int, channels: int, max_timescale: float = 10_000) -> Tensor:
    """Fixed sinusoidal position embeddings for the encoder.

    The encoder uses *fixed* sinusoids rather than learned embeddings because
    its input is a fixed-length spectrogram whose positions are physical time —
    frame 100 is always 1 second in. There is nothing to learn about that
    mapping. The decoder's positions are token indices, which do carry
    learnable structure, so those embeddings are learned.
    """
    if channels % 2 != 0:
        raise ValueError(f"sinusoidal embeddings need an even width, got {channels}")
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


class LayerNorm(nn.LayerNorm):
    """LayerNorm that always computes in fp32.

    On MPS, half precision layer norm loses enough resolution to destabilise
    training. Casting up costs almost nothing and removes a class of
    hard-to-diagnose divergence.
    """

    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float()).type(x.dtype)


class MultiHeadAttention(nn.Module):
    """Standard scaled dot-product attention, used for both self- and cross-attention.

    The `key` projection has **no bias**, matching OpenAI's implementation. It
    is mathematically redundant — a constant added to every key shifts all
    attention logits for a query by the same amount, which softmax cancels — so
    omitting it saves parameters and changes nothing.
    """

    def __init__(self, n_state: int, n_head: int) -> None:
        super().__init__()
        self.n_head = n_head
        self.query = nn.Linear(n_state, n_state)
        self.key = nn.Linear(n_state, n_state, bias=False)
        self.value = nn.Linear(n_state, n_state)
        self.out = nn.Linear(n_state, n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ) -> Tensor:
        """`xa` is the encoder output for cross-attention; None means self-attention."""
        q = self.query(x)

        if xa is None:
            # Self-attention. Always project x: when incremental caching is
            # active, a forward hook on these Linears concatenates the new
            # position onto the stored history and returns the full sequence,
            # so `k` and `v` come back longer than `x`.
            k = self.key(x)
            v = self.value(x)
        elif kv_cache is not None and self.key in kv_cache:
            # Cross-attention keys/values depend only on the audio, which does
            # not change across decoding steps — so compute them once.
            k, v = kv_cache[self.key], kv_cache[self.value]
        else:
            k = self.key(xa)
            v = self.value(xa)
            if kv_cache is not None:
                kv_cache[self.key] = k
                kv_cache[self.value] = v

        return self.out(self._attend(q, k, v, mask))

    def _attend(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor]) -> Tensor:
        n_batch, n_ctx, n_state = q.shape
        head_dim = n_state // self.n_head

        # (batch, heads, time, head_dim)
        q = q.view(n_batch, -1, self.n_head, head_dim).transpose(1, 2)
        k = k.view(n_batch, -1, self.n_head, head_dim).transpose(1, 2)
        v = v.view(n_batch, -1, self.n_head, head_dim).transpose(1, 2)

        attn_mask = None
        # A single query needs no causal mask: every key it can see is already
        # in its past. This is also the incremental-decoding case, where q has
        # length 1 but k/v span the whole history, so mask[:1, :1] would be the
        # wrong shape.
        if mask is not None and n_ctx > 1:
            attn_mask = mask[:n_ctx, :n_ctx]

        # PyTorch's fused kernel; equivalent to softmax(qk^T/sqrt(d))v but
        # avoids materialising the (time x time) score matrix.
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        return out.transpose(1, 2).reshape(n_batch, n_ctx, n_state)


class ResidualAttentionBlock(nn.Module):
    """A pre-activation residual block (paper §2.2, citing Child et al. 2019).

    "Pre-activation" means the LayerNorm sits *inside* the residual branch:

        x = x + attn(ln(x))          not    x = ln(x + attn(x))

    This leaves an unnormalised identity path from input to output, so gradients
    reach early layers without passing through a normalisation at every step.
    It is what makes deep Transformers trainable without careful warmup — and
    it is why a final LayerNorm is needed at the end, since nothing else
    normalises the output.
    """

    def __init__(self, n_state: int, n_head: int, cross_attention: bool = False) -> None:
        super().__init__()
        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = LayerNorm(n_state)

        self.cross_attn = MultiHeadAttention(n_state, n_head) if cross_attention else None
        self.cross_attn_ln = LayerNorm(n_state) if cross_attention else None

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            nn.Linear(n_state, n_mlp), nn.GELU(), nn.Linear(n_mlp, n_state)
        )
        self.mlp_ln = LayerNorm(n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ) -> Tensor:
        x = x + self.attn(self.attn_ln(x), mask=mask, kv_cache=kv_cache)
        if self.cross_attn is not None:
            x = x + self.cross_attn(self.cross_attn_ln(x), xa=xa, kv_cache=kv_cache)
        x = x + self.mlp(self.mlp_ln(x))
        return x


class AudioEncoder(nn.Module):
    """Log-mel spectrogram -> a sequence of audio features.

    The conv stem does two jobs. It mixes across the 80 mel channels (which a
    Transformer would otherwise have to learn from scratch through a linear
    projection), and its stride-2 second layer halves the sequence length,
    turning 1500 frames into 750 positions. That halving is a 4x saving in
    attention cost, and it is why the frontend's frame count and the encoder's
    context length differ by exactly a factor of two.
    """

    def __init__(
        self, n_mels: int, n_ctx: int, n_state: int, n_head: int, n_layer: int
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        self.register_buffer("positional_embedding", sinusoids(n_ctx, n_state))

        self.blocks = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)]
        )
        self.ln_post = LayerNorm(n_state)

    def forward(self, x: Tensor) -> Tensor:
        """x: (batch, n_mels, n_frames) -> (batch, n_ctx, n_state)"""
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)  # (batch, time, channels)

        if x.shape[1:] != self.positional_embedding.shape:
            raise ValueError(
                f"encoder got {tuple(x.shape[1:])} after the stem but its positional "
                f"embedding is {tuple(self.positional_embedding.shape)}. The audio "
                f"window and ModelConfig.n_audio_ctx disagree."
            )
        x = (x + self.positional_embedding).to(x.dtype)

        for block in self.blocks:
            x = block(x)
        return self.ln_post(x)


class TextDecoder(nn.Module):
    """Autoregressive decoder, cross-attending to the audio.

    Input and output token representations are **tied** (paper §2.2, citing
    Press & Wolf 2017): the output projection reuses the embedding matrix
    transposed. It halves the vocabulary's parameter cost and ties the
    representation of "predicting token t" to "having read token t", which is a
    genuine regularity rather than a coincidence worth learning twice.
    """

    def __init__(
        self, n_vocab: int, n_ctx: int, n_state: int, n_head: int, n_layer: int
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(n_vocab, n_state)
        self.positional_embedding = nn.Parameter(torch.empty(n_ctx, n_state))

        self.blocks = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head, cross_attention=True) for _ in range(n_layer)]
        )
        self.ln = LayerNorm(n_state)

        # Causal mask: position i may attend to 0..i only. Stored as a buffer so
        # it moves with the model across devices.
        mask = torch.empty(n_ctx, n_ctx).fill_(float("-inf")).triu_(1)
        self.register_buffer("mask", mask, persistent=False)

    def forward(
        self,
        x: Tensor,
        xa: Tensor,
        kv_cache: Optional[dict] = None,
        offset: int = 0,
    ) -> Tensor:
        """x: (batch, n_tokens) token ids. xa: (batch, n_audio_ctx, n_state).

        During incremental decoding `x` holds only the new token, so `offset`
        says where it sits in the sequence and therefore which positional
        embedding it gets. Passed explicitly rather than inferred from the
        cache, because the cache holds both growing self-attention entries and
        fixed-length cross-attention ones and guessing from it is fragile.
        """
        x = self.token_embedding(x) + self.positional_embedding[offset : offset + x.shape[-1]]

        for block in self.blocks:
            x = block(x, xa, mask=self.mask, kv_cache=kv_cache)

        x = self.ln(x)
        # Tied output projection.
        return x @ self.token_embedding.weight.to(x.dtype).transpose(0, 1)


class Whispr(nn.Module):
    """The whole model."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = AudioEncoder(
            config.n_mels,
            config.n_audio_ctx,
            config.n_audio_state,
            config.n_audio_head,
            config.n_audio_layer,
        )
        self.decoder = TextDecoder(
            config.n_vocab,
            config.n_text_ctx,
            config.n_text_state,
            config.n_text_head,
            config.n_text_layer,
        )
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Paper Table 17: "Gaussian Fan-In" initialisation.

        std = 1/sqrt(fan_in), so the variance of a layer's output matches the
        variance of its input and activations neither explode nor vanish with
        depth.
        """
        if isinstance(module, nn.Linear):
            fan_in = module.weight.shape[1]
            nn.init.normal_(module.weight, mean=0.0, std=1.0 / math.sqrt(fan_in))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv1d):
            fan_in = module.weight.shape[1] * module.weight.shape[2]
            nn.init.normal_(module.weight, mean=0.0, std=1.0 / math.sqrt(fan_in))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=1.0 / math.sqrt(module.weight.shape[1]))
        elif isinstance(module, TextDecoder):
            nn.init.normal_(
                module.positional_embedding, mean=0.0, std=1.0 / math.sqrt(module.positional_embedding.shape[1])
            )

    def embed_audio(self, mel: Tensor) -> Tensor:
        return self.encoder(mel)

    def logits(self, tokens: Tensor, audio_features: Tensor) -> Tensor:
        return self.decoder(tokens, audio_features)

    def forward(self, mel: Tensor, tokens: Tensor) -> Tensor:
        """mel: (batch, n_mels, n_frames), tokens: (batch, n_tokens)
        -> logits (batch, n_tokens, n_vocab)"""
        return self.decoder(tokens, self.encoder(mel))

    # ------------------------------------------------------------- inspection

    def num_parameters(self, trainable_only: bool = True) -> int:
        params = self.parameters()
        if trainable_only:
            params = (p for p in params if p.requires_grad)
        return sum(p.numel() for p in params)

    def parameter_breakdown(self) -> dict[str, int]:
        """Where the parameters actually live — useful for sizing decisions."""
        enc = sum(p.numel() for p in self.encoder.parameters())
        dec = sum(p.numel() for p in self.decoder.parameters())
        emb = self.decoder.token_embedding.weight.numel()
        pos = self.decoder.positional_embedding.numel()
        return {
            "encoder": enc,
            "decoder": dec,
            "decoder_token_embedding": emb,
            "decoder_positional_embedding": pos,
            "total": enc + dec,
        }

    def install_kv_cache(self) -> tuple[dict, list]:
        """Cache keys/values so incremental decoding is O(1) per step, not O(t).

        Without this, generating token t re-runs the decoder over all t previous
        tokens and a T-token transcript costs O(T^2). The causal mask guarantees
        earlier positions never see later ones, so their keys and values are
        final once computed.

        Two kinds of entry with different lifetimes:
          - self-attention K/V grow by one position per step (handled by the
            forward hooks installed here, which concatenate and return the full
            history);
          - cross-attention K/V depend only on the audio and are computed once
            (handled by the explicit `kv_cache` dict in MultiHeadAttention).

        Returns the cache and the hook handles; call `.remove()` on each when
        done, or use `kv_cache()` which does it for you.
        """
        cache: dict = {}
        hooks = []

        def save(module, _, output):
            if module not in cache:
                cache[module] = output.detach()
            else:
                cache[module] = torch.cat([cache[module], output], dim=1).detach()
            return cache[module]

        for block in self.decoder.blocks:
            hooks.append(block.attn.key.register_forward_hook(save))
            hooks.append(block.attn.value.register_forward_hook(save))

        return cache, hooks

    @contextmanager
    def kv_cache(self):
        """Scoped incremental-decoding cache.

            with model.kv_cache() as cache:
                ...

        Removing the hooks on exit matters: left installed, they would keep
        concatenating during the *next* utterance and silently corrupt it.
        """
        cache, hooks = self.install_kv_cache()
        try:
            yield cache
        finally:
            for h in hooks:
                h.remove()


def build_model(config: ModelConfig | None = None) -> Whispr:
    return Whispr(config or ModelConfig())


# Whisper's released model dimensions (paper Table 1), for cross-checking that
# our architecture reproduces the real parameter counts.
WHISPER_DIMS = {
    "tiny": dict(n_mels=80, n_vocab=51865, n_audio_ctx=1500, n_audio_state=384,
                 n_audio_head=6, n_audio_layer=4, n_text_ctx=448, n_text_state=384,
                 n_text_head=6, n_text_layer=4),
    "base": dict(n_mels=80, n_vocab=51865, n_audio_ctx=1500, n_audio_state=512,
                 n_audio_head=8, n_audio_layer=6, n_text_ctx=448, n_text_state=512,
                 n_text_head=8, n_text_layer=6),
    "small": dict(n_mels=80, n_vocab=51865, n_audio_ctx=1500, n_audio_state=768,
                  n_audio_head=12, n_audio_layer=12, n_text_ctx=448, n_text_state=768,
                  n_text_head=12, n_text_layer=12),
}
