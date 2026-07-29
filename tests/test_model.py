"""Step 6 tests: architectural fidelity, shapes, causality, and gradient flow.

The strongest tests here load the *real* whisper-tiny checkpoint into our class
and check the outputs are identical. They need `openai-whisper` and a cached
checkpoint, and skip otherwise:

    uv run --with openai-whisper --with "numba>=0.61" pytest tests/test_model.py
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from whispr.config import Config, ModelConfig
from whispr.model import (
    WHISPER_DIMS,
    AudioEncoder,
    MultiHeadAttention,
    ResidualAttentionBlock,
    TextDecoder,
    Whispr,
    build_model,
    sinusoids,
)

WHISPER_CACHE = Path(os.path.expanduser("~/.cache/whisper/tiny.pt"))


@pytest.fixture(scope="module")
def tiny_config():
    """A small config for fast tests."""
    return ModelConfig(n_vocab=128, n_audio_ctx=50, n_audio_state=64,
                       n_audio_head=4, n_audio_layer=2, n_text_ctx=16,
                       n_text_state=64, n_text_head=4, n_text_layer=2)


@pytest.fixture(scope="module")
def model(tiny_config):
    return build_model(tiny_config)


class TestSinusoids:
    def test_shape(self):
        assert sinusoids(1500, 384).shape == (1500, 384)

    def test_odd_width_is_rejected(self):
        with pytest.raises(ValueError, match="even width"):
            sinusoids(10, 7)

    def test_values_are_bounded(self):
        s = sinusoids(100, 64)
        assert s.min() >= -1.0 and s.max() <= 1.0

    def test_positions_are_distinguishable(self):
        """The point of positional encoding: no two positions look alike."""
        s = sinusoids(200, 64)
        assert not torch.allclose(s[0], s[1])
        assert not torch.allclose(s[0], s[100])

    def test_is_deterministic_not_learned(self):
        assert torch.equal(sinusoids(50, 32), sinusoids(50, 32))

    def test_encoder_positional_embedding_is_a_buffer_not_a_parameter(self, model):
        """Fixed sinusoids must not receive gradients."""
        names = [n for n, _ in model.encoder.named_parameters()]
        assert "positional_embedding" not in names
        assert "positional_embedding" in dict(model.encoder.named_buffers())

    def test_decoder_positional_embedding_IS_learned(self, model):
        """Paper §2.2: encoder sinusoidal, decoder learned. Easy to get backwards."""
        names = [n for n, _ in model.decoder.named_parameters()]
        assert "positional_embedding" in names
        assert model.decoder.positional_embedding.requires_grad


class TestAttention:
    def test_key_projection_has_no_bias(self):
        """Matches OpenAI. A constant added to every key cancels in softmax."""
        attn = MultiHeadAttention(64, 4)
        assert attn.key.bias is None
        assert attn.query.bias is not None
        assert attn.value.bias is not None

    def test_self_attention_shape(self):
        attn = MultiHeadAttention(64, 4)
        x = torch.randn(2, 10, 64)
        assert attn(x).shape == (2, 10, 64)

    def test_cross_attention_handles_different_lengths(self):
        """Text of length 10 attending to audio of length 50."""
        attn = MultiHeadAttention(64, 4)
        x, xa = torch.randn(2, 10, 64), torch.randn(2, 50, 64)
        assert attn(x, xa=xa).shape == (2, 10, 64)

    def test_causal_mask_blocks_the_future(self):
        """Position i must not see position i+1."""
        attn = MultiHeadAttention(64, 4)
        mask = torch.empty(10, 10).fill_(float("-inf")).triu_(1)
        x = torch.randn(1, 10, 64)

        out_a = attn(x, mask=mask)
        x2 = x.clone()
        x2[0, 7:] = torch.randn(3, 64)  # change only the future
        out_b = attn(x2, mask=mask)

        assert torch.allclose(out_a[0, :7], out_b[0, :7], atol=1e-5)
        assert not torch.allclose(out_a[0, 7:], out_b[0, 7:], atol=1e-5)


class TestResidualBlock:
    def test_encoder_block_has_no_cross_attention(self):
        block = ResidualAttentionBlock(64, 4, cross_attention=False)
        assert block.cross_attn is None
        assert block.cross_attn_ln is None

    def test_decoder_block_has_cross_attention(self):
        block = ResidualAttentionBlock(64, 4, cross_attention=True)
        assert block.cross_attn is not None

    def test_mlp_expands_by_four(self):
        block = ResidualAttentionBlock(64, 4)
        assert block.mlp[0].out_features == 256
        assert block.mlp[2].in_features == 256

    def test_is_pre_activation_not_post(self):
        """x + attn(ln(x)), not ln(x + attn(x)).

        With pre-norm the residual path is unnormalised, so zeroing the branch
        weights makes the block an exact identity. A post-norm block would
        still normalise its output and could not.
        """
        block = ResidualAttentionBlock(64, 4).eval()
        with torch.no_grad():
            block.attn.out.weight.zero_()
            block.attn.out.bias.zero_()
            block.mlp[2].weight.zero_()
            block.mlp[2].bias.zero_()
        x = torch.randn(1, 5, 64) * 7 + 3  # deliberately not unit-scaled
        assert torch.allclose(block(x), x, atol=1e-5)


class TestAudioEncoder:
    def test_stem_halves_the_sequence(self):
        """1500 mel frames -> 750 encoder positions, via the stride-2 conv."""
        enc = AudioEncoder(n_mels=80, n_ctx=750, n_state=64, n_head=4, n_layer=1)
        out = enc(torch.randn(2, 80, 1500))
        assert out.shape == (2, 750, 64)

    def test_paper_window_gives_1500_positions(self):
        enc = AudioEncoder(n_mels=80, n_ctx=1500, n_state=64, n_head=4, n_layer=1)
        assert enc(torch.randn(1, 80, 3000)).shape == (1, 1500, 64)

    def test_mismatched_window_gives_a_clear_error(self):
        enc = AudioEncoder(n_mels=80, n_ctx=750, n_state=64, n_head=4, n_layer=1)
        with pytest.raises(ValueError, match="disagree"):
            enc(torch.randn(1, 80, 3000))

    def test_conv_stem_shapes(self):
        enc = AudioEncoder(n_mels=80, n_ctx=750, n_state=64, n_head=4, n_layer=1)
        assert enc.conv1.kernel_size == (3,) and enc.conv1.stride == (1,)
        assert enc.conv2.kernel_size == (3,) and enc.conv2.stride == (2,)


class TestTextDecoder:
    def test_output_is_vocabulary_sized(self):
        dec = TextDecoder(n_vocab=100, n_ctx=16, n_state=64, n_head=4, n_layer=2)
        out = dec(torch.randint(0, 100, (2, 8)), torch.randn(2, 50, 64))
        assert out.shape == (2, 8, 100)

    def test_embeddings_are_tied(self, model, tiny_config):
        """Paper §2.2: tied input-output token representations.

        Verified behaviourally: perturbing one row of the embedding table must
        change that token's *output logit*, which only happens if the output
        projection really is the embedding matrix rather than a separate one.

        The perturbation must be a random vector, not a constant. The decoder
        ends in a LayerNorm, so its output x is zero-mean; adding c to every
        component of W[7] changes the logit by c * sum(x) = 0. A constant shift
        of an embedding row is invisible through a tied projection — which is a
        real (and slightly surprising) property of this architecture, not a
        quirk of the test.
        """
        model.eval()
        torch.manual_seed(0)
        xa = torch.randn(1, tiny_config.n_audio_ctx, tiny_config.n_audio_state)
        tokens = torch.zeros(1, 3, dtype=torch.long)
        delta = torch.randn(tiny_config.n_text_state) * 5

        with torch.no_grad():
            before = model.decoder(tokens, xa).clone()
            model.decoder.token_embedding.weight[7] += delta
            after = model.decoder(tokens, xa).clone()
            model.decoder.token_embedding.weight[7] -= delta

        assert (after[..., 7] - before[..., 7]).abs().max() > 1.0

    def test_a_constant_shift_of_an_embedding_row_is_invisible(self, model, tiny_config):
        """The flip side, worth pinning: the final LayerNorm makes x zero-mean,
        so adding a constant to a whole embedding row cannot change its logit."""
        model.eval()
        xa = torch.randn(1, tiny_config.n_audio_ctx, tiny_config.n_audio_state)
        tokens = torch.zeros(1, 3, dtype=torch.long)

        with torch.no_grad():
            before = model.decoder(tokens, xa).clone()
            model.decoder.token_embedding.weight[7] += 10.0
            after = model.decoder(tokens, xa).clone()
            model.decoder.token_embedding.weight[7] -= 10.0

        assert (after[..., 7] - before[..., 7]).abs().max() < 1e-4

    def test_there_is_no_separate_output_projection(self, model):
        """No extra vocab-sized weight anywhere — that is the saving."""
        vocab = model.decoder.token_embedding.weight.shape[0]
        vocab_sized = [
            n for n, p in model.named_parameters()
            if p.dim() == 2 and vocab in p.shape
        ]
        assert vocab_sized == ["decoder.token_embedding.weight"]

    def test_causal_mask_is_registered(self, model):
        assert "mask" in dict(model.decoder.named_buffers())
        mask = model.decoder.mask
        assert torch.isinf(mask[0, 1]) and mask[1, 0] == 0

    def test_decoder_is_autoregressive(self, model):
        """Changing a later token must not change an earlier position's logits."""
        model.eval()
        xa = torch.randn(1, model.config.n_audio_ctx, model.config.n_audio_state)
        tokens = torch.randint(1, 100, (1, 10))
        with torch.no_grad():
            a = model.decoder(tokens, xa)
            tokens2 = tokens.clone()
            tokens2[0, 6:] = torch.randint(1, 100, (4,))
            b = model.decoder(tokens2, xa)
        assert torch.allclose(a[0, :6], b[0, :6], atol=1e-5)


class TestWhisprModel:
    def test_forward_shape(self, model, tiny_config):
        mel = torch.randn(2, tiny_config.n_mels, tiny_config.n_audio_ctx * 2)
        tokens = torch.randint(0, tiny_config.n_vocab, (2, 8))
        assert model(mel, tokens).shape == (2, 8, tiny_config.n_vocab)

    def test_gradients_reach_everything(self, model, tiny_config):
        """A missing gradient means a dead subnetwork — silent and fatal."""
        mel = torch.randn(1, tiny_config.n_mels, tiny_config.n_audio_ctx * 2)
        tokens = torch.randint(0, tiny_config.n_vocab, (1, 8))
        model(mel, tokens).sum().backward()

        dead = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
        assert dead == [], f"no gradient reached: {dead}"

    def test_no_nans_at_init(self, model, tiny_config):
        mel = torch.randn(1, tiny_config.n_mels, tiny_config.n_audio_ctx * 2)
        out = model(mel, torch.randint(0, tiny_config.n_vocab, (1, 8)))
        assert torch.isfinite(out).all()

    def test_init_keeps_activations_sane(self, tiny_config):
        """Fan-in init should give roughly unit-scale logits, not 1e6 or 1e-6."""
        m = build_model(tiny_config).eval()
        mel = torch.randn(4, tiny_config.n_mels, tiny_config.n_audio_ctx * 2)
        with torch.no_grad():
            out = m(mel, torch.randint(0, tiny_config.n_vocab, (4, 8)))
        assert 0.01 < out.std().item() < 100

    def test_parameter_breakdown_adds_up(self, model):
        b = model.parameter_breakdown()
        assert b["encoder"] + b["decoder"] == b["total"]
        assert b["total"] == model.num_parameters()

    def test_batch_independence(self, model, tiny_config):
        """Row 0's output must not depend on row 1 — a real bug when masks leak."""
        model.eval()
        mel = torch.randn(2, tiny_config.n_mels, tiny_config.n_audio_ctx * 2)
        tokens = torch.randint(0, tiny_config.n_vocab, (2, 8))
        with torch.no_grad():
            both = model(mel, tokens)
            alone = model(mel[:1], tokens[:1])
        assert torch.allclose(both[0], alone[0], atol=1e-5)


class TestArchitectureMatchesThePaper:
    @pytest.mark.parametrize(
        "name,expected_millions",
        [("tiny", 37.18), ("base", 71.83), ("small", 240.58)],
    )
    def test_reproduces_released_parameter_counts(self, name, expected_millions):
        """Our architecture must produce the real checkpoints' parameter counts.

        Note these are the *actual* counts (tiny = 37.18M), which differ from
        the rounded figures in Table 1 (39M).
        """
        m = Whispr(ModelConfig(**WHISPER_DIMS[name]))
        assert m.num_parameters() / 1e6 == pytest.approx(expected_millions, abs=0.01)

    def test_tiny_shape_matches_table_1(self):
        d = WHISPER_DIMS["tiny"]
        assert (d["n_audio_layer"], d["n_audio_state"], d["n_audio_head"]) == (4, 384, 6)

    def test_our_config_is_tiny_shaped(self):
        assert Config().model.is_tiny_shaped


@pytest.mark.skipif(
    not WHISPER_CACHE.exists(),
    reason="whisper-tiny checkpoint not cached; run the reference tests once to fetch it",
)
class TestAgainstRealCheckpoint:
    """The decisive test: real Whisper weights loaded into our class."""

    @pytest.fixture(scope="class")
    def pair(self):
        whisper = pytest.importorskip("whisper")
        real = whisper.load_model("tiny").eval()
        ours = Whispr(ModelConfig(**WHISPER_DIMS["tiny"])).eval()
        ours.load_state_dict(real.state_dict(), strict=True)
        return real, ours

    def test_state_dict_loads_strictly(self, pair):
        """strict=True means every key and every shape matches exactly."""
        real, ours = pair
        assert set(real.state_dict()) == set(ours.state_dict())

    def test_encoder_output_is_identical(self, pair):
        real, ours = pair
        mel = torch.randn(1, 80, 3000)
        with torch.no_grad():
            assert torch.equal(real.encoder(mel), ours.encoder(mel))

    def test_decoder_logits_are_identical(self, pair):
        real, ours = pair
        mel = torch.randn(1, 80, 3000)
        tokens = torch.tensor([[50258, 50259, 50359, 50363, 1770, 13]])
        with torch.no_grad():
            xa = real.encoder(mel)
            assert torch.equal(real.decoder(tokens, xa), ours.decoder(tokens, xa))
