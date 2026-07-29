"""Step 6 — the model. See notes/06-model.md.

    uv run python scripts/06_model.py
"""

from __future__ import annotations

import time

import torch

from whispr.config import Config, ModelConfig
from whispr.device import describe as describe_device
from whispr.device import get_device
from whispr.model import WHISPER_DIMS, Whispr, build_model, sinusoids
from whispr.plotting import ACCENT, ACCENT2, MUTED, plt, save, use_style


def fig_sinusoids() -> None:
    pe = sinusoids(750, 384)
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.0))

    im = axes[0].imshow(pe.numpy().T, aspect="auto", cmap="RdBu_r", origin="lower")
    axes[0].set_xlabel("position (encoder frame)")
    axes[0].set_ylabel("channel")
    axes[0].set_title("Sinusoidal positional encoding (750 x 384)", loc="left")
    fig.colorbar(im, ax=axes[0], fraction=0.046)

    for pos, color in ((0, ACCENT), (1, ACCENT2), (375, MUTED)):
        axes[1].plot(pe[pos].numpy()[:96], lw=1.1, color=color, label=f"position {pos}")
    axes[1].set_xlabel("channel")
    axes[1].set_title("Each position gets a distinct signature", loc="left")
    axes[1].legend(fontsize=7)

    sim = (pe @ pe.T)[0] / pe[0].norm() ** 2
    axes[2].plot(sim.numpy(), color=ACCENT, lw=1.2)
    axes[2].set_xlabel("position")
    axes[2].set_ylabel("similarity to position 0")
    axes[2].set_title("Similarity decays smoothly with distance", loc="left")
    axes[2].grid(True, lw=0.5)

    fig.suptitle("Encoder positions are physical time, so the encoding is fixed, not learned",
                 x=0.02, ha="left", weight="bold")
    fig.tight_layout()
    save(fig, "06_sinusoids.png")


def fig_causal_mask() -> None:
    n = 24
    mask = torch.empty(n, n).fill_(float("-inf")).triu_(1)
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.2))

    axes[0].imshow((mask == 0).float().numpy(), cmap="magma", origin="upper")
    axes[0].set_xlabel("key position (what we attend TO)")
    axes[0].set_ylabel("query position (what we attend FROM)")
    axes[0].set_title("Decoder: causal — position i sees only 0..i", loc="left")

    axes[1].imshow(torch.ones(n, n).numpy(), cmap="magma", origin="upper", vmin=0, vmax=1)
    axes[1].set_xlabel("key position")
    axes[1].set_title("Encoder: bidirectional — the audio is all available", loc="left")

    fig.suptitle("The only structural difference in attention between the two stacks",
                 x=0.02, ha="left", weight="bold")
    fig.tight_layout()
    save(fig, "06_causal_mask.png")


def fig_parameter_breakdown() -> None:
    cfg = Config()
    ours = build_model(cfg.model)
    tiny = Whispr(ModelConfig(**WHISPER_DIMS["tiny"]))

    def parts(m):
        enc = sum(p.numel() for p in m.encoder.parameters())
        emb = m.decoder.token_embedding.weight.numel()
        pos = m.decoder.positional_embedding.numel()
        dec = sum(p.numel() for p in m.decoder.parameters()) - emb - pos
        return [enc, dec, emb, pos]

    labels = ["encoder", "decoder\n(non-embedding)", "token\nembedding", "positional\nembedding"]
    colors = [ACCENT, ACCENT2, MUTED, "#d8b45a"]

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))
    for ax, m, name in ((axes[0], ours, "ours (vocab 2048)"),
                        (axes[1], tiny, "whisper-tiny (vocab 51,865)")):
        p = parts(m)
        ax.bar(labels, [v / 1e6 for v in p], color=colors)
        ax.set_ylabel("parameters (M)")
        ax.set_title(f"{name} — {sum(p)/1e6:.1f}M total", loc="left")
        ax.tick_params(axis="x", labelsize=7)
        for i, v in enumerate(p):
            ax.text(i, v / 1e6, f"{v/1e6:.1f}M", ha="center", va="bottom", fontsize=7)

    fig.suptitle("Almost the entire size difference is the vocabulary table",
                 x=0.02, ha="left", weight="bold")
    fig.tight_layout()
    save(fig, "06_parameters.png")


def report() -> None:
    print("\nReproducing the released models' parameter counts:")
    print(f"  {'model':<8}{'layers':>7}{'width':>7}{'heads':>7}{'ours':>11}{'Table 1':>10}")
    for name, table1 in (("tiny", "39M"), ("base", "74M"), ("small", "244M")):
        d = WHISPER_DIMS[name]
        m = Whispr(ModelConfig(**d))
        print(f"  {name:<8}{d['n_audio_layer']:>7}{d['n_audio_state']:>7}"
              f"{d['n_audio_head']:>7}{m.num_parameters()/1e6:>10.2f}M{table1:>10}")
    print("  Table 1 rounds up by ~5%; we match the actual checkpoints exactly.")

    cfg = Config()
    model = build_model(cfg.model)
    b = model.parameter_breakdown()
    print(f"\nOur replication model ({cfg.audio.window_seconds:g}s window, vocab {cfg.model.n_vocab}):")
    for k, v in b.items():
        print(f"  {k:<32}{v:>12,}")

    print("\nShapes through the stack:")
    mel = torch.randn(2, cfg.model.n_mels, cfg.audio.n_frames)
    tokens = torch.randint(0, cfg.model.n_vocab, (2, 24))
    with torch.no_grad():
        feats = model.encoder(mel)
        logits = model.decoder(tokens, feats)
    print(f"  mel            {tuple(mel.shape)}")
    print(f"  audio features {tuple(feats.shape)}   (conv stem halved 1500 -> 750)")
    print(f"  tokens         {tuple(tokens.shape)}")
    print(f"  logits         {tuple(logits.shape)}")

    # Gradient coverage — a dead subnetwork is silent and fatal.
    model.zero_grad()
    model(mel, tokens).sum().backward()
    dead = [n for n, p in model.named_parameters() if p.grad is None]
    print(f"\n  parameters with no gradient: {dead or 'none'}")

    device = get_device()
    print(f"\nTiming on {describe_device(device)} (measured, not extrapolated):")
    model = model.to(device)

    def time_batch(bs: int, n: int = 8) -> float:
        m = torch.randn(bs, cfg.model.n_mels, cfg.audio.n_frames, device=device)
        t = torch.randint(0, cfg.model.n_vocab, (bs, 24), device=device)
        for _ in range(3):  # warm up the Metal kernels
            model(m, t).sum().backward()
        if device.type == "mps":
            torch.mps.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            model.zero_grad(set_to_none=True)
            model(m, t).sum().backward()
        if device.type == "mps":
            torch.mps.synchronize()
        return (time.perf_counter() - t0) / n

    print(f"  {'batch':>6}{'ms/step':>10}{'utts/s':>9}{'12k steps':>12}")
    for bs in (2, 4, 8, 16):
        dt = time_batch(bs)
        print(f"  {bs:>6}{dt*1000:>10.0f}{bs/dt:>9.1f}{dt*12_000/60:>10.0f} min")


if __name__ == "__main__":
    use_style()
    print("Step 6 — the encoder-decoder Transformer\n")
    fig_sinusoids()
    fig_causal_mask()
    fig_parameter_breakdown()
    report()
