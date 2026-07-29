"""Step 5 — the tokenizer and the multitask format. See notes/05-tokenizer.md.

    uv run python scripts/05_tokenizer.py
"""

from __future__ import annotations

import collections
import sys

import numpy as np

from whispr.config import ModelConfig
from whispr.data import DEFAULT_ROOT, cached_index, speaker_split
from whispr.plotting import ACCENT, ACCENT2, MUTED, plt, save, use_style
from whispr.tokenizer import SPECIAL_TOKENS, WhisprTokenizer

TOKENIZER_PATH = DEFAULT_ROOT.parent / "tokenizer.json"
VOCAB_SIZES = (256, 512, 1024, 2048, 4096)


def sweep(train_texts, val_texts):
    rows = []
    for v in VOCAB_SIZES:
        tok = WhisprTokenizer.train(train_texts, vocab_size=v)
        rows.append(
            {
                "vocab": tok.n_vocab,
                "train": tok.compression_stats(train_texts)["tokens_per_word"],
                "val": tok.compression_stats(val_texts)["tokens_per_word"],
                "params": tok.n_vocab * ModelConfig().n_text_state,
            }
        )
    return rows


def fig_vocab_tradeoff(rows) -> None:
    v = [r["vocab"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.2))

    axes[0].plot(v, [r["train"] for r in rows], "o-", color=ACCENT, lw=1.8, label="train")
    axes[0].plot(v, [r["val"] for r in rows], "s--", color=ACCENT2, lw=1.6, label="held-out")
    axes[0].axvline(2048, color=MUTED, lw=1.0, ls=":")
    axes[0].set_xscale("log", base=2)
    axes[0].set_xlabel("vocabulary size")
    axes[0].set_ylabel("tokens per word")
    axes[0].set_title("Compression flattens past 2048\n(the gap is the vocabulary overfitting)",
                      loc="left")
    axes[0].legend(fontsize=7)
    axes[0].grid(True, lw=0.5)

    gpt2 = 50_257 * ModelConfig().n_text_state
    axes[1].plot(v, [r["params"] / 1e6 for r in rows], "o-", color=ACCENT, lw=1.8)
    axes[1].axhline(gpt2 / 1e6, color=MUTED, lw=1.4, ls="--")
    axes[1].text(300, gpt2 / 1e6 * 0.72,
                 f"GPT-2's 50,257 tokens = {gpt2/1e6:.1f}M\n(half of Tiny's entire 39M budget)",
                 fontsize=7.5, color=MUTED)
    axes[1].axvline(2048, color=MUTED, lw=1.0, ls=":")
    axes[1].set_xscale("log", base=2)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("vocabulary size")
    axes[1].set_ylabel("embedding parameters (M)")
    axes[1].set_title("Cost of the tied embedding table", loc="left")
    axes[1].grid(True, lw=0.5)

    fig.suptitle("Choosing the vocabulary by measurement, not by inheriting GPT-2's",
                 x=0.02, ha="left", weight="bold")
    fig.tight_layout()
    save(fig, "05_vocab_tradeoff.png")


def fig_sequence_lengths(tok, index) -> None:
    keep = [u for u in index if u.duration <= 15.0]
    lens = np.array([len(tok.encode_training(u.text)) for u in keep])
    ctx = ModelConfig().n_text_ctx

    fig, ax = plt.subplots(figsize=(7.5, 3.0))
    ax.hist(lens, bins=50, color=ACCENT, alpha=0.85)
    ax.axvline(ctx, color=ACCENT2, lw=1.6, label=f"n_text_ctx = {ctx}")
    ax.axvline(448, color=MUTED, lw=1.4, ls="--", label="paper's 448")
    ax.set_xlabel("tokens per training sequence (prefix + text + EOT)")
    ax.set_ylabel("count")
    ax.set_title(f"median {np.median(lens):.0f} · p99 {np.percentile(lens,99):.0f} · "
                 f"max {lens.max()} — 128 is comfortable", loc="left")
    ax.legend(fontsize=7)
    fig.tight_layout()
    save(fig, "05_sequence_lengths.png")


def fig_token_frequency(tok, texts) -> None:
    counts = collections.Counter()
    for t in texts:
        counts.update(tok.encode(t))
    freqs = np.array(sorted(counts.values(), reverse=True))

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.0))
    axes[0].loglog(np.arange(1, len(freqs) + 1), freqs, color=ACCENT, lw=1.4)
    axes[0].set_xlabel("token rank")
    axes[0].set_ylabel("frequency")
    axes[0].set_title("Zipf's law, as always", loc="left")
    axes[0].grid(True, lw=0.5, which="both")

    unseen = tok.n_vocab - len(counts)
    rare = sum(1 for c in counts.values() if c < 5)
    axes[1].bar(["seen 5+", "seen <5", "never"],
                [len(counts) - rare, rare, unseen],
                color=[ACCENT, ACCENT2, MUTED])
    axes[1].set_ylabel("tokens")
    axes[1].set_title(f"Of {tok.n_vocab} tokens, {unseen} never occur\n"
                      "and their embeddings never get a gradient", loc="left")
    fig.suptitle("Token statistics on 46k words of training text",
                 x=0.02, ha="left", weight="bold")
    fig.tight_layout()
    save(fig, "05_token_frequency.png")


def demo(tok) -> None:
    print("\nThe multitask format (paper §2.3) — the task is a prefix, not an architecture:")
    text = "THE QUICK BROWN FOX"
    for task, ts in (("transcribe", False), ("translate", False), ("transcribe", True)):
        ids = tok.sot_sequence(task=task, timestamps=ts)
        print(f"  {task:<11} timestamps={str(ts):<5} -> {tok.decode(ids, skip_special=False)}")

    print("\nA full training target:")
    ids = tok.encode_training(text)
    print(f"  text    : {text}")
    print(f"  ids     : {ids}")
    print(f"  decoded : {tok.decode(ids, skip_special=False)}")
    print(f"  prefix  : first {tok.prompt_length} tokens are task spec, not content")

    print("\nSubword behaviour:")
    for word in ("THE", "QUICK", "PARTICULARLY", "ZANZIBAR", "QUILTER'S"):
        pieces = [tok.decode([i]) for i in tok.encode(word)]
        print(f"  {word:<14} -> {pieces}")

    print("\nSpecial token ids:")
    for t in SPECIAL_TOKENS:
        print(f"  {tok.token_id(t):>3}  {t}")


def report(tok, train_texts, val_texts) -> None:
    s = tok.compression_stats(train_texts)
    sv = tok.compression_stats(val_texts)
    print(f"\nChosen vocabulary: {tok.n_vocab}")
    print(f"  tokens/word   : {s['tokens_per_word']:.2f} train, {sv['tokens_per_word']:.2f} held-out")
    print(f"  chars/token   : {s['chars_per_token']:.2f}")
    print(f"  embedding cost: {tok.n_vocab * 384:,} params (tied input/output)")
    print(f"  vs GPT-2's    : {50_257 * 384:,} params ({50_257*384/(tok.n_vocab*384):.0f}x more)")

    ok = all(tok.decode(tok.encode(t)) == t for t in train_texts[:500])
    print(f"\n  roundtrip decode(encode(t)) == t on 500 transcripts: {ok}")


if __name__ == "__main__":
    if not DEFAULT_ROOT.exists():
        sys.exit("Corpus not found. Run: uv run python scripts/04_data.py --download")

    use_style()
    print("Step 5 — the tokenizer and the multitask format\n")

    index = cached_index(DEFAULT_ROOT, "dev-clean")
    train, val = speaker_split(index)
    train_texts = [u.text for u in train]  # fit on TRAIN ONLY — see notes §4
    val_texts = [u.text for u in val]

    print(f"Fitting BPE on {len(train_texts):,} training transcripts "
          f"({sum(len(t.split()) for t in train_texts):,} words)")

    rows = sweep(train_texts, val_texts)
    print(f"\n{'vocab':>6} {'tok/word':>9} {'held-out':>9} {'emb params':>12}")
    for r in rows:
        mark = "  <- chosen" if r["vocab"] == 2048 else ""
        print(f"{r['vocab']:>6} {r['train']:>9.2f} {r['val']:>9.2f} {r['params']:>12,}{mark}")

    fig_vocab_tradeoff(rows)

    tok = WhisprTokenizer.train(train_texts, vocab_size=ModelConfig().n_vocab)
    tok.save(TOKENIZER_PATH)
    print(f"\nSaved tokenizer to {TOKENIZER_PATH}")

    fig_sequence_lengths(tok, index)
    fig_token_frequency(tok, train_texts)
    report(tok, train_texts, val_texts)
    demo(tok)
