"""Step 4 — the LibriSpeech pipeline. See notes/04-data.md.

    uv run python scripts/04_data.py --download   # fetch the corpus first
    uv run python scripts/04_data.py
"""

from __future__ import annotations

import argparse
import collections
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np

from whispr.config import AudioConfig
from whispr.data import (
    DEFAULT_ROOT,
    LibriSpeechDataset,
    cached_index,
    collate,
    describe,
    speaker_split,
)
from whispr.plotting import ACCENT, ACCENT2, MUTED, plt, save, use_style

DATA = DEFAULT_ROOT.parent
URL = "https://www.openslr.org/resources/12/dev-clean.tar.gz"


def download(split: str = "dev-clean") -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    tgz = DATA / f"{split}.tar.gz"
    url = f"https://www.openslr.org/resources/12/{split}.tar.gz"
    if not tgz.exists():
        print(f"Downloading {url} ...")
        subprocess.run(["curl", "-L", "--retry", "3", "-o", str(tgz), url], check=True)
    if not (DEFAULT_ROOT / split).exists():
        print("Extracting...")
        with tarfile.open(tgz) as t:
            t.extractall(DATA, filter="data")
    print(f"Ready: {DEFAULT_ROOT / split}")


def precompute(split: str, window: float | None) -> None:
    """Write a memory-mapped log-mel cache for `split`.

    Worth knowing what this does and does not buy (measured, notes/07 §7):
    data loading drops from 33 ms to 2 ms per batch of 8 — 20x — but that was
    only ~4.5% of a training step, so training gets ~4% faster, not 1.5x.
    Validation gains more (61 s -> 51 s over dev-clean) because it has no
    backward pass to hide the loading behind.
    """
    from whispr import melcache

    cfg = AudioConfig(window_seconds=window) if window else AudioConfig()
    utts = cached_index(DEFAULT_ROOT, split)

    def prog(i, n, elapsed):
        rate = i / max(elapsed, 1e-9)
        print(f"\r  {i}/{n}  {rate:.0f} utt/s  eta {(n-i)/max(rate,1e-9)/60:.1f} min",
              end="", flush=True)

    cache = melcache.build(utts, cfg, split, on_progress=prog, overwrite=True)
    npy, _ = melcache.cache_paths(split, cfg)
    print(f"\r  {split}: {len(cache)} utts at {cfg.window_seconds:g}s -> "
          f"{npy.stat().st_size/1e9:.2f} GB" + " " * 30)
    print(f"  {npy}")


def fig_corpus_stats(index) -> None:
    durs = np.array([u.duration for u in index])
    words = np.array([len(u.text.split()) for u in index])
    cfg = AudioConfig()

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.0))

    axes[0].hist(durs, bins=60, color=ACCENT, alpha=0.85)
    axes[0].axvline(cfg.window_seconds, color=ACCENT2, lw=1.6,
                    label=f"our window {cfg.window_seconds:g}s")
    axes[0].axvline(30, color=MUTED, lw=1.4, ls="--", label="paper window 30s")
    axes[0].set_xlabel("utterance duration (s)")
    axes[0].set_ylabel("count")
    axes[0].set_title(f"Median {np.median(durs):.1f}s — a 30s window\nwould be mostly padding",
                      loc="left")
    axes[0].legend(fontsize=7)

    axes[1].scatter(durs, words, s=3, alpha=0.25, color=ACCENT)
    axes[1].set_xlabel("duration (s)")
    axes[1].set_ylabel("words")
    rate = words.sum() / durs.sum()
    axes[1].set_title(f"Tight linear fit — {rate:.1f} words/s\n(read speech is very regular)",
                      loc="left")

    # Coverage / cost tradeoff.
    windows = np.arange(5, 31)
    kept = [100 * (durs <= w).mean() for w in windows]
    cost = [(w / cfg.window_seconds) ** 2 for w in windows]
    ax2 = axes[2]
    ax2.plot(windows, kept, color=ACCENT, lw=1.8, label="% utterances kept")
    ax2.set_xlabel("window (s)")
    ax2.set_ylabel("% kept", color=ACCENT)
    ax2.axvline(cfg.window_seconds, color=ACCENT2, lw=1.2, ls=":")
    twin = ax2.twinx()
    twin.plot(windows, cost, color=MUTED, lw=1.6, ls="--")
    twin.set_ylabel("relative attention cost", color=MUTED)
    twin.spines["right"].set_visible(True)
    ax2.set_title("The tradeoff we resolved at 15s", loc="left")

    fig.suptitle("LibriSpeech dev-clean: " + describe(index), x=0.02, ha="left", weight="bold")
    fig.tight_layout()
    save(fig, "04_corpus_stats.png")


def fig_speakers(index) -> None:
    train, val = speaker_split(index)
    val_speakers = {u.speaker for u in val}

    by_spk = collections.Counter()
    for u in index:
        by_spk[u.speaker] += u.duration
    speakers = sorted(by_spk, key=lambda s: -by_spk[s])
    mins = [by_spk[s] / 60 for s in speakers]
    colors = [ACCENT2 if s in val_speakers else ACCENT for s in speakers]

    fig, ax = plt.subplots(figsize=(9.5, 3.0))
    ax.bar(range(len(speakers)), mins, color=colors)
    ax.set_xlabel("speaker (sorted by amount of audio)")
    ax.set_ylabel("minutes")
    ax.set_title(
        f"Speaker-disjoint split — {len(val_speakers)} held-out speakers in blue.\n"
        "A random utterance split would put the same voice on both sides.",
        loc="left",
    )
    fig.tight_layout()
    save(fig, "04_speaker_split.png")


def fig_batch(train) -> None:
    ds = LibriSpeechDataset(train, AudioConfig())
    batch = collate([ds[i] for i in range(4)])
    mels = batch["mel"]

    fig, axes = plt.subplots(4, 1, figsize=(9.0, 6.0))
    for ax, m, text, utt in zip(axes, mels, batch["text"], batch["utt_id"]):
        ax.imshow(m.numpy(), origin="lower", aspect="auto", cmap="magma")
        ax.set_ylabel("mel")
        ax.set_title(f"{utt} — {text[:64]}{'...' if len(text) > 64 else ''}",
                     loc="left", fontsize=8)
        ax.set_xticks([])
    axes[-1].set_xlabel("frame (padding is the flat region on the right)")
    fig.suptitle(f"One batch: {tuple(mels.shape)} — this is what the encoder eats",
                 x=0.02, ha="left", weight="bold")
    fig.tight_layout()
    save(fig, "04_batch.png")


def report(index) -> None:
    cfg = AudioConfig()
    train, val = speaker_split(index)
    train_ds = LibriSpeechDataset(train, cfg)
    val_ds = LibriSpeechDataset(val, cfg)

    print("\nCorpus:")
    print("  all      :", describe(index))
    print("  train    :", describe(train))
    print("  val      :", describe(val))
    print(f"  overlap  : {({u.speaker for u in train} & {u.speaker for u in val}) or 'none'}")

    print(f"\nAfter dropping utterances over {cfg.window_seconds:g}s:")
    print(f"  train    : {len(train_ds):,} utts, {train_ds.total_hours():.2f} h "
          f"({train_ds.dropped} dropped)")
    print(f"  val      : {len(val_ds):,} utts, {val_ds.total_hours():.2f} h "
          f"({val_ds.dropped} dropped)")

    print("\nTensor shapes:")
    item = train_ds[0]
    print(f"  mel      : {tuple(item['mel'].shape)}  (n_mels x n_frames)")
    print(f"  encoder  : {cfg.n_audio_ctx} positions after the conv stem")

    chars = collections.Counter("".join(u.text for u in index))
    print(f"\nLabel alphabet: {len(chars)} characters")
    print(f"  {''.join(sorted(chars))!r}")
    print(f"  GPT-2's vocabulary is 50,257 tokens. For 28 characters and "
          f"{sum(len(u.text.split()) for u in index):,} words,")
    print("  that is the argument for training our own BPE in step 5.")

    print("\nScale check against the paper:")
    hours = sum(u.duration for u in index) / 3600
    print(f"  ours     : {hours:.2f} h")
    print(f"  paper    : 680,000 h  ({680_000/hours:,.0f}x more)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--download", action="store_true", help="fetch and extract the corpus")
    p.add_argument("--split", default="dev-clean", help="which LibriSpeech split")
    p.add_argument("--precompute", action="store_true",
                   help="write a precomputed log-mel cache for --split")
    p.add_argument("--window", type=float, default=None,
                   help="window seconds for --precompute (default 15)")
    args = p.parse_args()

    if args.download:
        download(args.split)

    if args.precompute:
        precompute(args.split, args.window)
        sys.exit(0)

    if not DEFAULT_ROOT.exists():
        sys.exit("Corpus not found. Run: uv run python scripts/04_data.py --download")

    use_style()
    print("Step 4 — the LibriSpeech pipeline\n")
    index = cached_index(DEFAULT_ROOT, "dev-clean")
    train, _ = speaker_split(index)
    fig_corpus_stats(index)
    fig_speakers(index)
    fig_batch(train)
    report(index)
