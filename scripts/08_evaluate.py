"""Step 8 — decoding and WER. See notes/08-decoding-and-wer.md.

    uv run python scripts/08_evaluate.py
    uv run python scripts/08_evaluate.py --beam 5 --limit 100
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from whispr.config import Config
from whispr.data import (
    DEFAULT_ROOT,
    LibriSpeechDataset,
    cached_index,
    collate,
    speaker_split,
)
from whispr.decode import Decoder, normalise, word_error_rate
from whispr.device import get_device
from whispr.model import build_model
from whispr.plotting import ACCENT, ACCENT2, MUTED, plt, save, use_style
from whispr.tokenizer import WhisprTokenizer

CHECKPOINTS = Path(__file__).resolve().parent.parent / "checkpoints"
TOKENIZER_PATH = DEFAULT_ROOT.parent / "tokenizer.json"
RESULTS = Path(__file__).resolve().parent.parent / "results"


def load_trained(config: Config, checkpoint: str, device):
    path = CHECKPOINTS / checkpoint
    if not path.exists():
        sys.exit(f"{path} not found — train first with scripts/07_train.py")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model(config.model)
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), ckpt.get("step"), ckpt.get("best_val_loss")


@torch.no_grad()
def transcribe_split(decoder, dataset, tokenizer, config, limit=None, beam=0, batch_size=8):
    refs, hyps, logprobs = [], [], []
    n = len(dataset) if limit is None else min(limit, len(dataset))

    start = time.perf_counter()
    for i in range(0, n, batch_size):
        items = [dataset[j] for j in range(i, min(i + batch_size, n))]
        batch = collate(items, pad_token=tokenizer.special.pad)
        results = (
            decoder.beam(batch["mel"], beam_size=beam)
            if beam
            else decoder.greedy(batch["mel"])
        )
        for item, r in zip(items, results):
            refs.append(item["text"])
            hyps.append(r.text)
            logprobs.append(r.avg_logprob)
        print(f"\r  {len(refs)}/{n} utterances", end="", flush=True)

    audio_seconds = sum(dataset.utterances[j].duration for j in range(n))
    elapsed = time.perf_counter() - start
    print(f"\r  {len(refs)}/{n} utterances — {elapsed:.0f}s "
          f"({audio_seconds/elapsed:.1f}x realtime)")
    return refs, hyps, logprobs


def fig_examples(refs, hyps, name: str) -> None:
    """Show the best, median and worst transcriptions — the honest picture."""
    scored = sorted(
        (
            (word_error_rate([r], [h])["wer"], r, h)
            for r, h in zip(refs, hyps)
            if len(r.split()) >= 6
        ),
        key=lambda x: x[0],
    )
    if not scored:
        return
    picks = [
        ("best", scored[0]),
        ("25th pct", scored[len(scored) // 4]),
        ("median", scored[len(scored) // 2]),
        ("worst", scored[-1]),
    ]

    fig, ax = plt.subplots(figsize=(11.0, 4.2))
    ax.axis("off")
    y = 0.97
    for label, (wer, ref, hyp) in picks:
        ax.text(0, y, f"{label}  —  WER {wer:.0%}", fontsize=9, weight="bold",
                color=ACCENT if wer > 0.5 else ACCENT2, transform=ax.transAxes)
        y -= 0.075
        for tag, text, color in (("REF", ref, MUTED), ("HYP", hyp, ACCENT)):
            ax.text(0.02, y, f"{tag}  {text[:96]}", fontsize=7.5, family="monospace",
                    color=color, transform=ax.transAxes)
            y -= 0.055
        y -= 0.045
    fig.suptitle(f"Transcriptions on held-out speakers ({name})",
                 x=0.02, ha="left", weight="bold")
    fig.tight_layout()
    save(fig, f"08_examples_{name}.png")


def fig_wer_distribution(refs, hyps, logprobs) -> None:
    wers = [word_error_rate([r], [h])["wer"] for r, h in zip(refs, hyps)]
    lengths = [len(normalise(r).split()) for r in refs]

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.0))

    axes[0].hist(wers, bins=30, color=ACCENT, alpha=0.85)
    axes[0].axvline(word_error_rate(refs, hyps)["wer"], color=ACCENT2, lw=1.8,
                    label="corpus WER")
    axes[0].set_xlabel("per-utterance WER")
    axes[0].set_ylabel("count")
    axes[0].set_title("Most utterances are bad; a few are fine", loc="left")
    axes[0].legend(fontsize=7)

    axes[1].scatter(lengths, wers, s=8, alpha=0.4, color=ACCENT)
    axes[1].set_xlabel("reference length (words)")
    axes[1].set_ylabel("WER")
    axes[1].set_title("Longer utterances are harder\n(more chances to derail)", loc="left")

    axes[2].scatter(logprobs, wers, s=8, alpha=0.4, color=ACCENT)
    axes[2].set_xlabel("model's own avg logprob")
    axes[2].set_ylabel("WER")
    axes[2].set_title("Does confidence predict correctness?", loc="left")

    fig.suptitle("Where the errors are", x=0.02, ha="left", weight="bold")
    fig.tight_layout()
    save(fig, "08_wer_distribution.png")


def report(refs, hyps, label: str) -> dict:
    r = word_error_rate(refs, hyps)
    exact = sum(normalise(a) == normalise(b) for a, b in zip(refs, hyps))
    empty = sum(1 for h in hyps if not normalise(h))
    print(f"\n{label}:")
    print(f"  WER              {r['wer']:.2%}  ({r['errors']:,} errors / {r['words']:,} words)")
    print(f"  exact matches    {exact}/{len(refs)} ({exact/len(refs):.1%})")
    print(f"  empty outputs    {empty}")
    return {"label": label, **r, "exact": exact, "n": len(refs)}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="best.pt")
    p.add_argument("--beam", type=int, default=0, help="0 = greedy")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--compare", action="store_true", help="greedy vs beam on a subset")
    args = p.parse_args()

    if not DEFAULT_ROOT.exists():
        sys.exit("Corpus not found. Run: uv run python scripts/04_data.py --download")

    use_style()
    RESULTS.mkdir(exist_ok=True)
    config = Config()
    device = get_device()

    tokenizer = WhisprTokenizer.load(TOKENIZER_PATH)
    model, step, best_val = load_trained(config, args.checkpoint, device)
    decoder = Decoder(model, tokenizer, device=device)

    index = cached_index(DEFAULT_ROOT, "dev-clean")
    train_utts, val_utts = speaker_split(index)
    val_ds = LibriSpeechDataset(val_utts, config.audio, tokenizer)
    train_ds = LibriSpeechDataset(train_utts, config.audio, tokenizer)

    print(f"Step 8 — decoding and WER\n")
    print(f"checkpoint : {args.checkpoint} (step {step}, val loss {best_val:.4f})")
    print(f"parameters : {model.num_parameters():,}")
    print(f"decoding   : {'beam ' + str(args.beam) if args.beam else 'greedy'}\n")

    summary = []

    print("Held-out speakers (the honest number):")
    refs, hyps, lps = transcribe_split(decoder, val_ds, tokenizer, config,
                                       limit=args.limit, beam=args.beam)
    summary.append(report(refs, hyps, "held-out speakers (unseen)"))
    fig_examples(refs, hyps, "heldout")
    fig_wer_distribution(refs, hyps, lps)

    print("\nTraining speakers (for comparison — shows the overfitting):")
    n_train = args.limit or len(refs)
    t_refs, t_hyps, _ = transcribe_split(decoder, train_ds, tokenizer, config,
                                         limit=n_train, beam=args.beam)
    summary.append(report(t_refs, t_hyps, "training speakers (seen)"))
    fig_examples(t_refs, t_hyps, "train")

    gap = summary[0]["wer"] - summary[1]["wer"]
    print(f"\n  generalisation gap: {gap:+.1%} WER "
          f"({summary[1]['wer']:.1%} seen -> {summary[0]['wer']:.1%} unseen)")

    if args.compare:
        print("\nGreedy vs beam on 60 held-out utterances:")
        for b in (0, 3, 5):
            r, h, _ = transcribe_split(decoder, val_ds, tokenizer, config, limit=60, beam=b)
            w = word_error_rate(r, h)["wer"]
            print(f"  {'greedy' if not b else f'beam {b}':<10} WER {w:.2%}")

    (RESULTS / "wer.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {RESULTS/'wer.json'}")
