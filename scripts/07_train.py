"""Step 7 — training. See notes/07-training.md.

    uv run python scripts/07_train.py --sanity     # overfit one batch first
    uv run python scripts/07_train.py              # the real run
    uv run python scripts/07_train.py --plot       # figures from history.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from whispr.config import AudioConfig, Config, ModelConfig, TrainConfig
from whispr.data import (
    DEFAULT_ROOT,
    LibriSpeechDataset,
    cached_index,
    speaker_split,
    standard_split,
)
from whispr.device import describe as describe_device
from whispr.device import get_device
from whispr.model import build_model
from whispr.plotting import ACCENT, ACCENT2, MUTED, plt, save, use_style
from whispr.tokenizer import WhisprTokenizer, load_or_train
from whispr.train import Trainer, expected_initial_loss, overfit_one_batch

CHECKPOINTS = Path(__file__).resolve().parent.parent / "checkpoints"
TOKENIZER_PATH = DEFAULT_ROOT.parent / "tokenizer.json"


def setup(config: Config, augment: bool = True, corpus: str = "dev-clean"):
    """Build datasets.

    `corpus="dev-clean"` splits dev-clean by speaker (the 3.7 h baseline).
    `corpus="train-clean-100"` uses LibriSpeech's own protocol — train on
    train-clean-100, validate on all of dev-clean — which is both
    speaker-disjoint and the partition every published number uses.
    """
    if corpus == "dev-clean":
        index = cached_index(DEFAULT_ROOT, "dev-clean")
        train_utts, val_utts = speaker_split(index)
        tokenizer_path = TOKENIZER_PATH
    else:
        train_utts, val_utts = standard_split(DEFAULT_ROOT, corpus, "dev-clean")
        tokenizer_path = DEFAULT_ROOT.parent / f"tokenizer_{corpus}.json"

    tokenizer = load_or_train(
        tokenizer_path, [u.text for u in train_utts], vocab_size=config.model.n_vocab
    )
    train_ds = LibriSpeechDataset(train_utts, config.audio, tokenizer, augment=augment)
    val_ds = LibriSpeechDataset(val_utts, config.audio, tokenizer, augment=False)
    return tokenizer, train_ds, val_ds


def sanity(config: Config, corpus: str = "dev-clean") -> None:
    """Overfit one batch. If this fails, nothing else is worth running."""
    tokenizer, train_ds, _ = setup(config, augment=False, corpus=corpus)
    model = build_model(config.model)

    expected = expected_initial_loss(config.model.n_vocab)
    print(f"A model that knows nothing should start near ln({config.model.n_vocab}) = {expected:.3f}")
    print(f"Overfitting one batch of {config.train.batch_size} for 150 steps...\n")

    losses = overfit_one_batch(model, tokenizer, config, train_ds, steps=150)

    for i in (0, 10, 25, 50, 100, 149):
        print(f"  step {i:>4}  loss {losses[i]:.4f}")

    ok = losses[-1] < 0.05
    print(f"\n  initial loss   {losses[0]:.3f}  (expected ~{expected:.3f})")
    print(f"  final loss     {losses[-1]:.4f}")
    print(f"  VERDICT: {'PASS — the model can memorise, so the plumbing is correct' if ok else 'FAIL — something is structurally broken'}")

    use_style()
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    ax.plot(losses, color=ACCENT, lw=1.4)
    ax.axhline(expected, color=MUTED, ls="--", lw=1.0, label=f"ln(vocab) = {expected:.2f}")
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("loss (log scale)")
    ax.set_title("Overfit-one-batch: loss must collapse, or the model is broken", loc="left")
    ax.legend(fontsize=7)
    fig.tight_layout()
    save(fig, "07_overfit_one_batch.png")

    if not ok:
        sys.exit(1)


def train(config: Config, resume: bool = False, corpus: str = "dev-clean",
          out_dir: Path = CHECKPOINTS) -> None:
    device = get_device()
    tokenizer, train_ds, val_ds = setup(config, corpus=corpus)
    model = build_model(config.model)

    print(f"device      : {describe_device(device)}")
    print(f"parameters  : {model.num_parameters():,}")
    print(f"train       : {len(train_ds):,} utts ({train_ds.total_hours():.2f} h)")
    print(f"val         : {len(val_ds):,} utts ({val_ds.total_hours():.2f} h), unseen speakers")
    steps_per_epoch = len(train_ds) // config.train.batch_size
    print(f"schedule    : {config.train.max_updates:,} updates @ batch {config.train.batch_size} "
          f"= {config.train.max_updates/steps_per_epoch:.0f} epochs")
    print(f"lr          : {config.train.learning_rate} with {config.train.warmup_updates} warmup steps")
    print()

    trainer = Trainer(model, tokenizer, config, train_ds, val_ds, out_dir=out_dir)
    if resume:
        name = resume if isinstance(resume, str) else "last.pt"
        if (out_dir / name).exists():
            trainer.load(name)
            print(f"resumed from {name} at step {trainer.state.step}\n")
        else:
            print(f"no {name} to resume from; starting fresh\n")

    header = f"{'step':>7}{'train':>9}{'val':>9}{'lr':>10}{'|g|':>8}{'elapsed':>9}"
    print(header)
    print("-" * len(header))

    def on_log(r):
        val = f"{r['val_loss']:.4f}" if "val_loss" in r else ""
        star = " *" if r.get("best") else ""
        print(f"{r['step']:>7}{r['train_loss']:>9.4f}{val:>9}"
              f"{r['lr']:>10.2e}{r['grad_norm']:>8.2f}{r['elapsed']/60:>8.1f}m{star}")

    state = trainer.fit(log_every=50, eval_every=250, on_log=on_log)

    print(f"\nbest validation loss: {state.best_val_loss:.4f}")
    print(f"checkpoints in {out_dir}/")


def plot_history(out_dir: Path = CHECKPOINTS) -> None:
    path = out_dir / "history.json"
    if not path.exists():
        sys.exit(f"{path} not found — run training first")
    history = json.loads(path.read_text())

    use_style()
    steps = [h["step"] for h in history]
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.2))

    axes[0].plot(steps, [h["train_loss"] for h in history], color=ACCENT, lw=1.3, label="train")
    val = [(h["step"], h["val_loss"]) for h in history if "val_loss" in h]
    if val:
        axes[0].plot(*zip(*val), color=ACCENT2, lw=1.6, marker="o", ms=3, label="val (unseen speakers)")
        best = min(val, key=lambda x: x[1])
        axes[0].axvline(best[0], color=MUTED, ls=":", lw=1.0)
        axes[0].annotate(f"best {best[1]:.3f}", best, textcoords="offset points",
                         xytext=(6, 10), fontsize=7, color=ACCENT2)
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("cross-entropy")
    axes[0].set_title("Loss — the gap is overfitting on 3.7 hours", loc="left")
    axes[0].legend(fontsize=7)
    axes[0].grid(True, lw=0.5)

    axes[1].plot(steps, [h["lr"] for h in history], color=ACCENT, lw=1.4)
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("learning rate")
    axes[1].set_title("Warmup then linear decay to zero (paper §2.4)", loc="left")
    axes[1].grid(True, lw=0.5)

    axes[2].plot(steps, [h["grad_norm"] for h in history], color=ACCENT, lw=0.9)
    axes[2].axhline(1.0, color=ACCENT2, ls="--", lw=1.2, label="clip threshold")
    axes[2].set_xlabel("step")
    axes[2].set_ylabel("gradient norm")
    axes[2].set_yscale("log")
    axes[2].set_title("Gradient norm vs the clip at 1.0", loc="left")
    axes[2].legend(fontsize=7)
    axes[2].grid(True, lw=0.5)

    fig.suptitle("Training a 18M-parameter Whisper on 3.7 hours of LibriSpeech",
                 x=0.02, ha="left", weight="bold")
    fig.tight_layout()
    save(fig, "07_training_curves.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sanity", action="store_true", help="overfit one batch and exit")
    p.add_argument("--plot", action="store_true", help="plot from a finished run")
    p.add_argument("--resume", nargs="?", const=True, default=False,
                   help="resume from last.pt, or from a named checkpoint")
    p.add_argument("--corpus", default="dev-clean",
                   choices=["dev-clean", "train-clean-100"],
                   help="dev-clean = 3.7h baseline; train-clean-100 = the real run")
    p.add_argument("--out", default=None, help="checkpoint directory")
    p.add_argument("--window", type=float, default=None,
                   help="audio window in seconds (default 15; 17 covers all of "
                        "train-clean-100)")
    p.add_argument("--steps", type=int, default=None, help="override max_updates")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    args = p.parse_args()

    if not DEFAULT_ROOT.exists():
        sys.exit("Corpus not found. Run: uv run python scripts/04_data.py --download")

    base = Config()
    audio = AudioConfig(window_seconds=args.window) if args.window else base.audio
    # n_audio_ctx is derived from the window, so the model config must follow it.
    model = ModelConfig(n_audio_ctx=audio.n_audio_ctx) if args.window else base.model
    config = Config(
        audio=audio,
        model=model,
        train=TrainConfig(
            max_updates=args.steps or base.train.max_updates,
            batch_size=args.batch_size or base.train.batch_size,
            learning_rate=args.lr or base.train.learning_rate,
        ),
    )

    out_dir = Path(args.out) if args.out else CHECKPOINTS / (
        "run_100h" if args.corpus == "train-clean-100" else "run_3.7h"
    )

    if args.plot:
        plot_history(out_dir)
    elif args.sanity:
        sanity(config, corpus=args.corpus)
    else:
        train(config, resume=args.resume, corpus=args.corpus, out_dir=out_dir)
