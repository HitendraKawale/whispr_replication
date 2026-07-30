"""Training loop.

Optimiser settings follow paper Table 17 exactly — AdamW with betas (0.9, 0.98),
eps 1e-6, weight decay 0.1, gradient norm clipping at 1.0, and a linear decay to
zero after a warmup. Only the *scale* differs, and every scale deviation is in
`whispr/config.py` with its justification.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from whispr.config import Config
from whispr.data import collate
from whispr.device import get_device
from whispr.model import Whispr, build_model
from whispr.tokenizer import WhisprTokenizer


def lr_at(step: int, cfg) -> float:
    """Linear warmup, then linear decay to zero (paper §2.4).

    Warmup exists because Adam's second-moment estimate is unreliable in the
    first few hundred steps; a full learning rate then produces enormous,
    badly-scaled updates that can permanently damage the model. Ramping up
    lets the optimiser's statistics stabilise first.
    """
    if step < cfg.warmup_updates:
        return cfg.learning_rate * step / max(1, cfg.warmup_updates)
    progress = (step - cfg.warmup_updates) / max(1, cfg.max_updates - cfg.warmup_updates)
    return cfg.learning_rate * max(0.0, 1.0 - progress)


def build_optimizer(model: Whispr, cfg) -> torch.optim.AdamW:
    """AdamW, with weight decay applied only where it belongs.

    Decaying biases, LayerNorm gains and (especially) embeddings pulls them
    toward zero for no good reason — they are not the parameters that overfit.
    This split is standard practice and not stated in the paper, but decaying
    everything measurably hurts.
    """
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() < 2 or "embedding" in name:
            no_decay.append(param)
        else:
            decay.append(param)

    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg.learning_rate,
        betas=cfg.betas,
        eps=cfg.eps,
    )


def compute_loss(model: Whispr, batch: dict, device: torch.device) -> torch.Tensor:
    mel = batch["mel"].to(device, non_blocking=True)
    tokens = batch["tokens"].to(device, non_blocking=True)
    labels = batch["labels"].to(device, non_blocking=True)

    logits = model(mel, tokens)
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100
    )


@dataclass
class TrainState:
    step: int = 0
    best_val_loss: float = float("inf")
    history: list[dict] = field(default_factory=list)


class Trainer:
    def __init__(
        self,
        model: Whispr,
        tokenizer: WhisprTokenizer,
        config: Config,
        train_dataset,
        val_dataset=None,
        out_dir: str | Path = "checkpoints",
        device: torch.device | None = None,
    ) -> None:
        self.config = config
        self.tokenizer = tokenizer
        self.device = device or get_device()
        self.model = model.to(self.device)
        self.optimizer = build_optimizer(self.model, config.train)
        self.state = TrainState()
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        collate_fn = self._make_collate()
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.train.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=config.train.num_workers,
            drop_last=True,
        )
        self.val_loader = (
            DataLoader(
                val_dataset,
                batch_size=config.train.batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=config.train.num_workers,
            )
            if val_dataset is not None
            else None
        )

    def _make_collate(self):
        pad = self.tokenizer.special.pad
        # Exclude the constant task-specification tokens from the loss.
        prefix = self.tokenizer.prompt_length - 1
        return lambda b: collate(b, pad_token=pad, mask_prefix=prefix)

    # ------------------------------------------------------------------ steps

    def train_step(self, batch: dict) -> dict:
        self.model.train()
        lr = lr_at(self.state.step, self.config.train)
        for group in self.optimizer.param_groups:
            group["lr"] = lr

        loss = compute_loss(self.model, batch, self.device)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.train.max_grad_norm
        )

        self.optimizer.step()
        self.state.step += 1
        return {"loss": loss.item(), "lr": lr, "grad_norm": float(grad_norm)}

    @torch.no_grad()
    def evaluate(self, max_batches: int | None = None) -> float:
        if self.val_loader is None:
            return float("nan")
        self.model.eval()
        total, count = 0.0, 0
        for i, batch in enumerate(self.val_loader):
            if max_batches is not None and i >= max_batches:
                break
            total += compute_loss(self.model, batch, self.device).item()
            count += 1
        return total / max(count, 1)

    # --------------------------------------------------------------- the loop

    def fit(
        self,
        log_every: int = 50,
        eval_every: int = 250,
        eval_batches: int | None = 40,
        on_log=None,
    ) -> TrainState:
        """Train.

        `eval_batches` caps the periodic validation pass. A full sweep of
        dev-clean is 324 batches, and at a 250-step eval interval that made
        validation roughly *half* of total wall time — the loading and
        frontend cost of 2,590 FLAC files dominates. 40 batches (320
        utterances) estimates the loss to within a few hundredths, which is
        ample for choosing a checkpoint. Use `evaluate(None)` for the real
        number at the end.
        """
        cfg = self.config.train
        torch.manual_seed(cfg.seed)

        start = time.perf_counter()
        running = []
        loader = iter(self.train_loader)

        while self.state.step < cfg.max_updates:
            try:
                batch = next(loader)
            except StopIteration:  # next epoch
                loader = iter(self.train_loader)
                batch = next(loader)

            metrics = self.train_step(batch)
            running.append(metrics["loss"])

            if self.state.step % log_every == 0:
                elapsed = time.perf_counter() - start
                record = {
                    "step": self.state.step,
                    "train_loss": sum(running) / len(running),
                    "lr": metrics["lr"],
                    "grad_norm": metrics["grad_norm"],
                    "elapsed": elapsed,
                }
                running = []

                if self.state.step % eval_every == 0:
                    record["val_loss"] = self.evaluate(max_batches=eval_batches)
                    if record["val_loss"] < self.state.best_val_loss:
                        self.state.best_val_loss = record["val_loss"]
                        self.save("best.pt")
                        record["best"] = True

                self.state.history.append(record)
                # Written every log, not just at the end: a run interrupted at
                # hour three should not lose its curve.
                self._write_history()
                if on_log:
                    on_log(record)

        self.save("last.pt")
        self._write_history()
        return self.state

    # --------------------------------------------------------- checkpointing

    def save(self, name: str) -> Path:
        path = self.out_dir / name
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "step": self.state.step,
                "best_val_loss": self.state.best_val_loss,
                "config": asdict(self.config),
            },
            path,
        )
        return path

    def load(self, name: str) -> None:
        ckpt = torch.load(self.out_dir / name, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.state.step = ckpt["step"]
        self.state.best_val_loss = ckpt["best_val_loss"]

    def _write_history(self) -> None:
        (self.out_dir / "history.json").write_text(json.dumps(self.state.history, indent=1))


def overfit_one_batch(
    model: Whispr,
    tokenizer: WhisprTokenizer,
    config: Config,
    dataset,
    steps: int = 200,
    device: torch.device | None = None,
) -> list[float]:
    """The single most valuable sanity check in deep learning.

    Take one batch and train on it repeatedly. A correct model *must* be able to
    memorise a handful of examples — loss should collapse toward zero. If it
    plateaus, something is structurally broken (a detached gradient, a wrong
    mask, a label misalignment) and no amount of hyperparameter tuning will
    help. Running this before the real training run saves hours.
    """
    device = device or get_device()
    model = model.to(device).train()

    pad = tokenizer.special.pad
    prefix = tokenizer.prompt_length - 1
    batch = collate([dataset[i] for i in range(config.train.batch_size)],
                    pad_token=pad, mask_prefix=prefix)

    optimizer = build_optimizer(model, config.train)
    for group in optimizer.param_groups:
        group["lr"] = config.train.learning_rate

    losses = []
    for _ in range(steps):
        loss = compute_loss(model, batch, device)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
        optimizer.step()
        losses.append(loss.item())
    return losses


def expected_initial_loss(n_vocab: int) -> float:
    """A model that knows nothing predicts uniformly, giving loss = ln(vocab).

    Comparing the first measured loss against this catches a large family of
    bugs — a wrong vocabulary size, a broken initialisation, an inverted mask —
    before wasting an hour of training on them.
    """
    return math.log(n_vocab)
