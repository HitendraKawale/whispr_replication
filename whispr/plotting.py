"""Shared matplotlib setup so every figure in the repo looks like a sibling."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # scripts run headless
import matplotlib.pyplot as plt  # noqa: E402

FIGURES = Path(__file__).resolve().parent.parent / "figures"

INK = "#1b1b1f"
ACCENT = "#c1440e"  # a warm rust, readable on both light and dark
ACCENT2 = "#0e6ba8"
MUTED = "#9a9aa2"


def use_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "text.color": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "font.size": 9,
            "figure.dpi": 130,
            "savefig.bbox": "tight",
            "grid.color": "#e6e6ea",
            "legend.frameon": False,
        }
    )


def save(fig, name: str) -> Path:
    """Save into figures/ and report the path."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / name
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path.relative_to(FIGURES.parent)}")
    return path
