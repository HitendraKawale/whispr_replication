"""Device selection.

On Apple Silicon we want MPS, but a few ops still fall back to CPU (and a few
are outright missing). Setting PYTORCH_ENABLE_MPS_FALLBACK=1 makes those fall
back silently rather than raising, which is what we want for training.
"""

from __future__ import annotations

import os

import torch


def get_device(prefer: str | None = None) -> torch.device:
    """Return the best available device, or `prefer` if given and usable."""
    if prefer is not None:
        return torch.device(prefer)
    if torch.backends.mps.is_available():
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def describe(device: torch.device) -> str:
    if device.type == "mps":
        return "Apple Silicon GPU (Metal Performance Shaders)"
    if device.type == "cuda":
        return torch.cuda.get_device_name(0)
    return "CPU"
