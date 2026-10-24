"""Small shared utilities for training and inference."""

from __future__ import annotations

import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int = 1024) -> None:
    """Seed Python / NumPy / PyTorch RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(preferred: Optional[str] = None) -> torch.device:
    """Return the best available device (or the explicitly requested one)."""
    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_parameters(model: torch.nn.Module) -> int:
    """Total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)