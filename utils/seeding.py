"""Reproducibility helpers."""
from __future__ import annotations

import os
import random

import numpy as np


def make_rng(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(seed)


def seed_everything(seed: int) -> None:
    """Seed python, numpy and (if installed) torch."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:  # pragma: no cover
        pass
