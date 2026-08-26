"""Determinism helpers.

Reproducibility is a claim this repository makes in its README, so seeding is
centralised here and applied to Python, NumPy and PyTorch (CPU and CUDA) in one
call. ``seed_everything`` returns a ``torch.Generator`` so data loaders can be
seeded independently of the global RNG state.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 1337, deterministic: bool = True) -> torch.Generator:
    """Seed every RNG this project touches.

    Args:
        seed: Base seed.
        deterministic: If True, ask cuDNN for deterministic kernels and disable
            the autotuner. Slower, but required for bit-exact repeats.

    Returns:
        A CPU ``torch.Generator`` seeded from ``seed``, for use as the
        ``generator=`` argument of a ``DataLoader``.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True

    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def worker_init_fn(worker_id: int) -> None:
    """Give every data-loader worker a distinct, reproducible RNG stream."""
    base = torch.initial_seed() % 2**32
    seed = (base + worker_id) % 2**32
    np.random.seed(seed)
    random.seed(seed)


def resolve_device(spec: str = "auto") -> torch.device:
    """Turn a config device string into a concrete ``torch.device``.

    Args:
        spec: ``"auto"``, ``"cpu"``, ``"cuda"`` or ``"cuda:N"``.

    Returns:
        ``cuda`` when requested and available, otherwise ``cpu``.
    """
    spec = (spec or "auto").strip().lower()
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if spec.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(spec)
