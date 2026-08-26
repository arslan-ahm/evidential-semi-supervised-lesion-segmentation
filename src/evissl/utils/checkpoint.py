"""Checkpoint I/O.

A checkpoint carries the student weights, the EMA teacher weights, the epoch,
the monitored metric and the full config dict, so any saved model can be
re-evaluated without the original command line.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    *,
    ema_model: nn.Module | None = None,
    epoch: int = 0,
    metrics: dict[str, float] | None = None,
    config: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write a checkpoint atomically (temp file then replace)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model": model.state_dict(),
        "ema_model": ema_model.state_dict() if ema_model is not None else None,
        "epoch": epoch,
        "metrics": metrics or {},
        "config": config or {},
    }
    if extra:
        payload.update(extra)

    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)
    return path


def load_checkpoint(
    path: str | Path,
    model: nn.Module | None = None,
    *,
    prefer_ema: bool = True,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    """Load a checkpoint and optionally populate ``model``.

    Args:
        path: Checkpoint file.
        model: If given, its weights are replaced in place.
        prefer_ema: Load the EMA teacher when the checkpoint contains one.
        map_location: Passed through to ``torch.load``.
        strict: Forwarded to ``load_state_dict``.

    Returns:
        The raw checkpoint payload, with an added ``"loaded"`` key naming which
        weight set was applied (``"ema_model"``, ``"model"`` or ``None``).
    """
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    loaded: str | None = None
    if model is not None:
        key = "ema_model" if (prefer_ema and payload.get("ema_model")) else "model"
        model.load_state_dict(payload[key], strict=strict)
        loaded = key
    payload["loaded"] = loaded
    return payload
