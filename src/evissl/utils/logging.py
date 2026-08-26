"""Console and JSONL logging.

Two sinks: a human-readable stream for the terminal, and an append-only JSONL
file per run so training curves can be replotted later without re-training.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

_CONFIGURED: set[str] = set()


def get_logger(name: str = "evissl", level: int = logging.INFO) -> logging.Logger:
    """Return a console logger, configured exactly once per name."""
    logger = logging.getLogger(name)
    if name not in _CONFIGURED:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-7s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(handler)
        logger.propagate = False
        _CONFIGURED.add(name)
    logger.setLevel(level)
    return logger


class JsonlLogger:
    """Append-only JSONL sink for per-epoch metrics.

    Example:
        >>> log = JsonlLogger("results/run/history.jsonl")   # doctest: +SKIP
        >>> log.log(epoch=1, dice=0.81, loss=0.42)           # doctest: +SKIP
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        #: Truncate any previous history so a run's file describes only that run.
        self.path.write_text("", encoding="utf-8")
        self._rows: list[dict[str, Any]] = []

    def log(self, **record: Any) -> None:
        """Append one record, coercing tensors/NumPy scalars to plain floats."""
        clean = {k: _jsonable(v) for k, v in record.items()}
        self._rows.append(clean)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(clean) + "\n")

    @property
    def rows(self) -> list[dict[str, Any]]:
        """Every record logged so far, in order."""
        return list(self._rows)

    def history(self, key: str) -> list[Any]:
        """All values recorded under ``key``, in order, skipping gaps."""
        return [r[key] for r in self._rows if key in r]


def _jsonable(value: Any) -> Any:
    if hasattr(value, "item") and getattr(value, "ndim", 0) == 0:
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    return value
