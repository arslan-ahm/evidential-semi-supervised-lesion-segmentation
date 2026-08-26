"""Evaluation: predictions, metrics, calibration and per-image records.

Separated from :mod:`evissl.engine.trainer` so that a saved checkpoint can be
re-scored without re-training, and so the *same* evaluation code serves the
training loop, the ablation script and the notebooks. Per-image records are kept
rather than only their means, because the statistical comparisons in
:mod:`evissl.metrics.stats` are paired and need the individual values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from evissl.config import Config
from evissl.metrics.calibration import compute_calibration, sparsification_curve
from evissl.metrics.segmentation import aggregate, compute_batch
from evissl.uncertainty import (
    dirichlet_alpha,
    dissonance,
    mc_dropout_predict,
    predictive_entropy,
    probabilities,
    tta_predict,
    vacuity,
)


@dataclass
class Predictions:
    """Dense outputs for one split, all as NumPy arrays of shape ``(N, H, W)``.

    Attributes:
        prob: Predicted lesion probability.
        target: Ground-truth binary mask.
        vacuity: Epistemic uncertainty (evidential head only, else ``None``).
        dissonance: Aleatoric uncertainty (evidential head only, else ``None``).
        entropy: Normalised predictive entropy - always available, and the only
            uncertainty a softmax model can offer.
        epistemic: MC-dropout epistemic component, when requested.
    """

    prob: np.ndarray
    target: np.ndarray
    vacuity: np.ndarray | None = None
    dissonance: np.ndarray | None = None
    entropy: np.ndarray | None = None
    epistemic: np.ndarray | None = None

    def binarize(self, threshold: float = 0.5) -> np.ndarray:
        """Threshold the probability map into a ``uint8`` mask."""
        return (self.prob >= threshold).astype(np.uint8)

    def primary_uncertainty(self) -> np.ndarray:
        """The uncertainty map used for AUSE and the sparsification analysis.

        Prefers the evidential total (vacuity + dissonance) when available,
        since that is the quantity the method actually produces; falls back to
        MC-dropout epistemic, then to predictive entropy. The fallback order
        means a softmax baseline is still scored on *its* best available
        uncertainty rather than being handicapped.
        """
        if self.vacuity is not None and self.dissonance is not None:
            return np.clip(self.vacuity + self.dissonance, 0.0, 1.0)
        if self.epistemic is not None:
            return self.epistemic
        if self.entropy is not None:
            return self.entropy
        # Last resort: distance from a hard decision.
        return 1.0 - np.abs(self.prob - 0.5) * 2.0


@dataclass
class EvaluationResult:
    """Everything produced by one evaluation pass."""

    name: str
    #: Mean and std of every segmentation metric (see ``metrics.aggregate``).
    summary: dict[str, float]
    #: Per-image metric dicts, in loader order.
    per_image: list[dict[str, float]]
    #: Calibration and uncertainty-quality scalars.
    calibration: dict[str, float]
    predictions: Predictions | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def metric_array(self, metric: str) -> np.ndarray:
        """Per-image values of one metric, for the paired statistical tests."""
        return np.array([row[metric] for row in self.per_image], dtype=np.float64)

    def to_frame(self) -> pd.DataFrame:
        """One row per image, with the run name attached."""
        frame = pd.DataFrame(self.per_image)
        frame.insert(0, "run", self.name)
        frame.insert(1, "image_index", np.arange(len(frame)))
        return frame

    def summary_row(self) -> dict[str, Any]:
        """A single flat row combining every scalar, for the results table."""
        return {"run": self.name, **self.summary, **self.calibration, **self.extra}

    def save(self, out_dir: str | Path) -> dict[str, Path]:
        """Write ``per_image.csv`` and ``summary.json`` under ``out_dir``."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        per_image_path = out_dir / "per_image.csv"
        summary_path = out_dir / "summary.json"
        self.to_frame().to_csv(per_image_path, index=False)
        pd.Series(self.summary_row()).to_json(summary_path, indent=2)
        return {"per_image": per_image_path, "summary": summary_path}


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    cfg: Config,
    device: torch.device | str = "cpu",
) -> Predictions:
    """Run the model over a loader and collect dense outputs.

    Honours ``cfg.eval.tta`` and ``cfg.eval.mc_dropout``. Note that TTA and MC
    dropout are mutually exclusive here: combining them multiplies the forward
    count without a corresponding gain, and it would make the reported
    inference cost ambiguous.

    ``vacuity`` and ``dissonance`` are populated **only** in the single-pass
    mode. Averaging probabilities across flips or dropout samples discards the
    Dirichlet parameterisation those quantities are defined on, so reporting
    them alongside an averaged prediction would be meaningless.
    :meth:`Predictions.primary_uncertainty` falls back to the MC-dropout
    epistemic term or to predictive entropy in those modes.

    Args:
        model: Network to evaluate (student or teacher).
        loader: Loader yielding ``{"image", "mask"}`` batches.
        cfg: Run configuration.
        device: Compute device.

    Returns:
        A :class:`Predictions` bundle.
    """
    device = torch.device(device)
    head = cfg.loss.supervised.strip().lower()
    model = model.to(device)
    was_training = model.training
    model.eval()

    chunks: dict[str, list[np.ndarray]] = {
        "prob": [],
        "target": [],
        "vacuity": [],
        "dissonance": [],
        "entropy": [],
        "epistemic": [],
    }

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        chunks["target"].append(batch["mask"].numpy().astype(np.uint8))

        if cfg.eval.mc_dropout > 0:
            sampled = mc_dropout_predict(model, images, cfg.eval.mc_dropout, head)
            prob_full = sampled["prob"]
            chunks["epistemic"].append(sampled["epistemic"].cpu().numpy())
            chunks["entropy"].append(sampled["total"].cpu().numpy())
        elif cfg.eval.tta:
            prob_full = tta_predict(model, images, head)
            chunks["entropy"].append(predictive_entropy(prob_full).cpu().numpy())
        else:
            logits = model(images)
            prob_full = probabilities(logits, head)
            chunks["entropy"].append(predictive_entropy(prob_full).cpu().numpy())
            if head == "evidential":
                alpha = dirichlet_alpha(logits)
                chunks["vacuity"].append(vacuity(alpha).cpu().numpy())
                chunks["dissonance"].append(dissonance(alpha).cpu().numpy())

        chunks["prob"].append(prob_full[:, 1].cpu().numpy())

    model.train(was_training)

    def stack(key: str) -> np.ndarray | None:
        return np.concatenate(chunks[key], axis=0) if chunks[key] else None

    return Predictions(
        prob=stack("prob"),
        target=stack("target"),
        vacuity=stack("vacuity"),
        dissonance=stack("dissonance"),
        entropy=stack("entropy"),
        epistemic=stack("epistemic"),
    )


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    cfg: Config,
    name: str = "run",
    device: torch.device | str = "cpu",
    keep_predictions: bool = False,
    extra: dict[str, Any] | None = None,
) -> EvaluationResult:
    """Full evaluation: segmentation metrics, calibration, uncertainty quality.

    Args:
        model: Network to evaluate.
        loader: Evaluation loader.
        cfg: Run configuration.
        name: Label carried into the results table.
        device: Compute device.
        keep_predictions: Retain the dense maps on the result. Useful for
            figures; costs ``N * H * W * 4`` bytes per map, so off by default.
        extra: Additional scalars to merge into the summary row (parameter
            count, latency, and so on).

    Returns:
        An :class:`EvaluationResult`.
    """
    predictions = predict(model, loader, cfg, device)
    pred_masks = predictions.binarize(cfg.eval.threshold)
    per_image = compute_batch(pred_masks, predictions.target)

    calibration = compute_calibration(
        predictions.prob,
        predictions.target,
        uncertainty=predictions.primary_uncertainty(),
        bins=cfg.eval.calibration_bins,
        threshold=cfg.eval.threshold,
    )

    return EvaluationResult(
        name=name,
        summary=aggregate(per_image),
        per_image=per_image,
        calibration=calibration,
        predictions=predictions if keep_predictions else None,
        extra=dict(extra or {}),
    )


def sparsification_for(result: EvaluationResult, steps: int = 20) -> dict[str, np.ndarray]:
    """Sparsification curves for a completed evaluation.

    Raises:
        ValueError: if the result was produced with ``keep_predictions=False``.
    """
    if result.predictions is None:
        raise ValueError(
            "sparsification_for needs dense maps; call evaluate(..., keep_predictions=True)"
        )
    predictions = result.predictions
    correct = (predictions.binarize() == predictions.target).astype(np.float64)
    return sparsification_curve(predictions.primary_uncertainty(), 1.0 - correct, steps)
