"""Segmentation metrics: overlap, boundary distance and boundary agreement.

Overlap alone (Dice, IoU) is not enough to characterise a lesion segmenter.
Dice is dominated by the interior, which is large and easy, so two models can
tie on Dice while differing visibly at the contour - and the contour is what a
dermatologist actually reads. This module therefore reports three families:

* **Overlap** - Dice, IoU, sensitivity, specificity, precision, accuracy.
* **Boundary distance** - HD95 and ASSD, in pixels, from symmetric surface
  distances.
* **Boundary agreement** - the BF score at a pixel tolerance, which unlike
  Hausdorff is bounded and does not hinge on a single worst point.

Every metric is computed **per image** and aggregated afterwards. Pooling all
pixels first would let large lesions dominate, so a model could improve its
headline number by getting better at the easy big cases while regressing on the
small ones that matter clinically.

Empty-mask convention
---------------------
When both prediction and ground truth are empty the segmentation is perfect, so
Dice and IoU are 1.0 and the distances are 0.0. When exactly one is empty the
overlap metrics are 0.0 but the surface distances are genuinely **undefined** -
there is no surface to measure to - so they are returned as ``NaN`` and excluded
from means by :func:`aggregate`. Substituting the image diagonal, as some
implementations do, silently rewards a model for failing completely on hard
images instead of reporting that it failed.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt

#: Metrics where a lower value is better; used for sorting and reporting.
LOWER_IS_BETTER: frozenset[str] = frozenset(
    {"hd95", "assd", "ece", "ace", "mce", "brier", "nll", "ause"}
)

#: The metric set produced by :func:`compute_all`.
METRIC_NAMES: tuple[str, ...] = (
    "dice",
    "iou",
    "sensitivity",
    "specificity",
    "precision",
    "accuracy",
    "hd95",
    "assd",
    "boundary_f1",
)


def _as_bool(mask: np.ndarray) -> np.ndarray:
    return np.asarray(mask).astype(bool)


# --------------------------------------------------------------------------- #
# Overlap
# --------------------------------------------------------------------------- #


def confusion(pred: np.ndarray, target: np.ndarray) -> tuple[int, int, int, int]:
    """Return ``(tp, fp, fn, tn)`` pixel counts for one image."""
    p, t = _as_bool(pred), _as_bool(target)
    tp = int(np.count_nonzero(p & t))
    fp = int(np.count_nonzero(p & ~t))
    fn = int(np.count_nonzero(~p & t))
    tn = int(np.count_nonzero(~p & ~t))
    return tp, fp, fn, tn


def dice_score(pred: np.ndarray, target: np.ndarray) -> float:
    """Dice similarity coefficient. 1.0 when both masks are empty."""
    tp, fp, fn, _ = confusion(pred, target)
    denominator = 2 * tp + fp + fn
    return 1.0 if denominator == 0 else 2.0 * tp / denominator


def iou_score(pred: np.ndarray, target: np.ndarray) -> float:
    """Intersection over union (Jaccard). 1.0 when both masks are empty."""
    tp, fp, fn, _ = confusion(pred, target)
    denominator = tp + fp + fn
    return 1.0 if denominator == 0 else tp / denominator


# --------------------------------------------------------------------------- #
# Boundary distance
# --------------------------------------------------------------------------- #


def surface_mask(mask: np.ndarray) -> np.ndarray:
    """One-pixel-wide inner boundary of a binary mask.

    Uses a 4-connected erosion so diagonal steps are not treated as adjacency,
    which keeps the extracted contour a single pixel thick.
    """
    m = _as_bool(mask)
    if not m.any():
        return np.zeros_like(m)
    structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    return m & ~binary_erosion(m, structure=structure, border_value=0)


def surface_distances(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Symmetric surface-to-surface distances in pixels.

    Returns:
        Concatenation of the distance from every predicted-boundary pixel to
        the nearest ground-truth-boundary pixel and vice versa. Empty array if
        either surface is empty (see the module docstring).
    """
    pred_surface = surface_mask(pred)
    target_surface = surface_mask(target)
    if not pred_surface.any() or not target_surface.any():
        return np.empty(0, dtype=np.float64)

    # distance_transform_edt measures distance to the nearest zero, so the
    # surface must be inverted before transforming.
    dt_to_target = distance_transform_edt(~target_surface)
    dt_to_pred = distance_transform_edt(~pred_surface)
    return np.concatenate([dt_to_target[pred_surface], dt_to_pred[target_surface]])


def hd95(pred: np.ndarray, target: np.ndarray) -> float:
    """95th-percentile symmetric Hausdorff distance, in pixels.

    The 95th percentile rather than the maximum: a single stray pixel would
    otherwise dominate the true Hausdorff distance and make it unusable as a
    comparison statistic.
    """
    p, t = _as_bool(pred), _as_bool(target)
    if not p.any() and not t.any():
        return 0.0
    distances = surface_distances(p, t)
    return float("nan") if distances.size == 0 else float(np.percentile(distances, 95))


def assd(pred: np.ndarray, target: np.ndarray) -> float:
    """Average symmetric surface distance, in pixels."""
    p, t = _as_bool(pred), _as_bool(target)
    if not p.any() and not t.any():
        return 0.0
    distances = surface_distances(p, t)
    return float("nan") if distances.size == 0 else float(distances.mean())


def boundary_f1(pred: np.ndarray, target: np.ndarray, tolerance: float = 2.0) -> float:
    """BF score: F1 between boundaries matched within ``tolerance`` pixels.

    Args:
        pred: Predicted binary mask.
        target: Ground-truth binary mask.
        tolerance: Match radius in pixels. 2 px at 128x128 is roughly the
            inter-annotator agreement band reported for dermoscopic contours.

    Returns:
        F1 in ``[0, 1]``. 1.0 if both masks are empty, 0.0 if exactly one is.
    """
    p, t = _as_bool(pred), _as_bool(target)
    if not p.any() and not t.any():
        return 1.0
    pred_surface, target_surface = surface_mask(p), surface_mask(t)
    if not pred_surface.any() or not target_surface.any():
        return 0.0

    dt_to_target = distance_transform_edt(~target_surface)
    dt_to_pred = distance_transform_edt(~pred_surface)
    precision = float((dt_to_target[pred_surface] <= tolerance).mean())
    recall = float((dt_to_pred[target_surface] <= tolerance).mean())
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #


def compute_all(
    pred: np.ndarray,
    target: np.ndarray,
    boundary_tolerance: float = 2.0,
    boundary_metrics: bool = True,
) -> dict[str, float]:
    """Every metric for one image pair.

    Args:
        pred: ``(H, W)`` predicted binary mask.
        target: ``(H, W)`` ground-truth binary mask.
        boundary_tolerance: Passed to :func:`boundary_f1`.
        boundary_metrics: Compute HD95, ASSD and BF1. These need four Euclidean
            distance transforms per image and dominate the cost of scoring, so
            the training loop disables them for its per-epoch validation and
            they are computed only for the reported test results.

    Returns:
        Dict keyed by :data:`METRIC_NAMES`, or by the overlap subset when
        ``boundary_metrics`` is False.
    """
    tp, fp, fn, tn = confusion(pred, target)
    total = tp + fp + fn + tn

    def ratio(numerator: int, denominator: int) -> float:
        """Undefined ratios return NaN so :func:`aggregate` excludes them.

        The alternative - substituting 1.0 - would report perfect precision for
        a model that predicts nothing at all, and perfect sensitivity on every
        lesion-free image. Both are silent rewards for degenerate behaviour.
        """
        return float("nan") if denominator == 0 else numerator / denominator

    metrics = {
        "dice": dice_score(pred, target),
        "iou": iou_score(pred, target),
        "sensitivity": ratio(tp, tp + fn),
        "specificity": ratio(tn, tn + fp),
        "precision": ratio(tp, tp + fp),
        "accuracy": ratio(tp + tn, total),
    }
    if boundary_metrics:
        metrics["hd95"] = hd95(pred, target)
        metrics["assd"] = assd(pred, target)
        metrics["boundary_f1"] = boundary_f1(pred, target, boundary_tolerance)
    return metrics


def compute_batch(
    preds: np.ndarray,
    targets: np.ndarray,
    boundary_tolerance: float = 2.0,
    boundary_metrics: bool = True,
) -> list[dict[str, float]]:
    """Per-image metrics for a batch of masks, shape ``(N, H, W)``."""
    if preds.shape != targets.shape:
        raise ValueError(f"Shape mismatch: preds {preds.shape} vs targets {targets.shape}")
    return [
        compute_all(p, t, boundary_tolerance, boundary_metrics)
        for p, t in zip(preds, targets, strict=True)
    ]


def aggregate(per_image: list[dict[str, float]]) -> dict[str, float]:
    """Mean and standard deviation of each metric, ignoring ``NaN``.

    Returns:
        Dict with ``<name>`` (mean), ``<name>_std`` and ``<name>_n`` (the number
        of images that contributed) for every metric. Reporting ``_n`` keeps the
        undefined-distance exclusions visible rather than hidden.
    """
    if not per_image:
        return {}
    out: dict[str, float] = {}
    for key in per_image[0]:
        values = np.array([row[key] for row in per_image], dtype=np.float64)
        finite = values[np.isfinite(values)]
        out[key] = float(finite.mean()) if finite.size else float("nan")
        out[f"{key}_std"] = float(finite.std(ddof=1)) if finite.size > 1 else 0.0
        out[f"{key}_n"] = int(finite.size)
    return out
