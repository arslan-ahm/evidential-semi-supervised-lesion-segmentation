"""Calibration and uncertainty-quality metrics.

A segmenter deployed in a clinical workflow is only useful if its confidence
means something: a model that reports 0.9 on pixels it gets right 60% of the
time cannot be used to route ambiguous cases to a human. Dice says nothing about
this, so it is measured separately here.

Two distinct questions are answered:

**Is the confidence numerically honest?** Expected calibration error (ECE) and
its equal-mass variant (ACE), maximum calibration error, the Brier score and the
negative log-likelihood. ECE is reported in the standard equal-width binning and
in equal-mass binning, because equal-width ECE is known to understate error when
predictions pile up near 0 and 1 - which for a segmenter they overwhelmingly do,
since most pixels are confidently background.

**Does the uncertainty rank the errors?** The sparsification error curve and its
summary, AUSE. Progressively discard the pixels the model is least sure about
and track the error on what remains; a useful uncertainty makes that curve fall
as fast as an oracle that discards the actually-wrong pixels. AUSE is the gap to
that oracle, so 0 is perfect and it is invariant to the uncertainty's scale -
which is what makes vacuity, dissonance, softmax entropy and MC-dropout spread
directly comparable despite living on different scales.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-12


def _flatten(*arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    return tuple(np.asarray(a).ravel() for a in arrays)


# --------------------------------------------------------------------------- #
# Calibration error
# --------------------------------------------------------------------------- #


def expected_calibration_error(
    confidence: np.ndarray, correct: np.ndarray, bins: int = 15
) -> dict[str, float]:
    """Equal-width ECE, MCE and the per-bin reliability curve.

    Args:
        confidence: Predicted probability of the predicted class, in ``[0, 1]``.
        correct: 1 where that prediction was right, else 0.
        bins: Number of equal-width bins over ``[0, 1]``.

    Returns:
        Dict with ``ece``, ``mce`` and the arrays ``bin_confidence``,
        ``bin_accuracy``, ``bin_count`` (empty bins removed).
    """
    conf, corr = _flatten(confidence, correct)
    if conf.size == 0:
        return {"ece": float("nan"), "mce": float("nan")}

    edges = np.linspace(0.0, 1.0, bins + 1)
    # `right=True` with a -inf-safe first bin so confidence exactly 0 is binned.
    index = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, bins - 1)

    bin_conf, bin_acc, bin_count = [], [], []
    ece = 0.0
    mce = 0.0
    total = conf.size
    for b in range(bins):
        selected = index == b
        n = int(selected.sum())
        if n == 0:
            continue
        mean_conf = float(conf[selected].mean())
        mean_acc = float(corr[selected].mean())
        gap = abs(mean_conf - mean_acc)
        ece += (n / total) * gap
        mce = max(mce, gap)
        bin_conf.append(mean_conf)
        bin_acc.append(mean_acc)
        bin_count.append(n)

    return {
        "ece": float(ece),
        "mce": float(mce),
        "bin_confidence": np.array(bin_conf),
        "bin_accuracy": np.array(bin_acc),
        "bin_count": np.array(bin_count),
    }


def adaptive_calibration_error(
    confidence: np.ndarray, correct: np.ndarray, bins: int = 15
) -> float:
    """Equal-**mass** calibration error (ACE).

    Each bin holds the same number of samples, so no bin's contribution is
    estimated from a handful of points. For segmentation this matters: with
    equal-width bins almost every pixel lands in the top bin and the estimator
    reduces to a single average, hiding miscalibration in the sparse middle.
    """
    conf, corr = _flatten(confidence, correct)
    if conf.size == 0:
        return float("nan")
    bins = max(1, min(bins, conf.size))

    order = np.argsort(conf)
    chunks = np.array_split(order, bins)
    total = conf.size
    ace = 0.0
    for chunk in chunks:
        if chunk.size == 0:
            continue
        gap = abs(float(conf[chunk].mean()) - float(corr[chunk].mean()))
        ace += (chunk.size / total) * gap
    return float(ace)


def brier_score(prob: np.ndarray, target: np.ndarray) -> float:
    """Mean squared error between the positive-class probability and the label."""
    p, t = _flatten(prob, target)
    return float(np.mean((p - t) ** 2)) if p.size else float("nan")


def negative_log_likelihood(prob: np.ndarray, target: np.ndarray) -> float:
    """Mean binary NLL of the positive-class probability."""
    p, t = _flatten(prob, target)
    if p.size == 0:
        return float("nan")
    p = np.clip(p, EPS, 1.0 - EPS)
    return float(-np.mean(t * np.log(p) + (1.0 - t) * np.log(1.0 - p)))


# --------------------------------------------------------------------------- #
# Uncertainty quality
# --------------------------------------------------------------------------- #


def sparsification_curve(
    uncertainty: np.ndarray,
    error: np.ndarray,
    steps: int = 20,
) -> dict[str, np.ndarray]:
    """Error remaining after discarding the most uncertain pixels.

    Args:
        uncertainty: Per-pixel uncertainty; any monotone scale works.
        error: Per-pixel error, typically ``1 - correct`` or absolute residual.
        steps: Number of removal fractions sampled over ``[0, 1)``.

    Returns:
        Dict with ``fraction`` (removed), ``model`` (error after removing the
        model's most-uncertain pixels), ``oracle`` (after removing the truly
        worst pixels) and ``random`` (the no-information reference).
    """
    unc, err = _flatten(uncertainty, error)
    n = unc.size
    if n == 0:
        empty = np.empty(0)
        return {"fraction": empty, "model": empty, "oracle": empty, "random": empty}

    fractions = np.linspace(0.0, 0.95, steps)
    by_model = np.argsort(-unc)  # most uncertain first
    by_oracle = np.argsort(-err)  # actually worst first
    baseline = float(err.mean())

    model_curve, oracle_curve = [], []
    for fraction in fractions:
        keep = n - int(round(fraction * n))
        keep = max(keep, 1)
        model_curve.append(float(err[by_model[-keep:]].mean()))
        oracle_curve.append(float(err[by_oracle[-keep:]].mean()))

    return {
        "fraction": fractions,
        "model": np.array(model_curve),
        "oracle": np.array(oracle_curve),
        # Removing pixels at random leaves the error unchanged in expectation.
        "random": np.full(fractions.shape, baseline),
    }


def ause(uncertainty: np.ndarray, error: np.ndarray, steps: int = 20) -> float:
    """Area under the sparsification **error** curve (model minus oracle).

    Zero means the uncertainty ranks errors as well as an oracle. Normalised by
    the area between the random and oracle curves, so the value is comparable
    across datasets and error definitions, and bounded at 1.0 for an
    uninformative uncertainty.
    """
    curve = sparsification_curve(uncertainty, error, steps)
    if curve["fraction"].size == 0:
        return float("nan")
    gap = np.trapezoid(curve["model"] - curve["oracle"], curve["fraction"])
    span = np.trapezoid(curve["random"] - curve["oracle"], curve["fraction"])
    if abs(span) < EPS:
        return 0.0
    return float(np.clip(gap / span, 0.0, 1.0))


def uncertainty_error_auroc(uncertainty: np.ndarray, error: np.ndarray) -> float:
    """AUROC for using uncertainty to detect erroneous pixels.

    Computed via the rank-sum (Mann-Whitney U) identity rather than by sweeping
    thresholds, which is exact and avoids a dependency on a curve resolution.

    Returns:
        AUROC in ``[0, 1]``; 0.5 is uninformative. ``NaN`` if the image is all
        correct or all wrong, where the quantity is undefined.
    """
    unc, err = _flatten(uncertainty, error)
    positive = err > 0.5
    n_pos = int(positive.sum())
    n_neg = int(unc.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    # Average ranks so ties are handled correctly.
    order = np.argsort(unc, kind="stable")
    ranks = np.empty(unc.size, dtype=np.float64)
    ranks[order] = np.arange(1, unc.size + 1, dtype=np.float64)
    sorted_unc = unc[order]
    start = 0
    for i in range(1, sorted_unc.size + 1):
        if i == sorted_unc.size or sorted_unc[i] != sorted_unc[start]:
            if i - start > 1:
                ranks[order[start:i]] = ranks[order[start:i]].mean()
            start = i

    rank_sum = float(ranks[positive].sum())
    u = rank_sum - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


# --------------------------------------------------------------------------- #
# Bundle
# --------------------------------------------------------------------------- #


def compute_calibration(
    lesion_prob: np.ndarray,
    target: np.ndarray,
    uncertainty: np.ndarray | None = None,
    bins: int = 15,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Full calibration and uncertainty-quality report for a set of pixels.

    Args:
        lesion_prob: Predicted probability of the lesion class, any shape.
        target: Binary ground truth, same shape.
        uncertainty: Optional per-pixel uncertainty for AUSE and AUROC. When
            omitted, ``1 - max(p, 1 - p)`` is used, i.e. softmax uncertainty.
        bins: Bin count for ECE and ACE.
        threshold: Decision threshold defining correctness.

    Returns:
        Flat dict of scalar metrics.
    """
    prob, tgt = _flatten(lesion_prob, target)
    predicted = (prob >= threshold).astype(np.float64)
    correct = (predicted == tgt).astype(np.float64)
    # Confidence in the *predicted* class, which is what calibration is about.
    confidence = np.where(predicted > 0.5, prob, 1.0 - prob)

    unc = confidence * 0.0 + (1.0 - confidence) if uncertainty is None else _flatten(uncertainty)[0]

    ece = expected_calibration_error(confidence, correct, bins)
    return {
        "ece": ece["ece"],
        "mce": ece["mce"],
        "ace": adaptive_calibration_error(confidence, correct, bins),
        "brier": brier_score(prob, tgt),
        "nll": negative_log_likelihood(prob, tgt),
        "ause": ause(unc, 1.0 - correct),
        "unc_error_auroc": uncertainty_error_auroc(unc, 1.0 - correct),
        "mean_confidence": float(confidence.mean()),
        "pixel_accuracy": float(correct.mean()),
    }
