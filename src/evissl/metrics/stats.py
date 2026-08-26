"""Statistical comparison of segmentation runs.

Reporting that one method scored 0.842 and another 0.836 is not a result: with
100 test images and per-image standard deviations around 0.1, that gap is well
inside the noise. This module supplies the machinery needed to say whether a
difference is real.

Three tools, each for a specific job:

**Bootstrap confidence intervals.** Non-parametric, so no normality assumption -
which matters because per-image Dice is heavily left-skewed (a ceiling at 1.0 and
a long tail of failures).

**Paired tests.** The two methods are evaluated on the *same* images, so the
comparison must be paired. A paired bootstrap on the per-image differences, plus
the Wilcoxon signed-rank test as a rank-based check that does not assume the
differences are symmetric in magnitude.

**Multiplicity correction.** An ablation comparing many configurations against
one baseline runs many tests, and at alpha = 0.05 roughly one in twenty will
appear significant by chance. Holm-Bonferroni corrects for that while being
uniformly more powerful than plain Bonferroni.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import stats


@dataclass
class Interval:
    """A point estimate with a confidence interval."""

    estimate: float
    lower: float
    upper: float
    level: float = 0.95
    n: int = 0

    def __str__(self) -> str:
        return f"{self.estimate:.4f} [{self.lower:.4f}, {self.upper:.4f}]"

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass
class Comparison:
    """The outcome of comparing two methods on the same images."""

    name_a: str
    name_b: str
    mean_a: float
    mean_b: float
    #: ``mean_a - mean_b``, with a bootstrap interval on the difference.
    difference: Interval
    #: Wilcoxon signed-rank p-value on the paired differences.
    p_value: float
    #: Paired Cohen's d - the mean difference in units of its own SD.
    effect_size: float
    n: int
    #: Set by :func:`holm_bonferroni`; ``None`` until corrected.
    p_adjusted: float | None = None

    @property
    def significant(self) -> bool:
        """True when the corrected (or raw) p-value clears 0.05."""
        p = self.p_value if self.p_adjusted is None else self.p_adjusted
        return bool(np.isfinite(p) and p < 0.05)

    def summary(self) -> str:
        p = self.p_value if self.p_adjusted is None else self.p_adjusted
        marker = "*" if self.significant else " "
        return (
            f"{self.name_a} vs {self.name_b}: "
            f"{self.mean_a:.4f} vs {self.mean_b:.4f}, "
            f"delta={self.difference}, p={p:.4g}{marker}, d={self.effect_size:.3f}"
        )


def bootstrap_ci(
    values: np.ndarray,
    n_resamples: int = 2000,
    level: float = 0.95,
    seed: int = 0,
    statistic: str = "mean",
) -> Interval:
    """Percentile bootstrap interval for a summary of ``values``.

    Args:
        values: Per-image metric values. ``NaN`` entries are dropped.
        n_resamples: Bootstrap replicates.
        level: Coverage, e.g. 0.95.
        seed: RNG seed, so intervals are reproducible.
        statistic: ``"mean"`` or ``"median"``.

    Returns:
        An :class:`Interval`. Degenerate but valid when fewer than two finite
        values are available (the interval collapses to the point estimate).
    """
    data = np.asarray(values, dtype=np.float64)
    data = data[np.isfinite(data)]
    reduce = np.mean if statistic == "mean" else np.median

    if data.size == 0:
        return Interval(float("nan"), float("nan"), float("nan"), level, 0)
    point = float(reduce(data))
    if data.size < 2:
        return Interval(point, point, point, level, int(data.size))

    rng = np.random.default_rng(seed)
    # One vectorised draw of all replicates; far faster than a Python loop and
    # the memory cost is trivial at these sizes.
    picks = rng.integers(0, data.size, size=(n_resamples, data.size))
    replicates = reduce(data[picks], axis=1)

    alpha = (1.0 - level) / 2.0
    lower, upper = np.quantile(replicates, [alpha, 1.0 - alpha])
    return Interval(point, float(lower), float(upper), level, int(data.size))


def paired_bootstrap_difference(
    a: np.ndarray,
    b: np.ndarray,
    n_resamples: int = 2000,
    level: float = 0.95,
    seed: int = 0,
) -> Interval:
    """Bootstrap interval for the mean paired difference ``a - b``.

    Resamples *image indices*, not the two arrays independently, which is what
    preserves the pairing and gives the tighter interval that the paired design
    earns.
    """
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"Paired arrays must match in shape: {x.shape} vs {y.shape}")

    valid = np.isfinite(x) & np.isfinite(y)
    diff = (x - y)[valid]
    return bootstrap_ci(diff, n_resamples, level, seed, statistic="mean")


def compare(
    a: np.ndarray,
    b: np.ndarray,
    name_a: str = "a",
    name_b: str = "b",
    n_resamples: int = 2000,
    seed: int = 0,
) -> Comparison:
    """Full paired comparison of two methods on the same images.

    Args:
        a: Per-image metric for method A.
        b: Per-image metric for method B, same images in the same order.
        name_a: Label for A.
        name_b: Label for B.
        n_resamples: Bootstrap replicates for the difference interval.
        seed: RNG seed.

    Returns:
        A :class:`Comparison`. The p-value is ``NaN`` when every paired
        difference is exactly zero, where the signed-rank test is undefined.
    """
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    diff = x - y

    if diff.size == 0 or np.allclose(diff, 0.0):
        p_value = float("nan")
    else:
        # zero_method="wilcox" drops exact ties, the standard convention.
        p_value = float(stats.wilcoxon(x, y, zero_method="wilcox").pvalue)

    sd = float(diff.std(ddof=1)) if diff.size > 1 else 0.0
    effect = float(diff.mean() / sd) if sd > 0 else 0.0

    return Comparison(
        name_a=name_a,
        name_b=name_b,
        mean_a=float(x.mean()) if x.size else float("nan"),
        mean_b=float(y.mean()) if y.size else float("nan"),
        difference=paired_bootstrap_difference(x, y, n_resamples, seed=seed),
        p_value=p_value,
        effect_size=effect,
        n=int(diff.size),
    )


def holm_bonferroni(comparisons: list[Comparison], alpha: float = 0.05) -> list[Comparison]:
    """Apply the Holm-Bonferroni step-down correction in place.

    Sorts the raw p-values ascending and multiplies the ``i``-th by ``m - i``,
    then enforces monotonicity so an adjusted p-value never decreases. Sets
    ``p_adjusted`` on each comparison and returns the same list.

    ``NaN`` p-values (undefined tests) are left uncorrected and excluded from
    the family size, since they carry no evidence either way.
    """
    testable = [c for c in comparisons if np.isfinite(c.p_value)]
    m = len(testable)
    if m == 0:
        return comparisons

    order = sorted(range(m), key=lambda i: testable[i].p_value)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adjusted = min(1.0, (m - rank) * testable[idx].p_value)
        running_max = max(running_max, adjusted)
        testable[idx].p_adjusted = running_max

    del alpha  # significance is read from p_adjusted via Comparison.significant
    return comparisons


def summarize_metric(
    per_image: dict[str, np.ndarray],
    metric: str,
    baseline: str,
    n_resamples: int = 2000,
    seed: int = 0,
) -> tuple[dict[str, Interval], list[Comparison]]:
    """Intervals for every method and paired comparisons against a baseline.

    Args:
        per_image: Maps method name to its per-image metric array.
        metric: Metric name, used only for labelling.
        baseline: Key in ``per_image`` to compare everything against.
        n_resamples: Bootstrap replicates.
        seed: RNG seed.

    Returns:
        ``(intervals, comparisons)`` with Holm-Bonferroni already applied.

    Raises:
        KeyError: if ``baseline`` is not present.
    """
    if baseline not in per_image:
        raise KeyError(f"Baseline {baseline!r} not among methods {sorted(per_image)}")

    intervals = {
        name: bootstrap_ci(values, n_resamples, seed=seed)
        for name, values in per_image.items()
    }
    comparisons = [
        compare(values, per_image[baseline], f"{name}.{metric}", f"{baseline}.{metric}",
                n_resamples, seed)
        for name, values in per_image.items()
        if name != baseline
    ]
    return intervals, holm_bonferroni(comparisons)
