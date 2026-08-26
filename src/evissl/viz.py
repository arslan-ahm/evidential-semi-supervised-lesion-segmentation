"""Publication-quality figures.

Every figure this repository reports is produced by a function here, so a
result in the README can be regenerated from a checkpoint rather than being a
screenshot nobody can reproduce. Style choices are centralised in
:func:`use_style`: a single serif-free typeface, no chartjunk, and colours from
a colour-blind-safe qualitative set with the baseline always grey and the
proposed method always the accent - so a reader can identify the method under
test without consulting the legend.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import matplotlib

# Scripts run headless and must not require a display, so they get Agg. A
# notebook, however, already has an inline backend configured, and switching it
# here would make every plt.show() in notebooks/ silently draw nothing.
if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (must follow the backend choice)
import numpy as np  # noqa: E402

#: Okabe-Ito derived palette: distinguishable in the three common forms of
#: colour-vision deficiency and in greyscale print.
COLOURS: dict[str, str] = {
    "supervised": "#7f7f7f",
    "mean_teacher": "#0072B2",
    "fixmatch": "#E69F00",
    "evidential": "#009E73",
    "unet": "#333333",
    "oracle": "#CC79A7",
    "random": "#BBBBBB",
    "accent": "#009E73",
}

#: Colormaps used consistently across every uncertainty figure.
UNCERTAINTY_CMAP = "magma"
PROB_CMAP = "viridis"


def use_style() -> None:
    """Apply the shared matplotlib style. Idempotent."""
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "legend.frameon": False,
            "lines.linewidth": 1.8,
            "image.interpolation": "nearest",
        }
    )


def colour_for(name: str) -> str:
    """Colour for a method name, matching on substring so run names work."""
    key = name.lower()
    for candidate, colour in COLOURS.items():
        if candidate in key:
            return colour
    return COLOURS["accent"]


def save(fig: plt.Figure, path: str | Path) -> Path:
    """Write a figure to ``path`` (creating parents) and close it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #


def plot_samples(
    images: np.ndarray,
    masks: np.ndarray,
    difficulty: np.ndarray | None = None,
    n: int = 8,
    title: str = "Synthetic dermoscopy samples",
) -> plt.Figure:
    """Grid of images with their masks overlaid as a contour.

    Args:
        images: ``(N, H, W, 3)`` uint8.
        masks: ``(N, H, W)`` binary.
        difficulty: Optional per-sample difficulty, shown in each title.
        n: Number of samples to draw.
        title: Suptitle.
    """
    use_style()
    n = min(n, len(images))
    cols = min(n, 4)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.4 * cols, 2.6 * rows), squeeze=False)

    for i in range(rows * cols):
        ax = axes[i // cols][i % cols]
        ax.axis("off")
        if i >= n:
            continue
        ax.imshow(images[i])
        # Contour rather than a translucent fill: the boundary is what matters
        # and a fill hides the very texture the reader needs to judge difficulty.
        ax.contour(masks[i], levels=[0.5], colors=[COLOURS["accent"]], linewidths=1.4)
        label = f"area {masks[i].mean():.2f}"
        if difficulty is not None:
            label += f" | difficulty {difficulty[i]:.2f}"
        ax.set_title(label, fontsize=8)

    fig.suptitle(title, y=1.0)
    return fig


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #


def plot_training_curves(
    histories: dict[str, list[dict[str, Any]]],
    metrics: tuple[str, ...] = ("train_loss", "val_dice", "train_mask_rate"),
    title: str = "Training dynamics",
) -> plt.Figure:
    """One panel per metric, one line per run.

    Args:
        histories: Maps run name to its list of per-epoch records.
        metrics: Record keys to plot. Missing keys are skipped silently, since
            supervised runs have no ``mask_rate``.
        title: Suptitle.
    """
    use_style()
    present = [
        m for m in metrics if any(m in row for rows in histories.values() for row in rows)
    ]
    if not present:
        present = ["train_loss"]

    fig, axes = plt.subplots(1, len(present), figsize=(4.2 * len(present), 3.2), squeeze=False)
    for ax, metric in zip(axes[0], present, strict=True):
        for name, rows in histories.items():
            xs = [r["epoch"] for r in rows if metric in r]
            ys = [r[metric] for r in rows if metric in r]
            if xs:
                ax.plot(xs, ys, label=name, color=colour_for(name))
        ax.set_xlabel("epoch")
        ax.set_ylabel(metric.replace("_", " "))
        ax.set_title(metric.replace("_", " "))
    axes[0][0].legend(loc="best", fontsize=8)
    fig.suptitle(title, y=1.02)
    return fig


# --------------------------------------------------------------------------- #
# Qualitative
# --------------------------------------------------------------------------- #


def plot_qualitative(
    images: np.ndarray,
    targets: np.ndarray,
    probs: np.ndarray,
    vacuity: np.ndarray | None = None,
    dissonance: np.ndarray | None = None,
    n: int = 4,
    threshold: float = 0.5,
    title: str = "Prediction and uncertainty decomposition",
) -> plt.Figure:
    """Per-row panel: image, prediction, error, vacuity, dissonance.

    The error column is the point of the figure: it lets a reader check by eye
    whether the uncertainty maps light up where the model is actually wrong,
    which is the qualitative counterpart of the AUSE number.
    """
    use_style()
    n = min(n, len(images))
    columns = ["image + GT", "P(lesion)", "error"]
    if vacuity is not None:
        columns.append("vacuity (epistemic)")
    if dissonance is not None:
        columns.append("dissonance (aleatoric)")

    fig, axes = plt.subplots(n, len(columns), figsize=(2.5 * len(columns), 2.6 * n), squeeze=False)
    for row in range(n):
        pred = (probs[row] >= threshold).astype(np.uint8)
        error = (pred != targets[row]).astype(float)
        panels: list[tuple[np.ndarray, str | None, tuple[float, float] | None]] = [
            (images[row], None, None),
            (probs[row], PROB_CMAP, (0.0, 1.0)),
            (error, "Reds", (0.0, 1.0)),
        ]
        if vacuity is not None:
            panels.append((vacuity[row], UNCERTAINTY_CMAP, (0.0, 1.0)))
        if dissonance is not None:
            panels.append((dissonance[row], UNCERTAINTY_CMAP, (0.0, 1.0)))

        for col, (data, cmap, limits) in enumerate(panels):
            ax = axes[row][col]
            ax.axis("off")
            if cmap is None:
                ax.imshow(data)
            else:
                ax.imshow(data, cmap=cmap, vmin=limits[0], vmax=limits[1])
            if col == 0:
                ax.contour(targets[row], levels=[0.5], colors=[COLOURS["accent"]], linewidths=1.2)
            if row == 0:
                ax.set_title(columns[col], fontsize=9)

    fig.suptitle(title, y=1.0)
    return fig


def plot_uncertainty_separation(
    vacuity: np.ndarray,
    dissonance: np.ndarray,
    prob: np.ndarray,
    max_points: int = 20000,
    seed: int = 0,
    title: str = "Vacuity and dissonance in a trained model",
) -> plt.Figure:
    """Diagnostic: do the two uncertainty types actually separate in practice?

    The left panel plots vacuity against dissonance for pixels whose predicted
    probability sits near 0.5. *Analytically* that slice can contain both
    high-vacuity points (no evidence) and high-dissonance points (conflicting
    evidence) - two situations a single confidence number reports identically,
    which is the premise of the method.

    Whether a **trained** model occupies both regions is a separate, empirical
    question, and this figure is how it gets answered rather than assumed. If
    the near-0.5 pixels form a single tight cluster, the two components are
    collinear in that model and the decomposition has collapsed to one degree of
    freedom - see the discussion in ``docs/RESULTS.md``. The title is
    deliberately descriptive rather than asserting the separation.

    The right panel shows how each component varies with the probability itself.
    """
    use_style()
    rng = np.random.default_rng(seed)
    u, d, p = vacuity.ravel(), dissonance.ravel(), prob.ravel()
    if u.size > max_points:
        pick = rng.choice(u.size, max_points, replace=False)
        u, d, p = u[pick], d[pick], p[pick]

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))

    ambiguous = np.abs(p - 0.5) < 0.10
    ax = axes[0]
    ax.scatter(u[~ambiguous], d[~ambiguous], s=2, alpha=0.10, c=COLOURS["random"],
               label="confident pixels")
    ax.scatter(u[ambiguous], d[ambiguous], s=4, alpha=0.45, c=COLOURS["accent"],
               label="P(lesion) within 0.1 of 0.5")
    ax.set_xlabel("vacuity  (epistemic: no evidence)")
    ax.set_ylabel("dissonance  (aleatoric: conflicting evidence)")
    # A tight cluster here means the two components are collinear in this model;
    # a spread along both axes means the decomposition is doing real work. The
    # axis limits are fixed to [0, 1] so the two cases are visually comparable
    # across runs rather than being auto-scaled into looking the same.
    ax.set_title("Where the near-ambiguous pixels actually sit")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=8, markerscale=3)

    ax = axes[1]
    bins = np.linspace(0.0, 1.0, 21)
    index = np.clip(np.digitize(p, bins[1:-1]), 0, len(bins) - 2)
    centres = 0.5 * (bins[:-1] + bins[1:])
    for values, label, colour in (
        (u, "vacuity", COLOURS["fixmatch"]),
        (d, "dissonance", COLOURS["mean_teacher"]),
    ):
        means = np.array(
            [values[index == b].mean() if np.any(index == b) else np.nan
             for b in range(len(centres))]
        )
        ax.plot(centres, means, marker="o", markersize=3, label=label, color=colour)
    ax.set_xlabel("predicted P(lesion)")
    ax.set_ylabel("mean uncertainty")
    ax.set_title("Components against predicted probability")
    ax.legend(fontsize=8)

    fig.suptitle(title, y=1.03)
    return fig


# --------------------------------------------------------------------------- #
# Calibration and uncertainty quality
# --------------------------------------------------------------------------- #


def plot_reliability(
    curves: dict[str, dict[str, np.ndarray]],
    eces: dict[str, float] | None = None,
    title: str = "Reliability",
) -> plt.Figure:
    """Reliability diagram: accuracy against confidence, per run."""
    use_style()
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot([0, 1], [0, 1], ls="--", lw=1.2, c="black", label="perfect calibration")

    for name, curve in curves.items():
        label = name
        if eces and name in eces:
            label = f"{name} (ECE {eces[name]:.3f})"
        ax.plot(
            curve["bin_confidence"],
            curve["bin_accuracy"],
            marker="o",
            markersize=4,
            label=label,
            color=colour_for(name),
        )

    ax.set_xlabel("mean predicted confidence")
    ax.set_ylabel("observed accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="upper left")
    return fig


def plot_sparsification(
    curves: dict[str, dict[str, np.ndarray]],
    auses: dict[str, float] | None = None,
    title: str = "Sparsification: does uncertainty rank the errors?",
) -> plt.Figure:
    """Sparsification curves with the oracle and random references."""
    use_style()
    fig, ax = plt.subplots(figsize=(5.2, 4.0))

    first = next(iter(curves.values()), None)
    if first is not None and first["fraction"].size:
        ax.plot(first["fraction"], first["oracle"], ls="--", lw=1.3,
                c=COLOURS["oracle"], label="oracle (perfect ranking)")
        ax.plot(first["fraction"], first["random"], ls=":", lw=1.3,
                c=COLOURS["random"], label="random (no information)")

    for name, curve in curves.items():
        label = name
        if auses and name in auses:
            label = f"{name} (AUSE {auses[name]:.3f})"
        ax.plot(curve["fraction"], curve["model"], label=label, color=colour_for(name))

    ax.set_xlabel("fraction of most-uncertain pixels removed")
    ax.set_ylabel("error rate on remaining pixels")
    ax.set_title(title)
    ax.legend(fontsize=8)
    return fig


# --------------------------------------------------------------------------- #
# Comparison
# --------------------------------------------------------------------------- #


def plot_method_comparison(
    intervals: dict[str, Any],
    metric: str = "dice",
    title: str | None = None,
    lower_is_better: bool = False,
) -> plt.Figure:
    """Horizontal point-and-interval plot, one row per method.

    Confidence intervals rather than bare bars: a bar chart invites the reader
    to compare heights that are not significantly different, which is exactly
    the error the statistics in this repository exist to prevent.
    """
    use_style()
    names = list(intervals)
    fig, ax = plt.subplots(figsize=(6.0, 0.55 * len(names) + 1.6))

    for y, name in enumerate(names):
        interval = intervals[name]
        estimate = getattr(interval, "estimate", interval)
        lower = getattr(interval, "lower", estimate)
        upper = getattr(interval, "upper", estimate)
        ax.errorbar(
            estimate,
            y,
            xerr=[[estimate - lower], [upper - estimate]],
            fmt="o",
            markersize=6,
            capsize=4,
            color=colour_for(name),
        )
        ax.text(upper, y, f"  {estimate:.4f}", va="center", fontsize=8)

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    direction = "lower is better" if lower_is_better else "higher is better"
    ax.set_xlabel(f"{metric}  ({direction})  with 95% bootstrap CI")
    ax.set_title(title or f"Method comparison: {metric}")
    ax.grid(axis="y", visible=False)
    return fig


def plot_sweep(
    sweep: dict[str, dict[float, Any]],
    metric: str = "dice",
    xlabel: str = "labelled fraction",
    title: str | None = None,
) -> plt.Figure:
    """Metric against a swept quantity, one line per method, with CI bands.

    Args:
        sweep: ``{method: {x_value: Interval or float}}``.
        metric: Metric name for the y-axis.
        xlabel: Label for the swept axis.
        title: Optional title.
    """
    use_style()
    fig, ax = plt.subplots(figsize=(5.6, 4.0))

    for name, points in sweep.items():
        xs = sorted(points)
        estimates = np.array([getattr(points[x], "estimate", points[x]) for x in xs])
        ax.plot(xs, estimates, marker="o", markersize=4, label=name, color=colour_for(name))
        lowers = np.array([getattr(points[x], "lower", np.nan) for x in xs])
        uppers = np.array([getattr(points[x], "upper", np.nan) for x in xs])
        if np.all(np.isfinite(lowers)):
            ax.fill_between(xs, lowers, uppers, alpha=0.15, color=colour_for(name), lw=0)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(metric)
    ax.set_title(title or f"{metric} against {xlabel}")
    ax.legend(fontsize=8)
    return fig


def plot_efficiency(
    rows: list[dict[str, Any]],
    x: str = "params_m",
    y: str = "dice",
    size: str = "gmacs",
    title: str = "Accuracy against model size",
) -> plt.Figure:
    """Accuracy-versus-cost scatter, marker area proportional to ``size``.

    The figure that makes the efficiency argument: a point up and to the left
    dominates. A log x-axis is used because the models span two orders of
    magnitude in parameter count.
    """
    use_style()
    fig, ax = plt.subplots(figsize=(5.4, 4.0))

    sizes = np.array([float(r.get(size, 1.0)) for r in rows])
    scale = 900.0 / max(sizes.max(), 1e-6)
    for row, marker_size in zip(rows, sizes * scale, strict=True):
        name = str(row.get("run", "?"))
        ax.scatter(
            row[x], row[y], s=max(float(marker_size), 30.0), alpha=0.75,
            color=colour_for(name), edgecolors="white", linewidths=0.8,
        )
        ax.annotate(
            name, (row[x], row[y]), textcoords="offset points", xytext=(7, 5), fontsize=8
        )

    ax.set_xscale("log")
    ax.set_xlabel(f"{x.replace('_', ' ')}  (log scale)")
    ax.set_ylabel(y)
    ax.set_title(f"{title}   (marker area proportional to {size})")
    return fig
