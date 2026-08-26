"""Figure and report generation from pipeline output.

Kept separate from :mod:`evissl.viz` (which owns the drawing primitives) and
from :mod:`evissl.pipelines` (which owns the experiments), so that adding a
figure to the report never means touching the code that produced the numbers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from evissl.eval import sparsification_for
from evissl.metrics.calibration import expected_calibration_error
from evissl.metrics.segmentation import LOWER_IS_BETTER
from evissl.pipelines import format_table, load_history
from evissl.utils.logging import get_logger
from evissl.viz import (
    plot_efficiency,
    plot_method_comparison,
    plot_qualitative,
    plot_reliability,
    plot_sparsification,
    plot_sweep,
    plot_training_curves,
    plot_uncertainty_separation,
    save,
)

logger = get_logger("evissl.report")


def write_comparison_figures(
    comparison: dict[str, Any],
    out_dir: str | Path = "results",
    qualitative_from: str = "evidential",
) -> list[Path]:
    """Write every figure derivable from a :func:`run_comparison` result.

    Args:
        comparison: The dict returned by
            :func:`evissl.pipelines.run_comparison`. Dense prediction maps are
            needed for the qualitative, sparsification and separation figures,
            so run the comparison with ``keep_predictions=True`` to get them.
        out_dir: Root; figures land in ``<out_dir>/figures``.
        qualitative_from: Run name whose predictions illustrate the qualitative
            and uncertainty-separation panels.

    Returns:
        Paths of the figures actually written.
    """
    figures_dir = Path(out_dir) / "figures"
    written: list[Path] = []
    results = comparison["results"]

    # -- training dynamics ------------------------------------------------- #
    histories = comparison.get("histories") or {}
    if any(histories.values()):
        written.append(
            save(
                plot_training_curves(histories),
                figures_dir / "training_curves.png",
            )
        )

    # -- point-and-interval comparison, one figure per metric -------------- #
    for metric, intervals in comparison.get("intervals", {}).items():
        written.append(
            save(
                plot_method_comparison(
                    intervals, metric, lower_is_better=metric in LOWER_IS_BETTER
                ),
                figures_dir / f"comparison_{metric}.png",
            )
        )

    # -- calibration -------------------------------------------------------- #
    curves: dict[str, dict[str, np.ndarray]] = {}
    eces: dict[str, float] = {}
    for name, result in results.items():
        predictions = result.predictions
        if predictions is None:
            continue
        probability = predictions.prob.ravel()
        target = predictions.target.ravel()
        predicted = (probability >= 0.5).astype(np.float64)
        confidence = np.where(predicted > 0.5, probability, 1.0 - probability)
        correct = (predicted == target).astype(np.float64)
        curve = expected_calibration_error(confidence, correct, bins=15)
        if "bin_confidence" in curve:
            curves[name] = curve
            eces[name] = curve["ece"]
    if curves:
        written.append(
            save(plot_reliability(curves, eces), figures_dir / "reliability.png")
        )

    # -- sparsification / AUSE --------------------------------------------- #
    sparsification: dict[str, dict[str, np.ndarray]] = {}
    auses: dict[str, float] = {}
    for name, result in results.items():
        if result.predictions is None:
            continue
        sparsification[name] = sparsification_for(result)
        auses[name] = result.calibration.get("ause", float("nan"))
    if sparsification:
        written.append(
            save(
                plot_sparsification(sparsification, auses),
                figures_dir / "sparsification.png",
            )
        )

    # -- qualitative panels and the uncertainty-separation diagnostic ------- #
    target_run = qualitative_from if qualitative_from in results else next(iter(results), None)
    if target_run is not None:
        result = results[target_run]
        predictions = result.predictions
        if predictions is not None:
            images = _images_for(comparison, target_run, len(predictions.prob))
            if images is not None:
                written.append(
                    save(
                        plot_qualitative(
                            images,
                            predictions.target,
                            predictions.prob,
                            predictions.vacuity,
                            predictions.dissonance,
                            n=4,
                            title=f"{target_run}: prediction and uncertainty",
                        ),
                        figures_dir / "qualitative.png",
                    )
                )
            if predictions.vacuity is not None and predictions.dissonance is not None:
                written.append(
                    save(
                        plot_uncertainty_separation(
                            predictions.vacuity, predictions.dissonance, predictions.prob
                        ),
                        figures_dir / "uncertainty_separation.png",
                    )
                )

    # -- accuracy against cost --------------------------------------------- #
    rows = [r.summary_row() for r in results.values()]
    if rows and all("params_m" in r for r in rows):
        written.append(save(plot_efficiency(rows), figures_dir / "efficiency.png"))

    for path in written:
        logger.info("wrote %s", path)
    return written


def _images_for(comparison: dict[str, Any], run_name: str, n: int) -> np.ndarray | None:
    """Recover the raw test images for a run, for the qualitative figure."""
    run = comparison.get("runs", {}).get(run_name)
    if run is None:
        return None
    cfg = run["config"]
    from evissl.data import build_dataset

    bundle = build_dataset(cfg.data, cfg.run.seed)
    if len(bundle.test) < n:  # pragma: no cover - shapes always match in practice
        return None
    return bundle.test.images[:n]


def write_sweep_figure(
    sweep: dict[str, Any], out_dir: str | Path = "results", metric: str = "dice"
) -> Path:
    """Write the labelled-fraction sweep figure."""
    figure = plot_sweep(
        sweep["sweep"],
        metric=metric,
        xlabel="labelled fraction of the training set",
        title="Semi-supervised gain against label budget",
    )
    path = save(figure, Path(out_dir) / "figures" / "labeled_fraction_sweep.png")
    logger.info("wrote %s", path)
    return path


def regenerate_figures(
    config_paths: list[str], out_dir: str | Path = "results"
) -> list[Path]:
    """Redraw the training-curve figure from JSONL histories already on disk.

    The figures that need dense prediction maps cannot be rebuilt this way -
    those require re-running ``compare --figures``, since the maps are far too
    large to commit.
    """
    from evissl.config import load_config

    histories = {}
    for path in config_paths:
        cfg = load_config(path)
        history = load_history(cfg.run.name, out_dir)
        if history:
            histories[cfg.run.name] = history

    if not histories:
        logger.warning("no run histories found under %s/runs", out_dir)
        return []

    figure = plot_training_curves(histories)
    return [save(figure, Path(out_dir) / "figures" / "training_curves.png")]


def write_markdown_report(
    comparison: dict[str, Any], out_path: str | Path = "results/RESULTS.md"
) -> Path:
    """Write a markdown summary of a comparison: tables plus the paired tests."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = [
        "# Results",
        "",
        "Generated by `evissl compare`. Arrows mark the direction of improvement.",
        "",
        "## Test-set summary",
        "",
        format_table(comparison["table"]),
        "",
    ]

    if comparison.get("comparisons"):
        lines += [
            "## Paired statistical tests",
            "",
            "Wilcoxon signed-rank on per-image values, Holm-Bonferroni corrected across",
            "the whole family of tests. `delta` is the mean paired difference with a 95%",
            "bootstrap interval; an interval excluding zero is the claim being made.",
            "",
            "| metric | method | delta vs baseline | 95% CI | p (Holm) | Cohen d | significant |",
            "|---|---|---|---|---|---|---|",
        ]
        for metric, items in comparison["comparisons"].items():
            for c in items:
                p = c.p_adjusted if c.p_adjusted is not None else c.p_value
                lines.append(
                    f"| {metric} | {c.name_a.split('.')[0]} | {c.difference.estimate:+.4f} | "
                    f"[{c.difference.lower:+.4f}, {c.difference.upper:+.4f}] | "
                    f"{p:.4g} | {c.effect_size:+.3f} | {'yes' if c.significant else 'no'} |"
                )
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("wrote %s", out_path)
    return out_path
