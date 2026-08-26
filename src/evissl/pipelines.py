"""End-to-end pipelines: train, compare, ablate, benchmark.

These are the operations the scripts and notebooks call. Keeping them here
rather than in ``scripts/`` means the notebooks run exactly the code that
produced the committed tables, and the whole flow is importable from a test.

Every pipeline writes its artefacts under ``results/`` and returns them in
memory too, so a notebook can plot without re-reading from disk.
"""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from evissl.config import Config, load_config
from evissl.data import build_dataset, build_loaders
from evissl.engine import Trainer
from evissl.eval import EvaluationResult, evaluate
from evissl.metrics.segmentation import LOWER_IS_BETTER
from evissl.metrics.stats import (
    Comparison,
    bootstrap_ci,
    compare,
    holm_bonferroni,
    summarize_metric,
)
from evissl.models import build_model
from evissl.utils.complexity import estimate_macs, measure_latency
from evissl.utils.logging import get_logger
from evissl.utils.seed import resolve_device, seed_everything

logger = get_logger("evissl.pipeline")

#: Columns promoted to the front of every results table, in reading order.
HEADLINE_COLUMNS: tuple[str, ...] = (
    "run",
    "dice",
    "iou",
    "hd95",
    "assd",
    "boundary_f1",
    "ece",
    "ause",
    "params_m",
    "gmacs",
)


# --------------------------------------------------------------------------- #
# Single run
# --------------------------------------------------------------------------- #


def run_training(
    cfg: Config,
    keep_predictions: bool = False,
    evaluate_test: bool = True,
) -> dict[str, Any]:
    """Train one configuration and evaluate it on the test split.

    Args:
        cfg: Fully resolved configuration.
        keep_predictions: Retain dense prediction maps on the result, for figures.
        evaluate_test: Evaluate the test split after training. Disable for
            hyper-parameter work so the test set stays untouched.

    Returns:
        Dict with keys ``config``, ``state``, ``result`` (an
        :class:`~evissl.eval.EvaluationResult` or ``None``), ``complexity``,
        ``latency``, ``dataset`` and ``wall_seconds``.
    """
    started = time.perf_counter()
    generator = seed_everything(cfg.run.seed, cfg.run.deterministic)
    device = resolve_device(cfg.run.device)

    bundle = build_dataset(cfg.data, cfg.run.seed)
    loaders = build_loaders(cfg, bundle, generator=generator)
    model = build_model(cfg.model)

    complexity = estimate_macs(
        model, (cfg.model.in_channels, cfg.data.image_size, cfg.data.image_size), device
    )
    logger.info(
        "%s | %s | %.3fM params | %.3f GMACs | %s",
        cfg.run.name,
        cfg.model.name,
        complexity["params_m"],
        complexity["gmacs"],
        bundle.describe(),
    )

    trainer = Trainer(cfg, model, loaders, device)
    state = trainer.fit()

    result: EvaluationResult | None = None
    if evaluate_test:
        network = trainer.ema.module if cfg.eval.use_ema else trainer.model
        latency = measure_latency(
            network,
            (cfg.model.in_channels, cfg.data.image_size, cfg.data.image_size),
            device=device,
        )
        result = evaluate(
            network,
            loaders.test,
            cfg,
            name=cfg.run.name,
            device=device,
            keep_predictions=keep_predictions,
            extra={
                "params_m": round(complexity["params_m"], 4),
                "gmacs": round(complexity["gmacs"], 4),
                "latency_ms": round(latency["median_ms"], 3),
                "method": cfg.semi.method,
                "head": cfg.loss.supervised,
                "model": cfg.model.name,
                "labeled_fraction": round(bundle.meta["labeled_fraction_effective"], 4),
                "n_labeled": int(len(bundle.labeled_idx)),
                "best_val_dice": round(state.best_metric, 4),
                "best_epoch": state.best_epoch,
            },
        )
        result.save(Path(cfg.run.out_dir) / "runs" / cfg.run.name)
    else:
        latency = {}

    return {
        "config": cfg,
        "state": state,
        "result": result,
        "complexity": complexity,
        "latency": latency,
        "dataset": bundle.describe(),
        "wall_seconds": time.perf_counter() - started,
    }


# --------------------------------------------------------------------------- #
# Method comparison
# --------------------------------------------------------------------------- #


def run_comparison(
    config_paths: list[str | Path],
    overrides: list[str] | None = None,
    baseline: str = "supervised_baseline",
    metrics: tuple[str, ...] = ("dice", "iou", "hd95", "boundary_f1"),
    out_dir: str | Path = "results",
    keep_predictions: bool = False,
) -> dict[str, Any]:
    """Train several configurations and compare them with paired statistics.

    Args:
        config_paths: YAML files, one per method.
        overrides: ``section.key=value`` strings applied to every config, so a
            sweep can change one setting across all methods at once.
        baseline: Run name every other method is compared against.
        metrics: Metrics to run the paired tests on.
        out_dir: Root for tables and per-run artefacts.
        keep_predictions: Retain dense maps (needed for the qualitative and
            sparsification figures).

    Returns:
        Dict with ``runs``, ``table`` (a ``DataFrame``), ``intervals``,
        ``comparisons`` and ``histories``.
    """
    out_dir = Path(out_dir)
    runs: dict[str, dict[str, Any]] = {}

    for path in config_paths:
        cfg = load_config(path, overrides)
        logger.info("=" * 72)
        logger.info("training %s from %s", cfg.run.name, path)
        runs[cfg.run.name] = run_training(cfg, keep_predictions=keep_predictions)

    results = {name: run["result"] for name, run in runs.items() if run["result"] is not None}
    table = build_table(results, out_dir / "tables" / "method_comparison.csv")

    intervals: dict[str, dict[str, Any]] = {}
    comparisons: dict[str, list[Comparison]] = {}
    if baseline in results:
        for metric in metrics:
            per_image = {name: r.metric_array(metric) for name, r in results.items()}
            metric_intervals, metric_comparisons = summarize_metric(
                per_image, metric, baseline,
                n_resamples=next(iter(runs.values()))["config"].eval.bootstrap,
            )
            intervals[metric] = metric_intervals
            comparisons[metric] = metric_comparisons
        write_statistics(comparisons, out_dir / "tables" / "statistical_tests.csv")
    else:
        logger.warning("baseline %r not among runs %s; skipping paired tests",
                       baseline, sorted(results))

    histories = {name: run["state"].history for name, run in runs.items()}
    return {
        "runs": runs,
        "results": results,
        "table": table,
        "intervals": intervals,
        "comparisons": comparisons,
        "histories": histories,
    }


def build_table(
    results: dict[str, EvaluationResult], path: str | Path | None = None
) -> pd.DataFrame:
    """Assemble the summary table, headline columns first."""
    rows = [result.summary_row() for result in results.values()]
    frame = pd.DataFrame(rows)
    ordered = [c for c in HEADLINE_COLUMNS if c in frame.columns]
    frame = frame[ordered + [c for c in frame.columns if c not in ordered]]
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
        logger.info("wrote %s", path)
    return frame


def write_statistics(
    comparisons: dict[str, list[Comparison]], path: str | Path
) -> pd.DataFrame:
    """Flatten paired comparisons into a CSV with corrected p-values."""
    rows = []
    for metric, items in comparisons.items():
        for c in items:
            rows.append(
                {
                    "metric": metric,
                    "method": c.name_a.split(".")[0],
                    "baseline": c.name_b.split(".")[0],
                    "mean_method": round(c.mean_a, 5),
                    "mean_baseline": round(c.mean_b, 5),
                    "difference": round(c.difference.estimate, 5),
                    "ci_lower": round(c.difference.lower, 5),
                    "ci_upper": round(c.difference.upper, 5),
                    "p_wilcoxon": c.p_value,
                    "p_holm": c.p_adjusted,
                    "cohens_d": round(c.effect_size, 4),
                    "significant": c.significant,
                    "n": c.n,
                }
            )
    frame = pd.DataFrame(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    logger.info("wrote %s", path)
    return frame


# --------------------------------------------------------------------------- #
# Ablations
# --------------------------------------------------------------------------- #


def run_labeled_fraction_sweep(
    config_paths: list[str | Path],
    fractions: tuple[float, ...] = (0.05, 0.10, 0.20, 0.50),
    overrides: list[str] | None = None,
    out_dir: str | Path = "results",
    metric: str = "dice",
) -> dict[str, Any]:
    """Re-train every method at several labelled fractions.

    This is the sweep that decides whether a semi-supervised method is worth
    anything: any method looks good at 50% labels, where the supervised branch
    dominates. The interesting question is the 5-10% regime, and whether the
    gain shrinks monotonically as labels are added - if it does not, the method
    is probably just acting as a regulariser rather than exploiting the
    unlabelled pool.

    Returns:
        Dict with ``table``, ``sweep`` (``{method: {fraction: Interval}}``) and
        ``rows``.
    """
    rows: list[dict[str, Any]] = []
    sweep: dict[str, dict[float, Any]] = {}

    for fraction in fractions:
        for path in config_paths:
            cfg = load_config(path, overrides)
            cfg = replace(
                cfg,
                data=replace(cfg.data, labeled_fraction=fraction),
                run=replace(cfg.run, name=f"{cfg.run.name}_lf{int(fraction * 100):03d}"),
            )
            run = run_training(cfg)
            result = run["result"]
            if result is None:  # pragma: no cover - evaluate_test defaults True
                continue

            method = cfg.semi.method if cfg.semi.method != "none" else "supervised"
            interval = bootstrap_ci(result.metric_array(metric), cfg.eval.bootstrap)
            sweep.setdefault(method, {})[fraction] = interval
            rows.append(
                {
                    "method": method,
                    "labeled_fraction": fraction,
                    "n_labeled": result.extra["n_labeled"],
                    metric: interval.estimate,
                    "ci_lower": interval.lower,
                    "ci_upper": interval.upper,
                    **{k: result.summary.get(k) for k in ("iou", "hd95", "boundary_f1")},
                    "ece": result.calibration.get("ece"),
                    "ause": result.calibration.get("ause"),
                }
            )

    frame = pd.DataFrame(rows)
    path = Path(out_dir) / "tables" / "labeled_fraction_sweep.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    logger.info("wrote %s", path)
    return {"table": frame, "sweep": sweep, "rows": rows}


def run_component_ablation(
    base_config: str | Path = "configs/evidential.yaml",
    overrides: list[str] | None = None,
    out_dir: str | Path = "results",
) -> pd.DataFrame:
    """Remove one mechanism at a time from the proposed method.

    Each variant disables exactly one component, so the resulting drop is
    attributable. The variants are:

    ``full``
        The complete method.
    ``no_dissonance_temper``
        Fixed sharpening temperature everywhere. Isolates the value of
        softening ambiguous boundary pixels.
    ``no_vacuity_gate``
        Uniform consistency weight with the evidential head, soft targets and
        dissonance tempering all retained. Isolates the *weighting rule* alone.
    ``no_kl``
        Evidential head without the KL regulariser. Predicts whether vacuity
        stays informative when nothing penalises runaway evidence.
    ``no_axial``
        Bottleneck attention removed. Separates the architecture's contribution
        from the training ideology's.
    ``gate_without_evidential_loss``
        Keeps the vacuity gate and dissonance tempering, but trains the
        supervised branch with plain cross-entropy + Dice. The gate still runs,
        reading ``softplus(logits)`` as evidence, but nothing in the objective
        has taught those logits to *behave* like evidence. This asks whether the
        weighting rule needs the evidential training objective or merely the
        arithmetic. (An earlier version used a plain softmax + Mean Teacher
        variant here, which was simply a duplicate of the ``mean_teacher``
        baseline and answered nothing.)
    """
    variants: dict[str, list[str]] = {
        "full": [],
        "no_dissonance_temper": ["semi.temperature=1.0"],
        "no_vacuity_gate": ["semi.use_vacuity_gate=false"],
        "no_kl": ["loss.kl_weight=0.0"],
        "no_axial": ["model.axial_attention=false"],
        "gate_without_evidential_loss": ["loss.supervised=ce_dice"],
    }

    rows: list[dict[str, Any]] = []
    for name, variant_overrides in variants.items():
        cfg = load_config(base_config, (overrides or []) + variant_overrides)
        cfg = replace(cfg, run=replace(cfg.run, name=f"ablation_{name}"))
        result = run_training(cfg)["result"]
        if result is None:  # pragma: no cover
            continue
        rows.append(
            {
                "variant": name,
                "changed": ", ".join(variant_overrides) or "(none)",
                "dice": result.summary["dice"],
                "iou": result.summary["iou"],
                "hd95": result.summary["hd95"],
                "boundary_f1": result.summary["boundary_f1"],
                "ece": result.calibration["ece"],
                "ause": result.calibration["ause"],
            }
        )

    frame = pd.DataFrame(rows)
    if not frame.empty and "full" in set(frame["variant"]):
        # Delta against the full method makes each row readable on its own.
        full = frame[frame["variant"] == "full"].iloc[0]
        for metric in ("dice", "iou", "boundary_f1"):
            frame[f"delta_{metric}"] = (frame[metric] - full[metric]).round(4)

    path = Path(out_dir) / "tables" / "ablation_components.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    logger.info("wrote %s", path)
    return frame


def compare_ablation_variants(
    out_dir: str | Path = "results",
    baseline_variant: str = "full",
    metrics: tuple[str, ...] = ("dice", "iou", "boundary_f1", "hd95"),
    prefix: str = "ablation_",
    n_resamples: int = 2000,
) -> pd.DataFrame:
    """Paired significance tests of every ablation variant against the full method.

    An ablation table of raw deltas is not interpretable on its own. With
    per-image Dice standard deviations near 0.16 and 150 test images, the
    standard error of a mean is about 0.013 - so a reported drop of 0.005 is
    indistinguishable from zero, and presenting it as a component's
    "contribution" would be an overclaim. This reads the per-image CSVs each
    variant already wrote and runs the same paired Wilcoxon + bootstrap +
    Holm-Bonferroni machinery used for the method comparison.

    Because it works from saved CSVs it can also be run long after the fact,
    without re-training anything.

    Args:
        out_dir: Root containing ``runs/<prefix><variant>/per_image.csv``.
        baseline_variant: Variant every other is compared against.
        metrics: Metrics to test.
        prefix: Run-name prefix used by :func:`run_component_ablation`.
        n_resamples: Bootstrap replicates.

    Returns:
        A DataFrame with one row per (metric, variant), including the corrected
        p-value. Positive ``difference`` means *removing* that component
        improved the metric. Empty if the baseline variant is missing.
    """
    runs_dir = Path(out_dir) / "runs"
    baseline_path = runs_dir / f"{prefix}{baseline_variant}" / "per_image.csv"
    if not baseline_path.is_file():
        logger.warning("no ablation baseline at %s; skipping paired tests", baseline_path)
        return pd.DataFrame()

    baseline = pd.read_csv(baseline_path)
    variants = sorted(
        d.name[len(prefix):]
        for d in runs_dir.glob(f"{prefix}*")
        if d.is_dir()
        and (d / "per_image.csv").is_file()
        and d.name != f"{prefix}{baseline_variant}"
    )

    rows: list[dict[str, Any]] = []
    for metric in metrics:
        if metric not in baseline.columns:
            continue
        comparisons = [
            compare(
                pd.read_csv(runs_dir / f"{prefix}{v}" / "per_image.csv")[metric].to_numpy(),
                baseline[metric].to_numpy(),
                v,
                baseline_variant,
                n_resamples,
                seed=0,
            )
            for v in variants
        ]
        holm_bonferroni(comparisons)
        for c in comparisons:
            rows.append(
                {
                    "metric": metric,
                    "variant": c.name_a,
                    "mean_variant": round(c.mean_a, 5),
                    "mean_full": round(c.mean_b, 5),
                    "difference": round(c.difference.estimate, 5),
                    "ci_lower": round(c.difference.lower, 5),
                    "ci_upper": round(c.difference.upper, 5),
                    "p_wilcoxon": c.p_value,
                    "p_holm": c.p_adjusted,
                    "cohens_d": round(c.effect_size, 4),
                    "significant": c.significant,
                    "n": c.n,
                }
            )

    frame = pd.DataFrame(rows)
    path = Path(out_dir) / "tables" / "ablation_statistics.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    logger.info("wrote %s", path)
    return frame


# --------------------------------------------------------------------------- #
# Efficiency
# --------------------------------------------------------------------------- #


def run_efficiency_benchmark(
    model_names: tuple[str, ...] = ("separable_unet_tiny", "separable_unet", "unet"),
    image_size: int = 128,
    batch_sizes: tuple[int, ...] = (1, 8),
    device: str = "auto",
    out_dir: str | Path = "results",
    repeats: int = 20,
) -> pd.DataFrame:
    """Measure parameters, MACs and latency for each architecture.

    No training involved - this isolates the cost side of the efficiency claim
    so it can be reported independently of any accuracy number.
    """
    from evissl.config import ModelConfig

    torch_device = resolve_device(device)
    rows: list[dict[str, Any]] = []

    for name in model_names:
        model = build_model(ModelConfig(name=name))
        complexity = estimate_macs(model, (3, image_size, image_size), torch_device)
        row: dict[str, Any] = {
            "model": name,
            "params": complexity["params"],
            "params_m": round(complexity["params_m"], 4),
            "gmacs": round(complexity["gmacs"], 4),
            "image_size": image_size,
            "device": str(torch_device),
        }
        for batch_size in batch_sizes:
            latency = measure_latency(
                model, (3, image_size, image_size), batch_size,
                repeats=repeats, device=torch_device,
            )
            row[f"latency_bs{batch_size}_ms"] = round(latency["median_ms"], 3)
            row[f"throughput_bs{batch_size}_ips"] = round(latency["throughput_ips"], 2)
        rows.append(row)
        logger.info("%s: %.3fM params, %.3f GMACs", name, row["params_m"], row["gmacs"])

    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["params_reduction_x"] = (frame["params"].max() / frame["params"]).round(2)
        frame["macs_reduction_x"] = (frame["gmacs"].max() / frame["gmacs"]).round(2)
        reference_batch = batch_sizes[0]
        latency_column = f"latency_bs{reference_batch}_ms"
        if latency_column in frame:
            frame["latency_reduction_x"] = (
                frame[latency_column].max() / frame[latency_column]
            ).round(2)
            # Arithmetic efficiency: how many MACs the hardware actually retires
            # per millisecond. This is the number that explains why a 36x MAC
            # reduction does not buy a 36x speed-up - dense convolutions
            # vectorise well, narrow depthwise ones are memory-bandwidth bound.
            frame["macs_per_ms_M"] = (
                frame["gmacs"] * 1e3 / frame[latency_column]
            ).round(2)

    path = Path(out_dir) / "tables" / "efficiency.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    logger.info("wrote %s", path)
    return frame


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def load_history(run_name: str, out_dir: str | Path = "results") -> list[dict[str, Any]]:
    """Read a run's per-epoch JSONL history back from disk."""
    path = Path(out_dir) / "runs" / run_name / "history.jsonl"
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def format_table(frame: pd.DataFrame, metrics: tuple[str, ...] | None = None) -> str:
    """Render a results frame as a GitHub-flavoured markdown table.

    Arrows mark the direction of improvement so a reader does not have to
    remember that HD95 and ECE are better when small.
    """
    metrics = metrics or tuple(c for c in HEADLINE_COLUMNS if c in frame.columns)
    view = frame[[m for m in metrics if m in frame.columns]].copy()
    header = [
        f"{m} {'v' if m in LOWER_IS_BETTER else '^'}" if m != "run" else m for m in view.columns
    ]

    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for _, row in view.iterrows():
        cells = [
            f"{v:.4f}" if isinstance(v, (float, np.floating)) and np.isfinite(v) else str(v)
            for v in row
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
