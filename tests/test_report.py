"""Figures, reports and the CLI surface.

Plotting code fails silently more often than it fails loudly - a figure with an
empty axis still writes a valid PNG. These tests therefore assert that each
figure is produced *and* is non-trivial in size, and that every CLI subcommand
at least parses and dispatches.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from evissl.cli import build_parser
from evissl.eval import EvaluationResult, Predictions
from evissl.metrics import Interval, bootstrap_ci, compare, expected_calibration_error
from evissl.metrics.calibration import sparsification_curve
from evissl.pipelines import build_table, format_table, write_statistics
from evissl.report import write_markdown_report, write_sweep_figure
from evissl.viz import (
    colour_for,
    plot_efficiency,
    plot_method_comparison,
    plot_qualitative,
    plot_reliability,
    plot_samples,
    plot_sparsification,
    plot_sweep,
    plot_training_curves,
    plot_uncertainty_separation,
    save,
)

#: A figure smaller than this is almost certainly blank.
MIN_PNG_BYTES = 4_000


def _saved(figure, tmp_path, name):
    path = save(figure, tmp_path / name)
    assert path.is_file()
    size = path.stat().st_size
    assert size > MIN_PNG_BYTES, f"{name} is only {size} bytes - probably blank"
    return path


@pytest.fixture
def fake_predictions():
    rng = np.random.default_rng(0)
    n, size = 6, 32
    target = np.zeros((n, size, size), dtype=np.uint8)
    target[:, 8:24, 8:24] = 1
    # A plausible probability map: right most of the time, fuzzy at the border.
    prob = target * 0.85 + rng.random((n, size, size)) * 0.15
    vacuity = rng.random((n, size, size)) * 0.4
    dissonance = np.zeros_like(vacuity)
    dissonance[:, 7:9, :] = 0.8  # high conflict on the boundary band
    return Predictions(
        prob=prob.astype(np.float32),
        target=target,
        vacuity=vacuity.astype(np.float32),
        dissonance=dissonance.astype(np.float32),
        entropy=(1.0 - np.abs(prob - 0.5) * 2.0).astype(np.float32),
    )


@pytest.fixture
def fake_result(fake_predictions):
    rng = np.random.default_rng(1)
    per_image = [
        {"dice": float(d), "iou": float(d / (2 - d)), "hd95": float(h),
         "assd": float(h / 2), "boundary_f1": float(b)}
        for d, h, b in zip(rng.uniform(0.7, 0.95, 6), rng.uniform(1, 6, 6),
                           rng.uniform(0.5, 0.9, 6), strict=True)
    ]
    return EvaluationResult(
        name="evidential",
        summary={"dice": 0.85, "iou": 0.74, "hd95": 3.1, "assd": 1.5, "boundary_f1": 0.7},
        per_image=per_image,
        calibration={"ece": 0.04, "ace": 0.05, "ause": 0.3, "brier": 0.06, "nll": 0.2},
        predictions=fake_predictions,
        extra={"params_m": 0.96, "gmacs": 0.38, "latency_ms": 70.0},
    )


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #


def test_dataset_preview_figure(tmp_path):
    from evissl.data import generate_dataset

    images, masks, params = generate_dataset(6, 48, seed=2)
    difficulty = np.array([p.difficulty() for p in params])
    _saved(plot_samples(images, masks, difficulty, n=6), tmp_path, "samples.png")


def test_dataset_preview_without_difficulty(tmp_path):
    from evissl.data import generate_dataset

    images, masks, _ = generate_dataset(3, 32, seed=3)
    _saved(plot_samples(images, masks, None, n=3), tmp_path, "samples_plain.png")


def test_training_curves_tolerate_missing_keys(tmp_path):
    """Supervised runs have no mask_rate; the figure must not crash on that."""
    histories = {
        "supervised_baseline": [
            {"epoch": e, "train_loss": 1.0 - e * 0.1, "val_dice": 0.5 + e * 0.05}
            for e in range(5)
        ],
        "evidential": [
            {"epoch": e, "train_loss": 0.9 - e * 0.1, "val_dice": 0.55 + e * 0.05,
             "train_mask_rate": 0.2 + e * 0.1}
            for e in range(5)
        ],
    }
    _saved(plot_training_curves(histories), tmp_path, "curves.png")


def test_training_curves_with_no_usable_metric(tmp_path):
    figure = plot_training_curves({"a": [{"epoch": 0}]})
    _saved(figure, tmp_path, "curves_empty.png")


def test_qualitative_panels(tmp_path, fake_predictions):
    images = (np.random.default_rng(0).random((4, 32, 32, 3)) * 255).astype(np.uint8)
    figure = plot_qualitative(
        images, fake_predictions.target, fake_predictions.prob,
        fake_predictions.vacuity, fake_predictions.dissonance, n=3,
    )
    _saved(figure, tmp_path, "qualitative.png")


def test_qualitative_without_uncertainty_maps(tmp_path, fake_predictions):
    """A softmax baseline has no vacuity; the figure drops those columns."""
    images = (np.random.default_rng(0).random((3, 32, 32, 3)) * 255).astype(np.uint8)
    figure = plot_qualitative(images, fake_predictions.target, fake_predictions.prob, n=2)
    _saved(figure, tmp_path, "qualitative_plain.png")


def test_uncertainty_separation_figure(tmp_path, fake_predictions):
    figure = plot_uncertainty_separation(
        fake_predictions.vacuity, fake_predictions.dissonance, fake_predictions.prob
    )
    _saved(figure, tmp_path, "separation.png")


def test_uncertainty_separation_subsamples_large_inputs(tmp_path):
    rng = np.random.default_rng(0)
    n = 60_000
    figure = plot_uncertainty_separation(
        rng.random(n), rng.random(n), rng.random(n), max_points=5_000
    )
    _saved(figure, tmp_path, "separation_big.png")


def test_reliability_figure(tmp_path):
    rng = np.random.default_rng(4)
    confidence = rng.uniform(0.5, 1.0, 5000)
    correct = (rng.random(5000) < confidence).astype(float)
    curve = expected_calibration_error(confidence, correct, bins=12)
    figure = plot_reliability({"evidential": curve}, {"evidential": curve["ece"]})
    _saved(figure, tmp_path, "reliability.png")


def test_sparsification_figure(tmp_path):
    rng = np.random.default_rng(5)
    error = (rng.random(3000) < 0.25).astype(float)
    curve = sparsification_curve(error * 0.8 + rng.random(3000) * 0.2, error, steps=15)
    figure = plot_sparsification({"evidential": curve}, {"evidential": 0.31})
    _saved(figure, tmp_path, "sparsification.png")


def test_sparsification_figure_with_no_curves(tmp_path):
    _saved(plot_sparsification({}), tmp_path, "sparsification_empty.png")


def test_method_comparison_accepts_intervals_and_bare_floats(tmp_path):
    rng = np.random.default_rng(6)
    intervals = {
        "supervised_baseline": bootstrap_ci(rng.normal(0.80, 0.05, 60), 300, seed=0),
        "fixmatch": bootstrap_ci(rng.normal(0.82, 0.05, 60), 300, seed=1),
        "evidential": bootstrap_ci(rng.normal(0.86, 0.05, 60), 300, seed=2),
    }
    _saved(plot_method_comparison(intervals, "dice"), tmp_path, "comparison.png")
    _saved(
        plot_method_comparison({"a": 0.8, "b": 0.9}, "hd95", lower_is_better=True),
        tmp_path, "comparison_floats.png",
    )


def test_sweep_figure(tmp_path):
    sweep = {
        "supervised": {f: Interval(0.6 + f, 0.55 + f, 0.65 + f, 0.95, 50)
                       for f in (0.05, 0.1, 0.2, 0.5)},
        "evidential": {f: Interval(0.68 + f, 0.63 + f, 0.73 + f, 0.95, 50)
                       for f in (0.05, 0.1, 0.2, 0.5)},
    }
    _saved(plot_sweep(sweep, "dice"), tmp_path, "sweep.png")
    path = write_sweep_figure({"sweep": sweep}, tmp_path)
    assert path.is_file()


def test_efficiency_figure(tmp_path):
    rows = [
        {"run": "separable_unet", "params_m": 0.96, "gmacs": 0.38, "dice": 0.86},
        {"run": "unet", "params_m": 31.04, "gmacs": 13.65, "dice": 0.87},
        {"run": "separable_unet_tiny", "params_m": 0.11, "gmacs": 0.08, "dice": 0.79},
    ]
    _saved(plot_efficiency(rows), tmp_path, "efficiency.png")


def test_colour_assignment_is_stable_and_method_aware():
    assert colour_for("evidential") == colour_for("evidential_lf010")
    assert colour_for("fixmatch") != colour_for("evidential")
    assert colour_for("something_unregistered")  # falls back, does not raise


# --------------------------------------------------------------------------- #
# Tables and reports
# --------------------------------------------------------------------------- #


def test_build_table_orders_headline_columns_first(tmp_path, fake_result):
    frame = build_table({"evidential": fake_result}, tmp_path / "t.csv")
    assert list(frame.columns)[:3] == ["run", "dice", "iou"]
    assert (tmp_path / "t.csv").is_file()
    assert len(frame) == 1


def test_format_table_marks_metric_direction():
    frame = pd.DataFrame([{"run": "x", "dice": 0.9, "hd95": 3.0, "ece": 0.05}])
    text = format_table(frame)
    assert "dice ^" in text and "hd95 v" in text and "ece v" in text
    assert "| x |" in text


def test_format_table_handles_nan():
    frame = pd.DataFrame([{"run": "x", "dice": float("nan")}])
    assert "nan" in format_table(frame).lower()


def test_write_statistics_flattens_comparisons(tmp_path):
    rng = np.random.default_rng(7)
    baseline = rng.normal(0.8, 0.05, 60)
    comparisons = {
        "dice": [compare(baseline + 0.04, baseline, "evidential.dice", "supervised.dice",
                         n_resamples=300)]
    }
    frame = write_statistics(comparisons, tmp_path / "stats.csv")
    assert (tmp_path / "stats.csv").is_file()
    assert {"metric", "method", "baseline", "difference", "ci_lower", "ci_upper",
            "p_wilcoxon", "p_holm", "cohens_d", "significant"} <= set(frame.columns)
    assert frame.loc[0, "method"] == "evidential"


def test_markdown_report_contains_tables_and_tests(tmp_path, fake_result):
    rng = np.random.default_rng(8)
    baseline = rng.normal(0.8, 0.05, 40)
    comparison = {
        "table": build_table({"evidential": fake_result}),
        "comparisons": {
            "dice": [compare(baseline + 0.05, baseline, "evidential.dice",
                             "supervised.dice", n_resamples=200)]
        },
    }
    path = write_markdown_report(comparison, tmp_path / "RESULTS.md")
    text = path.read_text(encoding="utf-8")
    assert "# Results" in text
    assert "Paired statistical tests" in text
    assert "evidential" in text
    assert "Holm" in text


def test_markdown_report_without_statistics(tmp_path, fake_result):
    path = write_markdown_report(
        {"table": build_table({"evidential": fake_result})}, tmp_path / "R.md"
    )
    assert "Test-set summary" in path.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "argv",
    [
        ["train", "--config", "configs/smoke.yaml"],
        ["train", "--config", "configs/smoke.yaml", "--set", "optim.epochs=1"],
        ["compare"],
        ["compare", "--configs", "configs/evidential.yaml", "--figures"],
        ["sweep", "--fractions", "0.05", "0.2"],
        ["ablate"],
        ["bench", "--image-size", "64"],
        ["figures"],
        ["data", "--n", "4"],
    ],
)
def test_every_subcommand_parses(argv):
    args = build_parser().parse_args(argv)
    assert args.command == argv[0]
    assert hasattr(args, "overrides")
    assert hasattr(args, "out_dir")


def test_parser_requires_a_subcommand():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_train_requires_a_config():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["train"])


def test_data_subcommand_writes_a_preview(tmp_path):
    from evissl.cli import main

    code = main([
        "data", "--config", "configs/smoke.yaml", "--n", "4",
        "--out-dir", str(tmp_path),
    ])
    assert code == 0
    assert (tmp_path / "figures" / "dataset_samples.png").is_file()


def test_figures_subcommand_is_quiet_without_runs(tmp_path):
    from evissl.cli import main

    assert main(["figures", "--out-dir", str(tmp_path)]) == 0


# --------------------------------------------------------------------------- #
# Ablation statistics
# --------------------------------------------------------------------------- #


def _write_ablation_run(root, variant, values, metric="dice"):
    """Write a minimal per_image.csv the way run_component_ablation would."""
    d = root / "runs" / f"ablation_{variant}"
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({metric: values, "iou": values * 0.8,
                  "boundary_f1": values * 0.7, "hd95": 10 - values * 5}).to_csv(
        d / "per_image.csv", index=False)


def test_ablation_statistics_detects_a_real_drop(tmp_path):
    """A component that genuinely matters must come out significant."""
    rng = np.random.default_rng(0)
    base = rng.normal(0.80, 0.05, 150)
    _write_ablation_run(tmp_path, "full", base)
    _write_ablation_run(tmp_path, "no_important_part", base - 0.04)  # clearly worse

    # "Within noise" has to mean a noisy *difference*, not a small constant
    # offset. A perfectly consistent +0.0003 on every image is a real effect and
    # a paired test is right to detect it however small it is - which is exactly
    # why the per-image CSVs are needed rather than the means.
    _write_ablation_run(tmp_path, "no_irrelevant_part", base + rng.normal(0, 0.03, 150))

    from evissl.pipelines import compare_ablation_variants

    stats = compare_ablation_variants(tmp_path, metrics=("dice",))
    assert (tmp_path / "tables" / "ablation_statistics.csv").is_file()

    by_variant = stats.set_index("variant")
    assert by_variant.loc["no_important_part", "significant"]
    assert by_variant.loc["no_important_part", "difference"] < 0
    # And a difference inside the noise must NOT be reported as a contribution.
    assert not by_variant.loc["no_irrelevant_part", "significant"]


def test_ablation_statistics_without_a_baseline_is_empty(tmp_path):
    rng = np.random.default_rng(1)
    _write_ablation_run(tmp_path, "some_variant", rng.normal(0.8, 0.05, 20))
    from evissl.pipelines import compare_ablation_variants

    assert compare_ablation_variants(tmp_path).empty


def test_ablation_statistics_applies_holm_across_the_family(tmp_path):
    rng = np.random.default_rng(2)
    base = rng.normal(0.80, 0.05, 120)
    _write_ablation_run(tmp_path, "full", base)
    for i in range(4):
        _write_ablation_run(tmp_path, f"v{i}", base + rng.normal(0, 0.002, 120))

    from evissl.pipelines import compare_ablation_variants

    stats = compare_ablation_variants(tmp_path, metrics=("dice",))
    assert len(stats) == 4
    # Corrected p-values must never be below the raw ones.
    assert (stats["p_holm"] >= stats["p_wilcoxon"] - 1e-12).all()
