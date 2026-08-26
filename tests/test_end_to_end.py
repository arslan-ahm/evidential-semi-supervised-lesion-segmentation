"""End-to-end integration: every method must train, score and save.

These are the slowest tests in the suite (a couple of minutes on CPU) but they
are the ones that catch wiring bugs the unit tests cannot see - a mis-shaped
tensor between the trainer and the loss, a metric key the report expects and
the evaluator never produces, a checkpoint that cannot be reloaded.

Run just these with:  pytest -m slow
Skip them with:       pytest -m "not slow"
"""

from __future__ import annotations

import numpy as np
import pytest

from evissl.config import load_config
from evissl.eval import evaluate, sparsification_for
from evissl.pipelines import format_table, run_efficiency_benchmark, run_training

#: A configuration small enough to train inside a test, large enough to be real.
TINY = [
    "data.image_size=32",
    "data.train_size=24",
    "data.val_size=8",
    "data.test_size=8",
    "data.labeled_fraction=0.5",
    "data.batch_size=4",
    "data.mu=1",
    "model.width=8",
    "model.depth=2",
    "optim.epochs=2",
    "optim.steps_per_epoch=3",
    "optim.warmup_epochs=1",
    "loss.kl_anneal_epochs=1",
    "semi.rampup_epochs=1",
    "eval.bootstrap=50",
]

METHOD_CONFIGS = [
    "configs/supervised_baseline.yaml",
    "configs/mean_teacher.yaml",
    "configs/fixmatch.yaml",
    "configs/evidential.yaml",
]


@pytest.mark.slow
@pytest.mark.parametrize("config_path", METHOD_CONFIGS)
def test_every_method_trains_and_scores(config_path, tmp_path):
    cfg = load_config(
        config_path,
        [*TINY, f"run.out_dir={tmp_path.as_posix()}", f"run.ckpt_dir={tmp_path.as_posix()}"],
    )
    out = run_training(cfg)
    result = out["result"]

    assert result is not None
    # Every metric the report and tables reference must exist and be finite.
    for key in ("dice", "iou", "sensitivity", "specificity", "boundary_f1"):
        assert np.isfinite(result.summary[key]), f"{key} is not finite"
        assert 0.0 <= result.summary[key] <= 1.0
    for key in ("ece", "ause", "brier", "nll"):
        assert np.isfinite(result.calibration[key]), f"{key} is not finite"

    assert len(result.per_image) == 8
    assert out["state"].best_epoch >= 0
    assert (tmp_path / f"{cfg.run.name}.pt").is_file(), "no checkpoint written"
    assert (tmp_path / "runs" / cfg.run.name / "history.jsonl").is_file()
    assert (tmp_path / "runs" / cfg.run.name / "config.yaml").is_file()
    assert (tmp_path / "runs" / cfg.run.name / "per_image.csv").is_file()


@pytest.mark.slow
def test_evidential_run_produces_uncertainty_maps(tmp_path):
    cfg = load_config(
        "configs/evidential.yaml",
        [*TINY, f"run.out_dir={tmp_path.as_posix()}", f"run.ckpt_dir={tmp_path.as_posix()}"],
    )
    result = run_training(cfg, keep_predictions=True)["result"]
    predictions = result.predictions

    assert predictions is not None
    assert predictions.vacuity is not None, "evidential head produced no vacuity"
    assert predictions.dissonance is not None
    assert predictions.prob.shape == predictions.vacuity.shape
    assert float(predictions.vacuity.min()) >= 0.0
    assert float(predictions.vacuity.max()) <= 1.0 + 1e-6
    assert float(predictions.dissonance.min()) >= 0.0
    assert float(predictions.dissonance.max()) <= 1.0 + 1e-6

    # The sparsification analysis must run off a completed evaluation.
    curve = sparsification_for(result, steps=5)
    assert curve["model"].shape == curve["oracle"].shape


@pytest.mark.slow
def test_ce_head_run_has_no_evidential_maps(tmp_path):
    """A softmax model must not silently report vacuity it cannot compute."""
    cfg = load_config(
        "configs/fixmatch.yaml",
        [*TINY, f"run.out_dir={tmp_path.as_posix()}", f"run.ckpt_dir={tmp_path.as_posix()}"],
    )
    predictions = run_training(cfg, keep_predictions=True)["result"].predictions
    assert predictions.vacuity is None
    assert predictions.entropy is not None, "entropy must still be available"
    # primary_uncertainty must fall back gracefully rather than crash.
    assert predictions.primary_uncertainty().shape == predictions.prob.shape


@pytest.mark.slow
def test_sparsification_requires_predictions(tmp_path):
    cfg = load_config(
        "configs/evidential.yaml",
        [*TINY, f"run.out_dir={tmp_path.as_posix()}", f"run.ckpt_dir={tmp_path.as_posix()}"],
    )
    result = run_training(cfg, keep_predictions=False)["result"]
    with pytest.raises(ValueError, match="keep_predictions"):
        sparsification_for(result)


@pytest.mark.slow
def test_checkpoint_can_be_reloaded_and_rescored(tmp_path):
    """A saved checkpoint must reproduce a comparable score on reload."""
    from evissl.data import build_dataset, build_loaders
    from evissl.models import build_model
    from evissl.utils.checkpoint import load_checkpoint
    from evissl.utils.seed import seed_everything

    cfg = load_config(
        "configs/evidential.yaml",
        [*TINY, f"run.out_dir={tmp_path.as_posix()}", f"run.ckpt_dir={tmp_path.as_posix()}"],
    )
    original = run_training(cfg)["result"]

    generator = seed_everything(cfg.run.seed)
    bundle = build_dataset(cfg.data, cfg.run.seed)
    loaders = build_loaders(cfg, bundle, generator=generator)
    model = build_model(cfg.model)
    load_checkpoint(tmp_path / f"{cfg.run.name}.pt", model, prefer_ema=True)
    reloaded = evaluate(model, loaders.test, cfg, name="reloaded")

    # The checkpoint holds the best epoch, which may differ from the final one,
    # so require the reloaded score to be at least as good rather than equal.
    assert reloaded.summary["dice"] >= original.summary["dice"] - 1e-6


@pytest.mark.slow
def test_run_is_reproducible_from_its_seed(tmp_path):
    cfg_args = [*TINY, f"run.out_dir={tmp_path.as_posix()}", f"run.ckpt_dir={tmp_path.as_posix()}"]
    first = run_training(load_config("configs/evidential.yaml", cfg_args))["result"]
    second = run_training(load_config("configs/evidential.yaml", cfg_args))["result"]
    assert first.summary["dice"] == pytest.approx(second.summary["dice"], abs=1e-6)
    assert first.calibration["ece"] == pytest.approx(second.calibration["ece"], abs=1e-6)


@pytest.mark.slow
def test_seed_changes_the_result():
    """Guards against an accidentally constant pipeline."""
    base = [*TINY]
    a = run_training(load_config("configs/evidential.yaml", [*base, "run.seed=1"]))["result"]
    b = run_training(load_config("configs/evidential.yaml", [*base, "run.seed=99"]))["result"]
    assert a.summary["dice"] != b.summary["dice"]


def test_efficiency_benchmark_runs_without_training(tmp_path):
    frame = run_efficiency_benchmark(
        ("separable_unet_tiny", "separable_unet"),
        image_size=32,
        batch_sizes=(1,),
        out_dir=tmp_path,
        repeats=3,
    )
    assert len(frame) == 2
    assert (frame["params"] > 0).all()
    assert (frame["gmacs"] > 0).all()
    assert "latency_bs1_ms" in frame
    assert "macs_per_ms_M" in frame
    assert (tmp_path / "tables" / "efficiency.csv").is_file()


def test_format_table_renders_direction_arrows():
    import pandas as pd

    frame = pd.DataFrame([{"run": "x", "dice": 0.9, "hd95": 3.0, "ece": 0.05}])
    text = format_table(frame)
    assert "dice ^" in text     # higher is better
    assert "hd95 v" in text     # lower is better
    assert "ece v" in text
    assert "0.9000" in text
