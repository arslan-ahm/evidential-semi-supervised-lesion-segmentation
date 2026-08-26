"""Training engine: EMA semantics, schedule, checkpoints."""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from evissl.config import Config, load_config
from evissl.engine import ModelEMA, build_optimizer, lr_at
from evissl.utils.checkpoint import load_checkpoint, save_checkpoint

# --------------------------------------------------------------------------- #
# EMA
# --------------------------------------------------------------------------- #


def _constant_linear(value: float) -> nn.Linear:
    layer = nn.Linear(4, 4, bias=False)
    with torch.no_grad():
        layer.weight.fill_(value)
    return layer


def test_teacher_is_detached_and_in_eval_mode():
    ema = ModelEMA(_constant_linear(1.0).train())
    assert ema.module.training is False
    assert all(not p.requires_grad for p in ema.module.parameters())


def test_teacher_starts_as_a_copy_not_a_reference():
    student = _constant_linear(1.0)
    ema = ModelEMA(student)
    with torch.no_grad():
        student.weight.fill_(5.0)
    # Mutating the student must not move the teacher until update() is called.
    assert float(ema.module.weight[0, 0]) == pytest.approx(1.0)


def test_teacher_converges_towards_the_student():
    student = _constant_linear(2.0)
    ema = ModelEMA(_constant_linear(0.0), decay=0.9)
    ema.module.load_state_dict(_constant_linear(0.0).state_dict())
    previous = float(ema.module.weight[0, 0])
    for _ in range(60):
        ema.update(student)
        current = float(ema.module.weight[0, 0])
        assert current >= previous - 1e-9, "teacher moved away from the student"
        previous = current
    assert previous == pytest.approx(2.0, abs=1e-2)


def test_decay_warmup_starts_low_and_rises():
    ema = ModelEMA(_constant_linear(1.0), decay=0.99, warmup=True)
    # (1 + s) / (10 + s) reaches 0.99 at s = 890, so 1200 steps is enough to
    # observe the ramp saturate at the configured decay.
    decays = [ema.update(_constant_linear(2.0)) for _ in range(1200)]
    assert decays[0] < 0.2, "warm-up must start with a fast-tracking teacher"
    assert decays[-1] == pytest.approx(0.99, abs=1e-3)
    assert decays == sorted(decays)


def test_warmup_can_be_disabled():
    ema = ModelEMA(_constant_linear(1.0), decay=0.9, warmup=False)
    assert ema.update(_constant_linear(2.0)) == pytest.approx(0.9)


def test_invalid_decay_is_rejected():
    for decay in (-0.1, 1.0, 1.5):
        with pytest.raises(ValueError, match="decay must be"):
            ModelEMA(_constant_linear(1.0), decay=decay)


def test_buffers_are_copied_not_averaged():
    """Averaged BatchNorm statistics describe no real network."""
    student = nn.BatchNorm2d(3)
    ema = ModelEMA(student, decay=0.5)
    student(torch.randn(8, 3, 4, 4))  # populates running stats
    ema.update(student)
    assert torch.allclose(ema.module.running_mean, student.running_mean)
    assert torch.allclose(ema.module.running_var, student.running_var)


def test_ema_state_roundtrip():
    ema = ModelEMA(_constant_linear(1.0), decay=0.9)
    for _ in range(5):
        ema.update(_constant_linear(3.0))
    restored = ModelEMA(_constant_linear(0.0), decay=0.9)
    restored.load_state_dict(ema.state_dict())
    assert restored.step_count == ema.step_count
    assert torch.allclose(restored.module.weight, ema.module.weight)


def test_ema_forward_takes_no_gradient():
    ema = ModelEMA(_constant_linear(1.0))
    out = ema(torch.randn(2, 4, requires_grad=True))
    assert out.requires_grad is False


# --------------------------------------------------------------------------- #
# Schedule
# --------------------------------------------------------------------------- #


def test_warmup_then_cosine_decay():
    cfg = Config()
    cfg.optim.epochs = 30
    cfg.optim.warmup_epochs = 3
    cfg.optim.scheduler = "cosine"

    assert lr_at(cfg, 0.0) == pytest.approx(0.0, abs=1e-6)
    assert lr_at(cfg, 1.5) == pytest.approx(0.5, abs=1e-3)
    assert lr_at(cfg, 3.0) == pytest.approx(1.0, abs=1e-3)
    # Cosine reaches half-way at the mid-point of the post-warm-up span.
    assert lr_at(cfg, 3.0 + (30 - 3) / 2) == pytest.approx(0.5, abs=1e-2)
    assert lr_at(cfg, 30.0) == pytest.approx(0.0, abs=1e-6)


def test_schedule_never_goes_negative_past_the_end():
    cfg = Config()
    cfg.optim.epochs = 10
    assert lr_at(cfg, 50.0) >= 0.0


def test_scheduler_none_is_flat_after_warmup():
    cfg = Config()
    cfg.optim.scheduler = "none"
    cfg.optim.warmup_epochs = 2
    assert lr_at(cfg, 5.0) == pytest.approx(1.0)
    assert lr_at(cfg, 100.0) == pytest.approx(1.0)


def test_zero_warmup_starts_at_full_rate():
    cfg = Config()
    cfg.optim.warmup_epochs = 0
    cfg.optim.scheduler = "none"
    assert lr_at(cfg, 0.0) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Optimiser
# --------------------------------------------------------------------------- #


def test_norms_and_biases_are_excluded_from_weight_decay():
    from evissl.config import ModelConfig
    from evissl.models import build_model

    cfg = Config()
    cfg.optim.weight_decay = 0.05
    model = build_model(ModelConfig(name="separable_unet"))
    optimizer = build_optimizer(model, cfg)

    decayed, undecayed = optimizer.param_groups
    assert decayed["weight_decay"] == pytest.approx(0.05)
    assert undecayed["weight_decay"] == 0.0
    # Every 1-D parameter (norm scales, biases, gates) belongs to the no-decay
    # group, so nothing in the decayed group may be 1-D.
    assert all(p.ndim > 1 for p in decayed["params"])
    assert len(undecayed["params"]) > 0


def test_unknown_optimiser_is_rejected():
    cfg = Config()
    cfg.optim.name = "lbfgs"
    with pytest.raises(ValueError, match="Unknown optimiser"):
        build_optimizer(nn.Linear(2, 2), cfg)


def test_sgd_is_available():
    cfg = Config()
    cfg.optim.name = "sgd"
    assert build_optimizer(nn.Linear(4, 4), cfg) is not None


# --------------------------------------------------------------------------- #
# Checkpoints
# --------------------------------------------------------------------------- #


def test_checkpoint_roundtrip_prefers_the_teacher(tmp_path):
    student = _constant_linear(1.0)
    teacher = _constant_linear(7.0)
    path = save_checkpoint(
        tmp_path / "run.pt", student, ema_model=teacher, epoch=4,
        metrics={"dice": 0.9}, config={"run": {"name": "x"}},
    )
    target = _constant_linear(0.0)
    payload = load_checkpoint(path, target, prefer_ema=True)
    assert payload["loaded"] == "ema_model"
    assert payload["epoch"] == 4
    assert payload["metrics"]["dice"] == 0.9
    assert float(target.weight[0, 0].detach()) == pytest.approx(7.0)


def test_checkpoint_can_load_the_student_explicitly(tmp_path):
    path = save_checkpoint(
        tmp_path / "run.pt", _constant_linear(1.0), ema_model=_constant_linear(7.0)
    )
    target = _constant_linear(0.0)
    assert load_checkpoint(path, target, prefer_ema=False)["loaded"] == "model"
    assert float(target.weight[0, 0].detach()) == pytest.approx(1.0)


def test_checkpoint_without_ema_falls_back_to_the_student(tmp_path):
    path = save_checkpoint(tmp_path / "run.pt", _constant_linear(3.0))
    target = _constant_linear(0.0)
    assert load_checkpoint(path, target, prefer_ema=True)["loaded"] == "model"
    assert float(target.weight[0, 0].detach()) == pytest.approx(3.0)


def test_checkpoint_carries_the_config_for_later_re_evaluation(tmp_path):
    cfg = load_config("configs/evidential.yaml")
    path = save_checkpoint(tmp_path / "run.pt", _constant_linear(1.0), config=cfg.to_dict())
    payload = load_checkpoint(path)
    assert payload["config"]["semi"]["method"] == "evidential"


def test_best_metric_starts_at_negative_infinity():
    """So the first finite validation score always counts as an improvement."""
    from evissl.engine import TrainState

    assert TrainState().best_metric == -math.inf
