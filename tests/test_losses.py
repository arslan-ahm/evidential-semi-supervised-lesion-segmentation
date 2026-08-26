"""Losses: gradient flow, limiting cases, and the weighting semantics."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from evissl.config import LossConfig, SemiConfig
from evissl.losses import (
    boundary_weight_map,
    consistency_loss,
    cross_entropy_loss,
    evidential_kl,
    evidential_nll,
    sigmoid_rampup,
    soft_dice_loss,
    supervised_loss,
)
from evissl.uncertainty import dirichlet_alpha


def _target(shape=(2, 16, 16), positive=0.3, seed=0):
    rng = np.random.default_rng(seed)
    return torch.from_numpy((rng.random(shape) < positive).astype(np.int64))


def _confident_logits(target, magnitude=20.0):
    """Logits that predict ``target`` with large evidence."""
    return torch.stack([(1 - target) * magnitude, target * magnitude], dim=1).float()


# --------------------------------------------------------------------------- #
# Region losses
# --------------------------------------------------------------------------- #


def test_soft_dice_is_zero_for_a_perfect_prediction():
    target = _target()
    prob = torch.stack([1.0 - target, target], dim=1).float()
    assert float(soft_dice_loss(prob, target)) == pytest.approx(0.0, abs=1e-3)


def test_soft_dice_is_one_for_an_inverted_prediction():
    target = _target()
    prob = torch.stack([target, 1.0 - target], dim=1).float()
    assert float(soft_dice_loss(prob, target)) > 0.99


def test_soft_dice_handles_an_empty_target():
    target = torch.zeros(1, 8, 8, dtype=torch.long)
    prob = torch.zeros(1, 2, 8, 8)
    prob[:, 0] = 1.0
    # Empty prediction on an empty target is perfect, not undefined.
    assert float(soft_dice_loss(prob, target)) == pytest.approx(0.0, abs=1e-3)


def test_soft_dice_ignores_background_by_default():
    """Including background would let background accuracy mask lesion failure."""
    target = _target(positive=0.05)
    prob = torch.zeros(2, 2, 16, 16)
    prob[:, 0] = 1.0  # predict all background
    with_background = float(soft_dice_loss(prob, target, ignore_background=False))
    without = float(soft_dice_loss(prob, target, ignore_background=True))
    assert without > with_background


def test_cross_entropy_is_near_zero_when_confident_and_correct():
    target = _target()
    assert float(cross_entropy_loss(_confident_logits(target), target)) < 1e-6


# --------------------------------------------------------------------------- #
# Evidential terms
# --------------------------------------------------------------------------- #


def test_evidential_nll_decreases_as_correct_evidence_grows():
    target = _target()
    losses = [
        float(evidential_nll(dirichlet_alpha(_confident_logits(target, m)), target))
        for m in (0.5, 2.0, 10.0, 40.0)
    ]
    assert losses == sorted(losses, reverse=True), losses


def test_evidential_kl_is_zero_without_misleading_evidence():
    """Evidence only on the correct class must incur no KL penalty."""
    target = _target()
    logits = torch.stack([(1 - target) * 30.0 - 60.0 * target,
                          target * 30.0 - 60.0 * (1 - target)], dim=1).float()
    assert float(evidential_kl(dirichlet_alpha(logits), target)) == pytest.approx(0.0, abs=1e-3)


def test_evidential_kl_penalises_wrong_class_evidence():
    target = _target()
    wrong = torch.stack([target * 20.0, (1 - target) * 20.0], dim=1).float()
    assert float(evidential_kl(dirichlet_alpha(wrong), target)) > 1.0


@pytest.mark.parametrize("head", ["ce_dice", "evidential"])
def test_supervised_loss_flows_gradients(head):
    target = _target()
    logits = torch.randn(2, 2, 16, 16, requires_grad=True)
    supervised_loss(logits, target, LossConfig(supervised=head), epoch=5).total.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert float(logits.grad.abs().sum()) > 0


def test_supervised_loss_reports_its_components():
    target = _target()
    out = supervised_loss(
        torch.randn(2, 2, 16, 16), target, LossConfig(supervised="evidential"), epoch=3
    )
    assert {"evidential_nll", "evidential_kl", "kl_anneal", "dice"} <= set(out.parts)


def test_kl_annealing_ramps_from_small_to_full():
    target = _target()
    logits = torch.randn(2, 2, 16, 16)
    cfg = LossConfig(supervised="evidential", kl_anneal_epochs=10)
    early = supervised_loss(logits, target, cfg, epoch=0).parts["kl_anneal"]
    late = supervised_loss(logits, target, cfg, epoch=50).parts["kl_anneal"]
    assert early < 0.2
    assert late == pytest.approx(1.0)


def test_unknown_supervised_head_is_rejected():
    with pytest.raises(ValueError, match="Unknown supervised loss"):
        supervised_loss(torch.randn(1, 2, 4, 4), _target((1, 4, 4)), LossConfig(supervised="nope"))


def test_boundary_term_is_applied_when_weighted():
    target = _target()
    logits = torch.randn(2, 2, 16, 16)
    weights = torch.from_numpy(boundary_weight_map(target.numpy().astype(np.uint8)))
    plain = supervised_loss(logits, target, LossConfig(boundary_weight=0.0))
    weighted = supervised_loss(logits, target, LossConfig(boundary_weight=1.0),
                               weight_map=weights)
    assert "boundary" not in plain.parts
    assert "boundary" in weighted.parts
    assert float(weighted.total) > float(plain.total)


def test_boundary_weight_map_peaks_on_the_contour():
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[10:22, 10:22] = 1
    weights = boundary_weight_map(mask, sigma=3.0)
    edge = weights[10, 15]           # on the boundary
    interior = weights[16, 16]       # lesion centre
    far = weights[0, 0]              # far background
    assert edge > interior
    assert edge > far
    assert weights.min() >= 1.0 and weights.max() <= 2.0


def test_boundary_weight_map_handles_degenerate_masks():
    for mask in (np.zeros((8, 8), np.uint8), np.ones((8, 8), np.uint8)):
        assert np.allclose(boundary_weight_map(mask), 1.0)


# --------------------------------------------------------------------------- #
# Consistency
# --------------------------------------------------------------------------- #


ALL_METHODS = ["none", "mean_teacher", "fixmatch", "evidential"]


@pytest.mark.parametrize("method", ALL_METHODS)
def test_consistency_loss_is_finite_and_differentiable(method):
    student = torch.randn(2, 2, 16, 16, requires_grad=True)
    teacher = torch.randn(2, 2, 16, 16) * 3
    out = consistency_loss(student, teacher, SemiConfig(method=method))
    assert torch.isfinite(out.loss)
    out.loss.backward()
    assert student.grad is not None and torch.isfinite(student.grad).all()


def test_none_method_contributes_nothing():
    student = torch.randn(2, 2, 8, 8, requires_grad=True)
    out = consistency_loss(student, torch.randn(2, 2, 8, 8), SemiConfig(method="none"))
    assert float(out.loss.detach()) == 0.0
    assert out.mask_rate == 0.0


def test_teacher_receives_no_gradient():
    """The teacher is an EMA of the student; back-propagating into it is a bug."""
    student = torch.randn(2, 2, 8, 8, requires_grad=True)
    teacher = torch.randn(2, 2, 8, 8, requires_grad=True)
    consistency_loss(student, teacher, SemiConfig(method="evidential")).loss.backward()
    assert teacher.grad is None


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError, match="Unknown semi-supervised method"):
        consistency_loss(torch.randn(1, 2, 4, 4), torch.randn(1, 2, 4, 4),
                         SemiConfig(method="nope"))


def test_cutout_pixels_are_excluded_from_consistency():
    student = torch.randn(2, 2, 8, 8)
    teacher = torch.randn(2, 2, 8, 8) * 4
    everything = torch.ones(2, 8, 8)
    nothing = torch.zeros(2, 8, 8)
    for method in ("mean_teacher", "fixmatch", "evidential"):
        cfg = SemiConfig(method=method)
        assert consistency_loss(student, teacher, cfg, valid=nothing).mask_rate == 0.0
        full = consistency_loss(student, teacher, cfg, valid=everything)
        assert full.mask_rate >= 0.0


def test_fixmatch_discards_ambiguous_pixels_that_ours_keeps():
    """The central empirical claim, isolated.

    A teacher with modest, evenly split evidence produces probabilities near
    0.5. FixMatch's threshold rejects all of it; the evidential rule keeps a
    non-zero share weighted by the belief mass that is actually present.
    """
    student = torch.randn(2, 2, 16, 16)
    # Moderate evidence, close to evenly split -> P near 0.5.
    teacher = torch.zeros(2, 2, 16, 16) + torch.tensor([2.0, 2.2]).view(1, 2, 1, 1)

    fixmatch = consistency_loss(student, teacher, SemiConfig(method="fixmatch", threshold=0.95))
    ours = consistency_loss(student, teacher, SemiConfig(method="evidential"))

    assert fixmatch.mask_rate == pytest.approx(0.0)
    assert ours.mask_rate > 0.1


def test_vacuity_gate_downweights_low_evidence_pixels():
    """Weight must scale with evidence, continuously and without a threshold."""
    student = torch.randn(1, 2, 8, 8)
    rates = []
    for magnitude in (-4.0, 0.0, 2.0, 6.0):
        teacher = torch.full((1, 2, 8, 8), magnitude)
        teacher[:, 1] += 1.0  # break the tie so this is evidence, not conflict
        rates.append(consistency_loss(student, teacher, SemiConfig(method="evidential")).mask_rate)
    assert rates == sorted(rates), rates
    assert rates[0] < 0.2 and rates[-1] > 0.7


def test_dissonance_tempering_softens_conflicted_pixels():
    """High dissonance must raise the effective temperature towards 1."""
    student = torch.randn(1, 2, 8, 8)
    conflicted = torch.full((1, 2, 8, 8), 20.0)                 # split evidence
    decided = torch.tensor([-20.0, 20.0]).view(1, 2, 1, 1).expand(1, 2, 8, 8).contiguous()

    cfg = SemiConfig(method="evidential", temperature=0.3)
    hot = consistency_loss(student, conflicted, cfg).stats["mean_temperature"]
    cold = consistency_loss(student, decided, cfg).stats["mean_temperature"]
    assert hot > cold
    assert hot == pytest.approx(1.0, abs=0.05)   # no sharpening where conflicted
    assert cold == pytest.approx(0.3, abs=0.05)  # full sharpening where decided


def test_vacuity_cutoff_removes_the_most_ignorant_pixels():
    student = torch.randn(1, 2, 8, 8)
    teacher = torch.full((1, 2, 8, 8), -6.0)  # almost no evidence -> vacuity ~1
    permissive = consistency_loss(
        student, teacher, SemiConfig(method="evidential", vacuity_cutoff=1.01)
    )
    strict = consistency_loss(
        student, teacher, SemiConfig(method="evidential", vacuity_cutoff=0.5)
    )
    assert strict.mask_rate == pytest.approx(0.0)
    assert permissive.mask_rate >= strict.mask_rate


def test_mean_teacher_uses_every_valid_pixel():
    student = torch.randn(2, 2, 8, 8)
    teacher = torch.randn(2, 2, 8, 8)
    out = consistency_loss(student, teacher, SemiConfig(method="mean_teacher"))
    assert out.mask_rate == pytest.approx(1.0)


def test_identical_views_give_zero_mean_teacher_loss():
    logits = torch.randn(2, 2, 8, 8)
    out = consistency_loss(logits, logits, SemiConfig(method="mean_teacher"))
    assert float(out.loss) == pytest.approx(0.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# Schedule
# --------------------------------------------------------------------------- #


def test_rampup_is_monotone_and_saturates():
    values = [sigmoid_rampup(e, 10) for e in range(0, 15)]
    assert values == sorted(values)
    assert values[0] < 0.05
    assert values[10] == pytest.approx(1.0)
    assert values[14] == pytest.approx(1.0)


def test_rampup_disabled_returns_one():
    assert sigmoid_rampup(0, 0) == 1.0


def test_vacuity_gate_switch_isolates_only_the_weighting():
    """The ablation switch must change the weight and nothing else.

    Turning the gate off has to leave the soft-target cross-entropy and the
    dissonance tempering intact. Swapping to `mean_teacher` instead - which an
    earlier version of the ablation did - would change the loss form and the
    target as well, so the measured drop could not be attributed to the gate.
    """
    student = torch.randn(2, 2, 16, 16)
    teacher = torch.randn(2, 2, 16, 16) * 2
    valid = torch.ones(2, 16, 16)

    gated = consistency_loss(
        student, teacher, SemiConfig(method="evidential", use_vacuity_gate=True), valid=valid
    )
    ungated = consistency_loss(
        student, teacher, SemiConfig(method="evidential", use_vacuity_gate=False), valid=valid
    )

    # Uniform weighting uses every valid pixel; the gate uses strictly less.
    assert ungated.mask_rate == pytest.approx(1.0)
    assert gated.mask_rate < ungated.mask_rate

    # Dissonance tempering is untouched by the switch.
    assert gated.stats["mean_temperature"] == pytest.approx(
        ungated.stats["mean_temperature"], abs=1e-6
    )
    assert gated.stats["mean_dissonance"] == pytest.approx(
        ungated.stats["mean_dissonance"], abs=1e-6
    )


def test_ungated_evidential_still_respects_cutout():
    student = torch.randn(1, 2, 8, 8)
    teacher = torch.randn(1, 2, 8, 8)
    cfg = SemiConfig(method="evidential", use_vacuity_gate=False)
    assert consistency_loss(student, teacher, cfg, valid=torch.zeros(1, 8, 8)).mask_rate == 0.0
