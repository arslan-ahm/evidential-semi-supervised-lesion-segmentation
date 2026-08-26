"""The evidential decomposition, checked against its closed forms.

These are the load-bearing tests of the repository. The whole method rests on
vacuity and dissonance meaning what the paper says they mean, so each identity
is asserted directly rather than inferred from downstream accuracy.
"""

from __future__ import annotations

import pytest
import torch

from evissl.uncertainty import (
    EPS,
    belief,
    dirichlet_alpha,
    dissonance,
    evidence_from_logits,
    evidential_output,
    expected_probability,
    log_probabilities,
    mc_dropout_predict,
    predictive_entropy,
    probabilities,
    sharpen,
    tta_predict,
    vacuity,
)


def test_evidence_is_non_negative_and_finite():
    logits = torch.tensor([-1e4, -50.0, -1.0, 0.0, 1.0, 50.0, 1e4])
    evidence = evidence_from_logits(logits)
    assert torch.all(evidence >= 0)
    assert torch.all(torch.isfinite(evidence))


def test_belief_plus_vacuity_is_one():
    """The subjective-logic identity sum_k b_k + u = 1, on random evidence."""
    torch.manual_seed(0)
    alpha = dirichlet_alpha(torch.randn(4, 2, 8, 8) * 5)
    total = belief(alpha).sum(dim=1) + vacuity(alpha)
    assert torch.allclose(total, torch.ones_like(total), atol=1e-5)


def test_no_evidence_is_total_vacuity():
    out = evidential_output(torch.full((1, 2, 4, 4), -60.0))
    assert out.vacuity.min() > 0.999
    assert out.dissonance.max() < 1e-4
    # With no evidence the Dirichlet is uniform, so the mean is exactly 0.5.
    assert torch.allclose(out.lesion_prob, torch.full_like(out.lesion_prob, 0.5), atol=1e-5)


def test_conflicting_evidence_is_dissonance_not_vacuity():
    """The case a softmax confidence cannot express.

    Equal *large* evidence for both classes gives P = 0.5 with low vacuity and
    high dissonance, whereas no evidence gives P = 0.5 with vacuity 1 and zero
    dissonance. Both report 0.5; only the decomposition tells them apart.
    """
    conflict = evidential_output(torch.full((1, 2, 4, 4), 30.0))
    ignorance = evidential_output(torch.full((1, 2, 4, 4), -60.0))

    assert torch.allclose(conflict.lesion_prob, ignorance.lesion_prob, atol=1e-4)
    assert conflict.dissonance.min() > 0.9
    assert conflict.vacuity.max() < 0.1
    assert ignorance.dissonance.max() < 1e-4
    assert ignorance.vacuity.min() > 0.999


def test_confident_evidence_is_neither():
    out = evidential_output(torch.tensor([[[[-30.0]], [[30.0]]]]))
    assert out.lesion_prob.item() > 0.95
    assert out.vacuity.max() < 0.1
    assert out.dissonance.max() < 0.1


def test_binary_dissonance_matches_general_definition():
    """The K=2 fast path must equal Josang's general expression."""
    torch.manual_seed(1)
    for scale in (0.5, 2.0, 8.0):
        alpha = dirichlet_alpha(torch.randn(3, 2, 6, 6) * scale)
        b = belief(alpha)
        b0, b1 = b[:, 0], b[:, 1]

        # Josang's general expression, written out for K=2:
        #   diss = sum_k b_k * Bal(b_j, b_k),  Bal = 1 - |b_0 - b_1| / (b_0 + b_1)
        pair = (b0 + b1).clamp_min(EPS)
        balance = 1.0 - (b0 - b1).abs() / pair
        general = b0 * balance + b1 * balance

        # ...and the closed form the implementation actually evaluates.
        assert torch.allclose(dissonance(alpha), general, atol=1e-5)
        assert torch.allclose(dissonance(alpha), 2.0 * torch.minimum(b0, b1), atol=1e-5)


def test_dissonance_multiclass_branch_is_bounded():
    torch.manual_seed(2)
    d = dissonance(dirichlet_alpha(torch.randn(2, 5, 4, 4) * 3))
    assert d.shape == (2, 4, 4)
    assert float(d.min()) >= 0.0
    assert float(d.max()) <= 1.0


def test_dissonance_zero_belief_multiclass():
    """Bal must be 0, not 1, when both beliefs vanish."""
    d = dissonance(dirichlet_alpha(torch.full((1, 4, 3, 3), -60.0)))
    assert float(d.max()) == pytest.approx(0.0, abs=1e-6)


def test_probabilities_sum_to_one_for_both_heads():
    torch.manual_seed(3)
    logits = torch.randn(2, 2, 5, 5) * 4
    for head in ("evidential", "ce_dice"):
        p = probabilities(logits, head)
        assert torch.allclose(p.sum(dim=1), torch.ones(2, 5, 5), atol=1e-5)
        assert torch.allclose(log_probabilities(logits, head).exp(), p, atol=1e-5)


def test_expected_probability_matches_manual_ratio():
    alpha = torch.tensor([[[[3.0]], [[1.0]]]])
    p = expected_probability(alpha)
    assert p[0, 0, 0, 0].item() == pytest.approx(0.75)
    assert p[0, 1, 0, 0].item() == pytest.approx(0.25)


def test_sharpen_scalar_temperature():
    p = torch.tensor([[[[0.7]], [[0.3]]]])
    hot = sharpen(p, 0.5)
    assert hot[0, 0, 0, 0] > p[0, 0, 0, 0]
    assert torch.allclose(hot.sum(dim=1), torch.ones(1, 1, 1))
    # T = 1 is the identity.
    assert torch.allclose(sharpen(p, 1.0), p, atol=1e-6)


def test_sharpen_accepts_per_pixel_temperature():
    """Per-pixel temperature is what dissonance tempering needs."""
    p = torch.tensor([[[[0.7, 0.7]], [[0.3, 0.3]]]])
    temperature = torch.tensor([[[0.2, 1.0]]])
    out = sharpen(p, temperature)
    assert out[0, 0, 0, 0] > out[0, 0, 0, 1]  # sharpened harder on the left
    assert out[0, 0, 0, 1].item() == pytest.approx(0.7, abs=1e-5)


def test_predictive_entropy_is_normalised():
    uniform = torch.full((1, 2, 3, 3), 0.5)
    certain = torch.zeros(1, 2, 3, 3)
    certain[:, 1] = 1.0
    assert torch.allclose(predictive_entropy(uniform), torch.ones(1, 3, 3), atol=1e-5)
    assert float(predictive_entropy(certain).max()) < 1e-5


class _DropoutNet(torch.nn.Module):
    """Minimal network with dropout, for the sampling baselines."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 2, 1)
        self.drop = torch.nn.Dropout2d(0.5)
        self.norm = torch.nn.BatchNorm2d(3)

    def forward(self, x):
        return self.conv(self.drop(self.norm(x)))

    def enable_mc_dropout(self):
        self.eval()
        for module in self.modules():
            if isinstance(module, torch.nn.Dropout2d):
                module.train()


def test_mc_dropout_decomposes_and_restores_mode():
    torch.manual_seed(4)
    net = _DropoutNet().train()
    out = mc_dropout_predict(net, torch.randn(2, 3, 6, 6), samples=6)
    assert set(out) == {"prob", "total", "aleatoric", "epistemic"}
    assert torch.all(out["epistemic"] >= 0)
    # Sampling must not leave the network in a different mode than it found it.
    assert net.training is True


def test_mc_dropout_rejects_single_sample():
    with pytest.raises(ValueError, match="at least 2"):
        mc_dropout_predict(_DropoutNet(), torch.randn(1, 3, 4, 4), samples=1)


def test_tta_is_flip_invariant_for_a_symmetric_model():
    """A pointwise model has flip-equivariant outputs, so TTA must be a no-op."""
    torch.manual_seed(5)
    net = torch.nn.Conv2d(3, 2, 1)
    images = torch.randn(2, 3, 8, 8)
    direct = probabilities(net(images), "evidential")
    assert torch.allclose(tta_predict(net, images, "evidential"), direct, atol=1e-5)
