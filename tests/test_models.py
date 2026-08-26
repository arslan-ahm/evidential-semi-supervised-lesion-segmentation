"""Architectures: shapes, budgets, gradient flow, initialisation contracts."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from evissl.config import ModelConfig
from evissl.models import SepUNet, UNet, available_models, build_model
from evissl.models.blocks import (
    GatedAxialAttention,
    ResidualSeparableBlock,
    SeparableConv,
    SqueezeExcite,
    group_norm,
)
from evissl.utils.complexity import count_parameters, estimate_macs, measure_latency


@pytest.mark.parametrize("name", available_models())
@pytest.mark.parametrize("size", [64, 128])
def test_output_shape_matches_input_resolution(name, size):
    model = build_model(ModelConfig(name=name))
    out = model(torch.zeros(2, 3, size, size))
    assert out.shape == (2, 2, size, size)


def test_unknown_model_is_rejected():
    with pytest.raises(ValueError, match="Unknown model"):
        build_model(ModelConfig(name="resnet9000"))


def test_parameter_budget_of_the_proposed_model():
    """The headline claim: under one million parameters."""
    params = count_parameters(build_model(ModelConfig(name="separable_unet")))
    assert params < 1_000_000, f"separable_unet grew to {params:,} parameters"
    assert params > 300_000, "suspiciously small - check the width/depth defaults"


def test_baseline_unet_is_the_standard_31m_model():
    """The baseline must stay at published width, or the comparison is unfair."""
    params = count_parameters(build_model(ModelConfig(name="unet")))
    assert 30_000_000 < params < 32_000_000


def test_baseline_width_ignores_config_width():
    """cfg.width shrinks the proposed model but must not shrink the baseline."""
    narrow = build_model(ModelConfig(name="unet", width=8))
    assert count_parameters(narrow) > 30_000_000


def test_tiny_variant_is_much_smaller():
    tiny = count_parameters(build_model(ModelConfig(name="separable_unet_tiny")))
    full = count_parameters(build_model(ModelConfig(name="separable_unet")))
    assert tiny < full / 3


def test_efficiency_gap_against_the_baseline():
    small = estimate_macs(build_model(ModelConfig(name="separable_unet")), (3, 128, 128))
    large = estimate_macs(build_model(ModelConfig(name="unet")), (3, 128, 128))
    assert large["params"] / small["params"] > 20
    assert large["macs"] / small["macs"] > 20


def test_gradients_reach_every_trainable_parameter():
    """A parameter with no gradient is dead weight and usually a wiring bug."""
    model = build_model(ModelConfig(name="separable_unet"))
    model(torch.randn(2, 3, 64, 64)).sum().backward()
    missing = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    assert not missing, f"no gradient for: {missing}"


def test_head_starts_at_uniform_prediction():
    """A zeroed head means zero evidence, i.e. total vacuity at step 0.

    That is the correct starting point for the evidential objective: any other
    initialisation asserts evidence the model has not yet seen.
    """
    model = build_model(ModelConfig(name="separable_unet"))
    logits = model(torch.randn(2, 3, 32, 32))
    assert torch.allclose(logits, torch.zeros_like(logits), atol=1e-6)


def test_no_batchnorm_in_the_proposed_model():
    """BatchNorm buffers interact badly with an EMA teacher; GroupNorm avoids it."""
    model = build_model(ModelConfig(name="separable_unet"))
    offenders = [
        type(m).__name__
        for m in model.modules()
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))
    ]
    assert not offenders, f"found normalisation with running state: {offenders}"
    assert not list(model.buffers()), "proposed model should carry no buffers"


def test_group_norm_divides_channels():
    for channels in (1, 3, 8, 12, 24, 96, 192):
        norm = group_norm(channels)
        assert channels % norm.num_groups == 0


def test_mc_dropout_mode_only_activates_dropout():
    model = build_model(ModelConfig(name="separable_unet", dropout=0.2))
    model.eval()
    model.enable_mc_dropout()
    dropouts = [m for m in model.modules() if isinstance(m, (nn.Dropout, nn.Dropout2d))]
    others = [
        m for m in model.modules()
        if isinstance(m, (nn.GroupNorm, nn.LayerNorm))
    ]
    assert dropouts and all(m.training for m in dropouts)
    assert all(not m.training for m in others)


def test_mc_dropout_produces_varying_outputs():
    model = build_model(ModelConfig(name="separable_unet", dropout=0.3))
    # Give the head non-zero weights, or a zeroed head hides all variation.
    nn.init.normal_(model.head.weight, std=0.5)
    model.enable_mc_dropout()
    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        a, b = model(x), model(x)
    assert not torch.allclose(a, b)


def test_axial_attention_is_identity_at_initialisation():
    """The zero gate means the block cannot destabilise early training."""
    attention = GatedAxialAttention(32)
    x = torch.randn(2, 32, 8, 8)
    assert torch.allclose(attention(x), x, atol=1e-6)


def test_squeeze_excite_is_identity_at_initialisation():
    block = SqueezeExcite(16)
    x = torch.randn(2, 16, 8, 8)
    assert torch.allclose(block(x), x, atol=1e-6)


def test_axial_attention_becomes_active_once_gated():
    attention = GatedAxialAttention(32)
    with torch.no_grad():
        attention.gamma.fill_(1.0)
    x = torch.randn(2, 32, 8, 8)
    assert not torch.allclose(attention(x), x, atol=1e-4)


def test_axial_attention_handles_non_square_maps():
    attention = GatedAxialAttention(16)
    with torch.no_grad():
        attention.gamma.fill_(0.5)
    out = attention(torch.randn(2, 16, 6, 10))
    assert out.shape == (2, 16, 6, 10)


def test_separable_conv_is_cheaper_than_dense():
    separable = count_parameters(SeparableConv(64, 64))
    dense = count_parameters(nn.Conv2d(64, 64, 3, padding=1, bias=False))
    assert separable < dense / 4


def test_residual_block_preserves_shape():
    block = ResidualSeparableBlock(24, dropout=0.1)
    x = torch.randn(2, 24, 16, 16)
    assert block(x).shape == x.shape


def test_depth_one_model_is_valid():
    model = SepUNet(depth=1, width=8)
    assert model(torch.zeros(1, 3, 32, 32)).shape == (1, 2, 32, 32)


def test_zero_depth_is_rejected():
    with pytest.raises(ValueError, match="depth must be"):
        SepUNet(depth=0)


def test_unet_handles_odd_input_size():
    """Odd sizes leave a one-pixel mismatch at each skip; output must still match."""
    model = UNet(width=8, depth=2)
    assert model(torch.zeros(1, 3, 33, 37)).shape == (1, 2, 33, 37)


def test_macs_counter_matches_a_hand_computed_conv():
    """Validate the MAC estimator against a case that can be worked out by hand."""
    model = nn.Conv2d(3, 4, 3, padding=1, bias=False)
    stats = estimate_macs(model, (3, 8, 8))
    # 4 output channels x 8 x 8 positions x (3x3 kernel x 3 input channels)
    assert stats["macs"] == 4 * 8 * 8 * 9 * 3


def test_macs_counter_accounts_for_groups():
    depthwise = nn.Conv2d(8, 8, 3, padding=1, groups=8, bias=False)
    stats = estimate_macs(depthwise, (8, 8, 8))
    assert stats["macs"] == 8 * 8 * 8 * 9 * 1


def test_complexity_probe_restores_training_mode():
    model = build_model(ModelConfig(name="separable_unet")).train()
    estimate_macs(model, (3, 32, 32))
    measure_latency(model, (3, 32, 32), warmup=1, repeats=2)
    assert model.training is True


def test_latency_report_is_self_consistent():
    model = build_model(ModelConfig(name="separable_unet_tiny"))
    stats = measure_latency(model, (3, 32, 32), batch_size=2, warmup=1, repeats=5)
    assert stats["min_ms"] <= stats["median_ms"]
    assert stats["throughput_ips"] > 0
