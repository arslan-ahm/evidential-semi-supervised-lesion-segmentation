"""Model registry.

Configs name architectures by string, so a new backbone becomes available to
every script and notebook by registering it here.
"""

from __future__ import annotations

from collections.abc import Callable

from torch import nn

from evissl.config import ModelConfig
from evissl.models.efficient_unet import SepUNet
from evissl.models.unet import UNet

Builder = Callable[[ModelConfig], nn.Module]


def _separable_unet(cfg: ModelConfig) -> nn.Module:
    return SepUNet(
        in_channels=cfg.in_channels,
        num_classes=cfg.num_classes,
        width=cfg.width,
        depth=cfg.depth,
        dropout=cfg.dropout,
        axial_attention=cfg.axial_attention,
    )


def _separable_unet_tiny(cfg: ModelConfig) -> nn.Module:
    """A deliberately minimal variant (~0.1M parameters) for the size ablation."""
    return SepUNet(
        in_channels=cfg.in_channels,
        num_classes=cfg.num_classes,
        width=max(8, cfg.width // 2),
        depth=max(3, cfg.depth - 1),
        dropout=cfg.dropout,
        axial_attention=cfg.axial_attention,
        max_width=96,
        blocks_per_stage=1,
    )


def _unet(cfg: ModelConfig) -> nn.Module:
    # The baseline keeps its published width regardless of cfg.width, so the
    # comparison is against the standard 31M-parameter model, not a shrunk one.
    return UNet(
        in_channels=cfg.in_channels,
        num_classes=cfg.num_classes,
        width=64,
        depth=cfg.depth,
        dropout=cfg.dropout,
    )


REGISTRY: dict[str, Builder] = {
    "separable_unet": _separable_unet,
    "separable_unet_tiny": _separable_unet_tiny,
    "unet": _unet,
}


def build_model(cfg: ModelConfig) -> nn.Module:
    """Instantiate the architecture named by ``cfg.name``.

    Raises:
        ValueError: if the name is not registered.
    """
    key = cfg.name.strip().lower()
    if key not in REGISTRY:
        raise ValueError(f"Unknown model {cfg.name!r}. Available: {sorted(REGISTRY)}")
    return REGISTRY[key](cfg)


def available_models() -> list[str]:
    """Names accepted by :func:`build_model`."""
    return sorted(REGISTRY)
