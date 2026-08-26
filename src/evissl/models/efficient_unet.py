"""SepUNet-A: a sub-1M-parameter encoder-decoder for lesion segmentation.

Shape of the network at the default settings (``width=24``, ``depth=4``,
128x128 input):

======  ==========  ==========  ==================================
stage   resolution  channels    contents
======  ==========  ==========  ==================================
stem    128         24          3x3 conv, GroupNorm, SiLU
enc1    64          48          separable downsample + 2 res blocks
enc2    32          96          separable downsample + 2 res blocks
enc3    16          192         separable downsample + 2 res blocks
enc4    8           192         separable downsample + 2 res blocks
neck    8           192         gated axial attention + res block
dec4    16          192         bilinear up, skip fuse, res block
dec3    32          96          bilinear up, skip fuse, res block
dec2    64          48          bilinear up, skip fuse, res block
dec1    128         24          bilinear up, skip fuse, res block
head    128         2           1x1 conv
======  ==========  ==========  ==================================

The channel count is capped (``max_width``) rather than doubled indefinitely,
because in a small network the deepest stage would otherwise hold most of the
parameters while contributing the least spatial detail.

Two logits are emitted. Under a cross-entropy head they are ordinary class
logits; under the evidential head they are read as ``softplus`` evidence for a
2-class Dirichlet. Keeping the head shape identical is what lets the two
training ideologies share one architecture and be compared without confounds.
"""

from __future__ import annotations

import torch
from torch import nn

from evissl.models.blocks import (
    Down,
    GatedAxialAttention,
    ResidualSeparableBlock,
    SeparableConv,
    Up,
    group_norm,
)


class SepUNet(nn.Module):
    """Depthwise-separable U-Net with a gated axial bottleneck.

    Args:
        in_channels: Input image channels.
        num_classes: Output logits per pixel (2 for background/lesion).
        width: Stem width; stage widths are ``width * 2**i``, capped.
        depth: Number of downsampling stages.
        dropout: Spatial dropout inside residual blocks. Also the sampling
            source when Monte-Carlo dropout is used as an uncertainty baseline.
        axial_attention: Enable the gated axial bottleneck.
        max_width: Channel cap for deep stages.
        blocks_per_stage: Residual blocks after each downsample.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        width: int = 24,
        depth: int = 4,
        dropout: float = 0.10,
        axial_attention: bool = True,
        max_width: int = 192,
        blocks_per_stage: int = 2,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")

        widths = [min(width * 2**i, max_width) for i in range(depth + 1)]
        self.widths = widths
        self.depth = depth

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, widths[0], 3, padding=1, bias=False),
            group_norm(widths[0]),
            nn.SiLU(inplace=True),
        )

        self.encoders = nn.ModuleList(
            [
                Down(widths[i], widths[i + 1], n_blocks=blocks_per_stage, dropout=dropout)
                for i in range(depth)
            ]
        )

        neck_channels = widths[-1]
        neck: list[nn.Module] = []
        if axial_attention:
            neck.append(GatedAxialAttention(neck_channels))
        neck.append(ResidualSeparableBlock(neck_channels, dropout=dropout))
        self.neck = nn.Sequential(*neck)

        # Decoder mirrors the encoder; skip i comes from encoder output i.
        self.decoders = nn.ModuleList(
            [
                Up(widths[i + 1], widths[i], widths[i], n_blocks=1, dropout=dropout)
                for i in reversed(range(depth))
            ]
        )

        self.refine = SeparableConv(widths[0], widths[0])
        self.head = nn.Conv2d(widths[0], num_classes, 1)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.GroupNorm, nn.LayerNorm)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        # Start from a near-uniform prediction: a large initial bias magnitude
        # makes the evidential KL term fight the data for the first few epochs.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map an image batch to per-pixel logits of the same spatial size."""
        skips: list[torch.Tensor] = []
        h = self.stem(x)
        for encoder in self.encoders:
            skips.append(h)
            h = encoder(h)

        h = self.neck(h)

        for decoder, skip in zip(self.decoders, reversed(skips), strict=True):
            h = decoder(h, skip)

        return self.head(self.refine(h))

    def enable_mc_dropout(self) -> None:
        """Put only the dropout layers into training mode.

        Used by the Monte-Carlo-dropout uncertainty baseline: normalisation
        must stay in eval mode, otherwise the sampling noise is contaminated by
        changing normalisation statistics rather than measuring model spread.
        """
        self.eval()
        for module in self.modules():
            if isinstance(module, (nn.Dropout, nn.Dropout2d)):
                module.train()
