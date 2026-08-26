"""Reference U-Net baseline.

A faithful Ronneberger et al. (2015) U-Net with BatchNorm added, at the standard
64-128-256-512-1024 widths. This is the ~31M-parameter model that dermoscopy
segmentation papers - including the prior work this project builds on - use by
default. It exists here so the efficiency comparison is measured against a real
implementation under identical data, schedule and evaluation code, rather than
against numbers quoted from another paper.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from evissl.models.blocks import DoubleConv


class UNet(nn.Module):
    """Classic U-Net with transposed-convolution upsampling.

    Args:
        in_channels: Input image channels.
        num_classes: Output logits per pixel.
        width: Base width; stages are ``width * 2**i``.
        depth: Number of downsampling stages.
        dropout: Dropout applied at the bottleneck only, as in the original.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        width: int = 64,
        depth: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        widths = [width * 2**i for i in range(depth + 1)]
        self.depth = depth
        self.widths = widths

        self.inc = DoubleConv(in_channels, widths[0])
        self.downs = nn.ModuleList(
            [DoubleConv(widths[i], widths[i + 1]) for i in range(depth)]
        )
        self.pool = nn.MaxPool2d(2)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        self.ups = nn.ModuleList(
            [
                nn.ConvTranspose2d(widths[i + 1], widths[i], 2, stride=2)
                for i in reversed(range(depth))
            ]
        )
        self.up_convs = nn.ModuleList(
            [DoubleConv(widths[i] * 2, widths[i]) for i in reversed(range(depth))]
        )
        self.head = nn.Conv2d(widths[0], num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: list[torch.Tensor] = []
        h = self.inc(x)
        for down in self.downs:
            skips.append(h)
            h = down(self.pool(h))
        h = self.drop(h)

        for up, conv, skip in zip(self.ups, self.up_convs, reversed(skips), strict=True):
            h = up(h)
            # Odd input sizes leave a one-pixel mismatch; pad rather than crop
            # so the output keeps the input resolution exactly.
            if h.shape[-2:] != skip.shape[-2:]:
                dy = skip.shape[-2] - h.shape[-2]
                dx = skip.shape[-1] - h.shape[-1]
                h = F.pad(h, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
            h = conv(torch.cat([skip, h], dim=1))

        return self.head(h)

    def enable_mc_dropout(self) -> None:
        """Put only dropout layers into training mode (see :class:`SepUNet`)."""
        self.eval()
        for module in self.modules():
            if isinstance(module, (nn.Dropout, nn.Dropout2d)):
                module.train()
