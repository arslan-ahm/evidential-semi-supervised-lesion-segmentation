"""Building blocks for the efficient segmentation backbone.

Three deliberate choices are worth calling out, because each one is a response
to a specific failure mode rather than a stylistic preference:

**GroupNorm, never BatchNorm.** Two independent reasons. (1) Semi-supervised
runs here train on 8-16 images per step on CPU, where batch statistics are too
noisy to be useful. (2) An EMA teacher averages *parameters*; BatchNorm running
statistics are buffers, not parameters, so they either leak the student's
statistics into the teacher or go stale. GroupNorm has no running state and the
whole class of bug disappears.

**Depthwise-separable convolutions.** A ``k x k`` convolution from ``C_in`` to
``C_out`` costs ``k^2 * C_in * C_out`` parameters; splitting it into a depthwise
``k x k`` plus a pointwise ``1 x 1`` costs ``k^2 * C_in + C_in * C_out``. At the
widths used here that is roughly an 8x reduction with a small accuracy cost,
which the wider receptive field from the axial bottleneck more than repays.

**Zero-initialised gates.** The attention and squeeze-excite branches are
multiplied by a parameter initialised at zero, so at step 0 the network is
exactly the plain convolutional net and the extra branches have to earn their
contribution. This removes the warm-up instability that otherwise shows up when
attention is dropped into a small network.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def group_norm(channels: int, max_groups: int = 8) -> nn.GroupNorm:
    """GroupNorm with a group count that always divides ``channels``."""
    groups = max_groups
    while groups > 1 and channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class SeparableConv(nn.Module):
    """Depthwise ``k x k`` followed by pointwise ``1 x 1``, normed and activated."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        activate: bool = True,
    ) -> None:
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.norm = group_norm(out_channels)
        self.act = nn.SiLU(inplace=True) if activate else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.pointwise(self.depthwise(x))))


class SqueezeExcite(nn.Module):
    """Channel recalibration with a zero-initialised gate.

    Standard SE multiplies features by ``sigmoid(...)``, which starts at 0.5 and
    halves the signal. Here the gate is ``1 + gamma * (sigmoid(...) - 0.5)``
    with ``gamma`` starting at zero, so the block starts as the identity.
    """

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(4, channels // reduction)
        self.fc1 = nn.Conv2d(channels, hidden, 1)
        self.fc2 = nn.Conv2d(hidden, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.sigmoid(self.fc2(F.silu(self.fc1(x.mean(dim=(2, 3), keepdim=True)))))
        return x * (1.0 + self.gamma * (scale - 0.5))


class ResidualSeparableBlock(nn.Module):
    """Two separable convolutions with a residual connection and SE gate."""

    def __init__(self, channels: int, dilation: int = 1, dropout: float = 0.0) -> None:
        super().__init__()
        self.conv1 = SeparableConv(channels, channels, dilation=dilation)
        self.conv2 = SeparableConv(channels, channels, dilation=dilation, activate=False)
        self.se = SqueezeExcite(channels)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv2(self.conv1(x))
        h = self.se(self.drop(h))
        return self.act(x + h)


class GatedAxialAttention(nn.Module):
    """Global context at the bottleneck, factorised over rows then columns.

    Full self-attention over an ``H x W`` map costs ``O((HW)^2)``. Attending
    along the two axes in sequence costs ``O(HW * (H + W))`` while still
    connecting any two positions in two hops (Ho et al., 2019). Channels are
    projected down by ``reduction`` first, which is where most of the parameter
    saving comes from.

    The output is added through a zero-initialised per-channel gate, so the
    module is a no-op at initialisation.
    """

    def __init__(self, channels: int, heads: int = 4, reduction: int = 2) -> None:
        super().__init__()
        inner = max(heads, (channels // reduction) // heads * heads)
        self.inner = inner
        self.proj_in = nn.Conv2d(channels, inner, 1, bias=False)
        self.norm = nn.LayerNorm(inner)
        self.attn_rows = nn.MultiheadAttention(inner, heads, batch_first=True)
        self.attn_cols = nn.MultiheadAttention(inner, heads, batch_first=True)
        self.proj_out = nn.Conv2d(inner, channels, 1, bias=False)
        self.gamma = nn.Parameter(torch.zeros(channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        z = self.proj_in(x)

        # --- attend along rows: sequence dimension is W, batched over (B, H) --
        seq = z.permute(0, 2, 3, 1).reshape(b * h, w, self.inner)
        seq = self.norm(seq)
        seq = seq + self.attn_rows(seq, seq, seq, need_weights=False)[0]
        z = seq.reshape(b, h, w, self.inner).permute(0, 3, 1, 2)

        # --- attend along columns: sequence dimension is H, batched over (B, W)
        seq = z.permute(0, 3, 2, 1).reshape(b * w, h, self.inner)
        seq = self.norm(seq)
        seq = seq + self.attn_cols(seq, seq, seq, need_weights=False)[0]
        z = seq.reshape(b, w, h, self.inner).permute(0, 3, 2, 1)

        return x + self.gamma * self.proj_out(z)


class Down(nn.Module):
    """Strided separable downsample followed by ``n_blocks`` residual blocks."""

    def __init__(
        self, in_channels: int, out_channels: int, n_blocks: int = 2, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.reduce = SeparableConv(in_channels, out_channels, stride=2)
        self.blocks = nn.Sequential(
            *[ResidualSeparableBlock(out_channels, dropout=dropout) for _ in range(n_blocks)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.reduce(x))


class Up(nn.Module):
    """Bilinear upsample, skip concatenation, then separable fusion.

    Bilinear upsampling plus a separable fuse is preferred over
    ``ConvTranspose2d``: it has no checkerboard artefacts and costs far fewer
    parameters, which matters for a sub-1M-parameter budget.
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        n_blocks: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.reduce = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.fuse = SeparableConv(out_channels + skip_channels, out_channels)
        self.blocks = nn.Sequential(
            *[ResidualSeparableBlock(out_channels, dropout=dropout) for _ in range(n_blocks)]
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.reduce(x)
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.blocks(self.fuse(torch.cat([x, skip], dim=1)))


class DoubleConv(nn.Module):
    """Plain ``conv-BN-ReLU`` pair - the original U-Net block, for the baseline."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)
