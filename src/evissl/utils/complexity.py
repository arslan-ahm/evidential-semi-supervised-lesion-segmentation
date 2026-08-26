"""Model complexity accounting: parameters, multiply-accumulates, latency.

The efficiency claim in this repository is quantitative, so it is measured
rather than asserted. MACs are counted with forward hooks over the layer types
this project actually uses (Conv2d, ConvTranspose2d, Linear), which keeps the
count exact for these models and avoids a heavyweight profiler dependency.
"""

from __future__ import annotations

import statistics
import time
from typing import Any

import torch
from torch import nn


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """Total number of parameters in ``model``."""
    params = model.parameters()
    if trainable_only:
        return sum(p.numel() for p in params if p.requires_grad)
    return sum(p.numel() for p in params)


def _conv_macs(module: nn.Conv2d, output: torch.Tensor) -> int:
    """MACs for a Conv2d: one MAC per output element per input-channel tap."""
    out_elems = output.numel() // output.shape[0]  # per-sample
    kernel = module.kernel_size[0] * module.kernel_size[1]
    in_per_group = module.in_channels // module.groups
    return out_elems * kernel * in_per_group


def _deconv_macs(module: nn.ConvTranspose2d, output: torch.Tensor) -> int:
    out_elems = output.numel() // output.shape[0]
    kernel = module.kernel_size[0] * module.kernel_size[1]
    in_per_group = module.in_channels // module.groups
    return out_elems * kernel * in_per_group


def estimate_macs(
    model: nn.Module,
    input_shape: tuple[int, int, int] = (3, 128, 128),
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Count multiply-accumulates for a single forward pass.

    Args:
        model: Network to measure. Left in its original training mode.
        input_shape: ``(channels, height, width)`` of one sample.
        device: Device to run the probe forward on.

    Returns:
        Dict with ``macs`` (int), ``gmacs`` (float), ``params`` (int),
        ``params_m`` (float) and ``per_layer`` (dict of module name to MACs).
    """
    device = torch.device(device)
    was_training = model.training
    model.eval().to(device)

    per_layer: dict[str, int] = {}
    handles: list[Any] = []

    def make_hook(name: str):
        def hook(module: nn.Module, _inputs: Any, output: Any) -> None:
            if isinstance(output, (tuple, list)):
                output = output[0]
            if not isinstance(output, torch.Tensor):
                return
            if isinstance(module, nn.Conv2d):
                per_layer[name] = per_layer.get(name, 0) + _conv_macs(module, output)
            elif isinstance(module, nn.ConvTranspose2d):
                per_layer[name] = per_layer.get(name, 0) + _deconv_macs(module, output)
            elif isinstance(module, nn.Linear):
                elems = output.numel() // output.shape[0]
                per_layer[name] = per_layer.get(name, 0) + elems * module.in_features

        return hook

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            handles.append(module.register_forward_hook(make_hook(name)))

    try:
        with torch.no_grad():
            model(torch.zeros(1, *input_shape, device=device))
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)

    total = sum(per_layer.values())
    params = count_parameters(model)
    return {
        "macs": total,
        "gmacs": total / 1e9,
        "params": params,
        "params_m": params / 1e6,
        "per_layer": per_layer,
    }


def measure_latency(
    model: nn.Module,
    input_shape: tuple[int, int, int] = (3, 128, 128),
    batch_size: int = 1,
    warmup: int = 8,
    repeats: int = 25,
    device: str | torch.device = "cpu",
) -> dict[str, float]:
    """Time a forward pass, reporting median and inter-quartile spread.

    Median rather than mean, because CPU timings on a shared machine are
    heavy-tailed and the mean tracks the outliers.

    ``warmup`` defaults to 8 deliberately. PyTorch selects and caches a
    convolution algorithm per input shape on first use, and for the narrow
    depthwise convolutions in :class:`~evissl.models.SepUNet` that first call
    can cost several times the steady-state figure. Measured with only three
    warm-up iterations, the smallest model in this repository appeared 5x slower
    than it is - and reported as *slower than a model 8x its size*. Anything
    below roughly eight iterations is not measuring steady state.

    Returns:
        Dict with ``median_ms``, ``p25_ms``, ``p75_ms``, ``min_ms`` and
        ``throughput_ips`` (images per second at the median).
    """
    device = torch.device(device)
    was_training = model.training
    model.eval().to(device)
    batch = torch.zeros(batch_size, *input_shape, device=device)

    try:
        with torch.no_grad():
            for _ in range(warmup):
                model(batch)
            if device.type == "cuda":
                torch.cuda.synchronize()

            samples: list[float] = []
            for _ in range(repeats):
                start = time.perf_counter()
                model(batch)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                samples.append((time.perf_counter() - start) * 1000.0)
    finally:
        model.train(was_training)

    samples.sort()
    median = statistics.median(samples)
    return {
        "median_ms": median,
        "p25_ms": samples[len(samples) // 4],
        "p75_ms": samples[(3 * len(samples)) // 4],
        "min_ms": samples[0],
        "throughput_ips": (batch_size * 1000.0 / median) if median > 0 else float("inf"),
    }
