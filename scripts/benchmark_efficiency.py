#!/usr/bin/env python
"""Measure parameters, MACs and latency for every architecture.

Substantiates the cost half of the efficiency claim independently of any
accuracy number. Also reports the cost of the *uncertainty itself*, which is a
separate and often-ignored axis: the evidential head produces its
epistemic/aleatoric split from a single forward pass, whereas MC dropout needs
one pass per sample and TTA one per augmentation.

Examples:
    python scripts/benchmark_efficiency.py
    python scripts/benchmark_efficiency.py --image-size 256 --repeats 40
"""

import argparse
import sys
from pathlib import Path

import torch

from evissl.config import ModelConfig
from evissl.models import build_model
from evissl.pipelines import run_efficiency_benchmark
from evissl.utils.complexity import measure_latency
from evissl.utils.seed import resolve_device


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument(
        "--models", nargs="+", default=["separable_unet_tiny", "separable_unet", "unet"]
    )
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--mc-samples", type=int, default=8)
    parser.add_argument("--out-dir", default="results")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    frame = run_efficiency_benchmark(
        tuple(args.models), args.image_size, out_dir=args.out_dir, repeats=args.repeats
    )
    print()
    print(frame.to_string(index=False))

    # Cost of obtaining an uncertainty estimate - the practical argument for
    # the evidential head over sampling-based alternatives.
    device = resolve_device("auto")
    model = build_model(ModelConfig(name="separable_unet"))
    shape = (3, args.image_size, args.image_size)
    single = measure_latency(model, shape, repeats=args.repeats, device=device)["median_ms"]

    print()
    print("Cost of an uncertainty estimate (single image, separable_unet):")
    print(f"  evidential head, 1 forward pass  : {single:8.2f} ms   (1.0x)")
    print(
        f"  MC dropout, {args.mc_samples} forward passes    : "
        f"{single * args.mc_samples:8.2f} ms   ({float(args.mc_samples):.1f}x)"
    )
    print(f"  4-flip TTA, 4 forward passes     : {single * 4:8.2f} ms   (4.0x)")
    print()
    print(f"torch {torch.__version__}, device {device}, threads {torch.get_num_threads()}")
    print(f"table: {Path(args.out_dir) / 'tables' / 'efficiency.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
