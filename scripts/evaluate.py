#!/usr/bin/env python
"""Re-evaluate a saved checkpoint without re-training.

Useful for scoring a Colab-trained checkpoint locally, or for re-scoring with
different evaluation options (TTA, MC dropout, a different decision threshold)
without touching the trained weights.

Examples:
    python scripts/evaluate.py --config configs/evidential.yaml
    python scripts/evaluate.py --config configs/evidential.yaml --split val
    python scripts/evaluate.py --config configs/evidential.yaml --set eval.tta=true
    python scripts/evaluate.py --config configs/evidential.yaml --set eval.mc_dropout=8
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from evissl.config import load_config
from evissl.data import build_dataset, build_loaders
from evissl.eval import evaluate
from evissl.models import build_model
from evissl.pipelines import format_table
from evissl.utils.checkpoint import load_checkpoint
from evissl.utils.complexity import estimate_macs, measure_latency
from evissl.utils.logging import get_logger
from evissl.utils.seed import resolve_device, seed_everything

logger = get_logger("evissl.evaluate")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--checkpoint", default=None, help="defaults to <ckpt_dir>/<run.name>.pt"
    )
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument(
        "--student",
        action="store_true",
        help="evaluate the student weights instead of the EMA teacher",
    )
    parser.add_argument("--set", dest="overrides", nargs="*", default=[])
    parser.add_argument("--out-dir", default="results")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config, args.overrides)

    generator = seed_everything(cfg.run.seed, cfg.run.deterministic)
    device = resolve_device(cfg.run.device)

    bundle = build_dataset(cfg.data, cfg.run.seed)
    loaders = build_loaders(cfg, bundle, generator=generator)
    model = build_model(cfg.model)

    ckpt_path = Path(args.checkpoint or Path(cfg.run.ckpt_dir) / f"{cfg.run.name}.pt")
    if not ckpt_path.is_file():
        logger.error("checkpoint not found: %s (train the run first)", ckpt_path)
        return 2

    payload = load_checkpoint(
        ckpt_path, model, prefer_ema=not args.student, map_location=device
    )
    logger.info(
        "loaded %s from %s (epoch %s)", payload["loaded"], ckpt_path, payload["epoch"]
    )

    image_shape = (cfg.model.in_channels, cfg.data.image_size, cfg.data.image_size)
    complexity = estimate_macs(model, image_shape, device)
    latency = measure_latency(model, image_shape, device=device)

    loader = loaders.test if args.split == "test" else loaders.val
    result = evaluate(
        model,
        loader,
        cfg,
        name=f"{cfg.run.name}_{args.split}",
        device=device,
        extra={
            "split": args.split,
            "params_m": round(complexity["params_m"], 4),
            "gmacs": round(complexity["gmacs"], 4),
            "latency_ms": round(latency["median_ms"], 3),
            "weights": payload["loaded"],
            "tta": cfg.eval.tta,
            "mc_dropout": cfg.eval.mc_dropout,
        },
    )
    paths = result.save(Path(args.out_dir) / "runs" / result.name)

    print()
    print(format_table(pd.DataFrame([result.summary_row()])))
    print()
    for key, path in paths.items():
        print(f"{key}: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
