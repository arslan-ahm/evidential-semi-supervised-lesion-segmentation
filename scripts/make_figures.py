#!/usr/bin/env python
"""Regenerate the dataset preview and training-curve figures from disk.

Figures that need dense prediction maps (qualitative panels, sparsification,
uncertainty separation) are written by ``compare_methods.py`` instead, because
those maps are far too large to commit and must be recomputed from a
checkpoint.

Examples:
    python scripts/make_figures.py
    python scripts/make_figures.py --config configs/isic2018.yaml
"""

import argparse
import sys
from pathlib import Path

from evissl.config import load_config
from evissl.data import build_dataset
from evissl.report import regenerate_figures
from evissl.viz import plot_samples, save

DEFAULT_METHODS = [
    "configs/supervised_baseline.yaml",
    "configs/mean_teacher.yaml",
    "configs/fixmatch.yaml",
    "configs/evidential.yaml",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--configs", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--n-samples", type=int, default=8)
    parser.add_argument("--out-dir", default="results")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    written = []

    cfg = load_config(args.config)
    bundle = build_dataset(cfg.data, cfg.run.seed)
    written.append(
        save(
            plot_samples(
                bundle.train.images,
                bundle.train.masks,
                bundle.train.difficulty,
                n=args.n_samples,
                title=f"{cfg.data.name} samples ({cfg.data.image_size}px)",
            ),
            Path(args.out_dir) / "figures" / "dataset_samples.png",
        )
    )
    written += regenerate_figures(args.configs, args.out_dir)

    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
