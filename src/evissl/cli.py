"""Command-line interface.

``evissl <subcommand>`` mirrors the scripts in ``scripts/`` - the scripts are
thin wrappers around these functions, so there is exactly one implementation of
each operation regardless of how it is invoked.

Examples:
    evissl train --config configs/evidential.yaml
    evissl train --config configs/evidential.yaml --set optim.epochs=60
    evissl compare --configs configs/supervised_baseline.yaml configs/evidential.yaml
    evissl sweep --fractions 0.05 0.10 0.20
    evissl ablate
    evissl bench --image-size 256
    evissl figures
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_METHODS: tuple[str, ...] = (
    "configs/supervised_baseline.yaml",
    "configs/mean_teacher.yaml",
    "configs/fixmatch.yaml",
    "configs/evidential.yaml",
)


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--set",
        dest="overrides",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="override any config leaf, e.g. --set optim.epochs=60 data.image_size=256",
    )
    parser.add_argument(
        "--out-dir", default="results", help="root directory for tables and figures"
    )


def build_parser() -> argparse.ArgumentParser:
    """Construct the full argument parser."""
    parser = argparse.ArgumentParser(
        prog="evissl",
        description="Evidential semi-supervised lesion segmentation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="train one configuration")
    train.add_argument("--config", required=True, help="path to a YAML config")
    train.add_argument(
        "--no-test", action="store_true", help="skip test evaluation (keeps the test set unseen)"
    )
    _add_common(train)

    compare = subparsers.add_parser(
        "compare", help="train several methods and run paired statistical tests"
    )
    compare.add_argument("--configs", nargs="+", default=list(DEFAULT_METHODS))
    compare.add_argument("--baseline", default="supervised_baseline")
    compare.add_argument(
        "--figures", action="store_true", help="also write comparison figures"
    )
    _add_common(compare)

    sweep = subparsers.add_parser("sweep", help="labelled-fraction sweep")
    sweep.add_argument("--configs", nargs="+", default=list(DEFAULT_METHODS))
    sweep.add_argument("--fractions", nargs="+", type=float, default=[0.05, 0.10, 0.20, 0.50])
    _add_common(sweep)

    ablate = subparsers.add_parser("ablate", help="component ablation of the proposed method")
    ablate.add_argument("--config", default="configs/evidential.yaml")
    _add_common(ablate)

    bench = subparsers.add_parser("bench", help="parameters, MACs and latency per architecture")
    bench.add_argument("--image-size", type=int, default=128)
    bench.add_argument("--models", nargs="+",
                       default=["separable_unet_tiny", "separable_unet", "unet"])
    bench.add_argument("--repeats", type=int, default=20)
    _add_common(bench)

    figures = subparsers.add_parser(
        "figures", help="regenerate figures from existing run artefacts"
    )
    figures.add_argument("--configs", nargs="+", default=list(DEFAULT_METHODS))
    _add_common(figures)

    data = subparsers.add_parser("data", help="materialise and preview the dataset")
    data.add_argument("--config", default="configs/base.yaml")
    data.add_argument("--n", type=int, default=8, help="samples to draw in the preview figure")
    _add_common(data)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    args = build_parser().parse_args(argv)

    # Imported lazily so `evissl --help` does not pay for loading torch.
    import pandas as pd

    from evissl.config import load_config
    from evissl.pipelines import (
        compare_ablation_variants,
        format_table,
        run_comparison,
        run_component_ablation,
        run_efficiency_benchmark,
        run_labeled_fraction_sweep,
        run_training,
    )

    if args.command == "train":
        cfg = load_config(args.config, args.overrides)
        out = run_training(cfg, evaluate_test=not args.no_test)
        result = out["result"]
        if result is not None:
            print()
            print(format_table(pd.DataFrame([result.summary_row()])))
        return 0

    if args.command == "compare":
        out = run_comparison(
            args.configs, args.overrides, baseline=args.baseline, out_dir=args.out_dir,
            keep_predictions=args.figures,
        )
        print()
        print(format_table(out["table"]))
        for metric, comparisons in out["comparisons"].items():
            print(f"\npaired tests on {metric} (Holm-corrected):")
            for c in comparisons:
                print("  " + c.summary())
        if args.figures:
            from evissl.report import write_comparison_figures

            written = write_comparison_figures(out, Path(args.out_dir))
            print("\nfigures:")
            for path in written:
                print(f"  {path}")
        return 0

    if args.command == "sweep":
        out = run_labeled_fraction_sweep(
            args.configs, tuple(args.fractions), args.overrides, args.out_dir
        )
        print()
        print(out["table"].to_string(index=False))
        return 0

    if args.command == "ablate":
        frame = run_component_ablation(args.config, args.overrides, args.out_dir)
        print()
        print(frame.to_string(index=False))
        stats = compare_ablation_variants(args.out_dir)
        if not stats.empty:
            print()
            print("paired tests against the full method (Holm-corrected):")
            print(stats.to_string(index=False))
        return 0

    if args.command == "bench":
        frame = run_efficiency_benchmark(
            tuple(args.models), args.image_size, out_dir=args.out_dir, repeats=args.repeats
        )
        print()
        print(frame.to_string(index=False))
        return 0

    if args.command == "figures":
        from evissl.report import regenerate_figures

        written = regenerate_figures(args.configs, args.out_dir)
        for path in written:
            print(path)
        return 0

    if args.command == "data":
        from evissl.data import build_dataset
        from evissl.viz import plot_samples, save

        cfg = load_config(args.config, args.overrides)
        bundle = build_dataset(cfg.data, cfg.run.seed)
        print(bundle.describe())
        figure = plot_samples(
            bundle.train.images, bundle.train.masks, bundle.train.difficulty, n=args.n
        )
        print(save(figure, Path(args.out_dir) / "figures" / "dataset_samples.png"))
        return 0

    return 1  # pragma: no cover - argparse enforces a valid subcommand


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
