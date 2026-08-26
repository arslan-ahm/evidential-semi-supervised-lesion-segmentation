#!/usr/bin/env python
"""Component ablation plus the labelled-fraction sweep.

Two studies, both needed to support the claim:

* **Component ablation** - remove one mechanism at a time (dissonance
  tempering, the vacuity gate, the KL regulariser, axial attention, the
  evidential head) so each contribution is attributable rather than bundled.
* **Labelled-fraction sweep** - re-train every method at 5/10/20/50% labels
  with the gradient-step count held fixed, to show where the gain actually
  lives. A method that only helps at 50% labels is not a semi-supervised
  method, it is a regulariser.

Examples:
    python scripts/run_ablation.py
    python scripts/run_ablation.py --skip-sweep
    python scripts/run_ablation.py --fractions 0.05 0.10 --set optim.epochs=40
"""

import argparse
import sys
from pathlib import Path

from evissl.pipelines import (
    compare_ablation_variants,
    run_component_ablation,
    run_labeled_fraction_sweep,
)
from evissl.report import write_sweep_figure

DEFAULT_METHODS = [
    "configs/supervised_baseline.yaml",
    "configs/fixmatch.yaml",
    "configs/evidential.yaml",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="configs/evidential.yaml")
    parser.add_argument("--configs", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--fractions", nargs="+", type=float, default=[0.05, 0.10, 0.20, 0.50])
    parser.add_argument("--skip-components", action="store_true")
    parser.add_argument("--skip-sweep", action="store_true")
    parser.add_argument("--set", dest="overrides", nargs="*", default=[])
    parser.add_argument("--out-dir", default="results")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.skip_components:
        print("=" * 72)
        print("component ablation")
        print("=" * 72)
        frame = run_component_ablation(args.config, args.overrides, args.out_dir)
        print(frame.to_string(index=False))

        # Raw deltas are not interpretable without this: the standard error of a
        # mean Dice here is ~0.013, so a 0.005 "contribution" is noise.
        print()
        print("paired tests against the full method (Holm-corrected):")
        stats = compare_ablation_variants(args.out_dir)
        if stats.empty:
            print("  (no per-image CSVs found)")
        else:
            print(stats.to_string(index=False))

    if not args.skip_sweep:
        print()
        print("=" * 72)
        print("labelled-fraction sweep")
        print("=" * 72)
        sweep = run_labeled_fraction_sweep(
            args.configs, tuple(args.fractions), args.overrides, args.out_dir
        )
        print(sweep["table"].to_string(index=False))
        print(write_sweep_figure(sweep, Path(args.out_dir)))

    return 0


if __name__ == "__main__":
    sys.exit(main())
