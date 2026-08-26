#!/usr/bin/env python
"""Train every method and compare them with paired statistical tests.

This is the script that produces the headline table. It trains the supervised
lower bound, Mean Teacher, FixMatch and the proposed evidential method through
one shared training loop, then runs paired Wilcoxon tests with Holm-Bonferroni
correction and writes both tables and figures.

Examples:
    python scripts/compare_methods.py
    python scripts/compare_methods.py --set optim.epochs=60 data.image_size=192
    python scripts/compare_methods.py --configs configs/fixmatch.yaml configs/evidential.yaml
"""

import sys

from evissl.cli import main

if __name__ == "__main__":
    sys.exit(main(["compare", "--figures", *sys.argv[1:]]))
