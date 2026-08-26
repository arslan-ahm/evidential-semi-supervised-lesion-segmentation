#!/usr/bin/env python
"""Train one configuration.

Examples:
    python scripts/train.py --config configs/evidential.yaml
    python scripts/train.py --config configs/evidential.yaml --set optim.epochs=60
    python scripts/train.py --config configs/isic2018.yaml --set run.device=cuda
"""

import sys

from evissl.cli import main

if __name__ == "__main__":
    sys.exit(main(["train", *sys.argv[1:]]))
