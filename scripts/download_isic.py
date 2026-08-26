#!/usr/bin/env python
"""Prepare the ISIC 2018 Task 1 archive for this repository.

ISIC requires accepting terms of use, so the archive cannot be fetched
non-interactively and this script does not pretend otherwise. It prints the
retrieval steps, then - once the ZIPs are present - extracts and arranges them
into the layout the data loader expects, and verifies the result.

Layout produced::

    data/isic2018/
      images/  ISIC_0000000.jpg ...
      masks/   ISIC_0000000_segmentation.png ...

Usage:
    python scripts/download_isic.py                 # print instructions, check state
    python scripts/download_isic.py --extract       # extract any ZIPs found
    python scripts/download_isic.py --verify        # count and pair-check only
"""

import argparse
import sys
import zipfile
from pathlib import Path

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")

INSTRUCTIONS = """
ISIC 2018 Task 1 (lesion boundary segmentation) - manual retrieval required
===========================================================================

1. Open https://challenge.isic-archive.com/data/#2018 and accept the terms.

2. Download these two archives from the Task 1 section:
     ISIC2018_Task1-2_Training_Input.zip     (~10.4 GB, 2594 dermoscopic images)
     ISIC2018_Task1_Training_GroundTruth.zip (~27 MB, 2594 binary masks)

3. Put both ZIPs in:
     {target}

4. Run:
     python scripts/download_isic.py --extract

   This flattens them into {target}/images and {target}/masks, which is the
   layout configs/isic2018.yaml expects. The first training run resizes and
   caches the set to data/isic2018/cache_<size>.npz, so the slow decode happens
   exactly once.

Alternative: the PH2 archive (200 images) is much smaller and is supported as an
external test set via `data.name=ph2`. Same layout under data/ph2/.

Nothing here needs the full archive: `data.name=synthetic` (the default) runs
the entire pipeline offline, and that is what the committed results use.
"""


def print_instructions(target: Path) -> None:
    print(INSTRUCTIONS.format(target=target.resolve()))


def extract(target: Path) -> int:
    """Extract any ISIC ZIPs in ``target`` into ``images/`` and ``masks/``."""
    zips = sorted(target.glob("*.zip"))
    if not zips:
        print(f"No .zip files found in {target.resolve()}")
        return 1

    images_dir = target / "images"
    masks_dir = target / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    for archive_path in zips:
        # Ground-truth archives are identified by name; everything else is input.
        name = archive_path.name.lower()
        destination = masks_dir if "groundtruth" in name or "mask" in name else images_dir
        print(f"extracting {archive_path.name} -> {destination}")

        with zipfile.ZipFile(archive_path) as archive:
            written = 0
            for member in archive.infolist():
                if member.is_dir():
                    continue
                member_name = Path(member.filename).name
                if not member_name or Path(member_name).suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                # Flatten: the archives nest one directory deep, and the loader
                # pairs by file stem rather than by directory structure.
                out_path = destination / member_name
                if out_path.exists():
                    continue
                with archive.open(member) as source, out_path.open("wb") as sink:
                    sink.write(source.read())
                written += 1
            print(f"  wrote {written} file(s)")

    return verify(target)


def verify(target: Path) -> int:
    """Report counts and how many image/mask pairs match by stem."""
    images_dir, masks_dir = target / "images", target / "masks"
    if not images_dir.is_dir() or not masks_dir.is_dir():
        print(f"Missing {images_dir} or {masks_dir}. Run with --extract first.")
        return 1

    def listing(directory: Path) -> list[Path]:
        return [p for p in sorted(directory.iterdir()) if p.suffix.lower() in IMAGE_SUFFIXES]

    images, masks = listing(images_dir), listing(masks_dir)
    mask_stems = {p.stem for p in masks}
    paired = sum(
        1
        for image in images
        if image.stem in mask_stems
        or f"{image.stem}_segmentation" in mask_stems
        or f"{image.stem}_lesion" in mask_stems
    )

    print(f"images : {len(images)}")
    print(f"masks  : {len(masks)}")
    print(f"paired : {paired}")
    if paired == 0:
        print("\nNo pairs matched. Expected masks named <stem>, <stem>_segmentation")
        print("or <stem>_lesion alongside each image.")
        return 1

    print("\nReady. Train with:\n  python scripts/train.py --config configs/isic2018.yaml")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", default="data", help="dataset root directory")
    parser.add_argument("--name", default="isic2018", help="subdirectory name")
    parser.add_argument("--extract", action="store_true", help="extract ZIPs found in the target")
    parser.add_argument("--verify", action="store_true", help="check counts and pairing only")
    args = parser.parse_args(argv)

    target = Path(args.root) / args.name
    target.mkdir(parents=True, exist_ok=True)

    if args.extract:
        return extract(target)
    if args.verify:
        return verify(target)

    print_instructions(target)
    if (target / "images").is_dir():
        print("Current state:")
        verify(target)
    return 0


if __name__ == "__main__":
    sys.exit(main())
