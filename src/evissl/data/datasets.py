"""Dataset construction: splits, labelled/unlabelled partition, torch wrappers.

Three sources are supported behind one interface:

``synthetic``
    :mod:`evissl.data.synthetic`. No download, fully reproducible, and the
    default so the repository is verifiable out of the box.
``isic2018``
    ISIC 2018 Task 1 (lesion boundary segmentation). Expects images and masks
    already extracted; see ``scripts/download_isic.py``.
``ph2``
    The PH2 dermoscopic archive, used as an *external* test set to measure
    cross-dataset generalisation.

The split protocol is fixed and seeded: images are shuffled once with the run
seed, then partitioned train/val/test. The labelled subset is drawn from the
*head* of the shuffled training order, so increasing ``labeled_fraction``
strictly adds images to the previous labelled set. That nesting is what makes
the labelled-fraction sweep in the ablation a controlled comparison instead of
a set of unrelated draws.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from evissl.config import DataConfig
from evissl.data.synthetic import generate_dataset
from evissl.data.transforms import (
    AugmentConfig,
    apply_cutout,
    apply_geometric,
    apply_strong_photometric,
    apply_weak_photometric,
    mask_to_tensor,
    to_tensor,
)

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


# --------------------------------------------------------------------------- #
# Bundle
# --------------------------------------------------------------------------- #


@dataclass
class Split:
    """One partition of a dataset, held in memory as uint8 arrays."""

    images: np.ndarray  # (N, H, W, 3) uint8
    masks: np.ndarray  # (N, H, W)    uint8 in {0, 1}
    #: Optional per-sample difficulty in [0, 1] (synthetic data only).
    difficulty: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.images)

    def subset(self, idx: np.ndarray) -> Split:
        return Split(
            images=self.images[idx],
            masks=self.masks[idx],
            difficulty=None if self.difficulty is None else self.difficulty[idx],
        )


@dataclass
class DatasetBundle:
    """Everything a training run needs from the data layer."""

    train: Split
    val: Split
    test: Split
    #: Indices into ``train`` that carry usable annotations.
    labeled_idx: np.ndarray
    #: Indices into ``train`` whose masks are withheld from the loss.
    unlabeled_idx: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)

    def describe(self) -> dict[str, Any]:
        """Compact summary for logs and the results table."""
        lesion_px = float(self.train.masks.mean())
        return {
            "dataset": self.meta.get("name", "unknown"),
            "image_size": int(self.train.images.shape[1]),
            "n_train": len(self.train),
            "n_labeled": int(len(self.labeled_idx)),
            "n_unlabeled": int(len(self.unlabeled_idx)),
            "n_val": len(self.val),
            "n_test": len(self.test),
            "lesion_pixel_fraction": round(lesion_px, 4),
        }


# --------------------------------------------------------------------------- #
# Sources
# --------------------------------------------------------------------------- #


def _load_synthetic(cfg: DataConfig, seed: int) -> tuple[Split, Split, Split, dict[str, Any]]:
    size = cfg.image_size
    # Distinct base seeds guarantee disjoint sample streams across splits.
    tr_i, tr_m, tr_p = generate_dataset(cfg.train_size, size, seed=seed * 7 + 1)
    va_i, va_m, va_p = generate_dataset(cfg.val_size, size, seed=seed * 7 + 2)
    te_i, te_m, te_p = generate_dataset(cfg.test_size, size, seed=seed * 7 + 3)

    def diff(params: list) -> np.ndarray:
        return np.array([p.difficulty() for p in params], dtype=np.float32)

    return (
        Split(tr_i, tr_m, diff(tr_p)),
        Split(va_i, va_m, diff(va_p)),
        Split(te_i, te_m, diff(te_p)),
        {"name": "synthetic", "generator_seed": seed},
    )


def _read_pair(image_path: Path, mask_path: Path, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Read and resize one image/mask pair."""
    image = Image.open(image_path).convert("RGB").resize((size, size), Image.BILINEAR)
    # Nearest-neighbour for the mask: bilinear would invent intermediate labels.
    mask = Image.open(mask_path).convert("L").resize((size, size), Image.NEAREST)
    return (
        np.asarray(image, dtype=np.uint8),
        (np.asarray(mask) > 127).astype(np.uint8),
    )


def _index_directory(image_dir: Path, mask_dir: Path) -> list[tuple[Path, Path]]:
    """Pair image files with masks by stem, tolerating common suffix schemes.

    ISIC ships masks as ``<stem>_segmentation.png``; PH2 uses ``<stem>_lesion``.
    Both, plus the plain ``<stem>`` case, are matched here.
    """
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not mask_dir.is_dir():
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

    masks_by_stem: dict[str, Path] = {}
    for path in sorted(mask_dir.rglob("*")):
        if path.suffix.lower() in IMAGE_SUFFIXES:
            masks_by_stem.setdefault(path.stem, path)

    pairs: list[tuple[Path, Path]] = []
    missing = 0
    for image_path in sorted(image_dir.rglob("*")):
        if image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        stem = image_path.stem
        candidates = (stem, f"{stem}_segmentation", f"{stem}_lesion", f"{stem}_Segmentation")
        mask_path = next((masks_by_stem[c] for c in candidates if c in masks_by_stem), None)
        if mask_path is None:
            missing += 1
            continue
        pairs.append((image_path, mask_path))

    if not pairs:
        raise FileNotFoundError(
            f"No image/mask pairs matched between {image_dir} and {mask_dir}. "
            "Expected masks named <stem>, <stem>_segmentation or <stem>_lesion."
        )
    if missing:
        print(f"[data] warning: {missing} image(s) had no matching mask and were skipped")
    return pairs


def _load_folder_dataset(
    name: str, cfg: DataConfig, seed: int
) -> tuple[Split, Split, Split, dict[str, Any]]:
    """Load a folder-based dataset, caching the resized arrays to ``.npz``."""
    root = Path(cfg.root) / name
    size = cfg.image_size
    cache_path = root / f"cache_{size}.npz"

    if cfg.cache and cache_path.is_file():
        blob = np.load(cache_path)
        images, masks = blob["images"], blob["masks"]
    else:
        pairs = _index_directory(root / "images", root / "masks")
        images = np.empty((len(pairs), size, size, 3), dtype=np.uint8)
        masks = np.empty((len(pairs), size, size), dtype=np.uint8)
        for i, (img_path, msk_path) in enumerate(pairs):
            images[i], masks[i] = _read_pair(img_path, msk_path, size)
        if cfg.cache:
            # Decoding and resizing 2594 JPEGs is the slowest step in a real-data
            # run; caching makes it a one-time cost.
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache_path, images=images, masks=masks)
            print(f"[data] cached {len(pairs)} pairs to {cache_path}")

    n = len(images)
    order = np.random.default_rng(seed).permutation(n)
    # 70 / 10 / 20 train / val / test, the convention used for ISIC 2018 Task 1
    # when the official test masks are unavailable.
    n_train = int(round(0.70 * n))
    n_val = int(round(0.10 * n))
    tr, va, te = order[:n_train], order[n_train : n_train + n_val], order[n_train + n_val :]

    return (
        Split(images[tr], masks[tr]),
        Split(images[va], masks[va]),
        Split(images[te], masks[te]),
        {"name": name, "n_total": int(n), "cache": str(cache_path)},
    )


# --------------------------------------------------------------------------- #
# Public builder
# --------------------------------------------------------------------------- #


def build_dataset(cfg: DataConfig, seed: int = 1337) -> DatasetBundle:
    """Build splits and the labelled/unlabelled partition from a config.

    Args:
        cfg: Data configuration.
        seed: Controls both synthetic generation and the split permutation.

    Returns:
        A :class:`DatasetBundle`.

    Raises:
        ValueError: if ``cfg.name`` is unknown or the labelled subset is empty.
    """
    name = cfg.name.strip().lower()
    if name == "synthetic":
        train, val, test, meta = _load_synthetic(cfg, seed)
    elif name in {"isic2018", "ph2"}:
        train, val, test, meta = _load_folder_dataset(name, cfg, seed)
    else:
        raise ValueError(f"Unknown dataset {cfg.name!r}; expected synthetic, isic2018 or ph2")

    n_train = len(train)
    if cfg.labeled_count is not None:
        n_labeled = int(cfg.labeled_count)
    else:
        n_labeled = int(round(cfg.labeled_fraction * n_train))
    n_labeled = max(1, min(n_labeled, n_train))

    # Nested labelled sets: a fixed shuffle, then take the head. See module docstring.
    order = np.random.default_rng(seed + 991).permutation(n_train)
    labeled_idx = np.sort(order[:n_labeled])
    unlabeled_idx = np.sort(order[n_labeled:])

    meta = {**meta, "labeled_fraction_effective": n_labeled / max(n_train, 1)}
    return DatasetBundle(train, val, test, labeled_idx, unlabeled_idx, meta)


# --------------------------------------------------------------------------- #
# torch Datasets
# --------------------------------------------------------------------------- #


class LabeledDataset(Dataset):
    """Labelled images with augmentation, for the supervised loss."""

    def __init__(
        self,
        split: Split,
        indices: np.ndarray | None = None,
        augment: AugmentConfig | None = None,
        train: bool = True,
        seed: int = 0,
    ) -> None:
        self.split = split
        self.indices = np.arange(len(split)) if indices is None else np.asarray(indices)
        self.augment = augment or AugmentConfig()
        self.train = train
        self.seed = seed
        #: Bumped by the trainer each epoch so augmentation differs but stays
        #: reproducible: the RNG stream is a pure function of (seed, epoch, i).
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.indices)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        idx = int(self.indices[i])
        image = self.split.images[idx]
        mask = self.split.masks[idx]

        if self.train:
            rng = np.random.default_rng((self.seed, self.epoch, idx))
            image, mask = apply_geometric(image, mask, rng, self.augment)
            image = apply_weak_photometric(image, rng, self.augment)

        item = {
            "image": to_tensor(image),
            "mask": mask_to_tensor(mask),
            "index": torch.tensor(idx, dtype=torch.long),
        }
        if self.split.difficulty is not None:
            item["difficulty"] = torch.tensor(
                float(self.split.difficulty[idx]), dtype=torch.float32
            )
        return item


class UnlabeledDataset(Dataset):
    """Unlabelled images returned as an aligned (weak, strong) view pair.

    The geometric transform is shared between the two views, so the returned
    tensors are pixel-aligned and a consistency loss between them is meaningful.
    ``valid`` marks pixels the student was actually shown (zero inside cut-out).
    """

    def __init__(
        self,
        split: Split,
        indices: np.ndarray,
        augment: AugmentConfig | None = None,
        strong: bool = True,
        seed: int = 0,
    ) -> None:
        self.split = split
        self.indices = np.asarray(indices)
        self.augment = augment or AugmentConfig()
        self.strong = strong
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.indices)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        idx = int(self.indices[i])
        rng = np.random.default_rng((self.seed, 10_000 + self.epoch, idx))

        # Shared geometry. The mask travels along only so it can be logged for
        # oracle diagnostics; the trainer never feeds it to the loss.
        image, mask = apply_geometric(
            self.split.images[idx], self.split.masks[idx], rng, self.augment
        )
        weak = apply_weak_photometric(image, rng, self.augment)

        if self.strong:
            strong = apply_strong_photometric(image, rng, self.augment)
            strong, valid = apply_cutout(strong, rng, self.augment)
        else:
            strong = weak
            valid = np.ones(weak.shape[:2], dtype=np.float32)

        return {
            "weak": to_tensor(weak),
            "strong": to_tensor(strong),
            "valid": torch.from_numpy(valid),
            "oracle_mask": mask_to_tensor(mask if mask is not None else np.zeros_like(valid)),
            "index": torch.tensor(idx, dtype=torch.long),
        }
