"""Data layer: generator determinism, split protocol, view alignment."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch

from evissl.config import DataConfig, load_config
from evissl.data import build_dataset, build_loaders, generate_dataset, generate_sample
from evissl.data.datasets import LabeledDataset, Split, UnlabeledDataset, _index_directory
from evissl.data.transforms import (
    AugmentConfig,
    apply_cutout,
    apply_geometric,
    apply_strong_photometric,
    denormalize,
    to_tensor,
)

# --------------------------------------------------------------------------- #
# Generator
# --------------------------------------------------------------------------- #


def test_sample_shapes_and_dtypes():
    image, mask, params = generate_sample(0, 64)
    assert image.shape == (64, 64, 3) and image.dtype == np.uint8
    assert mask.shape == (64, 64) and mask.dtype == np.uint8
    assert set(np.unique(mask)).issubset({0, 1})
    assert 0.0 <= params.difficulty() <= 1.0


def test_sample_is_a_pure_function_of_its_seed():
    a, ma, _ = generate_sample(42, 48)
    b, mb, _ = generate_sample(42, 48)
    assert np.array_equal(a, b)
    assert np.array_equal(ma, mb)


def test_different_seeds_give_different_images():
    a, _, _ = generate_sample(1, 48)
    b, _, _ = generate_sample(2, 48)
    assert not np.array_equal(a, b)


def test_lesions_are_present_and_not_degenerate():
    _, masks, _ = generate_dataset(24, 64, seed=3)
    areas = masks.reshape(len(masks), -1).mean(axis=1)
    # Every sample must contain a lesion, and none may fill the frame.
    assert np.all(areas > 0.0), "generator produced an empty mask"
    assert np.all(areas < 0.8), "generator produced a frame-filling lesion"


def test_hard_fraction_controls_difficulty():
    _, _, easy = generate_dataset(40, 64, seed=5, hard_fraction=0.0)
    _, _, hard = generate_dataset(40, 64, seed=5, hard_fraction=1.0)
    easy_mean = np.mean([p.difficulty() for p in easy])
    hard_mean = np.mean([p.difficulty() for p in hard])
    assert hard_mean > easy_mean + 0.1


def test_occluders_do_not_change_the_label():
    """Hair and rulers cross the lesion; the annotator sees through them.

    The mask must come from the pre-occlusion field, so two samples that differ
    only in occluders would share a mask. Checked here by confirming the mask is
    a single connected, filled region rather than one punctured by hair strands.
    """
    from scipy.ndimage import label

    for seed in range(6):
        _, mask, params = generate_sample(seed * 977, 96, hard_fraction=1.0)
        if params.n_hairs < 10:
            continue
        # Hair drawn into the mask would fragment it into many components.
        _, n_components = label(mask)
        assert n_components <= 1 + params.n_satellites


# --------------------------------------------------------------------------- #
# Splits
# --------------------------------------------------------------------------- #


def test_splits_are_disjoint_and_sized():
    cfg = DataConfig(name="synthetic", image_size=32, train_size=40, val_size=8, test_size=12)
    bundle = build_dataset(cfg, seed=11)
    assert len(bundle.train) == 40
    assert len(bundle.val) == 8
    assert len(bundle.test) == 12
    assert set(bundle.labeled_idx).isdisjoint(bundle.unlabeled_idx)
    assert len(bundle.labeled_idx) + len(bundle.unlabeled_idx) == 40


def test_labeled_subsets_are_nested_across_fractions():
    """Raising the labelled fraction must *add* images, never reshuffle them.

    Without this the labelled-fraction sweep compares unrelated random draws
    and any trend it shows could be sampling noise.
    """
    base = DataConfig(name="synthetic", image_size=32, train_size=60, val_size=4, test_size=4)
    previous: set[int] = set()
    for fraction in (0.05, 0.10, 0.25, 0.50, 1.0):
        bundle = build_dataset(dataclasses.replace(base, labeled_fraction=fraction), seed=13)
        current = set(bundle.labeled_idx.tolist())
        assert previous.issubset(current), f"fraction {fraction} dropped earlier images"
        previous = current


def test_labeled_count_overrides_fraction():
    cfg = DataConfig(
        name="synthetic", image_size=32, train_size=50, val_size=4, test_size=4,
        labeled_fraction=0.5, labeled_count=7,
    )
    assert len(build_dataset(cfg, seed=1).labeled_idx) == 7


def test_at_least_one_labeled_image_always():
    cfg = DataConfig(
        name="synthetic", image_size=32, train_size=10, val_size=2, test_size=2,
        labeled_fraction=0.001,
    )
    assert len(build_dataset(cfg, seed=1).labeled_idx) == 1


def test_unknown_dataset_is_rejected():
    with pytest.raises(ValueError, match="Unknown dataset"):
        build_dataset(DataConfig(name="not_a_dataset"), seed=0)


def test_missing_real_dataset_raises_clearly(tmp_path):
    cfg = DataConfig(name="isic2018", root=str(tmp_path), image_size=32)
    with pytest.raises(FileNotFoundError):
        build_dataset(cfg, seed=0)


def test_index_directory_matches_isic_and_ph2_naming(tmp_path):
    from PIL import Image

    images = tmp_path / "images"
    masks = tmp_path / "masks"
    images.mkdir()
    masks.mkdir()
    for stem, mask_name in [
        ("ISIC_0001", "ISIC_0001_segmentation"),
        ("ISIC_0002", "ISIC_0002"),
        ("IMD003", "IMD003_lesion"),
    ]:
        Image.new("RGB", (8, 8)).save(images / f"{stem}.jpg")
        Image.new("L", (8, 8)).save(masks / f"{mask_name}.png")
    Image.new("RGB", (8, 8)).save(images / "orphan.jpg")  # no mask -> skipped

    pairs = _index_directory(images, masks)
    assert len(pairs) == 3


# --------------------------------------------------------------------------- #
# Transforms
# --------------------------------------------------------------------------- #


def test_tensor_roundtrip_is_lossless_within_quantisation():
    image = (np.random.default_rng(0).random((16, 16, 3)) * 255).astype(np.uint8)
    recovered = denormalize(to_tensor(image))
    assert np.abs(recovered.astype(int) - image.astype(int)).max() <= 1


def test_geometric_keeps_mask_binary_and_shape():
    image, mask, _ = generate_sample(7, 48)
    rng = np.random.default_rng(0)
    out_image, out_mask = apply_geometric(image, mask, rng, AugmentConfig())
    assert out_image.shape == image.shape
    assert out_mask.shape == mask.shape
    assert set(np.unique(out_mask)).issubset({0, 1})


def test_geometric_handles_missing_mask():
    image, _, _ = generate_sample(7, 48)
    out_image, out_mask = apply_geometric(image, None, np.random.default_rng(0), AugmentConfig())
    assert out_image.shape == image.shape
    assert out_mask is None


def test_cutout_marks_erased_pixels_invalid():
    image, _, _ = generate_sample(9, 64)
    cfg = AugmentConfig(cutout_prob=1.0, cutout_count=2, cutout_size_frac=(0.2, 0.25))
    out, valid = apply_cutout(image, np.random.default_rng(0), cfg)
    assert valid.min() == 0.0, "cut-out must mark pixels invalid"
    assert valid.mean() < 1.0
    # Erased regions are filled with a constant, so they carry no gradient.
    assert out.shape == image.shape


def test_cutout_disabled_leaves_everything_valid():
    image, _, _ = generate_sample(9, 32)
    _, valid = apply_cutout(image, np.random.default_rng(0), AugmentConfig(cutout_prob=0.0))
    assert valid.min() == 1.0


def test_strong_photometric_changes_appearance_only():
    image, _, _ = generate_sample(11, 64)
    strong = apply_strong_photometric(image, np.random.default_rng(0), AugmentConfig())
    assert strong.shape == image.shape
    assert not np.array_equal(strong, image)


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #


def _profile_correlation(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean correlation of row and column intensity profiles.

    Invariant to global photometric change, highly sensitive to any spatial
    misalignment - which is exactly the property needed to test that the two
    consistency views share their geometric transform.
    """
    scores = []
    for i in range(a.shape[0]):
        for axis in (0, 1):
            x = a[i].mean(dim=axis).numpy()
            y = b[i].mean(dim=axis).numpy()
            scores.append(np.corrcoef(x, y)[0, 1])
    return float(np.nanmean(scores))


def test_weak_and_strong_views_are_spatially_aligned():
    """The load-bearing property of the consistency branch.

    A pixel-wise consistency loss between misaligned views is noise. This test
    fails loudly if the geometric transform ever stops being shared.
    """
    cfg = load_config("configs/smoke.yaml")
    bundle = build_dataset(cfg.data, cfg.run.seed)
    loaders = build_loaders(cfg, bundle)
    batch = loaders.next_unlabeled()

    weak = batch["weak"].mean(dim=1)
    strong = batch["strong"].mean(dim=1)

    aligned = _profile_correlation(weak, strong)
    shifted = _profile_correlation(weak, torch.roll(strong, 5, dims=-1))
    assert aligned > 0.9, f"views are not aligned (correlation {aligned:.3f})"
    assert aligned > shifted + 0.2, "test is not sensitive to misalignment"


def test_supervised_run_has_no_unlabeled_loader():
    cfg = load_config("configs/supervised_baseline.yaml", ["data.train_size=40", "data.val_size=4",
                                                          "data.test_size=4", "data.image_size=32"])
    loaders = build_loaders(cfg, build_dataset(cfg.data, cfg.run.seed))
    assert loaders.unlabeled is None
    assert loaders.next_unlabeled() is None


def test_steps_per_epoch_is_fixed_regardless_of_label_count():
    """The design decision that makes the labelled-fraction sweep controlled."""
    steps = []
    for fraction in (0.05, 0.50):
        cfg = load_config(
            "configs/evidential.yaml",
            [f"data.labeled_fraction={fraction}", "data.train_size=200", "data.val_size=4",
             "data.test_size=4", "data.image_size=32", "optim.steps_per_epoch=32"],
        )
        loaders = build_loaders(cfg, build_dataset(cfg.data, cfg.run.seed))
        steps.append(loaders.steps_per_epoch)
    assert steps == [32, 32]


def test_steps_per_epoch_zero_falls_back_to_one_pass():
    cfg = load_config(
        "configs/evidential.yaml",
        ["data.labeled_fraction=0.5", "data.train_size=100", "data.batch_size=10",
         "data.val_size=4", "data.test_size=4", "data.image_size=32",
         "optim.steps_per_epoch=0"],
    )
    loaders = build_loaders(cfg, build_dataset(cfg.data, cfg.run.seed))
    assert loaders.steps_per_epoch == 5  # 50 labelled / batch 10


def test_loaders_cycle_indefinitely():
    cfg = load_config("configs/smoke.yaml")
    loaders = build_loaders(cfg, build_dataset(cfg.data, cfg.run.seed))
    # More draws than the loader has batches must still succeed.
    for _ in range(len(loaders.labeled) * 3 + 2):
        assert loaders.next_labeled()["image"].shape[0] >= 1


def test_augmentation_is_reproducible_per_epoch():
    split = Split(*generate_dataset(6, 32, seed=1)[:2])
    dataset = LabeledDataset(split, np.arange(6), train=True, seed=5)
    dataset.set_epoch(2)
    first = dataset[0]["image"].clone()
    dataset.set_epoch(2)
    assert torch.allclose(first, dataset[0]["image"])
    dataset.set_epoch(3)
    assert not torch.allclose(first, dataset[0]["image"])


def test_eval_dataset_is_not_augmented():
    split = Split(*generate_dataset(4, 32, seed=1)[:2])
    dataset = LabeledDataset(split, np.arange(4), train=False, seed=5)
    dataset.set_epoch(1)
    first = dataset[0]["image"].clone()
    dataset.set_epoch(9)
    assert torch.allclose(first, dataset[0]["image"])


def test_unlabeled_dataset_without_strong_aug_returns_identical_views():
    split = Split(*generate_dataset(4, 32, seed=1)[:2])
    dataset = UnlabeledDataset(split, np.arange(4), strong=False, seed=0)
    item = dataset[0]
    assert torch.allclose(item["weak"], item["strong"])
    assert item["valid"].min() == 1.0
