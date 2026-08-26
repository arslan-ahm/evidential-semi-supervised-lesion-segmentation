"""DataLoader construction and the labelled/unlabelled iteration protocol.

An epoch is a **fixed number of gradient steps** (``optim.steps_per_epoch``),
not one pass over the labelled set. Both loaders are cycled indefinitely and the
trainer draws exactly that many labelled batches, each paired with ``mu`` times
as many unlabelled images.

This matters for the labelled-fraction sweep, and getting it wrong invalidates
the experiment. If an epoch were one pass over the labelled data, a run with 5%
labels would take 2 optimiser steps per epoch and a run with 50% labels would
take 25 - so the two runs would differ in *training length* as well as in label
count, and the resulting curve would measure a mixture of the two. Fixing the
step count makes the label fraction the only variable. (FixMatch fixes the step
count for the same reason.)

Setting ``steps_per_epoch: 0`` restores one-pass-per-epoch behaviour, which is
fine for a single run but should not be used for a sweep.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from typing import Any

import numpy as np
from torch.utils.data import DataLoader

from evissl.config import Config
from evissl.data.datasets import DatasetBundle, LabeledDataset, UnlabeledDataset
from evissl.data.transforms import AugmentConfig
from evissl.utils.seed import worker_init_fn


def infinite(loader: DataLoader) -> Iterator[Any]:
    """Cycle a loader forever, re-shuffling on each pass."""
    while True:
        yield from loader


class Loaders:
    """The loaders and datasets for one training run.

    Attributes:
        labeled: Shuffled loader over the labelled training subset.
        unlabeled: Loader over the unlabelled subset, or ``None`` when the run
            is purely supervised or every training image is labelled.
        val: Deterministic, un-augmented validation loader.
        test: Deterministic, un-augmented test loader.
        steps_per_epoch: Gradient steps the trainer should take per epoch.
    """

    def __init__(
        self,
        labeled: DataLoader,
        unlabeled: DataLoader | None,
        val: DataLoader,
        test: DataLoader,
        steps_per_epoch: int,
    ) -> None:
        self.labeled = labeled
        self.unlabeled = unlabeled
        self.val = val
        self.test = test
        self.steps_per_epoch = int(steps_per_epoch)
        self._labeled_iter = infinite(labeled)
        self._unlabeled_iter = infinite(unlabeled) if unlabeled is not None else None

    #: Number of labelled batches in one true pass over the labelled subset.
    @property
    def labeled_passes_per_epoch(self) -> float:
        """How many times an epoch cycles the labelled set. Logged for context."""
        return self.steps_per_epoch / max(len(self.labeled), 1)

    def next_labeled(self) -> Any:
        """Draw the next labelled batch, cycling the loader as needed."""
        return next(self._labeled_iter)

    def next_unlabeled(self) -> Any | None:
        """Draw the next unlabelled batch, or ``None`` if there is no branch."""
        if self._unlabeled_iter is None:
            return None
        return next(self._unlabeled_iter)

    def set_epoch(self, epoch: int) -> None:
        """Advance the augmentation RNG stream on every underlying dataset."""
        for loader in (self.labeled, self.unlabeled):
            if loader is None:
                continue
            dataset = loader.dataset
            if hasattr(dataset, "set_epoch"):
                dataset.set_epoch(epoch)


def build_loaders(
    cfg: Config,
    bundle: DatasetBundle,
    augment: AugmentConfig | None = None,
    generator: Any | None = None,
) -> Loaders:
    """Assemble every loader a run needs.

    Args:
        cfg: Full run configuration.
        bundle: Output of :func:`evissl.data.datasets.build_dataset`.
        augment: Augmentation settings; defaults are used when omitted.
        generator: ``torch.Generator`` for shuffling, from ``seed_everything``.

    Returns:
        A :class:`Loaders` container.
    """
    augment = augment or AugmentConfig()
    dcfg = cfg.data
    seed = cfg.run.seed

    common: dict[str, Any] = {
        "num_workers": dcfg.num_workers,
        "pin_memory": False,
        "persistent_workers": dcfg.num_workers > 0,
    }
    if dcfg.num_workers > 0:
        common["worker_init_fn"] = worker_init_fn

    labeled_ds = LabeledDataset(
        bundle.train, bundle.labeled_idx, augment=augment, train=True, seed=seed
    )
    # drop_last keeps batch size constant, but must not empty the loader when
    # the labelled subset is smaller than two batches (the 5% ablation rung).
    drop_last = len(labeled_ds) >= 2 * dcfg.batch_size
    labeled = DataLoader(
        labeled_ds,
        batch_size=dcfg.batch_size,
        shuffle=True,
        drop_last=drop_last,
        generator=generator,
        **common,
    )

    unlabeled: DataLoader | None = None
    if cfg.semi.method != "none" and len(bundle.unlabeled_idx) > 0:
        unlabeled_ds = UnlabeledDataset(
            bundle.train,
            bundle.unlabeled_idx,
            augment=augment,
            strong=cfg.semi.strong_aug,
            seed=seed,
        )
        unlabeled = DataLoader(
            unlabeled_ds,
            batch_size=max(1, dcfg.batch_size * dcfg.mu),
            shuffle=True,
            drop_last=True,
            generator=generator,
            **common,
        )

    configured = int(cfg.optim.steps_per_epoch)
    if configured > 0:
        steps = configured
    else:
        # One pass over the labelled subset. Correct for a single run; see the
        # module docstring for why a sweep should not use it.
        steps = max(1, math.ceil(len(labeled_ds) / max(dcfg.batch_size, 1)))

    def eval_loader(split: Any) -> DataLoader:
        return DataLoader(
            LabeledDataset(split, np.arange(len(split)), augment=augment, train=False, seed=seed),
            batch_size=max(1, dcfg.batch_size),
            shuffle=False,
            drop_last=False,
            **common,
        )

    return Loaders(
        labeled, unlabeled, eval_loader(bundle.val), eval_loader(bundle.test), steps
    )
