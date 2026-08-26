"""The training loop.

All four semi-supervised ideologies share **one** loop. This is a deliberate
experimental-design decision, not code golf: if the supervised-only baseline,
Mean Teacher, FixMatch and the evidential method each had their own training
script, any difference in results could be attributed to an incidental
difference in the schedule, the augmentation, the EMA handling or the checkpoint
selection rather than to the ideology under test. Routing every method through
this file makes ``semi.method`` the only thing that varies, so the comparison
in ``results/tables/`` is clean.

Per step, when a consistency branch is active:

1. Draw a labelled batch, compute the supervised loss on it.
2. Draw ``mu`` times as many unlabelled images, each as an aligned
   (weak, strong) view pair.
3. Forward the **teacher** on the weak view under ``no_grad``.
4. Forward the **student** on the strong view.
5. Add ``w(epoch) * consistency``, where ``w`` is the sigmoid ramp-up.
6. Step the optimiser, then update the teacher.

The student's labelled and unlabelled forwards are concatenated into a single
batch. On CPU this is a material speed-up - one kernel launch sequence instead
of two - and it also makes the two branches see identical normalisation, which
matters if a BatchNorm backbone is ever substituted.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import SGD, AdamW, Optimizer

from evissl.config import Config
from evissl.data.loaders import Loaders
from evissl.engine.ema import ModelEMA
from evissl.losses import boundary_weight_map, consistency_loss, sigmoid_rampup, supervised_loss
from evissl.metrics.segmentation import LOWER_IS_BETTER, aggregate, compute_batch
from evissl.uncertainty import probabilities
from evissl.utils.checkpoint import save_checkpoint
from evissl.utils.logging import JsonlLogger, get_logger

logger = get_logger("evissl.train")


@dataclass
class TrainState:
    """Mutable bookkeeping for one run."""

    epoch: int = 0
    global_step: int = 0
    best_metric: float = -math.inf
    best_epoch: int = -1
    history: list[dict[str, Any]] = field(default_factory=list)


def build_optimizer(model: nn.Module, cfg: Config) -> Optimizer:
    """Build the optimiser, excluding norm and bias parameters from decay.

    Weight decay on normalisation scales and biases shrinks parameters that
    have no business being shrunk; the exclusion is standard practice and worth
    roughly a point of Dice on small models.
    """
    decay_params, no_decay_params = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim <= 1 or name.endswith(".bias") or "gamma" in name:
            no_decay_params.append(parameter)
        else:
            decay_params.append(parameter)

    groups = [
        {"params": decay_params, "weight_decay": cfg.optim.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    name = cfg.optim.name.strip().lower()
    if name == "adamw":
        return AdamW(groups, lr=cfg.optim.lr, betas=(0.9, 0.999))
    if name == "sgd":
        return SGD(groups, lr=cfg.optim.lr, momentum=0.9, nesterov=True)
    raise ValueError(f"Unknown optimiser {cfg.optim.name!r}; use adamw or sgd")


def lr_at(cfg: Config, epoch_fraction: float) -> float:
    """Learning-rate multiplier: linear warm-up then cosine decay.

    Args:
        cfg: Run configuration.
        epoch_fraction: Fractional epoch, e.g. 2.5 midway through epoch 3.

    Returns:
        A multiplier in ``[0, 1]`` applied to ``cfg.optim.lr``.
    """
    warmup = cfg.optim.warmup_epochs
    total = max(cfg.optim.epochs, 1)
    if warmup > 0 and epoch_fraction < warmup:
        # Start at 1/warmup rather than 0 so the first step still moves.
        return (epoch_fraction + 1e-8) / warmup
    if cfg.optim.scheduler.strip().lower() != "cosine":
        return 1.0
    progress = (epoch_fraction - warmup) / max(total - warmup, 1e-8)
    progress = float(np.clip(progress, 0.0, 1.0))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


class Trainer:
    """Trains one configuration and reports validation metrics per epoch.

    Args:
        cfg: Full run configuration.
        model: Student network.
        loaders: Data loaders from :func:`evissl.data.build_loaders`.
        device: Target device.
        out_dir: Directory for the JSONL history; defaults to
            ``cfg.run.out_dir / runs / cfg.run.name``.
    """

    def __init__(
        self,
        cfg: Config,
        model: nn.Module,
        loaders: Loaders,
        device: torch.device,
        out_dir: str | Path | None = None,
    ) -> None:
        self.cfg = cfg
        self.model = model.to(device)
        self.loaders = loaders
        self.device = device
        self.head = cfg.loss.supervised.strip().lower()
        self.semi_enabled = cfg.semi.method.strip().lower() != "none"

        self.optimizer = build_optimizer(self.model, cfg)
        self.ema = ModelEMA(self.model, decay=cfg.semi.ema_decay)
        self.state = TrainState()

        self.out_dir = Path(out_dir or Path(cfg.run.out_dir) / "runs" / cfg.run.name)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl = JsonlLogger(self.out_dir / "history.jsonl")
        cfg.save(self.out_dir / "config.yaml")

        self.ckpt_path = Path(cfg.run.ckpt_dir) / f"{cfg.run.name}.pt"

    # -- one step --------------------------------------------------------- #

    def _set_lr(self, epoch_fraction: float) -> float:
        lr = self.cfg.optim.lr * lr_at(self.cfg, epoch_fraction)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr

    def _boundary_weights(self, masks: torch.Tensor) -> torch.Tensor | None:
        if self.cfg.loss.boundary_weight <= 0:
            return None
        weights = boundary_weight_map(masks.detach().cpu().numpy().astype(np.uint8))
        return torch.from_numpy(weights).to(self.device)

    def train_epoch(self, epoch: int) -> dict[str, float]:
        """Run one epoch and return averaged training diagnostics."""
        self.model.train()
        self.loaders.set_epoch(epoch)

        cons_weight = self.cfg.semi.consistency_weight * sigmoid_rampup(
            epoch, self.cfg.semi.rampup_epochs
        )
        steps = max(self.loaders.steps_per_epoch, 1)
        totals: dict[str, float] = {}
        n_batches = 0
        start = time.perf_counter()

        for i in range(steps):
            batch = self.loaders.next_labeled()
            lr = self._set_lr(epoch + i / steps)

            images = batch["image"].to(self.device, non_blocking=True)
            masks = batch["mask"].to(self.device, non_blocking=True)
            n_labeled = images.shape[0]

            unlabeled = self.loaders.next_unlabeled() if self.semi_enabled else None

            if unlabeled is not None:
                weak = unlabeled["weak"].to(self.device, non_blocking=True)
                strong = unlabeled["strong"].to(self.device, non_blocking=True)
                valid = unlabeled["valid"].to(self.device, non_blocking=True)

                # Teacher sees the weak view; no gradient path.
                teacher_logits = self.ema(weak)
                # One student forward over labelled + strong-unlabelled.
                logits = self.model(torch.cat([images, strong], dim=0))
                sup_logits, unsup_logits = logits[:n_labeled], logits[n_labeled:]
            else:
                sup_logits = self.model(images)
                teacher_logits = unsup_logits = valid = None

            sup = supervised_loss(
                sup_logits, masks, self.cfg.loss, epoch, self._boundary_weights(masks)
            )
            loss = sup.total
            parts = dict(sup.parts)

            if unsup_logits is not None:
                cons = consistency_loss(
                    unsup_logits, teacher_logits, self.cfg.semi, self.head, valid
                )
                loss = loss + cons_weight * cons.loss
                parts["consistency"] = float(cons.loss.detach())
                parts["mask_rate"] = cons.mask_rate
                parts.update(cons.stats)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if self.cfg.optim.grad_clip > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.optim.grad_clip
                )
                parts["grad_norm"] = float(grad_norm)
            self.optimizer.step()
            parts["ema_decay"] = self.ema.update(self.model)

            parts["loss"] = float(loss.detach())
            parts["lr"] = lr
            for key, value in parts.items():
                totals[key] = totals.get(key, 0.0) + value
            n_batches += 1
            self.state.global_step += 1

            if self.cfg.run.log_every and (i + 1) % self.cfg.run.log_every == 0:
                logger.info(
                    "epoch %d step %d/%d loss=%.4f", epoch, i + 1, steps, parts["loss"]
                )

        out = {k: v / max(n_batches, 1) for k, v in totals.items()}
        out["consistency_weight"] = cons_weight
        out["epoch_seconds"] = time.perf_counter() - start
        return out

    # -- evaluation ------------------------------------------------------- #

    @torch.no_grad()
    def evaluate(
        self,
        loader: Any,
        use_ema: bool | None = None,
        boundary_metrics: bool = False,
    ) -> dict[str, float]:
        """Segmentation metrics on a loader, using the teacher by default.

        Args:
            loader: Evaluation loader.
            use_ema: Score the EMA teacher (default from ``cfg.eval.use_ema``).
            boundary_metrics: Include HD95, ASSD and BF1. Off during training -
                the distance transforms cost more than the forward pass itself,
                and checkpoint selection only needs the monitored overlap
                metric. The full set is computed once, at test time, by
                :func:`evissl.eval.evaluate`.
        """
        use_ema = self.cfg.eval.use_ema if use_ema is None else use_ema
        network = self.ema.module if use_ema else self.model
        was_training = network.training
        network.eval()

        per_image: list[dict[str, float]] = []
        for batch in loader:
            images = batch["image"].to(self.device, non_blocking=True)
            prob = probabilities(network(images), self.head)[:, 1]
            pred = (prob >= self.cfg.eval.threshold).cpu().numpy().astype(np.uint8)
            target = batch["mask"].numpy().astype(np.uint8)
            per_image.extend(compute_batch(pred, target, boundary_metrics=boundary_metrics))

        network.train(was_training)
        return aggregate(per_image)

    # -- full run --------------------------------------------------------- #

    def fit(self) -> TrainState:
        """Train for ``cfg.optim.epochs``, tracking the best validation score."""
        monitor = self.cfg.run.monitor
        logger.info(
            "run=%s method=%s head=%s epochs=%d steps/epoch=%d "
            "(%.1f passes over the labelled set) device=%s",
            self.cfg.run.name,
            self.cfg.semi.method,
            self.head,
            self.cfg.optim.epochs,
            self.loaders.steps_per_epoch,
            self.loaders.labeled_passes_per_epoch,
            self.device,
        )

        # Only pay for the distance transforms when the monitored metric is one
        # of the boundary metrics; ``dice`` and ``iou`` are in the fast set.
        needs_boundary = monitor in {"hd95", "assd", "boundary_f1"}

        for epoch in range(self.cfg.optim.epochs):
            self.state.epoch = epoch
            train_stats = self.train_epoch(epoch)
            val_stats = self.evaluate(self.loaders.val, boundary_metrics=needs_boundary)

            record = {
                "epoch": epoch,
                **{f"train_{k}": v for k, v in train_stats.items()},
                **{f"val_{k}": v for k, v in val_stats.items()},
            }
            self.jsonl.log(**record)
            self.state.history.append(record)

            # Track the monitored metric in a maximise-always form: metrics
            # where lower is better are negated, so the comparison below is
            # correct for `monitor: hd95` as well as `monitor: dice`.
            raw_score = val_stats.get(monitor, float("nan"))
            score = -raw_score if monitor in LOWER_IS_BETTER else raw_score
            improved = np.isfinite(score) and score > self.state.best_metric
            if improved:
                self.state.best_metric = float(score)
                self.state.best_epoch = epoch
                if self.cfg.run.save_best:
                    save_checkpoint(
                        self.ckpt_path,
                        self.model,
                        ema_model=self.ema.module,
                        epoch=epoch,
                        metrics=val_stats,
                        config=self.cfg.to_dict(),
                    )

            logger.info(
                "epoch %d | loss=%.4f | val_%s=%.4f%s | %.1fs",
                epoch,
                train_stats.get("loss", float("nan")),
                monitor,
                raw_score,
                "  <- best" if improved else "",
                train_stats.get("epoch_seconds", 0.0),
            )

        best_raw = (
            -self.state.best_metric if monitor in LOWER_IS_BETTER else self.state.best_metric
        )
        logger.info(
            "done: best val_%s=%.4f at epoch %d", monitor, best_raw, self.state.best_epoch
        )
        return self.state
