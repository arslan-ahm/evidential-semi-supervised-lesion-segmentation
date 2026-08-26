"""Supervised and consistency objectives.

Supervised heads
----------------
``ce_dice``
    Cross-entropy plus soft Dice. The standard dermoscopy objective and the
    baseline this project measures against.
``evidential``
    Dirichlet type-II maximum likelihood plus an annealed KL regulariser that
    drives evidence towards zero on *misleading* classes, plus soft Dice on the
    Dirichlet mean. The Dice term is not optional garnish: pure evidential
    losses are pixel-wise and inherit cross-entropy's indifference to class
    imbalance, which for lesions covering 4-34% of the frame collapses the
    prediction towards background.

Consistency branches
--------------------
The four methods differ *only* in how an unlabelled pixel is weighted and what
target it is given. Everything else - the shared geometry, the EMA teacher, the
ramp-up - is held fixed, so a difference in results is attributable to the
weighting rule and nothing else.

======================  ============================  ==========================
method                  per-pixel weight              target
======================  ============================  ==========================
``mean_teacher``        1                             teacher probability
``fixmatch``            ``1[max_k p_k > tau]``        one-hot ``argmax``
``evidential`` (ours)   ``1 - u``, zeroed above cut   dissonance-tempered sharpen
======================  ============================  ==========================

The ours row is the contribution, and it separates two jobs a confidence
threshold conflates:

* **Vacuity gates.** The weight is the teacher's total belief mass ``1 - u``.
  A pixel contributes in proportion to how much evidence the teacher has
  actually accumulated there, continuously, with no threshold to tune. Pixels
  above ``vacuity_cutoff`` are dropped outright, which only removes pixels the
  weight had already driven near zero. Setting
  ``semi.use_vacuity_gate = False`` makes the weight uniform while holding
  everything else fixed, which is how the ablation isolates this rule.
* **Dissonance tempers.** The sharpening temperature is raised towards 1 (no
  sharpening) where dissonance is high. Sharpening a genuinely ambiguous
  boundary pixel manufactures a confident label for something the image does not
  determine, and that error is then distilled into the student. FixMatch cannot
  express this: a border pixel at ``p = 0.5`` is *discarded* by the threshold,
  so its supervision is lost rather than softened - and lesion borders are
  exactly where segmentation error concentrates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt

from evissl.config import LossConfig, SemiConfig
from evissl.uncertainty import (
    EPS,
    dirichlet_alpha,
    dissonance,
    log_probabilities,
    probabilities,
    sharpen,
    vacuity,
)

# --------------------------------------------------------------------------- #
# Region losses
# --------------------------------------------------------------------------- #


def soft_dice_loss(
    prob: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1.0,
    ignore_background: bool = True,
) -> torch.Tensor:
    """Soft Dice loss on probabilities.

    Args:
        prob: ``(B, K, H, W)`` probabilities.
        target: ``(B, H, W)`` int64 class indices.
        smooth: Laplace term; also defines the loss for empty predictions and
            empty targets, which co-occur on lesion-free crops.
        ignore_background: Score only the foreground channels. Background
            occupies most of the frame, so including it makes Dice track
            background accuracy and hides foreground failure.

    Returns:
        Scalar loss in ``[0, 1]``.
    """
    k = prob.shape[1]
    onehot = F.one_hot(target.clamp(0, k - 1), num_classes=k).permute(0, 3, 1, 2).float()
    channels = slice(1, None) if (ignore_background and k > 1) else slice(None)
    p, t = prob[:, channels], onehot[:, channels]

    dims = (0, 2, 3)  # aggregate over the batch: stabler than per-image Dice
    intersection = (p * t).sum(dims)
    cardinality = p.sum(dims) + t.sum(dims)
    dice = (2.0 * intersection + smooth) / (cardinality + smooth)
    return 1.0 - dice.mean()


def cross_entropy_loss(
    logits: torch.Tensor, target: torch.Tensor, label_smoothing: float = 0.0
) -> torch.Tensor:
    """Standard pixel-wise cross-entropy."""
    return F.cross_entropy(logits, target, label_smoothing=label_smoothing)


# --------------------------------------------------------------------------- #
# Evidential supervised loss
# --------------------------------------------------------------------------- #


def evidential_nll(alpha: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Type-II maximum likelihood under the Dirichlet.

    Marginalising the categorical likelihood over the Dirichlet gives, for a
    one-hot target ``y``,

    .. code-block:: text

        L = sum_k y_k * ( log S - log alpha_k )

    the digamma-free form of Sensoy et al.'s equation 3 evaluated at the mean.
    This is preferred here over their Brier-style variant because the log form
    keeps a usable gradient when the evidence for the correct class is tiny,
    which is the regime the first few epochs live in.
    """
    k = alpha.shape[1]
    onehot = F.one_hot(target.clamp(0, k - 1), num_classes=k).permute(0, 3, 1, 2).float()
    strength = alpha.sum(dim=1, keepdim=True)
    per_class = onehot * (strength.clamp_min(EPS).log() - alpha.clamp_min(EPS).log())
    return per_class.sum(dim=1).mean()


def evidential_kl(alpha: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """KL from the *misleading-evidence* Dirichlet to the uniform Dirichlet.

    Evidence assigned to the correct class is removed first,
    ``alpha_tilde = y + (1 - y) * alpha``, so the penalty applies only to
    evidence supporting wrong classes. Without this term the network can lower
    the NLL by inflating evidence everywhere, and vacuity stops being
    informative - it collapses to zero for every pixel, which is precisely the
    failure this project's weighting rule depends on not happening.
    """
    k = alpha.shape[1]
    onehot = F.one_hot(target.clamp(0, k - 1), num_classes=k).permute(0, 3, 1, 2).float()
    alpha_tilde = onehot + (1.0 - onehot) * alpha

    strength = alpha_tilde.sum(dim=1, keepdim=True)
    k_tensor = torch.tensor(float(k), device=alpha.device, dtype=alpha.dtype)

    term1 = torch.lgamma(strength.squeeze(1)) - torch.lgamma(k_tensor)
    term2 = -torch.lgamma(alpha_tilde).sum(dim=1)
    term3 = (
        (alpha_tilde - 1.0) * (torch.digamma(alpha_tilde) - torch.digamma(strength))
    ).sum(dim=1)
    return (term1 + term2 + term3).mean()


# --------------------------------------------------------------------------- #
# Boundary weighting
# --------------------------------------------------------------------------- #


def boundary_weight_map(masks: np.ndarray, sigma: float = 4.0) -> np.ndarray:
    """Per-pixel weights that peak on the lesion boundary.

    Built from the Euclidean distance transform of both the mask and its
    complement, so the weight decays smoothly on either side of the contour.
    Boundary error dominates Dice on small lesions; up-weighting the contour is
    the cheapest way to attack it.

    Args:
        masks: ``(B, H, W)`` or ``(H, W)`` uint8 array in ``{0, 1}``.
        sigma: Decay length in pixels.

    Returns:
        ``float32`` array of the same shape, with values in ``[1, 2]``.
    """
    single = masks.ndim == 2
    batch = masks[None] if single else masks
    out = np.empty(batch.shape, dtype=np.float32)

    for i, mask in enumerate(batch):
        binary = mask.astype(bool)
        if not binary.any() or binary.all():
            out[i] = 1.0
            continue
        inside = distance_transform_edt(binary)
        outside = distance_transform_edt(~binary)
        distance = np.where(binary, inside, outside).astype(np.float32)
        out[i] = 1.0 + np.exp(-((distance / max(sigma, 1e-3)) ** 2))

    return out[0] if single else out


# --------------------------------------------------------------------------- #
# Supervised loss dispatcher
# --------------------------------------------------------------------------- #


@dataclass
class LossBreakdown:
    """Scalar components of one loss evaluation, for logging."""

    total: torch.Tensor
    parts: dict[str, float]


def supervised_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    cfg: LossConfig,
    epoch: int = 0,
    weight_map: torch.Tensor | None = None,
) -> LossBreakdown:
    """Evaluate the configured supervised objective.

    Args:
        logits: ``(B, K, H, W)`` raw network output.
        target: ``(B, H, W)`` int64 labels.
        cfg: Loss configuration.
        epoch: Current epoch, used to anneal the evidential KL term.
        weight_map: Optional ``(B, H, W)`` per-pixel weights, used when
            ``cfg.boundary_weight > 0``.

    Returns:
        A :class:`LossBreakdown` whose ``total`` carries the graph.
    """
    head = cfg.supervised.strip().lower()
    parts: dict[str, float] = {}

    if head == "ce_dice":
        ce = cross_entropy_loss(logits, target, cfg.label_smoothing)
        dice = soft_dice_loss(logits.softmax(dim=1), target)
        total = cfg.ce_weight * ce + cfg.dice_weight * dice
        parts["ce"] = float(ce.detach())
        parts["dice"] = float(dice.detach())

    elif head == "evidential":
        alpha = dirichlet_alpha(logits)
        nll = evidential_nll(alpha, target)
        kl = evidential_kl(alpha, target)
        dice = soft_dice_loss(alpha / alpha.sum(dim=1, keepdim=True), target)
        # Linear anneal: the KL term must not dominate before any evidence has
        # been accumulated, or the model settles into total vacuity.
        anneal = min(1.0, (epoch + 1) / max(cfg.kl_anneal_epochs, 1))
        total = cfg.ce_weight * nll + cfg.kl_weight * anneal * kl + cfg.dice_weight * dice
        parts["evidential_nll"] = float(nll.detach())
        parts["evidential_kl"] = float(kl.detach())
        parts["kl_anneal"] = anneal
        parts["dice"] = float(dice.detach())

    else:
        raise ValueError(f"Unknown supervised loss {cfg.supervised!r}; use ce_dice or evidential")

    if cfg.boundary_weight > 0 and weight_map is not None:
        # Re-weighted cross-entropy on the boundary band, added on top.
        log_p = log_probabilities(logits, head)
        nll_map = -log_p.gather(1, target.unsqueeze(1)).squeeze(1)
        boundary = (nll_map * weight_map).sum() / weight_map.sum().clamp_min(EPS)
        total = total + cfg.boundary_weight * boundary
        parts["boundary"] = float(boundary.detach())

    parts["supervised_total"] = float(total.detach())
    return LossBreakdown(total=total, parts=parts)


# --------------------------------------------------------------------------- #
# Consistency losses
# --------------------------------------------------------------------------- #


@dataclass
class ConsistencyOutput:
    """Result of one consistency evaluation.

    Attributes:
        loss: Scalar consistency term (already normalised by its weight mass).
        mask_rate: Fraction of unlabelled pixels that contributed, in ``[0, 1]``.
            Directly comparable across methods and the clearest single number
            for the "how much of the unlabelled set does each rule actually
            use?" question.
        stats: Extra diagnostics for logging.
    """

    loss: torch.Tensor
    mask_rate: float
    stats: dict[str, float]


def _weighted_soft_ce(
    student_log_prob: torch.Tensor, target_prob: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    """Cross-entropy against soft targets, weighted per pixel and normalised.

    Normalising by ``sum(weight)`` rather than by pixel count is what keeps the
    effective learning rate of the consistency term constant as the weight mass
    grows during training. Dividing by pixel count instead makes the term ramp
    up twice - once from the schedule, once from rising confidence - which is a
    well-known source of late-training collapse in threshold-based methods.
    """
    per_pixel = -(target_prob * student_log_prob).sum(dim=1)
    return (per_pixel * weight).sum() / weight.sum().clamp_min(1.0)


def consistency_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    cfg: SemiConfig,
    head: str = "evidential",
    valid: torch.Tensor | None = None,
) -> ConsistencyOutput:
    """Evaluate the configured unlabelled-data objective.

    Args:
        student_logits: ``(B, K, H, W)`` student output on the strong view.
        teacher_logits: ``(B, K, H, W)`` teacher output on the weak view.
            Must already be detached from the graph by the caller.
        cfg: Semi-supervised configuration.
        head: Supervised head in use, so probabilities are read consistently.
        valid: Optional ``(B, H, W)`` mask, zero where the student's view was
            erased by cut-out. Those pixels carry no evidence for the student
            and are excluded.

    Returns:
        A :class:`ConsistencyOutput`.
    """
    method = cfg.method.strip().lower()
    if method == "none":
        zero = student_logits.sum() * 0.0
        return ConsistencyOutput(loss=zero, mask_rate=0.0, stats={})

    teacher_logits = teacher_logits.detach()
    if valid is None:
        valid = torch.ones_like(student_logits[:, 0])

    student_log_prob = log_probabilities(student_logits, head)
    stats: dict[str, float] = {}

    if method == "mean_teacher":
        # Symmetric MSE between probability maps: no weighting, no thresholds.
        teacher_prob = probabilities(teacher_logits, head)
        student_prob = probabilities(student_logits, head)
        per_pixel = (student_prob - teacher_prob).pow(2).sum(dim=1)
        loss = (per_pixel * valid).sum() / valid.sum().clamp_min(1.0)
        weight = valid

    elif method == "fixmatch":
        teacher_prob = probabilities(teacher_logits, head)
        max_prob, pseudo = teacher_prob.max(dim=1)
        keep = (max_prob > cfg.threshold).float() * valid
        onehot = F.one_hot(pseudo, num_classes=student_logits.shape[1])
        onehot = onehot.permute(0, 3, 1, 2).float()
        loss = _weighted_soft_ce(student_log_prob, onehot, keep)
        weight = keep
        stats["mean_max_prob"] = float(max_prob.mean())

    elif method == "evidential":
        alpha_t = dirichlet_alpha(teacher_logits)
        teacher_prob = alpha_t / alpha_t.sum(dim=1, keepdim=True).clamp_min(EPS)
        u = vacuity(alpha_t)
        diss = dissonance(alpha_t)

        if cfg.use_vacuity_gate:
            # Vacuity gates: weight is the teacher's total belief mass.
            weight = (1.0 - u) * valid
            # Hard floor. Only removes pixels whose weight is already <= 0.2 at
            # the default cutoff, so it is a safety rail, not the mechanism.
            weight = weight * (u < cfg.vacuity_cutoff).float()
        else:
            # Ablation: uniform weighting, everything else unchanged.
            weight = valid

        # Dissonance tempers: T -> 1 (no sharpening) as conflict rises.
        effective_t = cfg.temperature + (1.0 - cfg.temperature) * diss
        target = sharpen(teacher_prob, effective_t)

        loss = _weighted_soft_ce(student_log_prob, target, weight)
        stats["mean_vacuity"] = float(u.mean())
        stats["mean_dissonance"] = float(diss.mean())
        stats["mean_temperature"] = float(effective_t.mean())
        stats["mean_weight"] = float(weight.mean())

    else:
        raise ValueError(
            f"Unknown semi-supervised method {cfg.method!r}; "
            "use none, mean_teacher, fixmatch or evidential"
        )

    # A single comparable number: expected weight per pixel. For fixmatch this
    # is literally the retained fraction; for ours it is the soft equivalent.
    mask_rate = float(weight.mean())
    return ConsistencyOutput(loss=loss, mask_rate=mask_rate, stats=stats)


def sigmoid_rampup(epoch: int, length: int) -> float:
    """Gaussian ramp-up ``exp(-5 (1 - t)^2)`` from Laine & Aila (2017).

    The consistency term is meaningless until the teacher is better than noise,
    so it is faded in rather than switched on. Returns 1.0 once ``epoch``
    reaches ``length``, and 1.0 immediately if ``length <= 0``.
    """
    if length <= 0:
        return 1.0
    t = float(np.clip(epoch / length, 0.0, 1.0))
    return float(np.exp(-5.0 * (1.0 - t) ** 2))
