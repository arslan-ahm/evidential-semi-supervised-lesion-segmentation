"""Evidential uncertainty for dense prediction, and the baselines it replaces.

The network emits ``K`` non-negative *evidence* values per pixel,
``e = softplus(logits)``, which parameterise a Dirichlet distribution over the
class probabilities with ``alpha = e + 1``. Writing ``S = sum_k alpha_k`` for the
Dirichlet strength, subjective logic (Josang, 2016) gives a decomposition of the
opinion at that pixel into three interpretable quantities:

``belief``  ``b_k = e_k / S``
    Evidence actually accumulated for class ``k``.
``vacuity``  ``u = K / S``, and ``sum_k b_k + u = 1``
    *Epistemic* uncertainty - how much of the opinion is "I have not seen
    enough to say". Maximal (1.0) at initialisation, when there is no evidence.
``dissonance``
    *Aleatoric* uncertainty - how much of the accumulated evidence is mutually
    contradictory. For the binary case Josang's definition collapses to the
    closed form ``2 * min(b_0, b_1)``, derived in :func:`dissonance`.

That split is the whole reason this project prefers evidence to confidence. A
softmax probability of 0.5 is a single number that cannot distinguish "this
pixel is an ambiguous lesion border" (low vacuity, high dissonance - a real,
irreducible property of the image) from "I have never seen a pixel like this"
(high vacuity - a statement about the model, fixable with more data). A
confidence threshold discards both cases identically. The trainer in
:mod:`evissl.engine.trainer` treats them differently, and the ablation in
``notebooks/04`` measures what that separation is worth.

References:
    Sensoy, Kaplan & Kandemir. Evidential Deep Learning to Quantify
    Classification Uncertainty. NeurIPS 2018.
    Josang. Subjective Logic: A Formalism for Reasoning Under Uncertainty. 2016.
    Kendall & Gal. What Uncertainties Do We Need in Bayesian Deep Learning for
    Computer Vision? NeurIPS 2017.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

EPS = 1e-8


# --------------------------------------------------------------------------- #
# Dirichlet parameterisation
# --------------------------------------------------------------------------- #


def evidence_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Map logits to non-negative evidence with ``softplus``.

    ``softplus`` is preferred over ``exp`` (which overflows and produces
    enormous strengths early in training) and over ``relu`` (which yields exactly
    zero evidence, killing the gradient for every pixel the model gets wrong).
    """
    return F.softplus(logits)


def dirichlet_alpha(logits: torch.Tensor) -> torch.Tensor:
    """Dirichlet concentration ``alpha = softplus(logits) + 1``."""
    return evidence_from_logits(logits) + 1.0


def expected_probability(alpha: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """Mean of the Dirichlet, ``alpha_k / S`` - the point prediction."""
    return alpha / alpha.sum(dim=dim, keepdim=True).clamp_min(EPS)


def vacuity(alpha: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """Epistemic uncertainty ``u = K / S`` in ``(0, 1]``.

    Returns a tensor with the class dimension removed.
    """
    k = alpha.shape[dim]
    return k / alpha.sum(dim=dim).clamp_min(EPS)


def belief(alpha: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """Per-class belief mass ``b_k = (alpha_k - 1) / S``. Sums to ``1 - u``."""
    strength = alpha.sum(dim=dim, keepdim=True).clamp_min(EPS)
    return (alpha - 1.0) / strength


def dissonance(alpha: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """Aleatoric uncertainty from mutually conflicting evidence.

    Josang's dissonance is

    .. code-block:: text

        diss = sum_k [ b_k * sum_{j != k} b_j * Bal(b_j, b_k) ] / sum_{j != k} b_j
        Bal(b_j, b_k) = 1 - |b_j - b_k| / (b_j + b_k)

    For ``K = 2`` the inner sums have a single term each, so the expression
    telescopes:

    .. code-block:: text

        diss = (b_0 + b_1) * Bal(b_0, b_1)
             = (b_0 + b_1) - |b_0 - b_1|
             = 2 * min(b_0, b_1)

    which is the form evaluated here for the binary case. The general-``K``
    branch is implemented for completeness so the module is not silently wrong
    if a multi-class variant is ever trained.

    Returns:
        Tensor in ``[0, 1]`` with the class dimension removed. Maximal when
        evidence is split evenly between classes *and* is plentiful, i.e. the
        model is sure the pixel is genuinely ambiguous.
    """
    b = belief(alpha, dim=dim)
    k = alpha.shape[dim]

    if k == 2:
        return 2.0 * b.min(dim=dim).values

    total = b.sum(dim=dim, keepdim=True)
    # Pairwise tensors indexed [..., i, j, ...]: axis `dim` is i, `dim + 1` is j.
    b_i = b.unsqueeze(dim + 1)
    b_j = b.unsqueeze(dim)
    pair_sum = b_i + b_j
    # Bal is defined as 0 when either belief is 0; the naive ratio would give 1.
    balance = torch.where(
        pair_sum > EPS,
        1.0 - (b_i - b_j).abs() / pair_sum.clamp_min(EPS),
        torch.zeros_like(pair_sum),
    )
    # Zero the diagonal: a class is never in conflict with itself.
    eye = torch.eye(k, device=b.device, dtype=b.dtype)
    shape = [1] * balance.dim()
    shape[dim], shape[dim + 1] = k, k
    balance = balance * (1.0 - eye.view(shape))

    # Sum over j (axis dim + 1), leaving one value per class i.
    weighted = (b_j * balance).sum(dim=dim + 1)
    b_other = (total - b).clamp_min(EPS)
    return (b * weighted / b_other).sum(dim=dim).clamp(0.0, 1.0)


@dataclass
class EvidentialOutput:
    """Everything derivable from one forward pass of an evidential head.

    Attributes:
        alpha: ``(B, K, H, W)`` Dirichlet concentrations.
        prob: ``(B, K, H, W)`` expected class probabilities.
        vacuity: ``(B, H, W)`` epistemic uncertainty in ``(0, 1]``.
        dissonance: ``(B, H, W)`` aleatoric uncertainty in ``[0, 1]``.
        strength: ``(B, H, W)`` total Dirichlet strength ``S``.
    """

    alpha: torch.Tensor
    prob: torch.Tensor
    vacuity: torch.Tensor
    dissonance: torch.Tensor
    strength: torch.Tensor

    @property
    def lesion_prob(self) -> torch.Tensor:
        """Expected probability of the positive (lesion) class."""
        return self.prob[:, 1]

    @property
    def total_uncertainty(self) -> torch.Tensor:
        """Vacuity plus dissonance, clipped to ``[0, 1]``.

        A single scalar for visualisation and for the uncertainty-error
        correlation analysis. The two components stay available separately
        because they call for different responses.
        """
        return (self.vacuity + self.dissonance).clamp(0.0, 1.0)


def evidential_output(logits: torch.Tensor) -> EvidentialOutput:
    """Compute the full evidential summary from raw logits."""
    alpha = dirichlet_alpha(logits)
    strength = alpha.sum(dim=1)
    return EvidentialOutput(
        alpha=alpha,
        prob=expected_probability(alpha),
        vacuity=vacuity(alpha),
        dissonance=dissonance(alpha),
        strength=strength,
    )


# --------------------------------------------------------------------------- #
# Head-agnostic probability access
# --------------------------------------------------------------------------- #


def probabilities(logits: torch.Tensor, head: str = "evidential") -> torch.Tensor:
    """Class probabilities under either head.

    Args:
        logits: ``(B, K, H, W)`` raw network output.
        head: ``"evidential"`` (Dirichlet mean) or ``"ce_dice"`` (softmax).

    Returns:
        ``(B, K, H, W)`` probabilities summing to one over ``dim=1``.
    """
    if head == "evidential":
        return expected_probability(dirichlet_alpha(logits))
    return logits.softmax(dim=1)


def log_probabilities(logits: torch.Tensor, head: str = "evidential") -> torch.Tensor:
    """Log class probabilities under either head, computed stably."""
    if head == "evidential":
        alpha = dirichlet_alpha(logits)
        return alpha.clamp_min(EPS).log() - alpha.sum(dim=1, keepdim=True).clamp_min(EPS).log()
    return logits.log_softmax(dim=1)


def sharpen(prob: torch.Tensor, temperature: torch.Tensor | float, dim: int = 1) -> torch.Tensor:
    """Temperature-sharpen a probability tensor: ``p^(1/T)`` renormalised.

    ``temperature`` may be a scalar or a per-pixel tensor broadcastable against
    ``prob`` with the class dimension removed - which is what lets
    :mod:`evissl.losses` sharpen confident pixels hard while leaving genuinely
    ambiguous ones almost untouched.
    """
    if not isinstance(temperature, torch.Tensor):
        temperature = torch.tensor(float(temperature), device=prob.device, dtype=prob.dtype)
    if temperature.dim() == prob.dim() - 1:
        temperature = temperature.unsqueeze(dim)
    inv_t = 1.0 / temperature.clamp_min(1e-3)
    powered = prob.clamp_min(EPS) ** inv_t
    return powered / powered.sum(dim=dim, keepdim=True).clamp_min(EPS)


def predictive_entropy(prob: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """Shannon entropy of a probability tensor, normalised to ``[0, 1]``."""
    k = prob.shape[dim]
    raw = -(prob.clamp_min(EPS) * prob.clamp_min(EPS).log()).sum(dim=dim)
    return raw / torch.log(torch.tensor(float(k), device=prob.device, dtype=prob.dtype))


# --------------------------------------------------------------------------- #
# Sampling-based baselines
# --------------------------------------------------------------------------- #


@torch.no_grad()
def mc_dropout_predict(
    model: nn.Module, images: torch.Tensor, samples: int = 8, head: str = "evidential"
) -> dict[str, torch.Tensor]:
    """Monte-Carlo dropout uncertainty - the sampling baseline.

    Runs ``samples`` stochastic forward passes with dropout active and
    normalisation frozen, then splits the predictive entropy into an epistemic
    part (mutual information between prediction and weights) and an aleatoric
    part (expected entropy of individual samples), following Depeweg et al.

    Cost: ``samples`` forward passes per image. The evidential head obtains
    both components from **one** pass, which is the practical argument for it -
    quantified in ``scripts/benchmark_efficiency.py``.

    Args:
        model: Network exposing ``enable_mc_dropout()``.
        images: ``(B, 3, H, W)`` input batch.
        samples: Number of stochastic passes (>= 2).
        head: Output head, passed to :func:`probabilities`.

    Returns:
        Dict with ``prob``, ``epistemic``, ``aleatoric`` and ``total``.
    """
    if samples < 2:
        raise ValueError(f"MC dropout needs at least 2 samples, got {samples}")

    was_training = model.training
    if hasattr(model, "enable_mc_dropout"):
        model.enable_mc_dropout()
    else:  # pragma: no cover - every model here implements the hook
        model.train()

    entropies: list[torch.Tensor] = []
    probs: list[torch.Tensor] = []
    for _ in range(samples):
        p = probabilities(model(images), head)
        probs.append(p)
        entropies.append(predictive_entropy(p))

    model.train(was_training)

    mean_prob = torch.stack(probs).mean(dim=0)
    total = predictive_entropy(mean_prob)
    aleatoric = torch.stack(entropies).mean(dim=0)
    return {
        "prob": mean_prob,
        "total": total,
        "aleatoric": aleatoric,
        "epistemic": (total - aleatoric).clamp_min(0.0),
    }


@torch.no_grad()
def tta_predict(
    model: nn.Module, images: torch.Tensor, head: str = "evidential"
) -> torch.Tensor:
    """Average predictions over the four dihedral flips.

    Cheap, deterministic and always an improvement on a single pass; used at
    evaluation time when ``eval.tta`` is enabled so the headline numbers are
    not inflated by an option the baselines do not also get.
    """
    was_training = model.training
    model.eval()

    accumulated = probabilities(model(images), head)
    for dims in ((-1,), (-2,), (-1, -2)):
        flipped = torch.flip(images, dims=dims)
        p = probabilities(model(flipped), head)
        accumulated = accumulated + torch.flip(p, dims=dims)

    model.train(was_training)
    return accumulated / 4.0
