"""Procedural dermoscopy generator.

Real dermoscopic archives (ISIC, PH2) require registration and a multi-gigabyte
download, which makes a repository hard to verify. This module synthesises
images with the *failure modes that matter* for uncertainty research, so the
whole pipeline - training, calibration, statistics, figures - runs offline and
still tests something meaningful:

* **Soft, variable-sharpness boundaries.** Each lesion has an edge softness
  drawn per sample. Soft edges create genuine aleatoric ambiguity, which is
  exactly the regime where a hard confidence threshold discards usable pixels.
* **Low-contrast lesions.** Contrast is sampled down to near-invisible, so a
  fraction of the set is legitimately hard rather than uniformly easy.
* **Occluders.** Hair strands, ruler ticks and specular highlights cross the
  lesion boundary, the standard nuisance factors in dermoscopy.
* **Irregular borders.** The radial boundary is perturbed by a low-order
  Fourier series plus optional satellite blobs, giving non-convex shapes that
  punish models that only learn ellipses.

The generator is a pure function of its seed, so a dataset is reproducible from
an integer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter

# --------------------------------------------------------------------------- #
# Appearance priors
# --------------------------------------------------------------------------- #

#: Plausible skin tones in RGB, spanning Fitzpatrick I-VI.
SKIN_TONES: tuple[tuple[int, int, int], ...] = (
    (247, 219, 197),
    (240, 202, 172),
    (226, 180, 145),
    (198, 148, 110),
    (160, 112, 78),
    (114, 76, 52),
    (78, 52, 38),
)

#: Lesion pigment directions (multiplicative tint applied to the skin tone).
LESION_TINTS: tuple[tuple[float, float, float], ...] = (
    (0.55, 0.42, 0.38),  # dark brown / melanocytic
    (0.68, 0.50, 0.45),  # mid brown
    (0.45, 0.35, 0.40),  # blue-grey / nodular
    (0.85, 0.55, 0.50),  # erythematous / pink
    (0.35, 0.28, 0.30),  # near-black
)


@dataclass
class LesionParams:
    """The latent factors behind one synthesised image.

    Stored alongside every sample so difficulty can be correlated with error -
    the ablation in ``04_ablations_and_calibration.ipynb`` uses ``contrast``
    and ``edge_softness`` as the difficulty axes.
    """

    #: Peak absolute pigment contrast against the surrounding skin, in [0, 1].
    contrast: float
    #: Boundary blur sigma in pixels; larger means a more ambiguous border.
    edge_softness: float
    #: Lesion area as a fraction of the image.
    area_fraction: float
    #: Number of hair strands drawn over the image.
    n_hairs: int
    #: Whether ruler ticks were drawn along an image edge.
    has_ruler: bool
    #: Number of satellite blobs fused into the main lesion.
    n_satellites: int

    def difficulty(self) -> float:
        """A scalar difficulty score in roughly [0, 1] (higher is harder).

        Combines low contrast, soft edges and occlusion, which are the three
        drivers of boundary ambiguity in dermoscopy.
        """
        contrast_term = 1.0 - min(self.contrast / 0.45, 1.0)
        edge_term = min(self.edge_softness / 6.0, 1.0)
        occlusion_term = min(self.n_hairs / 30.0, 1.0)
        return float(0.5 * contrast_term + 0.35 * edge_term + 0.15 * occlusion_term)


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #


def _radial_field(size: int, cy: float, cx: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (radius, angle) fields for every pixel, relative to a centre."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    dy = yy - cy
    dx = xx - cx
    return np.sqrt(dy * dy + dx * dx), np.arctan2(dy, dx)


def _fourier_radius(
    angle: np.ndarray,
    base_radius: float,
    rng: np.random.Generator,
    n_harmonics: int = 5,
    roughness: float = 0.22,
) -> np.ndarray:
    """Perturb a circle's radius with a low-order Fourier series.

    A handful of harmonics with decaying amplitude produces the lobed,
    asymmetric outlines characteristic of melanocytic lesions without the
    high-frequency noise that a random field would introduce.
    """
    radius = np.full_like(angle, base_radius, dtype=np.float32)
    for k in range(2, 2 + n_harmonics):
        amplitude = roughness * base_radius * rng.uniform(0.3, 1.0) / k
        phase = rng.uniform(0, 2 * np.pi)
        radius = radius + amplitude * np.cos(k * angle + phase).astype(np.float32)
    return np.maximum(radius, base_radius * 0.35)


def _soft_blob(
    size: int,
    rng: np.random.Generator,
    base_radius: float,
    cy: float,
    cx: float,
    edge_softness: float,
) -> np.ndarray:
    """One anisotropic, Fourier-perturbed blob as a soft [0, 1] field."""
    # Anisotropy: squash along a random axis so lesions are not all circular.
    theta = rng.uniform(0, np.pi)
    aspect = rng.uniform(0.62, 1.0)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    dy, dx = yy - cy, xx - cx
    rot_y = dy * np.cos(theta) + dx * np.sin(theta)
    rot_x = -dy * np.sin(theta) + dx * np.cos(theta)
    warped = np.sqrt((rot_y / aspect) ** 2 + rot_x**2)
    angle = np.arctan2(rot_y, rot_x)

    boundary = _fourier_radius(angle, base_radius, rng)
    # Signed distance to the boundary, converted to a soft membership by a
    # logistic whose temperature is the sampled edge softness.
    signed = (boundary - warped) / max(edge_softness, 0.6)
    return 1.0 / (1.0 + np.exp(-signed))


# --------------------------------------------------------------------------- #
# Nuisance structures
# --------------------------------------------------------------------------- #


def _draw_hairs(
    image: np.ndarray, rng: np.random.Generator, n_hairs: int
) -> np.ndarray:
    """Overlay dark, tapering quadratic-Bezier strands (dermoscopic hair)."""
    if n_hairs <= 0:
        return image
    size = image.shape[0]
    hair_layer = np.zeros((size, size), dtype=np.float32)
    t = np.linspace(0.0, 1.0, max(size * 2, 64), dtype=np.float32)

    for _ in range(n_hairs):
        p0 = rng.uniform(-0.15, 1.15, size=2) * size
        p2 = rng.uniform(-0.15, 1.15, size=2) * size
        # Control point offset perpendicular-ish to the chord gives a curve.
        mid = (p0 + p2) / 2.0
        p1 = mid + rng.normal(0.0, size * 0.22, size=2)

        pts = (
            np.outer((1 - t) ** 2, p0)
            + np.outer(2 * (1 - t) * t, p1)
            + np.outer(t**2, p2)
        )
        ys = np.clip(pts[:, 0].astype(np.int32), 0, size - 1)
        xs = np.clip(pts[:, 1].astype(np.int32), 0, size - 1)
        # Opacity tapers along the strand so hairs fade in and out.
        opacity = rng.uniform(0.35, 0.95) * (0.4 + 0.6 * np.sin(np.pi * t))
        np.maximum.at(hair_layer, (ys, xs), opacity.astype(np.float32))

    # Thicken and soften: hairs are 1-3 px wide with anti-aliased edges.
    hair_layer = gaussian_filter(hair_layer, sigma=rng.uniform(0.6, 1.3))
    hair_layer = np.clip(hair_layer * rng.uniform(1.6, 2.6), 0.0, 1.0)
    hair_colour = np.array([0.12, 0.09, 0.08], dtype=np.float32)
    alpha = hair_layer[..., None]
    return image * (1.0 - alpha) + hair_colour * alpha


def _draw_ruler(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Draw calibrated ruler ticks along one image edge."""
    size = image.shape[0]
    layer = np.zeros((size, size), dtype=np.float32)
    edge = rng.integers(0, 4)
    spacing = int(rng.integers(max(size // 16, 4), max(size // 8, 6)))
    thickness = max(1, size // 96)
    long_tick = max(size // 12, 5)
    short_tick = max(size // 22, 3)

    for i, pos in enumerate(range(spacing, size - spacing, spacing)):
        length = long_tick if i % 5 == 0 else short_tick
        layer[:length, pos : pos + thickness] = 1.0

    # Rotate the tick strip onto the chosen edge.
    layer = np.rot90(layer, k=int(edge))
    layer = gaussian_filter(layer, sigma=0.7)
    alpha = np.clip(layer * rng.uniform(0.7, 1.0), 0.0, 1.0)[..., None]
    tick_colour = np.array([0.92, 0.92, 0.90], dtype=np.float32)
    return image * (1.0 - alpha) + tick_colour * alpha


def _add_specular(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Add immersion-fluid specular highlights (small bright gaussian spots)."""
    size = image.shape[0]
    n_spots = int(rng.integers(0, 5))
    if n_spots == 0:
        return image
    layer = np.zeros((size, size), dtype=np.float32)
    for _ in range(n_spots):
        cy, cx = rng.uniform(0.1, 0.9, size=2) * size
        radius = rng.uniform(size * 0.012, size * 0.045)
        rr, _ = _radial_field(size, cy, cx)
        layer = np.maximum(layer, np.exp(-0.5 * (rr / radius) ** 2))
    alpha = np.clip(layer * rng.uniform(0.25, 0.55), 0.0, 1.0)[..., None]
    return np.clip(image + alpha * (1.0 - image), 0.0, 1.0)


def _illumination(size: int, rng: np.random.Generator) -> np.ndarray:
    """Smooth multiplicative vignette plus a low-frequency lighting gradient."""
    cy, cx = (rng.uniform(0.35, 0.65, size=2) * size).tolist()
    rr, _ = _radial_field(size, cy, cx)
    vignette = 1.0 - rng.uniform(0.10, 0.38) * (rr / (size * 0.75)) ** 2

    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) / size
    gradient = 1.0 + rng.uniform(-0.12, 0.12) * yy + rng.uniform(-0.12, 0.12) * xx
    return np.clip(vignette * gradient, 0.25, 1.35).astype(np.float32)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def generate_sample(
    seed: int,
    size: int = 128,
    hard_fraction: float = 0.35,
) -> tuple[np.ndarray, np.ndarray, LesionParams]:
    """Synthesise one dermoscopy-like image and its binary lesion mask.

    Args:
        seed: Any integer. The output is a deterministic function of it.
        size: Output edge length in pixels (square).
        hard_fraction: Probability that a sample is drawn from the hard regime
            (low contrast, soft edge, heavy hair occlusion).

    Returns:
        ``(image, mask, params)`` where ``image`` is ``uint8 (size, size, 3)``,
        ``mask`` is ``uint8 (size, size)`` in ``{0, 1}``, and ``params`` records
        the latent difficulty factors.
    """
    rng = np.random.default_rng(seed)
    is_hard = rng.random() < hard_fraction

    # --- latent factors -------------------------------------------------- #
    if is_hard:
        contrast = float(rng.uniform(0.06, 0.20))
        edge_softness = float(rng.uniform(2.5, 7.0) * size / 128.0)
        n_hairs = int(rng.integers(10, 36))
    else:
        contrast = float(rng.uniform(0.22, 0.62))
        edge_softness = float(rng.uniform(0.7, 2.8) * size / 128.0)
        n_hairs = int(rng.integers(0, 14))

    area_fraction = float(rng.uniform(0.04, 0.34))
    base_radius = float(np.sqrt(area_fraction / np.pi) * size)
    n_satellites = int(rng.integers(0, 3))
    has_ruler = bool(rng.random() < 0.18)
    params = LesionParams(
        contrast=contrast,
        edge_softness=edge_softness,
        area_fraction=area_fraction,
        n_hairs=n_hairs,
        has_ruler=has_ruler,
        n_satellites=n_satellites,
    )

    # --- skin background ------------------------------------------------- #
    tone = np.array(SKIN_TONES[rng.integers(len(SKIN_TONES))], dtype=np.float32) / 255.0
    tone = np.clip(tone * rng.uniform(0.92, 1.08, size=3), 0.0, 1.0)
    image = np.broadcast_to(tone, (size, size, 3)).copy()

    # Mottled skin texture: low-frequency noise, not white noise.
    texture = gaussian_filter(
        rng.normal(0.0, 1.0, size=(size, size)).astype(np.float32),
        sigma=max(size / 24.0, 1.5),
    )
    texture /= max(float(np.abs(texture).max()), 1e-6)
    image = np.clip(image + rng.uniform(0.02, 0.07) * texture[..., None], 0.0, 1.0)

    # --- lesion ---------------------------------------------------------- #
    cy = float(rng.uniform(0.32, 0.68) * size)
    cx = float(rng.uniform(0.32, 0.68) * size)
    soft = _soft_blob(size, rng, base_radius, cy, cx, edge_softness)

    for _ in range(n_satellites):
        # Satellites sit just outside the main body, partially fused with it.
        angle = rng.uniform(0, 2 * np.pi)
        distance = base_radius * rng.uniform(0.75, 1.30)
        sat = _soft_blob(
            size,
            rng,
            base_radius * rng.uniform(0.22, 0.45),
            cy + distance * np.sin(angle),
            cx + distance * np.cos(angle),
            edge_softness * rng.uniform(0.8, 1.4),
        )
        soft = np.maximum(soft, sat)

    tint = np.array(LESION_TINTS[rng.integers(len(LESION_TINTS))], dtype=np.float32)
    pigment = tone * tint
    # Interior pigment variation: real lesions are not flat in colour.
    interior = gaussian_filter(
        rng.normal(0.0, 1.0, size=(size, size)).astype(np.float32),
        sigma=max(size / 16.0, 2.0),
    )
    interior /= max(float(np.abs(interior).max()), 1e-6)
    pigment_field = np.clip(
        pigment[None, None, :] + 0.10 * interior[..., None], 0.0, 1.0
    )

    alpha = (soft * contrast / max(float(soft.max()), 1e-6))[..., None]
    image = image * (1.0 - alpha) + pigment_field * alpha

    # --- nuisance structures --------------------------------------------- #
    image = image * _illumination(size, rng)[..., None]
    image = _draw_hairs(np.clip(image, 0.0, 1.0), rng, n_hairs)
    if has_ruler:
        image = _draw_ruler(image, rng)
    image = _add_specular(image, rng)

    # Sensor noise and a touch of optical blur.
    image = gaussian_filter(image, sigma=(rng.uniform(0.3, 0.9), rng.uniform(0.3, 0.9), 0))
    image = image + rng.normal(0.0, rng.uniform(0.004, 0.022), size=image.shape)
    image = np.clip(image, 0.0, 1.0)

    # Ground truth is the 0.5 level set of the *pre-occlusion* soft field: the
    # annotator sees through hair, so occluders must not change the label.
    mask = (soft >= 0.5).astype(np.uint8)

    return (image * 255.0).round().astype(np.uint8), mask, params


def generate_dataset(
    n: int,
    size: int = 128,
    seed: int = 0,
    hard_fraction: float = 0.35,
) -> tuple[np.ndarray, np.ndarray, list[LesionParams]]:
    """Synthesise ``n`` samples.

    Seeds are derived as ``seed * 1_000_003 + i``, a large odd stride, so
    train/val/test splits built from different base seeds never collide.

    Returns:
        ``(images, masks, params)`` with shapes ``(n, size, size, 3)`` uint8 and
        ``(n, size, size)`` uint8.
    """
    images = np.empty((n, size, size, 3), dtype=np.uint8)
    masks = np.empty((n, size, size), dtype=np.uint8)
    params: list[LesionParams] = []
    for i in range(n):
        img, msk, prm = generate_sample(seed * 1_000_003 + i, size, hard_fraction)
        images[i] = img
        masks[i] = msk
        params.append(prm)
    return images, masks, params
