"""Augmentation for supervised and consistency-based training.

The important design decision here concerns *view construction* for consistency
training. A teacher-student consistency loss is computed pixel-by-pixel, so the
two views must be spatially aligned. This module therefore applies the geometric
transform **once, shared by both views**, and lets the views differ only in
photometric perturbation and cut-out. Misaligning the geometry - a common bug -
turns the consistency term into noise and silently destroys the method.

Cut-out regions carry no evidence for the student, so they are reported as an
explicit ``valid`` mask and excluded from the consistency term rather than being
left to corrupt it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

#: ImageNet statistics. Dermoscopy encoders are routinely initialised from
#: ImageNet weights, so matching the input normalisation keeps transfer valid.
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)


@dataclass
class AugmentConfig:
    """Knobs for the augmentation pipeline."""

    #: Geometric (shared between views).
    hflip: bool = True
    vflip: bool = True
    rot90: bool = True
    rotate_degrees: float = 20.0
    scale_range: tuple[float, float] = (0.85, 1.20)
    translate_frac: float = 0.06

    #: Photometric, weak view (teacher input).
    weak_brightness: float = 0.10
    weak_contrast: float = 0.10

    #: Photometric, strong view (student input).
    strong_brightness: float = 0.40
    strong_contrast: float = 0.40
    strong_saturation: float = 0.40
    strong_blur_prob: float = 0.30
    strong_gray_prob: float = 0.15
    strong_posterize_prob: float = 0.15
    strong_noise_std: float = 0.03

    #: Cut-out on the strong view only.
    cutout_prob: float = 0.50
    cutout_count: int = 2
    cutout_size_frac: tuple[float, float] = (0.08, 0.25)


# --------------------------------------------------------------------------- #
# Geometric (shared across views)
# --------------------------------------------------------------------------- #


def apply_geometric(
    image: np.ndarray,
    mask: np.ndarray | None,
    rng: np.random.Generator,
    cfg: AugmentConfig,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Apply flips, 90-degree rotations and a random affine to image and mask.

    The mask is resampled with nearest-neighbour so labels stay in ``{0, 1}``.

    Args:
        image: ``uint8 (H, W, 3)``.
        mask: ``uint8 (H, W)`` or ``None`` (unlabelled data).
        rng: Source of randomness.
        cfg: Augmentation settings.

    Returns:
        The transformed ``(image, mask)`` pair.
    """
    if cfg.hflip and rng.random() < 0.5:
        image = image[:, ::-1]
        mask = None if mask is None else mask[:, ::-1]
    if cfg.vflip and rng.random() < 0.5:
        image = image[::-1]
        mask = None if mask is None else mask[::-1]
    if cfg.rot90:
        k = int(rng.integers(0, 4))
        if k:
            image = np.rot90(image, k)
            mask = None if mask is None else np.rot90(mask, k)

    image = np.ascontiguousarray(image)
    mask = None if mask is None else np.ascontiguousarray(mask)

    needs_affine = cfg.rotate_degrees > 0 or cfg.translate_frac > 0 or cfg.scale_range != (1.0, 1.0)
    if not needs_affine:
        return image, mask

    size = image.shape[0]
    angle = float(rng.uniform(-cfg.rotate_degrees, cfg.rotate_degrees))
    scale = float(rng.uniform(*cfg.scale_range))
    max_shift = cfg.translate_frac * size
    shift = rng.uniform(-max_shift, max_shift, size=2)

    img_pil = Image.fromarray(image).rotate(
        angle,
        resample=Image.BILINEAR,
        translate=(float(shift[0]), float(shift[1])),
        fillcolor=tuple(int(v) for v in image.reshape(-1, 3).mean(axis=0)),
    )
    if scale != 1.0:
        img_pil = _center_zoom(img_pil, scale, Image.BILINEAR)
    out_image = np.asarray(img_pil, dtype=np.uint8)

    out_mask = None
    if mask is not None:
        msk_pil = Image.fromarray(mask * 255).rotate(
            angle,
            resample=Image.NEAREST,
            translate=(float(shift[0]), float(shift[1])),
            fillcolor=0,
        )
        if scale != 1.0:
            msk_pil = _center_zoom(msk_pil, scale, Image.NEAREST)
        out_mask = (np.asarray(msk_pil) > 127).astype(np.uint8)

    return out_image, out_mask


def _center_zoom(img: Image.Image, scale: float, resample: int) -> Image.Image:
    """Zoom about the image centre, preserving the output size."""
    w, h = img.size
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = img.resize((new_w, new_h), resample)
    if scale >= 1.0:
        left, top = (new_w - w) // 2, (new_h - h) // 2
        return resized.crop((left, top, left + w, top + h))
    # Zooming out: pad back to size with edge-ish fill.
    canvas = Image.new(img.mode, (w, h), 0)
    canvas.paste(resized, ((w - new_w) // 2, (h - new_h) // 2))
    return canvas


# --------------------------------------------------------------------------- #
# Photometric
# --------------------------------------------------------------------------- #


def _jitter(img: Image.Image, rng: np.random.Generator, strength: float, kind: str) -> Image.Image:
    if strength <= 0:
        return img
    factor = 1.0 + float(rng.uniform(-strength, strength))
    enhancer = {
        "brightness": ImageEnhance.Brightness,
        "contrast": ImageEnhance.Contrast,
        "saturation": ImageEnhance.Color,
    }[kind]
    return enhancer(img).enhance(max(factor, 0.05))


def apply_weak_photometric(
    image: np.ndarray, rng: np.random.Generator, cfg: AugmentConfig
) -> np.ndarray:
    """Mild colour jitter - the teacher should see a near-clean image."""
    img = Image.fromarray(image)
    img = _jitter(img, rng, cfg.weak_brightness, "brightness")
    img = _jitter(img, rng, cfg.weak_contrast, "contrast")
    return np.asarray(img, dtype=np.uint8)


def apply_strong_photometric(
    image: np.ndarray, rng: np.random.Generator, cfg: AugmentConfig
) -> np.ndarray:
    """Aggressive appearance perturbation - the student's view."""
    img = Image.fromarray(image)
    img = _jitter(img, rng, cfg.strong_brightness, "brightness")
    img = _jitter(img, rng, cfg.strong_contrast, "contrast")
    img = _jitter(img, rng, cfg.strong_saturation, "saturation")
    if rng.random() < cfg.strong_blur_prob:
        img = img.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.4, 1.6))))
    if rng.random() < cfg.strong_gray_prob:
        img = ImageOps.grayscale(img).convert("RGB")
    if rng.random() < cfg.strong_posterize_prob:
        img = ImageOps.posterize(img, bits=int(rng.integers(4, 7)))

    out = np.asarray(img, dtype=np.float32)
    if cfg.strong_noise_std > 0:
        out = out + rng.normal(0.0, cfg.strong_noise_std * 255.0, size=out.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_cutout(
    image: np.ndarray, rng: np.random.Generator, cfg: AugmentConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Erase random rectangles.

    Returns:
        ``(image, valid)`` where ``valid`` is ``float32 (H, W)``, zero inside
        erased rectangles. Consistency losses must multiply by ``valid``: the
        student cannot infer a label for pixels it was never shown, so forcing
        agreement there injects pure noise.
    """
    size = image.shape[0]
    valid = np.ones((size, size), dtype=np.float32)
    if rng.random() >= cfg.cutout_prob or cfg.cutout_count <= 0:
        return image, valid

    out = image.copy()
    fill = out.reshape(-1, 3).mean(axis=0).astype(np.uint8)
    for _ in range(cfg.cutout_count):
        frac = float(rng.uniform(*cfg.cutout_size_frac))
        box = max(2, int(round(frac * size)))
        y0 = int(rng.integers(0, max(1, size - box)))
        x0 = int(rng.integers(0, max(1, size - box)))
        out[y0 : y0 + box, x0 : x0 + box] = fill
        valid[y0 : y0 + box, x0 : x0 + box] = 0.0
    return out, valid


# --------------------------------------------------------------------------- #
# Tensor conversion
# --------------------------------------------------------------------------- #


def to_tensor(
    image: np.ndarray,
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
) -> torch.Tensor:
    """``uint8 (H, W, 3)`` to normalised ``float32 (3, H, W)``."""
    arr = np.ascontiguousarray(image.transpose(2, 0, 1)).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr)
    mean_t = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
    std_t = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
    return (tensor - mean_t) / std_t


def denormalize(
    tensor: torch.Tensor,
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
) -> np.ndarray:
    """Invert :func:`to_tensor` for visualisation. Returns ``uint8 (H, W, 3)``."""
    mean_t = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
    std_t = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
    arr = (tensor.detach().cpu().float() * std_t + mean_t).clamp(0, 1)
    return (arr.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)


def mask_to_tensor(mask: np.ndarray) -> torch.Tensor:
    """``uint8 (H, W)`` in ``{0, 1}`` to ``int64 (H, W)``."""
    return torch.from_numpy(np.ascontiguousarray(mask)).long()
