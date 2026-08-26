"""Typed configuration objects with YAML ``_base_`` inheritance.

Every experiment in this repository is fully described by one YAML file, so a
result can always be traced back to the exact configuration that produced it.
Configs compose through a ``_base_`` key (resolved relative to the including
file), and any leaf value can be overridden from the command line with
``--set section.key=value``.
"""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

# --------------------------------------------------------------------------- #
# Dataclasses
# --------------------------------------------------------------------------- #


@dataclass
class DataConfig:
    """Dataset selection and the labelled / unlabelled split."""

    name: str = "synthetic"  # synthetic | isic2018 | ph2
    root: str = "data"
    image_size: int = 128
    labeled_fraction: float = 0.10
    #: Absolute cap on labelled training images (None -> use the fraction).
    labeled_count: int | None = None
    train_size: int = 400  # synthetic only
    val_size: int = 60  # synthetic only
    test_size: int = 100  # synthetic only
    num_workers: int = 0
    batch_size: int = 8
    #: Unlabelled batch is ``mu`` times the labelled batch (FixMatch convention).
    mu: int = 2
    #: Cache decoded arrays in RAM. Safe for a few thousand 128px images.
    cache: bool = True


@dataclass
class ModelConfig:
    """Architecture selection."""

    name: str = "separable_unet"  # separable_unet | separable_unet_tiny | unet
    in_channels: int = 3
    #: Two logits (background, lesion); evidential heads read these as evidence.
    num_classes: int = 2
    width: int = 24
    depth: int = 4
    dropout: float = 0.10
    #: Gated axial attention at the bottleneck (cheap global context).
    axial_attention: bool = True


@dataclass
class LossConfig:
    """Loss composition for the supervised branch."""

    #: ``ce_dice`` (standard) or ``evidential`` (Dirichlet NLL + KL regulariser).
    supervised: str = "evidential"
    dice_weight: float = 1.0
    ce_weight: float = 1.0
    #: Weight of the evidential KL-to-uniform term at full annealing.
    kl_weight: float = 0.10
    #: Epochs over which the KL term is linearly annealed in.
    kl_anneal_epochs: int = 20
    #: Boundary-aware term weight (loss weighted by distance to the boundary).
    boundary_weight: float = 0.0
    label_smoothing: float = 0.0


@dataclass
class SemiConfig:
    """Semi-supervised consistency branch.

    ``method`` selects the ideology under test:

    ``none``
        Supervised only - the labelled-data lower bound.
    ``mean_teacher``
        Unweighted MSE consistency against an EMA teacher
        (Tarvainen & Valpola, 2017).
    ``fixmatch``
        Hard pseudo-labels retained above a fixed confidence threshold - the
        prior-work ideology this project argues against.
    ``evidential``
        Ours: soft per-pixel consistency weighted by ``1 - vacuity`` derived
        from the teacher's Dirichlet evidence, with no hard threshold.
    """

    method: str = "evidential"
    #: Peak consistency weight after ramp-up.
    consistency_weight: float = 1.0
    #: Sigmoid ramp-up length in epochs (Laine & Aila, 2017).
    rampup_epochs: int = 15
    #: EMA decay for the teacher network.
    ema_decay: float = 0.99
    #: FixMatch confidence threshold (used by ``fixmatch`` only).
    threshold: float = 0.95
    #: Ours: pixels whose vacuity exceeds this are dropped (soft floor).
    vacuity_cutoff: float = 0.80
    #: Ours, ablation switch. When False the per-pixel weight becomes uniform
    #: while everything else about the method is held fixed - same soft-target
    #: cross-entropy, same dissonance tempering, same EMA teacher. This is what
    #: isolates the *weighting rule*; switching to ``mean_teacher`` instead would
    #: also change the loss form and the target, confounding the ablation.
    use_vacuity_gate: bool = True
    #: Sharpening temperature applied to teacher probabilities.
    temperature: float = 0.50
    #: Apply strong augmentation to the student view of unlabelled data.
    strong_aug: bool = True


@dataclass
class OptimConfig:
    """Optimiser and schedule."""

    name: str = "adamw"
    lr: float = 3e-3
    weight_decay: float = 1e-4
    epochs: int = 30
    #: Gradient steps per epoch, held **fixed** across labelled fractions so a
    #: sweep varies the label count and nothing else. ``0`` falls back to one
    #: pass over the labelled set, which is only appropriate for single runs.
    steps_per_epoch: int = 32
    scheduler: str = "cosine"  # cosine | none
    warmup_epochs: int = 2
    grad_clip: float = 1.0


@dataclass
class EvalConfig:
    """Evaluation, calibration and statistics."""

    #: Decision threshold on the lesion probability.
    threshold: float = 0.50
    #: Bins for the expected-calibration-error estimator.
    calibration_bins: int = 15
    #: Bootstrap resamples for confidence intervals.
    bootstrap: int = 2000
    #: Monte-Carlo dropout samples (0 disables; the evidential head needs none).
    mc_dropout: int = 0
    #: Test-time augmentation (flips) for the final prediction.
    tta: bool = False
    #: Report the EMA teacher weights instead of the student.
    use_ema: bool = True


@dataclass
class RunConfig:
    """Bookkeeping."""

    name: str = "default"
    seed: int = 1337
    device: str = "auto"  # auto | cpu | cuda
    out_dir: str = "results"
    ckpt_dir: str = "checkpoints"
    log_every: int = 20
    save_best: bool = True
    #: Metric used to select the best checkpoint.
    monitor: str = "dice"
    deterministic: bool = True


@dataclass
class Config:
    """Root configuration object."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    semi: SemiConfig = field(default_factory=SemiConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    run: RunConfig = field(default_factory=RunConfig)

    # -- serialisation ---------------------------------------------------- #

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_yaml(), encoding="utf-8")
        return path

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return json.dumps(self.to_dict(), indent=2, default=str)


#: Maps a root-level config key to the dataclass that models it.
NESTED: dict[str, type] = {
    "data": DataConfig,
    "model": ModelConfig,
    "loss": LossConfig,
    "semi": SemiConfig,
    "optim": OptimConfig,
    "eval": EvalConfig,
    "run": RunConfig,
}


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base`` without mutating either."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _read_yaml_with_bases(path: Path, seen: tuple[Path, ...] = ()) -> dict[str, Any]:
    """Read a YAML file, resolving its ``_base_`` chain depth-first."""
    path = path.resolve()
    if path in seen:
        chain = " -> ".join(p.name for p in (*seen, path))
        raise ValueError("Circular _base_ reference: " + chain)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config root must be a mapping: {path}")

    bases = raw.pop("_base_", [])
    if isinstance(bases, str):
        bases = [bases]

    merged: dict[str, Any] = {}
    for base in bases:
        base_path = (path.parent / base).resolve()
        merged = _deep_merge(merged, _read_yaml_with_bases(base_path, (*seen, path)))
    return _deep_merge(merged, raw)


def _coerce(value: Any, annotation: Any) -> Any:
    """Best-effort cast of a YAML/CLI scalar onto a dataclass field type."""
    text = str(annotation)
    optional = "None" in text
    if value is None:
        return None
    if optional and isinstance(value, str) and value.strip().lower() in {"null", "none", ""}:
        return None
    if "bool" in text:
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}
    if "int" in text and "float" not in text:
        return int(float(value))
    if "float" in text:
        return float(value)
    if "str" in text:
        return str(value)
    return value


def _from_dict(cls: type, data: dict[str, Any]) -> Any:
    """Instantiate a (possibly nested) dataclass from a plain dict, strictly."""
    kwargs: dict[str, Any] = {}
    known = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        raise ValueError(
            f"Unknown key(s) for {cls.__name__}: {sorted(unknown)}. Valid keys: {sorted(known)}"
        )
    for name, f in known.items():
        if name not in data:
            continue
        value = data[name]
        if isinstance(value, dict):
            nested_cls = NESTED.get(name)
            if nested_cls is None:
                raise ValueError(f"Unexpected mapping for scalar field {name!r}")
            kwargs[name] = _from_dict(nested_cls, value)
        else:
            kwargs[name] = _coerce(value, f.type)
    return cls(**kwargs)


def _parse_override(item: str) -> tuple[list[str], str]:
    if "=" not in item:
        raise ValueError(f"Override must look like section.key=value, got {item!r}")
    key, value = item.split("=", 1)
    return key.strip().split("."), value.strip()


def apply_overrides(data: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    """Apply ``--set a.b=c`` style overrides onto a raw config dict."""
    out = copy.deepcopy(data)
    for item in overrides or []:
        path, value = _parse_override(item)
        cursor = out
        for part in path[:-1]:
            nxt = cursor.setdefault(part, {})
            if not isinstance(nxt, dict):
                raise ValueError("Cannot descend into scalar at " + ".".join(path))
            cursor = nxt
        cursor[path[-1]] = yaml.safe_load(value)
    return out


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> Config:
    """Load a config from YAML (with ``_base_`` inheritance) plus CLI overrides.

    Args:
        path: YAML file. ``None`` yields the dataclass defaults.
        overrides: list of ``"section.key=value"`` strings.

    Returns:
        A validated :class:`Config`.
    """
    raw = _read_yaml_with_bases(Path(path)) if path is not None else {}
    raw = apply_overrides(raw, overrides)
    return _from_dict(Config, raw)
