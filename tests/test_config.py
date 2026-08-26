"""Config loading: inheritance, overrides and strict validation."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from evissl.config import Config, apply_overrides, load_config

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def test_defaults_are_usable():
    cfg = load_config()
    assert isinstance(cfg, Config)
    assert cfg.model.num_classes == 2
    assert cfg.data.image_size > 0
    assert 0.0 < cfg.data.labeled_fraction <= 1.0


@pytest.mark.parametrize("path", sorted(CONFIG_DIR.glob("*.yaml")), ids=lambda p: p.name)
def test_every_shipped_config_loads(path):
    """Every config in the repo must resolve - a broken one is a broken claim."""
    cfg = load_config(path)
    assert cfg.run.name
    assert cfg.semi.method in {"none", "mean_teacher", "fixmatch", "evidential"}
    assert cfg.loss.supervised in {"ce_dice", "evidential"}
    assert cfg.optim.epochs >= 1


def test_base_inheritance_merges_deeply(tmp_path):
    (tmp_path / "parent.yaml").write_text(
        yaml.safe_dump({"data": {"image_size": 96, "batch_size": 4}, "run": {"seed": 7}})
    )
    (tmp_path / "child.yaml").write_text(
        "_base_: parent.yaml\n" + yaml.safe_dump({"data": {"batch_size": 16}})
    )
    cfg = load_config(tmp_path / "child.yaml")
    # Child overrides one leaf; the sibling leaf and the other section survive.
    assert cfg.data.batch_size == 16
    assert cfg.data.image_size == 96
    assert cfg.run.seed == 7


def test_base_chain_is_transitive(tmp_path):
    (tmp_path / "a.yaml").write_text(yaml.safe_dump({"optim": {"epochs": 5, "lr": 0.1}}))
    (tmp_path / "b.yaml").write_text("_base_: a.yaml\n" + yaml.safe_dump({"optim": {"lr": 0.2}}))
    (tmp_path / "c.yaml").write_text("_base_: b.yaml\n" + yaml.safe_dump({"run": {"name": "c"}}))
    cfg = load_config(tmp_path / "c.yaml")
    assert (cfg.optim.epochs, cfg.optim.lr, cfg.run.name) == (5, 0.2, "c")


def test_circular_base_is_rejected(tmp_path):
    (tmp_path / "x.yaml").write_text("_base_: y.yaml\n")
    (tmp_path / "y.yaml").write_text("_base_: x.yaml\n")
    with pytest.raises(ValueError, match="Circular"):
        load_config(tmp_path / "x.yaml")


def test_missing_base_file_is_reported(tmp_path):
    (tmp_path / "x.yaml").write_text("_base_: nope.yaml\n")
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "x.yaml")


def test_unknown_key_is_rejected_not_ignored(tmp_path):
    """A typo must fail loudly; silently ignoring it would void an experiment."""
    (tmp_path / "bad.yaml").write_text(yaml.safe_dump({"optim": {"epoch": 5}}))
    with pytest.raises(ValueError, match="Unknown key"):
        load_config(tmp_path / "bad.yaml")


def test_overrides_apply_and_coerce():
    cfg = load_config(
        CONFIG_DIR / "evidential.yaml",
        ["optim.epochs=7", "data.image_size=96", "semi.temperature=0.25",
         "model.axial_attention=false", "run.name=override_test"],
    )
    assert cfg.optim.epochs == 7
    assert cfg.data.image_size == 96
    assert cfg.semi.temperature == pytest.approx(0.25)
    assert cfg.model.axial_attention is False
    assert cfg.run.name == "override_test"


def test_override_can_create_a_missing_section():
    out = apply_overrides({}, ["run.seed=99"])
    assert out == {"run": {"seed": 99}}


def test_malformed_override_is_rejected():
    with pytest.raises(ValueError, match="section.key=value"):
        apply_overrides({}, ["optim.epochs"])


def test_optional_int_accepts_null():
    cfg = load_config(None, None)
    assert cfg.data.labeled_count is None
    cfg = load_config(CONFIG_DIR / "base.yaml", ["data.labeled_count=25"])
    assert cfg.data.labeled_count == 25


def test_roundtrip_through_yaml(tmp_path):
    cfg = load_config(CONFIG_DIR / "evidential.yaml")
    path = cfg.save(tmp_path / "saved.yaml")
    assert load_config(path).to_dict() == cfg.to_dict()
