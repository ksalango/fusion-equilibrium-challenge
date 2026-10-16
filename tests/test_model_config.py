"""Tests of ``fusion_eq.model_config``: the base and arm files, the chain, the key schema,
value checks and the hash."""

from __future__ import annotations

import pytest

from fusion_eq.model_config import DEFAULT_CONFIG_DIR, ModelConfig


def test_base_defaults_match_the_specification():
    c = ModelConfig.load(DEFAULT_CONFIG_DIR / "base.yaml")
    assert c.get("features.set") == "set_a" and c.get("features.version") == 2
    assert c.get("loader.mirror") is True and c.get("loader.thomson_mask_rate") == 0.03 and c.get("loader.clip_bound") == 10
    assert c.get("loss.reference") == "basis" and c.get("loss.band_cells") == 2
    assert c.get("loss.frame_gamma") == 2 and c.get("loss.phase_weights") == [1, 1, 1]
    assert [t for t, v in c.get("loss.terms").items() if v["enabled"]] == ["flux"]
    assert c.get("model.name") == "linear_mlp" and c.get("model.hidden") == [512, 512, 512] and c.get("model.dropout") == 0.1
    assert c.get("model.warm_start") is True and c.get("model.bounded") is False
    assert (c.get("train.seed"), c.get("train.batch_frames"), c.get("train.max_epochs"), c.get("train.patience")) == (0, 16384, 200, 15)
    assert c.get("train.lr") == 3e-4 and c.get("train.weight_decay") == 1e-4


def test_arms_override_only_the_loss_terms():
    base = ModelConfig.load(DEFAULT_CONFIG_DIR / "base.yaml")
    f = ModelConfig.load(DEFAULT_CONFIG_DIR / "arm_f.yaml")
    g = ModelConfig.load(DEFAULT_CONFIG_DIR / "arm_g.yaml")
    assert [t for t, v in g.get("loss.terms").items() if v["enabled"]] == ["flux", "grad", "band"]
    assert f.hash() == base.hash() and g.hash() != f.hash()
    assert {k: v for k, v in g.data.items() if k != "loss"} == {k: v for k, v in base.data.items() if k != "loss"}
    assert len(g.sources) == 2 and g.sources[0].endswith("base.yaml")
    assert ModelConfig.load(DEFAULT_CONFIG_DIR / "arm_g.yaml").hash() == g.hash()


def test_unknown_key_and_bad_values_fail(tmp_path):
    (tmp_path / "base.yaml").write_text((DEFAULT_CONFIG_DIR / "base.yaml").read_text())
    (tmp_path / "typo.yaml").write_text("base: base.yaml\ntrain:\n  learning_rate: 0.1\n")
    with pytest.raises(ValueError, match="train.learning_rate"):
        ModelConfig.load(tmp_path / "typo.yaml")
    (tmp_path / "bad.yaml").write_text("base: base.yaml\nloss:\n  reference: full\n")
    with pytest.raises(ValueError, match="reference"):
        ModelConfig.load(tmp_path / "bad.yaml")
    (tmp_path / "none.yaml").write_text("base: base.yaml\nloss:\n  terms:\n    flux: {enabled: false}\n")
    with pytest.raises(ValueError, match="no loss term"):
        ModelConfig.load(tmp_path / "none.yaml")
    (tmp_path / "loop.yaml").write_text("base: loop.yaml\n")
    with pytest.raises(ValueError, match="repeats"):
        ModelConfig.load(tmp_path / "loop.yaml")


def test_with_values_and_virtual_space_keys():
    base = ModelConfig.load()
    c = base.with_values({"train.lr": 1e-3, "model.depth": 2, "model.width": 256})
    assert c.get("train.lr") == 1e-3 and c.get("model.hidden") == [256, 256]
    assert base.get("train.lr") == 3e-4 and c.hash() != base.hash()
    assert base.with_values({"model.width": 128}).get("model.hidden") == [128, 128, 128]
    with pytest.raises(KeyError):
        base.with_values({"train.momentum": 0.9})
    with pytest.raises(ValueError):
        base.with_values({"model.dropout": 1.5})


@pytest.mark.parametrize("arm, feature_set, n_columns", [("arm_setc_g", "set_c", 30), ("arm_setb10_g", "set_b10", 15)])
def test_comparison_arms_differ_from_arm_g_only_in_the_feature_set(arm, feature_set, n_columns):
    from fusion_eq import features as fe
    from fusion_eq.loader import model_columns

    g = ModelConfig.load(DEFAULT_CONFIG_DIR / "arm_g.yaml")
    c = ModelConfig.load(DEFAULT_CONFIG_DIR / f"{arm}.yaml")
    assert c.get("features.set") == feature_set and c.hash() != g.hash()
    assert c.with_values({"features.set": "set_a"}).hash() == g.hash()
    assert c.data["loss"] == g.data["loss"] and c.get("features.drop_groups") == []
    config = fe.FeatureConfig.load(fe.DEFAULT_CONFIG.with_name(f"features_v{c.get('features.version')}.yaml"))
    assert len(model_columns(config, c.get("features.set"), c.get("features.drop_groups"))) == n_columns
