"""Tests of the robust feature scaler and of the scaled feature matrix."""

from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd
import pytest

from fusion_eq import feature_checks as fc, features as fe
from fusion_eq.feature_scaler import RobustScaler
from tests.test_features import CONFIG, CONFIG_V2, make_frames, point_table, resources


@pytest.fixture(scope="module")
def built() -> pd.DataFrame:
    return fe.build_features(make_frames(n_shots=4, n_frames=10), "DIII-D", CONFIG, resources(point_table()))


def test_fit_on_originals_and_mirrors_centres_odd_features_and_shares_group_scales(built):
    mirrored = fe.mirror_features(built, CONFIG)
    scaler = RobustScaler.fit(built, mirrored, CONFIG)
    assert scaler.columns == tuple(CONFIG.all_columns()) and scaler.n_rows == 2 * len(built)
    for name in ("coil_c01", "coil_c11", "coil_c21", "coil_c03"):
        assert scaler.source[name][0] == 0.0
    # One median and one interquartile range for the whole coil-point group, over all values pooled.
    points = CONFIG.columns_of("coil_pt")
    stats = {tuple(scaler.source[c]) for c in points}
    assert len(stats) == 1
    pooled = np.concatenate([built[points].to_numpy().ravel(), mirrored[points].to_numpy().ravel()])
    q25, q50, q75 = np.percentile(pooled, [25, 50, 75])
    assert stats.pop() == pytest.approx((q50, q75 - q25))
    # An ungrouped feature has its own statistics; ts_valid is not scaled.
    q = built["q_cyl"].dropna().to_numpy()
    q25, q50, q75 = np.percentile(np.concatenate([q, q]), [25, 50, 75])
    assert scaler.source["q_cyl"] == pytest.approx([q50, q75 - q25])
    assert scaler.source["ts_valid"] == [0.0, 1.0]


def test_fit_rules_reproduces_fit_and_serves_in_memory_variants(built):
    mirrored = fe.mirror_features(built, CONFIG)
    reference = RobustScaler.fit(built, mirrored, CONFIG)
    same = RobustScaler.fit_rules(built, mirrored, fc.variant_rules(CONFIG, CONFIG.all_columns()))
    assert same.to_dict() == reference.to_dict()
    # Variant: ip_frac unscaled, and a column outside the configuration added with a robust rule.
    original, mirror = built.assign(extra=2.0 * built["q_cyl"]), mirrored.assign(extra=2.0 * mirrored["q_cyl"])
    columns = ["coil_c10", "ip_frac", "extra"]
    variant = RobustScaler.fit_rules(original, mirror, fc.variant_rules(CONFIG, columns, ["ip_frac"], {"extra": {}}))
    assert variant.columns == tuple(columns) and variant.n_rows == 2 * len(built)
    assert variant.source["ip_frac"] == [0.0, 1.0] and variant.source["coil_c10"] == reference.source["coil_c10"]
    assert variant.source["extra"] == pytest.approx([2.0 * v for v in reference.source["q_cyl"]])
    x = variant.transform(original, "DIII-D")
    finite = built["ip_frac"].notna().to_numpy()
    assert np.array_equal(x[finite, 1], built["ip_frac"].to_numpy()[finite])


def test_transform_scales_with_the_training_statistics_and_sets_nan_to_zero(built):
    scaler = RobustScaler.fit(built, fe.mirror_features(built, CONFIG), CONFIG)
    x = scaler.transform(built, "DIII-D")
    j = scaler.columns.index("q_cyl")
    center, scale = scaler.source["q_cyl"]
    finite = built["q_cyl"].notna().to_numpy()
    assert np.allclose(x[finite, j], (built["q_cyl"].to_numpy()[finite] - center) / scale)
    k = scaler.columns.index("ts_beta_p")
    assert built["ts_beta_p"].isna().any() and np.all(x[built["ts_beta_p"].isna().to_numpy(), k] == 0.0)
    assert np.isfinite(x).all()
    # MAST rows use the same DIII-D statistics.
    assert np.array_equal(scaler.transform(built, "MAST"), x)


def test_a_constant_feature_raises_and_a_transform_applies_before_scaling(built):
    constant = built.copy()
    constant["q_cyl"] = 3.0
    with pytest.raises(ValueError, match="zero interquartile range"):
        RobustScaler.fit(constant, constant, CONFIG, columns=["q_cyl"])
    empty = built.copy()
    empty["q_cyl"] = np.nan
    with pytest.raises(ValueError, match="no finite value"):
        RobustScaler.fit(empty, empty, CONFIG, columns=["q_cyl"])
    d = copy.deepcopy(CONFIG.to_dict())
    for f in d["features"]:
        if f["name"] == "q_cyl":
            f["transform"] = "log1p"
    config = fe.FeatureConfig.from_dict(d)
    scaler = RobustScaler.fit(built, built, config, columns=["q_cyl"])
    logged = np.log1p(built["q_cyl"].dropna().to_numpy())
    q25, q50, q75 = np.percentile(np.concatenate([logged, logged]), [25, 50, 75])
    assert scaler.source["q_cyl"] == pytest.approx([q50, q75 - q25])
    x = scaler.transform(built, "DIII-D")[:, 0]
    assert np.allclose(x[built["q_cyl"].notna().to_numpy()], (logged - q50) / (q75 - q25))


def test_target_scaling_uses_the_machine_statistics_only_when_declared(built):
    d = copy.deepcopy(CONFIG.to_dict())
    for f in d["features"]:
        if f["name"] == "q_cyl":
            f["scale_with"] = "target"
    config = fe.FeatureConfig.from_dict(d)
    scaler = RobustScaler.fit(built, built, config, columns=["q_cyl", "ip_frac"])
    mast = built.copy()
    mast["q_cyl"] = built["q_cyl"] * 2.0 + 1.0
    with pytest.raises(KeyError, match="target-scaled"):
        scaler.transform(mast, "MAST")
    scaler.fit_target("MAST", mast)
    assert set(scaler.target["MAST"]) == {"q_cyl"}
    q = mast["q_cyl"].dropna().to_numpy()
    q25, q50, q75 = np.percentile(q, [25, 50, 75])
    assert scaler.statistics("q_cyl", "MAST") == pytest.approx([q50, q75 - q25])
    assert scaler.statistics("q_cyl", "DIII-D") == scaler.source["q_cyl"]
    assert scaler.statistics("ip_frac", "MAST") == scaler.source["ip_frac"]


def test_save_and_load_round_trip(built, tmp_path):
    scaler = RobustScaler.fit(built, fe.mirror_features(built, CONFIG), CONFIG)
    path = tmp_path / "v1" / "scaler.json"
    scaler.save(path)
    again = RobustScaler.load(path)
    assert again.to_dict() == scaler.to_dict()
    assert np.array_equal(again.transform(built, "DIII-D"), scaler.transform(built, "DIII-D"))
    stored = json.loads(path.read_text())
    assert stored["columns"] == list(scaler.columns) and stored["source"]["q_cyl"] == scaler.source["q_cyl"]


def test_feature_matrix_reads_mirrors_and_scales(built, tmp_path):
    root = tmp_path / "features"
    fe.features_dir(CONFIG.version, root).mkdir(parents=True)
    built.to_parquet(fe.features_path("diii_d_train", CONFIG.version, root), index=False)
    scaler = RobustScaler.fit(built, fe.mirror_features(built, CONFIG), CONFIG)
    x, columns = fe.feature_matrix("diii_d_train", ["s1", "s3"], "set_b", config=CONFIG, scaler=scaler, root=root)
    rows = built["shot"].isin(["s1", "s3"]).to_numpy()
    assert columns == CONFIG.set_columns("set_b") and x.dtype == np.float32 and x.shape == (rows.sum(), 16)
    assert np.allclose(x, scaler.transform(built.loc[rows], "DIII-D", columns), atol=1e-6)
    xm, _ = fe.feature_matrix("diii_d_train", ["s1", "s3"], "set_b", mirror=True, config=CONFIG, scaler=scaler, root=root)
    j = columns.index("coil_c01")
    assert np.allclose(xm[:, j], -x[:, j], atol=1e-6) and np.array_equal(xm[:, columns.index("q_cyl")], x[:, columns.index("q_cyl")])
    xa, columns_a = fe.feature_matrix("diii_d_train", None, "set_a", config=CONFIG, scaler=scaler, root=root)
    assert xa.shape == (len(built), CONFIG.layout.n_points + 7) and columns_a[0] == "coil_pt_00"


def test_with_identity_unscales_one_column_and_keeps_the_rest(built):
    scaler = RobustScaler.fit(built, fe.mirror_features(built, CONFIG), CONFIG)
    variant = scaler.with_identity(["ip_frac"])
    assert variant.source["ip_frac"] == [0.0, 1.0] and variant.rules["ip_frac"]["scale"] == "none"
    assert scaler.rules["ip_frac"]["scale"] == "robust" and scaler.source["ip_frac"] != [0.0, 1.0]
    x, y = scaler.transform(built, "DIII-D"), variant.transform(built, "DIII-D")
    j = scaler.columns.index("ip_frac")
    finite = built["ip_frac"].notna().to_numpy()
    assert np.array_equal(y[finite, j], built["ip_frac"].to_numpy()[finite])
    assert np.array_equal(np.delete(x, j, axis=1), np.delete(y, j, axis=1))
    with pytest.raises(ValueError, match="scale group"):
        scaler.with_identity(["coil_pt_00"])
    with pytest.raises(KeyError):
        scaler.with_identity(["shaping"])


def test_extended_appends_columns_with_their_own_statistics(built):
    scaler = RobustScaler.fit(built, fe.mirror_features(built, CONFIG), CONFIG)
    original = built.assign(extra=2.0 * built["q_cyl"])
    extra = RobustScaler.fit_rules(original, original, {"extra": dict(fc.ROBUST_RULE)})
    both = scaler.with_identity(["ip_frac"]).extended(extra)
    assert both.columns == (*scaler.columns, "extra") and both.source["extra"] == extra.source["extra"]
    assert both.source["q_cyl"] == scaler.source["q_cyl"] and both.source["ip_frac"] == [0.0, 1.0]
    x = both.transform(original, "DIII-D", ["q_cyl", "extra"])
    assert np.array_equal(x[:, 0], scaler.transform(built, "DIII-D", ["q_cyl"])[:, 0])
    with pytest.raises(ValueError, match="both"):
        scaler.extended(scaler)


def test_version_2_scaler_leaves_ip_frac_unscaled_and_scales_shaping_on_its_own():
    built_v2 = fe.build_features(make_frames(n_shots=4, n_frames=10), "DIII-D", CONFIG_V2, resources(point_table()))
    mirrored = fe.mirror_features(built_v2, CONFIG_V2)
    v2 = RobustScaler.fit(built_v2, mirrored, CONFIG_V2)
    v1 = RobustScaler.fit(built_v2[["shot", "frame", *CONFIG.all_columns()]], mirrored[["shot", "frame", *CONFIG.all_columns()]],
                          CONFIG)
    assert v2.columns == tuple(CONFIG_V2.all_columns())
    assert v2.rules["ip_frac"]["scale"] == "none" and v2.source["ip_frac"] == [0.0, 1.0]
    # Shaping is even, so originals and mirrors give the same values twice.
    q25, q50, q75 = np.percentile(np.tile(built_v2["shaping"].to_numpy(), 2), [25, 50, 75])
    assert v2.source["shaping"] == pytest.approx([q50, q75 - q25])
    # Every other statistic is the version-1 one.
    assert {c: v for c, v in v2.source.items() if c not in ("ip_frac", "shaping")} == \
        {c: v for c, v in v1.source.items() if c != "ip_frac"}
    x = v2.transform(built_v2, "DIII-D")
    j = v2.columns.index("ip_frac")
    finite = built_v2["ip_frac"].notna().to_numpy()
    assert np.array_equal(x[finite, j], built_v2["ip_frac"].to_numpy()[finite])
    xb = v2.transform(built_v2, "MAST", CONFIG_V2.set_columns("set_b"))
    assert xb.shape == (len(built_v2), 15) and np.isfinite(xb).all()
