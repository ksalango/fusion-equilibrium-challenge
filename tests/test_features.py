"""Tests of the feature registry, configuration, label guard, feature values, invariances and
the mirror, on synthetic frame rows and synthetic coil sets."""

from __future__ import annotations

import copy
import dataclasses

import numpy as np
import pandas as pd
import pytest

from fusion_eq import coil_flux as cf, coil_pattern as cp, features as fe
from fusion_eq.machines import DIII_D
from tests.test_coil_pattern import symmetric_geometry

CONFIG = fe.FeatureConfig.load()
COIL_COLUMNS = [c for c in CONFIG.all_columns() if c.startswith("coil_")]
R0, A0 = DIII_D.major_radius.value, DIII_D.minor_radius.value


def point_table(geometry=None, r0=R0, a0=A0, n_filaments=4) -> cf.PointTable:
    pr, pz = CONFIG.layout.physical(r0, a0)
    return cf.build_point_table(DIII_D, geometry or symmetric_geometry(), pr, pz, n_filaments)


def resources(table=None, scales=None, r0=R0, a0=A0) -> fe.MachineResources:
    res = fe.MachineResources.from_machine("DIII-D", table, scales)
    res.r0_m, res.a0_m = r0, a0
    return res


CURRENT_COLUMNS = ["ip__mean", "ip__rate50", "sol__mean", "sol_at_rise", "tf__mean",
                   *[f"{c}__mean" for c in DIII_D.shaping_coil_columns]]


def make_frames(n_shots: int = 2, n_frames: int = 6, seed: int = 0) -> pd.DataFrame:
    """Frame rows with every column the version-1 features read, positive Ip."""
    rng = np.random.default_rng(seed)
    parts = []
    for s in range(n_shots):
        n = n_frames
        d = {"shot": f"s{s}", "frame": np.arange(n), "time_ms": 100.0 + 20.0 * np.arange(n),
             "ip__mean": rng.uniform(600.0, 1200.0, n), "ip__rate50": rng.normal(0.0, 500.0, n),
             "sol__mean": rng.normal(0.0, 20.0, n), "sol_at_rise": np.full(n, 15.0 + s),
             "ip_rise_ms": np.full(n, 12.0 + s), "tf__mean": np.full(n, -120.0), "ip_over_peak": rng.uniform(0.3, 1.0, n),
             "ts_core_te_peak__interp": rng.uniform(500.0, 3000.0, n), "ts_core_ne_peak__interp": rng.uniform(2e19, 8e19, n),
             "ts_core_pe_mean__interp": rng.uniform(2.0, 20.0, n), "ts_core_valid": np.ones(n, dtype=bool)}
        d["ts_core_pe_peak__interp"] = d["ts_core_pe_mean__interp"] * rng.uniform(1.5, 3.0, n)
        if n > 1:
            d["ts_core_valid"][1] = False
        for c in DIII_D.shaping_coil_columns:
            d[f"{c}__mean"] = rng.normal(0.0, 300.0, n)
        parts.append(pd.DataFrame(d))
    return pd.concat(parts, ignore_index=True)


def reversed_currents(frames: pd.DataFrame) -> pd.DataFrame:
    out = frames.copy()
    out[CURRENT_COLUMNS] = -frames[CURRENT_COLUMNS]
    return out


def swapped_ab(frames: pd.DataFrame) -> pd.DataFrame:
    out = frames.copy()
    for i in range(1, 10):
        a, b = f"magnetics_F{i}A__mean", f"magnetics_F{i}B__mean"
        out[a], out[b] = frames[b].to_numpy(), frames[a].to_numpy()
    return out


@pytest.fixture(scope="module")
def table():
    return point_table()


# ------------------------------------------------------------------------ configuration
def test_every_configured_feature_declares_its_rules_and_every_set_member_is_configured():
    for spec in CONFIG.features:
        assert spec.function in fe.REGISTRY
        assert spec.mirror in fe.MIRROR_RULES and spec.scale in fe.SCALE_RULES and isinstance(spec.uses_future, bool)
    for members in CONFIG.sets.values():
        assert set(members) <= {s.name for s in CONFIG.features}
    assert len(CONFIG.set_columns("set_a")) == CONFIG.layout.n_points + 7 and len(CONFIG.set_columns("set_b")) == 16
    odd = {s.name for s in CONFIG.features if s.mirror == "odd"}
    assert odd == {"coil_c01", "coil_c11", "coil_c21", "coil_c03"}
    assert CONFIG.spec("coil_pt").scale_group == "coil_pattern" and CONFIG.spec("coil_pt").mirror == "permute"
    assert all(s.scale_with == "source" for s in CONFIG.features)


def _config_dict() -> dict:
    return copy.deepcopy(CONFIG.to_dict())


@pytest.mark.parametrize("change, message", [
    (lambda d: d["features"][1].pop("mirror"), "lacks"),
    (lambda d: d["features"][1].pop("uses_future"), "lacks"),
    (lambda d: d["features"][1].update(uses_future="yes"), "uses_future"),
    (lambda d: d["features"][1].update(mirror="flip"), "mirror"),
    (lambda d: d["features"][1].update(function="no_such_function"), "not registered"),
    (lambda d: d["features"][1].update(mirror="permute"), "permute"),
    (lambda d: d["features"][1].update(colour="red"), "unknown fields"),
    (lambda d: d["sets"].update(set_c=["coil_c10", "missing_feature"]), "unconfigured"),
    (lambda d: d["features"].append(dict(d["features"][1])), "repeated"),
    (lambda d: d["layout"]["points"].pop(), "mirror partner"),
])
def test_configuration_validation_rejects(change, message):
    d = _config_dict()
    change(d)
    with pytest.raises(ValueError, match=message):
        fe.FeatureConfig.from_dict(d)


# ----------------------------------------------------------------------------- the guard
def test_label_guard_and_declared_reads():
    assert fe.check_reads(["ip__mean", "ip_over_peak", "ts_core_valid", "time_ms", "ip_rise_ms"], DIII_D)
    with pytest.raises(ValueError, match="label"):
        fe.check_reads(["ip__mean", "efit_q95"], DIII_D)
    with pytest.raises(ValueError, match="label"):
        fe.check_reads(["magnetics_dsep"], DIII_D)
    with pytest.raises(ValueError, match="not readable"):
        fe.check_reads(["frame"], DIII_D)
    reads = fe.feature_reads(CONFIG, "DIII-D")
    assert all(fe.check_reads(cols, DIII_D) for cols in reads.values())
    assert reads["resistive_age"] == ["time_ms", "ip_rise_ms", "ts_core_te_peak__interp", "ts_core_valid"]


@pytest.fixture
def temporary_functions():
    added = []

    def add(name, reads, fn):
        fe.register(name, reads)(fn)
        added.append(name)
    yield add
    for name in added:
        fe.REGISTRY.pop(name, None)


def test_a_feature_reading_a_label_raises(temporary_functions):
    temporary_functions("test_reads_label", ("ip__mean", "efit_q95"), lambda ctx: ctx.column("ip__mean"))
    temporary_functions("test_undeclared", ("ip__mean",), lambda ctx: ctx.column("tf__mean"))
    d = _config_dict()
    d["features"].append({"name": "leak", "function": "test_reads_label", "group": "test", "mirror": "even",
                          "scale": "robust", "uses_future": False})
    d["features"].append({"name": "sneak", "function": "test_undeclared", "group": "test", "mirror": "even",
                          "scale": "robust", "uses_future": False})
    config = fe.FeatureConfig.from_dict(d)
    frames = make_frames()
    frames["efit_q95"] = 4.0
    with pytest.raises(ValueError, match="label"):
        fe.build_features(frames, "DIII-D", config, resources(), names=["leak"])
    with pytest.raises(ValueError, match="without being declared"):
        fe.build_features(frames, "DIII-D", config, resources(), names=["sneak"])


# --------------------------------------------------------------------------------- build
def test_build_is_keyed_sorted_and_complete(table):
    frames = make_frames()
    out = fe.build_features(frames.sample(frac=1.0, random_state=1), "DIII-D", CONFIG, resources(table))
    assert list(out.columns) == ["shot", "frame", *CONFIG.all_columns()]
    assert out[["shot", "frame"]].equals(frames[["shot", "frame"]].astype({"shot": str}))
    assert np.isfinite(out[COIL_COLUMNS].to_numpy()).all()
    with pytest.raises(ValueError, match="repeat"):
        fe.build_features(pd.concat([frames, frames.iloc[:1]]), "DIII-D", CONFIG, resources(table))
    with pytest.raises(KeyError):
        fe.build_features(frames.drop(columns=["tf__mean"]), "DIII-D", CONFIG, resources(table))
    only = fe.build_features(frames, "DIII-D", CONFIG, resources(), names=["q_cyl", "ts_valid"])
    assert list(only.columns) == ["shot", "frame", "q_cyl", "ts_valid"]


def test_named_terms_recover_a_coil_pattern_that_is_an_exact_cubic():
    # One "coil" whose flux at each point is mu0 R0 Ip P(x, y) for 1 kA of it, Ip being 1 MA.
    coefficients = np.array([0.8, -0.3, 0.25, 0.1, -0.4, 0.05, -0.07, 0.02, 0.09])
    pts = np.vstack([CONFIG.layout.points, [[0.0, 0.0]]])
    poly = cp.term_matrix(pts) @ coefficients + 0.3                           # a constant cancels
    patterns = (cf.MU0 * R0 * 1.0e6 * poly)[None, :]
    table = cf.PointTable("DIII-D", ("magnetics_ECOILA",), patterns, np.zeros(pts.shape[0]), np.zeros(pts.shape[0]), 1, "g", "p")
    frames = make_frames(n_shots=1, n_frames=2)
    frames["ip__mean"], frames["sol__mean"] = 1000.0, 1.0
    out = fe.build_features(frames, "DIII-D", CONFIG, resources(table))
    terms = out[[f"coil_{t}" for t in cp.TERM_NAMES]].to_numpy()
    assert np.allclose(terms, coefficients[None, :], atol=1e-12)


def test_uniform_vertical_field_gives_the_k9_ratio_and_the_shafranov_handle():
    # psi = Bz R^2 / 2 is a uniform vertical field Bz: c10 = Bz a0 / (mu0 Ip), c20 = c10 a0 / (2 R0).
    pts = np.vstack([CONFIG.layout.points, [[0.0, 0.0]]])
    r = R0 + A0 * pts[:, 0]
    bz_per_ka = 1e-3
    table = cf.PointTable("DIII-D", ("magnetics_ECOILA",), (bz_per_ka * r ** 2 / 2)[None, :], r, A0 * pts[:, 1], 1, "g", "p")
    frames = make_frames(n_shots=1, n_frames=1)
    frames["sol__mean"], frames["ip__mean"] = -800.0, 1000.0                  # Bz = -0.8 T
    out = fe.build_features(frames, "DIII-D", CONFIG, resources(table)).iloc[0]
    bz, ip_a = -0.8, 1.0e6
    assert out["coil_c10"] == pytest.approx(bz * A0 / (cf.MU0 * ip_a), rel=1e-10)
    assert out["coil_c20"] == pytest.approx(out["coil_c10"] * A0 / (2 * R0), rel=1e-10)
    for t in ("c01", "c11", "c02", "c30", "c21", "c12", "c03"):
        assert abs(out[f"coil_{t}"]) < 1e-10 * abs(out["coil_c10"])
    k9 = bz / (cf.MU0 * ip_a / (4 * np.pi * R0))                              # check K9's ratio
    assert 4 * np.pi * (R0 / A0) * out["coil_c10"] == pytest.approx(k9, rel=1e-10)
    assert out["shafranov_lambda"] == pytest.approx(-k9 - np.log(8 * R0 / A0) + 1.5, rel=1e-10)


def test_uniform_flux_offset_leaves_every_coil_feature_unchanged(table):
    frames = make_frames()
    plain = fe.build_features(frames, "DIII-D", CONFIG, resources(table, cf.CoilScales.default("DIII-D", table.columns)))
    flat = cf.CoilScales("DIII-D", {c: 1.0 for c in table.columns}, flat_wb_per_ka=0.02)
    offset = fe.build_features(frames, "DIII-D", CONFIG, resources(table, flat))
    scale = np.abs(plain[COIL_COLUMNS].to_numpy()).max()
    assert np.allclose(offset[COIL_COLUMNS], plain[COIL_COLUMNS], rtol=0, atol=1e-12 * scale)


def test_reversing_every_current_with_ip_leaves_every_feature_unchanged(table):
    frames = make_frames()
    scales = cf.CoilScales("DIII-D", {c: 1.1 for c in table.columns}, flat_wb_per_ka=0.01)
    forward = fe.build_features(frames, "DIII-D", CONFIG, resources(table, scales))
    backward = fe.build_features(reversed_currents(frames), "DIII-D", CONFIG, resources(table, scales))
    columns = CONFIG.all_columns()
    assert np.allclose(backward[columns], forward[columns], rtol=1e-12, atol=0, equal_nan=True)
    assert np.array_equal(np.isnan(backward[columns].to_numpy()), np.isnan(forward[columns].to_numpy()))


def test_features_have_no_units_under_a_joint_change_of_size_and_current():
    lam, k = 2.0, 3.0
    g = symmetric_geometry()
    big = cf.CoilGeometry(g.names, g.input_columns, lam * g.r, lam * g.z, lam * g.width, lam * g.height, g.angle1, g.angle2)
    frames = make_frames()
    small = fe.build_features(frames, "DIII-D", CONFIG, resources(point_table(g)))
    scaled_frames = frames.copy()
    scaled_frames[CURRENT_COLUMNS] = k * frames[CURRENT_COLUMNS]
    large = fe.build_features(scaled_frames, "DIII-D", CONFIG,
                              resources(point_table(big, lam * R0, lam * A0), r0=lam * R0, a0=lam * A0))
    columns = COIL_COLUMNS + ["shafranov_lambda", "ohmic_flux_used"]
    assert np.allclose(large[columns], small[columns], rtol=1e-9, atol=1e-12)


# -------------------------------------------------------------------------------- mirror
def test_mirroring_twice_is_the_identity_and_signs_follow_the_rules(table):
    out = fe.build_features(make_frames(), "DIII-D", CONFIG, resources(table))
    once = fe.mirror_features(out, CONFIG)
    assert fe.mirror_features(once, CONFIG).equals(out)
    assert np.array_equal(once["coil_c01"], -out["coil_c01"]) and np.array_equal(once["coil_c10"], out["coil_c10"])
    assert once["ts_beta_p"].equals(out["ts_beta_p"]) and once[["shot", "frame"]].equals(out[["shot", "frame"]])
    with pytest.raises(ValueError, match="all of its columns"):
        fe.mirror_features(out[["coil_pt_00", "coil_pt_01"]], CONFIG)


def test_mirror_of_the_points_equals_the_points_of_the_flipped_field(table):
    frames = make_frames()
    out = fe.build_features(frames, "DIII-D", CONFIG, resources(table))
    flipped = fe.build_features(swapped_ab(frames), "DIII-D", CONFIG, resources(table))
    mirrored = fe.mirror_features(out, CONFIG)
    scale = np.abs(out[COIL_COLUMNS].to_numpy()).max()
    assert np.allclose(mirrored[COIL_COLUMNS], flipped[COIL_COLUMNS], rtol=0, atol=1e-10 * scale)


def test_a_pattern_symmetric_in_y_has_zero_odd_terms(table):
    frames = make_frames()
    for i in range(1, 10):
        frames[f"magnetics_F{i}B__mean"] = frames[f"magnetics_F{i}A__mean"]
    out = fe.build_features(frames, "DIII-D", CONFIG, resources(table))
    scale = np.abs(out[COIL_COLUMNS].to_numpy()).max()
    for name in ("coil_c01", "coil_c11", "coil_c21", "coil_c03"):
        assert np.abs(out[name]).max() < 1e-10 * scale
    assert np.abs(out["coil_c10"]).min() > 1e-6 * scale
    partner = CONFIG.layout.mirror_index()
    points = out[CONFIG.columns_of("coil_pt")].to_numpy()
    assert np.allclose(points, points[:, partner], rtol=0, atol=1e-10 * scale)


# ------------------------------------------------------------------- values of the others
def one_frame(**values) -> pd.DataFrame:
    frames = make_frames(n_shots=1, n_frames=1)
    for k, v in values.items():
        frames[k] = v
    return frames


def test_plasma_content_field_and_density_values():
    out = fe.build_features(one_frame(ip__mean=-1000.0, tf__mean=120.0, ts_core_pe_mean__interp=10.0,
                                      ts_core_pe_peak__interp=25.0, ts_core_ne_peak__interp=5e19),
                            "DIII-D", CONFIG, resources(), names=["ts_beta_p", "ts_pe_peaking", "ts_valid", "q_cyl",
                                                                  "greenwald_frac"]).iloc[0]
    ip_a, p_pa = 1.0e6, 1.0e4
    bp = cf.MU0 * ip_a / (2 * np.pi * A0)                                     # poloidal field at the edge
    assert out["ts_beta_p"] == pytest.approx(2 * cf.MU0 * p_pa / bp ** 2, rel=1e-12)
    assert out["ts_pe_peaking"] == pytest.approx(2.5) and out["ts_valid"] == 1.0
    bt = cf.MU0 * 144 * 120e3 / (2 * np.pi * R0)
    assert out["q_cyl"] == pytest.approx(2 * np.pi * A0 ** 2 * bt / (cf.MU0 * R0 * ip_a), rel=1e-12)
    assert out["greenwald_frac"] == pytest.approx(0.5 / (1.0 / (np.pi * A0 ** 2)), rel=1e-12)


def test_missing_thomson_and_small_ip_give_nan_where_stated():
    frames = make_frames(n_shots=1, n_frames=3)
    frames.loc[0, "ip__mean"] = 5.0                                          # below min_abs_ip_ka
    frames.loc[1, "ts_core_valid"] = False
    out = fe.build_features(frames, "DIII-D", CONFIG, resources(),
                            names=["ts_beta_p", "ts_pe_peaking", "ts_valid", "q_cyl", "greenwald_frac", "ip_frac", "ip_rate_rel"])
    assert np.isnan(out.loc[0, ["ts_beta_p", "q_cyl", "greenwald_frac", "ip_rate_rel"]].to_numpy(dtype=float)).all()
    assert np.isfinite(out.loc[0, ["ts_pe_peaking", "ip_frac"]].to_numpy(dtype=float)).all()
    assert np.isnan(out.loc[1, ["ts_beta_p", "ts_pe_peaking", "greenwald_frac"]].to_numpy(dtype=float)).all()
    assert out["ts_valid"].tolist() == [1.0, 0.0, 1.0]
    assert np.isfinite(out.drop(columns=["shot", "frame"]).iloc[2].to_numpy(dtype=float)).all()


def test_ip_rate_rel_uses_the_resistive_time():
    out = fe.build_features(one_frame(ip__mean=-1000.0, ip__rate50=-500.0, ts_core_te_peak__interp=2000.0),
                            "DIII-D", CONFIG, resources(), names=["ip_rate_rel"]).iloc[0]
    tau = 7.8 * A0 ** 2 * 2.0 ** 1.5
    assert out["ip_rate_rel"] == pytest.approx(0.5 * tau, rel=1e-12)


def test_resistive_age_grows_along_a_shot_and_ignores_a_time_shift():
    frames = make_frames(n_shots=2, n_frames=8)
    out = fe.build_features(frames, "DIII-D", CONFIG, resources(), names=["resistive_age"])
    for _, g in out.groupby("shot"):
        assert np.all(np.diff(g["resistive_age"].to_numpy()) > 0)
    shifted = frames.copy()
    shifted["time_ms"] += 3500.0
    shifted["ip_rise_ms"] += 3500.0
    again = fe.build_features(shifted, "DIII-D", CONFIG, resources(), names=["resistive_age"])
    assert np.allclose(again["resistive_age"], out["resistive_age"], rtol=1e-12)
    # Constant temperature: the trapezoid is exact, (t - t_rise) / tau_R.
    flat = make_frames(n_shots=1, n_frames=5)
    flat["ts_core_te_peak__interp"] = 1500.0
    age = fe.build_features(flat, "DIII-D", CONFIG, resources(), names=["resistive_age"])["resistive_age"].to_numpy()
    tau = 7.8 * A0 ** 2 * 1.5 ** 1.5
    assert np.allclose(age, (flat["time_ms"] - flat["ip_rise_ms"]).to_numpy() / 1000.0 / tau, rtol=1e-12)


def test_tau_r_falls_back_to_the_shot_median_and_a_shot_without_thomson_gives_nan():
    frames = make_frames(n_shots=2, n_frames=4)
    frames["ts_core_te_peak__interp"] = [1000.0, 9000.0, 2000.0, 3000.0, 1000.0, 1000.0, 1000.0, 1000.0]
    frames["ts_core_valid"] = [True, False, True, True, False, False, False, False]
    out = fe.build_features(frames, "DIII-D", CONFIG, resources(), names=["ip_rate_rel", "resistive_age"])
    rel = (frames["ip__rate50"] / frames["ip__mean"]).to_numpy()
    tau_median = 7.8 * A0 ** 2 * 2.0 ** 1.5                                   # median of 1, 2 and 3 keV
    assert out.loc[1, "ip_rate_rel"] == pytest.approx(rel[1] * tau_median, rel=1e-12)
    assert out.loc[out["shot"] == "s1", ["ip_rate_rel", "resistive_age"]].isna().all().all()


def test_ohmic_flux_used_is_the_solenoid_flux_since_the_rise(table):
    frames = make_frames(n_shots=1, n_frames=2)
    out = fe.build_features(frames, "DIII-D", CONFIG, resources(table), names=["ohmic_flux_used"])
    sol = table.column_index(["magnetics_ECOILA"])[0]
    delta = (frames["sol__mean"] - frames["sol_at_rise"]).to_numpy()
    expected = table.patterns[sol, -1] * delta / (cf.MU0 * R0 * frames["ip__mean"].to_numpy() * 1000.0)
    assert np.allclose(out["ohmic_flux_used"], expected, rtol=1e-12)


# ------------------------------------------------------------------------------ version 2
CONFIG_V2 = fe.FeatureConfig.load(fe.DEFAULT_CONFIG.with_name("features_v2.yaml"))


def test_version_2_changes_only_shaping_ip_frac_scaling_and_set_b():
    assert CONFIG.version == 1 and CONFIG_V2.version == 2 and CONFIG_V2.hash() != CONFIG.hash()
    v1 = {s.name: s for s in CONFIG.features}
    v2 = {s.name: s for s in CONFIG_V2.features}
    # Every version-1 feature is still built; shaping is the only new one (no ts_te_peaking:
    # the frame table holds no core mean Te).
    assert set(v2) - set(v1) == {"shaping"} and set(v1) <= set(v2)
    assert {n for n in v1 if v1[n] != v2[n]} == {"ip_frac", "shafranov_lambda"}
    assert dataclasses.replace(v2["shafranov_lambda"], group="candidate", note="") == v1["shafranov_lambda"]
    assert v2["shafranov_lambda"].group == "coil_terms"
    assert v2["ip_frac"].scale == "none"
    assert dataclasses.replace(v2["ip_frac"], scale="robust", note=v1["ip_frac"].note) == v1["ip_frac"]
    s = v2["shaping"]
    assert (s.function, s.group, s.mirror, s.scale, s.uses_future, s.params) == \
        ("shaping", "coil_terms", "even", "robust", False, {})
    assert np.array_equal(CONFIG_V2.layout.points, CONFIG.layout.points)
    assert CONFIG_V2.layout.min_clearance_m == CONFIG.layout.min_clearance_m and CONFIG_V2.min_abs_ip_ka == CONFIG.min_abs_ip_ka
    assert CONFIG_V2.sets["set_a"] == CONFIG.sets["set_a"]
    assert CONFIG_V2.sets["set_b"] == ("shafranov_lambda", "coil_c01", "shaping", "coil_c11", "coil_c30", "coil_c21",
                                       "coil_c12", "coil_c03", *CONFIG.sets["set_b"][9:])
    assert len(CONFIG_V2.set_columns("set_b")) == 15 and set(CONFIG_V2.sets) == set(CONFIG.sets)
    # set_b without coil_terms leaves out the whole coil description: the seven plasma, field and position features.
    groups = {n: v2[n].group for n in CONFIG_V2.sets["set_b"]}
    assert [n for n, g in groups.items() if g != "coil_terms"] == list(CONFIG.sets["set_b"][9:])
    assert "candidate" not in groups.values()
    v1_columns = CONFIG.all_columns()
    after = v1_columns.index("coil_c03") + 1
    assert CONFIG_V2.all_columns() == [*v1_columns[:after], "shaping", *v1_columns[after:]]


def test_version_2_writes_to_its_own_folders():
    assert fe.features_dir(2) == fe.DEFAULT_FEATURE_ROOT / "v2" and fe.features_dir(1) == fe.DEFAULT_FEATURE_ROOT / "v1"
    assert fe.features_path("mast_public_test", 2).parent.name == "v2"
    assert fe.scaler_path(2) == fe.DEFAULT_REPORT_ROOT / "v2" / "scaler.json"


def test_shaping_is_half_of_c02_less_c20_and_leaves_every_version_1_feature_unchanged(table):
    frames = make_frames()
    v2 = fe.build_features(frames, "DIII-D", CONFIG_V2, resources(table))
    v1 = fe.build_features(frames, "DIII-D", CONFIG, resources(table))
    assert np.array_equal(v2["shaping"], 0.5 * (v2["coil_c02"] - v2["coil_c20"]))
    assert np.isfinite(v2["shaping"]).all() and np.abs(v2["shaping"]).min() > 0
    assert v2[["shot", "frame", *CONFIG.all_columns()]].equals(v1)


def test_shaping_recovers_the_quadrupole_of_an_exact_cubic_pattern():
    coefficients = np.array([0.8, -0.3, 0.25, 0.1, -0.4, 0.05, -0.07, 0.02, 0.09])
    pts = np.vstack([CONFIG_V2.layout.points, [[0.0, 0.0]]])
    poly = cp.term_matrix(pts) @ coefficients + 0.3
    patterns = (cf.MU0 * R0 * 1.0e6 * poly)[None, :]
    table = cf.PointTable("DIII-D", ("magnetics_ECOILA",), patterns, np.zeros(pts.shape[0]), np.zeros(pts.shape[0]), 1, "g", "p")
    frames = make_frames(n_shots=1, n_frames=2)
    frames["ip__mean"], frames["sol__mean"] = 1000.0, 1.0
    out = fe.build_features(frames, "DIII-D", CONFIG_V2, resources(table), names=["shaping"])
    c20, c02 = coefficients[cp.TERM_NAMES.index("c20")], coefficients[cp.TERM_NAMES.index("c02")]
    assert np.allclose(out["shaping"], 0.5 * (c02 - c20), rtol=0, atol=1e-12)


def test_shaping_reads_what_the_named_terms_read_and_passes_the_label_guard(table):
    for machine in ("DIII-D", "MAST"):
        reads = fe.feature_reads(CONFIG_V2, machine)
        assert reads["shaping"] == reads["coil_c20"] == reads["coil_c02"]
        assert fe.check_reads(reads["shaping"], machine)
        assert not any(c.startswith(("efit_", "magnetics_dsep")) for c in reads["shaping"])
    # A label column present in the rows is never read, and the result does not depend on it.
    frames = make_frames()
    plain = fe.build_features(frames, "DIII-D", CONFIG_V2, resources(table), names=["shaping"])
    labelled = fe.build_features(frames.assign(efit_q95=4.0, efit_li=1.0), "DIII-D", CONFIG_V2, resources(table),
                                 names=["shaping"])
    assert labelled.equals(plain)
    with pytest.raises(KeyError):
        fe.build_features(frames.drop(columns=["magnetics_F1A__mean"]), "DIII-D", CONFIG_V2, resources(table),
                          names=["shaping"])
    with pytest.raises(ValueError, match="point table"):
        fe.build_features(frames, "DIII-D", CONFIG_V2, resources(), names=["shaping"])


def test_shaping_is_even_under_the_mirror_and_equals_the_shaping_of_the_flipped_field(table):
    frames = make_frames()
    out = fe.build_features(frames, "DIII-D", CONFIG_V2, resources(table))
    once = fe.mirror_features(out, CONFIG_V2)
    assert once["shaping"].equals(out["shaping"]) and fe.mirror_features(once, CONFIG_V2).equals(out)
    assert np.array_equal(once["coil_c01"], -out["coil_c01"])
    flipped = fe.build_features(swapped_ab(frames), "DIII-D", CONFIG_V2, resources(table))
    scale = np.abs(out[COIL_COLUMNS].to_numpy()).max()
    assert np.allclose(flipped["shaping"], out["shaping"], rtol=0, atol=1e-10 * scale)
    # The flip changes the odd terms, so the agreement of shaping is not trivial.
    assert np.abs(flipped["coil_c01"] + out["coil_c01"]).max() < 1e-10 * scale < np.abs(out["coil_c01"]).max()
