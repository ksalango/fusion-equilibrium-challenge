"""Tests of ``scripts/betan_heads.py`` on synthetic frames only: the geometry-explicit base, the
clip rule, the training statistics and the transfer proxy, the label filter counts, the label
guard and the wiring of the four arms."""

from __future__ import annotations

import importlib.util
import sys

import numpy as np
import pandas as pd
import pytest

from fusion_eq import baseline as bl, features as fe, scalars as sc
from tests.test_build_features import REPO_ROOT

spec = importlib.util.spec_from_file_location("betan_heads", REPO_ROOT / "scripts" / "betan_heads.py")
bh = importlib.util.module_from_spec(spec)
# Registered before execution: dataclasses look the module up while building the classes.
sys.modules.setdefault("betan_heads", bh)
spec.loader.exec_module(bh)

CONFIG_V2 = fe.FeatureConfig.load(fe.DEFAULT_CONFIG.with_name("features_v2.yaml"))
MU0 = 4e-7 * np.pi


# ---------------------------------------------------------------------------------- base
def test_base_reproduces_a_hand_computed_betan():
    # p_e 5 kPa, a 0.6 m, I_tf 125 kA, Ip 1.2 MA; DIII-D: 144 turns, R0 1.67 m.
    bt0 = 2e-7 * 144 * 125e3 / 1.67
    expected = 100.0 * (2.0 * MU0 * 5000.0 / bt0 ** 2) * 0.6 * bt0 / 1.2
    assert bt0 == pytest.approx(2.155689, rel=1e-6)
    assert expected == pytest.approx(0.291470, rel=1e-6)
    got = bh.betan_base([5.0, 5.0, np.nan], [0.6, 0.6, 0.6], [125.0, -125.0, 125.0], [1200.0, -1200.0, 1200.0])
    assert got[0] == pytest.approx(expected, rel=1e-12)
    assert got[1] == pytest.approx(expected, rel=1e-12)     # signs of Ip and I_tf ignored
    assert np.isnan(got[2])
    # Linear in pressure and minor radius, inverse in field current and Ip.
    assert bh.betan_base(10.0, 0.3, 250.0, 600.0) == pytest.approx(expected, rel=1e-12)


def test_head_table_joins_shapes_and_derives_the_head_features():
    frames = pd.DataFrame({"shot": ["s1", "s1", "s2"], "frame": [0, 4, 0], bh.PRESSURE_COLUMN: [5.0, 5.0, 8.0],
                           "tf__mean": [125.0, 125.0, -125.0], "ip__mean": [1200.0, 1200.0, -1000.0], "q_cyl": [1.0, 3.0, 20.0]})
    shapes = pd.DataFrame({"shot": ["s1", "s2"], "frame": [0, 0], "a": [0.6, 0.67], "kappa": [1.7, 1.8],
                           "boundary_found": [True, True]})
    out = bh.head_table(frames, shapes, (2.0, 10.0))
    assert out[bh.CLIPPED_NAME].tolist() == [2.0, 3.0, 10.0]
    assert out["a_over_a0"].iloc[0] == pytest.approx(0.6 / 0.67) and out["a_over_a0"].iloc[2] == pytest.approx(1.0)
    assert out[bh.BASE].iloc[0] == pytest.approx(0.291470, rel=1e-6)
    assert out[bh.BASE].iloc[2] == pytest.approx(bh.betan_base(8.0, 0.67, 125.0, 1000.0), rel=1e-12)
    # The frame without a shape row has no base and no boundary.
    assert np.isnan(out[bh.BASE].iloc[1]) and out["boundary_found"].tolist() == [True, False, True]


# ------------------------------------------------------------------------ band and corner
def test_clip_rule_bounds_values_to_the_band_and_keeps_missing_values_missing():
    got = bh.clip_to_band([0.5, 2.0, 5.0, 12.0, np.nan, np.inf], (2.0, 10.0))
    assert got[:4].tolist() == [2.0, 2.0, 5.0, 10.0]
    assert np.isnan(got[4]) and np.isnan(got[5])
    with pytest.raises(ValueError, match="above"):
        bh.clip_to_band([1.0], (3.0, 2.0))


def test_training_statistics_are_percentiles_of_the_finite_values():
    q = np.concatenate([np.arange(1.0, 101.0), [np.nan, np.nan]])
    b = np.concatenate([np.linspace(0.0, 1.0, 101), [np.nan]])
    stats = bh.training_statistics(q, b)
    finite = np.arange(1.0, 101.0)
    assert stats["q_cyl"]["n"] == 100 and stats["ts_beta_p"]["n"] == 101
    assert stats["q_cyl"]["q01"] == pytest.approx(np.percentile(finite, 1.0))
    assert stats["q_cyl"]["q10"] == pytest.approx(np.percentile(finite, 10.0))
    assert stats["q_cyl"]["q99"] == pytest.approx(np.percentile(finite, 99.0))
    assert stats["ts_beta_p"]["q50"] == pytest.approx(0.5) and stats["ts_beta_p"]["q75"] == pytest.approx(0.75)


def test_corner_uses_the_median_rule_and_relaxes_to_the_75th_percentile_when_too_few_frames():
    stats = {"q_cyl": {"q10": 3.0}, "ts_beta_p": {"q50": 0.5, "q75": 0.75}}
    q = np.array([2.0, 2.0, 2.0, 2.0, 4.0, np.nan, 2.0])
    b = np.array([0.4, 0.6, 0.7, 0.9, 0.1, 0.1, np.nan])
    eligible = np.array([True, True, True, True, True, True, True])
    mask, rec = bh.corner_mask(q, b, stats, eligible, min_frames=1)
    assert mask.tolist() == [True, False, False, False, False, False, False] and not rec["relaxed"]
    assert rec["n_frames"] == 1 and rec["ts_beta_p_percentile"] == 50
    mask, rec = bh.corner_mask(q, b, stats, eligible, min_frames=2)
    assert mask.tolist() == [True, True, True, False, False, False, False]
    assert rec["relaxed"] and rec["n_strict"] == 1 and rec["n_frames"] == 3 and rec["ts_beta_p_below"] == 0.75
    eligible[1] = False
    mask, _ = bh.corner_mask(q, b, stats, eligible, min_frames=2)
    assert mask.tolist() == [True, False, True, False, False, False, False]


# -------------------------------------------------------------------------------- frames
def test_filter_counts_each_rule_alone_and_together():
    frames = pd.DataFrame({
        "shot": ["a"] * 6 + ["5df1630aa7"] * 2 + ["b"] * 2,
        "efit_q95": [4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, -1.0],
        "efit_beta_n": [1.0, np.nan, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        "tf__mean": [-120.0, -120.0, -30.0, -120.0, 40.0, -120.0, -120.0, -120.0, -120.0, -120.0],
        "ip_over_peak": [0.5, 0.5, 0.5, 0.05, 0.05, 0.5, 0.5, 0.5, 0.5, 0.5]})
    keep, counts = bh.training_filter(frames)
    by = counts.set_index("rule")
    # Impossible labels: NaN betaN, betaN 0, q95 below 0.
    assert by.loc["impossible_label", "frames"] == 3 and by.loc["impossible_label", "shots"] == 2
    assert by.loc["tf_under_threshold", "frames"] == 2
    assert by.loc["ip_under_fraction_of_peak", "frames"] == 2
    assert by.loc["flagged_shot", "frames"] == 2 and by.loc["flagged_shot", "shots"] == 1
    # Frame 4 fails two rules and counts once in the union.
    assert by.loc["removed (union)", "frames"] == 8 and by.loc["kept", "frames"] == 2
    assert keep.tolist() == [True, False, False, False, False, False, False, False, True, False]
    assert by.loc["kept", "shots"] == 2
    assert bh.validation_mask(frames).tolist() == [True, False, *[True] * 8]


def test_stride_takes_every_nth_frame_of_each_shot_whatever_the_input_order():
    frames = pd.DataFrame({"shot": ["b"] * 6 + ["a"] * 5, "frame": [5, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4]})
    out = bh.strided(frames, 4)
    assert out[["shot", "frame"]].values.tolist() == [["a", 0], ["a", 4], ["b", 0], ["b", 4]]


# --------------------------------------------------------------------------------- guard
def test_guard_accepts_the_study_inputs_and_rejects_labels_and_unbuilt_features():
    reads = bh.guard_inputs(bh.FRAME_INPUTS, [*bh.FEATURE_COLUMNS, *bh.CORNER_COLUMNS], CONFIG_V2)
    assert set(reads) == {*bh.FEATURE_COLUMNS, "ts_beta_p"}
    assert len(bh.FRAME_INPUTS) == 25 and set(bl.input_columns()) == set(bh.FRAME_INPUTS)
    with pytest.raises(ValueError, match="label"):
        bh.guard_inputs([*bh.FRAME_INPUTS, "efit_li"], bh.FEATURE_COLUMNS, CONFIG_V2)
    with pytest.raises(ValueError, match="not built"):
        bh.guard_inputs(bh.FRAME_INPUTS, [*bh.FEATURE_COLUMNS, "ts_te_peaking"], CONFIG_V2)
    # No raw currents, stretch terms or coil points among the head inputs.
    excluded = {*bl.input_columns(), "coil_c20", "coil_c02", *CONFIG_V2.columns_of("coil_pt")}
    assert not excluded & set(bh.HEAD_FEATURES)
    assert len(bh.HEAD_FEATURES) == 10 and bh.HEAD_FEATURES[-2:] == ("kappa", "a_over_a0")


# ---------------------------------------------------------------------------------- arms
def synthetic_table(n_shots: int, n_frames: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Frame rows with every column the study reads, and their shapes. betaN is
    ``3 base exp(0.3 (kappa - 1.6) - 0.1 (ts_pe_peaking - 2))`` with 1 % noise."""
    rng = np.random.default_rng(seed)
    n = n_shots * n_frames
    sign = np.repeat(np.where(np.arange(n_shots) % 3 == 0, -1.0, 1.0), n_frames)
    t = pd.DataFrame({"shot": np.repeat([f"s{seed}_{i:02d}" for i in range(n_shots)], n_frames),
                      "frame": np.tile(np.arange(n_frames), n_shots)})
    for c in bl.input_columns():
        t[c] = rng.normal(0.0, 100.0, n)
    t["ip__mean"] = sign * rng.uniform(800.0, 1500.0, n)
    t["tf__mean"] = -rng.uniform(100.0, 140.0, n)
    t[bh.PRESSURE_COLUMN] = rng.uniform(2.0, 15.0, n)
    t["ts_core_te_peak__interp"] = rng.uniform(1.0, 5.0, n)
    t["ts_core_ne_peak__interp"] = rng.uniform(2.0, 8.0, n)
    t["ts_core_valid"] = 1.0
    t["ip_over_peak"] = rng.uniform(0.2, 1.0, n)
    t["ip_direction"] = sign.astype(int)
    for c in ("shafranov_lambda", "ohmic_flux_used", "shaping"):
        t[c] = rng.normal(0.0, 1.0, n)
    t["ts_pe_peaking"] = rng.uniform(1.0, 3.0, n)
    t["ts_valid"] = 1.0
    t["greenwald_frac"] = rng.uniform(0.2, 0.9, n)
    t.loc[::7, "greenwald_frac"] = np.nan            # filled with the training median
    t["ip_frac"] = t["ip_over_peak"]
    t["q_cyl"] = rng.uniform(1.5, 9.0, n)
    t["ts_beta_p"] = rng.uniform(0.2, 1.5, n)
    shapes = pd.DataFrame({"shot": t["shot"], "frame": t["frame"], "a": rng.uniform(0.5, 0.65, n),
                           "kappa": rng.uniform(1.3, 2.0, n), "boundary_found": True})
    base = bh.betan_base(t[bh.PRESSURE_COLUMN], shapes["a"], t["tf__mean"], t["ip__mean"])
    factor = np.exp(0.3 * (shapes["kappa"] - 1.6) - 0.1 * (t["ts_pe_peaking"] - 2.0))
    t[bh.LABEL] = 3.0 * base * factor * (1.0 + 0.01 * rng.normal(size=n))
    t[bh.OTHER_LABEL] = 4.0
    return t, shapes


def test_each_arm_returns_finite_predictions_and_the_formula_arms_recover_the_synthetic_law():
    train_rows, train_shapes = synthetic_table(12, 40, seed=1)
    val_rows, val_shapes = synthetic_table(6, 30, seed=2)
    band = (2.0, 8.0)
    train = bh.head_table(train_rows, train_shapes, band)
    val = bh.head_table(val_rows, val_shapes, band)
    keep = np.ones(len(train), dtype=bool)
    keep[:40] = False                                  # one shot left out by the filter
    fit = bh.fit_stage(train, keep, val)

    assert set(fit.val) == {*bh.ARMS, bh.CONTEXT_ARM}
    for arm in (*bh.ARMS, bh.CONTEXT_ARM):
        assert fit.val[arm].shape == (len(val),) and np.isfinite(fit.val[arm]).all(), arm
        assert fit.train[arm].shape == (len(train),) and np.isfinite(fit.train[arm]).all(), arm
    assert fit.n_fit["ridge_raw"] == fit.n_fit["ridge_features"] == fit.n_fit["formula_gbm"] == int(keep.sum())
    assert fit.n_inputs == {"ridge_raw": 25, "ridge_features": 10, "formula_ridge": 10, "formula_gbm": 10, bh.CONTEXT_ARM: 0}
    # The constant is the least-squares multiplier on the kept frames, near 3 times the mean factor.
    assert fit.constant == pytest.approx(sc.best_multiplier(train.loc[keep, bh.LABEL], train.loc[keep, bh.BASE]), rel=1e-12)
    assert 2.5 < fit.constant < 3.5

    y = val[bh.LABEL].to_numpy()
    masks = {"all": np.ones(len(val), dtype=bool), "negative_ip": val["ip_direction"].to_numpy() == -1,
             "corner": val["q_cyl"].to_numpy() < 3.0}
    rows = {r["arm"]: r for r in bh.score_stage(2, fit, train[bh.LABEL], keep, y, masks)}
    assert rows["formula_ridge"]["r2_all"] > 0.99 and rows["formula_gbm"]["r2_all"] > 0.95
    assert rows["formula_ridge"]["r2_all"] > rows[bh.CONTEXT_ARM]["r2_all"]
    assert rows["formula_ridge"]["r2_negative_ip"] > 0.98 and rows["formula_ridge"]["r2_train"] > 0.99
    assert rows["ridge_raw"]["n_negative_ip"] == 60 and rows["ridge_raw"]["n_corner"] == int(masks["corner"].sum())
    assert all(r["n_missing_pred"] == 0 and r["constant"] == fit.constant for r in rows.values())

    results = pd.DataFrame([*bh.score_stage(1, fit, train[bh.LABEL], keep, y, masks), *rows.values()])
    corner = {"q_cyl_below": 3.0, "ts_beta_p_below": 0.75, "ts_beta_p_percentile": 75, "n_frames": 50, "n_strict": 20,
              "min_frames": 300, "relaxed": True}
    stats = {"q_cyl": {"n": 100, "q01": 2.0, "q10": 3.0, "q50": 5.0, "q99": 8.0}, "ts_beta_p": {"n": 100, "q50": 0.5, "q75": 0.75}}
    shape_rec = {"file": "shapes.parquet", "reused": True, "n_without_row": 0, "n_without_boundary": 0}
    info = {"lists": "working", "split_name": "split_v1.json", "frame_stride": 4, "n_train_shots": 12, "n_val_shots": 6,
            "n_train_frames": len(train), "n_train_kept": int(keep.sum()), "n_val": len(val), "feature_version": 2,
            "feature_dir": "../data/processed/features/v2", "alpha": 1.0, "n_tf": 144.0, "r0_m": 1.67, "corner": corner,
            "training_statistics": stats, "n_stat_shots": 12, "filter_counts": [{"rule": "kept", "frames": 440, "shots": 11}],
            "shapes": {"1": {"train": shape_rec, "val": shape_rec}, "2": {"train": shape_rec, "val": shape_rec}},
            "runtime_s": 12.0}
    report = bh.render(results, info)
    assert report.startswith("# betaN heads, working lists of `split_v1.json`, stride 4")
    assert "| reference | ridge_raw | 25 |" in report and "| review | formula_gbm | 10 |" in report
    assert "relaxed: 20 frames under the median rule" in report and "(stage 2: c = " in report
    assert "base x constant (context)" in report and report.rstrip().endswith("Runtime 12 s.")
