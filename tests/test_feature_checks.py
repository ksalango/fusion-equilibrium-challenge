"""Tests of the feature-check calculations (X3 to X16) on synthetic arrays and frames."""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pandas as pd
import pytest

from fusion_eq import feature_checks as fc, features as fe, frame_table as ft
from fusion_eq.coil_flux import MU0
from fusion_eq.coil_scales import phase_of
from fusion_eq.machines import DIII_D, MAST
from tests.test_coil_pattern import symmetric_geometry
from tests.test_features import CONFIG, make_frames, point_table, resources, swapped_ab


# ------------------------------------------------------------------------ table, phases
def test_check_rejects_an_unknown_status_and_renders_a_table():
    with pytest.raises(ValueError, match="status"):
        fc.Check("X1", "Rows", "ok", "")
    text = fc.check_table_markdown([fc.Check("X1", "Rows and keys", "pass", "a | b"), fc.Check("X3", "Completeness", "review", "c")])
    lines = text.splitlines()
    assert len(lines) == 4 and lines[2] == "| X1 | Rows and keys | pass | a / b |" and lines[3].startswith("| X3 |")


def test_markdown_table_formats_numbers_and_escapes_pipes():
    table = pd.DataFrame({"name": ["a|b", "c"], "n": [3, 4], "x": [0.000123456, np.nan], "ok": [True, False]})
    lines = fc.markdown_table(table, digits=3).splitlines()
    assert lines == ["| name | n | x | ok |", "| --- | --- | --- | --- |", "| a/b | 3 | 0.000123 | True |", "| c | 4 | nan | False |"]


def test_frame_phases_follow_phase_of_frame_by_frame():
    over = np.array([0.95, 0.5, 0.5, np.nan, 0.9, 0.89, np.nan])
    frame = np.array([0, 1, 8, 2, 9, 3, 7])
    n = np.array([10, 10, 10, 10, 10, 6, 10])
    expected = [phase_of(o, f, k) for o, f, k in zip(over, frame, n)]
    assert fc.frame_phases(over, frame, n).tolist() == expected
    assert expected == ["flat_top", "rising", "late", "rising", "flat_top", "late", "late"]
    assert fc.frames_per_shot(["a", "a", "b", "a"]).tolist() == [3, 3, 1, 3]


def test_join_frame_columns_adds_columns_phases_and_checks_keys():
    features = pd.DataFrame({"shot": ["a", "a", "b"], "frame": [0, 1, 0], "ip_frac": [0.5, 0.95, 0.2], "q_cyl": [3.0, 3.1, 2.0]})
    frames = pd.DataFrame({"shot": ["a", "a", "b"], "frame": [0, 1, 0], "ip__mean": [500.0, 900.0, 100.0]})
    out = fc.join_frame_columns(features, frames)
    assert out["ip__mean"].tolist() == [500.0, 900.0, 100.0] and out["n_frames"].tolist() == [2, 2, 1]
    # Shot b has one frame: frame 0 lies in the first half of the shot.
    assert out["phase"].tolist() == ["rising", "flat_top", "rising"]
    with pytest.raises(ValueError, match="differ"):
        fc.join_frame_columns(features, frames.assign(frame=[0, 2, 0]))
    with pytest.raises(ValueError, match="differ"):
        fc.join_frame_columns(features, frames.iloc[:2])
    with pytest.raises(ValueError, match="both tables"):
        fc.join_frame_columns(features, frames.assign(q_cyl=1.0))


def test_profile_frame_columns_are_inputs_of_each_machine():
    for cfg in (DIII_D, MAST):
        columns = fc.profile_frame_columns(cfg)
        assert not any(ft.is_label_column(c) for c in columns)
        assert set(columns) - {"time_ms"} <= set(ft.input_columns(cfg, include_flags=True))
        assert {"ip__mean", "ip__rate50", "ip_rise_ms", f"{cfg.shaping_coil_columns[0]}__mean"} <= set(columns)


# --------------------------------------------------------------------- completeness, ranges
def test_finite_shares_and_low_ip_counts():
    t = pd.DataFrame({"a": [1.0, np.nan, 2.0, 3.0], "b": [np.nan, np.nan, 1.0, np.inf]})
    assert fc.finite_shares(t, ["a", "b"]).tolist() == [0.75, 0.25]
    counts = fc.low_ip_counts([np.nan, 5.0, -19.9, 20.0, -800.0], 20.0)
    assert counts == {"n_frames": 5, "n_missing_ip": 1, "n_below_min": 2, "n_affected": 3, "share_affected": 0.6}


def test_compare_ranges_counts_values_outside_the_training_band_and_range():
    reference = pd.DataFrame({"f": np.linspace(0.0, 100.0, 10001)})
    split = pd.DataFrame({"f": [-5.0, 0.5, 50.0, 99.5, 150.0, np.nan]})
    row = fc.compare_ranges(reference, {"mast": split}, ["f"]).iloc[0]
    assert row["ref_q01"] == pytest.approx(1.0) and row["ref_q99"] == pytest.approx(99.0)
    assert row["ref_iqr"] == pytest.approx(50.0) and row["n_finite"] == 5
    assert row["share_outside_band"] == pytest.approx(4 / 5)       # -5, 0.5, 99.5, 150
    assert row["share_outside_range"] == pytest.approx(2 / 5)      # -5, 150
    assert row["median_shift_iqr"] == pytest.approx(0.0)
    shifted = fc.compare_ranges(reference, {"mast": pd.DataFrame({"f": [150.0]})}, ["f"]).iloc[0]
    assert shifted["median_shift_iqr"] == pytest.approx(2.0)


def test_compare_ranges_by_phase_restricts_the_reference_to_the_phase():
    reference = pd.DataFrame({"f": np.r_[np.zeros(100), np.full(100, 10.0)]})
    ref_phase = np.r_[np.full(100, "rising"), np.full(100, "flat_top")]
    split = pd.DataFrame({"f": [0.0, 10.0]})
    out = fc.compare_ranges_by_phase(reference, ref_phase, {"mast": split}, {"mast": np.array(["rising", "flat_top"])}, ["f"])
    assert out["phase"].tolist() == ["all", "rising", "flat_top", "late"]
    by = out.set_index("phase")
    assert by.loc["rising", "ref_median"] == 0.0 and by.loc["flat_top", "ref_median"] == 10.0
    assert by.loc["rising", "share_outside_range"] == 0.0 and by.loc["flat_top", "share_outside_range"] == 0.0
    assert by.loc["late", "n_finite"] == 0 and np.isnan(by.loc["late", "share_outside_band"])


def test_extreme_shares():
    x = np.array([[0.0, 11.0], [-10.5, 1.0], [10.0, 0.0], [0.0, -20.0]])
    assert fc.extreme_shares(x, ["a", "b"]).tolist() == [0.25, 0.5]


# ----------------------------------------------------------------------- nearest neighbours
def test_nearest_neighbours_match_an_exhaustive_search_and_group_shares_sum_to_one():
    rng = np.random.default_rng(1)
    reference = rng.normal(size=(300, 5))
    queries = rng.normal(size=(40, 5)) * 1.5
    distance, index = fc.nearest_neighbours(reference, queries, chunk=7)
    full = np.sqrt(((queries[:, None, :] - reference[None, :, :]) ** 2).sum(axis=2))
    assert np.array_equal(index, full.argmin(axis=1)) and np.allclose(distance, full.min(axis=1))
    # A query equal to a reference row has distance zero and that row as neighbour.
    d0, i0 = fc.nearest_neighbours(reference, reference[[17]])
    assert i0[0] == 17 and d0[0] == 0.0
    shares = fc.group_shares(reference, queries, index, {"g1": [0, 1], "g2": [2, 3, 4]})
    assert np.allclose(shares["g1"] + shares["g2"], 1.0)
    diff2 = (queries - reference[index]) ** 2
    assert np.allclose(shares["g1"], diff2[:, :2].sum(axis=1) / diff2.sum(axis=1))
    with pytest.raises(ValueError, match="same columns"):
        fc.nearest_neighbours(reference, queries[:, :4])


def test_sample_rows_is_seeded_sorted_and_complete_when_small():
    a, b = fc.sample_rows(1000, 50, 3), fc.sample_rows(1000, 50, 3)
    assert np.array_equal(a, b) and np.all(np.diff(a) > 0) and len(a) == 50
    assert not np.array_equal(a, fc.sample_rows(1000, 50, 4))
    assert np.array_equal(fc.sample_rows(10, 50, 0), np.arange(10))


def test_x10_samples_follow_the_seed_rule_of_the_profile():
    train, compared = fc.x10_samples(1000, {"validation": 300, "diii_d_public_test": 200, "mast_public_test": 5}, 100, 20, 7)
    assert np.array_equal(train, fc.sample_rows(1000, 100, 7))
    assert np.array_equal(compared["validation"], fc.sample_rows(300, 20, 8))
    assert np.array_equal(compared["diii_d_public_test"], fc.sample_rows(200, 20, 9))
    assert np.array_equal(compared["mast_public_test"], np.arange(5))
    assert fc.column_groups(["a", "b", "c", "d"], {"a": "g2", "b": "g1", "c": "g2", "d": "g3"}) == {"g2": [0, 2], "g1": [1], "g3": [3]}


def test_combined_distance_and_group_share_medians():
    rng = np.random.default_rng(4)
    reference = rng.normal(size=(200, 4))
    queries = {"validation": rng.normal(size=(15, 4)), "mast_public_test": rng.normal(size=(9, 4)) + 2.0}
    groups = {"g1": [0, 1], "g2": [2, 3]}
    table = fc.combined_distance(reference, queries, groups)
    assert table["split"].tolist() == ["validation"] * 15 + ["mast_public_test"] * 9
    for split, q in queries.items():
        rows = table[table["split"] == split]
        distance, nearest = fc.nearest_neighbours(reference, q)
        assert np.array_equal(rows["distance"].to_numpy(), distance) and np.array_equal(rows["nearest"].to_numpy(), nearest)
        assert np.allclose(rows["share_g1"].to_numpy(), fc.group_shares(reference, q, nearest, groups)["g1"])
    medians = fc.group_share_medians(table.assign(set="v"))
    assert medians["split"].tolist() == ["validation", "mast_public_test"]
    assert medians.loc[1, "share_g2"] == pytest.approx(table.loc[table["split"] == "mast_public_test", "share_g2"].median())


def test_distance_summary_reports_the_share_beyond_the_validation_99th_percentile():
    val = np.arange(1.0, 101.0)
    table = pd.DataFrame({"set": "set_a", "split": ["validation"] * 100 + ["mast"] * 4,
                          "distance": np.r_[val, [1.0, 50.0, 200.0, 300.0]]})
    out = fc.distance_summary(table).set_index("split")
    limit = np.percentile(val, 99.0)
    assert out.loc["mast", "validation_p99"] == pytest.approx(limit)
    assert out.loc["mast", "share_beyond_validation_p99"] == 0.5 and out.loc["validation", "share_beyond_validation_p99"] == 0.01
    assert out.loc["mast", "p50"] == pytest.approx(125.0) and out.loc["mast", "max"] == 300.0


# ------------------------------------------------------------------------- correlation
def test_correlations_and_their_summary():
    rng = np.random.default_rng(2)
    ip = rng.uniform(500, 1500, 2000)
    t = pd.DataFrame({"ip": ip, "a": 2.0 * ip + 1.0, "b": -ip + rng.normal(0, 1e-6, 2000), "c": rng.normal(size=2000)})
    t.loc[5, "c"] = np.nan
    long = fc.correlation_long(t, ["ip", "a", "b", "c"], "raw")
    assert len(long) == 16 and (long["n_rows"] == 1999).all()
    r = long.set_index(["a", "b"])["r"]
    assert r[("a", "a")] == pytest.approx(1.0) and r[("a", "b")] == pytest.approx(-1.0) and abs(r[("a", "c")]) < 0.1
    s = fc.correlation_summary(long, ["ip"])
    assert s["n_pairs"] == 3 and s["max_abs_r"] == pytest.approx(1.0) and sorted(s["max_pair"]) == ["a", "b"]
    assert s["with_ip"]["ip"]["max_abs_r"] == pytest.approx(1.0) and s["with_ip"]["ip"]["max_column"] in ("a", "b")


# ------------------------------------------------------------------------- Ip direction
def test_direction_comparison_measures_the_shift_and_the_overlap():
    rng = np.random.default_rng(3)
    v = rng.normal(size=20000)
    direction = np.r_[np.ones(10000), -np.ones(10000)]
    same = fc.direction_comparison(pd.DataFrame({"t": v}), direction, ["t"]).iloc[0]
    assert abs(same["median_shift_iqr"]) < 0.05 and same["share_negative_in_positive_band"] == pytest.approx(0.98, abs=0.01)
    shifted = v.copy()
    shifted[10000:] += 1.349 * 2.0                      # two interquartile ranges of a unit normal
    out = fc.direction_comparison(pd.DataFrame({"t": shifted}), direction, ["t"]).iloc[0]
    assert out["median_shift_iqr"] == pytest.approx(2.0, abs=0.1) and out["share_negative_in_positive_band"] < 0.75
    assert out["n_positive"] == 10000 and out["n_negative"] == 10000


# ------------------------------------------------------------------------ force balance
def test_bz_ratio_from_c10_and_the_force_balance_check():
    assert fc.bz_ratio_from_c10([-0.1], 1.67, 0.67)[0] == pytest.approx(-4 * np.pi * 1.67 / 0.67 * 0.1)
    k9 = {"DIII-D": -2.09, "MAST": -1.35}
    ok = fc.force_balance_check({"DIII-D": np.array([-2.0, -2.2, np.nan]), "MAST": np.array([-1.2, -1.4])}, k9)
    assert ok["status"] == "pass" and ok["medians"]["DIII-D"] == pytest.approx(-2.1) and ok["n_frames"]["DIII-D"] == 2
    sign = fc.force_balance_check({"DIII-D": np.array([-2.0]), "MAST": np.array([1.3])}, k9)
    assert sign["status"] == "fail" and not sign["signs_match"]
    order = fc.force_balance_check({"DIII-D": np.array([-1.0]), "MAST": np.array([-2.0])}, k9)
    assert order["status"] == "fail" and order["signs_match"] and not order["order_matches"]
    empty = fc.force_balance_check({"DIII-D": np.array([np.nan]), "MAST": np.array([-1.3])}, k9)
    assert empty["status"] == "fail" and not empty["order_matches"]
    missing = fc.force_balance_check({"DIII-D": np.array([-2.0])}, k9)
    assert missing["status"] == "fail"


# ------------------------------------------------------------------------- stretch terms
@pytest.mark.parametrize("cfg", [DIII_D, MAST], ids=["DIII-D", "MAST"])
def test_current_free_fields_give_a_small_vacuum_residual(cfg):
    r0, a0 = cfg.major_radius.value, cfg.minor_radius.value
    # psi = R^2 is quadratic in x: the order-3 fit recovers it exactly and the relation holds.
    quadratic = pd.DataFrame([fc.fitted_terms(CONFIG.layout, lambda r, z: r ** 2, r0, a0)])
    assert quadratic["coil_c20"].iloc[0] == pytest.approx(a0 ** 2) and quadratic["coil_c10"].iloc[0] == pytest.approx(2 * r0 * a0)
    assert abs(fc.vacuum_residual(quadratic, cfg)[0]) < 1e-12 * abs(quadratic["coil_c20"].iloc[0])
    # psi = R^4 - 4 R^2 Z^2 is current-free (psi_RR - psi_R / R + psi_ZZ = 0) with a quartic part
    # a0^4 (x^4 - 4 x^2 y^2) that the order-3 fit cannot hold. Its terms up to order 3 satisfy the
    # relation, so the residual is exactly what the fit takes in from the quartic part.
    quartic = pd.DataFrame([fc.fitted_terms(CONFIG.layout, lambda r, z: r ** 4 - 4 * r ** 2 * z ** 2, r0, a0)])
    alias = pd.DataFrame([fc.fitted_terms(
        CONFIG.layout, lambda r, z: ((r - r0) ** 4 - 4 * (r - r0) ** 2 * z ** 2), r0, a0)])
    residual = fc.vacuum_residual(quartic, cfg)[0]
    assert residual == pytest.approx(fc.vacuum_residual(alias, cfg)[0], rel=1e-9, abs=1e-12)
    assert abs(residual) < 0.1 * abs(quartic["coil_c20"].iloc[0])
    # Taylor coefficients of the quartic field: c10 = 4 R0^3 a0, c20 = 6 R0^2 a0^2, c02 = -4 R0^2 a0^2.
    taylor = pd.DataFrame([{"coil_c10": 4 * r0 ** 3 * a0, "coil_c20": 6 * r0 ** 2 * a0 ** 2, "coil_c02": -4 * r0 ** 2 * a0 ** 2}])
    assert abs(fc.vacuum_residual(taylor, cfg)[0]) < 1e-12 * taylor["coil_c20"].iloc[0]


def test_coil_flux_residual_vanishes_as_the_layout_shrinks():
    # The coil flux is current-free at the layout points, so the residual comes from orders 4
    # and above only: relative to coil_c20 it falls as a0^2. A field with a source at the
    # reference point would leave a constant ratio instead.
    frames = make_frames(n_shots=2, n_frames=6)
    r0 = DIII_D.major_radius.value
    sizes = []
    for a0 in DIII_D.minor_radius.value / np.array([1.0, 2.0, 4.0, 8.0]):
        built = fe.build_features(frames, "DIII-D", CONFIG, resources(point_table(a0=a0), a0=a0))
        cfg = dataclasses.replace(DIII_D, minor_radius=dataclasses.replace(DIII_D.minor_radius, value=float(a0)))
        residual = fc.vacuum_residual(built, cfg)
        assert np.isfinite(residual).all()
        sizes.append(fc.relative_size(residual, built["coil_c20"]))
    assert all(a > b for a, b in zip(sizes, sizes[1:]))
    assert sizes[-1] < 0.03 and sizes[-2] / sizes[-1] > 3.0


def test_a_known_quadrupole_gives_the_expected_shaping():
    r0, a0 = DIII_D.major_radius.value, DIII_D.minor_radius.value
    q, b = 0.4, -0.15

    def psi(r, z):
        x, y = (r - r0) / a0, z / a0
        return 0.3 * x + q * (y ** 2 - x ** 2) + b * (x ** 2 + y ** 2) + 0.05 * x * y + 0.02 * x ** 3 - 0.07 * y ** 3

    terms = pd.DataFrame([fc.fitted_terms(CONFIG.layout, psi, r0, a0)])
    assert terms["coil_c20"].iloc[0] == pytest.approx(b - q) and terms["coil_c02"].iloc[0] == pytest.approx(b + q)
    assert fc.shaping(terms)[0] == pytest.approx(q)
    assert fc.vacuum_residual(terms, DIII_D)[0] == pytest.approx(2 * b - a0 / (2 * r0) * 0.3)
    out = fc.stretch_quantities(terms.set_index(pd.Index([7])), DIII_D)
    assert list(out.columns) == ["vacuum_residual", "shaping"] and out.index.tolist() == [7]


def test_shaping_is_even_under_the_mirror():
    built = fe.build_features(make_frames(n_shots=2, n_frames=6), "DIII-D", CONFIG, resources(point_table()))
    mirrored = fe.mirror_features(built, CONFIG)
    assert np.array_equal(fc.shaping(mirrored), fc.shaping(built), equal_nan=True)
    assert np.array_equal(fc.vacuum_residual(mirrored, DIII_D), fc.vacuum_residual(built, DIII_D), equal_nan=True)


def test_relative_size_and_variant_rules():
    assert fc.relative_size([0.1, -0.2, np.nan, 0.3], [-1.0, 2.0, 3.0]) == pytest.approx(0.2 / 2.0)
    assert np.isnan(fc.relative_size([1.0], [0.0, 0.0]))
    columns = ["coil_c10", "shaping", "ts_valid", "ip_frac"]
    rules = fc.variant_rules(CONFIG, columns, unscaled=["ip_frac"], extra={"shaping": {}})
    assert list(rules) == columns
    assert rules["shaping"] == fc.ROBUST_RULE and rules["coil_c10"]["scale"] == "robust"
    assert rules["ip_frac"]["scale"] == "none" and rules["ts_valid"]["scale"] == "none"
    assert fc.variant_rules(CONFIG, ["coil_pt_00"])["coil_pt_00"]["scale_group"] == "coil_pattern"
    with pytest.raises(KeyError, match="neither configured"):
        fc.variant_rules(CONFIG, ["shaping"])
    with pytest.raises(KeyError, match="unscaled"):
        fc.variant_rules(CONFIG, ["coil_c10"], unscaled=["ip_frac"])


# -------------------------------------------------------------------------------- lines
def test_line_fit_recovers_the_line_and_the_offset_of_a_shifted_set():
    rng = np.random.default_rng(4)
    x = rng.uniform(0, 1, 5000)
    y = 2.0 + 3.0 * x + rng.normal(0, 0.01, 5000)
    y[:10] = 1e6                                         # outliers outside the band are left out
    fit = fc.line_fit(x, y)
    assert fit["slope"] == pytest.approx(3.0, abs=0.01) and fit["intercept"] == pytest.approx(2.0, abs=0.01)
    assert fit["r"] > 0.99 and fit["residual_std"] == pytest.approx(0.01, rel=0.1) and fit["n"] < 5000
    assert fc.median_offset(x[10:], y[10:] + 0.5, fit) == pytest.approx(0.5, abs=0.01)
    flat = fc.line_fit(np.ones(10), np.arange(10.0))
    assert np.isnan(flat["slope"]) and np.isnan(fc.median_offset([1.0], [1.0], flat))


# --------------------------------------------------------------------------- timescales
def test_band_table_and_changing_current_shares():
    values = {"rate": {"DIII-D": np.linspace(-1, 1, 1001), "MAST": np.linspace(-4, 4, 1001)}}
    t = fc.band_table(values).set_index("machine")
    assert t.loc["DIII-D", "width_over_reference"] == pytest.approx(1.0)
    assert t.loc["MAST", "width_over_reference"] == pytest.approx(4.0)
    assert t.loc["MAST", "q50"] == pytest.approx(0.0, abs=1e-12)
    shares = fc.changing_current_shares([0.0, 0.05, -0.2, 2.0, np.nan], ["flat_top", "flat_top", "rising", "late", "late"],
                                        [0.1, 1.0])
    assert shares["share_off_flat_top"] == 0.6 and shares["n_rate_finite"] == 4
    assert shares["share_abs_rate_rel_above_0.1"] == 0.5 and shares["share_abs_rate_rel_above_1"] == 0.25


# ------------------------------------------------------------------------------- mirror
def test_swapped_currents_give_the_mirrored_points_on_a_symmetric_coil_set():
    table = point_table()
    frames = make_frames()
    swapped = fc.swap_mirror_currents(frames, DIII_D, table.columns)
    # Only pairs with geometry in the table are exchanged (here F1 and F2).
    assert list(table.columns) == ["magnetics_F1A", "magnetics_F1B", "magnetics_F2A", "magnetics_F2B", "magnetics_ECOILA"]
    exchanged = [f"magnetics_F{i}{s}__mean" for i in (1, 2) for s in "AB"]
    assert swapped[exchanged].equals(swapped_ab(frames)[exchanged])
    assert swapped.drop(columns=exchanged).equals(frames.drop(columns=exchanged))
    original = fe.build_features(frames, "DIII-D", CONFIG, resources(table), names=["coil_pt"])
    other = fe.build_features(swapped, "DIII-D", CONFIG, resources(table), names=["coil_pt"])
    mirrored = fe.mirror_features(original, CONFIG)
    points = CONFIG.columns_of("coil_pt")
    diff = fc.pattern_difference(mirrored[points], other[points], original[points])
    assert np.nanmax(diff) < 1e-9 and np.isfinite(diff).all()
    # Without the permutation (a wrong mirror) the difference is large.
    assert np.nanmedian(fc.pattern_difference(original[points], other[points])) > 0.05
    assert fc.frames_equal(fe.mirror_features(mirrored, CONFIG), original, points)


def test_an_asymmetric_coil_set_gives_a_small_nonzero_swap_difference():
    geometry = symmetric_geometry()
    is_b = np.array([str(c).endswith("B") for c in geometry.input_columns])
    geometry.z[is_b] -= 0.005                            # B coils 5 mm low, as in F51
    table = point_table(geometry)
    frames = make_frames()
    original = fe.build_features(frames, "DIII-D", CONFIG, resources(table), names=["coil_pt"])
    other = fe.build_features(fc.swap_mirror_currents(frames, DIII_D, table.columns), "DIII-D", CONFIG, resources(table),
                              names=["coil_pt"])
    points = CONFIG.columns_of("coil_pt")
    diff = fc.pattern_difference(fe.mirror_features(original, CONFIG)[points], other[points], original[points])
    assert 1e-6 < np.median(diff) < 0.05


def test_pattern_difference_and_frames_equal_handle_nan():
    a = np.array([[0.0, 1.0, 2.0], [np.nan, 1.0, 2.0], [1.0, 1.0, 1.0]])
    b = np.array([[0.0, 1.5, 2.0], [0.0, 1.0, 2.0], [1.0, 1.0, 1.0]])
    out = fc.pattern_difference(a, b)
    assert out[0] == pytest.approx(0.25) and np.isnan(out[1]) and np.isnan(out[2])
    assert fc.pattern_difference(a[:1], b[:1], reference=np.array([[0.0, 5.0, 0.0]]))[0] == pytest.approx(0.1)
    x = pd.DataFrame({"c": [1.0, np.nan]})
    assert fc.frames_equal(x, x.copy(), ["c"]) and not fc.frames_equal(x, x.fillna(0.0), ["c"])


# -------------------------------------------------------------------------- linear check
def test_target_per_ip_round_trips_and_zeroes_small_ip():
    rng = np.random.default_rng(5)
    target = rng.normal(size=(4, 3, 3))
    ip = np.array([1000.0, -500.0, 10.0, np.nan])
    scaled, ok = fc.target_per_ip(target, ip, 1.67, 20.0)
    assert ok.tolist() == [True, True, False, False]
    assert np.allclose(scaled[0], target[0] / (MU0 * 1.67 * 1e6)) and np.allclose(scaled[1], target[1] / (MU0 * 1.67 * 5e5))
    assert (scaled[2:] == 0.0).all()
    back = fc.target_from_per_ip(scaled, ip, 1.67)
    assert np.allclose(back[:2], target[:2]) and (back[2:] == 0.0).all()
    assert fc.ip_usable([0.0, 20.0, -25.0, np.inf], 20.0).tolist() == [False, True, True, False]


def test_pooled_r2():
    t = np.array([[1.0, 2.0], [3.0, 5.0], [5.0, 8.0]])
    assert fc.pooled_r2(t, t) == 1.0
    assert fc.pooled_r2(t, np.broadcast_to(t.mean(axis=0), t.shape)) == pytest.approx(0.0)
    assert np.isnan(fc.pooled_r2(np.ones((3, 2)), np.ones((3, 2))))


def test_json_default_converts_numpy_values():
    text = json.dumps({"a": np.int64(3), "b": np.float32(0.5), "c": np.bool_(True), "d": np.arange(2)}, default=fc.json_default)
    assert json.loads(text) == {"a": 3, "b": 0.5, "c": True, "d": [0, 1]}
    with pytest.raises(TypeError):
        json.dumps({"x": object()}, default=fc.json_default)


def test_pooled_r2_by_group_and_variance_shares():
    rng = np.random.default_rng(5)
    true = rng.normal(size=(40, 3))
    true[:10] *= 10.0                                       # the first group holds most of the variance
    pred = true + rng.normal(scale=0.5, size=true.shape)
    labels = np.array(["a"] * 10 + ["b"] * 29 + ["c"])
    r2 = fc.pooled_r2_by_group(true, pred, labels, ["a", "b", "c", "d"])
    assert r2["a"] == pytest.approx(fc.pooled_r2(true[:10], pred[:10])) and np.isnan(r2["c"]) and np.isnan(r2["d"])
    assert r2["b"] < r2["a"]
    shares = fc.variance_shares(true, labels, ["a", "b", "c"])
    assert sum(shares.values()) == pytest.approx(1.0) and shares["a"] > 0.8
    d2 = ((true - true.mean(axis=0)) ** 2).sum(axis=1)
    assert shares["b"] == pytest.approx(d2[10:39].sum() / d2.sum())


def test_vacuum_residual_takes_the_radii_of_the_build():
    terms = pd.DataFrame({"coil_c10": [0.4], "coil_c20": [0.1], "coil_c02": [0.05]})
    assert fc.vacuum_residual(terms, MAST)[0] == pytest.approx(0.15 - 0.65 / (2 * 0.85) * 0.4)
    assert fc.vacuum_residual(terms, MAST, 0.70, 0.50)[0] == pytest.approx(0.15 - 0.50 / (2 * 0.70) * 0.4)
    out = fc.stretch_quantities(terms, MAST, 0.70, 0.50)
    assert out["vacuum_residual"].iloc[0] == pytest.approx(0.15 - 0.50 / 1.40 * 0.4) and out["shaping"].iloc[0] == pytest.approx(-0.025)
