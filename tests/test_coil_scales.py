"""Tests for the per-frame decomposition, the sign rule, the scale fit and the symmetry checks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fusion_eq import coil_flux as cf, coil_scales as cs, io, starter_kit
from fusion_eq.machines import DIII_D
from tests.conftest import make_d3d_shot, write_shot

GRID_R = np.linspace(0.84, 2.54, 65)
GRID_Z = np.linspace(-1.6, 1.6, 65)


def shot_index(rows):
    """Shot index rows ``(split, shot, n_frames, mag_rate_hz, ip_direction, error)``."""
    return pd.DataFrame(rows, columns=["split", "shot", "n_frames", "mag_rate_hz", "ip_direction", "error"])


def test_reference_shots_picks_three_kinds_in_sorted_order_without_overlap():
    index = shot_index([
        ("diii_d_train", "b", 100, 20000, 1, None), ("diii_d_train", "a", 100, 20000, 1, None),
        ("diii_d_train", "c", 100, 20000, 1, None), ("diii_d_train", "d", 100, 20000, 1, None),
        ("diii_d_train", "n2", 100, 2000, -1, None), ("diii_d_train", "n1", 100, 20000, -1, None),
        ("diii_d_train", "s1", 100, 2000, 1, None), ("diii_d_train", "s2", 100, 2000, 1, None),
        ("diii_d_train", "s3", 100, 2000, 1, None),
        ("diii_d_train", "short", 10, 20000, 1, None),           # too few frames
        ("diii_d_train", "bad", 100, 20000, 1, "ValueError: x"),  # build error
        ("diii_d_train", "mixed", 100, 20000, 0, None),           # mixed Ip direction
        ("diii_d_public_test", "0", 100, 20000, 1, None),         # wrong split
    ])
    refs = cs.reference_shots(index)
    assert refs == {"positive_20khz": ["a", "b", "c"], "negative_ip": ["n1", "n2"], "rate_2khz": ["s1", "s2"]}
    assert cs.reference_shots(index, n_fast_positive=1, n_negative=1, n_slow=3) == {
        "positive_20khz": ["a"], "negative_ip": ["n1"], "rate_2khz": ["s1", "s2", "s3"]}
    assert cs.reference_shots(index.iloc[:0]) == {"positive_20khz": [], "negative_ip": [], "rate_2khz": []}


def test_reference_frames_rising_mid_flat_top_late():
    trace = np.concatenate([np.linspace(0.0, 1.0, 11), np.ones(9), np.linspace(1.0, 0.0, 11)])
    frames = cs.reference_frames(trace)
    assert list(frames) == ["rising", "flat_top", "late"]
    assert frames["rising"] == 5 and trace[5] == pytest.approx(0.5) and trace[4] < 0.5
    assert frames["late"] == 25 and trace[25] == pytest.approx(0.5) and trace[26] < 0.5
    flat = np.flatnonzero(trace >= 0.9)
    assert frames["flat_top"] == flat[flat.size // 2] and trace[frames["flat_top"]] == 1.0
    # NaN frames never qualify and the rise can start inside the window.
    late_start = np.concatenate([[np.nan], np.linspace(0.6, 1.0, 5), [0.95, 0.7, 0.4]])
    assert cs.reference_frames(late_start) == {"rising": 1, "flat_top": 5, "late": 7}


def test_reference_frames_fallbacks():
    # Never reaches the flat-top level: the peak stands in for the flat-top frame.
    low = np.array([0.1, 0.55, 0.6, 0.7, 0.65, 0.55, 0.2])
    assert cs.reference_frames(low) == {"rising": 1, "flat_top": 3, "late": 5}
    # Choices coincide (one frame above the rise level): first, middle, last frame.
    spike = np.array([0.1, 0.1, 0.1, 1.0, 0.1, 0.1])
    assert cs.reference_frames(spike) == {"rising": 0, "flat_top": 3, "late": 5}
    with pytest.raises(ValueError):
        cs.reference_frames([0.5, 1.0])


def test_sign_rule_from_ratios():
    rule = cs.sign_rule([1.0, 0.98, -1.02, np.nan], [1.0, 1.0, -1.0, 1.0])
    assert rule["c"] == 1 and rule["consistent"] and rule["n_shots"] == 3
    assert rule["n_positive_ip"] == 2 and rule["n_negative_ip"] == 1
    assert rule["ratio_sign_by_direction"] == {1: [1], -1: [-1]}
    rule = cs.sign_rule([-1.0, 1.0], [1.0, -1.0])
    assert rule["c"] == -1 and rule["consistent"]
    rule = cs.sign_rule([-1.0, -1.0], [1.0, -1.0])
    assert rule["c"] == 0 and not rule["consistent"]
    assert cs.sign_rule([], [])["consistent"] is False
    assert cs.orientation_from_rule(-1, 800.0) == -1.0 and cs.orientation_from_rule(-1, -800.0) == 1.0


def test_rms_fraction():
    target = np.ones((3, 3))
    model = np.zeros((3, 3))
    mask = np.ones((3, 3), dtype=bool)
    assert cs.rms_fraction(target, model, mask, 2.0) == pytest.approx(0.5)
    assert cs.rms_fraction(target, model, mask, 2.0, constant=1.0) == 0.0
    assert np.isnan(cs.rms_fraction(target, model, np.zeros((3, 3), dtype=bool), 2.0))


def synthetic_fit_data(n_frames=12, n_cols=4, seed=0):
    rng = np.random.default_rng(seed)
    rr, zz = np.meshgrid(GRID_R, GRID_Z)
    patterns = np.stack([np.exp(-((rr - 1.2 - 0.3 * k) ** 2 + (zz - 0.5 * (k - 1.5)) ** 2) / 0.5) for k in range(n_cols)])
    true_scales = np.array([1.0, 0.8, 1.3, 23.0])[:n_cols]
    currents = rng.normal(size=(n_frames, n_cols))
    constants = rng.normal(size=n_frames) * 0.3
    mask = (rr - 1.7) ** 2 + zz ** 2 > 0.5 ** 2
    return patterns, true_scales, currents, constants, mask


COLUMNS = ("c0", "c1", "c2", "c3")
SOL = "c3"


def normal_of(target, coil_maps, mask, sol_ka, flux_range=1.0):
    """FrameNormal of a synthetic frame without going through analyze_frame."""
    a = cs.FrameAnalysis(orientation=1.0, boundary_found=True, contour=None, inside=~mask, fit_mask=mask,
                         near_coil_mask=np.zeros_like(mask), current_a=0.0, ratio=1.0, oriented_psi=target,
                         target=target, coil_maps=coil_maps, flux_range=flux_range)
    return cs.FrameNormal.from_frame(a, sol_ka)


def synthetic_frames(flat=0.0, constants=None, scales=None, seed=0):
    patterns, true_scales, currents, consts, mask = synthetic_fit_data(seed=seed)
    scales = true_scales if scales is None else scales
    frames = []
    for k, c in enumerate(currents):
        coil_maps = patterns * c[:, None, None]
        target = np.tensordot(scales, coil_maps, axes=1) + flat * c[3]
        if constants is not None:
            target = target + constants[k]
        frames.append((target, coil_maps, mask, c[3]))
    return frames, true_scales, consts


def test_standard_variants_cover_the_columns_and_check_overlaps():
    variants = cs.standard_variants(COLUMNS, SOL)
    assert set(variants) == {"per_coil", "per_coil_constant", "solenoid_only", "solenoid_only_constant",
                             "shared_plus_solenoid", "shared_plus_solenoid_constant", "solenoid_flat",
                             "solenoid_flat_plus_pattern"}
    assert variants["per_coil"].parameters == COLUMNS
    assert variants["solenoid_flat"].parameters == ("flat",) and variants["solenoid_flat"].fixed == ("c0", "c1", "c2")
    assert variants["solenoid_flat_plus_pattern"].parameters == ("c3", "flat")
    assert variants["shared_plus_solenoid"].parameters == ("c3", "shared")
    with pytest.raises(ValueError):
        cs.FitVariant("bad", free=("c0",), fixed=("c0",)).check(COLUMNS)
    with pytest.raises(ValueError):
        cs.standard_variants(COLUMNS, "c9")


def test_frame_normal_centred_matches_direct_sums():
    frames, _, _ = synthetic_frames(flat=0.4)
    target, coil_maps, mask, sol = frames[0]
    normal = normal_of(target, coil_maps, mask, sol)
    base = np.vstack([coil_maps[:, mask], np.full((1, mask.sum()), sol), np.ones((1, mask.sum()))])
    y = target[mask]
    assert np.allclose(normal.gram, base @ base.T) and np.allclose(normal.moment, base @ y) and normal.n == mask.sum()
    centred = normal.centred()
    bc, yc = base - base.mean(axis=1, keepdims=True), y - y.mean()
    assert np.allclose(centred.gram, bc @ bc.T) and np.allclose(centred.moment, bc @ yc)
    assert centred.sum_sq == pytest.approx(yc @ yc)


def test_scale_fit_per_coil_recovers_scales_and_frame_rms_matches_direct():
    frames, true_scales, _ = synthetic_frames()
    variants = cs.standard_variants(COLUMNS, SOL)
    normals = [normal_of(*f) for f in frames]
    solution = cs.fit_variants(normals, COLUMNS, {"per_coil": variants["per_coil"]})["per_coil"]
    assert np.allclose(solution.scales, true_scales, rtol=1e-8) and solution.flat_wb_per_ka == 0.0
    assert solution.coefficients() == pytest.approx(dict(zip(COLUMNS, true_scales)))
    target, coil_maps, mask, sol = frames[2]
    assert np.allclose(solution.model(coil_maps, sol), np.tensordot(true_scales, coil_maps, axes=1))
    assert solution.frame_rms(normals[2]) == pytest.approx(0.0, abs=1e-6)
    # A wrong solution: the rms from the sums equals the rms from the maps.
    wrong = cs.FitSolution(variants["per_coil"], COLUMNS, true_scales * 1.1)
    direct = cs.rms_fraction(target, wrong.model(coil_maps, sol), mask, 1.0)
    assert wrong.frame_rms(normals[2]) == pytest.approx(direct, rel=1e-8)
    assert wrong.frame_mean_residual(normals[2]) == pytest.approx(np.mean((target - wrong.model(coil_maps, sol))[mask]))
    assert np.isnan(cs.ScaleFit(COLUMNS, variants["per_coil"]).solve().theta).all()


def test_scale_fit_with_constants_recovers_scales_and_constants():
    _, true_scales, constants = synthetic_frames()
    frames, _, _ = synthetic_frames(constants=constants)
    variants = cs.standard_variants(COLUMNS, SOL)
    normals = [normal_of(*f) for f in frames]
    solutions = cs.fit_variants(normals, COLUMNS, variants)
    assert np.allclose(solutions["per_coil_constant"].scales, true_scales, rtol=1e-8)
    assert not np.allclose(solutions["per_coil"].scales, true_scales, rtol=1e-3)
    for normal, b in zip(normals[:3], constants[:3]):
        assert solutions["per_coil_constant"].frame_mean_residual(normal) == pytest.approx(b, abs=1e-8)
        assert solutions["per_coil_constant"].frame_rms(normal) == pytest.approx(0.0, abs=1e-6)


def test_solenoid_flat_variants_recover_the_flat_term():
    patterns = synthetic_fit_data()[0]
    variants = cs.standard_variants(COLUMNS, SOL)
    # Shaping coils at 1, solenoid pattern at 0.7, plus a flat term of 0.05 Wb/rad per kA.
    frames, _, _ = synthetic_frames(flat=0.05, scales=np.array([1.0, 1.0, 1.0, 0.7]))
    normals = [normal_of(*f) for f in frames]
    solutions = cs.fit_variants(normals, COLUMNS, variants)
    both = solutions["solenoid_flat_plus_pattern"]
    assert both.coefficients()["c3"] == pytest.approx(0.7, abs=1e-8) and both.flat_wb_per_ka == pytest.approx(0.05, abs=1e-8)
    assert np.allclose(both.scales, [1.0, 1.0, 1.0, 0.7]) and both.frame_rms(normals[0]) == pytest.approx(0.0, abs=1e-6)
    # The flat-only variant leaves the pattern out and absorbs what it can; its residual is larger.
    flat_only = solutions["solenoid_flat"]
    assert flat_only.scales.tolist() == [1.0, 1.0, 1.0, 0.0] and flat_only.frame_rms(normals[0]) > 1e-3
    # Without the pattern in the data the flat variant is exact.
    frames_flat, _, _ = synthetic_frames(flat=0.05, scales=np.array([1.0, 1.0, 1.0, 0.0]))
    normals_flat = [normal_of(*f) for f in frames_flat]
    exact = cs.fit_variants(normals_flat, COLUMNS, {"solenoid_flat": variants["solenoid_flat"]})["solenoid_flat"]
    assert exact.flat_wb_per_ka == pytest.approx(0.05, abs=1e-8) and exact.frame_rms(normals_flat[1]) == pytest.approx(0.0, abs=1e-6)
    # The flat term goes into a CoilScales record that the Green's table applies.
    record = both.to_coil_scales("DIII-D", source="test")
    assert record.flat_wb_per_ka == pytest.approx(0.05, abs=1e-8) and record.scales["c3"] == pytest.approx(0.7, abs=1e-8)
    assert record.scales["c0"] == 1.0 and record.source == "test"


def test_shared_and_solenoid_only_variants():
    variants = cs.standard_variants(COLUMNS, SOL)
    frames, _, _ = synthetic_frames(scales=np.array([1.2, 1.2, 1.2, 15.0]))
    normals = [normal_of(*f) for f in frames]
    solutions = cs.fit_variants(normals, COLUMNS, variants)
    shared = solutions["shared_plus_solenoid"]
    assert shared.coefficients()["shared"] == pytest.approx(1.2, abs=1e-8) and shared.coefficients()["c3"] == pytest.approx(15.0, abs=1e-8)
    assert np.allclose(shared.scales, [1.2, 1.2, 1.2, 15.0])
    only = solutions["solenoid_only"]
    assert only.scales[:3].tolist() == [1.0, 1.0, 1.0] and only.frame_rms(normals[0]) > 1e-3


def test_scale_fit_skips_frames_with_too_few_nodes_and_handles_dead_columns():
    frames, true_scales, _ = synthetic_frames()
    variant = cs.standard_variants(COLUMNS, SOL)["per_coil"]
    fit = cs.ScaleFit(COLUMNS, variant)
    tiny = np.zeros_like(frames[0][2])
    tiny[0, 0] = True
    fit.add(normal_of(frames[0][0], frames[0][1], tiny, 0.0))
    assert fit.n_frames == 0
    for target, coil_maps, mask, sol in frames:
        dead = coil_maps.copy()
        dead[3] = 0.0
        fit.add(normal_of(np.tensordot(true_scales, dead, axes=1), dead, mask, 0.0))
    solution = fit.solve()
    assert np.allclose(solution.scales[:3], true_scales[:3], rtol=1e-8) and np.isnan(solution.scales[3])
    assert np.allclose(solution.base[:3], true_scales[:3]) and solution.base[3] == 0.0


def test_xpoint_class_proxy():
    grid_r, grid_z = np.linspace(0.5, 1.5, 41), np.linspace(-1.0, 1.0, 41)
    rr, zz = np.meshgrid(grid_r, grid_z)
    theta = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    contour = np.column_stack([1.0 + 0.4 * np.cos(theta), 0.4 * np.sin(theta)])
    # Nulls of grad psi at Z = +-0.4 on the circle; the extra term lifts the upper one.
    double = (zz ** 2 - 0.16) ** 2 + 0.1 * (rr - 1.0) ** 2
    lower = double + 0.3 * (zz + 0.4) ** 2
    assert cs.xpoint_class(contour, lower, grid_r, grid_z) == "lower"
    assert cs.xpoint_class(contour, lower[::-1, :], grid_r, grid_z) == "upper"
    assert cs.xpoint_class(contour, double, grid_r, grid_z) == "both"
    assert cs.xpoint_class(None, lower, grid_r, grid_z) == "none"


def mirror_geometry(symmetric=True):
    r_b = 1.0 if symmetric else 1.3
    return cf.CoilGeometry(np.array(["F5A", "F5B", "E"]), np.array(["magnetics_F5A", "magnetics_F5B", "magnetics_ECOILA"]),
                           np.array([1.0, r_b, 0.72]), np.array([1.5, -1.5, 0.0]), np.array([0.14, 0.14, 0.13]),
                           np.array([0.12, 0.12, 3.5]), np.array([45.0, -45.0, 0.0]), np.zeros(3))


def test_mirror_difference_zero_for_mirror_coils_and_positive_otherwise():
    table = cf.build_green_table(DIII_D, mirror_geometry(), GRID_R, GRID_Z, n_filaments=2)
    assert cs.mirror_pairs(table.columns, DIII_D) == [(0, 1)]
    assert cs.mirror_difference(table, np.array([1.0, 2.5, -0.5]), DIII_D) < 1e-10
    skewed = cf.build_green_table(DIII_D, mirror_geometry(symmetric=False), GRID_R, GRID_Z, n_filaments=2)
    assert cs.mirror_difference(skewed, np.array([1.0, 2.5, -0.5]), DIII_D) > 1e-3


def test_reference_ratios_signs_and_solenoid_change():
    table = cf.build_green_table(DIII_D, mirror_geometry(), GRID_R, GRID_Z, n_filaments=2)
    currents = np.array([[0.0, 0.0, 10.0], [0.0, 0.0, 10.0], [5.0, 5.0, -30.0]])
    ip = np.array([1.0, 500.0, 1000.0])
    out = cs.reference_ratios(table, currents, ip, DIII_D, GRID_R, GRID_Z, start_frame=0, flat_frame=2)
    assert GRID_R[out["ir"]] == pytest.approx(DIII_D.reference_radius_m, abs=0.02) and GRID_Z[out["iz"]] == 0.0
    assert np.isfinite(out["bz_ratio"]) and out["sol_flux_ratio"] < 0          # solenoid swung down
    same = cs.reference_ratios(table, currents, ip, DIII_D, GRID_R, GRID_Z, start_frame=0, flat_frame=1)
    assert same["sol_flux_ratio"] == 0.0
    # A signed Ip keeps the ratio's sign on a reversed discharge; a flat solenoid term adds
    # flat x delta I_sol / (mu0 R0 Ip) to the solenoid flux ratio.
    reversed_ = cs.reference_ratios(table, -currents, -ip, DIII_D, GRID_R, GRID_Z, start_frame=0, flat_frame=2)
    assert reversed_["bz_ratio"] == pytest.approx(out["bz_ratio"]) and reversed_["sol_flux_ratio"] == pytest.approx(out["sol_flux_ratio"])
    from fusion_eq.coil_flux import CoilScales
    flat = CoilScales("DIII-D", {c: 1.0 for c in table.columns}, flat_wb_per_ka=0.01)
    with_flat = cs.reference_ratios(table, currents, ip, DIII_D, GRID_R, GRID_Z, 0, 2, scales=flat)
    expected = out["sol_flux_ratio"] + 0.01 * (-30.0 - 10.0) / (cs.MU0 * DIII_D.reference_radius_m * 1000.0 * 1000.0)
    assert with_flat["sol_flux_ratio"] == pytest.approx(expected, rel=1e-9)


def test_analyze_frame_on_synthetic_shot(tmp_path, starter):
    raw = make_d3d_shot(n_frames=2)
    shot = io.load_shot(write_shot(tmp_path / "d3d_shot_1.parquet", raw), apply=False)
    geometry = cf.CoilGeometry.from_shot(shot)
    table = cf.build_green_table(DIII_D, geometry, GRID_R, GRID_Z, n_filaments=1)
    env = starter_kit.envelope_mask(DIII_D)
    scorer = starter_kit.scoring()
    near = cf.near_coil_mask(geometry, GRID_R, GRID_Z)
    psi = np.asarray(raw["efit_psirz"][0], dtype=np.float64)
    currents = np.linspace(0.5, 1.5, len(table.columns))
    a = cs.analyze_frame(psi, GRID_R, GRID_Z, DIII_D, env, table, currents, ip_ka=1000.0, near_coil=near, scorer=scorer)
    assert a.boundary_found and a.inside.any()
    assert a.contour.ndim == 2 and a.contour.shape[1] == 2 and a.contour.shape[0] >= 3
    assert np.isfinite(a.current_a) and np.isfinite(a.ratio)
    # The synthetic map is a negative Gaussian (axis at the minimum, as on DIII-D), so the
    # stored map carries negative current density and the orientation flips it.
    assert a.orientation == -1.0 and a.current_a < 0
    assert not (a.fit_mask & a.inside).any() and not (a.fit_mask & a.near_coil_mask).any()
    assert np.isnan(a.target[a.inside]).all() and np.isfinite(a.target[a.fit_mask]).all()
    assert a.coil_maps.shape == (19, 65, 65) and np.allclose(a.coil_total, table.flux(currents))
    assert np.allclose(a.scaled_coil_flux(np.full(19, 2.0)), 2.0 * a.coil_total)
    assert a.flux_range == pytest.approx(psi.max() - psi.min())
    forced = cs.analyze_frame(psi, GRID_R, GRID_Z, DIII_D, env, table, currents, 1000.0, near, scorer, orientation=1.0)
    assert forced.orientation == 1.0 and np.allclose(forced.oriented_psi, psi)
    # The same stored map with a negative Ip is a negative-current discharge stored as it is: the
    # physical orientation keeps it, so that the coil flux from the (negative) measured currents
    # is compared in the same frame, and the remainder's hill is a minimum.
    negative = cs.analyze_frame(psi, GRID_R, GRID_Z, DIII_D, env, table, -currents, ip_ka=-1000.0, near_coil=near, scorer=scorer)
    assert negative.orientation == 1.0 and negative.ratio == pytest.approx(-a.ratio) and negative.ratio > 0
    assert np.allclose(negative.oriented_psi, psi) and np.allclose(negative.target, -a.target, equal_nan=True)
    assert negative.orientation == cs.orientation_from_rule(-1, -1000.0) and a.orientation == cs.orientation_from_rule(-1, 1000.0)
    rem = cs.remainder_metrics(a.oriented_psi, np.zeros_like(psi), GRID_R, GRID_Z, DIII_D, env, scorer,
                               float(raw["efit_r_axis"][0]), float(raw["efit_z_axis"][0]), 1000.0)
    assert rem["rem_axis_dist_m"] < 0.06 and rem["rem_amplitude"] > 0
    assert rem["rem_amplitude_over_ip"] == pytest.approx(rem["rem_amplitude"] / 1000.0)
    rem_neg = cs.remainder_metrics(negative.oriented_psi, np.zeros_like(psi), GRID_R, GRID_Z, DIII_D, env, scorer,
                                   float(raw["efit_r_axis"][0]), float(raw["efit_z_axis"][0]), -1000.0)
    assert rem_neg["rem_axis_dist_m"] == pytest.approx(rem["rem_axis_dist_m"])
    assert rem_neg["rem_amplitude"] == pytest.approx(rem["rem_amplitude"])
    assert rem_neg["rem_amplitude_over_ip"] == pytest.approx(rem["rem_amplitude_over_ip"])
