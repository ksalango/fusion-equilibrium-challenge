"""Tests for the per-frame decomposition, the sign rule, the scale fit and the symmetry checks."""

from __future__ import annotations

import numpy as np
import pytest

from fusion_eq import coil_flux as cf, coil_scales as cs, io, starter_kit
from fusion_eq.machines import DIII_D
from tests.conftest import make_d3d_shot, write_shot

GRID_R = np.linspace(0.84, 2.54, 65)
GRID_Z = np.linspace(-1.6, 1.6, 65)


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


def test_scale_fit_recovers_scales_without_constants():
    patterns, true_scales, currents, _, mask = synthetic_fit_data()
    columns = tuple(f"c{k}" for k in range(len(true_scales)))
    fit = cs.ScaleFit(columns)
    for c in currents:
        coil_maps = patterns * c[:, None, None]
        target = np.tensordot(true_scales, coil_maps, axes=1)
        fit.add(target, coil_maps, mask)
    assert fit.n_frames == len(currents)
    assert np.allclose(fit.solve(), true_scales, rtol=1e-8)
    assert np.isnan(cs.ScaleFit(columns).solve()).all()


def test_scale_fit_with_constants_recovers_scales_and_constants():
    patterns, true_scales, currents, constants, mask = synthetic_fit_data()
    columns = tuple(f"c{k}" for k in range(len(true_scales)))
    plain, with_const = cs.ScaleFit(columns), cs.ScaleFit(columns, with_constant=True)
    frames = []
    for c, b in zip(currents, constants):
        coil_maps = patterns * c[:, None, None]
        target = np.tensordot(true_scales, coil_maps, axes=1) + b
        plain.add(target, coil_maps, mask)
        with_const.add(target, coil_maps, mask)
        frames.append((target, coil_maps, b))
    scales = with_const.solve()
    assert np.allclose(scales, true_scales, rtol=1e-8)
    assert not np.allclose(plain.solve(), true_scales, rtol=1e-3)
    for target, coil_maps, b in frames[:3]:
        assert with_const.constant_for(target, coil_maps, mask, scales) == pytest.approx(b, abs=1e-8)
    assert plain.constant_for(frames[0][0], frames[0][1], mask, scales) == 0.0


def test_scale_fit_skips_frames_with_too_few_nodes_and_handles_dead_columns():
    patterns, true_scales, currents, _, mask = synthetic_fit_data()
    columns = ("c0", "c1", "c2", "c3")
    fit = cs.ScaleFit(columns)
    tiny = np.zeros_like(mask)
    tiny[0, 0] = True
    fit.add(np.zeros((65, 65)), patterns, tiny)
    assert fit.n_frames == 0
    for c in currents:
        c = c.copy()
        c[3] = 0.0
        coil_maps = patterns * c[:, None, None]
        fit.add(np.tensordot(true_scales, coil_maps, axes=1), coil_maps, mask)
    scales = fit.solve()
    assert np.allclose(scales[:3], true_scales[:3], rtol=1e-8) and np.isnan(scales[3])


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
    rem = cs.remainder_metrics(a.oriented_psi, np.zeros_like(psi), GRID_R, GRID_Z, DIII_D, env, scorer,
                               float(raw["efit_r_axis"][0]), float(raw["efit_z_axis"][0]), 1000.0)
    assert rem["rem_axis_dist_m"] < 0.06 and rem["rem_amplitude"] > 0
    assert rem["rem_amplitude_over_ip"] == pytest.approx(rem["rem_amplitude"] / 1000.0)
