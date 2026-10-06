"""Tests for the filament Green's function, element subdivision and Green's tables."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from matplotlib.path import Path as PolygonPath
from scipy.integrate import quad

from fusion_eq import coil_flux as cf, frame_table as ft, io
from fusion_eq.machines import DIII_D, MAST
from fusion_eq.qa.geometry import check_coil_geometry
from tests.conftest import make_d3d_shot, write_shot

GRID_R = np.linspace(0.84, 2.54, 65)
GRID_Z = np.linspace(-1.6, 1.6, 65)


def vector_potential_flux(r, z, rc, zc, current_a):
    """psi = R A_phi of a circular loop by direct quadrature of the Biot-Savart integral."""
    def integrand(phi):
        return rc * np.cos(phi) / np.sqrt(r * r + rc * rc - 2.0 * r * rc * np.cos(phi) + (z - zc) ** 2)
    a_phi = cf.MU0 * current_a / (4.0 * np.pi) * quad(integrand, 0.0, 2.0 * np.pi, epsabs=0, epsrel=1e-12, limit=400)[0]
    return r * a_phi


# ----------------------------------------------------------------------- Green's function
@pytest.mark.parametrize("point", [(1.5, 0.3), (0.9, -1.2), (2.2, 0.0), (1.02, 0.21)])
def test_filament_flux_matches_numerical_integration(point):
    r, z = point
    rc, zc, current = 1.0, 0.2, 1500.0
    expected = vector_potential_flux(r, z, rc, zc, current)
    assert cf.filament_flux(r, z, rc, zc, current) == pytest.approx(expected, rel=1e-6)


def test_field_on_axis_of_a_loop():
    a, current = 1.0, 1000.0
    for r in (1e-3, 2e-3):
        psi = cf.filament_flux(r, 0.0, a, 0.0, current)
        bz = 2.0 * psi / r ** 2
        assert bz == pytest.approx(cf.MU0 * current / (2.0 * a), rel=1e-5)
    assert cf.filament_flux(0.0, 0.3, a, 0.0, current) == 0.0


def test_filament_flux_symmetric_in_source_and_target():
    assert cf.filament_flux(1.3, 0.4, 0.9, -0.2) == pytest.approx(cf.filament_flux(0.9, -0.2, 1.3, 0.4), rel=1e-12)


def test_filament_flux_sign_linearity_and_floor():
    psi = cf.filament_flux(1.1, 0.0, 1.0, 0.0, 100.0)
    assert psi > 0
    assert cf.filament_flux(1.1, 0.0, 1.0, 0.0, -200.0) == pytest.approx(-2.0 * psi)
    assert np.isinf(cf.filament_flux(1.0, 0.0, 1.0, 0.0, 1.0))
    floored = cf.filament_flux(1.0, 0.0, 1.0, 0.0, 1.0, min_distance_m=0.01)
    assert np.isfinite(floored)
    assert floored == pytest.approx(cf.filament_flux(1.0, 0.01, 1.0, 0.0, 1.0), rel=1e-12)
    with pytest.raises(ValueError):
        cf.filament_flux(-0.1, 0.0, 1.0, 0.0)


def test_filament_flux_on_grid_sums_filaments():
    rc = np.array([1.0, 1.2, 0.9])
    zc = np.array([0.0, 0.5, -0.3])
    current = np.array([10.0, -5.0, 2.0])
    grid = cf.filament_flux_on_grid(GRID_R, GRID_Z, rc, zc, current)
    rr, zz = np.meshgrid(GRID_R, GRID_Z)
    direct = sum(cf.filament_flux(rr, zz, rc[i], zc[i], current[i]) for i in range(3))
    assert grid.shape == (65, 65) and np.allclose(grid, direct, rtol=1e-12, atol=0)


# ------------------------------------------------------------------------------ elements
def polygon_area(outline):
    x, y = outline[:, 0], outline[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def test_rectangle_filaments_and_sub_cell():
    r, z, sub_w, sub_h = cf.element_filaments(1.0, 0.5, 0.2, 0.2, 0.0, 0.0, 2)
    assert sorted(r.tolist()) == pytest.approx([0.95, 0.95, 1.05, 1.05])
    assert sorted(z.tolist()) == pytest.approx([0.45, 0.45, 0.55, 0.55])
    assert (sub_w, sub_h) == pytest.approx((0.1, 0.1))
    # A tall element gets more filaments along Z so that sub-cells stay near square.
    assert cf.filament_counts(0.05, 0.32, 4) == (4, 26) and cf.filament_counts(0.3, 0.1, 2) == (6, 2)
    r, z, sub_w, sub_h = cf.element_filaments(1.0, 0.5, 0.2, 0.4, 0.0, 0.0, 2)
    assert r.size == 8 and (sub_w, sub_h) == pytest.approx((0.1, 0.1))
    assert sorted(set(np.round(z, 9))) == pytest.approx([0.35, 0.45, 0.55, 0.65])
    with pytest.raises(ValueError):
        cf.element_filaments(1.0, 0.5, 0.2, 0.4, 45.0, 92.4, 2)
    with pytest.raises(ValueError):
        cf.element_filaments(1.0, 0.5, 0.2, 0.4, 0.0, 0.0, 0)


@pytest.mark.parametrize("angles", [(45.0, 0.0), (-45.0, 0.0), (0.0, 92.4), (0.0, -108.06)])
def test_parallelogram_filaments_inside_outline_with_preserved_area(angles):
    rc, zc, w, h = 1.0041, 1.5169, 0.1392, 0.1194
    outline = cf.element_outline(rc, zc, w, h, *angles)
    assert polygon_area(outline) == pytest.approx(w * h, rel=1e-12)
    r, z, _, _ = cf.element_filaments(rc, zc, w, h, *angles, 6)
    assert PolygonPath(outline).contains_points(np.column_stack([r, z]), radius=-1e-9).all()
    assert r.mean() == pytest.approx(rc) and z.mean() == pytest.approx(zc)


def test_skew_convention_matches_efit_rule():
    r, z, _, _ = cf.element_filaments(1.0, 0.0, 0.2, 0.2, 45.0, 0.0, 1)
    assert r.size == 1 and r[0] == pytest.approx(1.0) and z[0] == pytest.approx(0.0)
    r, z, _, _ = cf.element_filaments(1.0, 0.0, 0.2, 0.2, 45.0, 0.0, 2)
    order = np.argsort(r)
    # angle1: Z rises with R by tan(45) = 1 per metre of radial offset; sides stay vertical.
    assert np.allclose(z[order][:2] - np.sort(z[order][:2]), 0) and z[order][2:].mean() - z[order][:2].mean() == pytest.approx(0.1)
    # angle2 = 90 is the plain rectangle; 92.4 leans the top inward (R decreases with Z).
    r90, z90, _, _ = cf.element_filaments(2.6, 0.4, 0.17, 0.19, 0.0, 90.0, 3)
    rrect, zrect, _, _ = cf.element_filaments(2.6, 0.4, 0.17, 0.19, 0.0, 0.0, 3)
    assert np.allclose(r90, rrect) and np.allclose(z90, zrect)
    r, z, _, _ = cf.element_filaments(2.6, 0.4, 0.17, 0.19, 0.0, 92.4, 3)
    top, bottom = z > 0.4 + 0.05, z < 0.4 - 0.05
    assert r[top].mean() < r[bottom].mean()
    assert r[top].mean() - r[bottom].mean() == pytest.approx((z[top].mean() - z[bottom].mean()) / np.tan(np.radians(92.4)))


def test_convergence_of_subdivision_away_from_the_coil():
    geometry = cf.CoilGeometry(np.array(["F"]), np.array(["magnetics_F1A"]), np.array([0.8608]), np.array([0.1683]),
                               np.array([0.0508]), np.array([0.32106]), np.array([0.0]), np.array([0.0]))
    coarse = cf.column_pattern(DIII_D, geometry, "magnetics_F1A", GRID_R, GRID_Z, 2)
    fine = cf.column_pattern(DIII_D, geometry, "magnetics_F1A", GRID_R, GRID_Z, 4)
    finer = cf.column_pattern(DIII_D, geometry, "magnetics_F1A", GRID_R, GRID_Z, 8)
    near = cf.near_coil_mask(geometry, GRID_R, GRID_Z)
    rng = coarse.max() - coarse.min()
    step1 = np.abs(fine - coarse)[~near].max() / rng
    step2 = np.abs(finer - fine)[~near].max() / rng
    assert step2 < step1 and step2 < 1e-3
    # Nodes inside and next to the coil converge too, thanks to the distance floor.
    assert np.abs(finer - fine)[near].max() / rng < np.abs(fine - coarse)[near].max() / rng
    assert np.isfinite(coarse).all() and np.isfinite(fine).all()
    single = cf.filament_flux_on_grid(GRID_R, GRID_Z, [0.8608], [0.1683], [1000.0])
    # Beyond 1 m the finite height of the 0.32 m element changes the flux by under 1 percent.
    far = np.broadcast_to(np.abs(GRID_Z[:, None] - 0.1683) > 1.0, coarse.shape)
    assert np.allclose(coarse[far], single[far], rtol=2e-2)


def test_near_coil_mask():
    geometry = cf.CoilGeometry(np.array(["a"]), np.array(["magnetics_F1A"]), np.array([1.7]), np.array([0.0]),
                               np.array([0.05]), np.array([0.05]), np.array([0.0]), np.array([0.0]))
    near = cf.near_coil_mask(geometry, GRID_R, GRID_Z)
    iz, ir = np.argmin(np.abs(GRID_Z)), np.argmin(np.abs(GRID_R - 1.7))
    assert near[iz, ir] and near[iz + 1, ir] and not near[iz + 3, ir] and not near[iz, ir + 3]
    assert near.sum() < 20
    assert cf.near_coil_mask(geometry, GRID_R, GRID_Z, margin_cells=0.0).sum() <= 4


# -------------------------------------------------------------------------- Green's table
@pytest.fixture
def d3d_geometry(tmp_path):
    shot = io.load_shot(write_shot(tmp_path / "d3d_shot_1.parquet", make_d3d_shot()), apply=False)
    return shot, cf.CoilGeometry.from_shot(shot)


def test_geometry_from_shot_and_hash_matches_qa(d3d_geometry):
    shot, geometry = d3d_geometry
    assert geometry.n_elements == 19 and len(geometry.columns) == 19
    assert geometry.columns[-1] == "magnetics_ECOILA" and geometry.elements_per_column()["magnetics_F1A"] == 1
    assert geometry.check_against_machine(DIII_D, shot.file_columns) == []
    assert geometry.hash() == check_coil_geometry(shot)["coil_hash"]
    assert geometry.check_against_machine(MAST) != []


def test_green_table_equals_direct_sum_and_round_trips(d3d_geometry, tmp_path):
    _, geometry = d3d_geometry
    table = cf.build_green_table(DIII_D, geometry, GRID_R, GRID_Z, n_filaments=2)
    assert table.patterns.shape == (19, 65, 65) and table.conventions["magnetics_F1A"] == "shared"
    currents = {c: v for c, v in zip(table.columns, np.linspace(-2.0, 3.0, 19))}
    direct = cf.direct_coil_flux(DIII_D, geometry, currents, GRID_R, GRID_Z, n_filaments=2)
    from_table = table.flux(np.array([currents[c] for c in table.columns]))
    assert np.allclose(from_table, direct, rtol=1e-10, atol=1e-14)
    batch = table.flux(np.stack([np.array([currents[c] for c in table.columns])] * 3))
    assert batch.shape == (3, 65, 65) and np.allclose(batch[1], from_table)
    scaled = table.flux(np.array([currents[c] for c in table.columns]), scales=np.full(19, 2.0))
    assert np.allclose(scaled, 2.0 * from_table)
    path = tmp_path / "green.npz"
    table.save(path)
    loaded = cf.GreenTable.load(path)
    assert loaded.columns == table.columns and np.array_equal(loaded.patterns, table.patterns)
    assert loaded.geometry_hash == geometry.hash() and loaded.n_filaments == 2
    assert loaded.column_index(["magnetics_ECOILA", "magnetics_F1A"]).tolist() == [18, 0]
    with pytest.raises(ValueError):
        table.flux(np.zeros(5))


def test_shared_and_per_element_differ_by_element_count():
    geometry = cf.CoilGeometry(np.array([f"e{i}" for i in range(4)]), np.array(["magnetics_x"] * 4),
                               np.array([0.5, 0.5, 0.5, 0.5]), np.array([-0.3, -0.1, 0.1, 0.3]),
                               np.full(4, 0.02), np.full(4, 0.1), np.zeros(4), np.zeros(4))
    shared_cfg = replace(MAST, shaping_coil_columns=("magnetics_x",))
    per_element_cfg = replace(MAST, solenoid_column="magnetics_x", solenoid_turns="per_element")
    shared = cf.column_pattern(shared_cfg, geometry, "magnetics_x", GRID_R, GRID_Z, 2)
    per_element = cf.column_pattern(per_element_cfg, geometry, "magnetics_x", GRID_R, GRID_Z, 2)
    assert np.allclose(per_element, 4.0 * shared, rtol=1e-12)
    assert cf.element_current_fraction(shared_cfg, "magnetics_x", 4) == 0.25
    with pytest.raises(KeyError):
        cf.element_current_fraction(MAST, "magnetics_tf_current", 1)


def test_symmetric_coil_set_gives_symmetric_flux():
    geometry = cf.CoilGeometry(np.array(["A", "B"]), np.array(["magnetics_F5A", "magnetics_F5B"]),
                               np.array([1.0, 1.0]), np.array([1.5, -1.5]), np.array([0.14, 0.14]),
                               np.array([0.12, 0.12]), np.array([45.0, -45.0]), np.zeros(2))
    table = cf.build_green_table(DIII_D, geometry, GRID_R, GRID_Z, n_filaments=3)
    flux = table.flux(np.array([1.0, 1.0]))
    assert np.allclose(flux, flux[::-1, :], rtol=1e-12, atol=1e-15)
    assert np.allclose(table.patterns[0], table.patterns[1][::-1, :], rtol=1e-12, atol=1e-15)


def test_currents_at_frames_matches_frame_table_mean(tmp_path, starter):
    path = write_shot(tmp_path / "d3d_shot_1.parquet", make_d3d_shot())
    shot = io.load_shot(path, ft.columns_to_load(DIII_D, True, False))
    frames = ft.build_shot_frames(shot, "diii_d_train", ft.BuildParams(psi_summary=False), True)
    currents = cf.currents_at_frames(shot, ["magnetics_plasma_current", "magnetics_ECOILA", "missing"])
    assert currents.shape == (12, 3)
    assert np.allclose(currents[:, 0], frames["ip__mean"].to_numpy(dtype=np.float64), rtol=1e-6)
    assert np.allclose(currents[:, 1], frames["sol__mean"].to_numpy(dtype=np.float64), rtol=1e-6)
    assert np.isnan(currents[:, 2]).all()
