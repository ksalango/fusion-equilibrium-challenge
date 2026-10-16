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


# ------------------------------------------------------------------------ fitted scales
def scales_geometry():
    return cf.CoilGeometry(np.array(["F1A", "F1B", "E"]), np.array(["magnetics_F1A", "magnetics_F1B", "magnetics_ECOILA"]),
                           np.array([1.0, 1.0, 0.72]), np.array([1.2, -1.2, 0.0]), np.array([0.14, 0.14, 0.13]),
                           np.array([0.12, 0.12, 3.5]), np.zeros(3), np.zeros(3))


def test_coil_scales_apply_per_column_and_flat_term(tmp_path):
    table = cf.build_green_table(DIII_D, scales_geometry(), GRID_R, GRID_Z, n_filaments=2)
    currents = np.array([[100.0, -60.0, 30.0], [50.0, 50.0, -20.0]])
    default = cf.CoilScales.default("DIII-D", table.columns)
    assert default.flat_wb_per_ka == 0.0 and np.allclose(table.flux(currents, default), table.flux(currents))
    record = cf.CoilScales("DIII-D", {"magnetics_F1A": 0.5, "magnetics_F1B": 2.0, "magnetics_ECOILA": 150.0},
                           flat_wb_per_ka=0.01, source="test")
    expected = table.flux(currents, np.array([0.5, 2.0, 150.0])) + 0.01 * currents[:, 2][:, None, None]
    assert np.allclose(table.flux(currents, record), expected)
    assert np.allclose(table.flux(currents[0], record), expected[0]) and np.allclose(cf.coil_flux(table, currents, record), expected)
    # The flat term is the same at every node and proportional to the solenoid current.
    delta = table.flux(currents, record) - table.flux(currents, record.vector(table.columns))
    assert np.allclose(delta[0], 0.01 * 30.0) and np.allclose(delta[1], 0.01 * -20.0)
    record.save(tmp_path / "scales.json")
    back = cf.CoilScales.load(tmp_path / "scales.json")
    assert back == record and back.vector(table.columns).tolist() == [0.5, 2.0, 150.0]
    with pytest.raises(KeyError):
        back.vector(["magnetics_F2A"])
    with pytest.raises(ValueError):
        table.flux(currents, cf.CoilScales.default("MAST", table.columns))
    # A record without the flat field reads as zero, which leaves MAST unaffected.
    old = cf.CoilScales.from_dict({"machine": "MAST", "scales": {"magnetics_sol_current": 1.0}})
    assert old.flat_wb_per_ka == 0.0 and old.source == ""


def test_adopted_scales_are_the_default_of_coil_flux():
    table = cf.build_green_table(DIII_D, scales_geometry(), GRID_R, GRID_Z, n_filaments=2)
    adopted = cf.adopted_scales("DIII-D", table.columns)
    # The adopted DIII-D model: shaping coils as stored, the solenoid pattern left out, a flat term.
    assert adopted.machine == "DIII-D" and adopted.flat_wb_per_ka > 0.0 and "solenoid_flat" in adopted.source
    assert adopted.scales["magnetics_F1A"] == 1.0 and adopted.scales["magnetics_ECOILA"] == 0.0
    assert len(adopted.scales) == 19
    currents = np.array([[100.0, -60.0, 30.0], [50.0, 50.0, -20.0]])
    assert np.allclose(cf.coil_flux(table, currents), table.flux(currents, adopted))
    assert np.allclose(cf.coil_flux(table, currents, None), table.flux(currents))
    assert not np.allclose(cf.coil_flux(table, currents), table.flux(currents))
    with pytest.raises(KeyError):
        cf.adopted_scales("DIII-D", ["magnetics_F1A", "unknown"])
    # MAST: the adopted model is sol_half (solenoid elements at half the feed current, P-coils
    # as stored); the earlier model, unit, is the default record and leaves the flux unchanged.
    mast_columns = ["magnetics_p2l_current", "magnetics_sol_current"]
    mast = cf.adopted_scales("MAST", mast_columns)
    assert mast.name == "sol_half" and mast.vector(mast_columns).tolist() == [1.0, 0.5] and mast.flat_wb_per_ka == 0.0
    unit = cf.adopted_scales("MAST", mast_columns, "unit")
    assert unit == cf.CoilScales.default("MAST", mast_columns) and unit.flat_wb_per_ka == 0.0


# --------------------------------------------------------------------- named coil models
MAST_GRID_R = np.linspace(0.06, 2.0, 33)
MAST_GRID_Z = np.linspace(-2.0, 2.0, 33)


def mast_geometry():
    """Four single-turn solenoid elements and two P-coils (P4 lower and upper, two turns each)."""
    names = np.array(["sol_0", "sol_1", "sol_2", "sol_3", "p4_l0", "p4_l1", "p4_u0", "p4_u1"])
    columns = np.array(["magnetics_sol_current"] * 4 + ["magnetics_p4l_current"] * 2 + ["magnetics_p4u_current"] * 2)
    r = np.array([0.12, 0.12, 0.14, 0.14, 1.45, 1.50, 1.45, 1.50])
    z = np.array([-0.5, 0.5, -0.5, 0.5, -1.1, -1.1, 1.1, 1.1])
    n = names.size
    return cf.CoilGeometry(names, columns, r, z, np.full(n, 0.012), np.full(n, 0.02), np.zeros(n), np.zeros(n))


def test_a_scaled_solenoid_element_gives_the_scaled_flux():
    geometry = mast_geometry()
    table = cf.build_green_table(MAST, geometry, MAST_GRID_R, MAST_GRID_Z, n_filaments=2)
    columns = list(table.columns)
    sol = columns.index("magnetics_sol_current")
    currents = np.zeros((2, len(columns)))
    currents[:, sol] = [40.0, -25.0]
    currents[:, columns.index("magnetics_p4l_current")] = [-150.0, -90.0]
    currents[:, columns.index("magnetics_p4u_current")] = [-140.0, -95.0]
    half = cf.CoilScales("MAST", {c: (0.5 if c == "magnetics_sol_current" else 1.0) for c in columns}, name="test")
    unit = cf.coil_flux(table, currents, None)
    only_sol = np.zeros_like(currents)
    only_sol[:, sol] = currents[:, sol]
    sol_flux = cf.coil_flux(table, only_sol, None)
    # Halving the element current halves the solenoid's flux and leaves the P-coils' as they are.
    np.testing.assert_allclose(cf.coil_flux(table, currents, half), unit - 0.5 * sol_flux, rtol=1e-12, atol=1e-15)
    # The same as summing the elements with half of the feed current each.
    direct = cf.direct_coil_flux(MAST, geometry, {c: (0.5 if c == "magnetics_sol_current" else 1.0) * currents[0, j]
                                                  for j, c in enumerate(columns)}, MAST_GRID_R, MAST_GRID_Z, n_filaments=2)
    np.testing.assert_allclose(cf.coil_flux(table, currents[0], half), direct, rtol=1e-10, atol=1e-14)
    # Linear in the scale: the solenoid part at scale s is s times the unit part.
    for s in (0.25, 2.0):
        scaled = replace(half, scales={**half.scales, "magnetics_sol_current": s})
        np.testing.assert_allclose(cf.coil_flux(table, currents, scaled) - cf.coil_flux(table, currents - only_sol, None),
                                   s * sol_flux, rtol=1e-12, atol=1e-15)


def test_mast_coil_models_are_both_loadable_and_named():
    columns = [*MAST.shaping_coil_columns, MAST.solenoid_column]
    assert cf.coil_model_names("MAST") == ("sol_half", "unit") and cf.DEFAULT_COIL_MODELS["MAST"] == "sol_half"
    half = cf.coil_model("MAST", columns, "sol_half")
    assert half.name == "sol_half" and half.machine == "MAST" and half.flat_wb_per_ka == 0.0
    assert half.scales[MAST.solenoid_column] == 0.5 and all(half.scales[c] == 1.0 for c in MAST.shaping_coil_columns)
    assert set(half.scales) == set(columns) and "two-in-hand" in half.source
    assert cf.coil_model("MAST", columns) == half == cf.adopted_scales("MAST", columns)
    unit = cf.coil_model("MAST", columns, "unit")
    assert unit == cf.CoilScales.default("MAST", columns) and unit.name == "unit" and unit.vector(columns).tolist() == [1.0] * 11
    assert cf.coil_model("DIII-D", ["magnetics_F1A"]).name == "solenoid_flat"
    with pytest.raises(KeyError):
        cf.coil_model("MAST", columns, "solenoid_flat")
    with pytest.raises(KeyError):
        cf.coil_model("MAST", [*columns, "magnetics_unknown"], "sol_half")
    # The name survives a save and a load.
    assert cf.CoilScales.from_dict(half.to_dict()) == half and half.to_dict()["name"] == "sol_half"


def test_coil_model_name_of_record_identifies_named_and_legacy_records():
    columns = [*MAST.shaping_coil_columns, MAST.solenoid_column]
    for name in ("sol_half", "unit"):
        assert cf.coil_model_name_of_record("MAST", cf.coil_model("MAST", columns, name).to_dict()) == name
    # A record written before the models were named (the version-2 feature build): content only.
    legacy = {"machine": "MAST", "scales": {c: 1.0 for c in columns}, "flat_wb_per_ka": 0.0, "source": "default"}
    assert cf.coil_model_name_of_record("MAST", legacy) == "unit"
    legacy_half = {**legacy, "scales": {**legacy["scales"], MAST.solenoid_column: 0.5}, "source": "x"}
    assert cf.coil_model_name_of_record("MAST", legacy_half) == "sol_half"
    d3d = cf.coil_model("DIII-D", ["magnetics_F1A", "magnetics_ECOILA"]).to_dict()
    assert cf.coil_model_name_of_record("DIII-D", {**d3d, "name": ""}) == "solenoid_flat"
    # A name whose content differs, an unknown content and another machine are refused.
    with pytest.raises(ValueError):
        cf.coil_model_name_of_record("MAST", {**legacy, "name": "sol_half"})
    with pytest.raises(ValueError):
        cf.coil_model_name_of_record("MAST", {**legacy, "scales": {**legacy["scales"], MAST.solenoid_column: 0.7}})
    with pytest.raises(ValueError):
        cf.coil_model_name_of_record("DIII-D", legacy)
