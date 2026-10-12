"""Tests of the coil-pattern layout, clearances, point tables, point values and named terms."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fusion_eq import coil_flux as cf, coil_pattern as cp, features as fe, io
from fusion_eq.machines import DIII_D, MAST
from tests.conftest import drop_labels, make_d3d_shot, write_shot

GRID_R = np.linspace(0.84, 2.54, 65)
GRID_Z = np.linspace(-1.6, 1.6, 65)


def geometry(names, columns, r, z, w, h, angle1=None, angle2=None) -> cf.CoilGeometry:
    n = len(names)
    return cf.CoilGeometry(np.array(names), np.array(columns), np.asarray(r, float), np.asarray(z, float),
                           np.asarray(w, float), np.asarray(h, float),
                           np.zeros(n) if angle1 is None else np.asarray(angle1, float),
                           np.zeros(n) if angle2 is None else np.asarray(angle2, float))


def symmetric_geometry() -> cf.CoilGeometry:
    """A DIII-D-like set whose A and B coils mirror each other exactly in Z."""
    return geometry(["F1A", "F1B", "F2A", "F2B", "E"],
                    ["magnetics_F1A", "magnetics_F1B", "magnetics_F2A", "magnetics_F2B", "magnetics_ECOILA"],
                    [0.95, 0.95, 2.45, 2.45, 0.75], [0.45, -0.45, 0.9, -0.9, 0.0],
                    [0.06, 0.06, 0.1, 0.1, 0.05], [0.2, 0.2, 0.15, 0.15, 2.5])


# ------------------------------------------------------------------------------ layout
def test_layout_mirror_partners_and_rules():
    layout = cp.Layout(np.array([[0.5, -1.0], [0.5, 1.0], [0.5, 0.0], [-0.5, 0.0]]))
    assert layout.mirror_index().tolist() == [1, 0, 2, 3] and layout.is_symmetric()
    assert layout.column_names() == ["coil_pt_00", "coil_pt_01", "coil_pt_02", "coil_pt_03"]
    r, z = layout.physical(1.67, 0.67)
    assert np.allclose(r, [2.005, 2.005, 2.005, 1.335, 1.67]) and np.allclose(z, [-0.67, 0.67, 0.0, 0.0, 0.0])
    lopsided = cp.Layout(np.array([[0.5, 1.0], [0.5, 0.0]]))
    assert not lopsided.is_symmetric()
    with pytest.raises(ValueError):
        lopsided.mirror_index()
    with pytest.raises(ValueError):
        cp.Layout(np.array([[0.0, 0.0], [0.5, 0.0]]))           # the reference is not a sample point
    with pytest.raises(ValueError):
        cp.Layout(np.array([[0.5, 0.0], [0.5, 0.0]]))


def test_configured_layout_is_the_agreed_22_points():
    layout = fe.FeatureConfig.load().layout
    assert layout.n_points == 22 and layout.is_symmetric() and layout.min_clearance_m == 0.15
    pts = {tuple(p) for p in layout.points.tolist()}
    assert {(0.375, 1.125), (0.375, -1.125)} <= pts
    assert not pts & {(0.375, 1.5), (0.375, -1.5), (0.75, 1.5), (0.75, -1.5), (0.0, 0.0)}


# --------------------------------------------------------------------------- clearance
def test_clearance_to_rectangles_and_skewed_parallelograms():
    rect = geometry(["R"], ["magnetics_F1A"], [1.0], [0.0], [0.2], [0.2])
    dist, nearest = cp.clearances([1.0, 1.4, 1.0], [0.05, 0.0, -0.4], rect)
    assert np.allclose(dist, [0.0, 0.3, 0.3]) and nearest.tolist() == [0, 0, 0]
    # angle1 = 45 degrees shears the element: the lower edge runs from (0.9, -0.2) to (1.1, 0.0).
    skew = geometry(["S"], ["magnetics_F1A"], [1.0], [0.0], [0.2], [0.2], angle1=[45.0])
    dist, _ = cp.clearances([1.0, 1.05], [-0.15, 0.0], skew)
    assert dist[0] == pytest.approx(0.05 / np.sqrt(2)) and dist[1] == 0.0
    # Nearest of several elements.
    two = geometry(["A", "B"], ["magnetics_F1A", "magnetics_F1B"], [1.0, 2.0], [0.0, 0.0], [0.2, 0.2], [0.2, 0.2])
    dist, nearest = cp.clearances([1.7], [0.0], two)
    assert dist[0] == pytest.approx(0.2) and nearest[0] == 1


def test_clearance_table_and_layout_check_fail_on_a_close_coil():
    layout = cp.Layout(np.array([[0.75, -0.75], [0.75, 0.75], [-0.5, 0.0]]))
    far = geometry(["F1A", "F1B"], ["magnetics_F1A", "magnetics_F1B"], [0.9, 0.9], [1.4, -1.4], [0.05, 0.05], [0.1, 0.1])
    table = cp.clearance_table(layout, DIII_D, far, GRID_R, GRID_Z)
    assert table["ok"].all() and table["in_grid"].all()
    assert np.allclose(table["R_m"], [1.67 + 0.75 * 0.67, 1.67 + 0.75 * 0.67, 1.67 - 0.5 * 0.67])
    assert cp.check_layout(layout, {"DIII-D": table})["status"] == "pass"
    # A coil 0.10 m from the upper point fails it; the check names it.
    near = geometry(["F1A", "F1B"], ["magnetics_F1A", "magnetics_F1B"], [2.1725 + 0.125, 0.9], [0.5025, -1.4],
                    [0.05, 0.05], [0.1, 0.1])
    bad = cp.clearance_table(layout, DIII_D, near, GRID_R, GRID_Z)
    assert bad["ok"].tolist() == [True, False, True] and bad["clearance_m"].iloc[1] == pytest.approx(0.1)
    result = cp.check_layout(layout, {"DIII-D": table, "MAST-like": bad})
    assert result["status"] == "fail" and [f["column"] for f in result["failing"]] == ["coil_pt_01"]
    outside = cp.clearance_table(layout, DIII_D, far, GRID_R, GRID_Z, a0_m=1.5)    # pushes points off the grid
    assert not outside["in_grid"].all() and not outside["ok"].all()


def test_configured_layout_clears_the_shipped_geometry(starter):
    """Check X4 on the geometry of the starter-kit sample shots (identical in every shot, F06)."""
    shots = {m: (starter.sample_shot_paths(m) or [None])[0] for m in ("DIII-D", "MAST")}
    if any(v is None for v in shots.values()):
        pytest.skip("starter-kit sample shots not found")
    layout = fe.FeatureConfig.load().layout
    tables = {}
    for machine, cfg in (("DIII-D", DIII_D), ("MAST", MAST)):
        shot = io.load_shot(shots[machine], [*cf.GEOMETRY_COLUMNS, *cf.GRID_COLUMNS], apply=False)
        tables[machine] = cp.clearance_table(layout, cfg, cf.CoilGeometry.from_shot(shot),
                                             np.asarray(shot["efit_grid_R"], float), np.asarray(shot["efit_grid_Z"], float))
    result = cp.check_layout(layout, tables)
    assert result["status"] == "pass"
    # Values of the layout review of 12 October 2026.
    assert result["min_clearance_m"]["DIII-D"] == pytest.approx(0.279, abs=1e-3)
    assert result["min_clearance_m"]["MAST"] == pytest.approx(0.199, abs=1e-3)
    added = tables["MAST"].set_index(["x", "y"]).loc[(0.375, 1.125), "clearance_m"]
    assert added == pytest.approx(0.315, abs=1e-3)


# ------------------------------------------------------------------------- point tables
def test_point_table_matches_the_grid_table_at_grid_nodes():
    geom = symmetric_geometry()
    grid = cf.build_green_table(DIII_D, geom, GRID_R, GRID_Z, n_filaments=4)
    iz, ir = np.array([10, 32, 50, 3]), np.array([5, 31, 40, 60])
    points = cf.build_point_table(DIII_D, geom, GRID_R[ir], GRID_Z[iz], n_filaments=4)
    assert points.columns == grid.columns and points.patterns.shape == (5, 4)
    assert np.allclose(points.patterns, grid.patterns[:, iz, ir], rtol=1e-13, atol=0)
    currents = np.array([[120.0, -40.0, 300.0, 280.0, -25.0], [0.0, 10.0, -5.0, 5.0, 30.0]])
    scales = cf.CoilScales("DIII-D", {c: s for c, s in zip(grid.columns, (1.1, 0.9, 1.0, 1.05, 1.2))}, flat_wb_per_ka=0.004)
    assert np.allclose(points.flux(currents, scales), grid.flux(currents, scales)[:, iz, ir], rtol=1e-13)
    assert points.flux(currents[0]).shape == (4,)


def test_point_table_cache_reads_geometry_from_the_public_test_folder(tmp_path):
    data_root = tmp_path / "data"
    folder = data_root / "diii_d_public_test"
    folder.mkdir(parents=True)
    shot = drop_labels(make_d3d_shot(n_frames=2))
    write_shot(folder / "d3d_shot_b2.parquet", shot)
    write_shot(folder / "d3d_shot_a1.parquet", shot)
    geom = cf.CoilGeometry.from_shot(io.load_shot(folder / "d3d_shot_a1.parquet", list(cf.GEOMETRY_COLUMNS), apply=False))
    grid = cf.build_green_table(DIII_D, geom, GRID_R, GRID_Z, n_filaments=2)
    layout = cp.Layout(np.array([[0.5, -0.5], [0.5, 0.5], [-0.5, 0.0]]))
    table, record = cp.load_or_build_point_table(layout, DIII_D, grid, data_root, tmp_path / "cache")
    assert record["built"] and record["geometry_source"].endswith("d3d_shot_a1.parquet")
    assert table.geometry_hash == grid.geometry_hash and table.n_filaments == 2 and table.patterns.shape == (19, 4)
    again, record2 = cp.load_or_build_point_table(layout, DIII_D, grid, data_root, tmp_path / "cache")
    assert not record2["built"] and np.array_equal(again.patterns, table.patterns) and record2["path"] == record["path"]
    # The reference point is last and every point sits at R0 + a0 x, a0 y.
    assert np.allclose(table.points_r, [2.005, 2.005, 1.335, 1.67]) and np.allclose(table.points_z, [-0.335, 0.335, 0.0, 0.0])
    other = cf.build_green_table(DIII_D, symmetric_geometry(), GRID_R, GRID_Z, n_filaments=2)
    with pytest.raises(ValueError):
        cp.load_or_build_point_table(layout, DIII_D, other, data_root, tmp_path / "cache2")


# ------------------------------------------------------------------------ point values
def test_point_values_are_relative_to_the_reference_and_per_unit_ip():
    table = cf.PointTable("DIII-D", ("magnetics_ECOILA",), np.array([[2.0, 5.0, 1.0]]), np.zeros(3), np.zeros(3), 1, "g", "p")
    values = cp.coil_point_values(table, np.array([[3.0], [3.0], [3.0], [3.0]]), np.array([500.0, -500.0, 10.0, np.nan]),
                                  r0_m=1.67, min_abs_ip_ka=20.0)
    expected = (np.array([2.0, 5.0]) - 1.0) * 3.0 / (cf.MU0 * 1.67 * 500.0e3)
    assert np.allclose(values[0], expected) and np.allclose(values[1], -expected)
    assert np.isnan(values[2:]).all()                                        # Ip too small, Ip missing


def test_named_terms_recover_an_exact_cubic():
    layout = fe.FeatureConfig.load().layout
    coefficients = np.array([0.8, -0.3, 0.25, 0.1, -0.4, 0.05, -0.07, 0.02, 0.09])
    values = cp.term_matrix(layout.points) @ coefficients
    assert np.allclose(cp.term_operator(layout) @ values, coefficients, atol=1e-12)
    with pytest.raises(ValueError):
        cp.term_operator(cp.Layout(np.array([[0.5, 0.5], [0.5, -0.5], [-0.5, 0.0]])))
