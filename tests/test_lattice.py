"""Tests of the reference lattice and the conservative placement, on the two machines' grids
given explicitly (DIII-D R 0.84 to 2.54 m, Z -1.6 to 1.6 m; MAST R 0.06 to 2.0 m, Z -2 to 2 m)."""

from __future__ import annotations

import numpy as np
import pytest

from fusion_eq import lattice as lt
from fusion_eq.machines import DIII_D, MAST

D3D_R, D3D_Z = np.linspace(0.84, 2.54, 65), np.linspace(-1.6, 1.6, 65)
MAST_R, MAST_Z = np.linspace(0.06, 2.0, 65), np.linspace(-2.0, 2.0, 65)


@pytest.fixture(scope="module")
def reference() -> lt.Lattice:
    return lt.reference_lattice(D3D_R, D3D_Z)


@pytest.fixture(scope="module")
def mast(reference) -> tuple[lt.Lattice, lt.Placement]:
    native = lt.native_lattice("MAST", MAST_R, MAST_Z)
    return native, lt.placement("MAST", reference, native)


def blob(lattice: lt.Lattice, x0: float, y0: float, sx: float, sy: float) -> np.ndarray:
    """A smooth share, zero on the two outer rows and columns: the first reference column
    reaches past the MAST grid's inner edge, and native cells straddling the lattice's outer
    edges are only partly covered, so a share that keeps its total both ways stays clear of them."""
    xx, yy = np.meshgrid(lattice.x, lattice.y)
    share = np.exp(-0.5 * ((xx - x0) / sx) ** 2 - 0.5 * ((yy - y0) / sy) ** 2)
    share[:, :2] = share[:, -2:] = share[:2, :] = share[-2:, :] = 0.0
    return share


def moments(share: np.ndarray, lattice: lt.Lattice) -> tuple[float, float, float, float]:
    xx, yy = np.meshgrid(lattice.x, lattice.y)
    w = share / share.sum()
    cx, cy = (w * xx).sum(), (w * yy).sum()
    return cx, cy, np.sqrt((w * (xx - cx) ** 2).sum()), np.sqrt((w * (yy - cy) ** 2).sum())


def test_lattices_have_the_extents_of_the_specification(reference):
    assert reference.shape == (65, 65)
    assert reference.x[0] == pytest.approx(-1.239, abs=1e-3) and reference.x[-1] == pytest.approx(1.299, abs=1e-3)
    assert reference.dx == pytest.approx(0.0396, abs=1e-4) and reference.dy == pytest.approx(0.0746, abs=1e-4)
    assert reference.y[-1] == pytest.approx(2.388, abs=1e-3) and reference.y[0] == pytest.approx(-2.388, abs=1e-3)
    r, z = reference.in_metres(MAST.major_radius.value, MAST.minor_radius.value)
    assert r[0] == pytest.approx(0.045, abs=1e-3) and r[-1] == pytest.approx(1.694, abs=1e-3) and z[-1] == pytest.approx(1.552, abs=1e-3)
    native = lt.native_lattice("MAST", MAST_R, MAST_Z)
    assert native.x[0] == pytest.approx(-1.215, abs=1e-3) and native.x[-1] == pytest.approx(1.769, abs=1e-3)
    assert native.dx == pytest.approx(0.0466, abs=1e-4) and native.dy == pytest.approx(0.0962, abs=1e-4)
    assert native.dx / reference.dx == pytest.approx(1.18, abs=0.01) and native.dy / reference.dy == pytest.approx(1.29, abs=0.01)
    r, z = reference.in_metres(DIII_D.major_radius.value, DIII_D.minor_radius.value)
    assert np.allclose(r, D3D_R) and np.allclose(z, D3D_Z)
    with pytest.raises(ValueError, match="uniform"):
        lt.Lattice(np.array([0.0, 1.0, 3.0]), np.array([0.0, 1.0]))


def test_float32_grids_are_uniform_to_the_relative_tolerance():
    # The grids are stored as float32: the DIII-D R spacing varies by about 9e-6 of its median.
    r32 = np.linspace(0.84, 2.54, 65).astype(np.float32).astype(np.float64)
    z32 = np.linspace(-1.6, 1.6, 65).astype(np.float32).astype(np.float64)
    d = np.diff(r32)
    variation = np.abs(d - np.median(d)).max() / np.median(d)
    assert 1e-6 < variation < lt.UNIFORM_RTOL
    reference = lt.reference_lattice(r32, z32)
    native = lt.native_lattice("MAST", MAST_R.astype(np.float32).astype(np.float64), MAST_Z.astype(np.float32).astype(np.float64))
    place = lt.placement("MAST", reference, native)
    share = blob(reference, 0.1, 0.0, 0.3, 0.6)
    assert place.apply(share).sum() == pytest.approx(share.sum(), rel=1e-12)
    assert lt.placement("DIII-D", reference, lt.native_lattice("DIII-D", r32, z32)).identity
    # A spacing that varies by ten times the tolerance is refused.
    bent = r32.copy()
    bent[30:] += 10 * lt.UNIFORM_RTOL * np.median(d)
    with pytest.raises(ValueError, match="uniform"):
        lt.reference_lattice(bent, z32)


def test_placement_is_the_identity_on_diii_d(reference):
    place = lt.placement("DIII-D", reference, lt.native_lattice("DIII-D", D3D_R, D3D_Z))
    share = np.random.default_rng(0).random((3, 65, 65))
    assert place.identity and np.array_equal(place.apply(share), share)
    assert np.array_equal(lt.back_placement(place, reference, reference).apply(share), share)


def test_overlap_matrix_splits_cells_by_length():
    w = lt.overlap_matrix(np.array([0.0, 1.0, 2.0]), np.array([0.5, 1.5, 3.0]))
    assert np.allclose(w, [[0.5, 0.5], [0.0, 0.5]])
    # A column sums to the share of its source cell inside the target span.
    assert np.allclose(w.sum(axis=0), [0.5, 1.0])


def test_placement_on_mast_conserves_the_total_and_keeps_the_shape(reference, mast):
    native, place = mast
    assert not place.identity and place.wx.shape == (65, 65) and place.wy.shape == (65, 65)
    share = blob(reference, 0.1, 0.2, 0.25, 0.6)
    placed = place.apply(share)
    assert placed.sum() == pytest.approx(share.sum(), rel=1e-12)
    assert place.retained(share) == pytest.approx(1.0, rel=1e-12)
    # Centroid and widths in scaled coordinates within a tenth of a native cell.
    cx, cy, sx, sy = moments(share, reference)
    px, py, qx, qy = moments(placed, native)
    assert abs(px - cx) < 0.1 * native.dx and abs(py - cy) < 0.1 * native.dy
    assert abs(qx - sx) < 0.1 * native.dx and abs(qy - sy) < 0.1 * native.dy
    # Native cells outside the reference lattice receive nothing.
    x_edges, y_edges = reference.x_edges(), reference.y_edges()
    outside_x = (native.x - 0.5 * native.dx >= x_edges[-1]) | (native.x + 0.5 * native.dx <= x_edges[0])
    outside_y = (native.y - 0.5 * native.dy >= y_edges[-1]) | (native.y + 0.5 * native.dy <= y_edges[0])
    full = place.apply(np.ones(reference.shape))
    assert outside_x.any() and outside_y.any()
    assert np.all(full[:, outside_x] == 0) and np.all(full[outside_y, :] == 0)
    # The first reference column reaches past the native grid's inner edge: part of its weight is lost.
    edge = np.zeros(reference.shape)
    edge[32, 0] = 1.0
    assert 0.0 < place.retained(edge) < 1.0
    # Linear and batched.
    two = place.apply(np.stack([share, 2.0 * share]))
    assert two.shape == (2, 65, 65) and np.allclose(two[1], 2.0 * placed)
    with pytest.raises(ValueError):
        place.apply(np.zeros((33, 65)))


def test_mirror_commutes_with_placement(reference, mast):
    _, place = mast
    share = blob(reference, -0.2, 0.7, 0.3, 0.4)
    assert np.allclose(place.apply(share[::-1]), place.apply(share)[::-1], atol=1e-15)


def test_back_placement_conserves_and_coarsens(reference, mast, tmp_path):
    native, place = mast
    back = lt.back_placement(place, reference, native)
    assert back.direction == "to_reference" and back.wx.shape == (65, 65)
    share = blob(reference, 0.0, 0.0, 0.3, 0.6)
    round_trip = back.apply(place.apply(share))
    assert round_trip.sum() == pytest.approx(share.sum(), rel=1e-12)
    assert not np.allclose(round_trip, share) and np.abs(round_trip - share).max() < 0.05 * share.max()
    place.save(tmp_path / "p.npz")
    again = lt.Placement.load(tmp_path / "p.npz")
    assert np.array_equal(again.wx, place.wx) and again.machine == "MAST" and again.r0_m == 0.85 and not again.identity


def test_other_radii_move_the_placement(reference):
    nominal = lt.placement("MAST", reference, lt.native_lattice("MAST", MAST_R, MAST_Z))
    backup = lt.placement("MAST", reference, lt.native_lattice("MAST", MAST_R, MAST_Z, 0.70, 0.50), r0_m=0.70, a0_m=0.50)
    assert backup.r0_m == 0.70 and backup.a0_m == 0.50 and not np.allclose(backup.wx, nominal.wx)
    share = blob(reference, 0.0, 0.0, 0.3, 0.6)
    # Same metres of plasma: the centroid in R moves with R0 + a0 x.
    r_nominal = (nominal.apply(share).sum(axis=0) * MAST_R).sum() / share.sum()
    r_backup = (backup.apply(share).sum(axis=0) * MAST_R).sum() / share.sum()
    assert r_nominal == pytest.approx(0.85, abs=0.01) and r_backup == pytest.approx(0.70, abs=0.01)
