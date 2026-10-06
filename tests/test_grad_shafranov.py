"""Tests for the Grad-Shafranov operators."""

from __future__ import annotations

import numpy as np
import pytest

from fusion_eq import coil_flux as cf, grad_shafranov as gs

GRID_R = np.linspace(0.84, 2.54, 65)
GRID_Z = np.linspace(-1.6, 1.6, 65)
RR, ZZ = np.meshgrid(GRID_R, GRID_Z)
INTERIOR = (slice(1, -1), slice(1, -1))


def test_grid_steps_and_rejects_non_uniform():
    d_r, d_z = gs.grid_steps(GRID_R, GRID_Z)
    assert (d_r, d_z) == pytest.approx((1.7 / 64, 3.2 / 64))
    with pytest.raises(ValueError):
        gs.grid_steps(np.array([0.0, 1.0, 3.0]), GRID_Z)


@pytest.mark.parametrize("psi, expected", [
    (RR ** 2, 0.0 * RR),
    (ZZ ** 2, 2.0 + 0.0 * RR),
    (RR ** 2 * ZZ, 0.0 * RR),
    (RR ** 2 * ZZ ** 2, 2.0 * RR ** 2),
    (3.0 + 0.0 * RR, 0.0 * RR),
])
def test_delta_star_exact_on_low_order_solutions(psi, expected):
    out = gs.delta_star(psi, GRID_R, GRID_Z)
    assert np.isnan(out[0, :]).all() and np.isnan(out[:, -1]).all()
    assert np.allclose(out[INTERIOR], expected[INTERIOR], rtol=0, atol=1e-9)


def test_delta_star_second_order_on_quartic_and_batched():
    psi = RR ** 4                                 # psi_RR - psi_R / R = 12 R^2 - 4 R^2
    out = gs.delta_star(psi, GRID_R, GRID_Z)
    assert np.allclose(out[INTERIOR], 8.0 * RR[INTERIOR] ** 2, rtol=2e-3)
    batch = gs.delta_star(np.stack([psi, ZZ ** 2]), GRID_R, GRID_Z)
    assert batch.shape == (2, 65, 65) and np.allclose(batch[0], out, equal_nan=True)


def test_delta_star_of_a_filament_vanishes_away_from_it():
    psi = cf.filament_flux(RR, ZZ, 2.9, 0.3, 1000.0)       # coil outside the grid
    out = gs.delta_star(psi, GRID_R, GRID_Z)
    d_r, _ = gs.grid_steps(GRID_R, GRID_Z)
    assert np.nanmax(np.abs(out)) * d_r ** 2 / np.abs(psi).max() < 1e-3
    j = gs.current_density(psi, GRID_R, GRID_Z)
    assert abs(gs.total_current(j, GRID_R, GRID_Z)) < 1e-3 * 1000.0


def synthetic_plasma(total_a=500_000.0, centre=(1.7, 0.0), radius=0.4, sub=4):
    """Uniform current in a disk, as filaments on a sub-grid; returns (psi, J, disk mask)."""
    d_r, d_z = gs.grid_steps(GRID_R, GRID_Z)
    disk = (RR - centre[0]) ** 2 + (ZZ - centre[1]) ** 2 < radius ** 2
    j = np.where(disk, total_a / (disk.sum() * d_r * d_z), 0.0)
    offsets_r = (np.arange(sub) + 0.5) / sub * d_r - d_r / 2
    offsets_z = (np.arange(sub) + 0.5) / sub * d_z - d_z / 2
    rc = (RR[disk][:, None, None] + offsets_r[None, :, None] + 0 * offsets_z[None, None, :]).ravel()
    zc = (ZZ[disk][:, None, None] + 0 * offsets_r[None, :, None] + offsets_z[None, None, :]).ravel()
    current = np.full(rc.size, total_a / rc.size)
    psi = cf.filament_flux_on_grid(GRID_R, GRID_Z, rc, zc, current, cf.GMD_FACTOR * (d_r + d_z) / sub)
    return psi, j, disk


def test_integrated_current_density_recovers_the_plasma_current():
    psi, _, disk = synthetic_plasma()
    j = gs.current_density(psi, GRID_R, GRID_Z)
    region = (RR - 1.7) ** 2 + ZZ ** 2 < (0.4 + 3 * 0.05) ** 2
    assert gs.total_current(j, GRID_R, GRID_Z, region) == pytest.approx(500_000.0, rel=0.02)
    assert np.nanmean(j[disk]) > 0


def test_flux_from_current_reproduces_a_filament_plasma_outside_it():
    d_r, d_z = gs.grid_steps(GRID_R, GRID_Z)
    disk = (RR - 1.7) ** 2 + ZZ ** 2 < 0.3 ** 2
    current_per_cell = 2000.0
    j = np.where(disk, current_per_cell / (d_r * d_z), 0.0)
    expected = cf.filament_flux_on_grid(GRID_R, GRID_Z, RR[disk], ZZ[disk], np.full(disk.sum(), current_per_cell))
    got = gs.flux_from_current(j, GRID_R, GRID_Z, disk)
    assert np.isnan(got[disk]).all()
    assert np.allclose(got[~disk], expected[~disk], rtol=1e-10)
    only = np.zeros_like(disk)
    only[0, 0] = True
    got_one = gs.flux_from_current(j, GRID_R, GRID_Z, disk, target_mask=only)
    assert np.isfinite(got_one[0, 0]) and np.isnan(got_one).sum() == 65 * 65 - 1
    assert np.all(gs.flux_from_current(np.zeros_like(j), GRID_R, GRID_Z, np.zeros_like(disk))[~disk] == 0.0)


def test_vertical_field_of_quadratic_flux():
    bz = gs.vertical_field(0.5 * RR ** 2, GRID_R, GRID_Z)
    assert np.allclose(bz, 1.0, rtol=1e-9)


def test_boundary_mask_counts_interior_nodes():
    # Corners off the grid nodes, so no node lies exactly on an edge.
    square = np.array([[1.21, -0.52], [2.21, -0.52], [2.21, 0.52], [1.21, 0.52]])
    inside = gs.boundary_mask(square, GRID_R, GRID_Z)
    expected = (RR > 1.21) & (RR < 2.21) & (ZZ > -0.52) & (ZZ < 0.52)
    assert inside.sum() == expected.sum() and (inside == expected).all()
    assert not gs.boundary_mask(None, GRID_R, GRID_Z).any()
