"""Tests of the current-distribution target: the operator inverse, the Green's boundary, the
plasma mask and the frame handling, on synthetic maps."""

from __future__ import annotations

import numpy as np
import pytest

from fusion_eq import current_target as ct, grad_shafranov as gs, targets
from fusion_eq.coil_flux import MU0, filament_flux
from tests.conftest import make_d3d_shot

GRID_R = np.linspace(0.84, 2.54, 65)
GRID_Z = np.linspace(-1.6, 1.6, 65)
MAST_R = np.linspace(0.06, 2.0, 65)
MAST_Z = np.linspace(-2.0, 2.0, 65)


def blob(grid_r, grid_z, r0=1.7, z0=0.0, peak=1e6):
    """A smooth current density in A/m^2, zero on the border."""
    rr, zz = np.meshgrid(grid_r, grid_z)
    j = peak * np.exp(-((rr - r0) / 0.3) ** 2 - ((zz - z0) / 0.5) ** 2)
    j[0, :] = j[-1, :] = j[:, 0] = j[:, -1] = 0.0
    return j


@pytest.fixture(scope="module")
def response():
    return ct.PlasmaResponse.build("DIII-D", GRID_R, GRID_Z)


def test_interior_operator_is_the_delta_star_stencil():
    a, border_terms = ct.interior_operator(GRID_R, GRID_Z)
    rng = np.random.default_rng(0)
    psi = rng.normal(size=(65, 65))
    psi[0, :] = psi[-1, :] = psi[:, 0] = psi[:, -1] = 0.0   # no border contribution
    lhs = (a @ psi[1:-1, 1:-1].ravel()).reshape(63, 63)
    assert np.allclose(lhs, gs.delta_star(psi, GRID_R, GRID_Z)[1:-1, 1:-1], rtol=1e-12, atol=1e-9)
    assert set(border_terms) == {"east", "west", "north", "south"} and all(t[0].size == 63 for t in border_terms.values())


def test_response_solves_the_operator_and_round_trips_a_smooth_map(response):
    j0 = blob(GRID_R, GRID_Z)
    psi0 = response.flux(j0)
    rr, _ = np.meshgrid(GRID_R, GRID_Z)
    ds = gs.delta_star(psi0, GRID_R, GRID_Z)[1:-1, 1:-1]
    assert np.allclose(ds, -MU0 * rr[1:-1, 1:-1] * j0[1:-1, 1:-1], rtol=1e-9, atol=1e-9 * np.abs(ds).max())
    # Delta* then the response returns the map: the current and its flux are a fixed point.
    j1 = ct.current_from_map(psi0, GRID_R, GRID_Z)
    psi1 = response.flux(j1)
    assert np.allclose(psi1, psi0, atol=1e-10 * np.abs(psi0).max())
    assert np.all(j1[0, :] == 0) and np.allclose(j1[1:-1, 1:-1], j0[1:-1, 1:-1], atol=1e-6 * j0.max())
    # Total current is preserved through the round trip, and the response is linear and batched.
    assert gs.total_current(j1, GRID_R, GRID_Z) == pytest.approx(gs.total_current(j0, GRID_R, GRID_Z), rel=1e-9)
    batch = response.flux(np.stack([j0, 2.0 * j0]))
    assert batch.shape == (2, 65, 65) and np.allclose(batch[1], 2.0 * psi0) and np.allclose(batch[0], psi0)
    with pytest.raises(ValueError):
        response.flux(np.zeros((33, 65)))


def test_single_filament_matches_the_green_function_away_from_it(response):
    j = np.zeros((65, 65))
    j[32, 32] = 1e6
    psi = response.flux(j)
    d_r, d_z = gs.grid_steps(GRID_R, GRID_Z)
    rr, zz = np.meshgrid(GRID_R, GRID_Z)
    reference = filament_flux(rr, zz, GRID_R[32], GRID_Z[32], 1e6 * d_r * d_z)
    far = np.hypot(rr - GRID_R[32], zz - GRID_Z[32]) / d_r > 5
    assert np.abs(psi[far] - reference[far]).max() < 0.01 * np.abs(reference[far]).max()
    # The border is the Green's function exactly, the integrated current the filament's current.
    assert np.allclose(psi[0, :], reference[0, :]) and np.allclose(psi[:, -1], reference[:, -1])
    assert gs.total_current(ct.current_from_map(psi, GRID_R, GRID_Z), GRID_R, GRID_Z) == pytest.approx(1e6 * d_r * d_z, rel=1e-9)


def test_response_is_symmetric_under_a_z_flip_and_builds_for_mast(response):
    j = blob(GRID_R, GRID_Z, z0=0.4)
    assert np.allclose(response.flux(j[::-1]), response.flux(j)[::-1], atol=1e-12)
    mast = ct.PlasmaResponse.build("MAST", MAST_R, MAST_Z)
    psi = mast.flux(blob(MAST_R, MAST_Z, r0=0.9))
    assert psi.shape == (65, 65) and np.isfinite(psi).all() and psi.max() > 0
    assert mast.green.shape == (256, 63 * 63) and mast.grid_hash != response.grid_hash


def test_load_or_build_caches_the_green_matrix(tmp_path):
    first = ct.PlasmaResponse.load_or_build("DIII-D", GRID_R, GRID_Z, tmp_path)
    files = list(tmp_path.glob("response_d3d_*.npz"))
    assert len(files) == 1
    second = ct.PlasmaResponse.load_or_build("DIII-D", GRID_R, GRID_Z, tmp_path)
    assert np.array_equal(first.green, second.green) and second.machine == "DIII-D"
    assert np.allclose(first.flux(blob(GRID_R, GRID_Z)), second.flux(blob(GRID_R, GRID_Z)))


def test_plasma_current_from_stored_maps_recovers_the_current_in_both_directions(response):
    j0 = blob(GRID_R, GRID_Z)
    plasma = response.flux(j0)
    i0_ka = gs.total_current(j0, GRID_R, GRID_Z) / ct.A_PER_KA
    rr, zz = np.meshgrid(GRID_R, GRID_Z)
    coil = filament_flux(rr, zz, 3.2, 0.9, 2.0e5) + 0.05          # a coil outside the grid and a flat term
    for sign in (1.0, -1.0):
        ip = sign * i0_ka
        stored = targets.to_stored(coil + sign * plasma, ip)
        j = ct.plasma_current_from_stored(stored[None], coil[None], GRID_R, GRID_Z, np.array([ip]))
        assert j.shape == (1, 65, 65)
        assert gs.total_current(j[0], GRID_R, GRID_Z) / ct.A_PER_KA == pytest.approx(ip, rel=2e-3)
        assert np.allclose(j[0][1:-1, 1:-1], sign * j0[1:-1, 1:-1], atol=2e-3 * j0.max())


def test_vacuum_remainder_is_zero_for_current_inside_the_mask_and_closes_the_bookkeeping(response):
    rr, zz = np.meshgrid(GRID_R, GRID_Z)
    region = np.hypot((rr - 1.7) / 0.5, zz / 0.8) < 1.0
    j0 = blob(GRID_R, GRID_Z) * region                              # compact support inside the mask
    plasma = response.flux(j0)
    i0_ka = gs.total_current(j0, GRID_R, GRID_Z) / ct.A_PER_KA
    coil = filament_flux(rr, zz, 3.2, 0.9, 2.0e5)
    for sign in (1.0, -1.0):
        ip = np.array([sign * i0_ka, sign * i0_ka])
        stored = targets.to_stored(np.stack([coil + sign * plasma] * 2), ip)
        inside, remainder = ct.vacuum_remainder(stored, np.stack([coil] * 2), ip, np.stack([region] * 2), response)
        assert inside.shape == remainder.shape == (2, 65, 65)
        # Target frame: the same sign-free plasma flux for either current direction.
        assert np.allclose(inside[0], plasma, atol=1e-8 * np.abs(plasma).max())
        assert np.abs(remainder).max() < 1e-8 * np.abs(plasma).max()
        rebuilt = targets.stored_from_target(inside + remainder, np.stack([coil] * 2), ip)
        assert np.allclose(rebuilt, stored, rtol=0, atol=1e-12 * np.abs(stored).max())


def test_vacuum_remainder_holds_the_flat_term_and_the_flux_of_dropped_current(response):
    rr, zz = np.meshgrid(GRID_R, GRID_Z)
    region = np.hypot((rr - 1.7) / 0.5, zz / 0.8) < 1.0
    j_in = blob(GRID_R, GRID_Z) * region
    j_out = np.zeros((65, 65))
    j_out[5, 60] = 2e6                                              # current in the vessel, outside the mask
    coil = filament_flux(rr, zz, 3.2, 0.9, 2.0e5)
    flat = 0.05
    plasma = response.flux(j_in + j_out) + flat                     # a flat term has no current
    ip = gs.total_current(j_in, GRID_R, GRID_Z) / ct.A_PER_KA
    stored = targets.to_stored(coil + plasma, ip)
    inside, remainder = ct.vacuum_remainder(stored, coil, ip, region, response)
    # Linearity: the remainder is exactly the flat term plus the flux of the dropped current.
    expected = flat + response.flux(j_out)
    assert np.allclose(remainder, expected, atol=1e-8 * np.abs(expected).max())
    assert np.allclose(inside, response.flux(j_in), atol=1e-8 * np.abs(plasma).max())
    # Without a boundary the whole target is remainder.
    _, everything = ct.vacuum_remainder(stored, coil, ip, np.zeros((65, 65), bool), response)
    assert np.allclose(everything, targets.plasma_target(stored, coil, ip), atol=1e-14)


def test_plasma_mask_and_statistics(starter):
    shot = make_d3d_shot(n_frames=2)
    psi = shot["efit_psirz"][0]
    mask = ct.plasma_mask(psi, GRID_R, GRID_Z, "DIII-D")
    iz, ir = np.argmin(np.abs(GRID_Z + 0.3)), np.argmin(np.abs(GRID_R - 1.85))
    assert mask[iz, ir] and not mask[0, 0] and not mask[-1, -1] and 50 < mask.sum() < 65 * 65 / 2
    assert not ct.plasma_mask(np.zeros((65, 65)), GRID_R, GRID_Z, "DIII-D").any()
    j = blob(GRID_R, GRID_Z, r0=1.85, z0=-0.3)
    j[5, 5] = 1e5                                                   # stray current outside the plasma
    stats = ct.mask_statistics(j, mask, GRID_R, GRID_Z, ip_ka=gs.total_current(j, GRID_R, GRID_Z) / ct.A_PER_KA)
    assert stats["boundary_found"] and stats["n_inside"] == int(mask.sum())
    assert 0 < stats["dropped_abs_share"] < 1 and stats["dropped_abs_share"] == pytest.approx(np.abs(j[~mask]).sum() / np.abs(j).sum())
    assert stats["masked_total_over_ip"] == pytest.approx(1 - stats["dropped_abs_share"], rel=1e-9)
    empty = ct.mask_statistics(j, np.zeros_like(mask), GRID_R, GRID_Z, ip_ka=np.nan)
    assert not empty["boundary_found"] and empty["dropped_abs_share"] == 1.0 and np.isnan(empty["masked_total_over_ip"])


def test_load_or_build_without_writing_builds_in_memory(tmp_path):
    built = ct.PlasmaResponse.load_or_build("DIII-D", GRID_R, GRID_Z, tmp_path / "none", write=False)
    assert not (tmp_path / "none").exists()
    cached = ct.PlasmaResponse.load_or_build("DIII-D", GRID_R, GRID_Z, tmp_path)
    assert np.array_equal(built.green, cached.green)


def test_masks_pack_and_unpack():
    rng = np.random.default_rng(0)
    masks = rng.random((3, 65, 65)) > 0.6
    packed = ct.pack_masks(masks)
    assert packed.dtype == np.uint8 and packed.shape == (3, (65 * 65 + 7) // 8)
    assert np.array_equal(ct.unpack_masks(packed, (65, 65)), masks)
    assert ct.unpack_masks(ct.pack_masks(np.zeros((0, 65, 65), dtype=bool)), (65, 65)).shape == (0, 65, 65)


def test_shot_masks_are_the_plasma_masks_of_the_strided_maps(starter, tmp_path):
    from tests.conftest import write_shot
    shot = make_d3d_shot(n_frames=5)
    folder = tmp_path / "diii_d_train"
    folder.mkdir()
    write_shot(folder / "d3d_shot_7.parquet", shot)
    psi = np.asarray(shot["efit_psirz"], dtype=np.float32)
    for stride in (1, 2):
        packed = ct.shot_masks("7", str(tmp_path), "diii_d_train", stride, GRID_R, GRID_Z)
        masks = ct.unpack_masks(packed, (GRID_Z.size, GRID_R.size))
        expected = np.stack([ct.plasma_mask(p, GRID_R, GRID_Z, "DIII-D") for p in psi[::stride]])
        assert masks.shape[0] == len(psi[::stride]) and np.array_equal(masks, expected) and masks.any()
