"""Tests of the current share, its symmetric basis, the flux basis, compose_map and the
target-side mirror, on synthetic maps built from a known coil flux, share and remainder."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fusion_eq import current_basis as cb, current_target as ct, grad_shafranov as gs, lattice as lt, targets
from fusion_eq.coil_flux import MU0, filament_flux

D3D_R, D3D_Z = np.linspace(0.84, 2.54, 65), np.linspace(-1.6, 1.6, 65)
MAST_R, MAST_Z = np.linspace(0.06, 2.0, 65), np.linspace(-2.0, 2.0, 65)
RR, ZZ = np.meshgrid(D3D_R, D3D_Z)


@pytest.fixture(scope="module")
def d3d_response() -> ct.PlasmaResponse:
    return ct.PlasmaResponse.build("DIII-D", D3D_R, D3D_Z)


@pytest.fixture(scope="module")
def mast_response() -> ct.PlasmaResponse:
    return ct.PlasmaResponse.build("MAST", MAST_R, MAST_Z)


@pytest.fixture(scope="module")
def reference() -> lt.Lattice:
    return lt.reference_lattice(D3D_R, D3D_Z)


def ellipse(r0=1.72, z0=0.0, a=0.45, b=0.8) -> np.ndarray:
    return ((RR - r0) / a) ** 2 + ((ZZ - z0) / b) ** 2 < 1.0


def share_of(r0=1.72, z0=0.0, peak_r=1.75, scale=1.0) -> np.ndarray:
    """A smooth share inside the ellipse, positive, summing to about one."""
    s = np.exp(-((RR - peak_r) / 0.25) ** 2 - ((ZZ - z0) / 0.45) ** 2) * ellipse(r0, z0)
    return scale * s / s.sum()


def coil_flux() -> np.ndarray:
    """Coil flux of three filaments outside the grid, Wb/rad."""
    return (filament_flux(RR, ZZ, 0.6, 1.0, 2e5) + filament_flux(RR, ZZ, 0.6, -1.0, 2e5)
            + filament_flux(RR, ZZ, 2.9, 0.4, -3e5))


def vacuum_remainder(a=0.01, b=-0.004, c=0.002) -> np.ndarray:
    """``a + b R^2 + c R^2 Z``: current-free under the discrete operator as well as the continuum."""
    return a + b * RR ** 2 + c * RR ** 2 * ZZ


def stored_map(response, share, ip_ka, coil, remainder, machine="DIII-D") -> np.ndarray:
    physical = coil + ip_ka * 1e3 * cb.cell_flux(response, share) + remainder
    return targets.to_stored(physical[None], np.array([ip_ka]), machine)[0]


# ------------------------------------------------------------------------------ share
@pytest.mark.parametrize("ip_ka", [850.0, -620.0])
def test_a_built_map_returns_its_share_its_remainder_and_itself(d3d_response, reference, ip_ka):
    share, coil, rem = share_of(), coil_flux(), vacuum_remainder()
    stored = stored_map(d3d_response, share, ip_ka, coil, rem)
    out = cb.current_share(stored, coil, ip_ka, D3D_R, D3D_Z, masks=ellipse()[None])
    assert out.share.shape == (1, 65, 65) and out.boundary_found[0]
    assert np.allclose(out.share[0], share, atol=1e-9 * share.max())
    assert out.total[0] == pytest.approx(share.sum(), rel=1e-9) and out.outside_abs_share[0] < 1e-6
    # The remainder is what the share's flux leaves of the plasma flux.
    plasma = targets.to_physical(stored, ip_ka, "DIII-D") - coil
    recovered = plasma - ip_ka * 1e3 * cb.cell_flux(d3d_response, out.share[0])
    assert np.allclose(recovered, rem, atol=1e-9 * np.ptp(stored))
    # compose_map returns the stored map from the coefficients of a basis that spans the share.
    shares = np.stack([share, share_of(peak_r=1.65), share_of(z0=0.1, peak_r=1.8), share_of(peak_r=1.9)])
    basis = cb.CurrentBasis.fit([shares], n_components=8)
    c = basis.transform(out.share)
    assert np.allclose(basis.inverse(c)[0], out.share[0], atol=1e-12)
    fb = cb.flux_basis("DIII-D", basis, d3d_response, lt.placement("DIII-D", reference, reference))
    composed = cb.compose_map("DIII-D", coil[None], [ip_ka], c, fb, remainder=recovered[None])
    assert np.allclose(composed[0], stored, atol=1e-9 * np.ptp(stored))
    # Without the remainder the map differs by exactly the remainder in the stored frame.
    bare = cb.compose_map("DIII-D", coil[None], [ip_ka], c, fb)
    assert np.allclose(targets.to_physical(composed - bare, [ip_ka], "DIII-D")[0], recovered, atol=1e-12)


def test_reversing_every_current_with_ip_leaves_the_share_and_coefficients(d3d_response):
    share, coil, rem = share_of(), coil_flux(), vacuum_remainder()
    forward = stored_map(d3d_response, share, 900.0, coil, rem)
    reversed_ = stored_map(d3d_response, share, -900.0, -coil, -rem)
    a = cb.current_share(forward, coil, 900.0, D3D_R, D3D_Z, masks=ellipse()[None])
    b = cb.current_share(reversed_, -coil, -900.0, D3D_R, D3D_Z, masks=ellipse()[None])
    assert np.allclose(a.share, b.share, atol=1e-12 * share.max()) and np.all(a.share >= -1e-12)
    basis = cb.CurrentBasis.fit([np.stack([share, share_of(peak_r=1.6), share_of(peak_r=1.9)])], n_components=4)
    assert np.allclose(basis.transform(a.share), basis.transform(b.share), atol=1e-12)


def test_a_uniform_current_in_a_circle_gives_the_masked_current_over_ip(d3d_response):
    circle = np.hypot(RR - 1.7, ZZ) < 0.35
    j = np.where(circle, 2.0e6, 0.0)
    ip_ka = 750.0
    stored = targets.to_stored((coil_flux() + d3d_response.flux(j))[None], np.array([ip_ka]), "DIII-D")
    full = cb.current_share(stored, coil_flux(), ip_ka, D3D_R, D3D_Z, masks=circle[None])
    expected = gs.total_current(j, D3D_R, D3D_Z) / (ip_ka * 1e3)
    assert full.total[0] == pytest.approx(expected, rel=1e-9)
    inner = np.hypot(RR - 1.7, ZZ) < 0.25
    part = cb.current_share(stored, coil_flux(), ip_ka, D3D_R, D3D_Z, masks=inner[None])
    assert part.total[0] == pytest.approx(gs.total_current(j, D3D_R, D3D_Z, inner) / (ip_ka * 1e3), rel=1e-9)
    assert part.outside_abs_share[0] == pytest.approx(1.0 - inner.sum() / circle.sum(), rel=1e-6)
    # No finite Ip: the share is zero.
    none = cb.current_share(stored, coil_flux(), np.nan, D3D_R, D3D_Z, masks=circle[None])
    assert np.all(none.share == 0) and none.total[0] == 0


def test_share_with_the_scorers_boundary(starter):
    from tests.conftest import make_d3d_shot
    shot = make_d3d_shot(n_frames=3)
    psi = np.asarray(shot["efit_psirz"], dtype=np.float64)
    out = cb.current_share(psi, np.zeros_like(psi), np.full(3, 1000.0), D3D_R, D3D_Z)
    assert out.boundary_found.all() and out.mask.any(axis=(1, 2)).all()
    assert np.all(out.share[~out.mask] == 0)


# ------------------------------------------------------------------------------ basis
def random_shares(n_frames: int, shape: tuple[int, int], seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(4, *shape))
    return np.einsum("fk,kij->fij", rng.normal(size=(n_frames, 4)), base) + 0.1 * rng.normal(size=(n_frames, *shape))


@pytest.mark.parametrize("shape", [(9, 6), (8, 5)])
def test_basis_has_exact_parity_and_equals_the_pca_of_originals_and_mirrors(shape):
    shares = random_shares(60, shape)
    basis = cb.CurrentBasis.fit([shares[:25], shares[25:]], n_components=12)
    comps = basis.components.reshape(-1, *shape)
    assert all(np.array_equal(comps[i][::-1], basis.parity[i] * comps[i]) for i in range(basis.n_components))
    assert np.array_equal(basis.mean.reshape(shape)[::-1], basis.mean.reshape(shape))
    assert set(np.unique(basis.parity)) == {-1, 1}
    assert np.allclose(basis.components @ basis.components.T, np.eye(12), atol=1e-12)
    assert np.all(np.diff(basis.variances) <= 1e-12) and basis.n_frames == 60
    # The same as a PCA of the stacked originals and mirrors.
    stacked = np.concatenate([shares, shares[:, ::-1]]).reshape(120, -1)
    cov = np.cov(stacked.T, bias=True)
    values, vectors = np.linalg.eigh(cov)
    order = np.argsort(values)[::-1][:12]
    assert np.allclose(basis.variances, values[order], rtol=1e-9, atol=1e-12)
    assert np.allclose(basis.mean, stacked.mean(axis=0), atol=1e-12)
    projector = vectors[:, order] @ vectors[:, order].T
    assert np.allclose(basis.components.T @ basis.components, projector, atol=1e-8)
    assert basis.total_variance == pytest.approx(np.trace(cov), rel=1e-12)


def test_mirror_of_shares_coefficients_and_labels(tmp_path):
    shares = random_shares(30, (9, 6), seed=1)
    basis = cb.CurrentBasis.fit([shares], n_components=10)
    c = basis.transform(shares)
    flipped = basis.transform(cb.mirror_share(shares))
    assert np.allclose(flipped, cb.mirror_coeffs(c, basis.parity), atol=1e-12)
    assert np.array_equal(cb.mirror_share(cb.mirror_share(shares)), shares)
    assert np.array_equal(cb.mirror_coeffs(cb.mirror_coeffs(c, basis.parity), basis.parity), c)
    labels = pd.DataFrame({"R_axis": [1.7, 1.8], "Z_axis": [0.02, -0.05], "kappa": [1.7, 1.8], "tri_top": [0.5, 0.4],
                           "tri_bot": [0.3, 0.6], "volume": [19.0, 20.0], "li": [0.9, 1.0], "q95": [4.0, 5.0], "betaN": [2.0, 1.5]})
    m = cb.mirror_labels(labels)
    assert m["Z_axis"].tolist() == [-0.02, 0.05] and m["tri_top"].tolist() == [0.3, 0.6] and m["tri_bot"].tolist() == [0.5, 0.4]
    unchanged = ["R_axis", "kappa", "volume", "li", "q95", "betaN"]
    assert m[unchanged].equals(labels[unchanged]) and cb.mirror_labels(m).equals(labels)
    assert labels["Z_axis"].tolist() == [0.02, -0.05]
    with pytest.raises(KeyError, match="both"):
        cb.mirror_labels({"tri_top": [0.1]})
    basis.save(tmp_path / "b.npz")
    again = cb.CurrentBasis.load(tmp_path / "b.npz")
    assert np.array_equal(again.components, basis.components) and np.array_equal(again.parity, basis.parity)
    assert again.shape == (9, 6) and again.truncated(4).n_components == 4
    with pytest.raises(ValueError):
        basis.truncated(11)


# ------------------------------------------------------------------------- flux basis
def test_flux_basis_is_the_response_of_the_placed_share_on_both_machines(d3d_response, mast_response, reference, tmp_path):
    shares = np.stack([share_of(), share_of(peak_r=1.6), share_of(z0=0.2, peak_r=1.8), share_of(z0=-0.1, peak_r=1.7)])
    basis = cb.CurrentBasis.fit([shares], n_components=6)
    c = basis.transform(shares)
    ip = np.array([800.0, -500.0, 1200.0, 300.0])
    reconstructed = basis.inverse(c)
    for machine, response, native in (("DIII-D", d3d_response, reference),
                                      ("MAST", mast_response, lt.native_lattice("MAST", MAST_R, MAST_Z))):
        place = lt.placement(machine, reference, native)
        fb = cb.flux_basis(machine, basis, response, place)
        direct = ip[:, None, None] * 1e3 * cb.cell_flux(response, place.apply(reconstructed))
        assert np.allclose(fb.flux(ip, c), direct, atol=1e-10 * np.abs(direct).max())
        assert fb.mean_retained == pytest.approx(1.0, abs=1e-12) and fb.mean_border_weight < 1e-12
        fb.save(tmp_path / f"{machine}.npz")
        again = cb.MachineFluxBasis.load(tmp_path / f"{machine}.npz")
        assert np.array_equal(again.phi, fb.phi) and again.machine == machine and again.r0_m == place.r0_m
    with pytest.raises(ValueError, match="placement"):
        cb.flux_basis("MAST", basis, d3d_response, lt.placement("MAST", reference, lt.native_lattice("MAST", MAST_R, MAST_Z)))
    with pytest.raises(ValueError, match="rows"):
        fb.flux(ip[:2], c)


# --------------------------------------------------------------------------- response
@pytest.mark.parametrize("machine", ["DIII-D", "MAST"])
def test_response_is_symmetric_in_its_two_arguments_and_under_a_flip(machine, d3d_response, mast_response):
    response = d3d_response if machine == "DIII-D" else mast_response
    shape = (response.grid_z.size, response.grid_r.size)
    # Mutual flux between interior cells: symmetric in the continuum; the discrete operator with
    # the Green's border holds it to discretisation error (measured 3e-6 to 2e-4).
    for a, b in (((32, 32), (20, 50)), ((10, 10), (50, 40)), ((32, 5), (32, 60)), ((5, 30), (60, 30))):
        ea, eb = np.zeros(shape), np.zeros(shape)
        ea[a], eb[b] = 1.0, 1.0
        gab, gba = cb.cell_flux(response, ea)[b], cb.cell_flux(response, eb)[a]
        assert gab == pytest.approx(gba, rel=5e-4)
    rng = np.random.default_rng(3)
    s = np.zeros(shape)
    s[1:-1, 1:-1] = rng.random((shape[0] - 2, shape[1] - 2))
    flux = cb.cell_flux(response, s)
    assert np.allclose(cb.cell_flux(response, s[::-1]), flux[::-1], atol=1e-12 * np.abs(flux).max())


def test_flux_of_a_share_has_no_units():
    share = share_of()
    ratios = []
    for factor in (1.0, 0.5):
        r, z = factor * D3D_R, factor * D3D_Z
        response = ct.PlasmaResponse.build("DIII-D", r, z)
        r0 = factor * 1.67
        ratios.append(800e3 * cb.cell_flux(response, share) / (MU0 * r0 * 800e3))
    assert np.allclose(ratios[0], ratios[1], rtol=1e-9, atol=1e-12 * np.abs(ratios[0]).max())
