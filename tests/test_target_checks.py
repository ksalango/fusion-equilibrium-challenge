"""Tests of the target-check calculations (Y4, Y5, Y7 to Y10) on synthetic shares and grids."""

from __future__ import annotations

import numpy as np
import pytest

from fusion_eq import current_basis as cb, current_target as ct, lattice as lt, target_checks as tc
from fusion_eq.coil_flux import CoilGeometry

D3D_R, D3D_Z = np.linspace(0.84, 2.54, 65), np.linspace(-1.6, 1.6, 65)
MAST_R, MAST_Z = np.linspace(0.06, 2.0, 65), np.linspace(-2.0, 2.0, 65)


def shares(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rr, zz = np.meshgrid(D3D_R, D3D_Z)
    out = []
    for _ in range(n):
        r0, z0 = 1.75 + rng.normal(0, 0.05), rng.normal(0, 0.05)
        s = np.exp(-((rr - r0) / 0.3) ** 2 - ((zz - z0) / 0.6) ** 2)
        s[:3, :] = s[-3:, :] = s[:, :3] = s[:, -3:] = 0.0
        out.append(s / s.sum())
    return np.stack(out)


def test_status_checks_and_quantiles():
    with pytest.raises(ValueError, match="status"):
        tc.Check("Y1", "Rows", "ok", "")
    table = tc.check_table([tc.Check("Y1", "Rows and keys", "pass", "a | b")]).splitlines()
    assert table[2] == "| Y1 | Rows and keys | pass | a / b |"
    q = tc.quantiles([1.0, 2.0, np.nan, 3.0])
    assert q["n"] == 3 and q["q50"] == 2.0


def test_boundary_extent_of_a_known_contour():
    # An ellipse centred at R 1.7 m, Z 0.1 m with half-width 0.6 m and half-height 1.2 m, sampled
    # through its four extreme points.
    t = np.linspace(0.0, 2.0 * np.pi, 401)
    contour = np.column_stack([1.7 + 0.6 * np.cos(t), 0.1 + 1.2 * np.sin(t)])
    e = tc.boundary_extent(contour)
    assert e["inner_R"] == pytest.approx(1.1, abs=1e-12) and e["outer_R"] == pytest.approx(2.3, abs=1e-12)
    assert e["width"] == pytest.approx(1.2, abs=1e-12) and e["height"] == pytest.approx(2.4, abs=1e-12)
    # Moving the contour leaves width and height unchanged.
    moved = tc.boundary_extent(contour + [0.25, -0.4])
    assert moved["width"] == pytest.approx(e["width"]) and moved["height"] == pytest.approx(e["height"])
    assert moved["inner_R"] == pytest.approx(1.35)
    with pytest.raises(ValueError, match="contour"):
        tc.boundary_extent(np.zeros((5, 3)))


def test_parity_mirror_flux_basis_and_conservation():
    s = shares(20)
    basis = cb.CurrentBasis.fit([s], n_components=8)
    p = tc.parity_errors(basis)
    assert p["component_max"] == 0.0 and p["mean_max"] == 0.0 and p["even"] + p["odd"] == 8
    m = tc.mirror_errors(basis, s)
    assert m["max_relative"] < 1e-12 and m["twice_identity"] and m["n_frames"] == 20
    response = ct.PlasmaResponse.build("DIII-D", D3D_R, D3D_Z)
    reference = lt.reference_lattice(D3D_R, D3D_Z)
    place = lt.placement("DIII-D", reference, reference)
    flux = cb.flux_basis("DIII-D", basis, response, place)
    assert tc.flux_basis_error(basis, flux, response, place, basis.transform(s)) < 1e-12
    mast_place = lt.placement("MAST", reference, lt.native_lattice("MAST", MAST_R, MAST_Z))
    balance = tc.placement_balance(mast_place, s)
    assert balance["max_imbalance"] < 1e-12 and balance["outside_share"]["q50"] < 1e-12
    # A share in the first reference column loses weight outside the MAST grid, and the balance accounts for it.
    edge = np.zeros_like(s[0])
    edge[32, 0] = 1.0
    b = tc.placement_balance(mast_place, edge)
    assert b["max_imbalance"] < 1e-12 and 0.0 < b["outside_share"]["q50"] < 1.0


def test_response_against_filaments_measures_the_discretisation_error():
    response = ct.PlasmaResponse.build("DIII-D", D3D_R, D3D_Z)
    sources = tc.interior_sources((65, 65), step=21)
    assert sources == [(1, 1), (1, 22), (1, 43), (22, 1), (22, 22), (22, 43), (43, 1), (43, 22), (43, 43)]
    m = tc.response_against_filaments(response, sources)
    assert m["n_sources"] == 9 and 0.0 < m["beyond_5"] < 2e-2 and 0.0 < m["beyond_10"] < m["beyond_5"] * 2
    assert m["flip"] < 1e-12 and 0.0 < m["symmetry"] < 1e-2
    # Symmetry is ruled between the sources outside the first three interior rings: (22, 22) to (43, 43).
    assert m["n_symmetry_sources"] == 4 and m["symmetry_rings_excluded"] == 3
    assert m["symmetry"] <= m["symmetry_all_sources"]
    whole = tc.response_against_filaments(response, sources, rings=0)
    assert whole["n_symmetry_sources"] == 9 and whole["symmetry"] == m["symmetry_all_sources"]
    assert tc.y9_status("DIII-D", {**m, "symmetry": 0.0}) and not tc.y9_status("DIII-D", {**m, "beyond_5": 0.05})
    assert not tc.y9_status("DIII-D", {**m, "symmetry": float("nan")})


def test_ring_shares_and_the_guard():
    d = tc.border_distance((7, 9))
    assert d[0].tolist() == [0] * 9 and d[3].tolist() == [0, 1, 2, 3, 3, 3, 2, 1, 0] and d.max() == 3
    s = np.zeros((3, 11, 11))
    s[0, 5, 5] = 1.0                                   # centre only
    s[1, 5, 5], s[1, 3, 5], s[1, 0, 5] = 0.7, -0.2, 0.1  # ring 3 (signed) and the border
    out = tc.ring_shares(s, rings=3)
    assert out["rings"][0] == 0.0 and out["rings"][1] == pytest.approx(0.2) and out["border"][1] == pytest.approx(0.1)
    assert np.isnan(out["rings"][2]) and out["total"][1] == pytest.approx(1.0)
    assert tc.ring_shares(s, rings=2)["rings"][1] == 0.0
    g = tc.ring_guard([tc.ring_shares(s[:1]), tc.ring_shares(s[1:])])
    assert g["n_frames"] == 2 and g["max"] == pytest.approx(0.2) and g["pooled"] == pytest.approx(0.1)
    assert g["border_max"] == pytest.approx(0.1) and g["rule"] == tc.Y9_RING_SHARE and not g["pass"]
    assert tc.ring_guard([tc.ring_shares(s[0])])["pass"] and not tc.ring_guard([])["pass"]


def test_support_envelope_and_landing():
    s = np.zeros((3, 5, 5))
    s[:, 2, 2] = 0.9
    s[0, 2, 3] = 0.1
    s[1, 1, 2] = 0.1
    s[2, 2, 1] = 0.1
    sup = tc.support(s, fraction=0.99)
    assert sup[0, 2, 2] and sup[0, 2, 3] and sup.sum() == 6
    assert tc.support(s, fraction=0.5).sum() == 3
    env = tc.envelope(s, fraction=0.99, min_frames=0.5)
    assert env[2, 2] and env.sum() == 1
    assert tc.envelope(s, fraction=0.99, min_frames=0.01).sum() == 4
    r, z = np.linspace(0.0, 1.0, 5), np.linspace(-1.0, 1.0, 5)
    geometry = CoilGeometry(names=np.array(["A", "B"]), input_columns=np.array(["cA", "cB"]), r=np.array([0.5, 2.0]),
                            z=np.array([0.0, 0.0]), width=np.array([0.01, 0.1]), height=np.array([0.01, 0.1]),
                            angle1=np.zeros(2), angle2=np.zeros(2))
    region = np.zeros((5, 5), dtype=bool)
    region[1:4, 1:4] = True
    mean = np.zeros((5, 5))
    mean[2, 2] = 1.0
    out = tc.landing(mean, region, r, z, geometry)
    assert out["centroid_r_m"] == 0.5 and out["centroid_z_m"] == 0.0 and out["n_cells"] == 9
    assert out["inner_r_m"] == pytest.approx(0.125) and out["outer_r_m"] == pytest.approx(0.875) and out["height_m"] == pytest.approx(1.5)
    assert out["coils_inside"] == ["cA"] and out["n_elements_inside"] == 1 and out["coil_distance_m"] == 0.0
    assert out["edge_distance_m"] == pytest.approx(0.25)
    empty = tc.landing(mean, np.zeros((5, 5), dtype=bool), r, z, geometry)
    assert np.isnan(empty["inner_r_m"]) and empty["coils_inside"] == []
