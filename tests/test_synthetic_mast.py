"""Tests of the synthetic MAST equilibria and their ceiling on synthetic inputs: the FreeGS
machine against the coil model, the target design and its constraints, the native-grid maps and
records, the projections onto a basis, the chain on a planted share and the MAST scorer."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from fusion_eq import current_basis as cb, current_target as ct, lattice as lt, starter_kit, synthetic_mast as sm, targets
from fusion_eq.coil_flux import CoilGeometry, CoilScales, build_green_table, filament_flux, near_coil_mask
from fusion_eq.machines import machine_config

freegs = pytest.importorskip("freegs")

MAST = machine_config("MAST")
MAST_R, MAST_Z = np.linspace(0.06, 2.0, 65), np.linspace(-2.0, 2.0, 65)
D3D_R, D3D_Z = np.linspace(0.84, 2.54, 65), np.linspace(-1.6, 1.6, 65)
SOL = "magnetics_sol_current"
needs_scorer = pytest.mark.skipif(not starter_kit.is_available(), reason="starter kit not found")


def small_geometry() -> CoilGeometry:
    """Seven MAST columns: P2 and P4 pairs of two elements, P5 pair of one, a solenoid of four."""
    rows = []
    for name, r, z in (("p2l", 0.49, -1.75), ("p2u", 0.49, 1.75), ("p4l", 1.50, -1.10), ("p4u", 1.50, 1.10)):
        rows += [(f"magnetics_{name}_current", r - 0.02, z), (f"magnetics_{name}_current", r + 0.02, z)]
    rows += [("magnetics_p5l_current", 1.65, -0.5), ("magnetics_p5u_current", 1.65, 0.5)]
    rows += [(SOL, 0.14, z) for z in (-0.9, -0.3, 0.3, 0.9)]
    n = len(rows)
    return CoilGeometry(names=np.array([f"e{k}" for k in range(n)]), input_columns=np.array([c for c, _, _ in rows]),
                        r=np.array([r for _, r, _ in rows]), z=np.array([z for _, _, z in rows]),
                        width=np.full(n, 0.024), height=np.full(n, 0.037), angle1=np.zeros(n), angle2=np.zeros(n))


def sol_half(geometry: CoilGeometry) -> CoilScales:
    return CoilScales("MAST", {c: (0.5 if c == SOL else 1.0) for c in geometry.columns}, name="sol_half")


@pytest.fixture(scope="module")
def geometry() -> CoilGeometry:
    return small_geometry()


@pytest.fixture(scope="module")
def machine(geometry) -> sm.FreegsMachine:
    return sm.freegs_machine(geometry, sol_half(geometry))


def target(topology="DN", **kw) -> sm.EquilibriumTarget:
    base = dict(index=0, topology=topology, ip_ka=700.0, tf_ka=84.0, sol_ka=-20.0, kappa=1.9, delta=0.4, r_geo_m=0.86,
                minor_radius_m=0.56, beta_p=0.3, alpha_n=1.0, upper_gap_m=0.0 if topology == "DN" else 0.05)
    base.update(kw)
    return sm.EquilibriumTarget(**base)


# ---------------------------------------------------------------------------------- machine
def test_freegs_vacuum_flux_matches_the_coil_model(geometry, machine):
    currents = {c: v for c, v in zip(geometry.columns, (120.0, 95.0, -140.0, -150.0, -90.0, -85.0, -30.0))}
    machine.set_currents_ka(currents)
    rr, zz = np.meshgrid(MAST_R, MAST_Z)
    freegs_psi = machine.machine.psi(rr, zz)
    table = build_green_table("MAST", geometry, MAST_R, MAST_Z)
    ours = table.flux(np.array([currents[c] for c in table.columns]), sol_half(geometry))
    far = ~near_coil_mask(geometry, MAST_R, MAST_Z, margin_cells=2.0)
    span = ours.max() - ours.min()
    assert np.abs(freegs_psi - ours)[far].max() < 1e-5 * span
    # The column patterns reproduce the machine's flux at the same currents.
    patterns = machine.column_patterns(MAST_R, MAST_Z)
    combined = np.tensordot(np.array([currents[c] for c in machine.columns]), patterns, axes=1)
    np.testing.assert_allclose(combined, freegs_psi, rtol=0, atol=1e-12 * span)


def test_currents_convert_under_the_coil_model(geometry, machine):
    machine.set_currents_ka({"magnetics_p2l_current": 100.0, SOL: -30.0})
    # P2L: 100 kA-turn shared over two elements; solenoid: half of -30 kA in every element.
    assert machine.machine["magnetics_p2l_current"].current == pytest.approx(50e3)
    assert machine.machine[SOL].current == pytest.approx(-15e3)
    back = machine.currents_ka()
    assert back["magnetics_p2l_current"] == pytest.approx(100.0)
    assert back[SOL] == pytest.approx(-30.0)


def test_column_coil_equals_a_freegs_point_coil():
    from freegs.coil import Coil

    cls = sm.column_coil_class()
    ours = cls([0.9], [0.4], [1.0], n_elements=1)
    ref = Coil(0.9, 0.4)
    r, z = np.array([0.3, 1.2, 1.7]), np.array([-0.5, 0.1, 0.9])
    np.testing.assert_allclose(ours.controlPsi(r, z), ref.controlPsi(r, z), rtol=1e-12)
    np.testing.assert_allclose(ours.controlBr(r, z), ref.controlBr(r, z), rtol=1e-9)
    np.testing.assert_allclose(ours.controlBz(r, z), ref.controlBz(r, z), rtol=1e-9)
    assert isinstance(ours.controlPsi(1.2, 0.1), float)


def test_whole_grid_evaluations_are_cached_and_read_only():
    cls = sm.column_coil_class()
    coil = cls([0.9, 0.95], [0.4, 0.45], [0.5, 0.5], n_elements=1)
    rr, zz = np.meshgrid(MAST_R, MAST_Z)
    first = coil.controlBr(rr, zz)
    second = coil.controlBr(rr.copy(), zz.copy())
    assert second is first and not first.flags.writeable
    other = coil.controlBr(rr + 0.01, zz)
    assert other is not first and not np.allclose(other, first)


def test_fixed_columns_and_current_limits(geometry):
    fm = sm.freegs_machine(geometry, sol_half(geometry))
    control = {label: c.control for label, c in fm.machine.coils}
    assert control[SOL] is False and all(control[c] for c in geometry.columns if c != SOL)
    limits = sm.current_limits_a(fm, {"magnetics_p4l_current": (-288.0, -1.0)})
    by_column = dict(zip(fm.columns, limits))
    # Two elements share P4L's kA-turn: per element -144 kA to -0.5 kA.
    assert by_column["magnetics_p4l_current"] == pytest.approx((-144e3, -0.5e3))
    assert by_column[SOL] == (-np.inf, np.inf)
    assert sm.current_limits_a(fm, None) is None


def test_freegs_machine_rejects_a_coil_model_without_a_column(geometry):
    scales = CoilScales("MAST", {c: 1.0 for c in geometry.columns if c != SOL})
    with pytest.raises(KeyError):
        sm.freegs_machine(geometry, scales)


# ---------------------------------------------------------------------------------- targets
def test_target_design_is_deterministic_and_inside_the_ranges():
    a, b = sm.target_design(16, seed=3), sm.target_design(16, seed=3)
    assert a == b and a != sm.target_design(16, seed=4)
    assert [t.index for t in a] == list(range(32))
    assert sum(t.topology == "DN" for t in a) == 16
    for t in a:
        for key, (lo, hi) in sm.DEFAULT_RANGES.items():
            if key in ("r_geo_m", "upper_gap_m"):
                continue
            assert lo <= getattr(t, key) <= hi
        assert (t.upper_gap_m == 0.0) == (t.topology == "DN")
        assert t.r_geo_m - t.minor_radius_m >= sm.MIN_INNER_EDGE_M - 1e-12
        assert t.r_geo_m + t.minor_radius_m <= sm.MAX_OUTER_EDGE_M + 1e-12
    restored = [sm.EquilibriumTarget.from_dict(t.to_dict()) for t in a]
    assert restored == a


def test_targets_reject_inconsistent_topologies():
    with pytest.raises(ValueError):
        target("DN", upper_gap_m=0.05)
    with pytest.raises(ValueError):
        target("LSN", upper_gap_m=0.0)
    with pytest.raises(ValueError):
        target(r_geo_m=0.5, minor_radius_m=0.6)


def test_miller_boundary_has_the_target_shape():
    t = target(kappa=2.0, delta=0.35)
    r, z = sm.miller_boundary(t, np.linspace(0, 2 * np.pi, 20001))
    width = r.max() - r.min()
    assert (z.max() - z.min()) / width == pytest.approx(2.0, rel=1e-6)
    assert (r.max() + r.min()) / 2 == pytest.approx(t.r_geo_m, abs=1e-9)
    assert (t.r_geo_m - r[np.argmax(z)]) / (width / 2) == pytest.approx(0.35, abs=1e-6)


def test_double_null_constraints_are_symmetric():
    xpoints, isoflux = sm.shape_constraints(target("DN"))
    (rl, zl), (ru, zu) = xpoints
    assert rl == ru and zl == -zu < 0
    tied = {(p[0], p[1]) for p in isoflux}
    assert (rl, zl) in tied and (ru, zu) in tied
    # Every pair ties a point to the outer midplane.
    assert all((p[2], p[3]) == (0.86 + 0.56, 0.0) for p in isoflux)


def test_lower_single_null_leaves_the_upper_xpoint_untied():
    t = target("LSN", upper_gap_m=0.07)
    xpoints, isoflux = sm.shape_constraints(t)
    (rl, zl), (ru, zu) = xpoints
    assert zu == pytest.approx(-zl + 0.07)
    tied = {(p[0], p[1]) for p in isoflux}
    assert (rl, zl) in tied and (ru, zu) not in tied
    # The boundary points of the isoflux pairs lie on the target contour.
    r_b, z_b = sm.miller_boundary(t, sm.ISOFLUX_ANGLES)
    assert {(float(r), float(z)) for r, z in zip(r_b, z_b)} <= tied


def test_fvac_from_the_tf_conductor_current():
    t = target(tf_ka=-84.0)
    assert t.fvac(MAST) == pytest.approx(4e-7 * np.pi * 24 * 84e3 / (2 * np.pi))
    # Bt at R0 of order 0.5 T, MAST's field.
    assert 0.4 < t.fvac(MAST) / 0.85 < 0.6


# ---------------------------------------------------------------------------- maps, records
def test_interpolation_is_exact_on_shared_nodes_and_smooth_between():
    grid = sm.SolverGrid.refined(MAST_R, MAST_Z, 2)
    r, z = grid.axes()
    assert (grid.n_r, grid.n_z) == (129, 129)
    rr, zz = np.meshgrid(r, z, indexing="ij")
    f = np.sin(2 * rr) * np.cos(1.5 * zz) + rr ** 2
    native = sm.interpolate_to_grid(f, r, z, MAST_R, MAST_Z)
    rn, zn = np.meshgrid(MAST_R, MAST_Z)
    np.testing.assert_allclose(native, (np.sin(2 * rr) * np.cos(1.5 * zz) + rr ** 2)[::2, ::2].T, atol=1e-12)
    off_r, off_z = MAST_R[:-1] + 0.01, MAST_Z[:-1] + 0.02
    between = sm.interpolate_to_grid(f, r, z, off_r, off_z)
    ro, zo = np.meshgrid(off_r, off_z)
    np.testing.assert_allclose(between, np.sin(2 * ro) * np.cos(1.5 * zo) + ro ** 2, atol=1e-5)


def test_records_round_trip(tmp_path):
    record = {"version": np.array(1), "psi": np.arange(6.0).reshape(2, 3), "columns": np.array(["a", "b"]),
              "currents_ka": np.array([1.5, -2.0]), "ip_ka": np.array(650.0),
              "target": np.array('{"topology": "DN"}'), "profile": np.array("betap_ip"), "meta": np.array("{}")}
    for index in (12, 3):
        sm.write_record(sm.record_path(tmp_path, index), record)
    paths = sm.list_records(tmp_path)
    assert [p.name for p in paths] == ["eq_00003.npz", "eq_00012.npz"]
    assert not list(tmp_path.glob("*.tmp.npz"))
    back = sm.load_record(paths[0])
    np.testing.assert_array_equal(back["psi"], record["psi"])
    assert back["columns"] == ["a", "b"] and back["ip_ka"] == 650.0
    assert back["target"] == {"topology": "DN"} and back["profile"] == "betap_ip"


@pytest.mark.parametrize("split", ["mast_samples", "diii_d_train", "diii_d_public_test"])
def test_geometry_is_never_read_from_a_labelled_or_other_split(tmp_path, split):
    with pytest.raises(ValueError):
        sm.load_geometry(tmp_path, split)


def test_check_solution_rejects_a_wrong_current_and_a_missing_xpoint():
    r, z = np.linspace(0.1, 2.0, 65), np.linspace(-2.0, 2.0, 65)
    rr, zz = np.meshgrid(r, z, indexing="ij")
    psi = np.exp(-((rr - 0.9) / 0.4) ** 2 - (zz / 0.7) ** 2)            # an O-point, no X-point
    fake = SimpleNamespace(R=rr, Z=zz, psi=lambda: psi, plasmaCurrent=lambda: 700e3)
    assert "no X-point" in sm.check_solution(fake, target())
    fake.plasmaCurrent = lambda: 600e3
    assert "plasma current" in sm.check_solution(fake, target())


# --------------------------------------------------------------------------- projections
def smooth_shares(lattice_r, lattice_z, n, seed=0) -> np.ndarray:
    """Positive shares of compact support ``(1 - rho^2)^2`` inside an ellipse near the lattice
    centre, each summing to one; zero on the border."""
    rng = np.random.default_rng(seed)
    rr, zz = np.meshgrid(lattice_r, lattice_z)
    out = []
    for _ in range(n):
        r0, z0, w = 1.7 + 0.05 * rng.normal(), 0.05 * rng.normal(), 0.25 + 0.05 * rng.random()
        rho2 = ((rr - r0) / w) ** 2 + ((zz - z0) / (1.6 * w)) ** 2
        s = np.clip(1.0 - rho2, 0.0, None) ** 2
        out.append(s / s.sum())
    return np.stack(out)


@pytest.fixture(scope="module")
def basis() -> cb.CurrentBasis:
    return cb.CurrentBasis.fit([smooth_shares(D3D_R, D3D_Z, 200)], n_components=8)


@pytest.fixture(scope="module")
def mast_place() -> lt.Placement:
    return lt.placement("MAST", reference=lt.reference_lattice(D3D_R, D3D_Z), native=lt.native_lattice("MAST", MAST_R, MAST_Z))


def test_project_share_recovers_planted_coefficients(basis, mast_place):
    c0 = np.linspace(-0.004, 0.003, basis.n_components)
    identity = lt.placement("DIII-D", reference=lt.reference_lattice(D3D_R, D3D_Z),
                            native=lt.reference_lattice(D3D_R, D3D_Z))
    np.testing.assert_allclose(sm.project_share(basis.inverse(c0[None])[0], basis, identity), c0, atol=1e-12)
    placed = mast_place.apply(basis.inverse(c0[None])[0])
    np.testing.assert_allclose(sm.project_share(placed, basis, mast_place), c0, atol=1e-9)
    # Without placement the projection is the basis's own transform.
    s = smooth_shares(D3D_R, D3D_Z, 1, seed=9)[0]
    np.testing.assert_allclose(sm.project_share(s, basis, identity), basis.transform(s)[0], atol=1e-12)


def test_project_flux_recovers_planted_coefficients():
    rng = np.random.default_rng(1)
    fb = cb.MachineFluxBasis("MAST", rng.normal(size=4225) * 1e-7, rng.normal(size=(6, 4225)) * 1e-7, (65, 65), 0.85, 0.65)
    c0 = rng.normal(size=6)
    flux = fb.flux(720.0, c0[None])[0]
    np.testing.assert_allclose(sm.project_flux(flux, 720.0, fb), c0, rtol=1e-9)


def test_footprint_and_helpers(mast_place):
    footprint = sm.lattice_footprint(mast_place)
    assert footprint.shape == (65, 65) and footprint.any() and not footprint.all()
    assert sm.rms_over_range(np.full((3, 3), 0.1), np.array([[0.0, 2.0]])) == pytest.approx(0.05)
    labels = sm.bin_labels([1.5, 1.75, 2.5, np.nan, 1.0], (1.4, 1.7, 2.0))
    assert list(labels) == ["1.4-1.7", "1.7-2", ">=2", "missing", "<1.4"]
    square = np.array([[1.0, -1.0], [2.0, -1.0], [2.0, 1.0], [1.0, 1.0]])
    assert sm.contour_aspect_ratio(square) == pytest.approx(3.0)
    assert np.isnan(sm.contour_aspect_ratio(None))


# -------------------------------------------------------------------- the chain and scorer
@pytest.fixture(scope="module")
def planted(basis, mast_place):
    """A MAST frame whose plasma current is a share the basis holds exactly: coil flux of two
    filaments outside the grid plus Ip G P (s_mean + B^T c0)."""
    response = ct.PlasmaResponse.build("MAST", MAST_R, MAST_Z)
    fb = cb.flux_basis("MAST", basis, response, mast_place)
    c0 = np.linspace(-0.002, 0.002, basis.n_components)
    rr, zz = np.meshgrid(MAST_R, MAST_Z)
    coil = filament_flux(rr, zz, 2.6, 1.2, -2e5) + filament_flux(rr, zz, 2.6, -1.2, -2e5)
    ip_ka = 650.0
    physical = coil + fb.flux(ip_ka, c0[None])[0]
    psi = targets.to_stored(physical, ip_ka, "MAST")
    return SimpleNamespace(response=response, fb=fb, c0=c0, coil=coil, ip_ka=ip_ka, psi=psi)


@needs_scorer
def test_chain_recovers_a_planted_share(planted, basis, mast_place):
    ev = sm.evaluate_equilibrium(planted.psi, planted.coil, planted.ip_ka, MAST_R, MAST_Z, basis, planted.fb, mast_place,
                                 planted.response)
    assert ev.diagnostics["boundary_found"] == 1.0
    assert ev.diagnostics["outside_footprint_abs_share"] == pytest.approx(0.0, abs=1e-12)
    span = planted.psi.max() - planted.psi.min()
    np.testing.assert_array_equal(ev.maps["truth"], planted.psi)
    # The planted share lies inside the boundary, so the masked share and its projection are exact.
    np.testing.assert_allclose(ev.coefficients["basis_share"], planted.c0, atol=1e-6)
    np.testing.assert_allclose(ev.maps["basis_share"], planted.psi, atol=1e-6 * span)
    np.testing.assert_allclose(ev.maps["basis_flux"], planted.psi, atol=1e-8 * span)
    np.testing.assert_allclose(ev.maps["exact_share"], planted.psi, atol=1e-6 * span)
    expected_mean = cb.compose_map("MAST", planted.coil[None], np.array([planted.ip_ka]),
                                   np.zeros((1, basis.n_components)), planted.fb)[0]
    np.testing.assert_allclose(ev.maps["mean_share"], expected_mean)
    assert ev.diagnostics["share_rel_error"] < 1e-6


@needs_scorer
def test_mast_scorer_self_check(planted):
    truth = sm.synthetic_truth(planted.psi, q95=6.0, beta_n=3.0)
    other = sm.synthetic_truth(planted.psi * 1.02 + 0.001, q95=5.0, beta_n=2.5)
    truths = [truth, other]
    means = sm.means_of(truths)
    perfect = [truth.psi, other.psi]
    sign = sm.global_psi_sign(truths, perfect, means[0])
    parts = [sm.score_part(t, p, sign, means) for t, p in zip(truths, perfect)]
    terms = sm.finalize_parts(parts, [t.stats for t in truths], sign)
    assert terms["S"] == pytest.approx(1.0) and terms["consistency"] == pytest.approx(1.0)
    # The stored MAST frame has its axis at a maximum: the scorer keeps the sign.
    assert sign == 1.0
    zeros = [np.zeros_like(truth.psi), np.zeros_like(other.psi)]
    sign0 = sm.global_psi_sign(truths, zeros, means[0])
    parts0 = [sm.score_part(t, p, sign0, means) for t, p in zip(truths, zeros)]
    assert sm.finalize_parts(parts0, [t.stats for t in truths], sign0)["S"] < 0.5
    m = sm.frame_metrics(truth, truth.psi, 1.0)
    assert m["d_lcfs_m"] == 0.0 and m["d_lcfs_rel"] == 0.0
    assert all(m[f"pred_{k}"] == m[f"true_{k}"] or np.isnan(m[f"true_{k}"]) for k in ("R_axis", "kappa", "volume", "li"))
