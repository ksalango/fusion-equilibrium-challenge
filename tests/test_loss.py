"""Tests of ``fusion_eq.loss`` on a synthetic basis with exact parity and known maps: loss
bookkeeping (Z1), mirror equivariance (Z2), the gradient identity (Z3), the frame weights and
the term registry. Everything runs in float64 on the CPU."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fusion_eq import loss as ls
from tests.model_fixtures import ellipse_masks, parity_basis, symmetric_grid
from fusion_eq import loss_inputs as li

DT = torch.float64
ALL_TERMS = {"flux": {"enabled": True, "weight": 1.0}, "grad": {"enabled": True, "weight": 1.0},
             "band": {"enabled": True, "weight": 1.0},
             "coeff": {"enabled": True, "weight": 1.0, "components": [0, 1, 3], "component_weights": [1.0, 0.5, 2.0]}}


def setup(n: int = 100, k: int = 6, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    grid_r, grid_z = symmetric_grid()
    basis = parity_basis(k, (grid_z.size, grid_r.size), rng)
    sigma = np.sqrt(basis["variances"])
    op = ls.FluxOperator(basis["phi_mean"], basis["phi"], sigma, grid_r, grid_z, dtype=DT)
    inside = ellipse_masks(n, grid_r, grid_z, rng)
    batch = {"c": torch.as_tensor(rng.normal(size=(n, k)), dtype=DT),
             "ip_a": torch.as_tensor(rng.uniform(4e5, 1.5e6, size=n) * np.where(rng.uniform(size=n) < 0.3, -1, 1), dtype=DT),
             "inside": torch.as_tensor(inside), "band": torch.as_tensor(li.band_mask(inside, 2)),
             "weight": torch.as_tensor(rng.uniform(0.5, 2.0, size=n), dtype=DT)}
    return {"rng": rng, "op": op, "batch": batch, "basis": basis, "grid_r": grid_r, "grid_z": grid_z, "k": k, "n": n}


def mirrored(batch: dict, parity) -> dict:
    p = torch.as_tensor(parity, dtype=DT)
    out = dict(batch)
    out["c"] = batch["c"] * p
    out["inside"] = batch["inside"].flip(1)
    out["band"] = batch["band"].flip(1)
    if "psi_ref" in batch:
        out["psi_ref"] = batch["psi_ref"].flip(1)
    return out


# ------------------------------------------------------------------------------------ Z1
@pytest.mark.parametrize("reference", ["basis", "exact"])
def test_z1_bookkeeping(reference):
    s = setup()
    batch = dict(s["batch"])
    if reference == "exact":
        batch["psi_ref"] = s["op"].plasma_flux(batch["c"], batch["ip_a"])
    fn = ls.LossFunction(ALL_TERMS, s["op"], reference)
    norms = fn.compute_normalization(batch)
    fn.set_normalization(norms)
    exact_terms = fn.per_frame(batch["c"], batch)
    for name, values in exact_terms.items():
        assert float(values.abs().max()) <= 1e-10 * norms[name], name
    parts = fn.normalized_means(fn.weighted_sums(torch.zeros_like(batch["c"]), batch), s["n"])
    for name, value in parts.items():
        assert float(value) == pytest.approx(1.0, abs=1e-12), name
    total, logged = fn(torch.zeros_like(batch["c"]), batch)
    assert float(total) == pytest.approx(len(ALL_TERMS), abs=1e-10)
    assert set(logged) == set(ALL_TERMS)


def test_basis_error_is_exactly_zero_for_the_true_coefficients():
    s = setup(n=10)
    err = s["op"].flux_error(s["batch"]["c"], s["batch"]["c"], s["batch"]["ip_a"])
    assert float(err.abs().max()) == 0.0


def test_normalization_is_weighted_and_rejects_zero():
    s = setup()
    fn = ls.LossFunction({"flux": {"enabled": True, "weight": 1.0}}, s["op"])
    b = s["batch"]
    per = fn.per_frame(torch.zeros_like(b["c"]), b)["flux"]
    expected = float((b["weight"] * per).sum()) / s["n"]
    assert fn.compute_normalization(b)["flux"] == pytest.approx(expected, rel=1e-14)
    with pytest.raises(ValueError):
        fn.set_normalization({"flux": 0.0})
    with pytest.raises(ValueError):
        fn.set_normalization({})


# ------------------------------------------------------------------------------------ Z2
@pytest.mark.parametrize("reference", ["basis", "exact"])
def test_z2_mirror_equivariance(reference):
    s = setup()
    batch = dict(s["batch"])
    if reference == "exact":
        batch["psi_ref"] = s["op"].plasma_flux(batch["c"], batch["ip_a"])
    fn = ls.LossFunction(ALL_TERMS, s["op"], reference)
    c_hat = torch.as_tensor(s["rng"].normal(size=batch["c"].shape), dtype=DT)
    p = torch.as_tensor(s["basis"]["parity"], dtype=DT)
    original = fn.per_frame(c_hat, batch)
    flipped = fn.per_frame(c_hat * p, mirrored(batch, s["basis"]["parity"]))
    for name in original:
        scale = float(original[name].abs().max())
        assert float((original[name] - flipped[name]).abs().max()) <= 1e-6 * scale, name


def test_mirrored_flux_is_the_flip_of_the_flux():
    s = setup(n=8)
    p = torch.as_tensor(s["basis"]["parity"], dtype=DT)
    b = s["batch"]
    a = s["op"].plasma_flux(b["c"], b["ip_a"])
    m = s["op"].plasma_flux(b["c"] * p, b["ip_a"])
    assert float((a.flip(1) - m).abs().max()) <= 1e-12 * float(a.abs().max())


# ------------------------------------------------------------------------------------ Z3
def test_z3_gradient_identity():
    """The gradient of the plasma-flux error equals that of the full-map error (coil flux and
    remainder common to both sides, stored frame k sign(Ip)), and the terms agree."""
    s = setup(n=100)
    b = s["batch"]
    rng = s["rng"]
    shape = s["op"].shape
    c_hat = torch.as_tensor(rng.normal(size=b["c"].shape), dtype=DT)
    coil = torch.as_tensor(rng.normal(size=(s["n"], *shape)), dtype=DT)
    remainder = torch.as_tensor(0.01 * rng.normal(size=(s["n"], *shape)), dtype=DT)
    k_sign = -torch.sign(b["ip_a"])[:, None, None]
    full_hat = k_sign * (coil + s["op"].plasma_flux(c_hat, b["ip_a"]) + remainder)
    full_true = k_sign * (coil + s["op"].plasma_flux(b["c"], b["ip_a"]) + remainder)
    plasma_err = s["op"].flux_error(c_hat, b["c"], b["ip_a"])
    full_err = k_sign * (full_hat - full_true)
    gr_p, gz_p = ls.central_gradient(plasma_err, s["op"].d_r, s["op"].d_z)
    gr_f, gz_f = ls.central_gradient(full_err, s["op"].d_r, s["op"].d_z)
    scale = float(torch.maximum(gr_p.abs().max(), gz_p.abs().max()))
    assert float((gr_p - gr_f).abs().max()) <= 1e-10 * scale
    assert float((gz_p - gz_f).abs().max()) <= 1e-10 * scale
    # Terms from the full stored maps (sign included) equal the terms from the plasma flux.
    for name, fn in (("flux", ls.flux_term), ("grad", ls.grad_term), ("band", ls.band_term)):
        from_plasma = fn(ls.TermContext(c_hat, b, s["op"], "basis", _error=plasma_err), {})
        from_full = fn(ls.TermContext(c_hat, b, s["op"], "basis", _error=full_hat - full_true), {})
        assert float((from_plasma - from_full).abs().max()) <= 1e-10 * float(from_plasma.abs().max()), name


def test_z3_parameter_gradient_identity():
    s = setup(n=100)
    b = s["batch"]
    rng = s["rng"]
    shape = s["op"].shape
    coil = torch.as_tensor(rng.normal(size=(s["n"], *shape)), dtype=DT)
    start = rng.normal(size=b["c"].shape)
    grads = []
    for full in (False, True):
        c_hat = torch.as_tensor(start, dtype=DT).requires_grad_(True)
        if full:
            err = (coil + s["op"].plasma_flux(c_hat, b["ip_a"])) - (coil + s["op"].plasma_flux(b["c"], b["ip_a"]))
        else:
            err = s["op"].flux_error(c_hat, b["c"], b["ip_a"])
        ctx = ls.TermContext(c_hat, b, s["op"], "basis", _error=err)
        value = (ls.flux_term(ctx, {}) + ls.grad_term(ctx, {})).sum()
        value.backward()
        grads.append(c_hat.grad.clone())
    assert float((grads[0] - grads[1]).abs().max()) <= 1e-10 * float(grads[0].abs().max())


# ----------------------------------------------------------------------------- weights
def test_frame_weights_reproduce_hand_values():
    ip = np.array([1e6, -2e6, 0.5e6])
    w = ls.frame_weights(ip, [0, 1, 2], ip_med=1e6, gamma=3.0, phase_weights=[1.0, 2.0, 0.5])
    np.testing.assert_allclose(w, [1.0 * 1.0, 2.0 * 2.0, 0.5 * 0.5], rtol=1e-15)
    w = ls.frame_weights(ip, ["rising", "flat_top", "late"], ip_med=1e6, gamma=1.0, phase_weights=[1.0, 1.0, 1.0])
    np.testing.assert_allclose(w, [1.0, 0.5, 2.0], rtol=1e-15)
    np.testing.assert_array_equal(ls.frame_weights(ip, [0, 0, 0], ip_med=7.0), [1.0, 1.0, 1.0])
    assert ls.ip_median([-3.0, 1.0, 2.0, np.nan]) == 2.0
    with pytest.raises(ValueError):
        ls.phase_index(["flat"])


# ---------------------------------------------------------------------------- registry
def test_disabled_term_contributes_nothing():
    s = setup()
    b = s["batch"]
    c_hat = torch.as_tensor(s["rng"].normal(size=b["c"].shape), dtype=DT)
    terms = {"flux": {"enabled": True, "weight": 1.0}, "grad": {"enabled": False, "weight": 5.0}}
    fn = ls.LossFunction(terms, s["op"])
    assert fn.enabled == ["flux"]
    assert set(fn.per_frame(c_hat, b)) == {"flux"}
    fn.set_normalization(fn.compute_normalization(b))
    alone = ls.LossFunction({"flux": {"enabled": True, "weight": 1.0}}, s["op"], normalization=fn.normalization)
    assert float(fn(c_hat, b)[0]) == float(alone(c_hat, b)[0])


def test_term_weights_scale_the_total():
    s = setup()
    b = s["batch"]
    c_hat = torch.as_tensor(s["rng"].normal(size=b["c"].shape), dtype=DT)
    one = ls.LossFunction({"flux": {"enabled": True, "weight": 1.0}, "grad": {"enabled": True, "weight": 1.0}}, s["op"])
    one.set_normalization(one.compute_normalization(b))
    three = ls.LossFunction({"flux": {"enabled": True, "weight": 1.0}, "grad": {"enabled": True, "weight": 3.0}}, s["op"],
                            normalization=one.normalization)
    _, parts = one(c_hat, b)
    assert float(three(c_hat, b)[0]) == pytest.approx(parts["flux"] + 3.0 * parts["grad"], rel=1e-12)


def test_unimplemented_and_unknown_terms_raise():
    s = setup(n=4)
    with pytest.raises(NotImplementedError):
        ls.LossFunction({"contour": {"enabled": True, "weight": 1.0}}, s["op"])
    with pytest.raises(NotImplementedError):
        ls.LossFunction({"li_term": {"enabled": True, "weight": 1.0}}, s["op"])
    ls.LossFunction({"flux": {"enabled": True}, "contour": {"enabled": False}}, s["op"])
    with pytest.raises(KeyError):
        ls.LossFunction({"curvature": {"enabled": True}}, s["op"])


def test_coeff_term_value():
    s = setup(n=3)
    b = s["batch"]
    c_hat = b["c"].clone()
    c_hat[:, 1] += 2.0
    c_hat[:, 2] += 7.0
    value = ls.coeff_term(ls.TermContext(c_hat, b, s["op"], "basis"), {"components": [0, 1], "component_weights": [1.0, 0.5]})
    np.testing.assert_allclose(value.numpy(), 0.5 * 4.0)


def test_masked_mean_and_central_gradient_analytic():
    grid_r, grid_z = symmetric_grid()
    rr, zz = np.meshgrid(grid_r, grid_z)
    field = torch.as_tensor(np.stack([3.0 * rr - 2.0 * zz]), dtype=DT)
    g_r, g_z = ls.central_gradient(field, float(np.diff(grid_r).mean()), float(np.diff(grid_z).mean()))
    np.testing.assert_allclose(g_r.numpy(), 3.0, rtol=1e-12)
    np.testing.assert_allclose(g_z.numpy(), -2.0, rtol=1e-12)
    values = torch.ones((2, 4, 4), dtype=DT)
    mask = torch.zeros((2, 4, 4), dtype=torch.bool)
    mask[0, 1:3, 1:3] = True
    np.testing.assert_array_equal(ls.masked_mean(values * 5.0, mask).numpy(), [5.0, 0.0])
