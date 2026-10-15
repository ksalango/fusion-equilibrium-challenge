"""Tests of ``fusion_eq.plasma_models``: the warm-start ridge recovers a planted linear relation,
the zero-initialized MLP preserves it at step 0, output shape and the standardization round trip,
the bounded variant saturates, and a checkpoint reproduces the outputs exactly."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fusion_eq import loss as ls, plasma_models as pm
from tests.model_fixtures import parity_basis, symmetric_grid


def planted(n: int = 4000, f: int = 29, k: int = 6, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, f))
    a = rng.normal(size=(f, k))
    b = rng.normal(size=k)
    y = x @ a + b + 1e-3 * rng.normal(size=(n, k))
    return x, y, a, b


def test_warm_start_recovers_the_planted_relation_and_the_mlp_preserves_it():
    x, y, a, b = planted()
    w, c = pm.warm_start_ridge(x, y, alpha=1.0)
    np.testing.assert_allclose(w, a, atol=2e-3)
    np.testing.assert_allclose(c, b, atol=2e-3)
    torch.manual_seed(0)
    model = pm.build_model({"name": "linear_mlp", "hidden": [64, 64, 64], "activation": "gelu", "dropout": 0.1, "bounded": False,
                            "bound": 3.0}, x.shape[1], y.shape[1])
    pm.load_linear(model, w, c)
    out = pm.predict(model, x.astype(np.float32))
    assert out.shape == y.shape
    np.testing.assert_allclose(out, x @ w + c, atol=2e-4)
    assert float(model.mlp(torch.as_tensor(x[:10], dtype=torch.float32)).detach().abs().max()) == 0.0


def test_standardization_round_trip_through_the_flux_operator():
    rng = np.random.default_rng(2)
    grid_r, grid_z = symmetric_grid()
    basis = parity_basis(5, (grid_z.size, grid_r.size), rng)
    sigma = np.sqrt(basis["variances"])
    op = ls.FluxOperator(basis["phi_mean"], basis["phi"], sigma, grid_r, grid_z, dtype=torch.float64)
    c_raw = rng.normal(size=(4, 5)) * sigma
    ip = np.array([1e6, -8e5, 5e5, 1.2e6])
    flux = op.plasma_flux(torch.as_tensor(c_raw / sigma), torch.as_tensor(ip)).numpy()
    expected = (ip[:, None] * (basis["phi_mean"] + c_raw @ basis["phi"])).reshape(4, *op.shape)
    np.testing.assert_allclose(flux, expected, rtol=1e-12, atol=1e-15)


def test_bounded_variant_saturates():
    torch.manual_seed(0)
    model = pm.LinearMLP(4, 3, hidden=[8], dropout=0.0, bounded=True, bound=2.0)
    with torch.no_grad():
        model.output.weight.fill_(50.0)
        model.output.bias.fill_(50.0)
    x = torch.randn(100, 4) * 10
    m = model.mlp(x).detach()
    assert float(m.abs().max()) <= 2.0 and float(m.abs().max()) > 1.99
    free = pm.LinearMLP(4, 3, hidden=[8], dropout=0.0, bounded=False)
    free.load_state_dict(model.state_dict())
    assert float(free.mlp(x).detach().abs().max()) > 10.0


def test_checkpoint_reproduces_outputs_exactly(tmp_path):
    torch.manual_seed(1)
    model = pm.LinearMLP(7, 5, hidden=[16, 16], dropout=0.2)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.1 * torch.randn_like(p))
    pm.save_checkpoint(tmp_path / "m.pt", model, {"columns": list("abcdefg"), "epoch": 3})
    loaded, meta = pm.load_checkpoint(tmp_path / "m.pt")
    x = np.random.default_rng(0).normal(size=(50, 7)).astype(np.float32)
    assert np.array_equal(pm.predict(model, x), pm.predict(loaded, x))
    assert meta == {"columns": list("abcdefg"), "epoch": 3} and loaded.init_args == model.init_args
    assert not loaded.training


def test_registry_and_argument_checks():
    assert {"linear_mlp", "gru_seq", "tcn_seq"} <= set(pm.MODEL_REGISTRY)
    with pytest.raises(KeyError):
        pm.build_model({"name": "transformer_seq"}, 3, 2)
    with pytest.raises(ValueError):
        pm.LinearMLP(3, 2, activation="swish2")
