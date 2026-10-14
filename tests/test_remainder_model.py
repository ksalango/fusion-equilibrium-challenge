"""Tests of the remainder model (basis and ridges per rate window) and of the target
configuration, on synthetic frame rows and maps."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fusion_eq import baseline as bl, frame_table as ft, remainder_model as rm, target_build as tb
from fusion_eq.machines import DIII_D

SIGNALS = list(ft.current_signals(DIII_D))


def rows(n: int, seed: int = 0, negative_share: float = 0.3) -> pd.DataFrame:
    """Frame rows with every current mean and both rate windows (no 50 ms toroidal-field rate)."""
    rng = np.random.default_rng(seed)
    d = {}
    for name in SIGNALS:
        d[f"{name}__mean"] = rng.normal(0.0, 100.0, n)
        d[f"{name}__rate"] = rng.normal(0.0, 1000.0, n)
        if name != "tf":
            d[f"{name}__rate50"] = rng.normal(0.0, 1000.0, n)
    d["ip__mean"] = np.abs(rng.normal(900.0, 100.0, n)) * np.where(rng.random(n) < negative_share, -1.0, 1.0)
    return pd.DataFrame(d)


def remainder_basis(k: int = 12, seed: int = 1) -> bl.FluxBasis:
    rng = np.random.default_rng(seed)
    maps = np.einsum("fk,kp->fp", rng.normal(size=(200, k)), rng.normal(size=(k, bl.N_PIXELS))) * 1e-3
    return bl.fit_flux_basis([maps.reshape(-1, bl.GRID, bl.GRID)], n_components=k)


def test_ridge_inputs_per_window_and_the_signed_frame():
    names = set(rows(3).columns)
    windows = rm.available_windows(names)
    assert set(windows) == {"10ms", "50ms"} and all(len(c) == 42 for c in windows.values())
    assert "tf__rate" in windows["50ms"] and "ip__rate50" in windows["50ms"] and "tf__rate50" not in windows["50ms"]
    assert windows["10ms"] == rm.ridge_columns()
    assert set(rm.available_windows({"ip__mean", "ip__rate"})) == {"10ms"}
    with pytest.raises(KeyError):
        rm.window_name(20)
    r = rows(4)
    signed = rm.ridge_frames(r)
    s = np.sign(r["ip__mean"].to_numpy())
    for column in ("ip__mean", "ip__rate", "ip__rate50", "magnetics_F1A__rate50", "sol__rate"):
        assert np.allclose(signed[column], r[column] * s)
    assert np.array_equal(signed["tf__mean"], r["tf__mean"]) and np.array_equal(signed["tf__rate"], r["tf__rate"])


def test_model_recovers_a_planted_linear_relation_and_survives_save_and_load(tmp_path):
    basis = remainder_basis()
    model = rm.RemainderModel.from_basis(basis, n_components=10, alpha=1e-6)
    assert model.explained_cumulative[-1] == pytest.approx(1.0) and len(model.explained_cumulative) == 12
    train, val = rows(400, seed=2), rows(150, seed=3)
    rng = np.random.default_rng(4)
    for window, columns in rm.available_windows(set(train.columns)).items():
        w = rng.normal(size=(len(columns), 10))
        planted = lambda frame: bl.InputScaler.fit(rm.ridge_frames(train), columns).transform(rm.ridge_frames(frame))[0] @ w  # noqa: E731
        ridge = model.fit_ridge(window, columns, train, planted(train))
        assert ridge.scores["train"] > 0.999999 and ridge.columns == tuple(columns)
        assert model.score(window, "validation", val, planted(val)) > 0.999999
    assert model.choose_default() in {"10ms", "50ms"}
    # Encode and decode in the target frame; predict returns the physical frame.
    maps = basis.decode(np.random.default_rng(5).normal(size=(3, 12)) * np.r_[np.ones(10), np.zeros(2)])
    assert np.allclose(model.decode(model.encode(maps)), maps, atol=1e-12)
    target = model.predict_target(val.iloc[:5])
    physical = model.predict(val.iloc[:5])
    s = np.sign(val["ip__mean"].to_numpy()[:5])[:, None, None]
    assert np.allclose(physical, s * target) and physical.shape == (5, bl.GRID, bl.GRID)
    model.save(tmp_path / "rm.npz")
    again = rm.RemainderModel.load(tmp_path / "rm.npz")
    assert again.default_window == model.default_window and again.n_components == 10
    assert np.allclose(again.predict(val), model.predict(val)) and again.ridges["50ms"].scores == model.ridges["50ms"].scores
    with pytest.raises(ValueError, match="components"):
        rm.RemainderModel.from_basis(basis, n_components=13)
    with pytest.raises(ValueError, match="rows"):
        model.fit_ridge("10ms", rm.ridge_columns(), train, np.zeros((3, 10)))


def test_target_config_reads_validates_and_hashes(tmp_path):
    config = tb.TargetConfig.load()
    assert (config.n_current, config.fit_frames, config.seed, config.n_remainder) == (50, 100_000, 0, 10)
    assert config.rate_windows_ms == (10, 50) and config.shape_numbers and config.ridge_alpha == 1.0
    assert config.hash() == tb.TargetConfig.load().hash()
    d = {"version": 1, "current": {"n_components": 60}}
    assert tb.TargetConfig.from_dict(d).n_current == 60 and tb.TargetConfig.from_dict(d).hash() != config.hash()
    for bad, message in (({"version": 1, "current": {"symmetric": False}}, "symmetric"),
                         ({"version": 1, "mask": "unmasked"}, "mask"),
                         ({"version": 1, "remainder": {"rate_windows_ms": [50]}}, "include 10"),
                         ({"version": 1, "extra": 1}, "unknown")):
        with pytest.raises(ValueError, match=message):
            tb.TargetConfig.from_dict(bad)


def test_the_build_refuses_splits_without_labels_and_mast():
    tb.require_labelled("diii_d_train")
    with pytest.raises(ValueError, match="MAST equilibrium labels are not read"):
        tb.require_labelled("mast_public_test")
    with pytest.raises(ValueError, match="no equilibrium labels"):
        tb.require_labelled("diii_d_public_test")
    with pytest.raises(ValueError, match="MAST"):
        tb.read_truth("x", "/nowhere", "mast_public_test", 1)
