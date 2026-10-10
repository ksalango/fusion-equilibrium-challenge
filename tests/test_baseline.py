"""Tests for the PCA and ridge baseline and the local scorer wrapper."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fusion_eq import baseline as bl, frame_table as ft
from fusion_eq.machines import DIII_D
from tests.conftest import make_d3d_shot, write_shot


def random_frames(n, seed=0, incomplete=0):
    """Frame-table-like rows with the baseline inputs and the two scalar labels."""
    rng = np.random.default_rng(seed)
    cols = bl.input_columns()
    data = {c: rng.normal(size=n) * 100.0 for c in cols if c != bl.VALID_FLAG}
    data[bl.VALID_FLAG] = rng.random(n) < 0.8
    for c in bl.THOMSON_INPUTS:
        data[c][~data[bl.VALID_FLAG]] = np.nan
    frames = pd.DataFrame(data)
    frames.insert(0, "frame", np.arange(n))
    frames.insert(0, "shot", "s")
    for i in range(incomplete):
        frames.loc[i, "ip__mean"] = np.nan
    return frames


def test_input_columns_are_inputs_only():
    cols = bl.input_columns()
    assert len(cols) == 21 + 3 + 1 and cols[-1] == bl.VALID_FLAG
    assert not any(ft.is_label_column(c) for c in cols)
    assert "ip__mean" in cols and "tf__mean" in cols and "magnetics_F9B__mean" in cols


def test_input_scaler_fills_thomson_standardizes_and_flags_incomplete_rows():
    frames = random_frames(500, incomplete=3)
    scaler = bl.InputScaler.fit(frames, bl.input_columns())
    assert set(scaler.fill) == set(bl.THOMSON_INPUTS)
    assert scaler.fill["ts_core_te_peak__interp"] == pytest.approx(np.nanmedian(frames["ts_core_te_peak__interp"]))
    x, complete = scaler.transform(frames)
    assert complete.sum() == 497 and not complete[:3].any()
    assert np.isfinite(x).all() and np.allclose(x[complete].mean(axis=0), 0.0, atol=1e-12)
    assert np.allclose(x[complete].std(axis=0), 1.0)
    assert np.allclose(x[:3, 0], 0.0)                     # missing current predicted from the mean
    back = bl.InputScaler.from_dict(scaler.to_dict())
    assert back.columns == scaler.columns and np.allclose(back.mean, scaler.mean)


def synthetic_maps(n, seed=0):
    rng = np.random.default_rng(seed)
    rr, zz = np.meshgrid(np.linspace(0.84, 2.54, 65), np.linspace(-1.6, 1.6, 65))
    mean = -np.exp(-((rr - 1.7) ** 2 + zz ** 2) / 0.4)
    c1 = (rr - 1.7) * np.exp(-((rr - 1.7) ** 2 + zz ** 2) / 0.5)
    c2 = zz * np.exp(-((rr - 1.7) ** 2 + zz ** 2) / 0.5)
    a, b = rng.normal(size=n) * 0.3, rng.normal(size=n) * 0.1
    return mean + a[:, None, None] * c1 + b[:, None, None] * c2


def test_flux_basis_recovers_two_components_and_roundtrips(tmp_path):
    maps = synthetic_maps(60)
    basis = bl.fit_flux_basis([maps[:25], maps[25:], maps], n_components=3, max_frames=50)
    assert basis.n_frames == 60 and basis.n_components == 3
    assert basis.explained_ratio[:2].sum() > 0.999 and basis.explained_ratio[2] < 1e-6
    assert np.allclose(basis.components @ basis.components.T, np.eye(3), atol=1e-10)
    coefs = basis.encode(maps)
    assert coefs.shape == (60, 3) and np.abs(basis.decode(coefs) - maps).max() < 1e-8
    for row in basis.components:
        assert row[np.argmax(np.abs(row))] > 0
    basis.save(tmp_path / "basis.npz")
    back = bl.FluxBasis.load(tmp_path / "basis.npz")
    assert back.hash() == basis.hash() and np.allclose(back.encode(maps[:2]), coefs[:2])
    with pytest.raises(ValueError):
        bl.BasisAccumulator().finish(2)


def test_fit_pca_ridge_recovers_a_linear_map_and_predict_keeps_order(tmp_path):
    rng = np.random.default_rng(5)
    frames = random_frames(2000, seed=2, incomplete=5)
    basis = bl.fit_flux_basis([synthetic_maps(40)], n_components=2)
    scaler = bl.InputScaler.fit(frames, bl.input_columns())
    x, complete = scaler.transform(frames)
    w_true = rng.normal(size=(x.shape[1], 2))
    coefs = x @ w_true + np.array([0.5, -0.2])
    frames["efit_q95"] = x @ rng.normal(size=x.shape[1]) + 4.0
    frames["efit_beta_n"] = x @ rng.normal(size=x.shape[1]) + 1.0
    frames.loc[10:14, "efit_q95"] = np.nan                   # a label gap, not an input gap
    params = bl.BaselineParams(n_components=2, alpha=1e-6)
    model = bl.fit_pca_ridge(frames, coefs, basis, params)
    assert model.n_train_frames == 1995 and np.allclose(model.map_weights, w_true, atol=1e-6)
    pred = bl.predict(model, frames)
    assert pred.psirz.shape == (2000, 65, 65) and pred.psirz.dtype == np.float32
    q95 = frames["efit_q95"].to_numpy()
    labelled = complete & np.isfinite(q95)
    assert np.allclose(pred.q95[labelled], q95[labelled], atol=1e-5) and np.isfinite(pred.q95).all()
    assert np.allclose(pred.betaN[complete], frames["efit_beta_n"].to_numpy()[complete], atol=1e-5)
    expected = basis.decode(coefs[complete])
    assert np.abs(pred.psirz[complete] - expected).max() < 1e-4
    # Row order: the prediction of a reordered table follows the rows given.
    reversed_pred = bl.predict(model, frames.iloc[::-1])
    assert np.allclose(reversed_pred.q95, pred.q95[::-1])
    model.save(tmp_path / "model.npz")
    back = bl.PcaRidge.load(tmp_path / "model.npz")
    assert np.allclose(bl.predict(back, frames.iloc[:3]).q95, pred.q95[:3]) and back.params == params
    with pytest.raises(ValueError):
        bl.fit_pca_ridge(frames, coefs[:-1], basis, params)


@pytest.fixture
def data_root(tmp_path):
    root = tmp_path / "data"
    (root / "diii_d_train").mkdir(parents=True)
    for shot_id, n in (("b2", 6), ("a1", 4)):
        write_shot(root / "diii_d_train" / f"d3d_shot_{shot_id}.parquet", make_d3d_shot(n_frames=n))
    return root


def test_encode_shots_caches_and_keeps_frame_table_order(data_root, tmp_path):
    maps = ft.read_flux_maps(["a1", "b2"], "diii_d_train", data_root)
    basis = bl.fit_flux_basis([maps], n_components=2)
    cache = tmp_path / "cache"
    coefs = bl.encode_shots(["b2", "a1"], basis, data_root, cache)
    assert coefs.shape == (10, 2) and coefs.dtype == np.float32
    assert np.allclose(coefs, basis.encode(maps), atol=1e-5)
    files = sorted(p.name for p in (cache / basis.hash()[:16]).glob("*.npy"))
    assert files == ["a1.npy", "b2.npy"]
    (cache / basis.hash()[:16] / "a1.npy").unlink()
    np.save(cache / basis.hash()[:16] / "a1.npy", np.full((4, 2), 7.0, dtype=np.float32))
    again = bl.encode_shots(["a1", "b2"], basis, data_root, cache)
    assert np.all(again[:4] == 7.0) and np.allclose(again[4:], coefs[4:])      # cache hit, b2 unchanged


def test_score_local_perfect_scores_one_and_zeros_zero(data_root, starter):
    ids = ["a1", "b2"]
    truth = {s: bl.load_truth(s, data_root) for s in ids}
    perfect = {s: bl.Prediction(t["psi"], t["q95"], t["betaN"]) for s, t in truth.items()}
    result = bl.score_local(perfect, ids, data_root)
    final = result.finalize()
    assert final["S"] == pytest.approx(1.0, abs=1e-6) and final["n_shots"] == 2 and result.psi_sign == 1.0
    terms = bl.score_terms(final)
    assert terms["r2_psi"] == pytest.approx(1.0) and terms["one_minus_dlcfs"] == pytest.approx(1.0)
    assert terms["r2_q95"] == pytest.approx(1.0) and set(terms) >= {"r2_R_axis", "r2_li", "consistency"}
    only_first = result.finalize([True, False])
    assert only_first["n_shots"] == 1 and only_first["S"] == pytest.approx(1.0, abs=1e-6)
    zeros = {s: bl.Prediction(np.zeros_like(t["psi"]), np.zeros_like(t["q95"]), np.zeros_like(t["betaN"]))
             for s, t in truth.items()}
    # Flux, scalars and boundary score 0 on a zero map. The consistency term is not exactly 0
    # here because the synthetic shots share one shape, so a derived scalar is nearly constant
    # and a fallback value can land near its mean; on real shots the local scorer gives S = 0.
    zero_terms = bl.score_terms(bl.score_local(zeros, ids, data_root).finalize())
    assert zero_terms["r2_psi"] <= 0.0 and zero_terms["r2_qb"] <= 0.0 and zero_terms["one_minus_dlcfs"] == 0.0
    assert zero_terms["S"] < 0.1
    # A sign-flipped perfect map is normalized by the global sign, as on the platform.
    flipped = {s: bl.Prediction(-t["psi"], t["q95"], t["betaN"]) for s, t in truth.items()}
    flipped_result = bl.score_local(flipped, ids, data_root)
    assert flipped_result.psi_sign == -1.0 and flipped_result.finalize()["S"] == pytest.approx(1.0, abs=1e-6)
    # A frame stride keeps every second frame on both sides; references built once are reused.
    refs = bl.build_references(ids, data_root, frame_stride=2, workers=1, cache_dir=None)
    strided = bl.score_local(lambda s: bl.Prediction(truth[s]["psi"][::2], truth[s]["q95"][::2], truth[s]["betaN"][::2]),
                             ids, data_root, frame_stride=2, references=refs)
    assert strided.ref_stats[0]["psi_n"] == 2 * 65 * 65 and strided.finalize()["S"] == pytest.approx(1.0, abs=1e-6)
    with pytest.raises(ValueError):
        bl.score_local(perfect, ids, data_root, references=refs)


def _same_parts(a: dict, b: dict) -> bool:
    if a.keys() != b.keys():
        return False
    for key in a:
        x, y = a[key], b[key]
        if isinstance(x, dict):
            if x.keys() != y.keys() or any(x[k] != y[k] for k in x):
                return False
        elif isinstance(x, np.ndarray):
            if not np.array_equal(x, y):
                return False
        elif x != y:
            return False
    return True


def test_parallel_scoring_and_reference_cache_match_the_single_process_path(data_root, starter, tmp_path, monkeypatch):
    ids = ["a1", "b2"]
    truth = {s: bl.load_truth(s, data_root) for s in ids}
    rng = np.random.default_rng(0)
    rough = {s: bl.Prediction(t["psi"] * 0.9 + rng.normal(scale=0.02, size=t["psi"].shape).astype(np.float32),
                              t["q95"] + 0.3, t["betaN"] * 1.1) for s, t in truth.items()}
    cache = tmp_path / "references"
    serial = bl.build_references(ids, data_root, workers=1, cache_dir=None)
    parallel = bl.build_references(ids, data_root, workers=2, cache_dir=cache)
    assert not serial.from_cache and not parallel.from_cache
    files = list(cache.glob("refs_*.npz"))
    assert len(files) == 1 and files[0].name == f"refs_{bl.reference_key(ids, 'diii_d_train', 1)}.npz"
    # Read back from the cache, without building anything.
    monkeypatch.setattr(bl, "_shot_reference", lambda *a: (_ for _ in ()).throw(AssertionError("built instead of cached")))
    cached = bl.build_references(ids, data_root, workers=1, cache_dir=cache)
    assert cached.from_cache
    for other in (parallel, cached):
        assert other.shot_ids == serial.shot_ids and other.frame_stride == serial.frame_stride
        for (c0, cons0, m0, r0), (c1, cons1, m1, r1) in zip(serial.refs, other.refs):
            assert len(c0) == len(c1) and all((x is None and y is None) or np.array_equal(x, y) for x, y in zip(c0, c1))
            assert np.array_equal(cons0, cons1, equal_nan=True) and np.array_equal(m0, m1) and r0 == r1
        for s0, s1 in zip(serial.ref_stats, other.ref_stats):
            assert all(np.array_equal(np.asarray(s0[k]), np.asarray(s1[k])) for k in s0)
    monkeypatch.undo()
    # Another stride or shot order is another key.
    assert bl.reference_key(ids, "diii_d_train", 2) != bl.reference_key(ids, "diii_d_train", 1)
    assert bl.reference_key(ids[::-1], "diii_d_train", 1) != bl.reference_key(ids, "diii_d_train", 1)
    # Scoring: identical parts and totals from one process and from two.
    one = bl.score_local(rough, ids, data_root, references=serial, workers=1)
    two = bl.score_local(rough, ids, data_root, references=cached, workers=2)
    assert one.psi_sign == two.psi_sign and one.shot_ids == two.shot_ids
    assert all(_same_parts(p, q) for p, q in zip(one.parts, two.parts))
    assert one.finalize() == two.finalize() and one.finalize([True, False]) == two.finalize([True, False])
    assert 0.0 < one.finalize()["S"] < 1.0
    assert bl.default_workers() >= 1
