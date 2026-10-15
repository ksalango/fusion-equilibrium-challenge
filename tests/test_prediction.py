"""Tests of ``fusion_eq.prediction`` on synthetic inputs: mirror averaging (Z11), seed averaging
and the checkpoint checks, the MAST sign assertion (Z12), the label guard, the input join, and the
q95 and shape workers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from fusion_eq import features as fe, plasma_models as pm, prediction as pr, starter_kit, targets
from fusion_eq.feature_scaler import RobustScaler
from fusion_eq.model_config import ModelConfig
from tests.model_fixtures import FEATURE_CONFIG, feature_values, mirror_matrix

K = 6
PARITY = np.array([1, -1, 1, -1, 1, -1], dtype=np.float64)


@pytest.fixture(scope="module")
def setup():
    rng = np.random.default_rng(0)
    fconfig = fe.FeatureConfig.load(FEATURE_CONFIG)
    table = feature_values(fconfig, 400, rng)
    scaler = RobustScaler.fit(table, fe.mirror_features(table, fconfig), fconfig)
    columns = fconfig.set_columns("set_a")
    return {"fconfig": fconfig, "table": table, "scaler": scaler, "columns": columns, "a": mirror_matrix(fconfig, columns)}


def member(model: torch.nn.Module, columns, config: ModelConfig | None = None, run_id: str = "r") -> pr.Member:
    config = config or ModelConfig.load()
    meta = {"columns": list(columns), "parity": PARITY.tolist(), "sigma": [1.0] * K, "config": config.to_dict(),
            "config_hash": config.hash(), "feature_version": 2, "target_version": 1, "epoch": 0}
    return pr.Member(run_id=run_id, path=None, sha256="", model=model.eval(), meta=meta)


def equivariant_model(n_in: int, a: np.ndarray, seed: int) -> pm.LinearMLP:
    """A linear model with ``A W = W diag(p)`` and an even intercept, so that it predicts the
    mirror of a frame as the parity-flipped prediction of the frame."""
    rng = np.random.default_rng(seed)
    w0 = rng.normal(size=(n_in, K))
    w = 0.5 * (w0 + a @ w0 * PARITY)
    b = rng.normal(size=K) * (PARITY > 0)
    model = pm.LinearMLP(n_in, K, hidden=[8], dropout=0.0)
    pm.load_linear(model, w, b)
    return model


def symmetric(table: pd.DataFrame, fconfig: fe.FeatureConfig) -> pd.DataFrame:
    """Frames equal to their own mirror: the mean of each frame and its mirror."""
    mirrored = fe.mirror_features(table, fconfig)
    out = table.copy()
    out[fconfig.all_columns()] = 0.5 * (table[fconfig.all_columns()].to_numpy() + mirrored[fconfig.all_columns()].to_numpy())
    return out


def test_mirror_averaging_of_a_symmetric_frame_equals_the_single_prediction(setup):
    s = setup
    sym = symmetric(s["table"].iloc[:50].reset_index(drop=True), s["fconfig"])
    pd.testing.assert_frame_equal(fe.mirror_features(sym, s["fconfig"]), sym)
    torch.manual_seed(0)
    model = equivariant_model(len(s["columns"]), s["a"], 1)
    on = pr.CoefficientModel([member(model, s["columns"])], s["fconfig"], s["scaler"], 10.0, mirror_average=True)
    off = pr.CoefficientModel([member(model, s["columns"])], s["fconfig"], s["scaler"], 10.0, mirror_average=False)
    single = off.predict(sym, "DIII-D")
    np.testing.assert_allclose(on.predict(sym, "DIII-D"), single, atol=1e-6)
    # The scaled mirror is x A, so an equivariant model is unchanged by averaging on any frame.
    x = on.scaled(s["table"], "DIII-D")
    np.testing.assert_allclose(on.scaled(s["table"], "DIII-D", mirror=True), x @ s["a"], atol=1e-6)
    np.testing.assert_allclose(on.predict(s["table"], "DIII-D"), off.predict(s["table"], "DIII-D"), atol=1e-5)


def test_mirror_averaging_of_any_model_is_the_mean_with_the_flipped_mirror(setup):
    s = setup
    torch.manual_seed(3)
    model = pm.LinearMLP(len(s["columns"]), K, hidden=[16], dropout=0.0)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.3 * torch.randn_like(p))
    on = pr.CoefficientModel([member(model, s["columns"])], s["fconfig"], s["scaler"], 10.0, mirror_average=True)
    x, xm = on.scaled(s["table"], "DIII-D"), on.scaled(s["table"], "DIII-D", mirror=True)
    expected = 0.5 * (pm.predict(model, x) + pm.predict(model, xm) * PARITY)
    np.testing.assert_array_equal(on.predict(s["table"], "DIII-D"), expected)
    # On a symmetric frame the averaged odd components vanish.
    sym = symmetric(s["table"].iloc[:20].reset_index(drop=True), s["fconfig"])
    assert np.abs(on.predict(sym, "DIII-D")[:, PARITY < 0]).max() < 1e-6
    # Clipping applies before the model, to the original and the mirror alike.
    wild = s["table"].iloc[:5].copy()
    wild[s["columns"][0]] = 1e6
    assert on.scaled(wild, "DIII-D")[:, 0].max() == 10.0


def test_seed_averaging_and_member_checks(setup, tmp_path):
    s = setup
    models = []
    for seed in (0, 1):
        torch.manual_seed(seed)
        m = pm.LinearMLP(len(s["columns"]), K, hidden=[8], dropout=0.0)
        with torch.no_grad():
            for p in m.parameters():
                p.add_(0.2 * torch.randn_like(p))
        models.append(m)
    base = ModelConfig.load()
    for seed, m in enumerate(models):
        config = base.with_values({"train.seed": seed})
        meta = member(m, s["columns"], config).meta
        pm.save_checkpoint(tmp_path / f"run{seed}" / "best.pt", m, meta)
    members = pr.load_members(["run0", "run1"], tmp_path)
    assert [m.run_id for m in members] == ["run0", "run1"] and all(len(m.sha256) == 64 for m in members)
    both = pr.CoefficientModel(members, s["fconfig"], s["scaler"], 10.0, mirror_average=False)
    one = [pr.CoefficientModel([m], s["fconfig"], s["scaler"], 10.0, mirror_average=False).predict(s["table"], "DIII-D")
           for m in members]
    np.testing.assert_allclose(both.predict(s["table"], "DIII-D"), 0.5 * (one[0] + one[1]), rtol=1e-12)
    # A run that differs beyond the seed is refused.
    other = base.with_values({"train.seed": 2, "train.lr": 1e-3})
    pm.save_checkpoint(tmp_path / "run2" / "best.pt", models[0], member(models[0], s["columns"], other).meta)
    with pytest.raises(ValueError):
        pr.load_members(["run0", "run2"], tmp_path)
    pm.save_checkpoint(tmp_path / "run3" / "best.pt", models[0], {**member(models[0], s["columns"]).meta, "columns": s["columns"][::-1]})
    with pytest.raises(ValueError):
        pr.load_members(["run0", "run3"], tmp_path)
    with pytest.raises(FileNotFoundError):
        pr.load_members(["absent"], tmp_path)
    with pytest.raises(ValueError):
        pr.load_members([], tmp_path)


def test_the_chain_refuses_a_mast_map_without_the_plus_one_sign(monkeypatch):
    assert pr.assert_mast_sign() == 1.0
    composer = pr.MapComposer(objects=None, tables={}, scales={}, sigma=np.ones(K))
    monkeypatch.setitem(targets.PSI_SIGN_RULE_C, "MAST", -1.0)
    with pytest.raises(ValueError, match="MAST sign constant"):
        pr.assert_mast_sign()
    with pytest.raises(ValueError, match="MAST sign constant"):
        composer.maps("MAST", pd.DataFrame({"ip__mean": [500.0]}), np.zeros((1, K)))


def test_the_label_guard_and_mast_labels(tmp_path):
    assert pr.require_inputs(["shot", "frame", "ip__mean", "tf__mean"], "MAST") == ["shot", "frame", "ip__mean", "tf__mean"]
    for label in ("efit_q95", "efit_psirz", "magnetics_dsep"):
        with pytest.raises(ValueError):
            pr.require_inputs(["ip__mean", label], "MAST")
    with pytest.raises(ValueError):
        pr.TableSource(tmp_path, tmp_path, 2, {"MAST": ["ip__mean", "efit_beta_n"]}, [])
    source = pr.TableSource(tmp_path, tmp_path, 2, {"MAST": ["ip__mean"]}, [])
    # Asking a MAST split for labels fails before anything is read.
    with pytest.raises(ValueError, match="MAST equilibrium labels"):
        pr.read_rows(source, "mast_public_test", ["m1"], labels=("efit_q95",))
    with pytest.raises(ValueError):
        pr.read_rows(source, "diii_d_public_test", ["t1"], labels=("efit_q95",))


def test_merge_inputs_joins_and_guards():
    frames = pd.DataFrame({"shot": ["b", "a", "a"], "frame": np.array([0, 1, 0], dtype=np.int32), "ip__mean": [1.0, 2.0, 3.0]})
    feats = pd.DataFrame({"shot": ["a", "a", "b"], "frame": np.array([0, 1, 0], dtype=np.int32), "q_cyl": [4.0, 5.0, 6.0]})
    out = pr.merge_inputs(frames, feats)
    assert out["shot"].tolist() == ["a", "a", "b"] and out["frame"].tolist() == [0, 1, 0]
    assert out["ip__mean"].tolist() == [3.0, 2.0, 1.0] and out["q_cyl"].tolist() == [4.0, 5.0, 6.0]
    with pytest.raises(ValueError):
        pr.merge_inputs(frames, feats.iloc[:2])
    with pytest.raises(ValueError):
        pr.merge_inputs(frames, feats.assign(ip__mean=0.0))


@pytest.mark.skipif(not starter_kit.is_available(), reason="starter kit not available")
def test_scalar_workers_do_not_change_the_values():
    r, z = np.linspace(0.84, 2.54, 65), np.linspace(-1.6, 1.6, 65)
    rr, zz = np.meshgrid(r, z)
    psi = np.stack([-(((rr - 1.7) / (0.5 + 0.01 * k)) ** 2 + (zz / 0.9) ** 2) for k in range(6)]).astype(np.float32)
    tf = np.full(6, 120.0)
    one = pr.ScalarWorkers(1)("DIII-D", psi, tf, r, z, True)
    pool = pr.ScalarWorkers(2, chunk=2)
    try:
        two = pool("DIII-D", psi, tf, r, z, True)
    finally:
        pool.close()
    np.testing.assert_array_equal(one[0], two[0])
    assert pd.DataFrame(one[1]).equals(pd.DataFrame(two[1]))
    frame = pr.shapes_frame(["s"] * 6, range(6), one[1])
    assert list(frame.columns) == ["shot", "frame", "R_geo", "a", "kappa", "tri_top", "tri_bot", "eps", "boundary_found"]
