"""Tests for the four baselines of the comparison, on synthetic shots."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fusion_eq import baseline as bl, coil_flux as cf, frame_table as ft, io, models, targets
from fusion_eq.machines import DIII_D
from tests.conftest import make_d3d_shot, write_shot


@pytest.fixture
def setup(tmp_path, starter):
    """Three synthetic training shots (one with negative Ip), their frame rows and a Green's table."""
    root = tmp_path / "data"
    (root / "diii_d_train").mkdir(parents=True)
    rows = []
    for shot_id, n, sign in (("a1", 6, 1.0), ("b2", 5, -1.0), ("c3", 7, 1.0)):
        raw = make_d3d_shot(n_frames=n)
        for col in DIII_D.current_columns:
            raw[col] = (np.asarray(raw[col], dtype=np.float64) * sign).astype(np.float32)
        path = write_shot(root / "diii_d_train" / f"d3d_shot_{shot_id}.parquet", raw)
        shot = io.load_shot(path, ft.columns_to_load(DIII_D, True, False))
        rows.append(ft.build_shot_frames(shot, "diii_d_train", ft.BuildParams(psi_summary=False), True))
    frames = pd.concat(rows, ignore_index=True)
    frames["shot"] = frames["shot"].astype(str)
    geometry = cf.CoilGeometry.from_shot(io.load_shot(root / "diii_d_train" / "d3d_shot_a1.parquet", cf.GEOMETRY_COLUMNS, apply=False))
    table = cf.build_green_table(DIII_D, geometry, np.linspace(0.84, 2.54, 65), np.linspace(-1.6, 1.6, 65), n_filaments=1)
    return root, frames, table


def test_fit_model_set_and_predictions(setup, tmp_path):
    root, frames, table = setup
    ids = ["a1", "b2", "c3"]
    params = bl.BaselineParams(n_components=2, alpha=1.0, pca_frames=None, seed=0)
    cache = tmp_path / "cache"
    scales = cf.adopted_scales("DIII-D", table.columns)
    model_set = models.fit_model_set(frames, ids, root, table, params, cache, scales, log=lambda m: None)
    assert model_set.basis_psi.hash() != model_set.basis_plasma.hash()
    assert model_set.n_mean_frames == len(frames) and model_set.mean_shape.shape == (65, 65)
    folders = sorted(p.name for p in cache.iterdir())
    assert any(n.startswith("basis_") and n.endswith(".npz") and "plasma" in n for n in folders)
    assert any("_plasma_" in n and (cache / n).is_dir() for n in folders)
    baselines = model_set.baselines()
    assert set(baselines) == set(models.MODEL_NAMES)
    assert baselines["physics_mean"].scalars_from == "ridge_signed" and baselines["ridge_raw"].scalars_from is None
    val = frames[frames["shot"] == "b2"].iloc[::2]
    predictions = {name: m.predict(val) for name, m in baselines.items()}
    for p in predictions.values():
        assert p.psirz.shape == (len(val), 65, 65) and p.psirz.dtype == np.float32 and np.isfinite(p.psirz).all()
        assert p.q95.shape == (len(val),) and p.betaN.shape == (len(val),)
    # Models 3 and 4 carry the scalars of model 2 exactly.
    for name in ("physics_mean", "hybrid_ridge"):
        assert np.array_equal(predictions[name].q95, predictions["ridge_signed"].q95)
        assert np.array_equal(predictions[name].betaN, predictions["ridge_signed"].betaN)
    # physics_mean is the coil flux plus Ip times the mean shape, in the stored frame.
    ip = val["ip__mean"].to_numpy(dtype=np.float64)
    coil = targets.coil_flux_of_frames(table, val, scales)
    expected = targets.stored_from_target(np.abs(ip)[:, None, None] * model_set.mean_shape, coil, ip)
    assert np.allclose(predictions["physics_mean"].psirz, expected, atol=1e-5)
    # hybrid_ridge decodes its own basis: the target of a training frame round-trips through the basis.
    t = targets.plasma_target_of_frames(ft.read_flux_maps(["b2"], "diii_d_train", root), table, frames[frames["shot"] == "b2"], scales)
    coefs = bl.encode_shots(["b2"], model_set.basis_plasma, root, cache, target=models.plasma_target_fn(table, {"b2": frames[frames["shot"] == "b2"]}, scales), tag="t")
    assert np.allclose(model_set.basis_plasma.decode(coefs), t, atol=1e-3) or model_set.basis_plasma.explained_ratio.sum() < 0.999
    # A second fit reads everything from the cache and gives the same models.
    again = models.fit_model_set(frames, ids, root, table, params, cache, scales, log=lambda m: None)
    assert np.allclose(again.ridge_raw.map_weights, model_set.ridge_raw.map_weights)
    assert np.allclose(again.mean_shape, model_set.mean_shape)


def test_ridge_signed_equals_ridge_raw_when_every_ip_is_positive(setup, tmp_path):
    root, frames, table = setup
    positive = frames[frames["shot"] != "b2"].reset_index(drop=True)
    params = bl.BaselineParams(n_components=2, alpha=1.0, pca_frames=None, seed=0)
    model_set = models.fit_model_set(positive, ["a1", "c3"], root, table, params, tmp_path / "cache", log=lambda m: None)
    baselines = model_set.baselines()
    val = positive[positive["shot"] == "c3"]
    raw, signed = baselines["ridge_raw"].predict(val), baselines["ridge_signed"].predict(val)
    assert np.array_equal(raw.psirz, signed.psirz) and np.array_equal(raw.q95, signed.q95)
    # On a negative-Ip shot the signed model sees the currents flipped into the physical frame.
    negative = frames[frames["shot"] == "b2"]
    x_raw, _ = model_set.ridge_raw.scaler.transform(negative)
    x_signed, _ = model_set.ridge_signed.scaler.transform(targets.signed_frames(negative))
    columns = model_set.ridge_signed.scaler.columns
    flipped = [i for i, c in enumerate(columns) if c in targets.signed_current_columns()]
    assert not np.allclose(x_raw[:, flipped], x_signed[:, flipped])
    assert np.allclose(x_raw[:, columns.index("tf__mean")], x_signed[:, columns.index("tf__mean")])


def test_residual_summary_and_score_parts(setup, tmp_path):
    root, frames, table = setup
    params = bl.BaselineParams(n_components=2, alpha=1.0, pca_frames=None, seed=0)
    model_set = models.fit_model_set(frames, ["a1", "b2", "c3"], root, table, params, tmp_path / "cache", log=lambda m: None)
    val_by_shot = {s: g.sort_values("frame").iloc[::2] for s, g in frames.groupby("shot") if s in ("a1", "b2")}
    # A perfect model: the truth itself, in the stored frame.
    truth = {s: bl.load_truth(s, root, stride=2) for s in val_by_shot}
    perfect = models.Baseline("perfect", lambda f: truth[f["shot"].iloc[0]]["psi"],
                              lambda f: (truth[f["shot"].iloc[0]]["q95"], truth[f["shot"].iloc[0]]["betaN"]))
    summary = models.residual_summary("perfect", perfect, val_by_shot, root, 2, psi_sign=1.0)
    assert summary.counts["all"] == sum(len(v) for v in val_by_shot.values()) == len(summary.frames)
    assert np.allclose(summary.mean_abs["all"], 0.0) and summary.frames["rms_residual"].max() == 0.0
    assert set(summary.frames["phase"]) <= set(models.RESIDUAL_GROUPS[1:])
    assert summary.frames.loc[summary.frames["shot"] == "b2", "ip_direction"].eq(-1).all()
    assert sum(summary.counts[g] for g in models.RESIDUAL_GROUPS[1:]) == summary.counts["all"]
    # A real model: residuals are positive, and the sign the scorer chose is applied.
    model = model_set.baselines()["physics_mean"]
    real = models.residual_summary("physics_mean", model, val_by_shot, root, 2, psi_sign=1.0)
    flipped = models.residual_summary("physics_mean", model, val_by_shot, root, 2, psi_sign=-1.0)
    assert real.frames["rms_residual"].gt(0).all() and not np.allclose(real.mean_abs["all"], flipped.mean_abs["all"])
    assert real.frames["flux_range"].gt(0).all() and real.frames["sol_ka"].notna().all()
    # Score parts: one row per shot with the scorer's sums.
    ids = sorted(val_by_shot)
    result = bl.score_local(lambda s: model.predict(val_by_shot[s]), ids, root, frame_stride=2, workers=1,
                            references=bl.build_references(ids, root, frame_stride=2, workers=1, cache_dir=None))
    parts = models.score_parts_table("physics_mean", result)
    assert parts["shot"].tolist() == ids and parts["psi_pixels"].tolist() == [truth[s]["psi"].size for s in ids]
    assert {"ss_res_psi", "ss_res_q95", "ss_res_betaN", "ss_res_li", "shot_dlcfs", "cons_fails_kappa"} <= set(parts.columns)
    assert parts["ss_res_psi"].ge(0).all()


def test_ceiling_baselines_decode_the_truth_through_the_bases(setup, tmp_path):
    root, frames, table = setup
    params = bl.BaselineParams(n_components=2, alpha=1.0, pca_frames=None, seed=0)
    model_set = models.fit_model_set(frames, ["a1", "b2", "c3"], root, table, params, tmp_path / "cache", log=lambda m: None)
    ceilings = models.ceiling_baselines(model_set, root, frame_stride=2)
    assert set(ceilings) == set(models.CEILING_NAMES) and all(c.scalars_from == "ridge_signed" for c in ceilings.values())
    val = frames[frames["shot"] == "b2"].sort_values("frame").iloc[::2]
    truth = bl.load_truth("b2", root, stride=2)
    psi_pred = ceilings["ceiling_psi_basis"].predict(val)
    expected = model_set.basis_psi.decode(model_set.basis_psi.encode(truth["psi"].astype(np.float64)))
    assert np.allclose(psi_pred.psirz, expected, atol=1e-5)
    plasma_pred = ceilings["ceiling_plasma_basis"].predict(val)
    ip = val["ip__mean"].to_numpy(dtype=np.float64)
    coil = targets.coil_flux_of_frames(table, val, model_set.scales)
    t = targets.plasma_target(truth["psi"].astype(np.float64), coil, ip)
    expected = targets.stored_from_target(model_set.basis_plasma.decode(model_set.basis_plasma.encode(t)), coil, ip)
    assert np.allclose(plasma_pred.psirz, expected, atol=1e-5)
    # Scalars are the ridge_signed ones; the synthetic maps are spanned by two components, so both ceilings are tight.
    signed = model_set.baselines()["ridge_signed"].predict(val)
    assert np.array_equal(psi_pred.q95, signed.q95) and np.array_equal(plasma_pred.betaN, signed.betaN)
    assert np.abs(psi_pred.psirz - truth["psi"]).max() < 1e-3 and np.abs(plasma_pred.psirz - truth["psi"]).max() < 1e-3
    with pytest.raises(ValueError):
        ceilings["ceiling_psi_basis"].predict(frames.iloc[::2])
