"""Tests of ``fusion_eq.scalar_heads`` on synthetic frames: the production heads reproduce the
study's arms, the MAST head applies the MAST constants, the stage-2 refit changes only the MAST
part, and the heads survive a save and load."""

from __future__ import annotations

import numpy as np
import pytest

from fusion_eq import label_filter as lf, scalar_heads as sh, scalars as sc
from tests.test_betan_heads import bh, synthetic_table

BAND = (2.0, 8.0)


def training_rows(seed: int = 1, n_shots: int = 12, n_frames: int = 40):
    rows, shapes = synthetic_table(n_shots, n_frames, seed)
    # Columns of the lighter label filter.
    rows["efit_q95"] = 4.0
    rows["tf__mean"] = -120.0
    return rows, shapes


def test_heads_match_the_study_arms():
    rows, shapes = training_rows()
    heads = sh.fit_heads(rows, shapes, BAND, {"note": "synthetic"})
    keep = lf.lighter_label_filter(rows)
    assert keep.all() and heads.record["n_train_kept"] == len(rows) and heads.record["note"] == "synthetic"
    table = sh.head_table(rows, shapes, BAND)
    fit = bh.fit_stage(table, keep, table)
    np.testing.assert_allclose(heads.predict("DIII-D", rows), fit.val["ridge_raw"], rtol=1e-12)
    assert heads.mast_constant == pytest.approx(fit.constant, rel=1e-12)
    via_doc = heads.mast_correction.predict(table, heads.mast_constant * table[sh.BASE].to_numpy())
    np.testing.assert_allclose(via_doc, fit.val["formula_gbm"], rtol=1e-12)


def test_mast_head_uses_the_mast_constants():
    rows, shapes = training_rows()
    heads = sh.fit_heads(rows, shapes, BAND)
    mast_rows, mast_shapes = training_rows(seed=5, n_shots=3, n_frames=10)
    got = heads.predict("MAST", mast_rows, mast_shapes)
    table = sh.head_table(mast_rows, mast_shapes, BAND, "MAST")
    # MAST: R0 0.85 m and 24 turns in the base, a0 0.65 m in a / a0.
    bt0 = 2e-7 * 24 * np.abs(mast_rows["tf__mean"].to_numpy()) * 1e3 / 0.85
    base = 100.0 * (2 * 4e-7 * np.pi * mast_rows[sh.PRESSURE_COLUMN].to_numpy() * 1e3 / bt0 ** 2) * mast_shapes["a"].to_numpy() \
        * bt0 / (np.abs(mast_rows["ip__mean"].to_numpy()) * 1e-3)
    np.testing.assert_allclose(table[sh.BASE].to_numpy(), base, rtol=1e-12)
    np.testing.assert_allclose(table["a_over_a0"].to_numpy(), mast_shapes["a"].to_numpy() / 0.65, rtol=1e-12)
    np.testing.assert_allclose(got, heads.mast_constant * base * heads.mast_correction.factor(table), rtol=1e-12)
    # Frames without a boundary have no base, so no betaN.
    mast_shapes.loc[:2, ["a", "kappa"]] = np.nan
    assert np.isnan(heads.predict("MAST", mast_rows, mast_shapes)[:3]).all()
    with pytest.raises(ValueError):
        heads.predict("MAST", mast_rows)
    with pytest.raises(ValueError):
        heads.predict("JET", mast_rows)


def test_stage2_refit_changes_only_the_mast_correction():
    rows, shapes = training_rows()
    before = sh.fit_heads(rows, shapes, BAND)
    other = shapes.assign(a=shapes["a"] * 1.1, kappa=shapes["kappa"] + 0.05)
    after = sh.refit_mast(before, rows, other, {"checkpoints": "synthetic"})
    assert after.d3d is before.d3d
    np.testing.assert_array_equal(after.predict("DIII-D", rows), before.predict("DIII-D", rows))
    assert after.q_cyl_band == before.q_cyl_band
    assert after.mast_constant != before.mast_constant
    assert after.mast_correction is not before.mast_correction
    mast_rows, mast_shapes = training_rows(seed=7, n_shots=2, n_frames=8)
    assert not np.allclose(after.predict("MAST", mast_rows, mast_shapes), before.predict("MAST", mast_rows, mast_shapes))
    refit = after.record["stage2_refit"]
    assert refit["constant_before"] == before.mast_constant and refit["constant_after"] == after.mast_constant
    assert refit["checkpoints"] == "synthetic" and "stage2_refit" not in before.record
    # The refit equals a fresh fit on the new shapes.
    fresh = sh.fit_heads(rows, other, BAND)
    np.testing.assert_allclose(after.predict("MAST", mast_rows, mast_shapes), fresh.predict("MAST", mast_rows, mast_shapes), rtol=1e-12)


def test_heads_without_mast_and_the_save_round_trip(tmp_path):
    rows, shapes = training_rows()
    d3d_only = sh.fit_heads(rows, None, BAND)
    assert d3d_only.mast_correction is None and d3d_only.summary()["mast_fitted"] is False
    with pytest.raises(ValueError):
        d3d_only.predict("MAST", rows, shapes)
    heads = sh.fit_heads(rows, shapes, BAND)
    digest = heads.save(tmp_path / "heads.pkl")
    again = sh.BetanHeads.load(tmp_path / "heads.pkl")
    np.testing.assert_array_equal(again.predict("MAST", rows, shapes), heads.predict("MAST", rows, shapes))
    np.testing.assert_array_equal(again.predict("DIII-D", rows), heads.predict("DIII-D", rows))
    assert heads.save(tmp_path / "heads2.pkl") == digest
    summary = heads.summary()
    assert summary["heads"] == {"DIII-D": "ridge_raw", "MAST": "formula_gbm"} and summary["mast_features"] == list(sh.HEAD_FEATURES)
    (tmp_path / "bad.pkl").write_bytes(__import__("pickle").dumps({"not": "heads"}))
    with pytest.raises(TypeError):
        sh.BetanHeads.load(tmp_path / "bad.pkl")


def test_too_few_frames_for_the_mast_head():
    rows, shapes = training_rows(n_shots=1, n_frames=4)
    shapes[["a", "kappa"]] = np.nan
    with pytest.raises(ValueError):
        sh.fit_heads(rows, shapes, BAND)
    assert sc.TF_TURNS == {"DIII-D": 144, "MAST": 24}
