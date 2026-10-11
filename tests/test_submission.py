"""Tests of the submission pipeline on synthetic shots: key naming and order, frame
counts, the float16 round trip, every fallback, the validator, the pointer manifest and
the token's isolation, and the ``baseline_v0`` predictor end to end through the script."""

from __future__ import annotations

import importlib.util
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fusion_eq import baseline as bl, coil_flux as cf, frame_table as ft, io, models, split as sp, submission as sub, targets
from fusion_eq.dataset import REPO_ROOT
from fusion_eq.machines import DIII_D, MAST
from tests.conftest import drop_labels, make_d3d_shot, make_mast_shot, write_shot

D3D_GRID = (np.linspace(0.84, 2.54, 65), np.linspace(-1.6, 1.6, 65))
MAST_GRID = (np.linspace(0.06, 2.0, 65), np.linspace(-2.0, 2.0, 65))
MEDIANS = {"q95": 4.5, "betaN": 1.25}


class Dummy:
    """A predictor whose output is a known function of the frame index, with chosen
    non-finite frames, or an exception on a chosen shot."""

    name = "dummy"

    def __init__(self, bad_frames=(), bad_q95=(), betan_all_nan=False, all_bad=False, raise_on=None):
        self.bad_frames, self.bad_q95, self.betan_all_nan, self.all_bad, self.raise_on = bad_frames, bad_q95, betan_all_nan, all_bad, raise_on

    def setup(self, log=print) -> dict:
        return {"dummy": True}

    def predict_shot(self, shot, machine):
        if self.raise_on is not None and shot.shot_id == self.raise_on:
            raise RuntimeError("boom")
        n = shot.n_frames
        psi = 0.01 * (np.arange(n, dtype=np.float64)[:, None, None] + 1.0) * np.ones((n, 65, 65))
        psi[:, 3, 4] += 0.5
        q95 = 4.0 + 0.1 * np.arange(n)
        betan = 1.0 + 0.01 * np.arange(n)
        for k in self.bad_frames:
            psi[k, 10, 10] = np.nan
        for k in self.bad_q95:
            q95[k] = np.inf
        if self.betan_all_nan:
            betan[:] = np.nan
        if self.all_bad:
            psi[:] = np.nan
        return bl.Prediction(psi, q95, betan)


def green(machine_cfg, path, grid):
    geometry = cf.CoilGeometry.from_shot(io.load_shot(path, cf.GEOMETRY_COLUMNS, apply=False))
    return cf.build_green_table(machine_cfg, geometry, *grid, n_filaments=1)


@pytest.fixture
def test_root(tmp_path, starter):
    """Two inputs-only shots per machine, written out of name order to test the stream order."""
    root = tmp_path / "raw"
    for folder in sub.TEST_CONFIGS.values():
        (root / folder).mkdir(parents=True)
    write_shot(root / "diii_d_public_test" / "d3d_shot_b2.parquet", drop_labels(make_d3d_shot(n_frames=5)))
    write_shot(root / "diii_d_public_test" / "d3d_shot_a1.parquet", drop_labels(make_d3d_shot(n_frames=12)))
    write_shot(root / "mast_public_test" / "mast_shot_m2.parquet", drop_labels(make_mast_shot(n_frames=4)))
    write_shot(root / "mast_public_test" / "mast_shot_m1.parquet", drop_labels(make_mast_shot(n_frames=8)))
    return root


def d3d_fallbacks(root, with_mean_shape=True):
    table = green(DIII_D, root / "diii_d_public_test" / "d3d_shot_a1.parquet", D3D_GRID)
    mean_shape = np.full((65, 65), 1e-4) if with_mean_shape else None
    return sub.Fallbacks("DIII-D", dict(MEDIANS), table, cf.adopted_scales("DIII-D", table.columns), mean_shape)


def test_keys_follow_sorted_file_order_and_t_is_preserved(test_root):
    files = sub.test_shot_files(test_root, "DIII-D")
    assert [p.name for p in files] == ["d3d_shot_a1.parquet", "d3d_shot_b2.parquet"]
    assert sub.shot_key(7, "betaN") == "shot_0007_betaN" and sub.expected_keys(1) == ["shot_0000_psirz", "shot_0000_q95", "shot_0000_betaN"]
    arrays = {}
    record = sub.predict_fold(Dummy(), files, "DIII-D", d3d_fallbacks(test_root, False), arrays.__setitem__, log=lambda m: None)
    assert list(arrays) == sub.expected_keys(2)
    assert arrays["shot_0000_psirz"].shape == (12, 65, 65) and arrays["shot_0001_psirz"].shape == (5, 65, 65)
    assert arrays["shot_0000_q95"].shape == (12,) and arrays["shot_0001_betaN"].shape == (5,)
    assert arrays["shot_0000_psirz"].dtype == np.float16 and arrays["shot_0000_q95"].dtype == np.float32
    assert record.shot_ids == ["a1", "b2"] and record.n_frames_per_shot == [12, 5] and record.counts.n_frames == 17
    assert all(v == 0 for v in record.counts.counts.values()) and not record.counts.predictor_errors
    with pytest.raises(ValueError):
        sub.shot_key(0, "li")


def test_label_columns_never_reach_the_predictor(tmp_path, starter):
    path = write_shot(tmp_path / "d3d_shot_x.parquet", make_d3d_shot(n_frames=3))
    seen = {}

    class Spy(Dummy):
        def predict_shot(self, shot, machine):
            seen.update({"columns": shot.columns})
            return super().predict_shot(shot, machine)

    sub.predict_fold(Spy(), [path], "DIII-D", sub.Fallbacks("DIII-D", dict(MEDIANS)), dict().__setitem__, log=lambda m: None)
    assert not any(ft.is_label_column(c) for c in seen["columns"])
    assert "efit_times" in seen["columns"] and "magnetics_plasma_current" in seen["columns"]


def test_npz_round_trip_is_float16_in_key_order_and_byte_identical(test_root, tmp_path):
    files = sub.test_shot_files(test_root, "DIII-D")
    arrays = {}
    sub.predict_fold(Dummy(), files, "DIII-D", d3d_fallbacks(test_root, False), arrays.__setitem__, log=lambda m: None)
    path = sub.write_npz(tmp_path / "d.npz", arrays)
    back = sub.read_npz(path)
    assert list(back) == list(arrays)
    for key in arrays:
        assert back[key].dtype == arrays[key].dtype and np.array_equal(back[key], arrays[key])
    full = Dummy().predict_shot(io.load_shot(files[0]), "DIII-D")
    assert np.array_equal(back["shot_0000_psirz"], full.psirz.astype(np.float16))
    assert np.abs(back["shot_0000_psirz"].astype(np.float64) - full.psirz).max() < 1e-3 * np.abs(full.psirz).max()
    pred = sub.read_prediction(back, 0, stride=4)
    assert pred.psirz.shape == (3, 65, 65) and np.array_equal(pred.q95, back["shot_0000_q95"][::4])
    with np.load(path) as f:
        assert f["shot_0001_q95"].shape == (5,)
    again = sub.write_npz(tmp_path / "d2.npz", arrays)
    assert sub.sha256_of_file(path) == sub.sha256_of_file(again)


def test_each_fallback_is_reached_and_counted(test_root):
    path = sub.test_shot_files(test_root, "DIII-D")[0]
    shot = io.load_shot(path)
    n = shot.n_frames

    def run(predictor, fallbacks):
        counts = sub.FallbackCounts()
        pred = sub.apply_fallbacks(predictor.predict_shot(shot, "DIII-D"), shot, fallbacks, counts)
        assert np.isfinite(pred.psirz).all() and np.isfinite(pred.q95).all() and np.isfinite(pred.betaN).all()
        return pred, counts.counts

    # 1. Coil flux plus the mean shape scaled by Ip, DIII-D with resources.
    fb = d3d_fallbacks(test_root, True)
    pred, c = run(Dummy(bad_frames=(2,)), fb)
    assert c["coil_mean_shape"] == 1 and c["nearest_frame"] == 0 and c["constant_map"] == 0
    currents = cf.currents_at_frames(shot, fb.table.columns)
    ip = cf.currents_at_frames(shot, [DIII_D.ip_column])[:, 0]
    expected = targets.stored_from_target(np.abs(ip)[:, None, None] * fb.mean_shape, fb.table.flux(currents, fb.scales), ip)
    assert np.allclose(pred.psirz[2], expected[2]) and not np.allclose(pred.psirz[1], expected[1])
    # 2. Nearest finite frame of the shot when no mean shape exists (MAST, or no resources).
    pred, c = run(Dummy(bad_frames=(0, 2, n - 1)), d3d_fallbacks(test_root, False))
    assert c["nearest_frame"] == 3 and c["coil_mean_shape"] == 0
    assert np.array_equal(pred.psirz[0], pred.psirz[1]) and np.array_equal(pred.psirz[2], pred.psirz[1])
    assert np.array_equal(pred.psirz[n - 1], pred.psirz[n - 2]) and not np.array_equal(pred.psirz[3], pred.psirz[1])
    # 3. A constant map when no frame of the shot is finite.
    pred, c = run(Dummy(all_bad=True), sub.Fallbacks("MAST", dict(MEDIANS), constant=0.0))
    assert c["constant_map"] == n and np.all(pred.psirz == 0.0)
    # Scalars: the nearest finite frame, then the training median.
    pred, c = run(Dummy(bad_q95=(3,), betan_all_nan=True), d3d_fallbacks(test_root, False))
    assert c["scalar_nearest_frame"] == 1 and c["scalar_median"] == n
    assert pred.q95[3] == pred.q95[2] and np.all(pred.betaN == MEDIANS["betaN"])
    # A predictor exception on a shot: recorded, the shot predicted by the fallbacks alone.
    arrays = {}
    record = sub.predict_fold(Dummy(raise_on="a1"), [path], "DIII-D", fb, arrays.__setitem__, log=lambda m: None)
    assert len(record.counts.predictor_errors) == 1 and record.counts.predictor_errors[0]["shot"] == "a1"
    assert record.counts.counts["coil_mean_shape"] == n and record.counts.counts["scalar_median"] == 2 * n
    assert np.isfinite(arrays["shot_0000_psirz"]).all()
    # A wrong shape is a programming error, not a fallback case.
    class Wrong(Dummy):
        def predict_shot(self, shot, machine):
            p = super().predict_shot(shot, machine)
            return bl.Prediction(p.psirz, p.q95[:-1], p.betaN)
    with pytest.raises(ValueError):
        sub.apply_fallbacks(Wrong().predict_shot(shot, "DIII-D"), shot, fb, sub.FallbackCounts())


def test_validator_catches_missing_shot_wrong_shape_nan_and_extra_keys(test_root, tmp_path):
    files = sub.test_shot_files(test_root, "DIII-D")
    arrays = {}
    sub.predict_fold(Dummy(), files, "DIII-D", d3d_fallbacks(test_root, False), arrays.__setitem__, log=lambda m: None)
    good = sub.write_npz(tmp_path / "good.npz", arrays)
    result = sub.validate_npz(good, files, "DIII-D")
    assert result.ok and result.n_shots == 2 and result.n_frames == 17 and result.psi_dtype == "float16"
    assert result.to_dict()["ok"] is True

    missing = {k: v for k, v in arrays.items() if not k.startswith("shot_0001")}
    r = sub.validate_npz(sub.write_npz(tmp_path / "missing.npz", missing), files, "DIII-D")
    assert not r.ok and len(r.errors) == 3 and all("missing" in e for e in r.errors)

    wrong = dict(arrays)
    wrong["shot_0000_psirz"] = arrays["shot_0000_psirz"][:-1]
    r = sub.validate_npz(sub.write_npz(tmp_path / "wrong.npz", wrong), files, "DIII-D")
    assert len(r.errors) == 1 and "shape (11, 65, 65), expected (12, 65, 65)" in r.errors[0]

    nan = dict(arrays)
    nan["shot_0001_psirz"] = arrays["shot_0001_psirz"].copy()
    nan["shot_0001_psirz"][2, 0, 0] = np.nan
    nan["shot_0001_q95"] = arrays["shot_0001_q95"].copy()
    nan["shot_0001_q95"][0] = np.inf
    r = sub.validate_npz(sub.write_npz(tmp_path / "nan.npz", nan), files, "DIII-D")
    assert len(r.errors) == 2 and all("non-finite" in e for e in r.errors)

    extra = dict(arrays)
    extra["shot_0002_psirz"] = arrays["shot_0001_psirz"]
    r = sub.validate_npz(sub.write_npz(tmp_path / "extra.npz", extra), files, "DIII-D")
    assert len(r.errors) == 1 and "unexpected keys" in r.errors[0]

    wrong_dtype = dict(arrays)
    wrong_dtype["shot_0000_psirz"] = arrays["shot_0000_psirz"].astype(np.float32)
    r = sub.validate_npz(sub.write_npz(tmp_path / "dtype.npz", wrong_dtype), files, "DIII-D")
    assert len(r.errors) == 1 and "float16" in r.errors[0]

    # MAST is held to the same finiteness rule.
    m_files = sub.test_shot_files(test_root, "MAST")
    m_arrays = {}
    sub.predict_fold(Dummy(), m_files, "MAST", sub.Fallbacks("MAST", dict(MEDIANS)), m_arrays.__setitem__, log=lambda m: None)
    assert sub.validate_npz(sub.write_npz(tmp_path / "mast.npz", m_arrays), m_files, "MAST").ok


def test_pointer_manifest_has_three_fields_and_holds_the_token_alone(tmp_path):
    sha = "a" * 40
    manifest = sub.manifest_dict("user/preds", sha, "hf_secret_token_xyz")
    assert tuple(manifest) == ("repo_id", "revision", "token")
    zip_path = sub.write_pointer_zip(tmp_path / "run" / sub.POINTER_NAME, manifest)
    with zipfile.ZipFile(zip_path) as z:
        assert z.namelist() == ["manifest.json"]
    assert sub.read_pointer_zip(zip_path) == manifest
    assert [p.name for p in (tmp_path / "run").iterdir()] == [sub.POINTER_NAME]
    for bad in ("abc", "g" * 40):
        with pytest.raises(ValueError):
            sub.manifest_dict("user/preds", bad, "t")
    with pytest.raises(ValueError):
        sub.manifest_dict("user/preds", sha, "")
    with pytest.raises(ValueError):
        sub.write_pointer_zip(tmp_path / "x.zip", {**manifest, "extra": 1})


def test_mast_shot_checks_and_summary(test_root):
    path = sub.test_shot_files(test_root, "MAST")[0]
    shot = io.load_shot(path)
    table = green(MAST, path, MAST_GRID)
    checks = sub.mast_shot_checks(shot, table)
    assert checks["ip_sign"] == 1 and checks["ip_abs_median_ka"] == pytest.approx(800.0)
    assert checks["sol_median_ka"] == pytest.approx(-40.0) and 0 < checks["sol_flux_rel_std"] < np.inf
    assert 0 < checks["sol_range_share_of_coil_range"] <= 1.0
    flipped = io.load_shot(path)
    flipped.data["magnetics_plasma_current"] = -flipped["magnetics_plasma_current"]
    assert sub.mast_shot_checks(flipped, table)["ip_sign"] == -1
    summary = sub.summarize_mast_checks([checks, sub.mast_shot_checks(flipped, table)])
    assert summary["shots_by_ip_sign"] == {"positive": 1, "negative": 1, "mixed_or_absent": 0}
    assert summary["sol_range_share_of_coil_range_quantiles"]["0.5"] == pytest.approx(checks["sol_range_share_of_coil_range"])


def test_check_order_fingerprints_match_only_at_the_same_rank(test_root):
    files = sub.test_shot_files(test_root, "DIII-D")
    # Hub rows are plain dicts of lists; the local side reads the file as shipped.
    rows = [{k: v.tolist() for k, v in io.load_shot(p, sub.FINGERPRINT_COLUMNS, apply=False).data.items()} for p in files]
    fp = sub.shot_fingerprint(rows[0])
    assert fp["n_frames"] == 12 and fp["first_ms"] == 300.0 and fp["last_ms"] == 520.0 and len(fp["sha256"]) == 16
    assert sub.local_fingerprint(files[0]) == fp and sub.local_fingerprint(files[1]) != fp
    # The Ip fix must not enter the local fingerprint: the fixed axis is not what the Hub streams.
    assert sub.shot_fingerprint(io.load_shot(files[0], sub.FINGERPRINT_COLUMNS, apply=True)) == fp
    script = load_script()
    result = script.compare_order(rows, files)
    assert [r["match"] for r in result] == [True, True] and result[1]["local_file"] == "d3d_shot_b2.parquet"
    swapped = script.compare_order(rows[::-1], files)
    assert [r["match"] for r in swapped] == [False, False]
    beyond = script.compare_order(rows + rows, files)
    assert [r["match"] for r in beyond] == [True, True, False, False] and beyond[2]["local"] is None


def test_registry_lists_baseline_v0_and_rejects_unknown_names():
    with pytest.raises(KeyError):
        sub.make_predictor("no_such_predictor")
    assert "baseline_v0" in sub.REGISTRY
    assert isinstance(sub.make_predictor("baseline_v0"), sub.Predictor)


# ---------------------------------------------------------------- baseline_v0 end to end
def load_script():
    spec = importlib.util.spec_from_file_location("make_submission", REPO_ROOT / "scripts" / "make_submission.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def environment(tmp_path, test_root):
    """A synthetic dataset with three labelled training shots, a frame table, a split, Green's
    tables and a baseline cache, laid out as the real ones."""
    root = test_root
    (root / "diii_d_train").mkdir()
    rows = []
    for shot_id, n, sign in (("t1", 6, 1.0), ("t2", 5, -1.0), ("t3", 7, 1.0)):
        raw = make_d3d_shot(n_frames=n)
        for col in DIII_D.current_columns:
            raw[col] = (np.asarray(raw[col], dtype=np.float64) * sign).astype(np.float32)
        path = write_shot(root / "diii_d_train" / f"d3d_shot_{shot_id}.parquet", raw)
        shot = io.load_shot(path, ft.columns_to_load(DIII_D, True, False))
        rows.append(ft.build_shot_frames(shot, "diii_d_train", ft.BuildParams(psi_summary=False), True))
    frames = pd.concat(rows, ignore_index=True)
    table_root = tmp_path / "frame_table"
    table_root.mkdir()
    frames.to_parquet(table_root / "diii_d_train.parquet", index=False)
    green_dir = tmp_path / "green"
    green_dir.mkdir()
    green(DIII_D, root / "diii_d_train" / "d3d_shot_t1.parquet", D3D_GRID).save(green_dir / "green_d3d.npz")
    green(MAST, root / "mast_public_test" / "mast_shot_m1.parquet", MAST_GRID).save(green_dir / "green_mast.npz")
    split_path = tmp_path / "split.json"
    ids = ["t1", "t2", "t3"]
    sp.write_split(split_path, {"mode": "random", "repo_commit": "test", "train": ids, "validation": ["t1", "t2"], "local_test": [],
                                "working_train": ids, "working_validation": ["t1"]})
    cache = tmp_path / "cache"
    cache.mkdir()
    return {"root": root, "table_root": table_root, "green_dir": green_dir, "split": split_path, "cache": cache,
            "frames": frames, "ids": ids}


def common_args(env, tmp_path):
    return ["--submissions-dir", str(tmp_path / "submissions"), "--reports-dir", str(tmp_path / "reports"), "--data-root", str(env["root"]),
            "--split", str(env["split"]), "--cache-dir", str(env["cache"]), "--green-dir", str(env["green_dir"]),
            "--table-root", str(env["table_root"])]


def test_baseline_v0_fits_once_then_loads_and_predicts_both_machines(environment):
    env = environment
    from fusion_eq.predictors import BaselineV0, model_files, training_scalar_medians
    params = bl.BaselineParams(n_components=2, alpha=1.0, pca_frames=None, seed=0)
    predictor = BaselineV0(env["split"], env["cache"], env["green_dir"], env["table_root"], env["root"], params=params)
    ids = predictor.setup(log=lambda m: None)
    files = model_files(env["cache"], env["ids"])
    assert all(files[k].is_file() for k in ("hybrid_ridge", "ridge_signed", "mean_shape"))
    assert ids["models"]["hybrid_ridge"]["sha256"] == sub.sha256_of_file(files["hybrid_ridge"])
    medians = training_scalar_medians(env["split"], env["cache"], env["table_root"])
    f = env["frames"]
    assert medians == {"q95": pytest.approx(float(np.nanmedian(f["efit_q95"]))), "betaN": pytest.approx(float(np.nanmedian(f["efit_beta_n"])))}
    assert ids["scalar_medians"] == medians and ids["coil_model"]["MAST"]["scales"]["flat_wb_per_ka"] == 0.0
    # A second setup loads the saved files: same identifiers, no refit.
    again = BaselineV0(env["split"], env["cache"], env["green_dir"], env["table_root"], env["root"], params=params)
    assert again.setup(log=lambda m: None)["models"] == ids["models"]
    # DIII-D: the hybrid ridge from the shot's own frame rows, equal to the comparison's arithmetic.
    shot = sub.strip_labels(io.load_shot(env["root"] / "diii_d_public_test" / "d3d_shot_a1.parquet"))
    pred = predictor.predict_shot(shot, "DIII-D")
    assert pred.psirz.shape == (12, 65, 65) and np.isfinite(pred.psirz).all() and pred.q95.shape == (12,)
    frames = ft.build_shot_frames(shot, "diii_d_public_test", ft.BuildParams(), False)
    expected = predictor.model_set.baselines()["hybrid_ridge"].predict(frames)
    assert np.array_equal(pred.psirz, expected.psirz) and np.array_equal(pred.q95, expected.q95)
    # MAST: minus sign(Ip) times the calculated coil flux, scalars at the medians.
    m_shot = sub.strip_labels(io.load_shot(env["root"] / "mast_public_test" / "mast_shot_m1.parquet"))
    m_pred = predictor.predict_shot(m_shot, "MAST")
    currents = cf.currents_at_frames(m_shot, predictor.table_mast.columns)
    assert np.allclose(m_pred.psirz, -predictor.table_mast.flux(currents, None), atol=1e-7)
    assert np.all(m_pred.q95 == medians["q95"]) and np.all(m_pred.betaN == medians["betaN"]) and m_pred.psirz.shape == (8, 65, 65)
    m_shot.data["magnetics_plasma_current"] = -m_shot["magnetics_plasma_current"]
    assert np.allclose(predictor.predict_shot(m_shot, "MAST").psirz, -m_pred.psirz)


def test_script_build_validate_score_and_pointer_keep_the_token_in_the_zip(environment, tmp_path, monkeypatch):
    env = environment
    script = load_script()
    args =["--predictor", "baseline_v0", "--name", "smoke", "--date", "2026-01-01", *common_args(env, tmp_path)]
    assert script.main(["build", *args]) == 0
    run = "2026-01-01_smoke"
    sub_dir, report_dir = tmp_path / "submissions" / run, tmp_path / "reports" / run
    assert (sub_dir / "diii_d_public_test.npz").is_file() and (sub_dir / "mast_public_test.npz").is_file()
    record = json.loads((report_dir / "run.json").read_text())
    assert record["predictor"] == "baseline_v0" and record["split_file"] == str(env["split"])
    assert record["folds"]["DIII-D"]["n_shots"] == 2 and record["folds"]["DIII-D"]["n_frames"] == 17
    assert record["folds"]["MAST"]["n_frames"] == 12 and record["folds"]["MAST"]["mast_checks_summary"]["shots_by_ip_sign"]["positive"] == 2
    assert len(record["folds"]["MAST"]["shot_checks"]) == 2 and record["folds"]["MAST"]["shot_checks"][0]["shot"] == "m1"
    assert all(v == 0 for v in record["folds"]["DIII-D"]["fallback_frames"].values())
    assert all(record["validation"][m]["ok"] for m in ("DIII-D", "MAST"))
    assert record["predictor_identifiers"]["models"]["hybrid_ridge"]["sha256"]
    assert script.main(["build", *args]) == 1                                   # refuses to overwrite
    assert script.main(["validate", "--run", run, *common_args(env, tmp_path)]) == 0
    assert script.main(["score-local", "--run", run, "--frame-stride", "1", "--workers", "1", *common_args(env, tmp_path)]) == 0
    record = json.loads((report_dir / "run.json").read_text())
    assert record["score_local"]["n_shots"] == 2 and np.isfinite(record["score_local"]["terms"]["S"])
    assert (sub_dir / "local_validation.npz").is_file()
    # The pointer: token from the environment, written into the zip and nowhere else.
    token = "hf_test_token_0123456789"
    monkeypatch.setenv(script.TOKEN_ENV, token)
    assert script.main(["pointer", "--run", run, "--repo", "user/preds", "--revision", "b" * 40, "--skip-token-check",
                        *common_args(env, tmp_path)]) == 0
    manifest = sub.read_pointer_zip(sub_dir / sub.POINTER_NAME)
    assert manifest == {"repo_id": "user/preds", "revision": "b" * 40, "token": token}
    for path in [*tmp_path.rglob("*")]:
        if path.is_file() and path.name != sub.POINTER_NAME:
            assert token.encode() not in path.read_bytes(), path
    record = json.loads((report_dir / "run.json").read_text())
    assert record["pointer"]["hf_revision"] == "b" * 40 and token not in json.dumps(record)
