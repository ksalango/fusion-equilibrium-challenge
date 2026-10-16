"""End-to-end tests of stage 3 on synthetic raw data (the workspace of the feature and target
tests): ``scripts/predict.py`` builds both public-test folds through the submission writer from
the stored tables and from the raw files with byte-identical results, the determinism mode (Z5)
passes, the chain's mean-current maps equal ``mean_current_v0``'s on both machines, and
``scripts/evaluate.py`` writes every section. The checkpoint is a linear model saved directly
(training is covered by ``test_train_end_to_end.py``)."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from fusion_eq import (features as fe, loader as ld, plasma_models as pm, prediction as pr, scalar_heads as sh, starter_kit,
                       submission as sub, target_build as tb, targets)
from fusion_eq.coil_flux import adopted_scales
from fusion_eq.model_config import DEFAULT_BASE, ModelConfig
from tests.test_build_features import REPO_ROOT, make_workspace
from tests.test_target_build import SMALL_CONFIG, build as build_targets

FEATURES_V2 = REPO_ROOT / "configs" / "features_v2.yaml"
RUN_ID = "synthetic_run"
RUN2_ID = "synthetic_rung2"


def run_script(name: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / name), *map(str, args)], capture_output=True, text=True,
                          timeout=600)


@pytest.fixture(scope="module")
def ws(tmp_path_factory) -> dict:
    if not starter_kit.is_available():
        pytest.skip("starter kit not available")
    root = tmp_path_factory.mktemp("predict_e2e")
    ws = make_workspace(root)
    ws["config"] = root / "target.yaml"
    ws["config"].write_text(SMALL_CONFIG)
    assert build_targets(ws, root / "targets").returncode == 0
    ws["targets"] = root / "targets"
    result = run_script("build_features.py", "--config", FEATURES_V2, "--data-root", ws["data"], "--table-dir", ws["table_dir"],
                        "--green-dir", ws["green"], "--split", ws["split"], "--out", root / "features", "--report-root",
                        root / "feature_reports")
    assert result.returncode == 0, result.stderr
    ws["features"] = root / "features"
    ws["scaler"] = root / "feature_reports" / "v2" / "scaler.json"

    # A linear checkpoint: the ridge from the scaled features to the standardized coefficients
    # of the training frames, with a nonzero MLP so that mirror averaging changes the output.
    # Scored at stride 1: the synthetic shots are short, and a permutation of two frames is
    # often the identity.
    config = ModelConfig.load(DEFAULT_BASE).with_values({"model.hidden": [8], "model.dropout": 0.0, "scoring.corner_min_frames": 1,
                                                         "scoring.stride": 1})
    paths = ld.DataPaths(feature_root=ws["features"], scaler=ws["scaler"], target_root=ws["targets"], split=ws["split"])
    fconfig = fe.FeatureConfig.load(FEATURES_V2)
    columns = ld.model_columns(fconfig, "set_a")
    basis, flux = ld.load_bases(paths)
    sigma = ld.component_sigma(basis)
    torch.manual_seed(0)
    model = pm.LinearMLP(len(columns), flux.n_components, hidden=[8], dropout=0.0)
    with torch.no_grad():
        model.linear.weight.mul_(0.1)
        model.output.weight.normal_(0.0, 0.05)
    meta = {"columns": columns, "parity": basis.parity.tolist(), "sigma": sigma.tolist(), "config_hash": config.hash(),
            "config": config.to_dict(), "run_id": RUN_ID, "feature_version": 2, "target_version": 1, "epoch": 1}
    pm.save_checkpoint(root / "ckpt" / RUN_ID / "best.pt", model, meta)
    warm = pm.LinearMLP(len(columns), flux.n_components, hidden=[8], dropout=0.0)
    warm.linear.load_state_dict(model.linear.state_dict())
    pm.save_checkpoint(root / "ckpt" / RUN_ID / "warm.pt", warm, {**meta, "epoch": -1})
    run_dir = root / "runs" / RUN_ID
    run_dir.mkdir(parents=True)
    score = lambda s: {k: {"S": s - 0.01 * j} for j, k in enumerate(("all", "no_near_repeat", "ip_negative", "corner"))}  # noqa: E731
    record = {"run_id": RUN_ID, "seed": 0, "lists": {"train": "train", "validation": "validation"},
              "early_stopping": {"best_epoch": 1, "stopped_epoch": 2, "stopped_early": True},
              "warm_start": {"score": score(0.80)}, "best": {"score": score(0.82)}}
    (run_dir / "run.json").write_text(json.dumps(record))
    with (run_dir / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_total", "val_total", "val_nnr_total", "best"])
        writer.writeheader()
        for e in range(3):
            writer.writerow({"epoch": e, "train_total": 1.0 - 0.1 * e, "val_total": 1.1 - 0.05 * e, "val_nnr_total": 1.2, "best": e == 1})

    # betaN heads on the stored training rows with synthetic shapes (the synthetic maps need not
    # carry a boundary the scorer finds).
    chain, _ = pr.build_chain([RUN_ID], chain_paths(ws, root))
    source = pr.TableSource(ws["table_dir"], ws["features"], 2, {m: chain.frame_columns(m) for m in pr.MACHINES}, chain.feature_columns())
    rows = pr.heads_training_rows(source, ws["split"], "train", 1)
    rng = np.random.default_rng(0)
    shapes = pd.DataFrame({"shot": rows["shot"], "frame": rows["frame"], "a": rng.uniform(0.55, 0.65, len(rows)),
                           "kappa": rng.uniform(1.5, 1.9, len(rows)), "boundary_found": True})
    heads = sh.fit_heads(rows, shapes, (1.0, 10.0), {"note": "synthetic shapes"})
    heads.save(root / "heads.pkl")

    # A rung-2 checkpoint (GRU) with the same skip and a nonzero head, and its run record.
    config2 = config.with_values({"model.name": "gru_seq", "rung2.name": "gru_seq", "rung2.hidden": 8, "rung2.layers": 1,
                                  "rung2.head_hidden": [8]})
    torch.manual_seed(1)
    gru = pm.build_model(config2.get("model"), len(columns), flux.n_components, config2.get("rung2"))
    with torch.no_grad():
        gru.linear.load_state_dict(model.linear.state_dict())
        gru.output.weight.normal_(0.0, 0.05)
    pm.save_checkpoint(root / "ckpt" / RUN2_ID / "best.pt", gru, {**meta, "config": config2.to_dict(), "config_hash": config2.hash(),
                                                                   "run_id": RUN2_ID, "sequence": True, "uses_future": False})
    (root / "runs" / RUN2_ID).mkdir()
    (root / "runs" / RUN2_ID / "run.json").write_text(json.dumps({**record, "run_id": RUN2_ID}))
    ws.update(root=root, heads=root / "heads.pkl", paths=paths, config=config, chain_rows=rows)
    return ws


def chain_paths(ws: dict, root: Path) -> pr.ChainPaths:
    return pr.ChainPaths(table_root=ws["table_dir"], feature_root=root / "features", scaler=root / "feature_reports" / "v2" / "scaler.json",
                         targets_root=root / "targets", green_dir=ws["green"], data_root=ws["data"], checkpoint_root=root / "ckpt")


def predict_args(ws: dict, *extra: str) -> list[str]:
    root = ws["root"]
    return ["--run", RUN_ID, "--heads", ws["heads"], "--data-root", ws["data"], "--split", ws["split"], "--cache-dir", root / "cache",
            "--green-dir", ws["green"], "--table-root", ws["table_dir"], "--feature-root", ws["features"], "--scaler", ws["scaler"],
            "--targets-root", ws["targets"], "--run-root", root / "runs", "--checkpoint-root", root / "ckpt", "--workers", "1",
            "--score-workers", "1", "--submissions-dir", root / "submissions", "--reports-dir", root / "reports", *extra]


def test_predict_builds_both_folds_identically_from_tables_and_raw(ws):
    for source in ("tables", "raw"):
        result = run_script("predict.py", *predict_args(ws, "--source", source, "--name", source, "--date", "d",
                                                        *([] if source == "tables" else ["--no-score-local"])))
        assert result.returncode == 0, result.stderr + result.stdout
    root = ws["root"]
    for config in sub.TEST_CONFIGS.values():
        a, b = root / "submissions" / "d_tables" / f"{config}.npz", root / "submissions" / "d_raw" / f"{config}.npz"
        assert sub.sha256_of_file(a) == sub.sha256_of_file(b)
    record = json.loads((root / "reports" / "d_tables" / "run.json").read_text())
    assert all(v["ok"] for v in record["validation"].values()) and record["folds"]["MAST"]["mast_sign_asserted"] == 1.0
    # The fold writer replaces a failing shot by fallbacks; every shot must come from the model.
    for name in ("d_tables", "d_raw"):
        r = json.loads((root / "reports" / name / "run.json").read_text())
        assert all(f["n_predictor_errors"] == 0 for f in r["folds"].values()), r["folds"]
        assert all(f["fallback_frames"]["constant_map"] == f["fallback_frames"]["nearest_frame"] == 0 for f in r["folds"].values())
    assert record["score_local"]["fold"]["n_predictor_errors"] == 0
    ident = record["predictor_identifiers"]
    assert ident["source"] == "tables" and ident["mirror_average"] is True and ident["mast_sign_asserted"] is True
    assert ident["runs"] == [RUN_ID] and len(ident["checkpoints"][RUN_ID]["sha256"]) == 64
    assert record["heads"]["sha256"] == pr.sha256_of(ws["heads"]) and record["heads"]["summary"]["note"] == "synthetic shapes"
    assert set(record["score_local"]["terms"]) >= {"S", "r2_psi", "r2_qb", "consistency"}
    raw_record = json.loads((root / "reports" / "d_raw" / "run.json").read_text())
    assert "score_local" not in raw_record
    assert raw_record["predictor_identifiers"]["frame_build_params"]["params"]["align_ip"] is True

    # The written MAST maps are the chain's maps cast to float16.
    arrays = sub.read_npz(root / "submissions" / "d_tables" / "mast_public_test.npz")
    chain, _ = pr.build_chain([RUN_ID], chain_paths(ws, root), heads=sh.BetanHeads.load(ws["heads"]))
    source = pr.TableSource(ws["table_dir"], ws["features"], 2, {m: chain.frame_columns(m) for m in pr.MACHINES}, chain.feature_columns())
    pred = chain.predict(source.rows("mast_public_test", "m1"), "MAST")
    assert arrays["shot_0000_psirz"].dtype == np.float16
    np.testing.assert_array_equal(arrays["shot_0000_psirz"], pred.psi.astype(np.float32).astype(np.float16))
    assert pred.shapes is not None and len(pred.shapes) == len(pred.psi)


def test_determinism_mode_passes_and_detects_a_difference(ws, tmp_path):
    shots = tmp_path / "shots.json"
    shots.write_text(json.dumps({"diii_d_train": ["a1", "a2", "a3"], "mast_public_test": ["m1"]}))
    result = run_script("predict.py", *predict_args(ws, "--determinism", "--shots-file", shots, "--report-dir", tmp_path / "det"))
    assert result.returncode == 0, result.stderr + result.stdout
    info = json.loads((tmp_path / "det" / "prediction_check.json").read_text())
    assert info["pass"] and info["n_shots"] == 4 and set(info["results"]) == {"diii_d_train", "mast_public_test"}
    assert all(r["repeat"]["identical"] and r["raw_vs_tables"]["identical"] for r in info["results"].values())
    assert "pass" in (tmp_path / "det" / "prediction_check.md").read_text()
    spec = __import__("importlib.util").util.spec_from_file_location("predict_script", REPO_ROOT / "scripts" / "predict.py")
    module = __import__("importlib.util").util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sub.write_npz(tmp_path / "x.npz", {"shot_0000_q95": np.array([1.0], np.float32)})
    sub.write_npz(tmp_path / "y.npz", {"shot_0000_q95": np.array([1.0 + 1e-6], np.float32)})
    assert module.compare_npz(tmp_path / "x.npz", tmp_path / "y.npz")["differing"] == ["shot_0000_q95"]
    # The local_test shot of the D7 subset is left out.
    args = argparse.Namespace(shots_file=None, split=REPO_ROOT / "splits" / "split_v1.json")
    chosen, left_out = module.d7_subset(args)
    assert left_out == ["45027c603f"] and ("diii_d_train", "45027c603f") not in chosen and len(chosen) == 24


def test_mean_current_maps_of_the_chain_equal_mean_current_v0(ws):
    chain, record = pr.build_chain([RUN_ID], chain_paths(ws, ws["root"]))
    objects = tb.TargetObjects.load(tb.target_dir(1, ws["targets"]))
    source = pr.TableSource(ws["table_dir"], ws["features"], 2, {m: chain.frame_columns(m) for m in pr.MACHINES}, chain.feature_columns())
    for split, shot in (("diii_d_train", "a1"), ("diii_d_public_test", "t1"), ("mast_public_test", "m1")):
        machine = "MAST" if split.startswith("mast") else "DIII-D"
        rows = source.rows(split, shot)
        got = chain.predict(rows, machine, with_scalars=False, c_std=np.zeros((len(rows), chain.coefficients.sigma.size)))
        table = chain.composer.tables[machine]
        expected = tb.mean_current_maps(machine, rows, table, adopted_scales(machine, table.columns), objects)
        np.testing.assert_array_equal(got.psi, expected)
    assert record["sign_rule_c"] == {"DIII-D": -1.0, "MAST": 1.0} and record["q95"]["tf_turns"] == {"DIII-D": 144, "MAST": 24}


def test_evaluate_writes_every_section(ws):
    root = ws["root"]
    result = run_script("evaluate.py", "--run", RUN_ID, "--run-root", root / "runs", "--checkpoint-root", root / "ckpt",
                        "--split", ws["split"], "--data-root", ws["data"], "--table-root", ws["table_dir"], "--feature-root",
                        ws["features"], "--scaler", ws["scaler"], "--targets-root", ws["targets"], "--green-dir", ws["green"],
                        "--references-dir", root / "refs", "--workers", "1", "--scalar-workers", "1", "--mast-shot-step", "1",
                        "--heads-stride", "1", "--seed-runs", RUN2_ID)
    assert result.returncode == 0, result.stderr + result.stdout
    text = (root / "runs" / RUN_ID / "eval.md").read_text()
    for heading in ("## Checks", "## Score terms by slice", "## Score terms by phase", "## Worst 20 validation shots by R2 psi",
                    "## Overfit panel", "## Warm start beside the trained model", "## Clip rates",
                    "## Grouped permutation importance", "## MAST sensitivity (label-free)"):
        assert heading in text, heading
    assert text.count("MAST floor (F100") >= 4
    out = json.loads((root / "runs" / RUN_ID / "eval.json").read_text())
    assert {c["id"]: c["status"] for c in out["checks"]}["Z7"] == "pass"
    assert set(out["scores"]) >= {"all", "no_near_repeat", "ip_negative", "corner", "phase_rising", "phase_flat_top", "phase_late"}
    if np.isfinite(out["scores"]["all"]["S"]):
        assert abs(out["identity"]["all"]["difference"]) < 1e-5
    assert [w["shot"] for w in out["worst_20"]] == ["a2"]
    assert set(out["permutation"]) == {"coil_pattern", "plasma_content", "field_current", "shot_position"}
    assert out["mast_sensitivity"]["n_shots"] == 1 and set(out["mast_sensitivity"]["groups"]) == set(out["permutation"])
    assert out["eval_set"]["n_frames"] == 5 and out["mast_sensitivity"]["n_frames"] == 8
    # A permuted group changes the MAST maps exactly when its columns vary across the frames
    # (the synthetic MAST shot holds most inputs constant).
    mast = fe.read_features("mast_public_test", 2, root=ws["features"])
    for group, g in out["mast_sensitivity"]["groups"].items():
        varies = bool((mast[g["columns"]].nunique() > 1).any())
        assert (g["map_rms_change_relative"] > 0) == varies, group
    assert out["mast_sensitivity"]["groups"]["shot_position"]["map_rms_change_relative"] > 0
    assert any(abs(g["drop"]["r2_psi"]) > 0 for g in out["permutation"].values())
    assert out["mirror"]["on"]["S"] != out["mirror"]["off"]["S"]
    assert out["warm_start"]["z4_trained_at_or_above"] is True and out["overfit"]["seed_spread"]["n"] == 2
    seeds = out["overfit"]["seeds"]
    assert [x["run_id"] for x in seeds] == [RUN_ID, RUN2_ID] and seeds[0]["S"] == out["scores"]["all"]["S"]
    assert np.isfinite(seeds[1]["S"]) and seeds[1]["S"] != seeds[0]["S"]
    assert out["overfit"]["trained_gap"]["all"] == out["scores"]["all"]["S"]
    assert set(out["clip_rates"]) == {"DIII-D validation", "DIII-D test", "MAST test"}
    assert out["heads"]["mast_fitted"] is False
    wc = out["warm_chain"]
    assert wc["available"] and np.isfinite(wc["all"]["warm_start"]) and wc["all"]["trained"] == out["scores"]["all"]["S"]
    assert "Prediction chain (warm-start checkpoint" in text


def load_predict_script():
    import importlib.util
    spec = importlib.util.spec_from_file_location("predict_script", REPO_ROOT / "scripts" / "predict.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_predict_fits_the_heads_with_the_stage2_refit(ws, tmp_path, monkeypatch):
    from fusion_eq import baseline as bl, split as sp
    module = load_predict_script()
    args = module.build_parser().parse_args([*map(str, predict_args(ws, "--refit-stride", "1", "--min-corner-frames", "1",
                                                                    "--shape-dir", tmp_path / "shapes"))])
    rng = np.random.default_rng(1)

    def fake_shapes(chain, rows, machine="DIII-D"):
        return pd.DataFrame({"shot": rows["shot"].astype(str), "frame": rows["frame"], "a": rng.uniform(0.5, 0.7, len(rows)),
                             "kappa": rng.uniform(1.4, 2.0, len(rows)), "boundary_found": True})

    monkeypatch.setattr(pr, "model_shapes", fake_shapes)
    lists = {"train": "train", "validation": "validation"}
    # Without MAST, only the DIII-D head is fitted and no shape is computed.
    d3d_only, info = module.fit_heads(args, lists, ["DIII-D"])
    assert d3d_only.mast_correction is None and info["z10"]["status"].startswith("not run")
    # With MAST and no study shape table, the head is fitted on the model's shapes directly.
    heads, info = module.fit_heads(args, lists, ["DIII-D", "MAST"])
    assert info["z10"]["before"] is None and "not found" in info["z10"]["before_note"]
    assert np.isfinite(info["z10"]["after"]["r2_all"]) and info["z10"]["after"]["n_all"] == 5
    # With the study's hybrid_ridge shape table, the refit starts from the head fitted on it.
    train_ids = sorted(sp.read_split(ws["split"])["train"])
    h = bl.hash_of_ids(train_ids)
    (tmp_path / "shapes").mkdir()
    study = fake_shapes(None, ws["chain_rows"])
    study.to_parquet(tmp_path / "shapes" / f"shapes_hybrid{h}_train_{h}_s1.parquet", index=False)
    refit, info = module.fit_heads(args, lists, ["DIII-D", "MAST"])
    assert info["z10"]["before"] is not None and np.isfinite(info["z10"]["before"]["r2_all"])
    assert refit.record["shapes"].startswith("hybrid_ridge") and refit.record["stage2_refit"]["constant_before"] != refit.mast_constant
    np.testing.assert_array_equal(refit.predict("DIII-D", ws["chain_rows"]), d3d_only.predict("DIII-D", ws["chain_rows"]))


def test_bulk_prediction_over_a_pool_equals_the_per_shot_chain(ws):
    heads = sh.BetanHeads.load(ws["heads"])
    chain, _ = pr.build_chain([RUN_ID], chain_paths(ws, ws["root"]), heads=heads, workers=2)
    source = pr.TableSource(ws["table_dir"], ws["features"], 2, {m: chain.frame_columns(m) for m in pr.MACHINES}, chain.feature_columns())
    try:
        for split, ids, machine in (("diii_d_train", ["a1", "a2", "a3"], "DIII-D"), ("mast_public_test", ["m1"], "MAST")):
            rows = pr.read_rows(source, split, ids)
            bulk = pr.predict_shots(chain, rows, machine)
            assert list(bulk) == ids
            for shot, g in rows.groupby("shot", sort=False):
                single = chain.predict(g.reset_index(drop=True), machine)
                np.testing.assert_array_equal(bulk[shot].psi, single.psi)
                np.testing.assert_array_equal(bulk[shot].q95, single.q95)
                np.testing.assert_array_equal(bulk[shot].beta_n, single.beta_n)
            shapes = pr.model_shapes(chain, rows, machine)
            assert len(shapes) == len(rows) and list(shapes.columns)[:2] == ["shot", "frame"]
    finally:
        chain.close()


def test_rung_two_through_the_chain_predict_and_evaluate(ws, tmp_path):
    root = ws["root"]
    chain, record = pr.build_chain([RUN2_ID], chain_paths(ws, root), heads=sh.BetanHeads.load(ws["heads"]))
    assert chain.coefficients.sequence and "time_ms" in chain.frame_columns("MAST")
    source = pr.TableSource(ws["table_dir"], ws["features"], 2, {m: chain.frame_columns(m) for m in pr.MACHINES}, chain.feature_columns())
    # Strided rows of a sequence model: coefficients from the whole shot, read at the scored frames.
    full = pr.read_rows(source, "diii_d_train", ["a1", "a3"])
    marked = pr.read_rows(source, "diii_d_train", ["a1", "a3"], stride=4, mark=True)
    assert len(marked) == len(full) and marked["scored"].sum() == len(pr.read_rows(source, "diii_d_train", ["a1", "a3"], stride=4))
    bulk = pr.predict_shots(chain, marked, "DIII-D")
    for shot, g in full.groupby("shot", sort=False):
        whole = chain.predict(g.reset_index(drop=True), "DIII-D", with_scalars=False)
        np.testing.assert_array_equal(bulk[shot].psi, whole.psi[::4])
        np.testing.assert_array_equal(bulk[shot].c_std, whole.c_std[::4])
    # Seen alone, the strided frames give other coefficients: the time steps and the history differ.
    alone = chain.coefficients_of(pr.read_rows(source, "diii_d_train", ["a1"], stride=4), "DIII-D")
    assert not np.allclose(alone, bulk["a1"].c_std)
    # rung1_v1 refuses a sequence checkpoint, and rung2_v1 a frame one.
    with pytest.raises(ValueError, match="rung2_v1"):
        pr.Rung1V1(runs=[RUN2_ID], heads=ws["heads"], **predictor_paths(ws)).setup(lambda m: None)
    with pytest.raises(ValueError, match="rung1_v1"):
        pr.Rung2V1(runs=[RUN_ID], heads=ws["heads"], **predictor_paths(ws)).setup(lambda m: None)

    for source_name in ("tables", "raw"):
        args = predict_args(ws, "--predictor", "rung2_v1", "--source", source_name, "--name", f"r2{source_name}", "--date", "d",
                            "--no-score-local")
        args[args.index(RUN_ID)] = RUN2_ID
        result = run_script("predict.py", *args)
        assert result.returncode == 0, result.stderr + result.stdout
    for config in sub.TEST_CONFIGS.values():
        assert sub.sha256_of_file(root / "submissions" / "d_r2tables" / f"{config}.npz") == \
            sub.sha256_of_file(root / "submissions" / "d_r2raw" / f"{config}.npz")
    rec = json.loads((root / "reports" / "d_r2tables" / "run.json").read_text())
    assert rec["predictor"] == "rung2_v1" and all(f["n_predictor_errors"] == 0 for f in rec["folds"].values())

    result = run_script("evaluate.py", "--run", RUN2_ID, "--run-root", root / "runs", "--checkpoint-root", root / "ckpt",
                        "--split", ws["split"], "--data-root", ws["data"], "--table-root", ws["table_dir"], "--feature-root",
                        ws["features"], "--scaler", ws["scaler"], "--targets-root", ws["targets"], "--green-dir", ws["green"],
                        "--references-dir", root / "refs", "--workers", "1", "--scalar-workers", "1", "--mast-shot-step", "1",
                        "--heads-stride", "1", "--no-permutation")
    assert result.returncode == 0, result.stderr + result.stdout
    out = json.loads((root / "runs" / RUN2_ID / "eval.json").read_text())
    assert out["eval_set"]["n_frames"] == 5 and out["mast_sensitivity"]["n_frames"] == 8 and np.isfinite(out["scores"]["all"]["S"])


def predictor_paths(ws: dict) -> dict:
    root = ws["root"]
    return {"split_path": ws["split"], "green_dir": ws["green"], "table_root": ws["table_dir"], "data_root": ws["data"],
            "feature_root": ws["features"], "scaler": ws["scaler"], "targets_root": ws["targets"], "checkpoint_root": root / "ckpt",
            "source": "tables"}


# --------------------------------------------------------------------- MAST coil models
@pytest.fixture(scope="module")
def unit_mast(ws) -> Path:
    """A variant build of the MAST features under the unit coil model (the workspace's version-2
    build uses each machine's default, MAST ``sol_half``); returns its folder."""
    root = ws["root"]
    result = run_script("build_features.py", "--config", FEATURES_V2, "--data-root", ws["data"], "--table-dir", ws["table_dir"],
                        "--green-dir", ws["green"], "--split", ws["split"], "--out", ws["features"], "--report-root",
                        root / "feature_reports", "--splits", "mast_public_test", "--coil-model", "MAST=unit", "--variant", "unit")
    assert result.returncode == 0, result.stderr
    folder = fe.features_dir(2, ws["features"], "unit")
    assert fe.build_coil_model(folder, "MAST") == "unit" and fe.build_coil_model(fe.features_dir(2, ws["features"]), "MAST") == "sol_half"
    return folder


def test_the_chain_takes_the_coil_model_of_each_feature_build(ws, unit_mast):
    from dataclasses import replace
    root = ws["root"]
    paths = chain_paths(ws, root)
    half, rec_half = pr.build_chain([RUN_ID], paths)
    unit, rec_unit = pr.build_chain([RUN_ID], replace(paths, feature_dirs={"MAST": unit_mast}))
    assert rec_half["coil_models"] == {"DIII-D": "solenoid_flat", "MAST": "sol_half"} == rec_half["feature_coil_models"]
    assert rec_unit["coil_models"] == {"DIII-D": "solenoid_flat", "MAST": "unit"} == rec_unit["feature_coil_models"]
    assert rec_unit["coil_model_sources"]["MAST"] == "feature build record" and rec_unit["feature_dirs"]["MAST"] == str(unit_mast)
    assert rec_unit["coil_scales"]["MAST"]["name"] == "unit" and rec_half["coil_scales"]["MAST"]["scales"]["magnetics_sol_current"] == 0.5

    columns = {m: half.frame_columns(m) for m in pr.MACHINES}
    rows_half = pr.TableSource(ws["table_dir"], ws["features"], 2, columns, half.feature_columns()).rows("mast_public_test", "m1")
    source_unit = pr.TableSource(ws["table_dir"], ws["features"], 2, columns, half.feature_columns(), {"MAST": unit_mast})
    rows_unit = source_unit.rows("mast_public_test", "m1")
    assert source_unit.feature_dir("MAST") == unit_mast and source_unit.feature_dir("DIII-D") == fe.features_dir(2, ws["features"])
    coil_pt = [c for c in rows_half.columns if c.startswith("coil_pt_")]
    assert coil_pt and not np.allclose(rows_half[coil_pt].to_numpy(), rows_unit[coil_pt].to_numpy())
    # With the plasma coefficients at zero the two maps differ by half the solenoid's flux.
    zero = np.zeros((len(rows_half), half.coefficients.sigma.size))
    psi_half = half.predict(rows_half, "MAST", with_scalars=False, c_std=zero).psi
    psi_unit = unit.predict(rows_unit, "MAST", with_scalars=False, c_std=zero).psi
    table = half.composer.tables["MAST"]
    sol = list(table.columns).index("magnetics_sol_current")
    currents = np.zeros((len(rows_half), len(table.columns)))
    currents[:, sol] = np.nan_to_num(rows_half["sol__mean"].to_numpy(dtype=np.float64))
    expected = targets.to_stored(0.5 * table.flux(currents), rows_half["ip__mean"].to_numpy(dtype=np.float64), "MAST")
    assert np.abs(expected).max() > 0
    np.testing.assert_allclose(psi_unit - psi_half, expected, rtol=1e-9, atol=1e-12 * np.abs(psi_half).max())
    # DIII-D is untouched by the MAST folder.
    rows_d = pr.TableSource(ws["table_dir"], ws["features"], 2, columns, half.feature_columns()).rows("diii_d_public_test", "t1")
    np.testing.assert_array_equal(half.predict(rows_d, "DIII-D", with_scalars=False).psi, unit.predict(rows_d, "DIII-D", with_scalars=False).psi)

    # A named model that differs from the feature build is refused unless the mismatch is allowed.
    with pytest.raises(ValueError, match="differs from 'unit'"):
        pr.build_chain([RUN_ID], replace(paths, feature_dirs={"MAST": unit_mast}), coil_models={"MAST": "sol_half"})
    _, rec = pr.build_chain([RUN_ID], replace(paths, feature_dirs={"MAST": unit_mast}), coil_models={"MAST": "sol_half"},
                            allow_coil_model_mismatch=True)
    assert rec["coil_models"]["MAST"] == "sol_half" and rec["feature_coil_models"]["MAST"] == "unit"
    assert "differs" in rec["coil_model_sources"]["MAST"]


def test_predict_on_the_variant_mast_features_records_the_coil_model(ws, unit_mast, tmp_path):
    root = ws["root"]
    result = run_script("predict.py", *predict_args(ws, "--machines", "mast", "--mast-feature-dir", unit_mast, "--name", "unit",
                                                    "--date", "d", "--no-score-local", "--max-shots", "1"))
    assert result.returncode == 0, result.stderr + result.stdout
    record = json.loads((root / "reports" / "d_unit" / "run.json").read_text())
    assert record["coil_models"] == {"DIII-D": "solenoid_flat", "MAST": "unit"}
    assert record["feature_dirs"]["MAST"] == str(unit_mast) and record["coil_model_sources"]["MAST"] == "feature build record"
    assert record["predictor_identifiers"]["coil_scales"]["MAST"]["name"] == "unit"
    assert record["folds"]["MAST"]["n_predictor_errors"] == 0
    # A MAST model that differs from the folder's record stops the build.
    result = run_script("predict.py", *predict_args(ws, "--machines", "mast", "--mast-feature-dir", unit_mast, "--mast-coil-model",
                                                    "sol_half", "--name", "mismatch", "--date", "d", "--no-score-local"))
    assert result.returncode != 0 and "differs from 'unit'" in result.stderr
    # Z5 on the variant path: tables twice and raw, byte-identical, under the recorded model.
    shots = tmp_path / "shots.json"
    shots.write_text(json.dumps({"diii_d_train": ["a1"], "mast_public_test": ["m1"]}))
    result = run_script("predict.py", *predict_args(ws, "--determinism", "--shots-file", shots, "--report-dir", tmp_path / "det",
                                                    "--mast-feature-dir", unit_mast))
    assert result.returncode == 0, result.stderr + result.stdout
    info = json.loads((tmp_path / "det" / "prediction_check.json").read_text())
    assert info["pass"] and info["coil_models"]["MAST"] == "unit" and info["feature_dirs"]["MAST"] == str(unit_mast)
    assert info["sources"]["raw"]["feature_coil_models"]["MAST"] == "unit"
