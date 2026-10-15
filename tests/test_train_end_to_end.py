"""End-to-end test of the stage-2 chain on synthetic raw data (the workspace of the feature and
target tests): target build, feature build (version 2), loss inputs, the ``baseline_v0`` scalar
ridge, then ``scripts/train.py`` with the organizers' scorer on the composed validation maps.
Also checks that the scorer's map for the mean current equals ``mean_current_v0``'s map."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from fusion_eq import baseline as bl, loader as ld, model_scoring as ms, starter_kit, target_build as tb
from fusion_eq.coil_flux import GreenTable, adopted_scales
from fusion_eq.model_config import DEFAULT_BASE, ModelConfig
from tests.test_build_features import REPO_ROOT, make_workspace
from tests.test_loss_inputs import run_build as build_loss_inputs
from tests.test_target_build import SMALL_CONFIG, build as build_targets

SMALL_MODEL = f"""base: {DEFAULT_BASE}
model: {{hidden: [16], dropout: 0.0}}
train: {{batch_frames: 16, max_epochs: 3, patience: 2, warmup_epochs: 0}}
loss: {{norm_frames: 50}}
"""


@pytest.fixture(scope="module")
def ws(tmp_path_factory) -> dict:
    if not starter_kit.is_available():
        pytest.skip("starter kit not available")
    root = tmp_path_factory.mktemp("train_e2e")
    ws = make_workspace(root)
    ws["config"] = root / "target.yaml"
    ws["config"].write_text(SMALL_CONFIG)
    assert build_targets(ws, root / "targets").returncode == 0
    ws["targets"] = root / "targets"
    cmd = [sys.executable, str(REPO_ROOT / "scripts" / "build_features.py"), "--config", str(REPO_ROOT / "configs" / "features_v2.yaml"),
           "--splits", "diii_d_train", "--data-root", str(ws["data"]), "--table-dir", str(ws["table_dir"]), "--green-dir", str(ws["green"]),
           "--split", str(ws["split"]), "--out", str(root / "features"), "--report-root", str(root / "feature_reports")]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stderr
    result = build_loss_inputs(ws, root / "loss", "--lists", "train", "validation")
    assert result.returncode == 0, result.stderr
    from fusion_eq.predictors import BaselineV0
    BaselineV0(split_path=ws["split"], cache_dir=root / "cache", green_dir=ws["green"], table_root=ws["table_dir"],
               data_root=ws["data"]).setup(log=lambda m: None)
    ws["paths"] = ld.DataPaths(feature_root=root / "features", scaler=root / "feature_reports" / "v2" / "scaler.json",
                               target_root=root / "targets", loss_root=root / "loss", split=ws["split"])
    ws["scoring"] = ms.ScoringPaths(data_root=ws["data"], table_dir=ws["table_dir"], green_dir=ws["green"],
                                    references_dir=root / "refs", baseline_cache=root / "cache")
    ws["model_config"] = root / "small.yaml"
    ws["model_config"].write_text(SMALL_MODEL)
    return ws


def test_scorer_map_of_the_mean_current_is_mean_current_v0(ws):
    config = ModelConfig.load(ws["model_config"])
    scorer = ms.ValidationScorer("validation", "train", config, ws["paths"], ws["scoring"], workers=1, log=lambda m: None)
    objects = tb.TargetObjects.load(tb.target_dir(1, ws["targets"]))
    table = GreenTable.load(ws["green"] / "green_d3d.npz")
    for shot, s in scorer.shots.items():
        pred = scorer.maps(shot, np.zeros((len(s.rows), scorer.flux.n_components)))
        expected = tb.mean_current_maps("DIII-D", s.rows, table, adopted_scales("DIII-D", table.columns), objects)
        np.testing.assert_array_equal(pred.psirz, expected.astype(np.float32))
        assert len(s.rows) == len(bl.load_truth(shot, ws["data"], "diii_d_train", scorer.stride)["psi"])
    record = scorer.record()
    assert record["stride"] == 4 and record["n_shots"] == 1 and record["scalars"]["model"] == "ridge_signed"
    scores = scorer(lambda x: np.zeros((len(x), scorer.flux.n_components)))
    assert set(scores) == set(ms.SLICES) and np.isfinite(scores["all"]["S"])
    assert {"r2_psi", "consistency", "one_minus_dlcfs", "r2_qb"} <= set(scores["all"])


def test_train_script_end_to_end(ws, tmp_path):
    root = Path(ws["targets"]).parent
    cmd = [sys.executable, str(REPO_ROOT / "scripts" / "train.py"), "--config", str(ws["model_config"]), "--lists", "full",
           "--split", str(ws["split"]), "--feature-root", str(root / "features"),
           "--scaler", str(root / "feature_reports" / "v2" / "scaler.json"), "--targets-root", str(ws["targets"]),
           "--loss-root", str(root / "loss"), "--data-root", str(ws["data"]), "--table-dir", str(ws["table_dir"]),
           "--green-dir", str(ws["green"]), "--references-dir", str(root / "refs"), "--baseline-cache", str(root / "cache"),
           "--run-root", str(tmp_path / "runs"), "--checkpoint-root", str(tmp_path / "ckpt"), "--device", "cpu", "--workers", "1",
           "--run-id", "e2e"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stderr + result.stdout
    run = json.loads((tmp_path / "runs" / "e2e" / "run.json").read_text())
    for when in (run["warm_start"]["score"], run["best"]["score"]):
        assert set(when) == set(ms.SLICES) and np.isfinite(when["all"]["S"]) and np.isfinite(when["no_near_repeat"]["S"])
    assert run["scorer"]["scalars"]["model"] == "ridge_signed" and run["lists"] == {"train": "train", "validation": "validation"}
    assert run["data_hashes"]["targets_v1"] is not None and run["data_hashes"]["loss_inputs"] is not None
    assert (tmp_path / "runs" / "e2e" / "metrics.csv").is_file() and (tmp_path / "ckpt" / "e2e" / "best.pt").is_file()


def test_scorer_runs_a_sequence_model_on_whole_shots(ws, monkeypatch):
    config = ModelConfig.load(ws["model_config"])
    scorer = ms.ValidationScorer("validation", "train", config, ws["paths"], ws["scoring"], workers=1, log=lambda m: None)
    k = scorer.flux.n_components
    seen, live = {}, {}

    def sequence_fn(x, dt):
        seen[len(seen)] = (x.shape, dt.copy())
        c = np.zeros((len(x), k))
        c[:, 0] = np.arange(len(x))            # the frame position, to see which outputs are scored
        return c

    original = scorer.maps

    def maps(shot, c):
        live[shot] = c
        return original(shot, np.zeros_like(c))

    monkeypatch.setattr(scorer, "maps", maps)
    seq = scorer(sequence_fn, sequence=True)
    captured = dict(live)
    frame = scorer(lambda x: np.zeros((len(x), k)))
    assert seq["all"]["S"] == frame["all"]["S"]
    for shot, s in scorer.shots.items():
        n_all = len(bl.load_truth(shot, ws["data"], "diii_d_train", 1)["psi"])
        assert len(s.x_full) == len(s.dt_full) == n_all and s.dt_full[0] == 0.0 and (s.dt_full[1:] > 0).all()
        np.testing.assert_array_equal(captured[shot][:, 0], np.arange(n_all)[::scorer.stride])
        np.testing.assert_array_equal(s.x, s.x_full[::scorer.stride])


def test_train_script_runs_rung_two_with_the_scorer(ws, tmp_path):
    root = Path(ws["targets"]).parent
    config = tmp_path / "rung2.yaml"
    config.write_text(f"""base: {ws['model_config']}
model: {{name: gru_seq}}
rung2: {{name: gru_seq, hidden: 8, layers: 1, head_hidden: [8], batch_shots: 2}}
""")
    cmd = [sys.executable, str(REPO_ROOT / "scripts" / "train.py"), "--config", str(config), "--lists", "full", "--seed", "1",
           "--split", str(ws["split"]), "--feature-root", str(root / "features"),
           "--scaler", str(root / "feature_reports" / "v2" / "scaler.json"), "--targets-root", str(ws["targets"]),
           "--loss-root", str(root / "loss"), "--data-root", str(ws["data"]), "--table-dir", str(ws["table_dir"]),
           "--green-dir", str(ws["green"]), "--references-dir", str(root / "refs"), "--baseline-cache", str(root / "cache"),
           "--run-root", str(tmp_path / "runs"), "--checkpoint-root", str(tmp_path / "ckpt"), "--device", "cpu", "--workers", "1",
           "--run-id", "e2e_rung2"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stderr + result.stdout
    run = json.loads((tmp_path / "runs" / "e2e_rung2" / "run.json").read_text())
    assert run["model"]["rung"] == 2 and run["seed"] == 1 and run["config"]["train"]["seed"] == 1
    assert np.isfinite(run["warm_start"]["score"]["all"]["S"]) and np.isfinite(run["best"]["score"]["all"]["S"])
