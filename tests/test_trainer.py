"""Tests of ``fusion_eq.trainer`` on a synthetic workspace with a stub scorer: the schedule and
early stopping, a short run (loss decreases, checkpoints and run record complete, warm start
equal to the ridge), determinism of a repeated run, arm G, and the guards."""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest
import torch

from fusion_eq import loader as ld, plasma_models as pm, trainer as tr
from fusion_eq.model_config import ModelConfig
from tests.model_fixtures import make_model_workspace

SMALL = {"model.hidden": [32, 32], "model.dropout": 0.0, "train.batch_frames": 64, "train.max_epochs": 6, "train.patience": 3,
         "train.warmup_epochs": 1, "train.lr": 3e-3, "loss.norm_frames": 150}


class StubScorer:
    """Scores predictions on the validation frames by the share of the coefficient variance
    explained (a stand-in for the organizers' scorer)."""

    def __init__(self, frames: ld.FrameSet, bound: float) -> None:
        self.x = np.clip(frames.x, -bound, bound)
        self.c = frames.c.astype(np.float64)
        self.nnr = ~frames.near_repeat
        self.calls = 0

    def __call__(self, predict_c):
        self.calls += 1
        pred = predict_c(self.x)

        def s(mask):
            return {"S": float(1.0 - ((pred[mask] - self.c[mask]) ** 2).mean() / self.c[mask].var())}
        return {"all": s(np.ones(len(self.c), dtype=bool)), "no_near_repeat": s(self.nnr), "ip_negative": {"S": float("nan")},
                "corner": {"S": float("nan")}}


@pytest.fixture(scope="module")
def ws(tmp_path_factory) -> dict:
    return make_model_workspace(tmp_path_factory.mktemp("trainer"), n_train=10, n_val=6, frames_per_shot=24)


@pytest.fixture(scope="module")
def config() -> ModelConfig:
    return ModelConfig.load().with_values(SMALL)


def run(ws, config, tmp_path, run_id="r", scorer=True, **kw) -> tr.TrainResult:
    stub = StubScorer(ld.load_frames("validation", config, ws["paths"]), config.get("loader.clip_bound")) if scorer else None
    return tr.train(config, ("train", "validation"), ws["paths"], run_root=tmp_path / "runs", checkpoint_root=tmp_path / "ckpt",
                    device="cpu", scorer=stub, scorer_record={"kind": "stub"}, run_id=run_id, log=lambda m: None, **kw)


def test_schedule_and_early_stopping():
    assert tr.lr_factor(0, 100, 10) == pytest.approx(0.1)
    assert tr.lr_factor(9, 100, 10) == pytest.approx(1.0)
    assert tr.lr_factor(10, 100, 10) == pytest.approx(1.0)
    assert tr.lr_factor(55, 100, 10) == pytest.approx(0.5)
    assert tr.lr_factor(100, 100, 10) == pytest.approx(0.0)
    stop = tr.EarlyStopping(2)
    assert stop.update(1.0, 0) and not stop.update(1.0, 1) and not stop.stop
    assert not stop.update(2.0, 2) and stop.stop and stop.best_epoch == 0
    np.testing.assert_array_equal(tr.normalization_sample(5, 10, 0), np.arange(5))
    a = tr.normalization_sample(1000, 10, 0)
    assert np.array_equal(a, tr.normalization_sample(1000, 10, 0)) and np.all(np.diff(a) > 0)
    assert tr.select_device("cpu").type == "cpu"


def test_short_run(ws, config, tmp_path):
    result = run(ws, config, tmp_path)
    rec = result.record
    rows = list(csv.DictReader((result.run_dir / "metrics.csv").open()))
    assert len(rows) == len(result.metrics) == rec["early_stopping"]["stopped_epoch"] + 1
    assert {"train_flux", "train_total", "val_flux", "val_total", "val_nnr_total", "lr"} <= set(rows[0])
    train_loss = [float(r["train_total"]) for r in rows]
    assert train_loss[-1] < train_loss[0]
    # The warm start is the ridge on the clipped original training frames.
    frames = ld.load_frames("train", config, ws["paths"])
    w, b = pm.warm_start_ridge(np.clip(frames.x, -10, 10), frames.c, 1.0)
    stub = StubScorer(ld.load_frames("validation", config, ws["paths"]), 10.0)
    ridge_score = stub(lambda x: x @ w + b)
    assert rec["warm_start"]["score"]["all"]["S"] == pytest.approx(ridge_score["all"]["S"], abs=1e-5)
    assert rec["warm_start"]["score"]["all"]["S"] > 0.9
    # The warm-start checkpoint is the ridge itself.
    warm, warm_meta = pm.load_checkpoint(result.checkpoint_dir / "warm.pt")
    assert warm_meta["epoch"] == -1 and rec["checkpoints"]["warm_sha256"] is not None
    np.testing.assert_allclose(pm.predict(warm, np.clip(frames.x, -10, 10).astype(np.float32)), np.clip(frames.x, -10, 10) @ w + b,
                               atol=1e-4)
    # Best checkpoint: its validation loss is the minimum of the epochs and the scorer ran on it.
    best = rec["early_stopping"]["best_epoch"]
    assert rec["best"]["validation_loss"]["subset"]["total"] == pytest.approx(min(float(r["val_nnr_total"]) for r in rows), rel=1e-6)
    assert float(rows[best]["val_nnr_total"]) == pytest.approx(rec["early_stopping"]["best_value"])
    assert (result.checkpoint_dir / "best.pt").is_file() and (result.checkpoint_dir / "last.pt").is_file()
    model, meta = pm.load_checkpoint(result.checkpoint_dir / "best.pt")
    assert meta["epoch"] == best and meta["columns"] == frames.columns and meta["config_hash"] == config.hash()
    assert rec["best"]["score"]["all"]["S"] == pytest.approx(stub(lambda x: pm.predict(model, x))["all"]["S"], abs=1e-9)
    # Run record (Z13).
    saved = json.loads((result.run_dir / "run.json").read_text())
    for key in ("config_hash", "repo_commit", "data_hashes", "seed", "environment", "normalization", "early_stopping", "warm_start",
                "device", "timing", "machine_constants"):
        assert key in saved, key
    assert saved["config_hash"] == config.hash() and saved["seed"] == 0 and saved["device"] == "cpu"
    assert any(line.startswith("torch==") for line in saved["environment"])
    assert saved["data_hashes"]["loss_inputs"] is not None and saved["data_hashes"]["features_v2_scaler"] is not None
    assert saved["normalization"]["constants"]["flux"] > 0 and saved["normalization"]["n_frames"] == 150
    assert saved["early_stopping"]["monitor"].startswith("validation no-near-repeat")
    assert saved["n_frames"]["validation_no_near_repeat"] == int(stub.nnr.sum())


def test_repeated_run_is_identical(ws, config, tmp_path):
    a = run(ws, config, tmp_path, run_id="a")
    b = run(ws, config, tmp_path, run_id="b")
    assert [m["train_total"] for m in a.metrics] == [m["train_total"] for m in b.metrics]
    assert [m["val_total"] for m in a.metrics] == [m["val_total"] for m in b.metrics]
    for k, v in a.model.state_dict().items():
        assert torch.equal(v, b.model.state_dict()[k]), k
    assert a.record["config_hash"] == b.record["config_hash"] and a.record["data_hashes"] == b.record["data_hashes"]
    assert a.record["normalization"] == b.record["normalization"]
    c = run(ws, config.with_values({"train.seed": 1}), tmp_path, run_id="c")
    assert [m["train_total"] for m in c.metrics] != [m["train_total"] for m in a.metrics]


def test_early_stopping_engages(ws, config, tmp_path):
    frozen = config.with_values({"train.lr": 0.0, "train.patience": 1, "train.max_epochs": 10})
    result = run(ws, frozen, tmp_path, scorer=False)
    es = result.record["early_stopping"]
    assert es["stopped_early"] and es["best_epoch"] == 0 and es["stopped_epoch"] == 1
    assert result.record["best"]["score"] is None and "score" not in result.record["warm_start"]


def test_arm_g_runs_with_three_terms(ws, config, tmp_path):
    g = config.with_values({"loss.terms": {**config.get("loss.terms"), "grad": {"enabled": True, "weight": 1.0},
                                           "band": {"enabled": True, "weight": 1.0}}, "train.max_epochs": 2})
    result = run(ws, g, tmp_path, scorer=False)
    assert set(result.record["normalization"]["constants"]) == {"flux", "grad", "band"}
    assert all(v > 0 for v in result.record["normalization"]["constants"].values())
    assert {"train_grad", "val_band", "val_nnr_grad"} <= set(result.metrics[0])


def test_local_test_is_refused(ws, config, tmp_path):
    with pytest.raises(PermissionError):
        tr.train(config, ("train", "local_test"), ws["paths"], run_root=tmp_path, checkpoint_root=tmp_path, device="cpu",
                 log=lambda m: None)
