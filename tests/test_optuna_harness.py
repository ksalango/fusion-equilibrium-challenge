"""Tests of ``fusion_eq.optuna_harness`` (check Z14): a 2-trial study on the synthetic working
lists completes, logs both trials with their parameters and slice scores, resumes from its
database, and never reads ``local_test``."""

from __future__ import annotations

import json

import optuna
import pytest

from fusion_eq import loader as ld, loss_inputs as li, optuna_harness as oh
from fusion_eq.model_config import ModelConfig
from tests.model_fixtures import make_model_workspace
from tests.test_trainer import StubScorer

SPACE = {"train.lr": {"type": "log_float", "low": 1e-4, "high": 3e-3}, "model.dropout": {"type": "float", "low": 0.0, "high": 0.2},
         "model.depth": {"type": "int", "low": 1, "high": 2}, "model.width": {"type": "categorical", "choices": [16, 32]}}


@pytest.fixture(scope="module")
def ws(tmp_path_factory) -> dict:
    return make_model_workspace(tmp_path_factory.mktemp("optuna"), n_train=8, n_val=6, frames_per_shot=16)


@pytest.fixture(scope="module")
def config() -> ModelConfig:
    base = ModelConfig.load()
    return base.with_values({"train.batch_frames": 64, "train.warmup_epochs": 0, "train.patience": 2, "loss.norm_frames": 100,
                             "optuna.max_epochs": 2, "optuna.space": SPACE})


def test_two_trial_study_completes_logs_and_leaves_local_test_unread(ws, config, tmp_path, monkeypatch):
    read = []
    original_frames, original_inputs = ld.load_frames, li.read_loss_inputs

    def spy_frames(name, *a, **k):
        read.append(name)
        return original_frames(name, *a, **k)

    def spy_inputs(name, *a, **k):
        read.append(name)
        return original_inputs(name, *a, **k)

    monkeypatch.setattr(ld, "load_frames", spy_frames)
    monkeypatch.setattr(li, "read_loss_inputs", spy_inputs)
    scorer = StubScorer(original_frames("working_validation", config, ws["paths"]), config.get("loader.clip_bound"))
    kwargs = dict(storage_dir=tmp_path / "optuna", lists=("working_train", "working_validation"), paths=ws["paths"], scorer=scorer,
                  scorer_record={"kind": "stub"}, device="cpu", run_root=tmp_path / "runs", checkpoint_root=tmp_path / "ckpt",
                  log=lambda m: None)
    study = oh.run_study(config, 2, "z14", **kwargs)
    assert len(study.trials) == 2 and all(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    assert (tmp_path / "optuna" / "z14.db").is_file()
    lines = [json.loads(line) for line in (tmp_path / "optuna" / "z14_trials.jsonl").read_text().splitlines()]
    assert [line["trial"] for line in lines] == [0, 1]
    for line, trial in zip(lines, study.trials):
        assert set(line["params"]) == set(SPACE) and line["objective"] == pytest.approx(trial.value)
        assert line["objective_slice"] == "no_near_repeat"
        assert set(trial.user_attrs["slices"]) == {"all", "no_near_repeat", "ip_negative", "corner"}
        assert trial.user_attrs["run_id"] == f"z14_t{trial.number:03d}"
        run = json.loads((tmp_path / "runs" / trial.user_attrs["run_id"] / "run.json").read_text())
        assert run["optuna"]["params"] == line["params"] and run["early_stopping"]["max_epochs"] == 2
        assert run["config"]["model"]["hidden"] == [line["params"]["model.width"]] * line["params"]["model.depth"]
    assert "local_test" not in read and set(read) == {"working_train", "working_validation"}
    # The study resumes from its database.
    again = oh.run_study(config, 1, "z14", **kwargs)
    assert len(again.trials) == 3


def test_guards(ws, config, tmp_path):
    with pytest.raises(ValueError):
        oh.run_study(config, 1, "x", tmp_path, paths=ws["paths"], scorer=None)
    with pytest.raises(PermissionError):
        oh.run_study(config, 1, "x", tmp_path, lists=("working_train", "local_test"), paths=ws["paths"], scorer=lambda f: {})
    assert oh.objective_value({"all": {"S": 0.7}, "no_near_repeat": {"S": float("nan")}}) == (0.7, "all")
    assert oh.objective_value({"all": {"S": 0.7}, "no_near_repeat": {"S": 0.6}}) == (0.6, "no_near_repeat")
