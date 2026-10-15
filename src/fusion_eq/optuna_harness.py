"""Optuna harness of the plasma model (``docs/PIPELINE.md``, Part 4).

Each trial samples the search space of ``optuna.space`` (keys are configuration paths;
``model.depth`` and ``model.width`` set ``model.hidden``), trains on the working training list
with ``train.max_epochs`` bounded by ``optuna.max_epochs``, and scores the best checkpoint on
the working validation list. The objective is the scorer's S on the validation shots without a
near-repeat in the training list (the selection subset of D19 item 7), with the scalar term
fixed by the scorer's baseline scalars so that it moves with the map alone; when that subset is
empty the objective falls back to all shots and the trial records it. Every trial's parameters,
objective, slice scores (all, no near-repeat, negative Ip, MAST-like corner) and run id are
stored as trial attributes in the study database ``<storage_dir>/<study>.db`` and appended to
``<storage_dir>/<study>_trials.jsonl``. ``local_test`` is never read.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import optuna

from fusion_eq import dataset, loader as ld, trainer as tr
from fusion_eq.model_config import ModelConfig

DEFAULT_STORAGE = dataset.REPO_ROOT / "reports" / "optuna"
OBJECTIVE_SLICE = "no_near_repeat"


def suggest(trial: optuna.Trial, space: Mapping[str, Mapping]) -> dict[str, Any]:
    """Values of the search space for one trial, keyed by configuration path."""
    values = {}
    for key, spec in space.items():
        kind = spec["type"]
        if kind == "float":
            values[key] = trial.suggest_float(key, float(spec["low"]), float(spec["high"]))
        elif kind == "log_float":
            values[key] = trial.suggest_float(key, float(spec["low"]), float(spec["high"]), log=True)
        elif kind == "int":
            values[key] = trial.suggest_int(key, int(spec["low"]), int(spec["high"]))
        elif kind == "categorical":
            values[key] = trial.suggest_categorical(key, list(spec["choices"]))
        else:
            raise ValueError(f"search-space type {kind!r} of {key} is not supported")
    return values


def objective_value(scores: Mapping[str, Mapping]) -> tuple[float, str]:
    """S of the selection slice, or of all shots when the slice has none."""
    value = scores.get(OBJECTIVE_SLICE, {}).get("S", float("nan"))
    if value is not None and math.isfinite(value):
        return float(value), OBJECTIVE_SLICE
    return float(scores["all"]["S"]), "all"


def run_study(config: ModelConfig, n_trials: int, study_name: str, storage_dir: Path = DEFAULT_STORAGE,
              lists: Sequence[str] = tr.LIST_PAIRS["working"], paths: ld.DataPaths = ld.DataPaths(),
              scorer: tr.Scorer | None = None, scorer_record: Mapping | None = None, device: str | None = None,
              run_root: Path = tr.DEFAULT_RUN_ROOT, checkpoint_root: Path = tr.DEFAULT_CHECKPOINT_ROOT,
              train_fn: Callable[..., tr.TrainResult] = tr.train, log: Callable[[str], None] = tr.log_print,
              table_dir: Path | None = None) -> optuna.Study:
    """Runs ``n_trials`` trials of a study (resumed when the database already holds it).

    Raises:
        ValueError: without a scorer (the objective is the scorer's S).
        PermissionError: when ``local_test`` is among the lists.
    """
    if scorer is None:
        raise ValueError("the Optuna objective is the scorer's S; a scorer is required")
    for name in lists:
        ld.require_readable(name)
    storage_dir = Path(storage_dir)
    storage_dir.mkdir(parents=True, exist_ok=True)
    db = storage_dir / f"{study_name}.db"
    trials_log = storage_dir / f"{study_name}_trials.jsonl"
    sampler = optuna.samplers.TPESampler(seed=int(config.get("optuna.seed")))
    study = optuna.create_study(study_name=study_name, storage=f"sqlite:///{db}", direction="maximize", sampler=sampler,
                                load_if_exists=True)
    study.set_user_attr("config_hash", config.hash())
    study.set_user_attr("lists", list(lists))
    max_epochs = min(int(config.get("train.max_epochs")), int(config.get("optuna.max_epochs")))

    def objective(trial: optuna.Trial) -> float:
        values = suggest(trial, config.get("optuna.space"))
        trial_config = config.with_values({**values, "train.max_epochs": max_epochs})
        run_id = f"{study_name}_t{trial.number:03d}"
        result = train_fn(trial_config, lists, paths, run_root=run_root, checkpoint_root=checkpoint_root, device=device,
                          scorer=scorer, scorer_record=scorer_record, run_id=run_id, table_dir=table_dir, log=log,
                          extra_record={"optuna": {"study": study_name, "trial": trial.number, "params": values}})
        scores = result.record["best"]["score"]
        value, used = objective_value(scores)
        attrs = {"run_id": run_id, "objective_slice": used, "config_hash": trial_config.hash(),
                 "best_epoch": result.record["early_stopping"]["best_epoch"], "slices": _plain(scores),
                 "warm_start_S": _plain(result.record["warm_start"].get("score", {}).get("all", {}).get("S"))}
        for key, v in attrs.items():
            trial.set_user_attr(key, v)
        with trials_log.open("a") as f:
            f.write(json.dumps({"trial": trial.number, "params": values, "objective": value, **attrs}, sort_keys=True) + "\n")
        log(f"trial {trial.number}: objective {value:.4f} ({used}), params {values}")
        return value

    study.optimize(objective, n_trials=int(n_trials))
    return study


def _plain(value):
    """JSON-safe copy (NaN becomes None)."""
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "item"):
        return _plain(value.item())
    return value
