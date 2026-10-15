"""Optuna study of the plasma model on the working lists (``docs/PIPELINE.md``, Part 4).

Each trial trains the configuration with values from ``optuna.space`` (epochs bounded by
``optuna.max_epochs``) on ``working_train`` and scores the best checkpoint on
``working_validation`` with the organizers' scorer. Objective: S on the validation shots without
a near-repeat in the training list, the scalars fixed at the ``baseline_v0`` ridge. The study is
stored in ``reports/optuna/<study>.db`` (resumed when it exists) with every trial's parameters
and slice scores, also appended to ``reports/optuna/<study>_trials.jsonl``. Run records go to
``reports/runs/<study>_tNNN/``. ``local_test`` is never read. Usage::

    python scripts/optuna_study.py --config configs/model/arm_f.yaml --trials 20
    python scripts/optuna_study.py --config configs/model/arm_g.yaml --trials 20 --study arm_g_v1
"""

from __future__ import annotations

import argparse
from pathlib import Path

from fusion_eq import model_scoring as ms, optuna_harness as oh, trainer as tr
from fusion_eq.model_config import DEFAULT_BASE, ModelConfig

from train import add_path_arguments, data_paths, scoring_paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--trials", type=int, required=True)
    parser.add_argument("--study", default=None, help="study name; the configuration file stem and hash when omitted")
    parser.add_argument("--storage-dir", type=Path, default=oh.DEFAULT_STORAGE)
    add_path_arguments(parser)
    args = parser.parse_args(argv)
    if args.no_score:
        raise SystemExit("the Optuna objective is the scorer's S; --no-score does not apply")

    config = ModelConfig.load(args.config)
    study_name = args.study or f"{args.config.stem}_{config.hash()[:8]}"
    lists = tr.LIST_PAIRS["working"]
    paths = data_paths(args)
    scorer = ms.ValidationScorer(lists[1], lists[0], config, paths, scoring_paths(args), args.workers)
    study = oh.run_study(config, args.trials, study_name, args.storage_dir, lists, paths, scorer, scorer.record(), args.device,
                         args.run_root, args.checkpoint_root, table_dir=args.table_dir)
    best = study.best_trial
    print(f"study {study_name}: {len(study.trials)} trials; best trial {best.number}, objective {best.value:.4f}, params {best.params}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
