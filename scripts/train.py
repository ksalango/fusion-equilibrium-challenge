"""Trains the plasma model (``docs/PIPELINE.md``, Part 4): one run of a model configuration on
a training list, early stopping on its validation list, the organizers' scorer at the warm
start and at the best checkpoint.

Inputs: the version-2 feature tables and stored scaler, the version-1 target tables and fitted
objects, the loss inputs (``scripts/build_loss_inputs.py``), the split file. The scorer also
reads the raw validation shots (truth), the frame table (coil currents, Ip, remainder and scalar
ridge inputs), the DIII-D Green's table, the cached scorer references and the scalar ridge of
``baseline_v0`` for the training list (``ridge_signed`` in the baseline cache).

Outputs: ``reports/runs/<run_id>/run.json`` and ``metrics.csv``; checkpoints ``best.pt`` and
``last.pt`` under ``--checkpoint-root/<run_id>`` (outside git). ``local_test`` is never read.
Usage::

    python scripts/train.py --config configs/model/arm_f.yaml --lists working
    python scripts/train.py --config configs/model/arm_g.yaml
    python scripts/train.py --config configs/model/arm_f.yaml --seed 1 --device cuda
    python scripts/train.py --config configs/model/rung2_f.yaml --seed 0 --no-score

``--seed`` overrides ``train.seed`` (initialization, dropout, shuffling, mirroring, Thomson
masking); the configuration hash and the run id include it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from fusion_eq import baseline as bl, dataset, features as fe, frame_table as ft, loader as ld, loss_inputs as li, \
    model_scoring as ms, target_build as tb, trainer as tr
from fusion_eq.model_config import DEFAULT_BASE, ModelConfig


def add_path_arguments(parser: argparse.ArgumentParser) -> None:
    """Input locations shared by ``train.py`` and ``optuna_study.py``."""
    parser.add_argument("--split", type=Path, default=ld.DataPaths.split)
    parser.add_argument("--feature-root", type=Path, default=fe.DEFAULT_FEATURE_ROOT)
    parser.add_argument("--feature-config", type=Path, default=None,
                        help="feature configuration; configs/features_v<features.version>.yaml when omitted")
    parser.add_argument("--scaler", type=Path, default=None,
                        help="stored scaler; reports/features/v<features.version>/scaler.json when omitted")
    parser.add_argument("--targets-root", type=Path, default=tb.DEFAULT_TARGET_ROOT)
    parser.add_argument("--loss-root", type=Path, default=li.DEFAULT_ROOT)
    parser.add_argument("--data-root", type=Path, default=dataset.DEFAULT_DATA_ROOT)
    parser.add_argument("--table-dir", type=Path, default=ft.DEFAULT_TABLE_ROOT)
    parser.add_argument("--green-dir", type=Path, default=dataset.DEFAULT_PROCESSED_ROOT / "green")
    parser.add_argument("--references-dir", type=Path, default=bl.DEFAULT_REFERENCE_DIR)
    parser.add_argument("--baseline-cache", type=Path, default=bl.DEFAULT_CACHE_DIR)
    parser.add_argument("--run-root", type=Path, default=tr.DEFAULT_RUN_ROOT)
    parser.add_argument("--checkpoint-root", type=Path, default=tr.DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--device", default=None, help="cuda, mps or cpu; automatic when omitted")
    parser.add_argument("--workers", type=int, default=None, help="scoring processes")
    parser.add_argument("--no-score", action="store_true", help="skip the organizers' scorer (loss curves only)")


def data_paths(args) -> ld.DataPaths:
    return ld.DataPaths(feature_root=args.feature_root, feature_config=args.feature_config, scaler=args.scaler,
                        target_root=args.targets_root, loss_root=args.loss_root, split=args.split, table_dir=args.table_dir)


def scoring_paths(args) -> ms.ScoringPaths:
    return ms.ScoringPaths(data_root=args.data_root, table_dir=args.table_dir, green_dir=args.green_dir,
                           references_dir=args.references_dir, baseline_cache=args.baseline_cache)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--lists", choices=sorted(tr.LIST_PAIRS), default="full",
                        help="full: train and validation; working: the working lists")
    parser.add_argument("--seed", type=int, default=None, help="overrides train.seed")
    parser.add_argument("--max-epochs", type=int, default=None, help="overrides train.max_epochs")
    parser.add_argument("--run-id", default=None)
    add_path_arguments(parser)
    args = parser.parse_args(argv)

    config = ModelConfig.load(args.config)
    overrides = {k: v for k, v in (("train.seed", args.seed), ("train.max_epochs", args.max_epochs)) if v is not None}
    if overrides:
        config = config.with_values(overrides)
    lists = tr.LIST_PAIRS[args.lists]
    paths = data_paths(args)
    scorer = None
    if not args.no_score:
        scorer = ms.ValidationScorer(lists[1], lists[0], config, paths, scoring_paths(args), args.workers)
    tr.train(config, lists, paths, run_root=args.run_root, checkpoint_root=args.checkpoint_root, device=args.device, scorer=scorer,
             scorer_record=None if scorer is None else scorer.record(), run_id=args.run_id, table_dir=args.table_dir,
             argv=sys.argv[1:] if argv is None else argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
