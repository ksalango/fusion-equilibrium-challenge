"""Evaluates a training run (``docs/PIPELINE.md``, Part 6) and writes
``reports/runs/<run_id>/eval.md`` and ``eval.json``.

The run's best checkpoint goes through the prediction chain on its validation list (mirror
averaging on, q95 from the map, betaN from the DIII-D raw-input ridge fitted on its training list)
and is scored with the organizers' scorer: every term for all shots, the shots without a
near-repeat, the negative-Ip shots, the MAST-like corner and by phase, with the score identity and
the MAST floor of F100 beside them; the worst 20 shots; the overfit panel (``--seed-runs`` adds
the other seeds of the arm to the seed spread); clip rates per feature and machine; the warm start
beside the trained model and mirror averaging on against off; grouped permutation importance;
and the label-free MAST sensitivity check on every ``--mast-shot-step``-th MAST test shot.

Reads the run record and metrics, the checkpoint, the stored frame-table, feature and target
builds, the raw validation shots (truth) and the scorer reference cache. ``local_test`` is never
read. Usage::

    python scripts/evaluate.py --run <run_id>
    python scripts/evaluate.py --run <run_id> --seed-runs <run_id_seed1> <run_id_seed2>
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from fusion_eq import baseline as bl, dataset, evaluation as ev, features as fe, frame_table as ft, prediction as pr, \
    predictors, target_build as tb


def log(message: str) -> None:
    print(message, flush=True)


def json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serialisable: {type(value).__name__}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", required=True)
    parser.add_argument("--seed-runs", nargs="*", default=[], help="other seeds of the arm, for the seed spread")
    parser.add_argument("--no-permutation", action="store_true", help="skip the grouped permutation importance")
    parser.add_argument("--no-mast", action="store_true", help="skip the MAST sensitivity check")
    parser.add_argument("--mast-shot-step", type=int, default=20, help="every n-th MAST test shot enters the sensitivity check")
    parser.add_argument("--heads-stride", type=int, default=4, help="frame stride of the DIII-D betaN ridge's training frames")
    parser.add_argument("--seed", type=int, default=0, help="seed of the permutations")
    parser.add_argument("--workers", type=int, default=None, help="scoring processes")
    parser.add_argument("--scalar-workers", type=int, default=None, help="processes for q95 from the maps")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--run-root", type=Path, default=pr.DEFAULT_RUN_ROOT)
    parser.add_argument("--checkpoint-root", type=Path, default=pr.DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--split", type=Path, default=predictors.DEFAULT_SPLIT)
    parser.add_argument("--data-root", type=Path, default=dataset.default_data_root())
    parser.add_argument("--table-root", type=Path, default=ft.DEFAULT_TABLE_ROOT)
    parser.add_argument("--feature-root", type=Path, default=fe.DEFAULT_FEATURE_ROOT)
    parser.add_argument("--feature-config", type=Path, default=None)
    parser.add_argument("--scaler", type=Path, default=None)
    parser.add_argument("--targets-root", type=Path, default=tb.DEFAULT_TARGET_ROOT)
    parser.add_argument("--green-dir", type=Path, default=predictors.DEFAULT_GREEN_DIR)
    parser.add_argument("--references-dir", type=Path, default=bl.DEFAULT_REFERENCE_DIR)
    args = parser.parse_args(argv)

    started = time.time()
    chain_paths = pr.ChainPaths(table_root=args.table_root, feature_root=args.feature_root, feature_config=args.feature_config,
                                scaler=args.scaler, targets_root=args.targets_root, green_dir=args.green_dir,
                                data_root=args.data_root, checkpoint_root=args.checkpoint_root)
    paths = ev.EvalPaths(run_root=args.run_root, chain=chain_paths, split=args.split, references_dir=args.references_dir)
    result = ev.evaluate(args.run, paths, args.seed_runs, args.workers, args.scalar_workers, args.device,
                         permutation=not args.no_permutation, mast=not args.no_mast, mast_shot_step=args.mast_shot_step,
                         heads_stride=args.heads_stride, seed=args.seed, log=log)
    result["seconds"] = time.time() - started
    run_dir = args.run_root / args.run
    (run_dir / "eval.json").write_text(json.dumps(result, indent=2, default=json_default) + "\n")
    (run_dir / "eval.md").write_text(ev.render(result))
    log(f"wrote {run_dir / 'eval.md'} ({result['seconds']:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
