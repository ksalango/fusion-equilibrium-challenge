"""Draws the split of the DIII-D training shots and writes it as a JSON file.

Analysis A5 and checks S5, S6 of ``docs/SPLIT.md``. The draw is stratified by recording
rate and plasma-current direction and reproducible for ``--seed``. In ``grouped`` mode
the near-repeat groups at ``--threshold`` (from ``find_near_repeats.py``) are kept
whole. The file records, for every training shot, its group and whether it has a
near-repeat in the ``train`` list, from the distance matrix. The checks are appended
to ``reports/split/split_report.md``. An existing output file is never overwritten
unless ``--overwrite`` is given: a committed split is not redrawn. Usage::

    python scripts/make_split.py --mode random --out reports/split/split_provisional.json
    python scripts/make_split.py --mode random
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from fusion_eq import frame_table as ft, similarity as sim, split as sp, starter_kit
from fusion_eq.dataset import DEFAULT_PROCESSED_ROOT, REPO_ROOT, run_info

DEFAULT_OUT = REPO_ROOT / "splits" / "split_v1.json"
DEFAULT_REPORT_DIR = REPO_ROOT / "reports" / "split"
DEFAULT_DISTANCE = DEFAULT_PROCESSED_ROOT / "split" / "distance.npz"


def load_distance(path: Path) -> sim.PairwiseDistance:
    with np.load(path, allow_pickle=False) as f:
        return sim.PairwiseDistance(shots=f["shots"].astype(object), distance=f["distance"].astype(np.float64),
                                    n_common_times=f["n_common_times"].astype(np.int32))


def render_section(out: Path, args, counts: dict, status: str, balance: pd.DataFrame, n_near: int, n_train: int) -> str:
    lines = [f"## Split `{out}`", "",
             f"Mode `{args.mode}`, threshold {args.threshold}, seed {args.seed}; drawn {datetime.now(timezone.utc).date()}.", "",
             f"| S5 Split integrity | {status} | " + ", ".join(f"{k} {v}" for k, v in counts.items()) + " |", "",
             f"Training shots with a near-repeat in `train` (threshold {args.threshold}): {n_near} of {n_train}.", "",
             "### S6 Split balance (review)", "",
             "| List | Shots | Share 20 kHz | Share negative Ip | Median frames | Share no core Thomson |",
             "| --- | --- | --- | --- | --- | --- |"]
    for r in balance.itertuples(index=False):
        lines.append(f"| {r.list} | {r.shots} | {r.share_20khz:.3f} | {r.share_negative_ip:.3f} | {r.median_frames:.0f} | "
                     f"{r.share_no_core_thomson:.3f} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--mode", choices=sp.MODES, required=True)
    parser.add_argument("--threshold", type=float, default=sim.PROVISIONAL_THRESHOLD)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-validation", type=int, default=sp.SplitSizes.n_validation)
    parser.add_argument("--n-local-test", type=int, default=sp.SplitSizes.n_local_test)
    parser.add_argument("--n-working-train", type=int, default=sp.SplitSizes.n_working_train)
    parser.add_argument("--n-working-validation", type=int, default=sp.SplitSizes.n_working_validation)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--table-dir", type=Path, default=ft.DEFAULT_TABLE_ROOT)
    parser.add_argument("--groups", type=Path, default=DEFAULT_REPORT_DIR / "shot_groups.parquet")
    parser.add_argument("--distance", type=Path, default=DEFAULT_DISTANCE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_DIR / "split_report.md")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing output file")
    args = parser.parse_args(argv)

    if args.out.exists() and not args.overwrite:
        raise SystemExit(f"{args.out} exists; a committed split is not redrawn (use --overwrite for a provisional file)")
    started = time.time()
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    sizes = sp.SplitSizes(args.n_validation, args.n_local_test, args.n_working_train, args.n_working_validation)

    index = pd.read_parquet(args.table_dir / "shot_index.parquet")
    index["shot"] = index["shot"].astype(str)
    train = index[(index["split"] == sp.TRAIN_SPLIT) & index["error"].isna()]
    shot_table = train[["shot", *sp.STRATA]].reset_index(drop=True)

    groups = None
    if args.groups.is_file():
        gdf = pd.read_parquet(args.groups)
        column = sim.group_column(args.threshold)
        if column not in gdf:
            raise SystemExit(f"{args.groups} has no column {column}; rerun find_near_repeats.py with this threshold")
        groups = dict(zip(gdf["shot"].astype(str), gdf[column].astype(int)))
        missing = sorted(set(shot_table["shot"]) - set(groups))
        if missing:
            raise SystemExit(f"{len(missing)} training shots have no group, first {missing[:5]}")
    elif args.mode == "grouped":
        raise SystemExit(f"grouped mode needs {args.groups}; run find_near_repeats.py first")

    parts = sp.draw_split(shot_table, sizes, args.seed, groups if args.mode == "grouped" else None)
    near = None
    if args.distance.is_file():
        dist = load_distance(args.distance)
        flags = sim.has_near_repeat_in(dist, shot_table["shot"], parts["train"], args.threshold)
        near = dict(zip(shot_table["shot"], flags.tolist()))
    else:
        print(f"no distance matrix at {args.distance}; near-repeat flags not recorded", flush=True)

    check_sizes = sizes if args.mode == "random" else None
    counts = sp.check_split(parts, index, check_sizes, groups if args.mode == "grouped" else None)
    status = "pass"
    if args.mode == "grouped" and (counts["validation"] != sizes.n_validation or counts["local_test"] != sizes.n_local_test):
        status = "review (grouped mode could not fill the counts with whole groups)"
    balance = sp.split_balance(parts, index)

    record = sp.split_record(parts, args.mode, args.threshold, args.seed, sizes, starter_kit.git_commit(REPO_ROOT), groups, near)
    record["run"] = run_info(argv, vars(args), timestamp, time.time() - started)
    sp.write_split(args.out, record)
    n_near = int(sum(near.values())) if near else 0
    section = render_section(args.out, args, counts, status, balance, n_near, len(shot_table))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("a") as f:
        f.write("\n" + section)
    print(section)
    print(f"written {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
