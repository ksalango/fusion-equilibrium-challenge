"""Builds the frame table: one row per EFIT frame, inputs resampled onto frame times.

Selects the shot files per split, builds each shot's rows in worker processes with
``fusion_eq.frame_table.build_one``, and writes to ``--out``:

- ``<split>.parquet`` for each split: the frame rows, shots in sorted id order, about
  ``--shots-per-group`` shots per Parquet row group.
- ``shot_index.parquet``: one row per shot with split, machine, frame count, the
  shot-level flags, ``ip_rise_ms``, ``sol_at_rise`` and ``error`` (None, or why the shot
  produced no rows).
- ``build.json``: repository and starter-kit commits, timestamp, arguments, runtime.

Shots are processed in sorted order and written in that order, so the output does
not depend on which worker finishes first. One failing shot does not stop the run.
Nothing is written under the data root. Usage::

    python scripts/build_frame_table.py --shots-file reports/frame_table/smoke_shots.json \\
        --out ../data/processed/frame_table_smoke
    python scripts/build_frame_table.py                      # full run
    python scripts/build_frame_table.py --rate-window-ms 10 50   # slope windows (the default)
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from fusion_eq import starter_kit
from fusion_eq.dataset import (DATA_SPLITS, LABELLED_SPLITS, default_data_root, filter_shots, list_shots,
                               read_shots_file, run_info, select_shots)
from fusion_eq.frame_table import (DEFAULT_RATE_WINDOWS_MS, DEFAULT_TABLE_ROOT, BuildParams, arrow_schema, build_one,
                                   shot_index_frame, table_schema, to_arrow)

DEFAULT_SHOTS_PER_GROUP = 200


def build_jobs(args) -> list[tuple[str, Path]]:
    """(split, path) for every selected shot, sorted within each split."""
    shot_ids = read_shots_file(args.shots_file) if args.shots_file else None
    jobs = []
    for split in args.splits:
        files = list_shots(args.data_root, split)
        if shot_ids is not None:
            files = filter_shots(files, shot_ids)
        files = select_shots(files, args.n_shots, args.seed)
        jobs += [(split, path) for path in files]
    return jobs


def _work(job: tuple[str, Path, BuildParams]):
    split, path, params = job
    return build_one(path, split, params)


def write_split(split: str, results, out_dir: Path, shots_per_group: int,
                rate_windows_ms=DEFAULT_RATE_WINDOWS_MS) -> int:
    """Writes one split's rows as they arrive in order; returns the number of rows written."""
    writer, buffer, buffered, written, schema = None, [], 0, 0, None
    for frames in results:
        if frames is None:
            continue
        if schema is None:
            schema = table_schema(frames["machine"].iloc[0], split in LABELLED_SPLITS, rate_windows_ms)
            writer = pq.ParquetWriter(out_dir / f"{split}.parquet", arrow_schema(schema))
        buffer.append(frames)
        buffered += 1
        if buffered >= shots_per_group:
            written += _flush(writer, buffer, schema)
            buffer, buffered = [], 0
    if buffer:
        written += _flush(writer, buffer, schema)
    if writer is not None:
        writer.close()
    return written


def _flush(writer, buffer, schema) -> int:
    table = to_arrow(pd.concat(buffer, ignore_index=True), schema)
    writer.write_table(table, row_group_size=table.num_rows)
    return table.num_rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--splits", nargs="+", default=list(DATA_SPLITS), choices=DATA_SPLITS)
    parser.add_argument("--n-shots", type=int, default=None, help="shots per split (default: all)")
    parser.add_argument("--shots-file", type=Path, default=None,
                        help="JSON list of shot ids; only these shots are built")
    parser.add_argument("--seed", type=int, default=0, help="seed of the shot sampling")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--data-root", type=Path, default=default_data_root())
    parser.add_argument("--out", type=Path, default=None,
                        help=f"output directory (default {DEFAULT_TABLE_ROOT})")
    parser.add_argument("--window-ms", type=float, default=2.0)
    parser.add_argument("--rate-window-ms", type=float, nargs="+", default=list(DEFAULT_RATE_WINDOWS_MS),
                        help="slope windows in ms: the first gives __rate, each further one __rate<ms>")
    parser.add_argument("--max-gap-ms", type=float, default=5.0)
    parser.add_argument("--thomson-tol-ms", type=float, default=20.0)
    parser.add_argument("--no-align-ip", action="store_true", help="keep the organizers' Ip axis (requires --out)")
    parser.add_argument("--no-psi-summary", action="store_true", help="skip psi_min/max/mean (no flux maps read)")
    parser.add_argument("--shots-per-group", type=int, default=DEFAULT_SHOTS_PER_GROUP,
                        help="shots per Parquet row group")
    args = parser.parse_args(argv)
    if args.no_align_ip and args.out is None:
        parser.error("--no-align-ip requires an explicit --out, so the default table is never overwritten")
    if args.out is None:
        args.out = DEFAULT_TABLE_ROOT

    starter_kit.require()
    params = BuildParams(window_ms=args.window_ms, rate_window_ms=args.rate_window_ms,
                         max_gap_ms=args.max_gap_ms, thomson_tol_ms=args.thomson_tol_ms,
                         align_ip=not args.no_align_ip, psi_summary=not args.no_psi_summary)
    started = time.time()
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    jobs = build_jobs(args)
    print(f"{len(jobs)} shots over {len(args.splits)} splits, {args.workers} workers", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)

    index_rows: list[dict] = []
    rows_per_split: dict[str, int] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for split in args.splits:
            split_jobs = [(split, path, params) for s, path in jobs if s == split]
            if not split_jobs:
                continue
            done = 0

            def ordered():
                nonlocal done
                for frames, row in pool.map(_work, split_jobs, chunksize=4):
                    index_rows.append(row)
                    done += 1
                    if done % max(1, len(split_jobs) // 20) == 0 or done == len(split_jobs):
                        elapsed = time.time() - started
                        print(f"  {split}: {done}/{len(split_jobs)} shots, {elapsed:.0f} s elapsed", flush=True)
                    yield frames

            rows_per_split[split] = write_split(split, ordered(), args.out, args.shots_per_group, params.rate_window_ms)

    index = shot_index_frame(index_rows)
    index.to_parquet(args.out / "shot_index.parquet", index=False)
    runtime = time.time() - started
    info = run_info(argv, vars(args), timestamp, runtime, params=params.to_dict(),
                    n_shots=int(len(index)), n_errors=int(index["error"].notna().sum()),
                    shots_per_split={s: int(n) for s, n in index.groupby("split").size().items()},
                    rows_per_split=rows_per_split)
    (args.out / "build.json").write_text(json.dumps(info, indent=2) + "\n")
    print(f"wrote {args.out} in {runtime:.0f} s: {rows_per_split}, {info['n_errors']} shot errors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
