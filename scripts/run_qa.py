"""Runs the data quality checks over the dataset and writes the QA outputs.

Selects the shot files per split, runs the checks of ``fusion_eq.qa`` on each shot
in worker processes, and writes three files to ``--out``:

- ``qa_per_shot.parquet``: one row per shot, one column per metric, plus ``split``,
  ``machine``, ``shot``, ``path`` and ``error``.
- ``qa_report.md``: summary table and one section per check.
- ``qa_run.json``: repository and starter-kit commits, timestamp, arguments, runtime.

Nothing is written under the data root. Usage::

    python scripts/run_qa.py --n-shots 20          # smoke test
    python scripts/run_qa.py                       # full run
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from fusion_eq import starter_kit
from fusion_eq.qa.report import RunContext, render_report, summarize
from fusion_eq.qa.run import rows_to_frame, run_shot

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = REPO_ROOT.parent / "data" / "raw" / "data"
DATA_SPLITS = ("diii_d_train", "diii_d_public_test", "mast_public_test")
SAMPLE_SPLIT = "mast_samples"
ALL_SPLITS = DATA_SPLITS + (SAMPLE_SPLIT,)


def list_shots(data_root: Path, split: str) -> list[Path]:
    """Sorted shot files of one split; the MAST samples come from the starter kit."""
    if split == SAMPLE_SPLIT:
        return starter_kit.sample_shot_paths("MAST")
    folder = data_root / split
    if not folder.is_dir():
        raise FileNotFoundError(f"split folder not found: {folder}")
    return sorted(folder.glob("*.parquet"))


def select_shots(files: list[Path], n_shots: int | None, seed: int) -> list[Path]:
    """Deterministic sample of ``n_shots`` files, or all of them when None."""
    if n_shots is None or n_shots >= len(files):
        return list(files)
    rng = random.Random(seed)
    return sorted(rng.sample(files, n_shots))


def build_jobs(args) -> list[tuple[str, Path, bool]]:
    """(split, path, run scorer check) for every selected shot."""
    jobs = []
    for split in args.splits:
        files = list_shots(args.data_root, split)
        if split != SAMPLE_SPLIT:
            files = select_shots(files, args.n_shots, args.seed)
        scorer = set()
        if split == SAMPLE_SPLIT:
            scorer = set(files)
        elif split == "diii_d_train":
            scorer = set(select_shots(files, args.n_scorer_shots, args.seed + 1))
        jobs += [(split, path, path in scorer) for path in files]
    return jobs


def _work(job: tuple[str, Path, bool]) -> dict:
    split, path, scorer = job
    return run_shot(path, split, with_sampled=scorer)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--splits", nargs="+", default=list(ALL_SPLITS), choices=ALL_SPLITS,
                        help="splits to check (default: all, plus the MAST sample shots)")
    parser.add_argument("--n-shots", type=int, default=None,
                        help="shots per split (default: all); the MAST samples are never subsampled")
    parser.add_argument("--n-scorer-shots", type=int, default=200,
                        help="DIII-D train shots for the scorer comparison (check 0.15)")
    parser.add_argument("--seed", type=int, default=0, help="seed of the shot sampling")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "internal" / "qa")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    args = parser.parse_args(argv)

    starter_kit.require()
    started = time.time()
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    jobs = build_jobs(args)
    print(f"{len(jobs)} shots over {len(args.splits)} splits, {args.workers} workers", flush=True)

    rows: list[dict] = []
    report_every = max(1, len(jobs) // 50)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_work, job) for job in jobs]
        for i, fut in enumerate(as_completed(futures), start=1):
            rows.append(fut.result())
            if i % report_every == 0 or i == len(jobs):
                elapsed = time.time() - started
                print(f"  {i}/{len(jobs)} shots, {elapsed:.0f} s elapsed, "
                      f"{elapsed / i * (len(jobs) - i):.0f} s remaining", flush=True)

    df = rows_to_frame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out / "qa_per_shot.parquet", index=False)

    runtime = time.time() - started
    run_info = {
        "timestamp": timestamp,
        "repo_commit": starter_kit.git_commit(REPO_ROOT),
        "starter_commit": starter_kit.git_commit(starter_kit.starter_dir()),
        "args": " ".join(sys.argv[1:]) if argv is None else " ".join(argv),
        "arguments": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "n_shots": int(len(df)),
        "shots_per_split": {s: int(n) for s, n in df.groupby("split").size().items()},
        "runtime_s": runtime,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    (args.out / "qa_run.json").write_text(json.dumps(run_info, indent=2) + "\n")

    ctx = RunContext(full_run=args.n_shots is None and set(args.splits) >= set(DATA_SPLITS),
                     splits=tuple(args.splits))
    results = summarize(df, ctx)
    (args.out / "qa_report.md").write_text(render_report(results, run_info, df))

    print(f"wrote {args.out / 'qa_per_shot.parquet'}, qa_report.md, qa_run.json in {runtime:.0f} s")
    for r in results:
        print(f"  {r.check_id:5s} {r.status:6s} {r.name}: {r.headline}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
