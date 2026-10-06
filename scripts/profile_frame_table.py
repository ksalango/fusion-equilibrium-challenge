"""Profiles a built frame table and runs its acceptance checks.

Reads the tables under ``--table-dir`` and writes to ``--out``:

- ``frame_table_report.md``: the acceptance checks T1 to T8 of ``docs/FRAME_TABLE.md``
  with status pass, fail or review, then per-column missing counts.
- ``column_profile.parquet``: one row per column and split with range, percentiles,
  missing fraction and share of exact zeros.

Separate from the build so it can be re-run without rebuilding. Check T3 reads label
columns from a sample of raw training shots under ``--data-root``; check T5 reads the QA
per-shot table when ``--qa-table`` exists. Usage::

    python scripts/profile_frame_table.py
    python scripts/profile_frame_table.py --table-dir ../data/processed/frame_table_smoke \\
        --out reports/frame_table_smoke
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from fusion_eq.dataset import DATA_SPLITS, LABELLED_SPLITS, REPO_ROOT, SPLIT_MACHINE, default_data_root
from fusion_eq.frame_table import DEFAULT_TABLE_ROOT, table_path
from fusion_eq.profile import profile_table, render_report, run_checks, schema_roles

DEFAULT_OUT = REPO_ROOT / "reports" / "frame_table"
DEFAULT_QA_TABLE = REPO_ROOT / "reports" / "qa" / "qa_per_shot.parquet"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--table-dir", type=Path, default=DEFAULT_TABLE_ROOT, help="build directory")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--data-root", type=Path, default=default_data_root())
    parser.add_argument("--qa-table", type=Path, default=DEFAULT_QA_TABLE,
                        help="QA per-shot table for check T5; constants are used when absent")
    args = parser.parse_args(argv)

    build_info = {}
    build_file = args.table_dir / "build.json"
    if build_file.is_file():
        build_info = json.loads(build_file.read_text())

    profiles = []
    for split in DATA_SPLITS:
        path = table_path(split, args.table_dir)
        if not path.is_file():
            continue
        roles = schema_roles(SPLIT_MACHINE[split], split in LABELLED_SPLITS)
        profiles.append(profile_table(path, split, roles))
        print(f"profiled {split}", flush=True)
    profile = pd.concat(profiles, ignore_index=True) if profiles else pd.DataFrame()

    results = run_checks(args.table_dir, args.data_root, args.qa_table)
    args.out.mkdir(parents=True, exist_ok=True)
    profile.to_parquet(args.out / "column_profile.parquet", index=False)
    (args.out / "frame_table_report.md").write_text(render_report(results, profile, build_info))
    print(f"wrote {args.out / 'frame_table_report.md'} and column_profile.parquet")
    for r in results:
        print(f"  {r.check_id:3s} {r.status:6s} {r.name}: {r.headline}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
