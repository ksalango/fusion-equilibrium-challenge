"""Selects the smoke set for the frame-table build from the QA per-shot table.

Two shots of each kind, taken as the first two in sorted shot-id order so that the
selection is reproducible: DIII-D 20 kHz, DIII-D 2 kHz, DIII-D negative Ip, DIII-D
with no valid Thomson profile, DIII-D with fewer than 10 frames, DIII-D test, MAST
without NaN currents, MAST with NaN currents. Writes a JSON list of shot ids. Usage::

    python scripts/select_smoke_shots.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from fusion_eq.dataset import REPO_ROOT
from fusion_eq.profile import QA_RATE_SPLIT_MS

DEFAULT_QA_TABLE = REPO_ROOT / "reports" / "qa" / "qa_per_shot.parquet"
DEFAULT_OUT = REPO_ROOT / "reports" / "frame_table" / "smoke_shots.json"
PER_KIND = 2


def select(qa: pd.DataFrame, per_kind: int = PER_KIND) -> dict[str, list[str]]:
    """Shot ids per kind, first ``per_kind`` in sorted order."""
    train = qa[(qa["split"] == "diii_d_train") & qa["error"].isna()]
    test = qa[(qa["split"] == "diii_d_public_test") & qa["error"].isna()]
    mast = qa[(qa["split"] == "mast_public_test") & qa["error"].isna()]
    no_thomson = train["tsm_core_frames_valid_within_20ms"].isna() | (train["tsm_core_frames_valid_within_20ms"] == 0)
    kinds = {
        "d3d_20khz": train[train["time_magnetics_dt_median_ms"] <= QA_RATE_SPLIT_MS],
        "d3d_2khz": train[train["time_magnetics_dt_median_ms"] > QA_RATE_SPLIT_MS],
        "d3d_negative_ip": train[train["cur_plasma_current_max"] <= 0],
        "d3d_no_thomson": train[no_thomson],
        "d3d_under_10_frames": train[train["time_n_frames"] < 10],
        "d3d_test": test,
        "mast_without_nan": mast[~(mast["cur_plasma_current_nan_frac"] > 0)],
        "mast_with_nan": mast[mast["cur_plasma_current_nan_frac"] > 0],
    }
    return {kind: sorted(sub["shot"].astype(str))[:per_kind] for kind, sub in kinds.items()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--qa-table", type=Path, default=DEFAULT_QA_TABLE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    qa = pd.read_parquet(args.qa_table)
    kinds = select(qa)
    ids = sorted({s for shots in kinds.values() for s in shots})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(ids, indent=2) + "\n")
    for kind, shots in kinds.items():
        print(f"{kind:22s} {shots}")
    print(f"wrote {len(ids)} shot ids to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
