"""Builds the loss inputs of the model phase (``docs/PIPELINE.md``, Part 1, P3): per labelled
frame of each target table, the packed inside mask, the packed band mask and, with
``--reference exact``, the exact reference ``Ip G s`` in float16.

Frames: every row of the target table of each list (``../data/processed/targets/v<version>/
<list>.parquet``), in its order, so that the files align with the target rows by position and
by key. Masks are the target build's own masks (the scorer's boundary of the stored map,
``current_target.plasma_mask``); the band holds the nodes within ``--band-cells`` grid cells
of that boundary, both sides. The band width defaults to ``loss.band_cells`` of the model
configuration; the loader refuses a configuration whose band width differs from the build's.

The exact reference (about 11 GB for all lists) is built only on request; it reads the frame
table for the coil currents and Ip, recomputes each frame's share with the target build's
functions and applies the plasma response.

``local_test`` is refused unless ``--allow-local-test`` is given: it is read only at the final
pre-submission check. Only the labelled DIII-D training split is read; MAST labels never are.
Nothing is written under the raw data root. Usage::

    python scripts/build_loss_inputs.py                       # train, validation and the working lists
    python scripts/build_loss_inputs.py --lists working
    python scripts/build_loss_inputs.py --reference exact --lists working
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from fusion_eq import baseline as bl, current_target as ct, dataset, frame_table as ft, loss_inputs as li, target_build as tb, targets
from fusion_eq.coil_flux import GreenTable
from fusion_eq.dataset import default_data_root, run_info
from fusion_eq.model_config import DEFAULT_BASE, ModelConfig

DEFAULT_LISTS = ("train", "validation", "working_train", "working_validation")
WORKING_LISTS = ("working_train", "working_validation")
PROTECTED_LISTS = ("local_test",)


def log(message: str) -> None:
    print(message, flush=True)


def resolve_lists(requested: list[str], allow_local_test: bool) -> list[str]:
    """Lists to build; ``working`` stands for the two working lists.

    Raises:
        SystemExit: for ``local_test`` without ``--allow-local-test``.
    """
    lists = []
    for name in requested:
        lists += list(WORKING_LISTS) if name == "working" else [name]
    lists = list(dict.fromkeys(lists))
    blocked = [n for n in lists if n in PROTECTED_LISTS and not allow_local_test]
    if blocked:
        raise SystemExit(f"{blocked} is read only at the final pre-submission check; pass --allow-local-test to build it")
    return lists


def target_keys(list_name: str, version: int, root: Path) -> pd.DataFrame:
    """``shot``, ``frame`` of a target table, in its row order.

    Raises:
        SystemExit: when the table does not exist.
    """
    path = tb.table_path(list_name, version, root)
    if not path.is_file():
        raise SystemExit(f"{path} not found; build the targets first (python scripts/build_targets.py)")
    keys = pq.read_table(path, columns=["shot", "frame"]).to_pandas()
    keys["shot"] = keys["shot"].astype(str)
    return keys


def shot_groups(keys: pd.DataFrame) -> list[tuple[str, np.ndarray]]:
    """``(shot, frames)`` in the table's order; each shot's rows must be contiguous.

    Raises:
        ValueError: when a shot's rows are split.
    """
    shots = keys["shot"].to_numpy()
    starts = np.flatnonzero(np.r_[True, shots[1:] != shots[:-1]])
    order = shots[starts]
    if len(set(order)) != len(order):
        raise ValueError("a shot's rows are not contiguous in the target table")
    ends = np.r_[starts[1:], len(shots)]
    frames = keys["frame"].to_numpy(dtype=np.int64)
    return [(str(order[i]), frames[starts[i]:ends[i]]) for i in range(len(order))]


def build_list(name: str, args, keys: pd.DataFrame, grid_shape: tuple[int, int], exact: bool, workers: int,
               rows_by_shot: dict[str, pd.DataFrame] | None) -> dict:
    writer = li.ListWriter(name, keys, grid_shape, exact, args.target_version, args.out)
    groups = shot_groups(keys)
    items = [(shot, frames, str(args.data_root), args.split_name, str(args.green_dir), str(args.response_cache), args.band_cells,
              None if rows_by_shot is None else rows_by_shot[shot]) for shot, frames in groups]
    for k, result in enumerate(bl._map_ordered(li.shot_loss_inputs, items, workers)):
        writer.add(result)
        if (k + 1) % 200 == 0:
            log(f"  {name}: {k + 1}/{len(groups)} shots")
    record = writer.close()
    record["n_shots"] = len(groups)
    return record


def frame_rows(keys: pd.DataFrame, args, green: GreenTable) -> dict[str, pd.DataFrame]:
    """Frame-table rows (coil currents, ``ip__mean``) of the target table's frames, per shot."""
    columns = list(dict.fromkeys([*targets.current_columns_of(green, li.MACHINE), "ip__mean"]))
    shots = sorted(keys["shot"].unique())
    rows = bl.frames_of_shots(shots, columns, args.table_dir, args.split_name)
    rows["shot"] = rows["shot"].astype(str)
    rows = rows.set_index(["shot", "frame"]).loc[pd.MultiIndex.from_frame(keys)].reset_index()
    return {s: g.reset_index(drop=True) for s, g in rows.groupby("shot", sort=False)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_BASE, help="model configuration (band width, reference)")
    parser.add_argument("--lists", nargs="+", default=list(DEFAULT_LISTS), help="lists of the split file; 'working' for the two working lists")
    parser.add_argument("--reference", choices=("basis", "exact"), default=None,
                        help="exact also writes Ip G s per frame; the configuration's loss.reference when omitted")
    parser.add_argument("--band-cells", type=int, default=None, help="band half-width; loss.band_cells of the configuration when omitted")
    parser.add_argument("--allow-local-test", action="store_true")
    parser.add_argument("--target-version", type=int, default=1)
    parser.add_argument("--targets-root", type=Path, default=tb.DEFAULT_TARGET_ROOT)
    parser.add_argument("--split-name", default=tb.LABELLED_SPLIT)
    parser.add_argument("--data-root", type=Path, default=default_data_root())
    parser.add_argument("--table-dir", type=Path, default=ft.DEFAULT_TABLE_ROOT)
    parser.add_argument("--green-dir", type=Path, default=dataset.DEFAULT_PROCESSED_ROOT / "green")
    parser.add_argument("--response-cache", type=Path, default=ct.DEFAULT_CACHE_DIR)
    parser.add_argument("--out", type=Path, default=li.DEFAULT_ROOT)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args(argv)

    started = time.time()
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    try:
        tb.require_labelled(args.split_name)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    config = ModelConfig.load(args.config)
    args.band_cells = int(config.get("loss.band_cells") if args.band_cells is None else args.band_cells)
    reference = args.reference or config.get("loss.reference")
    exact = reference == "exact"
    lists = resolve_lists(args.lists, args.allow_local_test)
    workers = bl.default_workers() if args.workers is None else args.workers
    green = GreenTable.load(args.green_dir / "green_d3d.npz")
    grid_shape = (green.grid_z.size, green.grid_r.size)
    target_record = tb.target_dir(args.target_version, args.targets_root) / "build.json"
    log(f"lists {lists}; band {args.band_cells} cells; reference {reference}; {workers} workers")

    records, timing = {}, {}
    for name in lists:
        t0 = time.time()
        keys = target_keys(name, args.target_version, args.targets_root)
        rows = frame_rows(keys, args, green) if exact else None
        records[name] = build_list(name, args, keys, grid_shape, exact, workers, rows)
        timing[f"{name}_s"] = time.time() - t0
        log(f"{name}: {records[name]['n_rows']} frames of {records[name]['n_shots']} shots, "
            f"{records[name]['n_empty_inside']} without a boundary ({timing[f'{name}_s']:.0f} s)")

    out_dir = li.build_dir(args.target_version, args.out)
    previous = json.loads((out_dir / "build.json").read_text()) if (out_dir / "build.json").is_file() else {}
    if previous and (previous.get("band_cells") != args.band_cells or previous.get("grid_r") != green.grid_r.tolist()):
        log("previous build used another band width or grid; its list records are dropped")
        previous = {}
    all_lists = {**previous.get("lists", {}), **records}
    record = run_info(argv, vars(args), timestamp, time.time() - started, target_version=args.target_version,
                      band_cells=args.band_cells, reference=reference, grid_r=green.grid_r.tolist(), grid_z=green.grid_z.tolist(),
                      grid_hash=green.grid_hash, lists=all_lists, exact_lists=sorted(
                          set(previous.get("exact_lists", [])) - set(records) | (set(records) if exact else set())),
                      target_build_sha256=tb.sha256_of(target_record) if target_record.is_file() else None,
                      mask_rule="current_target.plasma_mask on the stored map in float64 (the target build's mask)",
                      timing=timing)
    (out_dir / "build.json").write_text(json.dumps(record, indent=2, default=str) + "\n")
    log(f"wrote {out_dir} in {time.time() - started:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
