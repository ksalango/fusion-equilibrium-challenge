"""Builds the targets of the target phase (``docs/TARGET.md``): fits the current and remainder
bases and the remainder ridges on the training list, then writes one target table per list.

Steps:

1. Frame rows of every list from the frame table (every ``--stride``-th frame per shot): the
   coil currents, the ridge inputs of each rate window the table carries, Ip, the phase
   columns and the columns of the lighter label filter.
2. Fit sample: a fixed-seed sample of ``current.fit_frames`` frames of the training list that
   pass the lighter label filter; frames without a boundary are replaced from the same
   permutation until the sample is full or the candidates run out. The current basis
   (``current_basis.CurrentBasis``, exact parity, originals and mirrors) and the remainder
   basis (originals, target frame ``sign(Ip) r``) are fitted on it.
3. Plasma responses of both machines (cached under ``--response-cache``), the placement of each
   machine (identity on DIII-D) and the flux bases.
4. One table per list, ``<out>/v<version>/<list>.parquet``, keyed on ``shot`` and ``frame``:
   current coefficients ``c_00`` onward, share total, share outside the boundary, relative
   reconstruction error of the share and of its mirror, remainder coefficients ``r_00``
   onward, remainder RMS over the frame's flux range, the seven shape numbers, phase, the
   boundary flag and the label-filter flags.
5. Remainder ridges, one per rate window, on the training-list frames that pass the filter;
   each is scored on the validation list and the better one is saved as the default.
6. The fitted objects and ``build.json`` (commits, arguments, configuration, machine constants
   with their verified status, sample, counts, ridge scores, checksums, runtime).

Only the labelled DIII-D training split is read (``--split-name``); MAST equilibrium labels are
never read. Nothing is written under the raw data root. Usage::

    python scripts/build_targets.py --config configs/target_v1.yaml --lists working
    python scripts/build_targets.py --config configs/target_v1.yaml
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from fusion_eq import (baseline as bl, current_basis as cb, current_target as ct, dataset, frame_table as ft,
                       label_filter as lf, lattice as lt, split as sp, target_build as tb, targets)
from fusion_eq.coil_flux import GreenTable
from fusion_eq.dataset import default_data_root, run_info
from fusion_eq.machines import machine_config
from fusion_eq.remainder_model import RemainderModel, available_windows, window_name

DEFAULT_SPLIT = dataset.REPO_ROOT / "splits" / "split_v1.json"
DEFAULT_LISTS = ("train", "validation", "local_test")
WORKING_LISTS = ("working_train", "working_validation")
FIT_LIST = {"train": "train", "working_train": "working_train"}
VALIDATION_LIST = {"train": "validation", "working_train": "working_validation"}


def log(message: str) -> None:
    print(message, flush=True)


def resolve_lists(requested: list[str], split: dict) -> tuple[list[str], str, str | None]:
    """The lists to build, the fit list and the validation list.

    Raises:
        SystemExit: for a list not in the split file or a request without a training list.
    """
    lists = list(WORKING_LISTS) if requested == ["working"] else list(dict.fromkeys(requested))
    unknown = [name for name in lists if name not in split]
    if unknown:
        raise SystemExit(f"lists {unknown} are not in the split file; known: {sorted(k for k, v in split.items() if isinstance(v, list))}")
    fit = next((name for name in lists if name in FIT_LIST), None)
    if fit is None:
        raise SystemExit("the lists must include a training list (train or working_train) to fit on")
    validation = VALIDATION_LIST[fit] if VALIDATION_LIST[fit] in lists else None
    return lists, fit, validation


def frame_rows(shot_ids: list[str], columns: list[str], table_dir: Path, stride: int) -> dict[str, pd.DataFrame]:
    """Frame-table rows per shot (sorted ids), every ``stride``-th frame in time order."""
    if not shot_ids:
        return {}
    rows = bl.frames_of_shots(shot_ids, columns, table_dir)
    rows["shot"] = rows["shot"].astype(str)
    return {s: g.sort_values("frame").iloc[::stride].reset_index(drop=True) for s, g in rows.groupby("shot", sort=True)}


def fit_bases(rows: dict[str, pd.DataFrame], keep: dict[str, np.ndarray], config: tb.TargetConfig, args,
              workers: int) -> tuple[cb.CurrentBasis, bl.FluxBasis, dict]:
    """Current and remainder bases on the fixed-seed sample of filter-passing training frames
    with a boundary."""
    candidates = [(s, i) for s in rows for i in np.flatnonzero(keep[s])]
    order = np.random.default_rng(config.seed).permutation(len(candidates))
    acc_current, acc_remainder = None, bl.BasisAccumulator()
    taken, collected, dropped = 0, 0, 0
    while collected < config.fit_frames and taken < len(order):
        batch = order[taken:taken + (config.fit_frames - collected)]
        taken += len(batch)
        chosen = {s: np.zeros(len(g), dtype=bool) for s, g in rows.items()}
        for k in batch:
            s, i = candidates[k]
            chosen[s][i] = True
        items = [(s, rows[s], chosen[s], str(args.data_root), args.split_name, args.stride, str(args.green_dir),
                  str(args.response_cache)) for s in rows if chosen[s].any()]
        for result in bl._map_ordered(tb.fit_shot, items, workers):
            found = result["boundary"]
            if found.any():
                acc_current = acc_current or cb.CurrentBasisAccumulator(result["share"].shape[1:])
                acc_current.add(result["share"][found])
                acc_remainder.add(result["remainder"][found])
            collected += int(found.sum())
            dropped += int((~found).sum())
        log(f"  fit sample: {collected} frames with a boundary, {dropped} without, {taken} of {len(candidates)} candidates drawn")
    if acc_current is None:
        raise SystemExit("no training frame passes the label filter with a boundary")
    current = acc_current.finish(config.n_current)
    remainder = acc_remainder.finish(max(20, config.n_remainder))
    return current, remainder, {"n_candidates": len(candidates), "n_drawn": int(taken), "n_frames": int(collected),
                                "n_without_boundary": int(dropped), "seed": config.seed, "requested": config.fit_frames}


def flux_objects(current: cb.CurrentBasis, args) -> tuple[dict, dict, dict]:
    """Responses, placements and flux bases of both machines."""
    responses, placements, fluxes = {}, {}, {}
    grids = {m: GreenTable.load(args.green_dir / f"green_{machine_config(m).file_prefix.split('_')[0]}.npz") for m in tb.MACHINES}
    reference = lt.reference_lattice(grids["DIII-D"].grid_r, grids["DIII-D"].grid_z)
    for m in tb.MACHINES:
        responses[m] = cb.machine_response(m, args.green_dir, args.response_cache)
        native = lt.native_lattice(m, grids[m].grid_r, grids[m].grid_z)
        placements[m] = lt.placement(m, reference, native)
        fluxes[m] = cb.flux_basis(m, current, responses[m], placements[m])
        log(f"  {m}: flux basis of {fluxes[m].n_components} components, mean share kept by placement "
            f"{fluxes[m].mean_retained:.12f}, on border cells {fluxes[m].mean_border_weight:.2e}")
    return responses, placements, fluxes


def build_table(name: str, rows: dict[str, pd.DataFrame], current: cb.CurrentBasis, remainder: RemainderModel,
                config: tb.TargetConfig, args, workers: int) -> tuple[pd.DataFrame, dict]:
    """The target table of one list and its record."""
    ids = list(rows)
    cons = tb.cached_shape_numbers(ids, args.split_name, args.stride, args.references_dir) if config.shape_numbers else None
    items = [(s, rows[s], str(args.data_root), args.split_name, args.stride, str(args.green_dir), str(args.response_cache),
              current, remainder, None if cons is None else cons[s], config.shape_numbers) for s in ids]
    parts = []
    for k, out in enumerate(bl._map_ordered(tb.table_shot, items, workers)):
        part = pd.DataFrame(out)
        rules = lf.lighter_filter_rules(rows[out["shot"][0]])
        part["passes_filter"] = ~rules.any(axis=1).to_numpy()
        for rule in lf.RULES:
            part[f"filter_{rule}"] = rules[rule].to_numpy()
        parts.append(part)
        if (k + 1) % 100 == 0:
            log(f"  {name}: {k + 1}/{len(ids)} shots")
    table = pd.concat(parts, ignore_index=True)
    record = {"n_shots": len(ids), "n_rows": int(len(table)), "n_without_boundary": int((~table["boundary_found"]).sum()),
              "n_passing_filter": int(table["passes_filter"].sum()),
              "shape_numbers": ("cached scorer references" if cons is not None else "scorer reference builder")
              if config.shape_numbers else "not built"}
    return table, record


def write_table(table: pd.DataFrame, path: Path) -> None:
    """Writes a table with float columns as float32, ``frame`` as int32, atomically."""
    out = table.copy()
    for c in out.columns:
        if out[c].dtype == np.float64:
            out[c] = out[c].astype(np.float32)
    out["frame"] = out["frame"].astype(np.int32)
    out["phase"] = out["phase"].astype(str)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.parquet")
    pq.write_table(pa.Table.from_pandas(out, preserve_index=False), tmp)
    tmp.replace(path)


def fit_ridges(model: RemainderModel, fit_rows: dict[str, pd.DataFrame], fit_table: pd.DataFrame,
               val_rows: dict[str, pd.DataFrame] | None, val_table: pd.DataFrame | None, windows: dict[str, list[str]]) -> dict:
    """One ridge per window on the filter-passing training frames, scored on the validation
    list; the better window becomes the default."""
    r_columns = [f"r_{j:02d}" for j in range(model.n_components)]

    def joined(rows, table):
        frames = pd.concat(list(rows.values()), ignore_index=True)
        if not (np.array_equal(frames["shot"].to_numpy(), table["shot"].to_numpy())
                and np.array_equal(frames["frame"].to_numpy(), table["frame"].to_numpy())):
            raise ValueError("frame rows and target rows differ in keys")
        keep = table["passes_filter"].to_numpy()
        return frames.loc[keep].reset_index(drop=True), table.loc[keep, r_columns].to_numpy(dtype=np.float64)

    train_frames, train_y = joined(fit_rows, fit_table)
    val = joined(val_rows, val_table) if val_rows is not None else None
    for window, columns in windows.items():
        ridge = model.fit_ridge(window, columns, train_frames, train_y)
        if val is not None:
            model.score(window, "validation", *val)
        log(f"  ridge {window}: training R2 {ridge.scores['train']:.4f}"
            + (f", validation R2 {ridge.scores['validation']:.4f}" if "validation" in ridge.scores else ""))
    default = model.choose_default("validation")
    return {"windows": {w: {"columns": list(r.columns), "scores": r.scores} for w, r in model.ridges.items()},
            "default_window": default, "alpha": model.alpha, "n_components": model.n_components,
            "explained_cumulative": {k: float(model.explained_cumulative[k - 1]) for k in (5, 10, 20)
                                     if k <= len(model.explained_cumulative)}}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, default=tb.DEFAULT_CONFIG)
    parser.add_argument("--lists", nargs="+", default=list(DEFAULT_LISTS),
                        help="lists of the split file to build; 'working' for the two working lists")
    parser.add_argument("--stride", type=int, default=1, help="every stride-th frame of each shot")
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--split-name", default=tb.LABELLED_SPLIT, help="labelled data split the lists refer to")
    parser.add_argument("--data-root", type=Path, default=default_data_root())
    parser.add_argument("--table-dir", type=Path, default=ft.DEFAULT_TABLE_ROOT)
    parser.add_argument("--green-dir", type=Path, default=dataset.DEFAULT_PROCESSED_ROOT / "green")
    parser.add_argument("--response-cache", type=Path, default=ct.DEFAULT_CACHE_DIR)
    parser.add_argument("--references-dir", type=Path, default=bl.DEFAULT_REFERENCE_DIR,
                        help="cached scorer references, read for the shape numbers when they match a list and stride")
    parser.add_argument("--out", type=Path, default=tb.DEFAULT_TARGET_ROOT)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args(argv)

    started = time.time()
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    try:
        tb.require_labelled(args.split_name)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    config = tb.TargetConfig.load(args.config)
    workers = bl.default_workers() if args.workers is None else args.workers
    split = sp.read_split(args.split)
    lists, fit_name, val_name = resolve_lists(args.lists, split)
    out_dir = tb.target_dir(config.version, args.out)
    timing = {}

    green_d3d = GreenTable.load(args.green_dir / "green_d3d.npz")
    schema = set(pq.ParquetFile(ft.table_path(args.split_name, args.table_dir)).schema_arrow.names)
    windows = available_windows(schema, config.rate_windows_ms)
    missing = [w for w in (window_name(ms) for ms in config.rate_windows_ms) if w not in windows]
    columns = list(dict.fromkeys([*targets.current_columns_of(green_d3d, "DIII-D"), "ip__mean", *lf.FILTER_COLUMNS[1:],
                                  "ip_over_peak", "n_frames", *(c for cols in windows.values() for c in cols)]))
    log(f"lists {lists} at stride {args.stride}, fit on {fit_name}; rate windows {list(windows)}"
        + (f" ({missing} not in the frame table)" if missing else "") + f"; {workers} workers")

    t0 = time.time()
    rows = {name: frame_rows(sorted(split[name]), columns, args.table_dir, args.stride) for name in lists}
    keep = {s: lf.lighter_label_filter(g) for s, g in rows[fit_name].items()}
    timing["frames_s"] = time.time() - t0

    t0 = time.time()
    current, remainder_basis, sample = fit_bases(rows[fit_name], keep, config, args, workers)
    remainder = RemainderModel.from_basis(remainder_basis, config.n_remainder, config.ridge_alpha)
    timing["bases_s"] = time.time() - t0
    log(f"current basis: {current.n_components} components ({int((current.parity > 0).sum())} even, "
        f"{int((current.parity < 0).sum())} odd) over {current.n_frames} frames and their mirrors; remainder basis: "
        f"{config.n_remainder} of {remainder_basis.n_components} components")

    t0 = time.time()
    responses, placements, fluxes = flux_objects(current, args)
    timing["flux_bases_s"] = time.time() - t0

    t0 = time.time()
    tables, list_records = {}, {}
    for name in lists:
        if not rows[name]:
            list_records[name] = {"n_shots": 0, "n_rows": 0, "skipped": "empty list"}
            log(f"{name}: empty list, no table")
            continue
        tables[name], list_records[name] = build_table(name, rows[name], current, remainder, config, args, workers)
        write_table(tables[name], tb.table_path(name, config.version, args.out))
        log(f"{name}: {list_records[name]['n_rows']} rows of {list_records[name]['n_shots']} shots")
    timing["tables_s"] = time.time() - t0

    t0 = time.time()
    ridge_record = fit_ridges(remainder, rows[fit_name], tables[fit_name], rows.get(val_name) if val_name in tables else None,
                              tables.get(val_name), windows)
    timing["ridges_s"] = time.time() - t0

    objects = tb.TargetObjects(current, remainder, fluxes, placements)
    checksums = objects.save(out_dir)
    cumulative = np.cumsum(current.explained_ratio)
    record = run_info(argv, vars(args), timestamp, time.time() - started, version=config.version, config=config.to_dict(),
                      config_hash=config.hash(), machine_constants={m: machine_config(m).constants_record() for m in tb.MACHINES},
                      lists=list_records, fit_list=fit_name, validation_list=val_name, stride=args.stride, sample=sample,
                      current_basis={"n_components": current.n_components, "even": int((current.parity > 0).sum()),
                                     "odd": int((current.parity < 0).sum()),
                                     "explained_cumulative": float(cumulative[-1])},
                      remainder=ridge_record, rate_windows_missing=missing,
                      flux_bases={m: {"mean_retained": f.mean_retained, "mean_border_weight": f.mean_border_weight,
                                      "r0_m": f.r0_m, "a0_m": f.a0_m} for m, f in fluxes.items()},
                      responses={m: cb.response_record(r) for m, r in responses.items()},
                      checksums=checksums, tables={n: tb.table_path(n, config.version, args.out).name for n in tables},
                      timing=timing)
    (out_dir / "build.json").write_text(json.dumps(record, indent=2, default=lambda v: v.tolist() if hasattr(v, "tolist") else str(v)) + "\n")
    log(f"wrote {out_dir} in {time.time() - started:.0f} s; default ridge window {ridge_record['default_window']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
