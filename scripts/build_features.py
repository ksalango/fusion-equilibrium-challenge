"""Builds the version's feature tables from the frame table (``docs/FEATURES.md``).

Steps:

1. Per machine: the point table of the layout (from the cache, or built from the geometry
   columns of the first shot of the machine's public test folder, checked against the grid
   table's geometry hash), the adopted coil model and the machine constants. The layout
   clearances are calculated on both machines (check X4); the build stops when the layout
   fails, and writes ``layout_clearance.csv`` either way.
2. Per split: the frame rows, read a chunk of shots at a time with only the columns the
   features declare (check X2), are turned into unscaled features (float32) and written to
   ``<out>/v<version>/<split>.parquet``, shots sorted by id, frames in order.
3. The robust scaler is fitted on the ``train`` list of ``--split``, originals and mirrors
   together, and written to ``reports/features/v<version>/scaler.json`` (committed).
4. ``<out>/v<version>/build.json``: commits, arguments, the configuration and its hash,
   the machine constants with their verified status, the coil model of each machine by name,
   the point tables, the X1 counts (feature rows against frame-table rows per split), the X2
   reads and the X4 result.

Coil models (``coil_flux.COIL_MODELS``): each machine's default unless ``--coil-model
MACHINE=NAME`` names another. A build into a folder whose ``build.json`` records another coil
model for a machine it rebuilds stops, so that a stored table never changes model in place.
``--variant NAME`` writes to ``<out>/v<version>_<NAME>`` (layout clearance to
``<report-root>/v<version>_<NAME>``) and fits no scaler: a variant build is read with the
version's scaler, fitted on DIII-D training.

Requires the frame table built with Part 0 (``__rate50``, ``ip_rise_ms``, ``sol_at_rise``).
Nothing is written under the raw data root. Usage::

    python scripts/build_features.py --config configs/features_v1.yaml
    python scripts/build_features.py --config configs/features_v2.yaml
    python scripts/build_features.py --config configs/features_v2.yaml --splits mast_public_test \
        --coil-model MAST=sol_half --variant solhalf
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

from fusion_eq import coil_pattern as cp, dataset, features as fe, frame_table as ft, split as sp
from fusion_eq.coil_flux import DEFAULT_COIL_MODELS, GreenTable, coil_model_names
from fusion_eq.dataset import default_data_root, read_shots_file, run_info
from fusion_eq.feature_scaler import RobustScaler
from fusion_eq.machines import machine_config

DEFAULT_SPLIT = dataset.REPO_ROOT / "splits" / "split_v1.json"
SCALER_SPLIT = "diii_d_train"


def log(message: str) -> None:
    print(message, flush=True)


def feature_schema(columns: list[str]) -> pa.Schema:
    return pa.schema([pa.field("shot", pa.string()), pa.field("frame", pa.int32()),
                      *[pa.field(c, pa.float32()) for c in columns]])


def split_shots(split: str, table_dir: Path) -> list[str]:
    """Shot ids of a split's frame table, sorted."""
    return sorted(ft.read_frame_table(split, ["shot"], root=table_dir)["shot"].astype(str).unique().tolist())


def check_frame_table(split: str, table_dir: Path, needed: list[str]) -> None:
    names = set(pq.ParquetFile(ft.table_path(split, table_dir)).schema_arrow.names)
    missing = [c for c in needed if c not in names]
    if missing:
        raise SystemExit(f"{ft.table_path(split, table_dir)} lacks {missing}; rebuild the frame table with Part 0 "
                         f"(python scripts/build_frame_table.py)")


def layout_check(config: fe.FeatureConfig, args) -> tuple[dict, pd.DataFrame]:
    """Clearance tables of both machines (X4), from the geometry of one public test shot each."""
    tables, sources = {}, {}
    for machine in ("DIII-D", "MAST"):
        cfg = machine_config(machine)
        grid = GreenTable.load(args.green_dir / f"green_{cfg.file_prefix.split('_')[0]}.npz")
        geometry, source = cp.geometry_from_split(args.data_root, machine)
        if geometry.hash() != grid.geometry_hash:
            raise SystemExit(f"geometry of {source} differs from the grid table's")
        tables[machine] = cp.clearance_table(config.layout, cfg, geometry, grid.grid_r, grid.grid_z)
        sources[machine] = source
    result = cp.check_layout(config.layout, tables)
    result["geometry_source"] = sources
    return result, pd.concat(tables.values(), ignore_index=True)


def build_split(split: str, config: fe.FeatureConfig, resources: fe.MachineResources, args,
                shot_filter: list[str] | None, variant: str | None = None) -> dict:
    """Writes one split's feature table (to the variant folder when ``variant``); returns its
    counts."""
    machine = dataset.SPLIT_MACHINE[split]
    reads = sorted({c for cols in fe.feature_reads(config, machine).values() for c in cols})
    check_frame_table(split, args.table_dir, reads)
    shots = split_shots(split, args.table_dir)
    if shot_filter is not None:
        shots = [s for s in shots if s in set(shot_filter)]
    columns = config.all_columns()
    path = fe.features_path(split, config.version, args.out, variant)
    tmp = path.with_suffix(".tmp.parquet")
    writer = pq.ParquetWriter(tmp, feature_schema(columns))
    n_rows, n_table_rows = 0, 0
    for start in range(0, len(shots), args.shots_per_chunk):
        chunk = shots[start:start + args.shots_per_chunk]
        frames = ft.read_frame_table(split, ["shot", "frame", *reads], shots=chunk, root=args.table_dir)
        out = fe.build_features(frames, machine, config, resources)
        out["frame"] = out["frame"].astype(np.int32)
        out[columns] = out[columns].astype(np.float32)
        writer.write_table(pa.Table.from_pandas(out, schema=feature_schema(columns), preserve_index=False),
                           row_group_size=len(out))
        n_rows += len(out)
        n_table_rows += len(frames)
        log(f"  {split}: {min(start + args.shots_per_chunk, len(shots))}/{len(shots)} shots, {n_rows} rows")
    writer.close()
    tmp.replace(path)
    if shot_filter is None:
        n_table_rows = pq.ParquetFile(ft.table_path(split, args.table_dir)).metadata.num_rows
    keys_match = n_rows == n_table_rows
    if keys_match and shot_filter is None:
        table_keys = ft.read_frame_table(split, ["shot", "frame"], root=args.table_dir)
        built_keys = pq.read_table(path, columns=["shot", "frame"]).to_pandas()
        keys_match = (table_keys["shot"].astype(str).to_numpy() == built_keys["shot"].to_numpy()).all() and \
            (table_keys["frame"].to_numpy() == built_keys["frame"].to_numpy()).all()
    return {"n_shots": len(shots), "n_rows": n_rows, "n_frame_table_rows": n_table_rows,
            "x1_rows_match": bool(keys_match)}


def parse_coil_models(items: list[str] | None) -> dict[str, str]:
    """Machine to coil-model name: the defaults, with ``MACHINE=NAME`` items applied.

    Raises:
        SystemExit: for a malformed item, an unknown machine or an unregistered name.
    """
    models = dict(DEFAULT_COIL_MODELS)
    for item in items or []:
        machine, sep, name = item.partition("=")
        if not sep or machine not in models:
            raise SystemExit(f"--coil-model {item!r}: expected MACHINE=NAME with MACHINE one of {sorted(models)}")
        if name not in coil_model_names(machine):
            raise SystemExit(f"--coil-model {item!r}: no model {name!r} on {machine}; known: {list(coil_model_names(machine))}")
        models[machine] = name
    return models


def check_existing_build(out_dir: Path, splits: list[str], coil_models: dict[str, str]) -> None:
    """Stops when ``out_dir`` holds a build whose coil model differs on a machine being rebuilt."""
    for machine in sorted({dataset.SPLIT_MACHINE[s] for s in splits}):
        recorded = fe.build_coil_model(out_dir, machine)
        if recorded is not None and recorded != coil_models[machine]:
            raise SystemExit(f"{out_dir} was built with the {machine} coil model {recorded!r}, not {coil_models[machine]!r}; "
                             f"pass --coil-model {machine}={recorded} to rebuild it, or --variant to write a separate build")


def fit_scaler(config: fe.FeatureConfig, args, shot_filter: list[str] | None) -> tuple[RobustScaler, dict]:
    """Robust scaler on the training list's frames, originals and mirrors together."""
    train = sorted(sp.read_split(args.split)["train"])
    if shot_filter is not None:
        train = [s for s in train if s in set(shot_filter)]
    table = fe.read_features(SCALER_SPLIT, config.version, config.all_columns(), train, args.out)
    found = int(table["shot"].nunique())
    scaler = RobustScaler.fit(table, fe.mirror_features(table, config), config)
    return scaler, {"split_file": str(args.split), "n_train_shots_listed": len(train), "n_train_shots_found": found,
                    "n_rows_with_mirrors": scaler.n_rows}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, default=fe.DEFAULT_CONFIG)
    parser.add_argument("--splits", nargs="+", default=list(dataset.DATA_SPLITS), choices=dataset.DATA_SPLITS)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT, help="split file whose train list fits the scaler")
    parser.add_argument("--shots-file", type=Path, default=None, help="JSON list of shot ids; only these are built")
    parser.add_argument("--data-root", type=Path, default=default_data_root())
    parser.add_argument("--table-dir", type=Path, default=ft.DEFAULT_TABLE_ROOT)
    parser.add_argument("--green-dir", type=Path, default=fe.DEFAULT_GREEN_DIR, help="grid tables and the point-table cache")
    parser.add_argument("--out", type=Path, default=fe.DEFAULT_FEATURE_ROOT)
    parser.add_argument("--report-root", type=Path, default=fe.DEFAULT_REPORT_ROOT)
    parser.add_argument("--shots-per-chunk", type=int, default=500)
    parser.add_argument("--coil-model", action="append", default=None, metavar="MACHINE=NAME",
                        help="coil model of a machine (coil_flux.COIL_MODELS); repeat per machine; the defaults otherwise")
    parser.add_argument("--variant", default=None, help="write to <out>/v<version>_<variant>; no scaler is fitted")
    args = parser.parse_args(argv)

    started = time.time()
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    config = fe.FeatureConfig.load(args.config)
    shot_filter = read_shots_file(args.shots_file) if args.shots_file else None
    coil_models = parse_coil_models(args.coil_model)
    out_dir = fe.features_dir(config.version, args.out, args.variant)
    report_dir = Path(args.report_root) / out_dir.name
    check_existing_build(out_dir, args.splits, coil_models)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    layout, clearance = layout_check(config, args)
    clearance.to_csv(report_dir / "layout_clearance.csv", index=False, float_format="%.6f")
    log(f"layout: {layout['status']}, smallest clearance " +
        ", ".join(f"{m} {v:.3f} m" for m, v in layout["min_clearance_m"].items()))
    if layout["status"] != "pass":
        raise SystemExit(f"the layout fails the clearance rule: {layout['failing']}")

    resources = {m: fe.load_resources(m, config, args.data_root, args.green_dir, coil_model=coil_models[m])
                 for m in ("DIII-D", "MAST")}
    for machine, res in resources.items():
        unverified = [k for k, v in res.record["constants"].items() if not v["verified"]]
        log(f"{machine}: coil model {res.scales.name}; point table "
            f"{'built' if res.record['point_table']['built'] else 'from the cache'}; "
            f"unverified constants: {', '.join(unverified) or 'none'}")

    counts = {}
    for split in args.splits:
        counts[split] = build_split(split, config, resources[dataset.SPLIT_MACHINE[split]], args, shot_filter,
                                    variant=args.variant)
        log(f"{split}: {counts[split]['n_rows']} rows over {counts[split]['n_shots']} shots")

    scaler_info = None
    if args.variant is not None:
        scaler_info = {"note": f"variant build: no scaler fitted; read with {fe.scaler_path(config.version, args.report_root)}"}
    elif SCALER_SPLIT in args.splits:
        scaler, scaler_info = fit_scaler(config, args, shot_filter)
        scaler.save(fe.scaler_path(config.version, args.report_root))
        log(f"scaler: {len(scaler.columns)} columns over {scaler.n_rows} rows (originals and mirrors)")

    info = run_info(argv, vars(args), timestamp, time.time() - started, config=config.to_dict(),
                    config_hash=config.hash(), variant=args.variant, coil_models=coil_models,
                    resources={m: r.record for m, r in resources.items()},
                    layout_check=layout, counts=counts,
                    x1_pass=all(c["x1_rows_match"] for c in counts.values()),
                    x2_reads={m: fe.feature_reads(config, m) for m in ("DIII-D", "MAST")},
                    x2_rule="every column read is in frame_table.input_columns(machine, include_flags=True) or is time_ms",
                    scaler=scaler_info)
    (out_dir / "build.json").write_text(json.dumps(info, indent=2, default=str) + "\n")
    log(f"wrote {out_dir} in {time.time() - started:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
