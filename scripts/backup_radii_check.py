"""Diagnostic: the MAST test features at backup reference radii, beside version 1.

The MAST reference radii of ``machines.py`` (R0 0.85 m, a0 0.65 m) are unverified. This
run builds the MAST public test features with other radii (by default R0 0.70 m and
a0 0.50 m, Morris 1998) passed to the build as an override, and compares them with the
version-1 MAST features against the DIII-D training frames. Version 1 is not changed:
``machines.py``, the configuration, the feature tables under ``<feature-root>/v1/`` and
``scaler.json`` stay as they are, DIII-D is not rebuilt, and the version-1 scaler is used
unchanged.

Steps:

1. Layout clearance of MAST at the backup radii (check X4, rule ``min_clearance_m`` of the
   configuration); the run stops when it fails, after writing the clearance to the report.
2. The point table of the moved points, from the cache or built and cached under its own
   key (``coil_pattern.load_or_build_point_table``), and the MAST test features (float32,
   as version 1) written to ``--scratch-dir`` with a ``build.json``. The scratch folder
   must lie outside the version's feature folder.
3. Against the ``train`` list of ``--split``, for MAST at the version-1 radii and at the
   backup radii: X8 ranges of every feature and of the 6b quantities (``vacuum_residual``,
   ``shaping``), all frames and by phase; the vacuum residual relative to ``coil_c20``
   beside the same ratio for the current-free field R^4 - 4 R^2 Z^2 through the same fit
   at each set of radii; X14 (``shafranov_lambda`` against ``ts_beta_p``), all frames and
   flat-top; X10 with its method (``--nn-train-frames``, ``--nn-eval-frames``, ``--seed``)
   for set_a, set_b and set_b with ``shaping`` in place of ``coil_c20`` and ``coil_c02``,
   each with ``ip_frac`` unscaled. MAST at both radii uses the same sampled frames.
   ``shaping`` is scaled with its median and interquartile range on the DIII-D training
   frames, originals and mirrors together (mirror rule even); every other column keeps the
   version-1 statistics.

Writes ``<report-root>/v<version>/backup_radii_check.md``. Usage::

    python scripts/backup_radii_check.py
    python scripts/backup_radii_check.py --r0 0.70 --a0 0.50 --radii-source "Morris 1998"
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from fusion_eq import coil_pattern as cp, dataset, feature_checks as fc, features as fe, frame_table as ft, split as sp
from fusion_eq.coil_flux import GreenTable
from fusion_eq.coil_scales import REFERENCE_PHASES
from fusion_eq.dataset import default_data_root, run_info
from fusion_eq.feature_scaler import RobustScaler
from fusion_eq.machines import machine_config

DEFAULT_SPLIT = dataset.REPO_ROOT / "splits" / "split_v1.json"
MAST_SPLIT = "mast_public_test"
REPORT_NAME = "backup_radii_check.md"
STRETCH = ["vacuum_residual", "shaping"]
# Keys of the compared tables, and their labels in the report.
MAST_V1, MAST_BACKUP = "mast_v1_radii", "mast_backup_radii"
LABEL = {"train": "DIII-D train", "validation": "DIII-D validation", "diii_d_public_test": "DIII-D test",
         MAST_V1: "MAST, v1 radii", MAST_BACKUP: "MAST, backup radii"}


def log(message: str) -> None:
    print(message, flush=True)


def display_path(path: Path | str) -> str:
    """A path relative to the repository, so that the report carries no home path."""
    return os.path.relpath(Path(path).resolve(), dataset.REPO_ROOT)


def default_scratch(feature_root: Path, r0: float, a0: float) -> Path:
    return Path(feature_root) / "scratch" / f"mast_r{r0:.2f}_a{a0:.2f}"


def check_scratch(scratch: Path, version_dir: Path) -> None:
    """Raises SystemExit when the scratch folder is the version's feature folder or inside it."""
    s, v = scratch.resolve(), version_dir.resolve()
    if s == v or v in s.parents:
        raise SystemExit(f"{scratch} lies inside {version_dir}; choose a scratch folder outside it")


# --------------------------------------------------------------------------- clearance
def clearance(config: fe.FeatureConfig, args) -> tuple[dict, pd.DataFrame, float]:
    """X4 of MAST at the backup radii, and the smallest clearance at the version-1 radii."""
    cfg = machine_config("MAST")
    grid = GreenTable.load(args.green_dir / f"green_{cfg.file_prefix.split('_')[0]}.npz")
    geometry, source = cp.geometry_from_split(args.data_root, "MAST")
    if geometry.hash() != grid.geometry_hash:
        raise SystemExit(f"geometry of {source} differs from the grid table's")
    table = cp.clearance_table(config.layout, cfg, geometry, grid.grid_r, grid.grid_z, args.r0, args.a0)
    v1 = cp.clearance_table(config.layout, cfg, geometry, grid.grid_r, grid.grid_z)
    result = cp.check_layout(config.layout, {"MAST": table})
    result["geometry_source"] = display_path(source)
    return result, table, float(v1["clearance_m"].min())


# ------------------------------------------------------------------------------- build
def build_mast(config: fe.FeatureConfig, resources: fe.MachineResources, args) -> tuple[Path, dict]:
    """The MAST test features at the backup radii, written to the scratch folder (float32)."""
    reads = sorted({c for cols in fe.feature_reads(config, "MAST").values() for c in cols})
    shots = sorted(ft.read_frame_table(MAST_SPLIT, ["shot"], root=args.table_dir)["shot"].astype(str).unique().tolist())
    columns = config.all_columns()
    schema = pa.schema([pa.field("shot", pa.string()), pa.field("frame", pa.int32()), *[pa.field(c, pa.float32()) for c in columns]])
    args.scratch_dir.mkdir(parents=True, exist_ok=True)
    path = args.scratch_dir / f"{MAST_SPLIT}.parquet"
    tmp = path.with_suffix(".tmp.parquet")
    writer = pq.ParquetWriter(tmp, schema)
    n_rows = 0
    for start in range(0, len(shots), args.shots_per_chunk):
        chunk = shots[start:start + args.shots_per_chunk]
        frames = ft.read_frame_table(MAST_SPLIT, ["shot", "frame", *reads], shots=chunk, root=args.table_dir)
        out = fe.build_features(frames, "MAST", config, resources)
        out["frame"] = out["frame"].astype(np.int32)
        out[columns] = out[columns].astype(np.float32)
        writer.write_table(pa.Table.from_pandas(out, schema=schema, preserve_index=False), row_group_size=len(out))
        n_rows += len(out)
    writer.close()
    tmp.replace(path)
    keys = ft.read_frame_table(MAST_SPLIT, ["shot", "frame"], root=args.table_dir)
    built = pq.read_table(path, columns=["shot", "frame"]).to_pandas()
    match = len(keys) == len(built) and np.array_equal(keys["shot"].astype(str).to_numpy(), built["shot"].to_numpy()) \
        and np.array_equal(keys["frame"].to_numpy(), built["frame"].to_numpy())
    return path, {"n_shots": len(shots), "n_rows": n_rows, "x1_rows_match": bool(match)}


# ------------------------------------------------------------------------------- load
def joined(features: pd.DataFrame, split: str, args, shots=None) -> pd.DataFrame:
    """Feature rows with ``n_frames`` and ``phase`` (X9 rule), keys checked against the frame table."""
    frames = ft.read_frame_table(split, ["shot", "frame"], shots=shots, root=args.table_dir)
    try:
        return fc.join_frame_columns(features, frames)
    except ValueError as exc:
        raise SystemExit(f"{split}: {exc}") from exc


def load_tables(config: fe.FeatureConfig, mast_path: Path, args) -> dict[str, pd.DataFrame]:
    columns = config.all_columns()
    split = sp.read_split(args.split)
    d3d = fe.read_features("diii_d_train", config.version, columns, None, args.feature_root)
    d3d = joined(d3d, "diii_d_train", args)
    out = {"train": d3d[d3d["shot"].isin(set(split["train"]))].reset_index(drop=True),
           "validation": d3d[d3d["shot"].isin(set(split["validation"]))].reset_index(drop=True)}
    del d3d
    out["diii_d_public_test"] = joined(fe.read_features("diii_d_public_test", config.version, columns, None, args.feature_root),
                                       "diii_d_public_test", args)
    out[MAST_V1] = joined(fe.read_features(MAST_SPLIT, config.version, columns, None, args.feature_root), MAST_SPLIT, args)
    out[MAST_BACKUP] = joined(pq.read_table(mast_path, columns=["shot", "frame", *columns]).to_pandas(), MAST_SPLIT, args)
    if not (out[MAST_V1][["shot", "frame"]].equals(out[MAST_BACKUP][["shot", "frame"]])
            and np.array_equal(out[MAST_V1]["phase"].to_numpy(), out[MAST_BACKUP]["phase"].to_numpy())):
        raise SystemExit("MAST rows or phases differ between the two radii")
    return out


def add_stretch(tables: dict[str, pd.DataFrame], args) -> None:
    for key, t in tables.items():
        machine = "MAST" if key in (MAST_V1, MAST_BACKUP) else "DIII-D"
        radii = (args.r0, args.a0) if key == MAST_BACKUP else (None, None)
        s = fc.stretch_quantities(t, machine, *radii)
        t["vacuum_residual"], t["shaping"] = s["vacuum_residual"].to_numpy(), s["shaping"].to_numpy()


# ----------------------------------------------------------------------- comparisons
def residual_ratios(tables: dict[str, pd.DataFrame], config: fe.FeatureConfig, args) -> pd.DataFrame:
    """Median abs(vacuum_residual) over median abs(coil_c20), all frames and by phase, beside
    the ratio of the current-free field R^4 - 4 R^2 Z^2 through the same fit at the same radii."""
    rows = []
    for key in ("train", MAST_V1, MAST_BACKUP):
        machine = "DIII-D" if key == "train" else "MAST"
        cfg = machine_config(machine)
        r0, a0 = (args.r0, args.a0) if key == MAST_BACKUP else (cfg.major_radius.value, cfg.minor_radius.value)
        quartic = pd.DataFrame([fc.fitted_terms(config.layout, lambda r, z: r ** 4 - 4 * r ** 2 * z ** 2, r0, a0)])
        benchmark = abs(fc.vacuum_residual(quartic, cfg, r0, a0)[0]) / abs(quartic["coil_c20"].iloc[0])
        t = tables[key]
        for phase in fc.PHASES:
            p = t if phase == "all" else t[t["phase"] == phase]
            rows.append({"split": LABEL[key], "R0 (m)": r0, "a0 (m)": a0, "phase": phase,
                         "n": int(p["vacuum_residual"].notna().sum()),
                         "ratio": fc.relative_size(p["vacuum_residual"], p["coil_c20"]),
                         "ratio for R^4 - 4 R^2 Z^2": benchmark})
    return pd.DataFrame(rows)


def stretch_statistics(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for q in [*STRETCH, "coil_c20", "coil_c02"]:
        for key in ("train", MAST_V1, MAST_BACKUP):
            s = fc.reference_statistics(tables[key][q])
            rows.append({"quantity": q, "split": LABEL[key], "n": s["n"], "median": s["q50"], "IQR": s["q75"] - s["q25"]})
    return pd.DataFrame(rows)


def x14(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for subset in ("all", "flat_top"):
        reference = None
        for key in ("train", MAST_V1, MAST_BACKUP):
            t = tables[key] if subset == "all" else tables[key][tables[key]["phase"] == "flat_top"]
            fit = fc.line_fit(t["ts_beta_p"], t["shafranov_lambda"])
            reference = fit if key == "train" else reference
            rows.append({"frames": subset, "split": LABEL[key], "n": fit["n"], "slope": fit["slope"],
                         "intercept": fit["intercept"], "r": fit["r"], "residual std": fit["residual_std"],
                         "median offset from training line": fc.median_offset(t["ts_beta_p"], t["shafranov_lambda"], reference)})
    return pd.DataFrame(rows)


def x10(tables: dict[str, pd.DataFrame], config: fe.FeatureConfig, scaler: RobustScaler, args) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Combined distance of the three variants, ``ip_frac`` unscaled, MAST at both radii on the same frames."""
    train = tables["train"]
    mirrored = fe.mirror_features(train[["coil_c20", "coil_c02"]], config)
    shaping_scaler = RobustScaler.fit_rules(train[["shaping"]], pd.DataFrame({"shaping": fc.shaping(mirrored)}),
                                            {"shaping": dict(fc.ROBUST_RULE)})
    variant_scaler = scaler.with_identity(["ip_frac"]).extended(shaping_scaler)
    set_a, set_b = config.set_columns("set_a"), config.set_columns("set_b")
    variants = {"set_a, ip_frac unscaled": set_a, "set_b, ip_frac unscaled": set_b,
                "set_b, shaping for coil_c20 and coil_c02, ip_frac unscaled":
                    ["shaping" if c == "coil_c20" else c for c in set_b if c != "coil_c02"]}
    group_of = {**{c: s.group for c, s in config.column_specs().items()}, "shaping": "coil_terms"}
    compared = {g: len(tables[g]) for g in fc.COMPARED_SPLITS[:2]}
    compared[MAST_SPLIT] = len(tables[MAST_V1])
    train_rows, eval_rows = fc.x10_samples(len(train), compared, args.nn_train_frames, args.nn_eval_frames, args.seed)
    eval_rows[MAST_V1] = eval_rows[MAST_BACKUP] = eval_rows.pop(MAST_SPLIT)
    parts = []
    for name, columns in variants.items():
        reference = variant_scaler.transform(train.iloc[train_rows], "DIII-D", columns)
        queries = {LABEL[k]: variant_scaler.transform(tables[k].iloc[rows], "MAST" if k in (MAST_V1, MAST_BACKUP) else "DIII-D",
                                                      columns) for k, rows in eval_rows.items()}
        parts.append(fc.combined_distance(reference, queries, fc.column_groups(columns, group_of)).assign(set=name))
        log(f"  X10 {name}: {len(columns)} columns, {len(reference)} training frames")
    table = pd.concat(parts, ignore_index=True)
    info = {"n_train_sample": int(len(train_rows)), "n_eval": {LABEL[k]: int(len(v)) for k, v in eval_rows.items()},
            "shaping_statistics": shaping_scaler.source["shaping"]}
    return fc.distance_summary(table, reference_split=LABEL["validation"]), fc.group_share_medians(table), info


# ------------------------------------------------------------------------------ report
def render(info: dict, clear: pd.DataFrame, ranges: pd.DataFrame | None, ratios: pd.DataFrame | None,
           stretch: pd.DataFrame | None, lines14: pd.DataFrame | None, nn: pd.DataFrame | None,
           shares: pd.DataFrame | None) -> str:
    a = info["arguments"]
    c = info["clearance"]
    lines = [f"# MAST features at backup reference radii, version {info['version']}", "",
             f"Diagnostic. MAST public test features built with R0 {a['r0']:g} m and a0 {a['a0']:g} m ({a['radii_source']}) "
             f"passed to the build as an override, beside the version-1 MAST features (R0 {info['v1_radii']['r0_m']:g} m, "
             f"a0 {info['v1_radii']['a0_m']:g} m, unverified in `machines.py`). Version 1 is unchanged: `machines.py`, the "
             f"configuration, the version-1 feature tables and `scaler.json`; DIII-D is not rebuilt. Reference of every "
             f"comparison: the `train` list of `{info['split_file']}`.", "",
             "Features that change with the radii: the coil points (moved, and divided by mu0 R0 Ip) and the named terms; "
             "`ts_beta_p`, `greenwald_frac` and the resistive time (`ip_rate_rel`, `resistive_age`) through a0^2; `q_cyl` "
             "through a0^2 / R0^2; `shafranov_lambda` through R0 / a0; `ohmic_flux_used` through R0 and the reference point. "
             "`ip_frac`, `ts_pe_peaking` and `ts_valid` do not depend on them.", "",
             "## 1. Layout clearance of MAST at the backup radii (X4)", "",
             f"Status **{c['status']}**: smallest clearance {c['min_clearance_m']['MAST']:.3f} m at the backup radii, "
             f"{info['v1_min_clearance_m']:.3f} m at the version-1 radii (rule {info['min_clearance_rule_m']:g} m); layout "
             f"symmetric {c['symmetric']}; failing points {len(c['failing'])}. Geometry from `{c['geometry_source']}`. "
             "Smallest clearances:", "",
             fc.markdown_table(clear.nsmallest(6, "clearance_m")[["column", "x", "y", "R_m", "Z_m", "clearance_m",
                                                                  "nearest_column", "in_grid", "ok"]])]
    if info.get("stopped"):
        lines += ["", f"Stopped: {info['stopped']}", ""]
        return "\n".join(lines)
    b = info["build"]
    lines += ["", "## 2. Build", "",
              f"{b['n_shots']} MAST shots, {b['n_rows']} rows, keys equal the frame table: {b['x1_rows_match']}. Features in "
              f"`{info['scratch_table']}` (float32, as version 1). Point table `{Path(info['point_table']['path']).name}` "
              f"({'built' if info['point_table']['built'] else 'from the cache'}), points hash `{info['points_hash'][:12]}`.", ""]

    lines += ["## 3. Ranges against DIII-D training (X8), all frames", "",
              "Share of frames outside the training 1st to 99th percentile band and outside the training range, and the "
              "median shift in training interquartile ranges, for MAST at both radii. `vacuum_residual` uses each build's "
              "own radii.", ""]
    allp = ranges[ranges["phase"] == "all"]
    rows = []
    for feature, g in allp.groupby("feature", sort=False):
        v1, bk = g[g["split"] == MAST_V1].iloc[0], g[g["split"] == MAST_BACKUP].iloc[0]
        rows.append({"feature": feature, "train 1st": v1["ref_q01"], "train 99th": v1["ref_q99"],
                     "v1 median": v1["median"], "backup median": bk["median"],
                     "v1 outside band": v1["share_outside_band"], "backup outside band": bk["share_outside_band"],
                     "v1 outside range": v1["share_outside_range"], "backup outside range": bk["share_outside_range"],
                     "v1 shift (IQR)": v1["median_shift_iqr"], "backup shift (IQR)": bk["median_shift_iqr"]})
    lines += [fc.markdown_table(pd.DataFrame(rows)), "", "### By phase", "",
              "Reference restricted to the same phase (X9 rule).", ""]
    rows = []
    for (feature, phase), g in ranges[ranges["phase"] != "all"].groupby(["feature", "phase"], sort=False):
        v1, bk = g[g["split"] == MAST_V1].iloc[0], g[g["split"] == MAST_BACKUP].iloc[0]
        rows.append({"feature": feature, "phase": phase, "MAST frames": v1["n_finite"],
                     "v1 outside band": v1["share_outside_band"], "backup outside band": bk["share_outside_band"],
                     "v1 outside range": v1["share_outside_range"], "backup outside range": bk["share_outside_range"],
                     "v1 shift (IQR)": v1["median_shift_iqr"], "backup shift (IQR)": bk["median_shift_iqr"]})
    order = {f: i for i, f in enumerate(allp["feature"].drop_duplicates())}
    by_phase = pd.DataFrame(rows)
    by_phase = by_phase.assign(_f=by_phase["feature"].map(order), _p=by_phase["phase"].map(
        {p: i for i, p in enumerate(REFERENCE_PHASES)})).sort_values(["_f", "_p"], kind="stable").drop(columns=["_f", "_p"])
    lines += [fc.markdown_table(by_phase)]

    lines += ["", "## 4. Stretch terms and vacuum relation (6b quantities)", "",
              "Median abs(`vacuum_residual`) over median abs(`coil_c20`), beside the same ratio for the current-free field "
              "R^4 - 4 R^2 Z^2 sampled at the layout points and fitted the same way at the same radii (the part of the "
              "residual that comes from the fit alone, for that field).", "", fc.markdown_table(ratios), "",
              "Median and interquartile range:", "", fc.markdown_table(stretch)]

    lines += ["", "## 5. Thomson against force balance (X14)", "",
              "Line `shafranov_lambda = intercept + slope ts_beta_p` over frames inside both 1st to 99th percentile bands of "
              "each split; median offset from the DIII-D training line of the same frames (all or flat-top).", "",
              fc.markdown_table(lines14)]

    x = info["x10"]
    lines += ["", "## 6. Combined distance (X10 method), ip_frac unscaled", "",
              f"{x['n_train_sample']} DIII-D training frames (seed {a['seed']}); "
              + ", ".join(f"{k} {n}" for k, n in x["n_eval"].items())
              + " frames (MAST at both radii on the same frames). Version-1 statistics for every column but `ip_frac` "
                f"(identity) and `shaping` (DIII-D training, originals and mirrors: median {x['shaping_statistics'][0]:.5g}, "
                f"interquartile range {x['shaping_statistics'][1]:.5g}).", ""]
    beyond = f"share_beyond_{LABEL['validation']}_p99"
    table = nn.rename(columns={"set": "variant", "p50": "median", "p90": "90th", "p99": "99th",
                               beyond: "beyond validation 99th"})
    lines += [fc.markdown_table(table[["variant", "split", "n", "median", "90th", "99th", "max", "beyond validation 99th"]]),
              "", "Median share of the squared distance per feature group:", "",
              fc.markdown_table(shares.rename(columns={"set": "variant", **{c: c[6:] for c in shares.columns if c.startswith("share_")}})),
              "", f"Runtime {info['runtime_s']:.0f} s.", ""]
    return "\n".join(lines)


# -------------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--r0", type=float, default=0.70, help="MAST reference major radius in metres")
    parser.add_argument("--a0", type=float, default=0.50, help="MAST reference minor radius in metres")
    parser.add_argument("--radii-source", default="Morris 1998")
    parser.add_argument("--config", type=Path, default=fe.DEFAULT_CONFIG)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--feature-root", type=Path, default=fe.DEFAULT_FEATURE_ROOT)
    parser.add_argument("--scratch-dir", type=Path, default=None,
                        help="folder of the backup-radii MAST table; <feature-root>/scratch/mast_r<R0>_a<a0> by default")
    parser.add_argument("--report-root", type=Path, default=fe.DEFAULT_REPORT_ROOT)
    parser.add_argument("--table-dir", type=Path, default=ft.DEFAULT_TABLE_ROOT)
    parser.add_argument("--green-dir", type=Path, default=fe.DEFAULT_GREEN_DIR)
    parser.add_argument("--data-root", type=Path, default=default_data_root())
    parser.add_argument("--shots-per-chunk", type=int, default=200)
    parser.add_argument("--nn-train-frames", type=int, default=200_000)
    parser.add_argument("--nn-eval-frames", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    started = time.time()
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    config = fe.FeatureConfig.load(args.config)
    version_dir = fe.features_dir(config.version, args.feature_root)
    args.scratch_dir = args.scratch_dir or default_scratch(args.feature_root, args.r0, args.a0)
    check_scratch(args.scratch_dir, version_dir)
    out_dir = Path(args.report_root) / f"v{config.version}"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / REPORT_NAME
    scaler = RobustScaler.load(fe.scaler_path(config.version, args.report_root))
    cfg = machine_config("MAST")
    info = {"version": config.version, "config_hash": config.hash(), "split_file": display_path(args.split),
            "v1_radii": {"r0_m": cfg.major_radius.value, "a0_m": cfg.minor_radius.value},
            "min_clearance_rule_m": config.layout.min_clearance_m}

    result, clear, v1_min = clearance(config, args)
    info.update(clearance=result, v1_min_clearance_m=v1_min)
    log(f"layout clearance of MAST at R0 {args.r0:g} m, a0 {args.a0:g} m: {result['status']}, smallest "
        f"{result['min_clearance_m']['MAST']:.3f} m (version-1 radii {v1_min:.3f} m)")
    if result["status"] != "pass":
        info.update(stopped="the layout fails the clearance rule at the backup radii; no feature was built",
                    **run_info(argv, vars(args), timestamp, time.time() - started))
        report_path.write_text(render(info, clear, None, None, None, None, None, None))
        raise SystemExit(f"layout clearance fails at the backup radii; see {report_path}")

    resources = fe.load_resources("MAST", config, args.data_root, args.green_dir, r0_m=args.r0, a0_m=args.a0,
                                  radii_source=args.radii_source)
    mast_path, build = build_mast(config, resources, args)
    record = {"config_hash": config.hash(), "radii": resources.record["radii_override"], "build": build,
              "point_table": resources.record["point_table"], "points_hash": resources.record["points_hash"],
              "clearance": result}
    (args.scratch_dir / "build.json").write_text(json.dumps(record, indent=2, default=fc.json_default) + "\n")
    info.update(build=build, scratch_table=display_path(mast_path), point_table=resources.record["point_table"],
                points_hash=resources.record["points_hash"])
    log(f"MAST features at the backup radii: {build['n_rows']} rows of {build['n_shots']} shots in {mast_path}")
    if not build["x1_rows_match"]:
        raise SystemExit("the backup-radii MAST rows differ from the frame table (X1)")

    tables = load_tables(config, mast_path, args)
    add_stretch(tables, args)
    columns = [*config.all_columns(), *STRETCH]
    compared = (MAST_V1, MAST_BACKUP)
    ranges = fc.compare_ranges_by_phase(tables["train"][columns], tables["train"]["phase"],
                                        {k: tables[k][columns] for k in compared}, {k: tables[k]["phase"] for k in compared},
                                        columns)
    ratios = residual_ratios(tables, config, args)
    stretch = stretch_statistics(tables)
    lines14 = x14(tables)
    log("X8, 6b and X14 done")
    nn, shares, info["x10"] = x10(tables, config, scaler, args)
    info.update(run_info(argv, vars(args), timestamp, time.time() - started))
    report_path.write_text(render(info, clear, ranges, ratios, stretch, lines14, nn, shares))
    log(f"wrote {report_path} in {time.time() - started:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
