"""Tests the coil flux against true DIII-D maps and fits the coil scales in every variant.

Analyses D3b, D4, D9 and D5 of ``docs/COIL_FLUX.md``. Reads the Green's tables from
``--green-dir``, a sample of DIII-D training shots (and the eight reference shots of
D3, chosen from the frame table's shot index by ``coil_scales.reference_shots``, the
same rule the review notebook uses) and 20 MAST test shots. Every frame is reduced
to its normal equations (``coil_scales.FrameNormal``) on the nodes away from the
coils and on the nodes near a coil, so that the eight fit variants of
``coil_scales.standard_variants`` are solved from the fit frames and evaluated on
every frame without keeping the maps. The adopted DIII-D coil model (``solenoid_flat``:
shaping coils as stored plus a flat solenoid term) is the headline of the report and is
also checked on validation shots of ``--split`` that took no part in the fit. Writes to
``--out``:

- ``coil_frames.parquet``: one row per analysed frame (reference, fit and validation
  shots) with the current ratio (K3), the orientation, the X-point class, and per
  variant the rms away from and near the coils (K5) and the mean residual; the
  remainder metrics (K7) on the fit frames.
- ``coil_scales.parquet``: one row per variant and parameter (the column scales and
  the flat solenoid term), with the per-shot spread of the per-coil variant (K6).
- ``k5_by_phase.parquet``: K5 quantiles per variant, split, node set and phase.
- ``scales/d3d_<variant>.json``: the ``CoilScales`` record of each variant, which
  ``GreenTable.flux`` applies; ``scales/mast_default.json`` is the unfitted record.
- ``coil_flux_report.md``: K3 to K9 with status pass, fail or review.
- ``fit.json``: provenance.

Labels (true maps, stored axis) are read for DIII-D training shots only and serve as
test targets. No MAST label is read. Usage::

    python scripts/fit_coil_scales.py
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from fusion_eq import coil_scales as cs, io, split as sp, starter_kit
from fusion_eq.coil_flux import (ADOPTED_FILES, GEOMETRY_COLUMNS, CoilGeometry, CoilScales, GreenTable,
                                 adopted_scales, currents_at_frames, near_coil_mask)
from fusion_eq.dataset import REPO_ROOT, default_data_root, filter_shots, list_shots, run_info, select_shots
from fusion_eq.frame_table import DEFAULT_TABLE_ROOT
from fusion_eq.machines import machine_config

DEFAULT_GREEN_DIR = REPO_ROOT.parent / "data" / "processed" / "green"
DEFAULT_OUT = REPO_ROOT / "reports" / "coil_flux"
DEFAULT_SHOT_INDEX = DEFAULT_TABLE_ROOT / "shot_index.parquet"
DEFAULT_SPLIT = REPO_ROOT / "reports" / "split" / "split_provisional.json"
FLAT_TOP = cs.FLAT_TOP_LEVEL
K5_REVIEW_FRACTION = 0.05
# The adopted DIII-D coil model (decision of 10 October 2026): shaping coils as stored, the
# solenoid as a flat term. ``per_coil`` keeps the per-shot spread diagnostic (K6).
HEADLINE_VARIANT = "solenoid_flat"
SPREAD_VARIANT = "per_coil"
PHASES = ("rising", "flat_top", "late")


def frame_sample(n_frames: int, n: int) -> np.ndarray:
    return np.unique(np.round(np.linspace(0, n_frames - 1, min(n, n_frames))).astype(int))


phase_of = cs.phase_of


def analyse_shot(path: Path, split: str, table: GreenTable, cfg, env, scorer, near, args, orientation_c=None):
    """Per-frame rows, normal equations and light records of one labelled shot.

    Returns:
        ``(shot, rows, normals, records, grid_r, grid_z)``: ``normals`` holds one
        ``(fit nodes, near-coil nodes)`` pair of ``FrameNormal`` per frame; ``records``
        the oriented map and currents of each frame, for the remainder metrics.
    """
    columns = [*table.columns, cfg.ip_column, "efit_times", "efit_psirz", "efit_grid_R", "efit_grid_Z",
               "efit_r_axis", "efit_z_axis", "magnetics_time", "magnetics_plasma_current_times"]
    shot = io.load_shot(path, columns)
    grid_r, grid_z = np.asarray(shot["efit_grid_R"], float), np.asarray(shot["efit_grid_Z"], float)
    currents = currents_at_frames(shot, table.columns, args.window_ms)
    ip = currents_at_frames(shot, [cfg.ip_column], args.window_ms)[:, 0]
    peak = float(np.nanpercentile(np.abs(np.asarray(shot[cfg.ip_column], float)), 99.5))
    psi = np.asarray(shot["efit_psirz"], float)
    r_axis, z_axis = np.asarray(shot["efit_r_axis"], float), np.asarray(shot["efit_z_axis"], float)
    rows, normals, records = [], [], []
    sol = table.column_index([cfg.solenoid_column])[0]
    for k in frame_sample(psi.shape[0], args.frames_per_shot):
        orientation = None if orientation_c is None else cs.orientation_from_rule(orientation_c, ip[k])
        a = cs.analyze_frame(psi[k], grid_r, grid_z, cfg, env, table, currents[k], ip[k], near, scorer, orientation)
        sol_ka = float(currents[k, sol])
        rows.append({"split": split, "shot": shot.shot_id, "frame": int(k), "time_ms": float(shot["efit_times"][k]),
                     "ip_ka": float(ip[k]), "sol_ka": sol_ka,
                     "ip_over_peak": abs(ip[k]) / peak if peak > 0 else np.nan,
                     "phase": phase_of(abs(ip[k]) / peak if peak > 0 else 0.0, k, psi.shape[0]),
                     "boundary_found": a.boundary_found, "orientation": a.orientation,
                     "current_ratio": a.ratio, "flux_range": a.flux_range,
                     "n_fit_nodes": int(a.fit_mask.sum()), "n_near_coil_nodes": int(a.near_coil_mask.sum()),
                     "rms_unscaled_fit": cs.rms_fraction(a.target, a.coil_total, a.fit_mask, a.flux_range),
                     "rms_unscaled_near": cs.rms_fraction(a.target, a.coil_total, a.near_coil_mask, a.flux_range),
                     "xpoint": cs.xpoint_class(a.contour, a.oriented_psi, grid_r, grid_z),
                     "r_axis": float(r_axis[k]), "z_axis": float(z_axis[k])})
        normals.append((cs.FrameNormal.from_frame(a, sol_ka), cs.FrameNormal.from_frame(a, sol_ka, a.near_coil_mask)))
        records.append({"oriented_psi": a.oriented_psi, "currents": currents[k].copy()})
    return shot, rows, normals, records, grid_r, grid_z


def evaluate(rows: list[dict], normals: list[tuple], solutions: dict[str, cs.FitSolution]) -> None:
    """Adds the per-variant rms (away from and near the coils) and mean residual to every row."""
    for row, (fit_normal, near_normal) in zip(rows, normals):
        for name, solution in solutions.items():
            row[f"rms_{name}"] = solution.frame_rms(fit_normal)
            row[f"rms_near_{name}"] = solution.frame_rms(near_normal)
            row[f"mean_{name}"] = solution.frame_mean_residual(fit_normal)


def k5_table(frames: pd.DataFrame, variants: list[str]) -> pd.DataFrame:
    """K5 quantiles per variant, split, node set and phase."""
    rows = []
    for name in ["unscaled", *variants]:
        for nodes, column in (("fit", f"rms_{name}" if name != "unscaled" else "rms_unscaled_fit"),
                              ("near", f"rms_near_{name}" if name != "unscaled" else "rms_unscaled_near")):
            for (split, phase), g in frames.groupby(["split", "phase"], sort=False):
                v = g[column].dropna()
                rows.append({"variant": name, "split": split, "nodes": nodes, "phase": phase, "n": int(v.size),
                             "median": v.median(), "p25": v.quantile(0.25), "p75": v.quantile(0.75)})
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data-root", type=Path, default=default_data_root())
    parser.add_argument("--green-dir", type=Path, default=DEFAULT_GREEN_DIR)
    parser.add_argument("--shot-index", type=Path, default=DEFAULT_SHOT_INDEX,
                        help="shot_index.parquet of the frame-table build the reference shots are chosen from")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n-shots", type=int, default=50, help="DIII-D training shots for the fit")
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT,
                        help="split file whose validation list the check of the adopted model draws from")
    parser.add_argument("--n-validation-shots", type=int, default=50,
                        help="validation shots for the check of the adopted model; 0 skips it")
    parser.add_argument("--n-mast-shots", type=int, default=20)
    parser.add_argument("--frames-per-shot", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--window-ms", type=float, default=2.0)
    args = parser.parse_args(argv)

    starter_kit.require()
    scorer = starter_kit.scoring()
    started = time.time()
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "scales").mkdir(exist_ok=True)

    cfg = machine_config("DIII-D")
    table = GreenTable.load(args.green_dir / "green_d3d.npz")
    env = starter_kit.envelope_mask(cfg)
    first = io.load_shot(list_shots(args.data_root, "diii_d_train")[0], GEOMETRY_COLUMNS, apply=False)
    geometry = CoilGeometry.from_shot(first)
    if geometry.hash() != table.geometry_hash:
        raise SystemExit("geometry hash of the data differs from the Green's table; rebuild the tables")
    near = near_coil_mask(geometry, table.grid_r, table.grid_z)
    variants = cs.standard_variants(table.columns, cfg.solenoid_column)

    files = list_shots(args.data_root, "diii_d_train")
    refs = cs.reference_shots(pd.read_parquet(args.shot_index))
    ref_ids = [s for ids in refs.values() for s in ids]
    ref_files = filter_shots(files, ref_ids)
    fit_files = select_shots([f for f in files if f not in set(ref_files)], args.n_shots, args.seed)

    # Pass 1: reference shots, every frame, orientation from the current density (K3, K4).
    ref_rows, ref_normals = [], []
    ref_args = argparse.Namespace(**{**vars(args), "frames_per_shot": 10_000})
    for path in ref_files:
        _, rows, normals, _, _, _ = analyse_shot(path, "reference", table, cfg, env, scorer, near, ref_args)
        ref_rows += rows
        ref_normals += normals
        print(f"reference {path.stem}: {len(rows)} frames", flush=True)
    ref_df = pd.DataFrame(ref_rows)
    per_shot = ref_df.groupby("shot").agg(ratio=("current_ratio", "median"), ip=("ip_ka", "median"))
    rule = cs.sign_rule(per_shot["ratio"], per_shot["ip"])
    print(f"sign rule: {rule}", flush=True)
    c = rule["c"] if rule["consistent"] else None

    # Pass 2: fit shots under the rule; every variant accumulates, plus a per-shot per-coil fit (K6).
    fits = {name: cs.ScaleFit(table.columns, v) for name, v in variants.items()}
    per_shot_scales, fit_rows, fit_normals, kept = [], [], [], []
    for i, path in enumerate(fit_files, start=1):
        shot, rows, normals, records, grid_r, grid_z = analyse_shot(path, "fit", table, cfg, env, scorer, near, args, c)
        shot_fit = cs.ScaleFit(table.columns, variants[SPREAD_VARIANT])
        for fit_normal, _ in normals:
            for fit in fits.values():
                fit.add(fit_normal)
            shot_fit.add(fit_normal)
        per_shot_scales.append({"shot": shot.shot_id, **dict(zip(table.columns, shot_fit.solve().scales))})
        fit_rows += rows
        fit_normals += normals
        kept.append((rows, records, grid_r, grid_z))
        if i % 5 == 0 or i == len(fit_files):
            print(f"  fit shots {i}/{len(fit_files)}, {time.time() - started:.0f} s", flush=True)
    solutions = {name: fit.solve() for name, fit in fits.items()}
    for name, solution in solutions.items():
        solution.to_coil_scales(cfg.name, source=f"{name}, fit of {timestamp}").save(args.out / "scales" / f"d3d_{name}.json")

    adopted = solutions[HEADLINE_VARIANT].to_coil_scales(cfg.name, source=f"{HEADLINE_VARIANT}, fit of {timestamp}")
    adopted_file = ADOPTED_FILES.get(cfg.name)
    adopted_matches = None
    if adopted_file is not None and adopted_file.is_file():
        on_file = CoilScales.load(adopted_file)
        adopted_matches = bool(np.allclose(on_file.vector(table.columns), adopted.vector(table.columns), atol=1e-9)
                               and abs(on_file.flat_wb_per_ka - adopted.flat_wb_per_ka) < 1e-9)
        if not adopted_matches:
            print(f"note: {adopted_file} differs from this fit's {HEADLINE_VARIANT} record; copy "
                  f"{args.out / 'scales' / f'd3d_{HEADLINE_VARIANT}.json'} there to adopt the new fit", flush=True)

    # Pass 3: validation shots of the split that took no part in the fit, for the adopted model.
    val_rows, val_normals, val_ids = [], [], []
    if args.n_validation_shots > 0 and args.split.is_file():
        used = set(ref_ids) | {io.parse_shot_filename(p)[1] for p in fit_files}
        candidates = [s for s in sp.read_split(args.split)["validation"] if s not in used]
        val_files = select_shots(filter_shots(files, candidates), args.n_validation_shots, args.seed + 2)
        val_ids = [io.parse_shot_filename(p)[1] for p in val_files]
        for i, path in enumerate(val_files, start=1):
            _, rows, normals, _, _, _ = analyse_shot(path, "validation", table, cfg, env, scorer, near, args, c)
            val_rows += rows
            val_normals += normals
            if i % 10 == 0 or i == len(val_files):
                print(f"  validation shots {i}/{len(val_files)}, {time.time() - started:.0f} s", flush=True)
    elif args.n_validation_shots > 0:
        print(f"no split file at {args.split}; the validation check is skipped", flush=True)

    # K5 for every variant on every frame; K7 on the fit frames with the headline variant.
    evaluate(ref_rows, ref_normals, solutions)
    evaluate(fit_rows, fit_normals, solutions)
    evaluate(val_rows, val_normals, solutions)
    headline = adopted
    for rows, records, grid_r, grid_z in kept:
        for row, record in zip(rows, records):
            model = table.flux(record["currents"], headline)
            row.update(cs.remainder_metrics(record["oriented_psi"], model, grid_r, grid_z, cfg, env, scorer,
                                            row["r_axis"], row["z_axis"], row["ip_ka"]))
    frames_df = pd.concat([pd.DataFrame(ref_rows), pd.DataFrame(fit_rows), pd.DataFrame(val_rows)], ignore_index=True)

    # K8 on one flat-top frame of the first fit shot.
    k8 = np.nan
    if kept:
        first_rows = kept[0][0]
        flat = [r for r in first_rows if r["phase"] == "flat_top"] or first_rows
        shot_path = filter_shots(files, [first_rows[0]["shot"]])[0]
        shot = io.load_shot(shot_path, [*table.columns, "efit_times", "magnetics_time"])
        currents = currents_at_frames(shot, table.columns, args.window_ms)
        k8 = cs.mirror_difference(table, currents[flat[0]["frame"]], cfg)

    # K9: MAST ratios from inputs only, against the same ratios on the DIII-D fit shots, each
    # machine under its adopted coil model (DIII-D: this fit's headline record; MAST: the default).
    ratios = []
    mast_table = None
    for machine, split, n in (("DIII-D", "diii_d_train", min(len(fit_files), 20)), ("MAST", "mast_public_test", args.n_mast_shots)):
        mcfg = machine_config(machine)
        mtable = GreenTable.load(args.green_dir / f"green_{mcfg.file_prefix.split('_')[0]}.npz")
        if machine == "MAST":
            mast_table = mtable
        model = adopted if machine == "DIII-D" else adopted_scales(machine, mtable.columns)
        for path in select_shots(list_shots(args.data_root, split), n, args.seed + 1):
            shot = io.load_shot(path, [*mtable.columns, mcfg.ip_column, "efit_times", "magnetics_time",
                                       "magnetics_plasma_current_times"])
            currents = currents_at_frames(shot, mtable.columns, args.window_ms)
            ip = currents_at_frames(shot, [mcfg.ip_column], args.window_ms)[:, 0]
            if not np.isfinite(ip).any():
                continue
            flat = int(np.nanargmax(np.abs(ip)))
            ratios.append({"machine": machine, "shot": shot.shot_id, "ip_ka": float(ip[flat]),
                           **cs.reference_ratios(mtable, currents, ip, mcfg, mtable.grid_r, mtable.grid_z, 0, flat, model)})
    ratios_df = pd.DataFrame(ratios)
    if mast_table is not None:
        CoilScales.default("MAST", mast_table.columns).save(args.out / "scales" / "mast_default.json")

    spread = pd.DataFrame(per_shot_scales).set_index("shot") if per_shot_scales else pd.DataFrame()
    scale_rows = []
    for name, solution in solutions.items():
        for column, value in zip(table.columns, solution.scales):
            kind = ("free" if column in solution.variant.free else "shared" if column in solution.variant.shared
                    else "fixed" if column in solution.variant.fixed else "left_out")
            scale_rows.append({"variant": name, "parameter": column, "kind": kind, "value": value,
                               "per_shot_median": spread[column].median() if name == SPREAD_VARIANT and len(spread) else np.nan,
                               "per_shot_iqr": (spread[column].quantile(0.75) - spread[column].quantile(0.25))
                               if name == SPREAD_VARIANT and len(spread) else np.nan})
        if solution.variant.flat:
            scale_rows.append({"variant": name, "parameter": cs.FLAT_PARAMETER, "kind": "flat_wb_per_ka",
                               "value": solution.flat_wb_per_ka, "per_shot_median": np.nan, "per_shot_iqr": np.nan})
    scales_df = pd.DataFrame(scale_rows)
    k5_df = k5_table(frames_df, list(variants))

    frames_df.to_parquet(args.out / "coil_frames.parquet", index=False)
    scales_df.to_parquet(args.out / "coil_scales.parquet", index=False)
    k5_df.to_parquet(args.out / "k5_by_phase.parquet", index=False)
    ratios_df.to_parquet(args.out / "reference_ratios.parquet", index=False)
    report = render_report(frames_df, scales_df, k5_df, ratios_df, rule, k8, refs, solutions)
    (args.out / "coil_flux_report.md").write_text(report)
    info = run_info(argv, vars(args), timestamp, time.time() - started, sign_rule=rule, k8_mirror=k8,
                    reference_shots=refs, n_fit_shots=len(fit_files), geometry_hash=table.geometry_hash,
                    variants=list(variants), headline_variant=HEADLINE_VARIANT, spread_variant=SPREAD_VARIANT,
                    validation_shots=val_ids, split_file=str(args.split), adopted_file=str(adopted_file),
                    adopted_file_matches=adopted_matches,
                    coefficients={name: s.coefficients() for name, s in solutions.items()})
    (args.out / "fit.json").write_text(json.dumps(info, indent=2, default=float) + "\n")
    print(report.split("\n## ")[0])
    return 0


def render_report(frames: pd.DataFrame, scales: pd.DataFrame, k5: pd.DataFrame, ratios: pd.DataFrame, rule: dict,
                  k8: float, refs: dict, solutions: dict[str, cs.FitSolution]) -> str:
    """Markdown report with K3 to K9."""
    ref = frames[frames["split"] == "reference"]
    fit = frames[frames["split"] == "fit"]
    lines = ["# Coil flux report", "",
             "| Id | Check | Status | Headline |", "| --- | --- | --- | --- |"]
    q = ref["current_ratio"].quantile([0.05, 0.5, 0.95]).round(3).tolist()
    lines.append(f"| K3 | Total current | review | ratio p05/p50/p95 on reference frames: {q} |")
    lines.append(f"| K4 | Sign rule | {'pass' if rule['consistent'] else 'fail'} | "
                 f"psi_physical = {rule['c']} x sign(Ip) x psi_stored over {rule['n_shots']} shots |")

    def by_phase(variant: str, nodes: str, split: str = "fit") -> dict:
        sub = k5[(k5["variant"] == variant) & (k5["nodes"] == nodes) & (k5["split"] == split)]
        return {r.phase: round(r.median, 4) for r in sub.itertuples(index=False)}

    for nodes, label in (("fit", "away from coils"), ("near", "near coils")):
        lines.append(f"| K5 | Agreement outside the plasma ({label}), {HEADLINE_VARIANT} | review | "
                     f"median rms / flux range by phase on fit frames: {by_phase(HEADLINE_VARIANT, nodes)} |")
    flat = fit[fit["phase"] == "flat_top"]
    headline_col = f"rms_{HEADLINE_VARIANT}"
    status = "review"
    if headline_col in flat and len(flat):
        status = "pass" if flat[headline_col].median() < K5_REVIEW_FRACTION else "review"
    lines.append(f"| K5 | Flat-top agreement away from coils ({HEADLINE_VARIANT}, fit frames) | {status} | "
                 f"median {flat[headline_col].median() if len(flat) else float('nan'):.4f} of flux range |")
    val = frames[frames["split"] == "validation"]
    if len(val):
        lines.append(f"| K5 | Validation shots not used in the fit ({HEADLINE_VARIANT}, {val['shot'].nunique()} shots) | review | "
                     f"median rms / flux range by phase: {by_phase(HEADLINE_VARIANT, 'fit', 'validation')}; "
                     f"near coils {by_phase(HEADLINE_VARIANT, 'near', 'validation')} |")
    pc = scales[(scales["variant"] == SPREAD_VARIANT) & (scales["kind"] == "free")]
    pcc = scales[(scales["variant"] == "per_coil_constant") & (scales["kind"] == "free") & (scales["parameter"] != "magnetics_ECOILA")]
    lines.append(f"| K6 | Coil scales | review | adopted {HEADLINE_VARIANT}: shaping coils at 1, flat solenoid term "
                 f"{solutions['solenoid_flat'].flat_wb_per_ka:.5f} Wb/rad per kA ("
                 f"{solutions['solenoid_flat_plus_pattern'].flat_wb_per_ka:.5f} with the pattern at "
                 f"{solutions['solenoid_flat_plus_pattern'].coefficients().get(solutions['solenoid_flat_plus_pattern'].variant.free[0], float('nan')):.3f}); "
                 f"per-coil scales with a per-frame constant {pcc['value'].min():.3f} to {pcc['value'].max():.3f}; "
                 f"{SPREAD_VARIANT} without constant {pc['value'].min():.3f} to {pc['value'].max():.3f}, per-shot IQR up to "
                 f"{pc.loc[pc['parameter'] != 'magnetics_ECOILA', 'per_shot_iqr'].max():.2f} |")
    if "rem_axis_dist_m" in fit:
        lines.append(f"| K7 | Remainder | review | axis distance median {fit['rem_axis_dist_m'].median():.4f} m; "
                     f"amplitude/Ip spread (IQR/median) on flat-top "
                     f"{(flat['rem_amplitude_over_ip'].quantile(0.75) - flat['rem_amplitude_over_ip'].quantile(0.25)) / abs(flat['rem_amplitude_over_ip'].median()) if len(flat) else float('nan'):.3f} |")
    lines.append(f"| K8 | Mirror symmetry | {'pass' if np.isfinite(k8) and k8 < 1e-3 else 'fail'} | "
                 f"max difference {k8:.2e} of flux range |")
    if len(ratios):
        by = ratios.groupby("machine")[["bz_ratio", "sol_flux_ratio"]].median().round(3).to_dict()
        lines.append(f"| K9 | MAST checks | review | medians by machine {by} (each machine under its adopted coil model, signed Ip) |")
    lines += ["", "## Reference shots", "", f"`{json.dumps(refs)}`", "",
              "## K5 by variant, median rms / flux range", "",
              "| Variant | Fit flat-top, away | Fit flat-top, near | Fit rising, away | Fit late, away | "
              "Validation flat-top, away | Validation rising, away | Validation late, away |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name in ["unscaled", *solutions]:
        away, near_, v = by_phase(name, "fit"), by_phase(name, "near"), by_phase(name, "fit", "validation")
        lines.append(f"| {name} | {away.get('flat_top', float('nan')):.4f} | {near_.get('flat_top', float('nan')):.4f} | "
                     f"{away.get('rising', float('nan')):.4f} | {away.get('late', float('nan')):.4f} | "
                     f"{v.get('flat_top', float('nan')):.4f} | {v.get('rising', float('nan')):.4f} | {v.get('late', float('nan')):.4f} |")
    lines += ["", "## Coefficients", "", "| Variant | Parameter | Kind | Value | Per-shot median | Per-shot IQR |",
              "| --- | --- | --- | --- | --- | --- |"]
    for r in scales.itertuples(index=False):
        if r.kind in ("fixed", "left_out"):
            continue
        lines.append(f"| {r.variant} | {r.parameter} | {r.kind} | {r.value:.5f} | {r.per_shot_median:.4f} | {r.per_shot_iqr:.4f} |")
    lines += ["", "## Frames", "", f"{len(frames)} frames analysed; boundary found on "
              f"{frames['boundary_found'].mean():.3%}.", ""]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
