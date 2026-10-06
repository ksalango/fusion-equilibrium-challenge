"""Tests the coil flux against true DIII-D maps and fits one scale per current column.

Analyses D3b, D4, D9 and D5 of ``docs/COIL_FLUX.md``. Reads the Green's tables from
``--green-dir``, a sample of DIII-D training shots (and the eight reference shots of
D3 chosen from the QA table) and 20 MAST test shots, and writes to ``--out``:

- ``coil_frames.parquet``: one row per analysed frame with the current ratio (K3), the
  agreement outside the plasma away from and near the coils (K5), the remainder
  metrics (K7) and the orientation.
- ``coil_scales.parquet``: one row per current column and fit variant with the scale,
  the per-shot spread (K6) and the normal-equation diagnostics.
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

from fusion_eq import coil_scales as cs, io, starter_kit
from fusion_eq.coil_flux import GEOMETRY_COLUMNS, CoilGeometry, GreenTable, currents_at_frames, near_coil_mask
from fusion_eq.dataset import REPO_ROOT, default_data_root, filter_shots, list_shots, run_info, select_shots
from fusion_eq.machines import machine_config
from fusion_eq.profile import QA_RATE_SPLIT_MS

DEFAULT_GREEN_DIR = REPO_ROOT.parent / "data" / "processed" / "green"
DEFAULT_OUT = REPO_ROOT / "reports" / "coil_flux"
DEFAULT_QA_TABLE = REPO_ROOT / "reports" / "qa" / "qa_per_shot.parquet"
FLAT_TOP = 0.9
K5_REVIEW_FRACTION = 0.05


def reference_shots(qa_path: Path) -> dict[str, list[str]]:
    """The eight D3 reference shots: 3 positive-Ip 20 kHz, 3 negative-Ip, 2 at 2 kHz."""
    qa = pd.read_parquet(qa_path)
    train = qa[(qa["split"] == "diii_d_train") & qa["error"].isna() & (qa["time_n_frames"] >= 30)]
    fast = train["time_magnetics_dt_median_ms"] <= QA_RATE_SPLIT_MS
    positive = train["cur_plasma_current_min"] >= 0
    negative = train["cur_plasma_current_max"] <= 0
    pick = lambda sub, n: sorted(sub["shot"].astype(str))[:n]  # noqa: E731
    return {"positive_20khz": pick(train[fast & positive], 3), "negative_ip": pick(train[negative], 3),
            "rate_2khz": pick(train[~fast & positive], 2)}


def frame_sample(n_frames: int, n: int) -> np.ndarray:
    return np.unique(np.round(np.linspace(0, n_frames - 1, min(n, n_frames))).astype(int))


def phase_of(ip_over_peak: float, frame: int, n_frames: int) -> str:
    if ip_over_peak < FLAT_TOP:
        return "rising" if frame < n_frames / 2 else "late"
    return "flat_top"


def analyse_shot(path: Path, split: str, table: GreenTable, cfg, env, scorer, near, args, orientation_c=None):
    """Per-frame rows and analyses of one labelled shot."""
    columns = [*table.columns, cfg.ip_column, "efit_times", "efit_psirz", "efit_grid_R", "efit_grid_Z",
               "efit_r_axis", "efit_z_axis", "magnetics_time", "magnetics_plasma_current_times"]
    shot = io.load_shot(path, columns)
    grid_r, grid_z = np.asarray(shot["efit_grid_R"], float), np.asarray(shot["efit_grid_Z"], float)
    currents = currents_at_frames(shot, table.columns, args.window_ms)
    ip = currents_at_frames(shot, [cfg.ip_column], args.window_ms)[:, 0]
    peak = float(np.nanpercentile(np.abs(np.asarray(shot[cfg.ip_column], float)), 99.5))
    psi = np.asarray(shot["efit_psirz"], float)
    r_axis, z_axis = np.asarray(shot["efit_r_axis"], float), np.asarray(shot["efit_z_axis"], float)
    rows, analyses = [], []
    sol = table.column_index([cfg.solenoid_column])[0]
    for k in frame_sample(psi.shape[0], args.frames_per_shot):
        orientation = None if orientation_c is None else cs.orientation_from_rule(orientation_c, ip[k])
        a = cs.analyze_frame(psi[k], grid_r, grid_z, cfg, env, table, currents[k], ip[k], near, scorer, orientation)
        rows.append({"split": split, "shot": shot.shot_id, "frame": int(k), "time_ms": float(shot["efit_times"][k]),
                     "ip_ka": float(ip[k]), "sol_ka": float(currents[k, sol]),
                     "ip_over_peak": abs(ip[k]) / peak if peak > 0 else np.nan,
                     "phase": phase_of(abs(ip[k]) / peak if peak > 0 else 0.0, k, psi.shape[0]),
                     "boundary_found": a.boundary_found, "orientation": a.orientation,
                     "current_ratio": a.ratio, "flux_range": a.flux_range,
                     "n_fit_nodes": int(a.fit_mask.sum()), "n_near_coil_nodes": int(a.near_coil_mask.sum()),
                     "rms_unscaled_fit": cs.rms_fraction(a.target, a.coil_total, a.fit_mask, a.flux_range),
                     "rms_unscaled_near": cs.rms_fraction(a.target, a.coil_total, a.near_coil_mask, a.flux_range),
                     "r_axis": float(r_axis[k]), "z_axis": float(z_axis[k])})
        analyses.append(a)
    return shot, rows, analyses, grid_r, grid_z


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data-root", type=Path, default=default_data_root())
    parser.add_argument("--green-dir", type=Path, default=DEFAULT_GREEN_DIR)
    parser.add_argument("--qa-table", type=Path, default=DEFAULT_QA_TABLE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n-shots", type=int, default=50, help="DIII-D training shots for the fit")
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

    cfg = machine_config("DIII-D")
    table = GreenTable.load(args.green_dir / "green_d3d.npz")
    env = starter_kit.envelope_mask(cfg)
    first = io.load_shot(list_shots(args.data_root, "diii_d_train")[0], GEOMETRY_COLUMNS, apply=False)
    geometry = CoilGeometry.from_shot(first)
    if geometry.hash() != table.geometry_hash:
        raise SystemExit("geometry hash of the data differs from the Green's table; rebuild the tables")
    near = near_coil_mask(geometry, table.grid_r, table.grid_z)

    files = list_shots(args.data_root, "diii_d_train")
    refs = reference_shots(args.qa_table)
    ref_ids = [s for ids in refs.values() for s in ids]
    ref_files = filter_shots(files, ref_ids)
    fit_files = select_shots([f for f in files if f not in set(ref_files)], args.n_shots, args.seed)

    # Pass 1: reference shots, every frame, orientation from the current density (K3, K4).
    ref_rows = []
    ref_args = argparse.Namespace(**{**vars(args), "frames_per_shot": 10_000})
    for path in ref_files:
        _, rows, _, _, _ = analyse_shot(path, "reference", table, cfg, env, scorer, near, ref_args)
        ref_rows += rows
        print(f"reference {path.stem}: {len(rows)} frames", flush=True)
    ref_df = pd.DataFrame(ref_rows)
    per_shot = ref_df.groupby("shot").agg(ratio=("current_ratio", "median"), ip=("ip_ka", "median"))
    rule = cs.sign_rule(per_shot["ratio"], per_shot["ip"])
    print(f"sign rule: {rule}", flush=True)
    c = rule["c"] if rule["consistent"] else None

    # Pass 2: fit shots under the rule, accumulating both fit variants and per-shot fits.
    fit_plain = cs.ScaleFit(table.columns, with_constant=False)
    fit_const = cs.ScaleFit(table.columns, with_constant=True)
    per_shot_scales, frame_rows, kept = [], [], []
    for i, path in enumerate(fit_files, start=1):
        shot, rows, analyses, grid_r, grid_z = analyse_shot(path, "fit", table, cfg, env, scorer, near, args, c)
        shot_fit = cs.ScaleFit(table.columns, with_constant=False)
        for a in analyses:
            fit_plain.add(a.target, a.coil_maps, a.fit_mask)
            fit_const.add(a.target, a.coil_maps, a.fit_mask)
            shot_fit.add(a.target, a.coil_maps, a.fit_mask)
        per_shot_scales.append({"shot": shot.shot_id, **dict(zip(table.columns, shot_fit.solve()))})
        frame_rows += rows
        kept.append((shot.shot_id, analyses, grid_r, grid_z, rows))
        if i % 5 == 0 or i == len(fit_files):
            print(f"  fit shots {i}/{len(fit_files)}, {time.time() - started:.0f} s", flush=True)
    scales_plain, scales_const = fit_plain.solve(), fit_const.solve()

    # Scaled agreement (K5) and remainder (K7) on the kept analyses.
    for shot_id, analyses, grid_r, grid_z, rows in kept:
        for a, row in zip(analyses, rows):
            model = a.scaled_coil_flux(scales_plain)
            row["rms_scaled_fit"] = cs.rms_fraction(a.target, model, a.fit_mask, a.flux_range)
            row["rms_scaled_near"] = cs.rms_fraction(a.target, model, a.near_coil_mask, a.flux_range)
            model_c = a.scaled_coil_flux(scales_const)
            b = fit_const.constant_for(a.target, a.coil_maps, a.fit_mask, scales_const)
            row["rms_scaled_const_fit"] = cs.rms_fraction(a.target, model_c, a.fit_mask, a.flux_range, b)
            row["frame_constant"] = b
            row.update(cs.remainder_metrics(a.oriented_psi, model, grid_r, grid_z, cfg, env, scorer,
                                            row["r_axis"], row["z_axis"], row["ip_ka"]))
    frames_df = pd.concat([ref_df, pd.DataFrame(frame_rows)], ignore_index=True)

    # K8 on one flat-top frame of the first fit shot.
    k8 = np.nan
    if kept:
        first_rows = kept[0][4]
        flat = [r for r in first_rows if r["phase"] == "flat_top"] or first_rows
        shot_path = filter_shots(files, [kept[0][0]])[0]
        shot = io.load_shot(shot_path, [*table.columns, "efit_times", "magnetics_time"])
        currents = currents_at_frames(shot, table.columns, args.window_ms)
        k8 = cs.mirror_difference(table, currents[flat[0]["frame"]], cfg)

    # K9: MAST ratios from inputs only, against the same ratios on the DIII-D fit shots.
    ratios = []
    for machine, split, n in (("DIII-D", "diii_d_train", min(len(fit_files), 20)), ("MAST", "mast_public_test", args.n_mast_shots)):
        mcfg = machine_config(machine)
        mtable = GreenTable.load(args.green_dir / f"green_{mcfg.file_prefix.split('_')[0]}.npz")
        for path in select_shots(list_shots(args.data_root, split), n, args.seed + 1):
            shot = io.load_shot(path, [*mtable.columns, mcfg.ip_column, "efit_times", "magnetics_time",
                                       "magnetics_plasma_current_times"])
            currents = currents_at_frames(shot, mtable.columns, args.window_ms)
            ip = currents_at_frames(shot, [mcfg.ip_column], args.window_ms)[:, 0]
            if not np.isfinite(ip).any():
                continue
            flat = int(np.nanargmax(np.abs(ip)))
            ratios.append({"machine": machine, "shot": shot.shot_id,
                           **cs.reference_ratios(mtable, currents, np.abs(ip), mcfg, mtable.grid_r, mtable.grid_z, 0, flat)})
    ratios_df = pd.DataFrame(ratios)

    scales_df = pd.DataFrame({
        "column": list(table.columns) * 2, "variant": ["plain"] * len(table.columns) + ["with_constant"] * len(table.columns),
        "scale": np.concatenate([scales_plain, scales_const]),
    })
    spread = pd.DataFrame(per_shot_scales).set_index("shot")
    scales_df["per_shot_median"] = [spread[c].median() for c in table.columns] * 2
    scales_df["per_shot_iqr"] = [spread[c].quantile(0.75) - spread[c].quantile(0.25) for c in table.columns] * 2

    frames_df.to_parquet(args.out / "coil_frames.parquet", index=False)
    scales_df.to_parquet(args.out / "coil_scales.parquet", index=False)
    ratios_df.to_parquet(args.out / "reference_ratios.parquet", index=False)
    report = render_report(frames_df, scales_df, ratios_df, rule, k8, refs)
    (args.out / "coil_flux_report.md").write_text(report)
    info = run_info(argv, vars(args), timestamp, time.time() - started, sign_rule=rule, k8_mirror=k8,
                    reference_shots=refs, n_fit_shots=len(fit_files), geometry_hash=table.geometry_hash)
    (args.out / "fit.json").write_text(json.dumps(info, indent=2, default=float) + "\n")
    print(report.split("\n## ")[0])
    return 0


def render_report(frames: pd.DataFrame, scales: pd.DataFrame, ratios: pd.DataFrame, rule: dict, k8: float,
                  refs: dict) -> str:
    """Markdown report with K3 to K9."""
    ref = frames[frames["split"] == "reference"]
    fit = frames[frames["split"] == "fit"]
    lines = ["# Coil flux report", "",
             "| Id | Check | Status | Headline |", "| --- | --- | --- | --- |"]
    q = ref["current_ratio"].quantile([0.05, 0.5, 0.95]).round(3).tolist()
    lines.append(f"| K3 | Total current | review | ratio p05/p50/p95 on reference frames: {q} |")
    lines.append(f"| K4 | Sign rule | {'pass' if rule['consistent'] else 'fail'} | "
                 f"psi_physical = {rule['c']} x sign(Ip) x psi_stored over {rule['n_shots']} shots |")
    for col, label in (("rms_scaled_fit", "away from coils"), ("rms_scaled_near", "near coils")):
        if col in fit:
            by_phase = fit.groupby("phase")[col].median().round(4).to_dict()
            lines.append(f"| K5 | Agreement outside the plasma ({label}) | review | median rms / flux range by phase: {by_phase} |")
    flat = fit[fit["phase"] == "flat_top"]
    status = "review"
    if "rms_scaled_fit" in flat and len(flat):
        status = "pass" if flat["rms_scaled_fit"].median() < K5_REVIEW_FRACTION else "review"
    lines.append(f"| K5 | Flat-top agreement away from coils | {status} | "
                 f"median {flat['rms_scaled_fit'].median() if len(flat) else float('nan'):.4f} of flux range |")
    plain = scales[scales["variant"] == "plain"]
    lines.append(f"| K6 | Coil scales | review | plain fit: median over columns {plain['scale'].median():.3f}, "
                 f"range {plain['scale'].min():.3f} to {plain['scale'].max():.3f} |")
    if "rem_axis_dist_m" in fit:
        lines.append(f"| K7 | Remainder | review | axis distance median {fit['rem_axis_dist_m'].median():.4f} m; "
                     f"amplitude/Ip spread (IQR/median) on flat-top "
                     f"{(flat['rem_amplitude_over_ip'].quantile(0.75) - flat['rem_amplitude_over_ip'].quantile(0.25)) / abs(flat['rem_amplitude_over_ip'].median()) if len(flat) else float('nan'):.3f} |")
    lines.append(f"| K8 | Mirror symmetry | {'pass' if np.isfinite(k8) and k8 < 1e-3 else 'fail'} | "
                 f"max difference {k8:.2e} of flux range |")
    if len(ratios):
        by = ratios.groupby("machine")[["bz_ratio", "sol_flux_ratio"]].median().round(3).to_dict()
        lines.append(f"| K9 | MAST checks | review | medians by machine {by} |")
    lines += ["", "## Reference shots", "", f"`{json.dumps(refs)}`", "", "## Coil scales", "",
              "| Column | Variant | Scale | Per-shot median | Per-shot IQR |", "| --- | --- | --- | --- | --- |"]
    for r in scales.itertuples(index=False):
        lines.append(f"| {r.column} | {r.variant} | {r.scale:.4f} | {r.per_shot_median:.4f} | {r.per_shot_iqr:.4f} |")
    lines += ["", "## Frames", "", f"{len(frames)} frames analysed; boundary found on "
              f"{frames['boundary_found'].mean():.3%}.", ""]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
