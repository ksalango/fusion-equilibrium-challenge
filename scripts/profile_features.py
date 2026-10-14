"""Profiles the feature tables of a version: checks X1 to X15 of ``docs/FEATURES.md``.

Reads the unscaled feature tables and the build record of ``build_features.py``, the
scaler and the layout clearances under ``reports/features/v<version>/``, the shot index,
and from the frame table only the columns the checks need besides the features: Ip, the
coil currents (the A/B exchange of X5 and the raw-current correlations of X12), the 50 ms
Ip slope, the frame time and the Ip rise (X15). The frame-table rows must match the
feature rows key for key (X1).

Nothing is fitted on test inputs. The reference of every comparison is the DIII-D
training frames of the ``train`` list of ``--split``; ``validation`` is that file's
validation list. Comparisons by phase use the rule of ``coil_scales.phase_of``.

Writes to ``<report-root>/v<version>/``:

- ``feature_report.md``: the check table and one section per check.
- ``range_comparison.parquet``: X8 and X9, one row per feature, compared split and phase
  (``all``, ``rising``, ``flat_top``, ``late``).
- ``nn_distance.parquet``: X10, one row per sampled frame and feature set: the distance to
  the nearest sampled training frame (originals; and originals with their mirrors), that
  frame's keys, and the share of the squared distance carried by each feature group.
- ``correlations.parquet``: X12, the correlation matrices in long form.
- ``profile.json``: every number of the report, for notebook 06.

For a version after the first, X10 and X11 also show the numbers of the reference version
(``--reference-version``, 1) from its ``profile.json`` beside the version's own, when that
run used the same split file, samples and seed (the sampled frames are then the same).
X8 is not compared: the unscaled values of a feature do not depend on the version.

Usage::

    python scripts/profile_features.py
    python scripts/profile_features.py --config configs/features_v2.yaml
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

from fusion_eq import dataset, feature_checks as fc, features as fe, frame_table as ft, split as sp
from fusion_eq.coil_pattern import TERM_NAMES
from fusion_eq.dataset import default_data_root, run_info
from fusion_eq.feature_scaler import RobustScaler
from fusion_eq.machines import machine_config

DEFAULT_SPLIT = dataset.REPO_ROOT / "splits" / "split_v1.json"
DEFAULT_K9 = dataset.REPO_ROOT / "reports" / "coil_flux" / "reference_ratios.parquet"
# K9 medians of findings F53, used when the K9 table is not found.
K9_FALLBACK = {"DIII-D": -2.09, "MAST": -1.35}
GROUPS = ("train", "validation", "diii_d_public_test", "mast_public_test")
GROUP_MACHINE = {"train": "DIII-D", "validation": "DIII-D", "diii_d_public_test": "DIII-D", "mast_public_test": "MAST"}
COMPARED = fc.COMPARED_SPLITS
TERM_COLUMNS = [f"coil_{t}" for t in TERM_NAMES]
STRETCH_COLUMNS = ("coil_c20", "coil_c02", "shaping")
# Arguments that must agree for the reference version's X10 and X11 to be shown.
REFERENCE_KEYS = ("split", "nn_train_frames", "nn_eval_frames", "seed")


def log(message: str) -> None:
    print(message, flush=True)


# ------------------------------------------------------------------------------ loading
def load_split(split: str, config: fe.FeatureConfig, args) -> pd.DataFrame:
    """Feature rows of one data split joined with the frame-table columns of the checks.

    Raises:
        SystemExit: when the feature rows and the frame-table rows differ in keys.
    """
    features = fe.read_features(split, config.version, config.all_columns(), None, args.feature_root)
    columns = fc.profile_frame_columns(dataset.SPLIT_MACHINE[split])
    frames = ft.read_frame_table(split, ["shot", "frame", *columns], root=args.table_dir)
    try:
        return fc.join_frame_columns(features, frames)
    except ValueError as exc:
        raise SystemExit(f"{split}: {exc}") from exc


def display_path(path: Path | str) -> str:
    """A path relative to the repository when possible, so that committed reports carry no
    absolute home path."""
    return os.path.relpath(Path(path).resolve(), dataset.REPO_ROOT)


def stretch_columns(config: fe.FeatureConfig) -> list[str]:
    """The stretch terms of X9 the configuration builds."""
    columns = config.all_columns()
    return [c for c in STRETCH_COLUMNS if c in columns]


def load_reference(config: fe.FeatureConfig, args, n_train_rows: int) -> dict | None:
    """The reference version's profile, or None for the reference version itself.

    Returns:
        ``{"version", "path", "differences", "comparable", "x10_summary", "x11"}``;
        ``differences`` names the arguments (and the training row count) in which the runs
        differ, and the numbers are shown only when there is none.
    """
    if config.version == args.reference_version:
        return None
    path = Path(args.report_root) / f"v{args.reference_version}" / "profile.json"
    out = {"version": args.reference_version, "path": display_path(path)}
    if not path.is_file():
        return {**out, "differences": {"profile.json": ["missing", "present"]}, "comparable": False}
    record = json.loads(path.read_text())
    differences = fc.reference_differences(record.get("arguments", {}), vars(args), REFERENCE_KEYS, path_keys=("split",))
    ref_rows = record.get("n_rows", {}).get("train")
    if ref_rows != n_train_rows:
        differences["n_rows train"] = [ref_rows, n_train_rows]
    out.update(differences=differences, comparable=not differences)
    if out["comparable"]:
        out.update(x10_summary=record["x10"]["summary"], x11=record["x11"])
    return out


def k9_medians(path: Path) -> tuple[dict, str]:
    if path.is_file():
        table = pd.read_parquet(path)
        return {m: float(v) for m, v in table.groupby("machine")["bz_ratio"].median().items()}, display_path(path)
    return dict(K9_FALLBACK), "findings F53 (K9 table not found)"


# -------------------------------------------------------------------------------- checks
def check_x1(build: dict, config: fe.FeatureConfig, data: dict[str, pd.DataFrame]) -> tuple[fc.Check, dict]:
    counts = build.get("counts", {})
    same_config = build.get("config_hash") == config.hash()
    record_ok = bool(counts) and all(c.get("x1_rows_match") for c in counts.values())
    splits_built = sorted(counts) == sorted(dataset.DATA_SPLITS)
    status = "pass" if record_ok and same_config and splits_built else "fail"
    rows = {s: len(data[s]) for s in dataset.DATA_SPLITS}
    summary = (f"rows and keys equal the frame table on every split ({', '.join(f'{s} {n}' for s, n in rows.items())}); "
               f"build record {'matches' if record_ok else 'does not match'}; configuration "
               f"{'unchanged since the build' if same_config else 'changed since the build'}")
    return fc.Check("X1", "Rows and keys", status, summary), {"rows": rows, "build_counts": counts,
                                                             "config_hash_matches": same_config}


def check_x2(build: dict, config: fe.FeatureConfig) -> tuple[fc.Check, dict]:
    reads, problems = {}, []
    for machine in ("DIII-D", "MAST"):
        reads[machine] = fe.feature_reads(config, machine)
        for name, cols in reads[machine].items():
            try:
                fe.check_reads(cols, machine)
            except ValueError as exc:
                problems.append(f"{machine} {name}: {exc}")
    matches = build.get("x2_reads") == reads
    status = "pass" if not problems and matches else "fail"
    n = len({c for m in reads.values() for cols in m.values() for c in cols})
    summary = (f"{n} distinct columns read; every one is in input_columns(machine, include_flags=True) or is time_ms"
               if not problems else "; ".join(problems))
    if not matches:
        summary += "; the reads differ from the build record"
    return fc.Check("X2", "Inputs only", status, summary), {"reads": reads, "problems": problems, "matches_build": matches}


def check_x3(data: dict[str, pd.DataFrame], config: fe.FeatureConfig) -> tuple[fc.Check, dict, pd.DataFrame]:
    columns = config.all_columns()
    shares = pd.DataFrame({s: fc.finite_shares(data[s], columns) for s in dataset.DATA_SPLITS})
    low = {s: fc.low_ip_counts(data[s]["ip__mean"], config.min_abs_ip_ka) for s in dataset.DATA_SPLITS}
    worst = {s: (shares[s].idxmin(), float(shares[s].min())) for s in dataset.DATA_SPLITS}
    summary = ("lowest finite share: " + "; ".join(f"{s} {c} {v:.4f}" for s, (c, v) in worst.items())
               + f". Frames with Ip missing or below {config.min_abs_ip_ka:g} kA: "
               + ", ".join(f"{s} {d['n_affected']}" for s, d in low.items()))
    return fc.Check("X3", "Completeness", "review", summary), {"low_ip": low, "min_abs_ip_ka": config.min_abs_ip_ka}, shares


def check_x4(build: dict, config: fe.FeatureConfig, clearance_path: Path) -> tuple[fc.Check, dict]:
    layout = build.get("layout_check", {})
    table = pd.read_csv(clearance_path) if clearance_path.is_file() else None
    csv_ok = table is not None and bool(table["ok"].all()) and len(table) == 2 * config.layout.n_points
    symmetric = config.layout.is_symmetric()
    status = "pass" if layout.get("status") == "pass" and csv_ok and symmetric else "fail"
    mins = layout.get("min_clearance_m", {})
    summary = (f"{config.layout.n_points} points, symmetric {symmetric}; smallest clearance "
               + ", ".join(f"{m} {v:.3f} m" for m, v in mins.items())
               + f" (rule {config.layout.min_clearance_m:g} m); failing points: {len(layout.get('failing', []))}")
    return fc.Check("X4", "Layout", status, summary), {"layout_check": layout, "clearance_file_ok": csv_ok}


def check_x5(data: dict[str, pd.DataFrame], config: fe.FeatureConfig, train_ids: list[str], args) -> tuple[fc.Check, dict]:
    columns = config.all_columns()
    identity = {s: fc.frames_equal(fe.mirror_features(fe.mirror_features(data[s][["shot", "frame", *columns]], config), config),
                                   data[s], columns) for s in dataset.DATA_SPLITS}
    machine = "DIII-D"
    cfg = machine_config(machine)
    resources = fe.load_resources(machine, config, args.data_root, args.green_dir)
    shots = sorted(set(train_ids) & set(data["diii_d_train"]["shot"].unique()))
    rng = np.random.default_rng(args.seed)
    chosen = sorted(rng.choice(shots, size=min(args.mirror_shots, len(shots)), replace=False).tolist()) if shots else []
    reads = fe.feature_reads(config, machine)["coil_pt"]
    frames = ft.read_frame_table("diii_d_train", ["shot", "frame", *reads], shots=chosen, root=args.table_dir)
    points = config.columns_of("coil_pt")
    original = fe.build_features(frames, machine, config, resources, names=["coil_pt"])
    mirrored = fe.mirror_features(original, config)
    swapped = fe.build_features(fc.swap_mirror_currents(frames, cfg, resources.point_table.columns), machine, config,
                                resources, names=["coil_pt"])
    diff = fc.pattern_difference(mirrored[points].to_numpy(), swapped[points].to_numpy(), original[points].to_numpy())
    d = diff[np.isfinite(diff)]
    stats = {"n_shots": len(chosen), "shots": chosen, "n_frames": int(d.size),
             "median": float(np.median(d)) if d.size else np.nan, "p95": float(np.percentile(d, 95)) if d.size else np.nan,
             "max": float(d.max()) if d.size else np.nan}
    twice = all(identity.values())
    if not twice:
        status = "fail"
    elif d.size and stats["p95"] <= fc.SWAP_DIFFERENCE_PASS:
        status = "pass"
    else:
        status = "review"
    summary = (f"mirroring twice returns the input on {sum(identity.values())} of {len(identity)} splits; mirrored points "
               f"against points from A and B exchanged, {stats['n_frames']} frames of {len(chosen)} training shots: "
               f"median {stats['median']:.4f}, 95th percentile {stats['p95']:.4f}, max {stats['max']:.4f} of the pattern's "
               f"range (F51 expects about 0.02; pass at or below {fc.SWAP_DIFFERENCE_PASS:g} at the 95th percentile)")
    return fc.Check("X5", "Mirror", status, summary), {"identity": identity, "swap_difference": stats}


def check_x6(train: pd.DataFrame, index: pd.DataFrame) -> tuple[fc.Check, dict, pd.DataFrame]:
    direction = train["shot"].map(index.set_index("shot")["ip_direction"].astype("float64")).to_numpy()
    table = fc.direction_comparison(train, direction, TERM_COLUMNS)
    worst = table.loc[table["median_shift_iqr"].abs().idxmax()] if table["median_shift_iqr"].notna().any() else None
    summary = (f"{int((direction == 1).sum())} positive-Ip and {int((direction == -1).sum())} negative-Ip training frames; "
               f"share of negative-Ip values inside the positive band from {table['share_negative_in_positive_band'].min():.3f} "
               f"to {table['share_negative_in_positive_band'].max():.3f}")
    if worst is not None:
        summary += f"; largest median shift {worst['feature']} {worst['median_shift_iqr']:+.2f} interquartile ranges"
    return fc.Check("X6", "Current direction", "review", summary), {"table": table.to_dict(orient="records")}, table


def check_x7(data: dict[str, pd.DataFrame], args) -> tuple[fc.Check, dict]:
    ratios = {}
    for group, machine in (("train", "DIII-D"), ("mast_public_test", "MAST")):
        cfg = machine_config(machine)
        rows = data[group].loc[data[group]["phase"] == "flat_top"]
        ratios[machine] = fc.bz_ratio_from_c10(rows["coil_c10"], cfg.major_radius.value, cfg.minor_radius.value)
    k9, source = k9_medians(args.k9)
    result = fc.force_balance_check(ratios, k9)
    result["k9_source"] = source
    summary = ("median 4 pi (R0/a0) coil_c10 on flat-top frames: "
               + ", ".join(f"{m} {v:.3f} (K9 {result['k9_medians'][m]:.3f})" for m, v in result["medians"].items())
               + f"; signs {'match' if result['signs_match'] else 'differ'}, order {'matches' if result['order_matches'] else 'differs'}")
    return fc.Check("X7", "Force-balance sign", result["status"], summary), result


def scaled_matrices(data: dict[str, pd.DataFrame], scaler: RobustScaler) -> dict[str, np.ndarray]:
    return {g: scaler.transform(data[g], GROUP_MACHINE[g]) for g in GROUPS}


def check_x10(data: dict[str, pd.DataFrame], scaled: dict[str, np.ndarray], scaler: RobustScaler,
              config: fe.FeatureConfig, args) -> tuple[fc.Check, dict, pd.DataFrame]:
    columns = list(scaler.columns)
    specs = config.column_specs()
    train_rows, eval_rows = fc.x10_samples(len(data["train"]), {g: len(data[g]) for g in COMPARED},
                                           args.nn_train_frames, args.nn_eval_frames, args.seed)
    train_sample = data["train"].iloc[train_rows]
    mirrored = scaler.transform(fe.mirror_features(train_sample, config), "DIII-D")
    parts = []
    for set_name in config.sets:
        set_columns = config.set_columns(set_name)
        idx = [columns.index(c) for c in set_columns]
        groups = fc.column_groups(set_columns, {c: specs[c].group for c in set_columns})
        reference = scaled["train"][train_rows][:, idx]
        with_mirrors = np.vstack([reference, mirrored[:, idx]])
        queries = {g: scaled[g][eval_rows[g]][:, idx] for g in COMPARED}
        distances = fc.combined_distance(reference, queries, groups)
        for group in COMPARED:
            rows, d = eval_rows[group], distances.loc[distances["split"] == group]
            nearest = d["nearest"].to_numpy()
            distance_m, _ = fc.nearest_neighbours(with_mirrors, queries[group])
            part = pd.DataFrame({"set": set_name, "split": group, "shot": data[group]["shot"].to_numpy()[rows],
                                 "frame": data[group]["frame"].to_numpy()[rows], "phase": data[group]["phase"].to_numpy()[rows],
                                 "distance": d["distance"].to_numpy(), "distance_with_mirrors": distance_m,
                                 "nn_shot": train_sample["shot"].to_numpy()[nearest],
                                 "nn_frame": train_sample["frame"].to_numpy()[nearest]})
            for c in (c for c in d.columns if c.startswith("share_")):
                part[c] = d[c].to_numpy()
            parts.append(part)
        log(f"  X10 {set_name}: nearest neighbours of {len(COMPARED)} splits among {len(reference)} training frames")
    table = pd.concat(parts, ignore_index=True)
    summary_table = fc.distance_summary(table)
    mirrors = fc.distance_summary(table.assign(distance=table["distance_with_mirrors"]))
    groups_median = fc.group_share_medians(table)
    mast = summary_table.loc[summary_table["split"] == "mast_public_test"]
    summary = "; ".join(f"{r.set}: MAST median {r.p50:.2f} against validation "
                        f"{summary_table.loc[(summary_table['set'] == r.set) & (summary_table['split'] == 'validation'), 'p50'].iloc[0]:.2f}, "
                        f"{r.share_beyond_validation_p99:.3f} of MAST frames beyond the validation 99th percentile"
                        for r in mast.itertuples(index=False))
    info = {"n_train_sample": int(len(train_rows)), "n_eval_sample": args.nn_eval_frames, "seed": args.seed,
            "summary": summary_table.to_dict(orient="records"), "summary_with_mirrors": mirrors.to_dict(orient="records"),
            "group_share_medians": groups_median.to_dict(orient="records")}
    return fc.Check("X10", "Combined distance", "review", summary), info, table


def check_x11(scaled: dict[str, np.ndarray], scaler: RobustScaler) -> tuple[fc.Check, pd.DataFrame]:
    table = pd.DataFrame({g: fc.extreme_shares(scaled[g], scaler.columns) for g in GROUPS})
    worst = {g: (table[g].idxmax(), float(table[g].max())) for g in GROUPS}
    summary = (f"largest share of scaled values beyond {fc.EXTREME_LIMIT:g}: "
               + "; ".join(f"{g} {c} {v:.4f}" for g, (c, v) in worst.items()))
    return fc.Check("X11", "Extremes after scaling", "review", summary), table


def check_x12(train: pd.DataFrame, config: fe.FeatureConfig, args) -> tuple[fc.Check, dict, pd.DataFrame]:
    rows = train.iloc[fc.sample_rows(len(train), args.corr_frames, args.seed)].copy()
    rows["abs_ip"] = rows["ip__mean"].abs()
    cfg = machine_config("DIII-D")
    coil = [*config.columns_of("coil_pt"), *TERM_COLUMNS]
    raw = ["sol__mean", *[f"{c}__mean" for c in cfg.shaping_coil_columns]]
    ip = ["ip__mean", "abs_ip"]
    long = pd.concat([fc.correlation_long(rows, [*ip, *coil], "coil_features"),
                      fc.correlation_long(rows, [*ip, *raw], "raw_currents")], ignore_index=True)
    summaries = {kind: fc.correlation_summary(long.loc[long["kind"] == kind], ip) for kind in ("coil_features", "raw_currents")}
    c, r = summaries["coil_features"], summaries["raw_currents"]
    summary = (f"{len(rows)} training frames. Largest |r| with Ip (signed, magnitude): coil features "
               f"{c['with_ip']['ip__mean']['max_abs_r']:.3f}, {c['with_ip']['abs_ip']['max_abs_r']:.3f}; raw currents "
               f"{r['with_ip']['ip__mean']['max_abs_r']:.3f}, {r['with_ip']['abs_ip']['max_abs_r']:.3f}. Median |r| between "
               f"columns: coil features {c['median_abs_r']:.3f}, raw currents {r['median_abs_r']:.3f}")
    return fc.Check("X12", "Shared factor removed", "review", summary), {"n_frames": int(len(rows)), **summaries}, long


def check_x14(data: dict[str, pd.DataFrame]) -> tuple[fc.Check, dict]:
    fits = {}
    for subset in ("all", "flat_top"):
        reference = None
        for group in ("train", "diii_d_public_test", "mast_public_test"):
            rows = data[group] if subset == "all" else data[group].loc[data[group]["phase"] == "flat_top"]
            fit = fc.line_fit(rows["ts_beta_p"], rows["shafranov_lambda"])
            reference = fit if group == "train" else reference
            fit["median_offset_from_train_line"] = fc.median_offset(rows["ts_beta_p"], rows["shafranov_lambda"], reference)
            fits.setdefault(subset, {})[group] = fit
    a = fits["all"]
    summary = ("shafranov_lambda against ts_beta_p, slope (r): "
               + ", ".join(f"{g} {f['slope']:.3f} ({f['r']:.3f})" for g, f in a.items())
               + "; median offset from the training line: "
               + ", ".join(f"{g} {f['median_offset_from_train_line']:+.3f}" for g, f in a.items() if g != "train"))
    return fc.Check("X14", "Thomson against force balance", "review", summary), fits


def check_x15(data: dict[str, pd.DataFrame], thresholds: list[float]) -> tuple[fc.Check, dict]:
    labels = {"train": "DIII-D train", "diii_d_public_test": "DIII-D test", "mast_public_test": "MAST"}
    quantities = {"ip__rate50 (kA/s)": {}, "ip_rate_rel": {}, "time since the Ip rise (s)": {}, "resistive_age": {}}
    rise, changing = {}, {}
    for group, label in labels.items():
        t = data[group]
        quantities["ip__rate50 (kA/s)"][label] = t["ip__rate50"].to_numpy(dtype=np.float64)
        quantities["ip_rate_rel"][label] = t["ip_rate_rel"].to_numpy(dtype=np.float64)
        quantities["time since the Ip rise (s)"][label] = (t["time_ms"] - t["ip_rise_ms"]).to_numpy(dtype=np.float64) / 1000.0
        quantities["resistive_age"][label] = t["resistive_age"].to_numpy(dtype=np.float64)
        first = t.drop_duplicates("shot")["ip_rise_ms"].to_numpy(dtype=np.float64)
        rise[label] = fc.reference_statistics(first)
        changing[label] = fc.changing_current_shares(t["ip_rate_rel"], t["phase"], thresholds)
    bands = fc.band_table(quantities, reference="DIII-D train")
    width = bands.set_index(["quantity", "machine"])["width_over_reference"]
    summary = (f"MAST band width over DIII-D training: ip__rate50 {width[('ip__rate50 (kA/s)', 'MAST')]:.2f}, "
               f"ip_rate_rel {width[('ip_rate_rel', 'MAST')]:.2f}; time since rise {width[('time since the Ip rise (s)', 'MAST')]:.2f}, "
               f"resistive_age {width[('resistive_age', 'MAST')]:.2f}. Frames off the flat-top: "
               + ", ".join(f"{k} {v['share_off_flat_top']:.3f}" for k, v in changing.items())
               + ". Median ip_rise_ms: " + ", ".join(f"{k} {v['q50']:.1f}" for k, v in rise.items()))
    return fc.Check("X15", "Timescales", "review", summary), {"bands": bands.to_dict(orient="records"), "ip_rise_ms": rise,
                                                              "changing_current": changing, "thresholds": thresholds}


# ------------------------------------------------------------------------------- report
def _fmt(v, digits: int = 4) -> str:
    return "nan" if v is None or not np.isfinite(v) else f"{v:.{digits}f}"


def reference_note(ref: dict) -> str:
    """One line on the reference version's numbers: shown, or why not."""
    if ref["comparable"]:
        return (f"Version {ref['version']} beside this version: `{ref['path']}`, same split file, samples and seed, so the "
                f"same sampled frames; only the sets and the scaling differ.")
    def shown_value(key: str, value):
        return display_path(value) if key == "split" and value not in (None, "missing") else value
    return (f"Version {ref['version']} not shown: its run differs in "
            + ", ".join(f"{k} ({shown_value(k, a)} against {shown_value(k, b)})" for k, (a, b) in ref["differences"].items())
            + ".")


def render_report(checks: list[fc.Check], info: dict, ranges: pd.DataFrame, shares: pd.DataFrame,
                  extremes: pd.DataFrame, direction: pd.DataFrame) -> str:
    lines = [f"# Feature profile, version {info['version']}", "",
             f"Feature tables `{info['feature_dir']}`, configuration hash `{info['config_hash'][:12]}`. Reference of every "
             f"comparison: the {info['n_rows']['train']} frames of the {info['n_shots']['train']} shots of the `train` list of "
             f"`{Path(info['split_file']).name}`. Nothing is fitted on test inputs.", "",
             fc.check_table_markdown(checks), "", "X13 (the pattern before and after scaling) is shown in notebook 06, "
             "section 3; X16 is written by `scripts/feature_linear_check.py` to `linear_check.md`.", ""]

    lines += ["## X3. Completeness", "", "Share of finite values per feature and split. Coil features are NaN where a coil "
              "current is missing or Ip is below the threshold; Thomson features where `ts_valid` is 0.", "",
              "| Feature | " + " | ".join(shares.columns) + " |", "| --- " * (len(shares.columns) + 1) + "|"]
    for feature, row in shares.iterrows():
        lines.append(f"| {feature} | " + " | ".join(_fmt(v) for v in row.to_numpy()) + " |")
    lines += ["", f"Frames with Ip missing or below {info['x3']['min_abs_ip_ka']:g} kA in magnitude:", "",
              "| Split | Frames | Ip missing | Below threshold | Share affected |", "| --- | --- | --- | --- | --- |"]
    for s, d in info["x3"]["low_ip"].items():
        lines.append(f"| {s} | {d['n_frames']} | {d['n_missing_ip']} | {d['n_below_min']} | {_fmt(d['share_affected'], 6)} |")

    sw = info["x5"]["swap_difference"]
    lines += ["", "## X5. Mirror", "", f"Mirroring twice: " + ", ".join(f"{s} {'identity' if v else 'differs'}"
                                                                       for s, v in info["x5"]["identity"].items()) + ".",
              f"Training shots of the A/B exchange: {', '.join(sw['shots'])}.", ""]

    lines += ["## X6. Coil terms by current direction (DIII-D training)", "",
              "| Term | Median + | Median - | IQR + | IQR - | Shift (IQR +) | Share - inside + band |", "| --- " * 7 + "|"]
    for r in direction.itertuples(index=False):
        lines.append(f"| {r.feature} | {_fmt(r.median_positive)} | {_fmt(r.median_negative)} | {_fmt(r.iqr_positive)} | "
                     f"{_fmt(r.iqr_negative)} | {_fmt(r.median_shift_iqr, 2)} | {_fmt(r.share_negative_in_positive_band, 3)} |")

    x7 = info["x7"]
    lines += ["", "## X7. Force-balance sign", "", f"K9 medians from {x7['k9_source']}. Flat-top frames: "
              + ", ".join(f"{m} {n}" for m, n in x7["n_frames"].items()) + ".", ""]

    lines += ["## X8. Per-feature ranges (all frames)", "",
              "Share outside the training 1st to 99th percentile band and outside the training range; median shift in "
              "training interquartile ranges.", "",
              "| Feature | Train band | Val band / range / shift | DIII-D test band / range / shift | MAST band / range / shift |",
              "| --- | --- | --- | --- | --- |"]
    all_rows = ranges.loc[ranges["phase"] == "all"]
    for feature, g in all_rows.groupby("feature", sort=False):
        cells = []
        for split in COMPARED:
            r = g.loc[g["split"] == split].iloc[0]
            cells.append(f"{_fmt(r.share_outside_band, 3)} / {_fmt(r.share_outside_range, 3)} / {_fmt(r.median_shift_iqr, 2)}")
        r0 = g.iloc[0]
        lines.append(f"| {feature} | {_fmt(r0['ref_q01'], 3)} to {_fmt(r0['ref_q99'], 3)} | " + " | ".join(cells) + " |")

    lines += ["", "## X9. Stretch terms by phase", "",
              "| Term | Phase | Train band | Train median | MAST median | MAST outside band | MAST outside range | "
              "DIII-D test outside band |", "| --- " * 8 + "|"]
    for feature in stretch_columns(info["config"]):
        for phase in fc.PHASES:
            g = ranges.loc[(ranges["feature"] == feature) & (ranges["phase"] == phase)].set_index("split")
            if g.empty:
                continue
            m, t = g.loc["mast_public_test"], g.loc["diii_d_public_test"]
            lines.append(f"| {feature} | {phase} | {_fmt(m['ref_q01'], 3)} to {_fmt(m['ref_q99'], 3)} | "
                         f"{_fmt(m['ref_median'], 3)} | {_fmt(m['median'], 3)} | {_fmt(m['share_outside_band'], 3)} | "
                         f"{_fmt(m['share_outside_range'], 3)} | {_fmt(t['share_outside_band'], 3)} |")

    x10 = info["x10"]
    ref = info.get("reference")
    shown = ref is not None and ref["comparable"]
    lines += ["", "## X10. Combined distance", "",
              f"Scaled features, {x10['n_train_sample']} training frames (seed {x10['seed']}), up to {x10['n_eval_sample']} "
              f"frames per compared split. With mirrors: the nearest of the training frames and their mirrors.", ""]
    if ref is not None:
        lines += [reference_note(ref), ""]
    header = "| Set | Split | Median | 90th | 99th | Max | Beyond validation 99th | Median with mirrors |"
    if shown:
        v = ref["version"]
        header += f" v{v} median | v{v} 99th | v{v} beyond validation 99th |"
    lines += [header, "| --- " * (11 if shown else 8) + "|"]
    with_m = {(r["set"], r["split"]): r for r in x10["summary_with_mirrors"]}
    ref_rows = {(r["set"], r["split"]): r for r in ref["x10_summary"]} if shown else {}
    for r in x10["summary"]:
        line = (f"| {r['set']} | {r['split']} | {_fmt(r['p50'], 3)} | {_fmt(r['p90'], 3)} | {_fmt(r['p99'], 3)} | "
                f"{_fmt(r['max'], 3)} | {_fmt(r['share_beyond_validation_p99'], 3)} | {_fmt(with_m[(r['set'], r['split'])]['p50'], 3)} |")
        if shown:
            q = ref_rows.get((r["set"], r["split"]), {})
            line += (f" {_fmt(q.get('p50'), 3)} | {_fmt(q.get('p99'), 3)} | {_fmt(q.get('share_beyond_validation_p99'), 3)} |")
        lines.append(line)
    lines += ["", "Median share of the squared distance carried by each feature group:", ""]
    for r in x10["group_share_medians"]:
        lines.append(f"- {r['set']}, {r['split']}: " + ", ".join(f"{k[6:]} {_fmt(v, 3)}" for k, v in r.items()
                                                                 if k.startswith("share_") and v == v))

    lines += ["", "## X11. Scaled values beyond 10 in magnitude", ""]
    if not shown:
        lines += ["Features with a non-zero share on any split.", "",
                  "| Feature | " + " | ".join(extremes.columns) + " |", "| --- " * (len(extremes.columns) + 1) + "|"]
        for feature, row in extremes.loc[(extremes > 0).any(axis=1)].iterrows():
            lines.append(f"| {feature} | " + " | ".join(_fmt(v, 5) for v in row.to_numpy()) + " |")
    else:
        v = ref["version"]
        ref_x11 = ref["x11"]
        lines += [f"Features with a non-zero share on any split in either version; each cell is this version's share, "
                  f"then version {v}'s in brackets (a dash where that version has no such feature).", "",
                  "| Feature | " + " | ".join(extremes.columns) + " |", "| --- " * (len(extremes.columns) + 1) + "|"]
        features = list(dict.fromkeys([*extremes.index, *ref_x11]))
        for feature in features:
            own = extremes.loc[feature] if feature in extremes.index else None
            old = ref_x11.get(feature)
            own_any = own is not None and bool((own > 0).any())
            old_any = old is not None and any(x is not None and x > 0 for x in old.values())
            if not (own_any or old_any):
                continue
            cells = [f"{'-' if own is None else _fmt(own[g], 5)} ({'-' if old is None else _fmt(old.get(g), 5)})"
                     for g in extremes.columns]
            lines.append(f"| {feature} | " + " | ".join(cells) + " |")

    x12 = info["x12"]
    lines += ["", "## X12. Correlations (DIII-D training)", ""]
    for kind, s in ((k, x12[k]) for k in ("raw_currents", "coil_features")):
        lines.append(f"- {kind}: median |r| {_fmt(s['median_abs_r'], 3)}, largest {_fmt(s['max_abs_r'], 3)} "
                     f"({' and '.join(s['max_pair'] or [])}); with Ip: " + ", ".join(
                         f"{ip} largest {_fmt(v['max_abs_r'], 3)} ({v['max_column']}), median {_fmt(v['median_abs_r'], 3)}"
                         for ip, v in s["with_ip"].items()))

    lines += ["", "## X14. Thomson against force balance", "", "Line `shafranov_lambda = intercept + slope ts_beta_p` over "
              "frames inside both 1st to 99th percentile bands of the split.", "",
              "| Frames | Split | n | Slope | Intercept | r | Residual std | Median offset from training line |", "| --- " * 8 + "|"]
    for subset, fits in info["x14"].items():
        for g, f in fits.items():
            lines.append(f"| {subset} | {g} | {f['n']} | {_fmt(f['slope'], 3)} | {_fmt(f['intercept'], 3)} | {_fmt(f['r'], 3)} | "
                         f"{_fmt(f['residual_std'], 3)} | {_fmt(f['median_offset_from_train_line'], 3)} |")

    x15 = info["x15"]
    lines += ["", "## X15. Timescales", "", "| Quantity | Machine | 1st | Median | 99th | Width over DIII-D training |",
              "| --- " * 6 + "|"]
    for r in x15["bands"]:
        lines.append(f"| {r['quantity']} | {r['machine']} | {_fmt(r['q01'], 3)} | {_fmt(r['q50'], 3)} | {_fmt(r['q99'], 3)} | "
                     f"{_fmt(r['width_over_reference'], 2)} |")
    lines += ["", "Changing current (open item 6): share of frames off the flat-top, and share with abs(`ip_rate_rel`) above "
              "each threshold (Ip changing by that fraction of itself per resistive time).", ""]
    for k, v in x15["changing_current"].items():
        lines.append(f"- {k}: off the flat-top {_fmt(v['share_off_flat_top'], 3)}; " + ", ".join(
            f"above {t:g} {_fmt(v[f'share_abs_rate_rel_above_{t:g}'], 3)}" for t in x15["thresholds"]))
    lines += ["", "`ip_rise_ms` per shot (10 percent crossing):", ""]
    for k, v in x15["ip_rise_ms"].items():
        lines.append(f"- {k}: {v['n']} shots, 1st {_fmt(v['q01'], 1)}, median {_fmt(v['q50'], 1)}, 99th {_fmt(v['q99'], 1)} ms")
    lines += ["", f"Runtime {info['runtime_s']:.0f} s.", ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, default=fe.DEFAULT_CONFIG)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT, help="split file: train list (reference) and validation list")
    parser.add_argument("--feature-root", type=Path, default=fe.DEFAULT_FEATURE_ROOT)
    parser.add_argument("--report-root", type=Path, default=fe.DEFAULT_REPORT_ROOT)
    parser.add_argument("--table-dir", type=Path, default=ft.DEFAULT_TABLE_ROOT)
    parser.add_argument("--green-dir", type=Path, default=fe.DEFAULT_GREEN_DIR)
    parser.add_argument("--data-root", type=Path, default=default_data_root())
    parser.add_argument("--k9", type=Path, default=DEFAULT_K9, help="reference_ratios.parquet of fit_coil_scales.py (K9)")
    parser.add_argument("--nn-train-frames", type=int, default=200_000)
    parser.add_argument("--nn-eval-frames", type=int, default=5_000)
    parser.add_argument("--corr-frames", type=int, default=200_000)
    parser.add_argument("--mirror-shots", type=int, default=20)
    parser.add_argument("--rate-thresholds", type=float, nargs="+", default=[0.1, 0.3, 1.0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reference-version", type=int, default=1,
                        help="version whose X10 and X11 are shown beside this version's (none for that version itself)")
    args = parser.parse_args(argv)

    started = time.time()
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    config = fe.FeatureConfig.load(args.config)
    out_dir = Path(args.report_root) / f"v{config.version}"
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_dir = fe.features_dir(config.version, args.feature_root)
    build = json.loads((feature_dir / "build.json").read_text())
    scaler = RobustScaler.load(fe.scaler_path(config.version, args.report_root))
    split = sp.read_split(args.split)
    train_ids, val_ids = sorted(split["train"]), sorted(split["validation"])

    data = {}
    for s in dataset.DATA_SPLITS:
        data[s] = load_split(s, config, args)
        log(f"{s}: {len(data[s])} rows")
    d3d = data["diii_d_train"]
    groups = {"train": d3d.loc[d3d["shot"].isin(train_ids)].reset_index(drop=True),
              "validation": d3d.loc[d3d["shot"].isin(val_ids)].reset_index(drop=True),
              "diii_d_public_test": data["diii_d_public_test"], "mast_public_test": data["mast_public_test"]}
    data.update(groups)
    index = pd.read_parquet(Path(args.table_dir) / "shot_index.parquet")
    index = index.loc[index["split"] == "diii_d_train"].assign(shot=lambda t: t["shot"].astype(str))

    checks, info = [], {"version": config.version, "config_hash": config.hash(), "feature_dir": display_path(feature_dir),
                        "split_file": display_path(args.split), "n_rows": {g: int(len(data[g])) for g in GROUPS},
                        "n_shots": {g: int(data[g]["shot"].nunique()) for g in GROUPS}}
    c, info["x1"] = check_x1(build, config, data); checks.append(c)
    c, info["x2"] = check_x2(build, config); checks.append(c)
    c, info["x3"], shares = check_x3(data, config); checks.append(c)
    c, info["x4"] = check_x4(build, config, out_dir / "layout_clearance.csv"); checks.append(c)
    c, info["x5"] = check_x5(data, config, train_ids, args); checks.append(c)
    log("X1 to X5 done")
    c, info["x6"], direction = check_x6(data["train"], index); checks.append(c)
    c, info["x7"] = check_x7(data, args); checks.append(c)

    columns = config.all_columns()
    ranges = fc.compare_ranges_by_phase(data["train"][columns], data["train"]["phase"], {g: data[g][columns] for g in COMPARED},
                                        {g: data[g]["phase"] for g in COMPARED}, columns)
    ranges.to_parquet(out_dir / "range_comparison.parquet", index=False)
    mast_all = ranges.loc[(ranges["phase"] == "all") & (ranges["split"] == "mast_public_test")]
    top = mast_all.nlargest(3, "share_outside_band")
    checks.append(fc.Check("X8", "Per-feature ranges", "review",
                           "largest MAST share outside the training band: " + ", ".join(
                               f"{r.feature} {r.share_outside_band:.3f}" for r in top.itertuples(index=False))))
    stretch = ranges.loc[ranges["feature"].isin(stretch_columns(config)) & (ranges["split"] == "mast_public_test")]
    checks.append(fc.Check("X9", "Stretch terms", "review", "MAST share outside the training band by phase: " + "; ".join(
        f"{r.feature} {r.phase} {_fmt(r.share_outside_band, 3)}" for r in stretch.itertuples(index=False))))
    log("X6 to X9 done")

    scaled = scaled_matrices(data, scaler)
    c, info["x10"], nn = check_x10(data, scaled, scaler, config, args); checks.append(c)
    reference = load_reference(config, args, len(data["train"]))
    if reference is not None:
        info["reference"] = {k: v for k, v in reference.items() if k not in ("x10_summary", "x11")}
    nn.to_parquet(out_dir / "nn_distance.parquet", index=False)
    c, extremes = check_x11(scaled, scaler); checks.append(c)
    info["x11"] = extremes.to_dict(orient="index")
    c, info["x12"], correlations = check_x12(data["train"], config, args); checks.append(c)
    correlations.to_parquet(out_dir / "correlations.parquet", index=False)
    checks.append(fc.Check("X13", "Pattern preserved", "review", "shown in notebook 06, section 3"))
    c, info["x14"] = check_x14(data); checks.append(c)
    c, info["x15"] = check_x15(data, list(args.rate_thresholds)); checks.append(c)
    log("X10 to X15 done")

    info["checks"] = [c.to_dict() for c in checks]
    info.update(run_info(argv, vars(args), timestamp, time.time() - started))
    (out_dir / "profile.json").write_text(json.dumps(info, indent=2, default=fc.json_default) + "\n")
    report = render_report(checks, {**info, "config": config, "reference": reference}, ranges, shares, extremes, direction)
    (out_dir / "feature_report.md").write_text(report)
    print(fc.check_table_markdown(checks))
    log(f"wrote {out_dir} in {time.time() - started:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
