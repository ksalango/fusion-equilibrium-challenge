"""Worst MAST shots of a submission against the rest of the MAST test split (diagnostic,
inputs only).

The scorer's detailed results list the 20 MAST shots with the lowest R2 psi by submission key
(``shot_XXXX``, the 0-indexed position in the submission file). The key is mapped to a shot id
through the submission's shot order, recorded in ``run.json`` (``folds.MAST.shot_checks``:
index and shot of every shot, in the order they were written). The worst shots are split by
D_LCFS at ``--dlcfs-split`` (0.45 m) into two groups and compared with the rest of the split.

Per shot and per group, from the frame table and the feature table (version 2): number of
frames, median and peak abs(Ip), share of frames by phase (``feature_checks.frame_phases``),
``ip_rise_ms``, and the medians of ``coil_c10``, ``shaping``, ``coil_c02``, ``q_cyl``,
``resistive_age`` and ``ts_beta_p``. On ``--frames-per-shot`` evenly spaced frames of each
worst shot and of ``--rest-shots`` shots of the rest (fixed seed), the mean-current map of the
target build (``target_build.mean_current_maps``, no remainder on MAST) and the boundary the
scorer finds in it, with the Y16 quantities of ``scripts/profile_targets.py``: R_axis, kappa
and the boundary's inner and outer radius, width and height.

No equilibrium label is read: the MAST test files have none, and every column read passes
``frame_table.assert_input_columns``. Writes ``reports/submissions/<run>/worst_mast_shots.md``.
Usage::

    python scripts/worst_mast_shots.py
    python scripts/worst_mast_shots.py --run 2026-10-14_mean_current_v0 --frames-per-shot 5 --rest-shots 100
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import time
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
import pandas as pd

from fusion_eq import (dataset, feature_checks as fc, features as fe, frame_table as ft, predictors, target_build as tb,
                       target_checks as tc, targets)
from fusion_eq.coil_flux import adopted_scales

DEFAULT_RUN = "2026-10-14_mean_current_v0"
MACHINE = "MAST"
SPLIT = "mast_public_test"
KEY_PATTERN = re.compile(r"^shot_(\d+)$")
FEATURE_COLUMNS = ("coil_c10", "shaping", "coil_c02", "q_cyl", "resistive_age", "ts_beta_p")
FRAME_COLUMNS = ("ip__mean", "ip_rise_ms")
PHASES = ("rising", "flat_top", "late")
# Y16 quantities of the boundary found in the mean-current map (profile_targets.derived_on_maps).
BOUNDARY_KEYS = ("R_axis", "kappa", "inner_R", "outer_R", "width", "height")
BOUNDARY_LABELS = {"R_axis": "R_axis (m)", "kappa": "kappa", "inner_R": "Inner R (m)", "outer_R": "Outer R (m)",
                   "width": "Width (m)", "height": "Height (m)"}


def log(message: str) -> None:
    print(message, flush=True)


def display_path(path: Path | str) -> str:
    return os.path.relpath(Path(path).resolve(), dataset.REPO_ROOT)


def load_script(name: str):
    """A repository script as a module, so the Y16 quantities come from the profile's own code."""
    spec = importlib.util.spec_from_file_location(name, dataset.REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------- scorer's report
class _Tables(HTMLParser):
    """Tables of an HTML page with the last h2 heading and the last h4 heading before each."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tables: list[dict] = []
        self._heading: str | None = None
        self._text: list[str] = []
        self._section = self._sub = ""
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag, attrs):
        if tag in ("h2", "h4"):
            self._heading, self._text = tag, []
        elif tag == "table":
            self.tables.append({"section": self._section, "title": self._sub, "rows": []})
        elif tag == "tr" and self.tables:
            self._row = []
        elif tag in ("td", "th") and self._row is not None:
            self._cell = []

    def handle_endtag(self, tag):
        if tag == self._heading:
            text = " ".join("".join(self._text).split())
            if tag == "h2":
                self._section, self._sub = text, ""
            else:
                self._sub = text
            self._heading = None
        elif tag in ("td", "th") and self._cell is not None:
            self._row.append(" ".join("".join(self._cell).split()))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            self.tables[-1]["rows"].append(self._row)
            self._row = None

    def handle_data(self, data):
        if self._heading:
            self._text.append(data)
        if self._cell is not None:
            self._cell.append(data)


def worst_table(page: str, machine: str = MACHINE) -> pd.DataFrame:
    """The "Worst 20 shots" table of one machine's section of the scorer's detailed results.

    Returns:
        Columns ``key``, ``index`` (the position in the submission file), ``r2_psi`` and
        ``d_lcfs`` (m), in the page's order.

    Raises:
        ValueError: when the section or its table is missing, or a key does not read
            ``shot_<index>``.
    """
    parser = _Tables()
    parser.feed(page)
    found = [t for t in parser.tables if t["section"] == machine and t["title"].startswith("Worst")]
    if not found:
        raise ValueError(f"no 'Worst' table in the {machine} section of the detailed results")
    header, *rows = found[0]["rows"]
    col = {name: i for i, name in enumerate(header)}
    for name in ("submission key", "R2_psi", "D_LCFS"):
        if name not in col:
            raise ValueError(f"the {machine} worst-shot table has no column {name!r}: {header}")
    out = []
    for r in rows:
        key = r[col["submission key"]]
        m = KEY_PATTERN.match(key)
        if not m:
            raise ValueError(f"submission key {key!r} does not read shot_<index>")
        out.append({"key": key, "index": int(m.group(1)), "r2_psi": float(r[col["R2_psi"]]), "d_lcfs": float(r[col["D_LCFS"]])})
    return pd.DataFrame(out, columns=["key", "index", "r2_psi", "d_lcfs"])


def shot_order(record: dict, machine: str = MACHINE) -> list[str]:
    """Shot ids of a machine's submission file in writing order, from ``run.json``.

    Raises:
        ValueError: when the recorded indices are not 0 to n - 1 or their number differs from
            the fold's shot count.
    """
    fold = record["folds"][machine]
    checks = fold["shot_checks"]
    indices = [int(c["index"]) for c in checks]
    if indices != list(range(len(checks))):
        raise ValueError(f"{machine} shot_checks indices are not 0 to {len(checks) - 1} in order")
    if len(checks) != int(fold["n_shots"]):
        raise ValueError(f"{machine}: {len(checks)} shot_checks for {fold['n_shots']} shots")
    return [str(c["shot"]) for c in checks]


def map_keys(worst: pd.DataFrame, order: list[str]) -> pd.DataFrame:
    """The worst-shot table with the shot id of each submission key."""
    bad = [k for k, i in zip(worst["key"], worst["index"]) if not 0 <= i < len(order)]
    if bad:
        raise ValueError(f"submission keys outside the {len(order)} shots of the file: {bad}")
    return worst.assign(shot=[order[i] for i in worst["index"]])


def assign_groups(worst: pd.DataFrame, split_m: float) -> pd.DataFrame:
    """``group`` per worst shot: ``low`` below ``split_m`` of D_LCFS, ``high`` at or above."""
    return worst.assign(group=np.where(worst["d_lcfs"] < split_m, "low", "high"))


# ------------------------------------------------------------------------------ inputs
def read_inputs(table_dir: Path, feature_root: Path, version: int) -> pd.DataFrame:
    """Feature rows of the MAST test split with ``ip__mean``, ``ip_rise_ms``, ``n_frames``
    and ``phase`` beside them."""
    ft.assert_input_columns(FRAME_COLUMNS, MACHINE)
    features = fe.read_features(SPLIT, version, [*FEATURE_COLUMNS, "ip_frac"], root=feature_root)
    frames = ft.read_frame_table(SPLIT, ["shot", "frame", *FRAME_COLUMNS], root=table_dir)
    frames["shot"] = frames["shot"].astype(str)
    return fc.join_frame_columns(features, frames)


def shot_stats(rows: pd.DataFrame) -> pd.DataFrame:
    """One row per shot: frames, median and peak abs(Ip) in kA, share of frames per phase,
    ``ip_rise_ms`` and the medians of the feature columns."""
    out = []
    for shot, g in rows.groupby("shot", sort=True):
        ip = np.abs(g["ip__mean"].to_numpy(dtype=np.float64))
        entry = {"shot": str(shot), "frames": len(g), "ip_median_ka": float(np.nanmedian(ip)) if np.isfinite(ip).any() else np.nan,
                 "ip_peak_ka": float(np.nanmax(ip)) if np.isfinite(ip).any() else np.nan,
                 **{f"share_{p}": float((g["phase"] == p).mean()) for p in PHASES},
                 "ip_rise_ms": float(g["ip_rise_ms"].iloc[0])}
        for c in FEATURE_COLUMNS:
            v = g[c].to_numpy(dtype=np.float64)
            entry[c] = float(np.nanmedian(v)) if np.isfinite(v).any() else np.nan
        out.append(entry)
    return pd.DataFrame(out)


def spaced_frames(n: int, k: int) -> np.ndarray:
    """``k`` evenly spaced positions in ``0 .. n - 1`` (all positions when ``n <= k``)."""
    if n <= 0:
        return np.zeros(0, dtype=int)
    return np.unique(np.rint(np.linspace(0, n - 1, min(k, n))).astype(int))


# ----------------------------------------------------------------------------- summary
def spread_text(values, digits: int = 3) -> str:
    """``median (5th to 95th)`` of the finite values, ``n/a`` when none."""
    q = tc.quantiles(values)
    if not q["n"]:
        return "n/a"
    return f"{q['q50']:.{digits}f} ({q['q5']:.{digits}f} to {q['q95']:.{digits}f})"


def group_rows(rows: pd.DataFrame, stats: pd.DataFrame, groups: dict[str, list[str]], boundary: dict[str, dict]) -> list[dict]:
    """Rows of the group table: shot-level quantities over the group's shots, frame-level
    quantities over its frames (median, 5th to 95th percentile), phase shares over its
    frames, and the boundary quantities over its sampled frames."""
    table = []

    def add(name: str, values_of) -> None:
        table.append({"quantity": name, **{g: values_of(g, shots) for g, shots in groups.items()}})

    def of_shots(column: str, digits: int = 3):
        return lambda g, shots: spread_text(stats.loc[stats["shot"].isin(shots), column], digits)

    def of_frames(column: str):
        return lambda g, shots: spread_text(rows.loc[rows["shot"].isin(shots), column])

    add("Shots", lambda g, shots: str(len(shots)))
    add("Frames per shot", of_shots("frames", 0))
    add("Median abs(Ip) per shot (kA)", of_shots("ip_median_ka", 0))
    add("Peak abs(Ip) per shot (kA)", of_shots("ip_peak_ka", 0))
    for p in PHASES:
        add(f"Share of frames, {p}", lambda g, shots, p=p: (
            f"{(rows.loc[rows['shot'].isin(shots), 'phase'] == p).mean():.3f}" if rows["shot"].isin(shots).any() else "n/a"))
    add("ip_rise_ms per shot", of_shots("ip_rise_ms", 1))
    for c in FEATURE_COLUMNS:
        add(c, of_frames(c))
    add("Boundary found (sampled frames)", lambda g, shots: (
        f"{boundary[g]['boundary_share']:.3f} of {boundary[g]['n_frames']}" if boundary.get(g, {}).get("n_frames") else "n/a"))
    for k in BOUNDARY_KEYS:
        add(f"Mean-current boundary: {BOUNDARY_LABELS[k]}", lambda g, shots, k=k: (
            spread_quantiles(boundary[g]["spread"][k]) if boundary.get(g, {}).get("n_frames") else "n/a"))
    return table


def spread_quantiles(q: dict, digits: int = 3) -> str:
    if not q.get("n"):
        return "n/a"
    return f"{q['q50']:.{digits}f} ({q['q5']:.{digits}f} to {q['q95']:.{digits}f})"


def markdown(rows: list[dict], columns: list[str], header: list[str]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "| --- " * len(header) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(r[c]) for c in columns) + " |")
    return lines


def fmt(value, digits: int = 3) -> str:
    return "n/a" if value is None or not np.isfinite(value) else f"{value:.{digits}f}"


def render(info: dict, worst: pd.DataFrame, stats: pd.DataFrame, per_shot_boundary: dict, groups: dict, group_table: list[dict]) -> str:
    names = info["group_names"]
    lines = [f"# Worst MAST shots of submission {info['run']}", "",
             f"Scorer's worst 20 MAST shots by R2 psi (`{info['detailed_results']}`), mapped from submission key to shot id "
             f"through the shot order of `{info['run_json']}` ({info['n_shots']} shots, indices 0 to {info['n_shots'] - 1}). "
             f"Groups: D_LCFS below {info['dlcfs_split']} m and at or above it; the rest is every other shot of "
             f"`{SPLIT}`. Inputs only: frame table `{info['table_dir']}` and features version {info['feature_version']} "
             f"(`{info['feature_root']}`). No equilibrium label is read.", "",
             f"Mean-current boundary: target build `{info['target_dir']}` (configuration hash `{info['config_hash'][:12]}`; "
             f"the submission's build `{info['submission_config_hash'][:12]}`), {info['frames_per_shot']} evenly spaced frames "
             f"per shot, of every worst shot and of {info['n_rest_sampled']} shots of the rest (seed {info['seed']}); quantities "
             "of Y16 (`scripts/profile_targets.py`).", ""]
    if info["config_hash"] != info["submission_config_hash"]:
        lines += ["The target build differs from the one the submission used; the boundary columns describe the current build.", ""]
    lines += ["## Worst 20 shots", ""]
    s = stats.set_index("shot")
    rows = []
    for r in worst.sort_values(["group", "d_lcfs"]).itertuples():
        st = s.loc[r.shot] if r.shot in s.index else None
        b = per_shot_boundary.get(r.shot, {})
        rows.append({"key": r.key, "shot": f"`{r.shot}`", "group": names[r.group], "r2": f"{r.r2_psi:.4f}", "d": f"{r.d_lcfs:.4f}",
                     "frames": "n/a" if st is None else int(st["frames"]),
                     "ipm": "n/a" if st is None else fmt(st["ip_median_ka"], 0), "ipp": "n/a" if st is None else fmt(st["ip_peak_ka"], 0),
                     **{f"ph_{p}": "n/a" if st is None else fmt(st[f"share_{p}"]) for p in PHASES},
                     "rise": "n/a" if st is None else fmt(st["ip_rise_ms"], 1),
                     **{c: "n/a" if st is None else fmt(st[c], 4) for c in FEATURE_COLUMNS},
                     "found": f"{b['boundary_share']:.2f}" if b.get("n_frames") else "n/a",
                     **{f"b_{k}": fmt(b.get(f"median_{k}", np.nan)) for k in BOUNDARY_KEYS}})
    lines += markdown(rows, ["key", "shot", "group", "r2", "d", "frames", "ipm", "ipp", "ph_rising", "ph_flat_top", "ph_late", "rise"],
                      ["Key", "Shot", "Group", "R2 psi", "D_LCFS (m)", "Frames", "Median abs(Ip) (kA)", "Peak abs(Ip) (kA)",
                       "Rising", "Flat-top", "Late", "ip_rise_ms"])
    lines += ["", "Feature medians over each shot's frames:", ""]
    lines += markdown(rows, ["key", "shot", "group", *FEATURE_COLUMNS], ["Key", "Shot", "Group", *FEATURE_COLUMNS])
    lines += ["", "Mean-current boundary, medians over the shot's sampled frames:", ""]
    lines += markdown(rows, ["key", "shot", "group", "found", *[f"b_{k}" for k in BOUNDARY_KEYS]],
                      ["Key", "Shot", "Group", "Boundary found", *BOUNDARY_LABELS.values()])
    lines += ["", "## Groups against the rest", "",
              "Median (5th to 95th percentile): over the group's shots for the per-shot rows, over its frames for the features, "
              "over its sampled frames for the boundary. Phase shares over the group's frames.", ""]
    lines += markdown(group_table, ["quantity", *groups], ["Quantity", *[f"{names[g]}" for g in groups]])
    lines += ["", f"Runtime {info['runtime_s']:.0f} s.", ""]
    return "\n".join(lines)


# -------------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", default=DEFAULT_RUN, help="run label <date>_<name>")
    parser.add_argument("--submissions-dir", type=Path, default=dataset.REPO_ROOT / "submissions")
    parser.add_argument("--reports-dir", type=Path, default=dataset.REPO_ROOT / "reports" / "submissions")
    parser.add_argument("--detailed-results", type=Path, default=None,
                        help="default: <submissions-dir>/<run>/codabench_scoring_result/detailed_results.html")
    parser.add_argument("--table-dir", type=Path, default=ft.DEFAULT_TABLE_ROOT)
    parser.add_argument("--feature-root", type=Path, default=fe.DEFAULT_FEATURE_ROOT)
    parser.add_argument("--feature-version", type=int, default=2)
    parser.add_argument("--targets", type=Path, default=tb.DEFAULT_TARGET_ROOT)
    parser.add_argument("--target-config", type=Path, default=tb.DEFAULT_CONFIG)
    parser.add_argument("--green-dir", type=Path, default=predictors.DEFAULT_GREEN_DIR)
    parser.add_argument("--dlcfs-split", type=float, default=0.45, help="D_LCFS (m) between the two worst-shot groups")
    parser.add_argument("--frames-per-shot", type=int, default=5)
    parser.add_argument("--rest-shots", type=int, default=100, help="shots of the rest sampled for the boundary")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None, help="default: <reports-dir>/<run>/worst_mast_shots.md")
    args = parser.parse_args(argv)
    started = time.time()

    report_dir = args.reports_dir / args.run
    run_json = report_dir / "run.json"
    detailed = args.detailed_results or args.submissions_dir / args.run / "codabench_scoring_result" / "detailed_results.html"
    out = args.out or report_dir / "worst_mast_shots.md"
    record = json.loads(run_json.read_text())
    order = shot_order(record)
    worst = assign_groups(map_keys(worst_table(detailed.read_text()), order), args.dlcfs_split)
    log(f"{len(worst)} worst MAST shots mapped through {display_path(run_json)}")

    rows = read_inputs(args.table_dir, args.feature_root, args.feature_version)
    missing = sorted(set(worst["shot"]) - set(rows["shot"]))
    if missing:
        raise ValueError(f"worst shots not in the {SPLIT} tables: {missing}")
    if sorted(set(rows["shot"])) != sorted(order):
        log("warning: the shots of the tables differ from the submission's shot order")
    stats = shot_stats(rows)

    groups = {g: sorted(worst.loc[worst["group"] == g, "shot"]) for g in ("low", "high")}
    groups["rest"] = sorted(set(rows["shot"]) - set(worst["shot"]))
    names = {g: f"D_LCFS {'below' if g == 'low' else 'at or above'} {args.dlcfs_split} m "
                f"(median {worst.loc[worst['group'] == g, 'd_lcfs'].median():.2f} m)" if groups[g] else g for g in ("low", "high")}
    names["rest"] = "Rest of the split"

    # Mean-current maps and the scorer's boundary on sampled frames.
    config = tb.TargetConfig.load(args.target_config)
    folder = tb.target_dir(config.version, args.targets)
    objects = tb.TargetObjects.load(folder)
    if objects.record.get("config_hash") != config.hash():
        raise ValueError(f"the target build in {folder} was made with another configuration than {args.target_config}")
    table = predictors.green_table(args.green_dir, MACHINE)
    scales = adopted_scales(MACHINE, table.columns)
    current = targets.current_columns_of(table, MACHINE)
    ft.assert_input_columns(current, MACHINE)
    currents = ft.read_frame_table(SPLIT, list(dict.fromkeys(["shot", "frame", *current, "ip__mean"])), root=args.table_dir)
    currents["shot"] = currents["shot"].astype(str)
    by_shot = {s: g.sort_values("frame").reset_index(drop=True) for s, g in currents.groupby("shot", sort=True)}
    rng = np.random.default_rng(args.seed)
    rest_sampled = sorted(rng.permutation(groups["rest"])[:args.rest_shots].tolist()) if groups["rest"] else []
    profile = load_script("profile_targets")

    def maps_of(shot: str) -> np.ndarray:
        g = by_shot[shot]
        frames = g.iloc[spaced_frames(len(g), args.frames_per_shot)].reset_index(drop=True)
        return tb.mean_current_maps(MACHINE, frames, table, scales, objects)

    per_shot_boundary, group_maps = {}, {"low": [], "high": [], "rest": []}
    for g, shots in (("low", groups["low"]), ("high", groups["high"]), ("rest", rest_sampled)):
        for s in shots:
            maps = maps_of(s)
            group_maps[g].append(maps)
            if g != "rest":
                per_shot_boundary[s] = profile.derived_on_maps(maps, MACHINE)
        log(f"{g}: mean-current maps of {len(shots)} shots")
    boundary = {g: profile.derived_on_maps(np.concatenate(m), MACHINE) for g, m in group_maps.items() if m}

    info = {"run": args.run, "detailed_results": display_path(detailed), "run_json": display_path(run_json), "n_shots": len(order),
            "dlcfs_split": args.dlcfs_split, "table_dir": display_path(args.table_dir), "feature_root": display_path(args.feature_root),
            "feature_version": args.feature_version, "target_dir": display_path(folder), "config_hash": objects.record["config_hash"],
            "submission_config_hash": record.get("predictor_identifiers", {}).get("target_build", {}).get("config_hash", ""),
            "frames_per_shot": args.frames_per_shot, "n_rest_sampled": len(rest_sampled), "seed": args.seed, "group_names": names,
            "runtime_s": time.time() - started}
    text = render(info, worst, stats, per_shot_boundary, groups, group_rows(rows, stats, groups, boundary))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    log(f"wrote {display_path(out)} in {info['runtime_s']:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
