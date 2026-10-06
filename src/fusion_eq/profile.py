"""Column profile and acceptance checks of a built frame table.

The profile reads one column at a time through pyarrow, so a split of 1.5 million
rows never sits in memory whole. The acceptance checks (T1 to T8 in
``docs/FRAME_TABLE.md``) compare the built tables with the published counts, the
raw shot files and the QA results, and are rendered as a Markdown report.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from fusion_eq import dataset, frame_table, io

PERCENTILES = (1, 5, 25, 50, 75, 95, 99)
N_LABEL_CHECK_SHOTS = 50
IP_FIRST_FRAME_MIN = 0.15
# Counts from the full QA run of 2026-10-04 (docs/findings.md F10, F14, F20) used when the
# QA per-shot table is not available.
QA_RATE_COUNTS = {20000: 5518, 2000: 2397}
QA_IP_DIRECTION = {"diii_d_train": {1: 6420, -1: 603, 0: 18}, "diii_d_public_test": {1: 798, -1: 76, 0: 0}}
QA_MAST_NAN_SHOTS = 114
QA_RATE_SPLIT_MS = 0.1


# ------------------------------------------------------------------------- column profile
def profile_column(values: np.ndarray | pd.Series, name: str, split: str, role: str) -> dict:
    """Summary statistics of one column.

    Returns:
        ``split``, ``column``, ``role``, ``dtype``, ``n``, ``n_missing``,
        ``missing_frac``; for numeric and boolean columns also ``n_zero``,
        ``zero_frac``, ``mean``, ``std``, ``min``, ``p01`` to ``p99``, ``max``; for
        strings ``n_unique``.
    """
    s = pd.Series(values)
    out: dict = {"split": split, "column": name, "role": role, "dtype": str(s.dtype), "n": int(len(s))}
    missing = s.isna()
    out["n_missing"] = int(missing.sum())
    out["missing_frac"] = float(missing.mean()) if len(s) else np.nan
    if pd.api.types.is_string_dtype(s.dtype) or s.dtype == object:
        out["n_unique"] = int(s.nunique(dropna=True))
        return out
    v = pd.to_numeric(s[~missing], errors="coerce").to_numpy(dtype=np.float64)
    out["n_zero"] = int((v == 0).sum())
    out["zero_frac"] = float((v == 0).mean()) if v.size else np.nan
    if v.size == 0:
        return out
    out["mean"] = float(v.mean())
    out["std"] = float(v.std())
    out["min"] = float(v.min())
    for p in PERCENTILES:
        out[f"p{p:02d}"] = float(np.percentile(v, p))
    out["max"] = float(v.max())
    return out


def profile_table(path: Path | str, split: str, roles: dict[str, str] | None = None) -> pd.DataFrame:
    """Per-column profile of one split's table, one row per column.

    Args:
        path: Parquet file of the split.
        split: Split name written to the rows.
        roles: Column name to role; unknown columns get ``"unknown"``.
    """
    roles = roles or {}
    pf = pq.ParquetFile(path)
    rows = []
    for name in pf.schema_arrow.names:
        column = pf.read(columns=[name]).column(name).to_pandas()
        rows.append(profile_column(column, name, split, roles.get(name, "unknown")))
    return pd.DataFrame(rows)


def schema_roles(machine: str, with_labels: bool) -> dict[str, str]:
    """Column name to role for one split."""
    return {c.name: c.role for c in frame_table.table_schema(machine, with_labels)}


# ---------------------------------------------------------------------- acceptance checks
@dataclass
class CheckResult:
    """Outcome of one acceptance check."""

    check_id: str
    name: str
    status: str
    headline: str
    details: list[str] = field(default_factory=list)


def _read(path: Path, columns: list[str]) -> pd.DataFrame:
    present = set(pq.ParquetFile(path).schema_arrow.names)
    return pq.read_table(path, columns=[c for c in columns if c in present]).to_pandas()


def check_counts(index: pd.DataFrame, splits: Iterable[str], full_run: bool) -> CheckResult:
    """T1: row and shot counts per split, and no shot with an error."""
    details, ok = [], True
    for split in splits:
        sub = index[index["split"] == split]
        n_shots, n_frames = len(sub), int(sub["n_frames"].fillna(0).sum())
        errors = int(sub["error"].notna().sum())
        expect = (dataset.EXPECTED_SHOTS.get(split), dataset.EXPECTED_FRAMES.get(split))
        match = (not full_run) or (n_shots == expect[0] and n_frames == expect[1])
        ok &= match and errors == 0
        details.append(f"- {split}: {n_shots} shots, {n_frames} frames, {errors} errors"
                       + (f" (expected {expect[0]} / {expect[1]})" if full_run else ""))
    for err in index.loc[index["error"].notna(), ["split", "shot", "error"]].itertuples(index=False):
        details.append(f"  - error {err.split}/{err.shot}: {err.error}")
    status = "pass" if ok else "fail"
    return CheckResult("T1", "Row and shot counts", status,
                       f"{len(index)} shots, {int(index['n_frames'].fillna(0).sum())} frames", details)


def check_keys(tables: dict[str, Path]) -> CheckResult:
    """T2: (shot, frame) unique and time_ms strictly increasing within every shot."""
    ok, details = True, []
    for split, path in tables.items():
        df = _read(path, ["shot", "frame", "time_ms"])
        dup = int(df.duplicated(["shot", "frame"]).sum())
        dt = df.groupby("shot", sort=False)["time_ms"].diff()
        bad_time = int((dt <= 0).sum())
        ok &= dup == 0 and bad_time == 0
        details.append(f"- {split}: {dup} duplicate keys, {bad_time} non-increasing time steps")
    return CheckResult("T2", "Keys", "pass" if ok else "fail", "unique keys, increasing time", details)


def check_labels_untouched(tables: dict[str, Path], data_root: Path, n_shots: int = N_LABEL_CHECK_SHOTS,
                           seed: int = 0) -> CheckResult:
    """T3: copied labels equal the shot file exactly; test tables hold no label column."""
    ok, details = True, []
    for split, path in tables.items():
        names = pq.ParquetFile(path).schema_arrow.names
        labels = [c for c in names if frame_table.is_label_column(c)]
        if split not in dataset.LABELLED_SPLITS:
            ok &= not labels
            details.append(f"- {split}: {len(labels)} label columns" + (f" {labels}" if labels else ""))
            continue
        df = _read(path, ["shot", "frame", *frame_table.COPIED_LABELS])
        shots = sorted(df["shot"].unique())
        rng = np.random.default_rng(seed)
        sample = sorted(rng.choice(shots, size=min(n_shots, len(shots)), replace=False)) if shots else []
        mismatched = []
        for shot_id in sample:
            shot = io.load_shot(dataset.shot_path(data_root, split, shot_id), list(frame_table.COPIED_LABELS),
                                apply=False)
            rows = df[df["shot"] == shot_id].sort_values("frame")
            for name in frame_table.COPIED_LABELS:
                raw = np.asarray(shot[name], dtype=np.float64).ravel()
                got = rows[name].to_numpy(dtype=np.float64)
                same = raw.size == got.size and np.array_equal(raw, got, equal_nan=True)
                if not same:
                    mismatched.append(f"{shot_id}:{name}")
        ok &= not mismatched
        details.append(f"- {split}: {len(sample)} shots compared, {len(mismatched)} mismatches"
                       + (f" {mismatched[:10]}" if mismatched else ""))
    return CheckResult("T3", "Labels untouched", "pass" if ok else "fail", "exact copies", details)


def check_input_completeness(tables: dict[str, Path]) -> CheckResult:
    """T4 (review): share of frames with every current mean finite, per split."""
    details = []
    for split, path in tables.items():
        df = _read(path, ["shot", "inputs_complete"])
        share = float(df["inputs_complete"].mean()) if len(df) else np.nan
        shots_incomplete = int((~df.groupby("shot")["inputs_complete"].all()).sum())
        details.append(f"- {split}: {share:.4f} of frames complete; {shots_incomplete} shots with an incomplete frame")
    return CheckResult("T4", "Input completeness", "review", "see per split", details)


def check_flags_against_qa(index: pd.DataFrame, qa_path: Path | None, full_run: bool) -> CheckResult:
    """T5: recording-rate, Ip direction and MAST NaN counts against the QA run."""
    details, ok = [], True
    d3d = index[(index["machine"] == "DIII-D") & index["error"].isna()]
    mast = index[(index["machine"] == "MAST") & index["error"].isna()]
    rates = d3d["mag_rate_hz"].value_counts().to_dict()
    details.append(f"- DIII-D recording rates: {dict(sorted((int(k), int(v)) for k, v in rates.items()))}")
    mast_rates = mast["mag_rate_hz"].value_counts().to_dict()
    details.append(f"- MAST recording rates: {dict(sorted((int(k), int(v)) for k, v in mast_rates.items()))}")
    directions = {s: d3d[d3d["split"] == s]["ip_direction"].value_counts().to_dict()
                  for s in ("diii_d_train", "diii_d_public_test")}
    details.append(f"- DIII-D Ip direction: {directions}")
    n_nan = int(mast["mag_has_nan"].fillna(False).sum())
    details.append(f"- MAST shots with NaN currents: {n_nan}")

    if qa_path is not None and Path(qa_path).is_file():
        qa = pd.read_parquet(qa_path, columns=["split", "machine", "shot", "time_magnetics_dt_median_ms",
                                               "cur_plasma_current_min", "cur_plasma_current_max",
                                               "cur_plasma_current_nan_frac"])
        qa = qa[qa["split"].isin(set(index["split"]))]
        qa_d3d = qa[qa["machine"] == "DIII-D"]
        qa_rates = {20000: int((qa_d3d["time_magnetics_dt_median_ms"] <= QA_RATE_SPLIT_MS).sum()),
                    2000: int((qa_d3d["time_magnetics_dt_median_ms"] > QA_RATE_SPLIT_MS).sum())}
        qa_dir = {}
        for s in ("diii_d_train", "diii_d_public_test"):
            sub = qa_d3d[qa_d3d["split"] == s]
            sign = np.select([sub["cur_plasma_current_min"] >= 0, sub["cur_plasma_current_max"] <= 0],
                             [1, -1], default=0)
            qa_dir[s] = {int(k): int(v) for k, v in zip(*np.unique(sign, return_counts=True))}
        qa_nan = int((qa[qa["machine"] == "MAST"]["cur_plasma_current_nan_frac"] > 0).sum())
        source = f"QA table {qa_path}"
    else:
        qa_rates, qa_dir, qa_nan = QA_RATE_COUNTS, QA_IP_DIRECTION, QA_MAST_NAN_SHOTS
        source = "constants from the full QA run (findings F10, F14, F20)"
        if not full_run:
            return CheckResult("T5", "Shot-level flags against QA", "review",
                               "partial run, QA table not available", details)
    rate_ok = {int(k): int(v) for k, v in rates.items()} == {k: v for k, v in qa_rates.items() if v}
    dir_ok = all({int(k): int(v) for k, v in directions[s].items()} == {k: v for k, v in qa_dir[s].items() if v}
                 for s in directions if s in set(index["split"]))
    nan_ok = n_nan == qa_nan or "mast_public_test" not in set(index["split"])
    ok = rate_ok and dir_ok and nan_ok
    details.append(f"- reference ({source}): rates {qa_rates}, directions {qa_dir}, MAST NaN shots {qa_nan}")
    return CheckResult("T5", "Shot-level flags against QA", "pass" if ok else "fail",
                       f"rates {'match' if rate_ok else 'differ'}, directions {'match' if dir_ok else 'differ'}, "
                       f"NaN count {'matches' if nan_ok else 'differs'}", details)


def check_ip_first_frame(tables: dict[str, Path]) -> CheckResult:
    """T6: ip_over_peak at the first frame of every DIII-D shot is at least 0.15."""
    details, ok, worst = [], True, np.inf
    for split, path in tables.items():
        if dataset.SPLIT_MACHINE.get(split) != "DIII-D":
            continue
        df = _read(path, ["shot", "frame", "ip_over_peak"])
        first = df[df["frame"] == 0]
        low = first[first["ip_over_peak"] < IP_FIRST_FRAME_MIN]
        worst = min(worst, float(first["ip_over_peak"].min())) if len(first) else worst
        ok &= len(low) == 0
        details.append(f"- {split}: minimum {first['ip_over_peak'].min():.3f}; {len(low)} shots below {IP_FIRST_FRAME_MIN}"
                       + (f": {sorted(low['shot'])[:20]}" if len(low) else ""))
    return CheckResult("T6", "Ip at the first frame", "pass" if ok else "fail", f"minimum {worst:.3f}", details)


def check_thomson_coverage(tables: dict[str, Path], index: pd.DataFrame) -> CheckResult:
    """T7 (review): Thomson coverage per split against the QA figures."""
    details = []
    for split, path in tables.items():
        df = _read(path, ["shot", "ts_core_valid"])
        share = df.groupby("shot")["ts_core_valid"].mean()
        covered = float((share >= 0.9).mean()) if len(share) else np.nan
        none = int((~index[(index["split"] == split) & index["error"].isna()]["ts_core_any_valid"].fillna(False)).sum())
        details.append(f"- {split}: {covered:.4f} of shots with ts_core_valid on at least 90% of frames; "
                       f"{none} shots with no valid core profile")
    return CheckResult("T7", "Thomson coverage against QA", "review",
                       "QA: about 97% covered; 53 / 6 / 1 shots without a profile", details)


def check_resampling_forms(tables: dict[str, Path]) -> CheckResult:
    """T8 (review): nearest minus mean per current column and recording rate, relative to spread."""
    details = []
    for split, path in tables.items():
        names = pq.ParquetFile(path).schema_arrow.names
        signals = sorted({n[: -len("__mean")] for n in names if n.endswith("__mean")})
        cols = ["mag_rate_hz"] + [f"{s}__nearest" for s in signals] + [f"{s}__mean" for s in signals]
        df = _read(path, cols)
        for rate, sub in df.groupby("mag_rate_hz"):
            parts = []
            for s in signals:
                diff = (sub[f"{s}__nearest"] - sub[f"{s}__mean"]).to_numpy(dtype=np.float64)
                spread = float(np.nanstd(sub[f"{s}__mean"].to_numpy(dtype=np.float64)))
                if not np.isfinite(spread) or spread == 0 or not np.isfinite(diff).any():
                    continue
                rel = np.abs(diff[np.isfinite(diff)]) / spread
                parts.append(f"{s} p50 {np.median(rel):.4f} p99 {np.percentile(rel, 99):.4f}")
            details.append(f"- {split} at {int(rate)} Hz ({len(sub)} frames): " + "; ".join(parts))
    return CheckResult("T8", "Resampling forms agree", "review", "|nearest - mean| / std(mean)", details)


def run_checks(out_dir: Path | str, data_root: Path | str, qa_path: Path | str | None) -> list[CheckResult]:
    """Runs T1 to T8 on a build directory."""
    out_dir = Path(out_dir)
    index = pd.read_parquet(out_dir / "shot_index.parquet")
    splits = [s for s in dataset.DATA_SPLITS if frame_table.table_path(s, out_dir).is_file()]
    tables = {s: frame_table.table_path(s, out_dir) for s in splits}
    full_run = all(len(index[index["split"] == s]) == dataset.EXPECTED_SHOTS[s] for s in dataset.DATA_SPLITS)
    return [
        check_counts(index, splits, full_run),
        check_keys(tables),
        check_labels_untouched(tables, Path(data_root)),
        check_input_completeness(tables),
        check_flags_against_qa(index, Path(qa_path) if qa_path else None, full_run),
        check_ip_first_frame(tables),
        check_thomson_coverage(tables, index),
        check_resampling_forms(tables),
    ]


def render_report(results: list[CheckResult], profile: pd.DataFrame, build_info: dict) -> str:
    """Markdown report: acceptance table, details, then per-column missing counts."""
    lines = ["# Frame table report", "",
             f"Build at {build_info.get('timestamp')} on commit {build_info.get('repo_commit')} "
             f"(starter kit {build_info.get('starter_commit')}), runtime {build_info.get('runtime_s', 0):.0f} s.",
             f"Arguments: `{build_info.get('args')}`.", "",
             "| Id | Check | Status | Headline |", "| --- | --- | --- | --- |"]
    for r in results:
        lines.append(f"| {r.check_id} | {r.name} | {r.status} | {r.headline} |")
    for r in results:
        lines += ["", f"## {r.check_id} {r.name} ({r.status})", ""] + r.details
    lines += ["", "## Missing values per column", ""]
    for split, sub in profile.groupby("split", sort=False):
        lines += [f"### {split}", "", "| Column | Role | Missing | Fraction | Zeros |", "| --- | --- | --- | --- | --- |"]
        for row in sub.itertuples(index=False):
            zeros = "" if pd.isna(getattr(row, "n_zero", np.nan)) else f"{int(row.n_zero)}"
            lines.append(f"| {row.column} | {row.role} | {row.n_missing} | {row.missing_frac:.4f} | {zeros} |")
        lines.append("")
    return "\n".join(lines) + "\n"
