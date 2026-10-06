"""Dataset-level aggregation of the per-shot QA table and the Markdown report.

Each ``summarize_*`` function turns the per-shot table into one ``CheckResult``
with a status, a headline number, detail lines and the flagged shots. The status
rules follow ``docs/QA.md``. Checks 0.1, 0.12, 0.14, 0.11b and 0.13 exist only at
this level.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

EXPECTED_COUNTS = {"diii_d_train": 7041, "diii_d_public_test": 874, "mast_public_test": 1206,
                   "mast_samples": 3}
EXPECTED_TEST_FRAMES = {"diii_d_public_test": 193_359, "mast_public_test": 69_075}
EXPECTED_LABELS = {
    "efit_psirz", "efit_q95", "efit_beta_n", "efit_li", "efit_r_axis", "efit_z_axis",
    "efit_lcfs_n", "efit_lcfs_r", "efit_lcfs_z", "magnetics_dsep", "magnetics_dsep_times",
}
EXPECTED_LCFS_PADDING = {"DIII-D": "zero", "MAST": "nan"}
# Check 0.2 review rules: disagreement between the rule A and rule B shifts, and the
# breakdown window expected of shots the organizers' fix corrected.
A_MINUS_B_REVIEW_MS = 10.0
FIXED_BREAKDOWN_WINDOW_MS = (0.0, 60.0)
FLAG_CAP = 20
MACHINES = ("DIII-D", "MAST")


@dataclass
class CheckResult:
    """Outcome of one check over the whole run."""

    check_id: str
    name: str
    status: str
    headline: str
    note: str = ""
    details: list[str] = field(default_factory=list)
    flagged: list[str] = field(default_factory=list)
    reviewed: list[str] = field(default_factory=list)


@dataclass
class RunContext:
    """What the aggregation needs to know about the run."""

    full_run: bool
    splits: tuple[str, ...]


# ----------------------------------------------------------------------------- helpers
def _col(df: pd.DataFrame, name: str) -> pd.Series:
    """The named column, or an all-NaN column when no shot produced it."""
    if name in df:
        return df[name]
    return pd.Series([np.nan] * len(df), index=df.index, dtype="float64")


def _true(series: pd.Series) -> pd.Series:
    return series.fillna(False).astype(bool)


def _false(series: pd.Series) -> pd.Series:
    """Rows where a boolean metric is present and False."""
    return series.notna() & ~_true(series)


def _fmt(x, spec: str = ".4g") -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)) or pd.isna(x):
        return "nan"
    return format(float(x), spec)


def _range(series: pd.Series, spec: str = ".4g") -> str:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return "no data"
    return f"min {_fmt(s.min(), spec)}, median {_fmt(s.median(), spec)}, max {_fmt(s.max(), spec)}"


def _flags(rows: pd.DataFrame, reason: str) -> list[str]:
    return [f"{r.split}/{r.shot}: {reason}" for r in rows.itertuples()]


def _table(header: list[str], rows: list[list]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return out


def _by_machine(df: pd.DataFrame):
    for machine in MACHINES:
        sub = df[df["machine"] == machine]
        if not sub.empty:
            yield machine, sub


def _value_counts(series: pd.Series, limit: int = 8) -> str:
    vc = series.dropna().value_counts()
    items = [f"{k}: {v}" for k, v in vc.head(limit).items()]
    if len(vc) > limit:
        items.append(f"... {len(vc) - limit} more")
    return ", ".join(items) if items else "no data"


def _column_sets(df: pd.DataFrame) -> dict[str, set[str]]:
    """Column set per split, from the most common ``col_names`` value."""
    out = {}
    for split, sub in df.groupby("split"):
        names = _col(sub, "col_names").dropna()
        if not names.empty:
            out[split] = set(names.mode().iloc[0].split("|"))
    return out


# ----------------------------------------------------------------------------- 0.1
def summarize_inventory(df: pd.DataFrame, ctx: RunContext) -> CheckResult:
    rows, flagged = [], []
    all_ok = True
    counts_ok = True
    for split, sub in df.groupby("split"):
        opens = _true(_col(sub, "inv_opens"))
        one_row = _col(sub, "inv_n_rows") == 1
        size_gb = pd.to_numeric(_col(sub, "inv_file_size_bytes"), errors="coerce").sum() / 1e9
        rows.append([split, len(sub), int(opens.sum()), int(one_row.sum()), f"{size_gb:.2f}",
                     EXPECTED_COUNTS.get(split, "")])
        bad = sub[~opens | ~one_row]
        flagged += _flags(bad, "does not open or row count is not 1")
        all_ok &= bad.empty
        if split in EXPECTED_COUNTS and len(sub) != EXPECTED_COUNTS[split]:
            counts_ok = False
    if not all_ok:
        status, note = "fail", "some files do not open or hold more than one row"
    elif ctx.full_run and counts_ok:
        status, note = "pass", ""
    elif ctx.full_run:
        status, note = "fail", "file counts differ from the expected 7041 / 874 / 1206"
    else:
        status, note = "review", "sampled run; file counts are not comparable to the expected totals"
    headline = ", ".join(f"{r[0]}: {r[1]}" for r in rows)
    details = _table(["split", "files", "open", "one row", "size GB", "expected"], rows)
    return CheckResult("0.1", "File inventory", status, headline, note, details, flagged)


# ----------------------------------------------------------------------------- 0.12
def summarize_split_integrity(df: pd.DataFrame, ctx: RunContext) -> CheckResult:
    # A shot id identifies a shot within one machine only, so the key carries both.
    ids = {s: set(zip(sub["machine"].fillna(""), sub["shot"].dropna())) for s, sub in df.groupby("split")}
    splits = sorted(ids)
    details, flagged = [], []
    for i, a in enumerate(splits):
        for b in splits[i + 1:]:
            overlap = sorted(ids[a] & ids[b])
            details.append(f"- {a} and {b}: {len(overlap)} shared shot ids")
            flagged += [f"{a}+{b}/{m} {s}: shot id in both splits" for m, s in overlap]
    status = "pass" if not flagged else "fail"
    headline = f"{len(flagged)} shot ids shared between splits"
    details.insert(0, "Shot ids per split: " + ", ".join(f"{s}: {len(ids[s])}" for s in splits))
    return CheckResult("0.12", "Split integrity", status, headline, "", details, flagged)


# ----------------------------------------------------------------------------- 0.14
def summarize_column_sets(df: pd.DataFrame, ctx: RunContext) -> CheckResult:
    details, flagged = [], []
    for split, sub in df.groupby("split"):
        hashes = _col(sub, "col_hash").dropna()
        n = hashes.nunique()
        details.append(f"- {split}: {n} distinct column set(s), "
                       f"{int(_col(sub, 'col_n').dropna().median()) if _col(sub, 'col_n').notna().any() else 0} columns")
        if n > 1:
            common = hashes.mode().iloc[0]
            flagged += _flags(sub[hashes != common], "column set differs from the split's most common one")
    sets = _column_sets(df)
    headline = "column sets recorded"
    train, d3d_test = sets.get("diii_d_train"), sets.get("diii_d_public_test")
    if train is not None and d3d_test is not None:
        labels = train - d3d_test
        details.append(f"- DIII-D label set (train minus test): {', '.join(sorted(labels))}")
        extra, missing = labels - EXPECTED_LABELS, EXPECTED_LABELS - labels
        if not extra and not missing:
            headline = "DIII-D label set matches the expected 11 columns"
        else:
            headline = "DIII-D label set differs from the expected one"
            details.append(f"- Unexpected labels: {sorted(extra) or 'none'}; expected but absent: {sorted(missing) or 'none'}")
        only_test = d3d_test - train
        details.append(f"- Columns in DIII-D test but not train: {sorted(only_test) or 'none'}")
    mast_test, mast_samples = sets.get("mast_public_test"), sets.get("mast_samples")
    if mast_test is not None and mast_samples is not None:
        details.append(f"- MAST label set (samples minus test): {', '.join(sorted(mast_samples - mast_test))}")
        details.append(f"- Columns in MAST test but not in the samples: {sorted(mast_test - mast_samples) or 'none'}")
    if d3d_test is not None and mast_test is not None:
        details.append(f"- Inputs only on DIII-D: {sorted(d3d_test - mast_test)}")
        details.append(f"- Inputs only on MAST: {sorted(mast_test - d3d_test)}")
    for split, cols in sets.items():
        details.append(f"- {split} columns: {', '.join(sorted(cols))}")
    return CheckResult("0.14", "Column sets", "review", headline, "", details, flagged)


# ----------------------------------------------------------------------------- 0.2
def summarize_ip_fix(df: pd.DataFrame, ctx: RunContext) -> CheckResult:
    d = df[(df["machine"] == "DIII-D") & _col(df, "ip_fix_applied").notna()]
    if d.empty:
        return CheckResult("0.2", "Ip time axis", "review", "no DIII-D shots with Ip loaded")
    applied = _true(d["ip_fix_applied"])
    aligned = _true(_col(d, "ip_alignment_applied"))
    fails = d[_false(_col(d, "ip_covers_efit")) | _true(_col(d, "ip_near_zero"))
              | _true(_col(d, "ip_first_frame_before_breakdown"))]
    details = []
    for split, sub in d.groupby("split", observed=True):
        details.append(f"- {split}: organizers' fix corrected {_true(sub['ip_fix_applied']).mean():.1%} of {len(sub)} shots; "
                       f"rule B alignment applied to {_true(_col(sub, 'ip_alignment_applied')).mean():.1%}")
    shifts = pd.to_numeric(d.loc[applied, "ip_shift_ms"], errors="coerce").round(2)
    details.append(f"- Organizers' shift on corrected shots (ms): {_value_counts(shifts)}")
    details.append(f"- Rule B shift on aligned shots (ms): {_range(_col(d.loc[aligned], 'ip_alignment_shift_ms'))}; "
                   f"positive on {(pd.to_numeric(_col(d.loc[aligned], 'ip_alignment_shift_ms'), errors='coerce') > 0).mean():.1%}")
    details.append(f"- Breakdown after the organizers' fix (ms), corrected shots: "
                   f"{_range(_col(d.loc[applied], 'ip_breakdown_fixed_ms'))}; untouched shots: "
                   f"{_range(_col(d.loc[~applied], 'ip_breakdown_fixed_ms'))}")
    details.append(f"- Breakdown on the final axis (ms): {_range(_col(d, 'ip_breakdown_aligned_ms'))}")
    details.append(f"- Breakdown on the shipped axis (ms), corrected shots: "
                   f"{_range(_col(d.loc[applied], 'ip_breakdown_shipped_ms'))}")
    details.append(f"- Median |Ip| at EFIT frames, final axis (kA): {_range(_col(d, 'ip_median_abs_efit_ka'))}; "
                   f"shipped axis: {_range(_col(d, 'ip_median_abs_efit_shipped_ka'))}")
    details.append(f"- Ratio median |Ip| over p99.5 |Ip|: {_range(_col(d, 'ip_median_over_p995'))}")
    details.append(f"- |Ip| at the first EFIT frame over p99.5 |Ip|: {_range(_col(d, 'ip_first_over_p995'), '.3f')}")
    details.append(f"- First EFIT frame minus breakdown (ms): {_range(_col(d, 'ip_breakdown_to_first_frame_ms'))}; "
                   f"negative on {_true(_col(d, 'ip_first_frame_before_breakdown')).sum()} shots")
    details.append(f"- 99.5th percentile |Ip| (kA): {_range(_col(d, 'ip_p995_abs_ka'))}")
    details.append(f"- Final axis covers the EFIT window: {_true(_col(d, 'ip_covers_efit')).mean():.1%}")
    anchor = pd.to_numeric(_col(d, "ip_sol_anchor_ms"), errors="coerce")
    a_minus_b = pd.to_numeric(_col(d, "ip_a_minus_b_ms"), errors="coerce")
    if anchor.notna().any():
        details.append(f"- Rule A diagnostic: solenoid swing-down anchor (ms) {_range(anchor)}; lag to fixed breakdown "
                       f"{_range(_col(d, 'ip_lag_a_ms'))}; A minus B shift (ms) {_range(a_minus_b)}; "
                       f"|A minus B| above {A_MINUS_B_REVIEW_MS:g} ms on {(a_minus_b.abs() > A_MINUS_B_REVIEW_MS).sum()} shots; "
                       f"no anchor on {int(anchor.isna().sum())} shots")
    for key in ("ip_fix_error", "ip_alignment_error"):
        errors = _col(d, key).dropna()
        if not errors.empty:
            details.append(f"- {key} on {len(errors)} shots: {_value_counts(errors, 3)}")
    m = df[(df["machine"] == "MAST") & _col(df, "ip_median_abs_efit_ka").notna()]
    if not m.empty:
        details.append(f"- MAST ({len(m)} shots): Ip covers EFIT window {_true(m['ip_covers_efit']).mean():.1%}; "
                       f"median |Ip| at EFIT frames (kA): {_range(m['ip_median_abs_efit_ka'])}; "
                       f"first-frame ratio {_range(_col(m, 'ip_first_over_p995'), '.3f')}; "
                       f"Ip NaN count: {_range(_col(m, 'ip_n_nan'), '.0f')}")

    flagged = _flags(fails, "Ip misses the EFIT window, reads near zero, or the first EFIT frame precedes breakdown")
    lo, hi = FIXED_BREAKDOWN_WINDOW_MS
    fixed_breakdown = pd.to_numeric(_col(d, "ip_breakdown_fixed_ms"), errors="coerce")
    reviewed = _flags(d[anchor.notna() & (a_minus_b.abs() > A_MINUS_B_REVIEW_MS)],
                      f"rule A and rule B shifts differ by more than {A_MINUS_B_REVIEW_MS:g} ms")
    if "ip_sol_anchor_ms" in d.columns:
        reviewed += _flags(d[anchor.isna()], "no solenoid swing-down anchor found")
    reviewed += _flags(d[applied & ((fixed_breakdown < lo) | (fixed_breakdown > hi))],
                       f"organizer-fixed shot with breakdown outside {lo:g} to {hi:g} ms")
    status = "fail" if flagged else "pass"
    headline = (f"{applied.mean():.1%} of {len(d)} DIII-D shots corrected by the organizers' fix, "
                f"{aligned.mean():.1%} aligned by rule B; {len(fails)} shot(s) fail; {len(reviewed)} for review")
    return CheckResult("0.2", "Ip time axis", status, headline, "", details, flagged, reviewed)


# ----------------------------------------------------------------------------- 0.3
def summarize_thomson_alignment(df: pd.DataFrame, ctx: RunContext) -> CheckResult:
    m = df[(df["machine"] == "MAST") & _col(df, "ts_channels_before").notna()]
    if m.empty:
        return CheckResult("0.3", "MAST Thomson alignment", "review", "no MAST shots with Thomson loaded")
    layout = (m["ts_channels_before"].astype("Int64").astype(str) + " -> "
              + m["ts_channels_after"].astype("Int64").astype(str) + " channels, "
              + m["ts_n_positions"].astype("Int64").astype(str) + " positions")
    details = [
        f"- Fix changed the shot: {_true(m['ts_fix_applied']).mean():.1%} of {len(m)} shots",
        f"- Layouts (before -> after, positions): {_value_counts(layout)}",
        f"- Aligned after fix: {_true(m['ts_aligned']).mean():.1%}",
        f"- NaN fraction of the dropped channel, Te: {_range(_col(m, 'ts_dropped_nan_frac_te'))}; "
        f"ne: {_range(_col(m, 'ts_dropped_nan_frac_ne'))}",
    ]
    errors = _col(m, "ts_fix_error").dropna()
    flagged = _flags(m[_false(m["ts_aligned"])], "channels and positions disagree after the fix")
    flagged += _flags(m[_col(m, "ts_fix_error").notna()], "fix raised")
    if not errors.empty:
        details.append(f"- Fix raised on {len(errors)} shots: {_value_counts(errors, 3)}")
    d3d = df[(df["machine"] == "DIII-D") & _col(df, "ts_channels_before").notna()]
    if not d3d.empty:
        details.append(f"- DIII-D pass-through: fix changed {_true(d3d['ts_fix_applied']).sum()} of {len(d3d)} shots; "
                       f"core channels {_value_counts(d3d['ts_channels_after'].astype('Int64'))}")
    headline = f"fix applied on {_true(m['ts_fix_applied']).mean():.1%} of {len(m)} MAST shots"
    return CheckResult("0.3", "MAST Thomson alignment", "review", headline, "", details, flagged)


# ----------------------------------------------------------------------------- 0.4
def summarize_current_units(df: pd.DataFrame, ctx: RunContext) -> CheckResult:
    details, head = [], []
    for machine, sub in _by_machine(df):
        names = sorted(c[len("cur_"):-len("_min")] for c in sub.columns
                       if c.startswith("cur_") and c.endswith("_min") and sub[c].notna().any())
        if not names:
            continue
        rows = []
        for name in names:
            mn = pd.to_numeric(sub[f"cur_{name}_min"], errors="coerce")
            mx = pd.to_numeric(sub[f"cur_{name}_max"], errors="coerce")
            p99 = pd.to_numeric(sub[f"cur_{name}_p99abs"], errors="coerce")
            nan = pd.to_numeric(_col(sub, f"cur_{name}_nan_frac"), errors="coerce")
            rows.append([name, int(mn.notna().sum()), _fmt(mn.min()), _fmt(mx.max()),
                         _fmt(p99.median()), _fmt(p99.max()), _fmt(nan.max(), ".3f")])
            if name == "plasma_current":
                head.append(f"{machine} Ip p99 |value| median {_fmt(p99.median())}")
        details.append(f"**{machine}** (values as stored, inside the EFIT window)")
        details.append("")
        details += _table(["column", "shots", "min of min", "max of max", "median p99 abs", "max p99 abs",
                           "max NaN frac"], rows)
        details.append("")
    return CheckResult("0.4", "Current units", "review", "; ".join(head) or "no data", "", details)


# ----------------------------------------------------------------------------- 0.5
def summarize_flux_map(df: pd.DataFrame, ctx: RunContext) -> CheckResult:
    d = df[_col(df, "psi_shape_ok").notna()]
    if d.empty:
        return CheckResult("0.5", "Flux map indexing", "review", "no shots with flux maps loaded")
    shape_bad = d[_false(d["psi_shape_ok"])]
    within_o = pd.to_numeric(_col(d, "psi_opoint_within_1cell_frac"), errors="coerce")
    axis_bad = d[within_o.notna() & (within_o < 1.0)]
    border = pd.to_numeric(_col(d, "psi_axis_on_border_frac"), errors="coerce")
    on_border = d[border.notna() & (border > 0)]
    flagged = _flags(shape_bad, "psi shape is not [n_frames, 65, 65]")
    flagged += [f"{r.split}/{r.shot}: scorer O-point more than one cell from the stored axis on "
                f"{1 - r.psi_opoint_within_1cell_frac:.0%} of sampled frames" for r in axis_bad.itertuples()]
    reviewed = [f"{r.split}/{r.shot}: global extremum on the grid border on {r.psi_axis_on_border_frac:.0%} "
                f"of sampled frames (O-point max distance {r.psi_opoint_cell_dist_max:.2f} cells)"
                for r in on_border.itertuples()]
    details = [f"- Shapes: {_value_counts(d['psi_shape'].str.replace(r'^\(\d+, ', '(T, ', regex=True))}"]
    for machine, sub in _by_machine(d):
        details.append(f"- {machine} ({len(sub)} shots): scorer O-point cell distance, per-shot max: "
                       f"{_range(_col(sub, 'psi_opoint_cell_dist_max'))}; interior local maximum found on "
                       f"{_range(_col(sub, 'psi_opoint_used_local_frac'), '.2f')} of frames; "
                       f"global extremum cell distance, per-shot max: {_range(_col(sub, 'psi_axis_cell_dist_max'))}; "
                       f"global extremum on the border: {int((pd.to_numeric(_col(sub, 'psi_axis_on_border_frac'), errors='coerce') > 0).sum())} shots; "
                       f"transposed reading median: {_range(_col(sub, 'psi_axis_transposed_cell_dist_median'))}; "
                       f"opposite extremum median: {_range(_col(sub, 'psi_axis_opposite_cell_dist_median'))}")
    status = "pass" if not flagged else "fail"
    headline = (f"{len(shape_bad)} shape failures; {len(axis_bad)} shots with the scorer O-point more than one "
                f"cell from the axis; {len(on_border)} shots with a border extremum for review")
    return CheckResult("0.5", "Flux map indexing", status, headline, "", details, flagged, reviewed)


# ----------------------------------------------------------------------------- 0.10
def summarize_grid(df: pd.DataFrame, ctx: RunContext) -> CheckResult:
    d = df[_col(df, "grid_hash").notna()]
    if d.empty:
        return CheckResult("0.10", "Grid", "review", "no shots with grids loaded")
    details, flagged, ok = [], [], True
    for machine, sub in _by_machine(d):
        hashes = sub["grid_hash"].value_counts()
        details.append(f"- {machine}: {len(hashes)} grid configuration(s) over {len(sub)} shots; "
                       f"R {_fmt(pd.to_numeric(sub['grid_R_min']).min())} to {_fmt(pd.to_numeric(sub['grid_R_max']).max())} m "
                       f"step {_fmt(pd.to_numeric(sub['grid_R_step']).median())}; "
                       f"Z {_fmt(pd.to_numeric(sub['grid_Z_min']).min())} to {_fmt(pd.to_numeric(sub['grid_Z_max']).max())} m "
                       f"step {_fmt(pd.to_numeric(sub['grid_Z_step']).median())}; "
                       f"strictly increasing and length 65: {_true(sub['grid_ok']).mean():.1%}")
        if len(hashes) > 1:
            ok = False
            common = hashes.index[0]
            flagged += _flags(sub[sub["grid_hash"] != common], "grid differs from the machine's most common grid")
        bad = sub[_false(sub["grid_ok"])]
        ok &= bad.empty
        flagged += _flags(bad, "grid axis not strictly increasing or not length 65")
    status = "pass" if ok else "fail"
    headline = "one grid per machine" if ok else "grid differences found"
    return CheckResult("0.10", "Grid", status, headline, "", details, flagged)


# ----------------------------------------------------------------------------- 0.9
def summarize_coil_geometry(df: pd.DataFrame, ctx: RunContext) -> CheckResult:
    d = df[_col(df, "coil_hash").notna()]
    if d.empty:
        return CheckResult("0.9", "Coil geometry", "review", "no shots with coil geometry loaded")
    details, flagged, ok = [], [], True
    for machine, sub in _by_machine(d):
        hashes = sub["coil_hash"].value_counts()
        details.append(f"- {machine}: {len(hashes)} geometry configuration(s) over {len(sub)} shots; "
                       f"elements {_value_counts(sub['coil_n_elements'].astype('Int64'))}; "
                       f"input columns mapped to a current column: {_true(sub['coil_mapping_ok']).mean():.1%}; "
                       f"solenoid rows present: {_true(sub['coil_solenoid_present']).mean():.1%} "
                       f"(elements {_value_counts(sub['coil_solenoid_n_elements'].astype('Int64'))})")
        details.append(f"  - elements per column: {_value_counts(sub['coil_elements_per_column'], 3)}")
        details.append(f"  - R {_fmt(pd.to_numeric(sub['coil_R_min']).min())} to {_fmt(pd.to_numeric(sub['coil_R_max']).max())} m, "
                       f"Z {_fmt(pd.to_numeric(sub['coil_Z_min']).min())} to {_fmt(pd.to_numeric(sub['coil_Z_max']).max())} m; "
                       f"all angles zero: {_true(_col(sub, 'coil_angles_all_zero')).mean():.1%}")
        if len(hashes) > 1:
            ok = False
            for h, n in hashes.items():
                shots = sub[sub["coil_hash"] == h]
                details.append(f"  - configuration {h[:12]}: {n} shots, for example "
                               + ", ".join(f"{r.split}/{r.shot}" for r in shots.head(FLAG_CAP).itertuples()))
            flagged += _flags(sub[sub["coil_hash"] != hashes.index[0]],
                              "coil geometry differs from the machine's most common configuration")
        bad = sub[_false(sub["coil_mapping_ok"]) | _false(sub["coil_solenoid_present"])]
        ok &= bad.empty
        flagged += _flags(bad, "unmapped coil_input_column or no solenoid rows")
    status = "pass" if ok else "fail"
    return CheckResult("0.9", "Coil geometry", status,
                       "one geometry per machine" if ok else "geometry differences found", "", details, flagged)


# ----------------------------------------------------------------------------- 0.6
def summarize_boundary(df: pd.DataFrame, ctx: RunContext) -> CheckResult:
    d = df[_col(df, "lcfs_pad_kind").notna()]
    if d.empty:
        return CheckResult("0.6", "Boundary contour", "review", "no shots with contours loaded")
    details, flagged = [], []
    for machine, sub in _by_machine(d):
        details.append(f"- {machine} ({len(sub)} shots): efit_lcfs_n {_range(sub['lcfs_n_min'], '.0f')} (per-shot min) to "
                       f"{_range(sub['lcfs_n_max'], '.0f')} (per-shot max); padded width {_value_counts(sub['lcfs_width'].astype('Int64'), 5)}; "
                       f"padding {_value_counts(sub['lcfs_pad_kind'])}; "
                       f"NaN inside valid points: max {_fmt(pd.to_numeric(sub['lcfs_valid_nan_frac']).max())}; "
                       f"closed contours: {_range(sub['lcfs_closed_frac'], '.3f')}; "
                       f"R {_fmt(pd.to_numeric(sub['lcfs_r_min']).min())} to {_fmt(pd.to_numeric(sub['lcfs_r_max']).max())} m, "
                       f"Z {_fmt(pd.to_numeric(sub['lcfs_z_min']).min())} to {_fmt(pd.to_numeric(sub['lcfs_z_max']).max())} m")
        expected = EXPECTED_LCFS_PADDING.get(machine)
        bad_pad = sub[~sub["lcfs_pad_kind"].isin([expected, "none"])]
        flagged += [f"{r.split}/{r.shot}: padding {r.lcfs_pad_kind}, expected {expected}" for r in bad_pad.itertuples()]
        flagged += _flags(sub[pd.to_numeric(sub["lcfs_valid_nan_frac"]) > 0], "NaN inside the valid contour points")
        flagged += _flags(sub[_false(sub["lcfs_n_integer"]) | _false(sub["lcfs_rows_match_frames"])],
                          "non-integer efit_lcfs_n or row count differs from frames")
    return CheckResult("0.6", "Boundary contour", "review",
                       f"{len(flagged)} shots flagged", "", details, flagged)


# ----------------------------------------------------------------------------- 0.7
def summarize_thomson_missing(df: pd.DataFrame, ctx: RunContext) -> CheckResult:
    details, flagged, head = [], [], []
    for machine, sub in _by_machine(df):
        for system in ("core", "edge"):
            p = f"tsm_{system}_"
            s = sub[_col(sub, p + "n_channels").notna()]
            if s.empty:
                continue
            rows = [
                ["channel count", _value_counts(s[p + "n_channels"].astype("Int64"))],
                ["Thomson samples", _range(s[p + "n_times"], ".0f")],
                ["samples inside EFIT window", _range(_col(s, p + "n_times_window"), ".0f")],
                ["median sampling interval (ms)", _range(_col(s, p + "dt_median_ms"))],
                ["(time, channel) orientation", f"{_true(s[p + 'shape_matches_times']).mean():.1%} of shots"],
                ["mean missing fraction over channels", _range(_col(s, p + "missing_frac_mean"), ".3f")],
                ["max missing fraction over channels", _range(_col(s, p + "missing_frac_max"), ".3f")],
                ["slices with no valid channel", _range(_col(s, p + "slice_all_missing_frac"), ".3f")],
                ["slices that are valid profiles", _range(_col(s, p + "slice_valid_profile_frac"), ".3f")],
                ["EFIT frames with a valid profile within 10 ms", _range(_col(s, p + "frames_valid_within_10ms"), ".3f")],
                ["EFIT frames with a valid profile within 20 ms", _range(_col(s, p + "frames_valid_within_20ms"), ".3f")],
                ["EFIT frames with any valid channel within 10 ms", _range(_col(s, p + "frames_any_within_10ms"), ".3f")],
                ["EFIT frames with any valid channel within 20 ms", _range(_col(s, p + "frames_any_within_20ms"), ".3f")],
            ]
            details.append(f"**{machine} {system}** ({len(s)} shots)")
            details.append("")
            details += _table(["metric", "over shots"], rows)
            details.append("")
            flagged += _flags(s[_false(s[p + "shape_matches_times"])], f"{system} Te/ne rows differ from the time axis")
            if system == "core":
                v = pd.to_numeric(_col(s, p + "frames_valid_within_20ms"), errors="coerce")
                head.append(f"{machine} core profile within 20 ms: median {_fmt(v.median(), '.2f')} of frames")
    return CheckResult("0.7", "Missing Thomson", "review", "; ".join(head) or "no data", "", details, flagged)


# ----------------------------------------------------------------------------- 0.8
def summarize_dsep(df: pd.DataFrame, ctx: RunContext) -> CheckResult:
    d = df[_col(df, "dsep_n").notna()]
    if d.empty:
        return CheckResult("0.8", "magnetics_dsep", "review", "no shots with magnetics_dsep loaded")
    details, head = [], []
    for machine, sub in _by_machine(d):
        sentinel = pd.to_numeric(sub["dsep_n_sentinel"])
        details.append(f"- {machine} ({len(sub)} shots): values {_fmt(pd.to_numeric(sub['dsep_min']).min())} to "
                       f"{_fmt(pd.to_numeric(sub['dsep_max']).max())} m excluding the sentinel; per-shot median "
                       f"{_range(sub['dsep_median'])}; negative fraction {_range(sub['dsep_frac_negative'], '.3f')}; "
                       f"-1 sentinel: {int(sentinel.sum())} values on {int((sentinel > 0).sum())} shots; "
                       f"NaN: {int(pd.to_numeric(sub['dsep_n_nan']).sum())}; "
                       f"dsep_times equals efit_times: {_true(_col(sub, 'dsep_times_equal_efit')).mean():.1%}")
        head.append(f"{machine}: {int(sentinel.sum())} sentinel values")
    return CheckResult("0.8", "magnetics_dsep", "review", "; ".join(head), "", details)


# ----------------------------------------------------------------------------- 0.11
def summarize_label_sanity(df: pd.DataFrame, ctx: RunContext) -> CheckResult:
    d = df[_col(df, "time_n_frames").notna()]
    if d.empty:
        return CheckResult("0.11", "Labels and time bases", "review", "no shots with efit_times loaded")
    n_frames = pd.to_numeric(d["time_n_frames"])
    fails = d[(n_frames == 0) | _false(d["time_efit_increasing"]) | _false(_col(d, "lab_all_finite"))]
    flagged = _flags(fails, "zero frames, non-increasing efit_times or non-finite scored labels")
    rows = []
    for split, sub in d.groupby("split"):
        n = pd.to_numeric(sub["time_n_frames"])
        rows.append([split, len(sub), int(n.min()), _fmt(n.median(), ".0f"), int(n.max()), int(n.sum()),
                     int((n < 10).sum()), int((n == 0).sum())])
    details = ["Frames per shot"]
    details += _table(["split", "shots", "min", "median", "max", "total frames", "shots < 10 frames", "shots with 0 frames"], rows)
    details.append("")
    for machine, sub in _by_machine(d):
        details.append(f"- {machine}: efit_times step median {_range(_col(sub, 'time_efit_dt_median_ms'))} ms; "
                       f"largest gap per shot {_range(_col(sub, 'time_efit_dt_max_ms'))} ms; "
                       f"window {_fmt(pd.to_numeric(_col(sub, 'time_efit_min_ms')).min())} to "
                       f"{_fmt(pd.to_numeric(_col(sub, 'time_efit_max_ms')).max())} ms")
        bases = []
        for short in ("magnetics", "ip", "ts_core", "ts_edge"):
            dt = pd.to_numeric(_col(sub, f"time_{short}_dt_median_ms"), errors="coerce")
            n = pd.to_numeric(_col(sub, f"time_{short}_n"), errors="coerce")
            if dt.notna().any():
                bases.append(f"{short}: dt {_value_counts(dt.round(3), 4)}, length {_value_counts(n.astype('Int64'), 4)}")
        if bases:
            details.append(f"  - time bases: " + "; ".join(bases))
        if _col(sub, "lab_q95_min").notna().any():
            details.append(f"  - q95 {_fmt(pd.to_numeric(sub['lab_q95_min']).min())} to {_fmt(pd.to_numeric(sub['lab_q95_max']).max())}; "
                           f"betaN {_fmt(pd.to_numeric(sub['lab_betan_min']).min())} to {_fmt(pd.to_numeric(sub['lab_betan_max']).max())}; "
                           f"psi {_fmt(pd.to_numeric(sub['lab_psi_min']).min())} to {_fmt(pd.to_numeric(sub['lab_psi_max']).max())}; "
                           f"li {_fmt(pd.to_numeric(_col(sub, 'lab_li_min')).min())} to {_fmt(pd.to_numeric(_col(sub, 'lab_li_max')).max())}; "
                           f"non-finite psi values {int(pd.to_numeric(sub['lab_psi_nan']).sum() + pd.to_numeric(sub['lab_psi_inf']).sum())}; "
                           f"non-finite q95 {int(pd.to_numeric(sub['lab_q95_nan']).sum() + pd.to_numeric(sub['lab_q95_inf']).sum())}; "
                           f"non-finite betaN {int(pd.to_numeric(sub['lab_betan_nan']).sum() + pd.to_numeric(sub['lab_betan_inf']).sum())}")
            lens = [c for c in sub.columns if c.startswith("lab_") and c.endswith("_len_ok")]
            bad_len = sub[np.logical_or.reduce([_false(sub[c]).to_numpy() for c in lens])] if lens else sub.iloc[0:0]
            flagged += _flags(bad_len, "label length differs from the number of frames")
    status = "fail" if flagged else "pass"
    headline = f"{len(fails)} shot(s) fail; {int((n_frames < 10).sum())} shots with fewer than 10 frames"
    return CheckResult("0.11", "Labels and time bases", status, headline, "", details, flagged)


# ----------------------------------------------------------------------------- 0.11b
def summarize_test_frames(df: pd.DataFrame, ctx: RunContext) -> CheckResult:
    details, ok, seen = [], True, False
    for split, expected in EXPECTED_TEST_FRAMES.items():
        sub = df[df["split"] == split]
        if sub.empty:
            continue
        seen = True
        total = int(pd.to_numeric(_col(sub, "time_n_frames"), errors="coerce").sum())
        details.append(f"- {split}: {total} frames over {len(sub)} shots (expected {expected})")
        ok &= total == expected
    if not seen:
        return CheckResult("0.11b", "Test frame totals", "review", "no test split in this run")
    if not ctx.full_run:
        return CheckResult("0.11b", "Test frame totals", "review", "sampled run",
                           "totals are only comparable on a full run", details)
    return CheckResult("0.11b", "Test frame totals", "pass" if ok else "fail",
                       "totals match" if ok else "totals differ from the expected values", "", details)


# ----------------------------------------------------------------------------- 0.15
def summarize_scorer(df: pd.DataFrame, ctx: RunContext) -> CheckResult:
    d = df[_col(df, "sc_n_frames").notna()]
    if d.empty:
        return CheckResult("0.15", "Scorer comparison", "review", "no shots sampled for the scorer check")
    details, head = [], []
    metrics = [
        ("sc_hausdorff_median", "Hausdorff distance, stored vs extracted contour (m), per-shot median"),
        ("sc_hausdorff_max", "Hausdorff distance (m), per-shot max"),
        ("sc_mean_dist_median", "mean distance of stored points to extracted contour (m), per-shot median"),
        ("sc_raxis_absdiff_median", "|R_axis difference| (m), per-shot median"),
        ("sc_raxis_absdiff_max", "|R_axis difference| (m), per-shot max"),
        ("sc_zaxis_absdiff_median", "|Z_axis difference| (m), per-shot median"),
        ("sc_zaxis_absdiff_max", "|Z_axis difference| (m), per-shot max"),
        ("sc_li_diff_median", "li, scorer minus stored, per-shot median"),
        ("sc_li_ratio_median", "li, scorer over stored, per-shot median"),
        ("sc_li_absdiff_max", "|li difference|, per-shot max"),
        ("sc_kappa_absdiff_median", "|kappa minus efit_elongation|, per-shot median"),
    ]
    for machine, sub in _by_machine(d):
        frames = int(pd.to_numeric(sub["sc_n_frames"]).sum())
        fails = int(pd.to_numeric(sub["sc_lcfs_fail_count"]).sum())
        details.append(f"**{machine}**: {len(sub)} shots, {frames} frames, {fails} frames without a scorer contour; "
                       f"mask grid vs shot grid max difference {_fmt(pd.to_numeric(sub['sc_mask_grid_max_abs_diff_m']).max())} m")
        details.append("")
        rows = [[label, _range(_col(sub, key))] for key, label in metrics if _col(sub, key).notna().any()]
        details += _table(["quantity", "over shots"], rows)
        details.append("")
        head.append(f"{machine}: Hausdorff median {_fmt(pd.to_numeric(sub['sc_hausdorff_median']).median())} m, "
                    f"li ratio median {_fmt(pd.to_numeric(sub['sc_li_ratio_median']).median(), '.3f')}")
    return CheckResult("0.15", "Scorer comparison", "review", "; ".join(head), "", details)


# ----------------------------------------------------------------------------- 0.13
def summarize_readme(df: pd.DataFrame, ctx: RunContext) -> CheckResult:
    """Check 0.13: the nine README contradictions against the full data."""
    d3d = df[df["machine"] == "DIII-D"]
    mast = df[df["machine"] == "MAST"]
    lines: list[str] = []
    updates: list[str] = []

    def p99(sub, name):
        return pd.to_numeric(_col(sub, f"cur_{name}_p99abs"), errors="coerce").median()

    fcoils = [c[len("cur_"):-len("_p99abs")] for c in d3d.columns
              if c.startswith("cur_F") and c.endswith("_p99abs")]
    fmax = max((pd.to_numeric(d3d[f"cur_{f}_p99abs"], errors="coerce").max() for f in fcoils), default=np.nan)
    lines.append(f"- D1 (DIII-D current units): median p99 |Ip| {_fmt(p99(d3d, 'plasma_current'))}, largest F-coil p99 "
                 f"{_fmt(fmax)}, ECOILA {_fmt(p99(d3d, 'ECOILA'))}, bcoil {_fmt(p99(d3d, 'bcoil'))}; MAST Ip "
                 f"{_fmt(p99(mast, 'plasma_current'))}. Magnitudes in the hundreds to about a thousand read as kA "
                 f"and kA-turn, consistent with the dictionary and against the README's 'A' table. Final decision "
                 f"stays with the Green's-function fit in Phase C.")

    g = d3d[_col(d3d, "grid_R_min").notna()]
    if not g.empty:
        lines.append(f"- D2 (DIII-D grid 'normalized 0-1'): efit_grid_R {_fmt(pd.to_numeric(g['grid_R_min']).min())} to "
                     f"{_fmt(pd.to_numeric(g['grid_R_max']).max())} m, efit_grid_Z {_fmt(pd.to_numeric(g['grid_Z_min']).min())} to "
                     f"{_fmt(pd.to_numeric(g['grid_Z_max']).max())} m on {len(g)} shots. Physical metres; dictionary agrees.")

    for machine, sub in _by_machine(df):
        s = sub[_col(sub, "tsm_core_shape_matches_times").notna()]
        if not s.empty:
            frac = _true(s["tsm_core_shape_matches_times"]).mean()
            lines.append(f"- D3 (Thomson orientation), {machine}: core Te rows equal the time axis on {frac:.1%} of "
                         f"{len(s)} shots, so storage is (time, channel). Dictionary agrees.")

    m = mast[_col(mast, "col_dtype_magnetics_time").notna()]
    if not m.empty:
        lines.append(f"- D4 (MAST dtypes): magnetics_time {_value_counts(m['col_dtype_magnetics_time'])}; efit_times "
                     f"{_value_counts(m['col_dtype_efit_times'])}; Ip {_value_counts(_col(m, 'col_dtype_magnetics_plasma_current'))}. "
                     f"README says float64 for the time axes.")
        if (m["col_dtype_magnetics_time"] == "float").all():
            lines[-1] += " Data is float32; dictionary agrees."
        else:
            updates.append("D4: MAST time-axis dtypes vary across the test set; record the distribution.")

    t = d3d[_col(d3d, "tsm_core_n_times").notna()]
    if not t.empty:
        core, edge = pd.to_numeric(t["tsm_core_n_times"]), pd.to_numeric(_col(t, "tsm_edge_n_times"), errors="coerce")
        lines.append(f"- D5 (DIII-D Thomson sample counts): core {_range(core, '.0f')}, edge {_range(edge, '.0f')} "
                     f"over {len(t)} shots. README says about 1300 to 1900 core and 200 to 500 edge.")
        updates.append(f"D5: replace the single-layout statement (3029 core, 1443 edge) with the observed ranges "
                       f"core {_range(core, '.0f')}, edge {_range(edge, '.0f')}.")

    ds = mast[_col(mast, "dsep_n_sentinel").notna()]
    if not ds.empty:
        lines.append(f"- D6 (MAST dsep sentinel): {int(pd.to_numeric(ds['dsep_n_sentinel']).sum())} values of -1 and "
                     f"{int(pd.to_numeric(ds['dsep_n_nan']).sum())} NaN over {len(ds)} labelled MAST shots. Only the three "
                     f"sample shots carry this label, so no new evidence beyond the dictionary.")

    sets = _column_sets(df)
    if "mast_samples" in sets and "diii_d_train" in sets:
        extra = sorted(c for c in sets["mast_samples"] - sets["diii_d_train"] if c.startswith("efit_"))
        lines.append(f"- D7 (extra MAST EFIT columns): {', '.join(extra) or 'none'} in the MAST samples and absent from "
                     f"DIII-D train. Dictionary agrees.")
    if "mast_public_test" in sets and "diii_d_public_test" in sets:
        lines.append(f"- D7 continued, test inputs: MAST-only inputs {sorted(sets['mast_public_test'] - sets['diii_d_public_test'])}; "
                     f"DIII-D-only inputs {sorted(sets['diii_d_public_test'] - sets['mast_public_test'])}.")

    nulls = pd.to_numeric(_col(df, "inv_null_leaves"), errors="coerce")
    if nulls.notna().any():
        with_nulls = df[nulls > 0]
        lines.append(f"- D8 (no Parquet nulls): {int(nulls.sum())} null values over {int(nulls.notna().sum())} shots"
                     + (f"; columns {_value_counts(with_nulls['inv_null_columns'], 5)}" if not with_nulls.empty else "")
                     + ". " + ("README and dictionary agree." if nulls.sum() == 0 else "Nulls exist; update the dictionary."))
        if nulls.sum() > 0:
            updates.append("D8: Parquet nulls exist in the release; record which columns and splits.")

    ch = d3d[_col(d3d, "tsm_core_n_channels").notna()]
    if not ch.empty:
        lines.append(f"- D9 (DIII-D Thomson channel counts): core {_value_counts(ch['tsm_core_n_channels'].astype('Int64'))}; "
                     f"edge {_value_counts(_col(ch, 'tsm_edge_n_channels').astype('Int64'))} over {len(ch)} shots. "
                     f"README says 40, 42, 43, 44 or 54 core and 10 or 14 edge.")
        updates.append("D9: record the observed channel-count distribution and confirm the per-shot layout warning.")

    details = lines + ["", "Proposed data-dictionary updates (not applied in this phase):"] + [f"- {u}" for u in updates]
    n_ids = len({line.split()[1] for line in lines})
    return CheckResult("0.13", "README against data", "review", f"{n_ids} of 9 contradictions assessed", "", details)


SUMMARIZERS = (
    summarize_inventory, summarize_split_integrity, summarize_column_sets, summarize_ip_fix,
    summarize_thomson_alignment, summarize_current_units, summarize_flux_map, summarize_grid,
    summarize_coil_geometry, summarize_boundary, summarize_thomson_missing, summarize_dsep,
    summarize_label_sanity, summarize_test_frames, summarize_scorer, summarize_readme,
)


def summarize(df: pd.DataFrame, ctx: RunContext) -> list[CheckResult]:
    """Runs every aggregation. An aggregation that raises becomes a ``review`` result."""
    results = []
    for fn in SUMMARIZERS:
        try:
            results.append(fn(df, ctx))
        except Exception as exc:  # noqa: BLE001 - the report must still be written
            check_id = fn.__doc__.split(":")[0].split()[-1] if fn.__doc__ and "Check" in fn.__doc__ else fn.__name__
            results.append(CheckResult(check_id, fn.__name__, "review", "aggregation failed",
                                       f"{type(exc).__name__}: {exc}"))
    return results


def render_report(results: list[CheckResult], run_info: dict, df: pd.DataFrame) -> str:
    """Renders the Markdown report: run facts, summary table, one section per check."""
    lines = ["# QA report", ""]
    lines.append(f"Run at {run_info.get('timestamp')} on commit {run_info.get('repo_commit')} "
                 f"(starter kit {run_info.get('starter_commit')}), runtime {run_info.get('runtime_s', 0):.0f} s.")
    lines.append(f"Arguments: `{run_info.get('args')}`.")
    counts = df.groupby("split").size()
    lines.append("Shots per split: " + ", ".join(f"{s}: {n}" for s, n in counts.items()) + ".")
    errors = df["error"].notna().sum() if "error" in df else 0
    lines.append(f"Shots with at least one error: {int(errors)}.")
    lines += ["", "## Summary", ""]
    lines += _table(["Id", "Check", "Status", "Headline", "Note"],
                    [[r.check_id, r.name, r.status, r.headline, r.note] for r in results])
    lines += ["", "## Checks", ""]
    for r in results:
        lines.append(f"### {r.check_id} {r.name} ({r.status})")
        lines.append("")
        lines.append(r.headline + (f". {r.note}" if r.note else ""))
        lines.append("")
        lines += r.details
        if r.flagged:
            lines.append("")
            lines.append(f"Flagged shots ({len(r.flagged)}, showing up to {FLAG_CAP}; the full list is in qa_per_shot.parquet):")
            lines += [f"- {f}" for f in r.flagged[:FLAG_CAP]]
        if r.reviewed:
            lines.append("")
            lines.append(f"Shots for review ({len(r.reviewed)}, showing up to {FLAG_CAP}):")
            lines += [f"- {f}" for f in r.reviewed[:FLAG_CAP]]
        lines.append("")
    if errors:
        lines += ["## Errors", ""]
        err = df[df["error"].notna()]
        lines.append(f"Error messages by type: {_value_counts(err['error'].str.split(':').str[:2].str.join(':'), 10)}")
        lines.append("")
        lines += [f"- {r.split}/{r.shot}: {r.error}" for r in err.head(FLAG_CAP).itertuples()]
        lines.append("")
    return "\n".join(lines)
