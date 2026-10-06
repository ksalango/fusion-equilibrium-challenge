"""Checks 0.2 (DIII-D Ip time-axis erratum) and 0.4 (units of current columns)."""

from __future__ import annotations

import numpy as np

from fusion_eq import ip_alignment
from fusion_eq.io import Shot
from fusion_eq.qa.efit_window import efit_window_ms, window_mask

# Below this ratio of median |Ip| at EFIT frames to the shot's 99.5th percentile of
# |Ip|, the Ip trace read at the EFIT frames is the pre-plasma noise floor.
IP_NEAR_ZERO_RATIO = 0.10

TIME_COLUMN_SUFFIXES = ("_time", "_times")


def check_ip_time_axis(shot: Shot) -> dict:
    """Check 0.2: the plasma-current time axis after the fixes.

    On DIII-D the organizers' fix may re-origin ``magnetics_plasma_current_times`` and
    the loader's rule B alignment may shift it further; both are recorded in
    ``shot.fixes``. On MAST Ip sits on ``magnetics_time`` and only the coverage and
    magnitude metrics apply. Breakdown times use the organizers' definition
    (``fusion_eq.ip_alignment.breakdown_time_ms``).

    Returns:
        Fix record: ``ip_fix_applied``, ``ip_shift_ms`` (organizers' shift),
        ``ip_alignment_applied``, ``ip_alignment_shift_ms`` (rule B shift),
        ``ip_breakdown_shipped_ms``, ``ip_breakdown_fixed_ms`` (after the organizers'
        fix, before alignment), ``ip_breakdown_aligned_ms`` (on the final axis).
        Final axis: ``ip_axis_min_ms``, ``ip_axis_max_ms``, ``ip_covers_efit``,
        ``ip_p995_abs_ka``, ``ip_median_abs_efit_ka`` (median |Ip| at the EFIT frames),
        ``ip_median_abs_efit_shipped_ka`` (DIII-D, shipped axis), ``ip_median_over_p995``
        and ``ip_near_zero`` (below ``IP_NEAR_ZERO_RATIO``), ``ip_first_abs_efit_ka`` and
        ``ip_first_over_p995`` (|Ip| at the first EFIT frame), ``ip_breakdown_to_first_frame_ms``
        (first EFIT time minus final breakdown) and ``ip_first_frame_before_breakdown``
        (that gap is negative; a fail).
        Rule A diagnostic (DIII-D with ``magnetics_ECOILA`` loaded): ``ip_sol_anchor_ms``
        (solenoid swing-down), ``ip_lag_a_ms`` (fixed breakdown minus anchor),
        ``ip_shift_a_ms``, ``ip_shift_b_ms`` (the shift rule B implies for every shot,
        applied or not) and ``ip_a_minus_b_ms``. Plus ``ip_n_samples``, ``ip_n_nan``.
    """
    cfg = shot.machine
    ip = np.asarray(shot["magnetics_plasma_current"], dtype=np.float64)
    efit = np.asarray(shot["efit_times"], dtype=np.float64)
    fixes = shot.fixes
    out: dict = {"ip_n_samples": int(ip.size), "ip_n_nan": int(np.isnan(ip).sum())}

    if cfg.ip_has_own_time_axis:
        axis = np.asarray(shot["magnetics_plasma_current_times"], dtype=np.float64)
        shipped = fixes.get("ip_times_shipped")
        shipped = axis if shipped is None else np.asarray(shipped, dtype=np.float64)
        out["ip_fix_applied"] = bool(fixes.get("ip_fix_applied", False))
        out["ip_shift_ms"] = float(fixes.get("ip_time_shift_ms", 0.0))
        out["ip_alignment_applied"] = bool(fixes.get("ip_alignment_applied", False))
        out["ip_alignment_shift_ms"] = float(fixes.get("ip_alignment_shift_ms", 0.0))
        for key in ("ip_fix_error", "ip_alignment_error"):
            if key in fixes:
                out[key] = fixes[key]
    else:
        axis = np.asarray(shot["magnetics_time"], dtype=np.float64)
        shipped = axis
        out["ip_fix_applied"] = False
        out["ip_shift_ms"] = 0.0
        out["ip_alignment_applied"] = False
        out["ip_alignment_shift_ms"] = 0.0

    finite = np.isfinite(ip) & np.isfinite(axis)
    ip_f, axis_f, shipped_f = ip[finite], axis[finite], shipped[finite]
    if ip_f.size == 0:
        out.update(ip_covers_efit=False, ip_p995_abs_ka=np.nan, ip_median_abs_efit_ka=np.nan,
                   ip_median_over_p995=np.nan, ip_near_zero=True, ip_first_over_p995=np.nan,
                   ip_breakdown_to_first_frame_ms=np.nan, ip_first_frame_before_breakdown=False)
        return out

    ip_abs = np.abs(ip_f)
    p995 = float(np.percentile(ip_abs, 99.5))
    out["ip_p995_abs_ka"] = p995
    out["ip_breakdown_shipped_ms"] = ip_alignment.breakdown_time_ms(shipped_f, ip_f)
    aligned_breakdown = ip_alignment.breakdown_time_ms(axis_f, ip_f)
    # Breakdown after the organizers' fix equals the final one minus the alignment shift.
    out["ip_breakdown_fixed_ms"] = float(fixes.get("ip_breakdown_before_ms",
                                                   aligned_breakdown - out["ip_alignment_shift_ms"]))
    out["ip_breakdown_aligned_ms"] = aligned_breakdown
    out["ip_axis_min_ms"] = float(axis_f.min())
    out["ip_axis_max_ms"] = float(axis_f.max())

    lo, hi = efit_window_ms(efit)
    out["ip_covers_efit"] = bool(np.isfinite(lo) and axis_f.min() <= lo and axis_f.max() >= hi)

    order = np.argsort(axis_f)
    at_efit = np.abs(np.interp(efit, axis_f[order], ip_f[order]))
    median_final = float(np.median(at_efit))
    out["ip_median_abs_efit_ka"] = median_final
    if cfg.ip_has_own_time_axis:
        order_s = np.argsort(shipped_f)
        out["ip_median_abs_efit_shipped_ka"] = float(
            np.median(np.abs(np.interp(efit, shipped_f[order_s], ip_f[order_s]))))
    ratio = median_final / p995 if p995 > 0 else np.nan
    out["ip_median_over_p995"] = ratio
    out["ip_near_zero"] = bool(not np.isfinite(ratio) or ratio < IP_NEAR_ZERO_RATIO)
    first = float(at_efit[0]) if at_efit.size else np.nan
    out["ip_first_abs_efit_ka"] = first
    out["ip_first_over_p995"] = first / p995 if p995 > 0 else np.nan
    gap = lo - aligned_breakdown if np.isfinite(lo) else np.nan
    out["ip_breakdown_to_first_frame_ms"] = gap
    out["ip_first_frame_before_breakdown"] = bool(np.isfinite(gap) and gap < 0)

    if cfg.ip_has_own_time_axis and "magnetics_ECOILA" in shot and "magnetics_time" in shot:
        anchor = ip_alignment.solenoid_swing_down_ms(shot["magnetics_time"], shot["magnetics_ECOILA"])
        out["ip_sol_anchor_ms"] = anchor
        diag = ip_alignment.rule_a_diagnostics(anchor, out["ip_breakdown_fixed_ms"])
        out["ip_lag_a_ms"] = diag["lag_a_ms"]
        out["ip_shift_a_ms"] = diag["shift_a_ms"]
        out["ip_shift_b_ms"] = diag["shift_b_ms"]
        out["ip_a_minus_b_ms"] = diag["a_minus_b_ms"]
    return out


def current_columns(columns) -> list[str]:
    """Selects the current-carrying ``magnetics_*`` columns from a column list.

    Excludes time axes and the EFIT-derived ``magnetics_dsep`` label.
    """
    return sorted(
        c for c in columns
        if c.startswith("magnetics_")
        and not c.endswith(TIME_COLUMN_SUFFIXES)
        and not c.startswith("magnetics_dsep")
    )


def check_current_units(shot: Shot) -> dict:
    """Check 0.4: magnitude of every current column inside the EFIT window.

    Values are reported as stored (kA or kA-turn under the working assumption, see
    D1 in the data dictionary). The plasma current on DIII-D uses its own, fixed,
    time axis; every other column uses ``magnetics_time``.

    Returns:
        For each current column ``<c>`` (name without the ``magnetics_`` prefix):
        ``cur_<c>_min``, ``cur_<c>_max``, ``cur_<c>_p99abs`` (99th percentile of the
        absolute value), ``cur_<c>_nan_frac`` (NaN fraction inside the window) and
        ``cur_<c>_n_window`` (finite samples inside the window). Plus
        ``cur_n_columns``.
    """
    efit = np.asarray(shot["efit_times"], dtype=np.float64)
    cols = current_columns(shot.columns)
    out: dict = {"cur_n_columns": len(cols)}
    masks: dict[str, np.ndarray] = {}
    for col in cols:
        short = col[len("magnetics_"):]
        values = np.asarray(shot[col], dtype=np.float64)
        axis_name = shot.machine.time_axis_column(col)
        if axis_name not in shot:
            out[f"cur_{short}_error"] = f"time axis {axis_name} not loaded"
            continue
        if axis_name not in masks:
            masks[axis_name] = window_mask(shot[axis_name], efit)
        mask = masks[axis_name]
        if mask.shape != values.shape:
            out[f"cur_{short}_error"] = (
                f"length {values.shape[0]} does not match {axis_name} {mask.shape[0]}")
            continue
        inside = values[mask]
        finite = inside[np.isfinite(inside)]
        out[f"cur_{short}_nan_frac"] = float(1.0 - finite.size / inside.size) if inside.size else np.nan
        out[f"cur_{short}_n_window"] = int(finite.size)
        if finite.size:
            out[f"cur_{short}_min"] = float(finite.min())
            out[f"cur_{short}_max"] = float(finite.max())
            out[f"cur_{short}_p99abs"] = float(np.percentile(np.abs(finite), 99))
        else:
            out[f"cur_{short}_min"] = np.nan
            out[f"cur_{short}_max"] = np.nan
            out[f"cur_{short}_p99abs"] = np.nan
    return out
