"""Checks 0.2 (DIII-D Ip time-axis erratum) and 0.4 (units of current columns)."""

from __future__ import annotations

import numpy as np

from fusion_eq.io import Shot
from fusion_eq.qa.efit_window import efit_window_ms, window_mask

# Fraction of the 99.5th percentile of |Ip| that marks plasma breakdown. The same
# threshold the starter's fix uses, so the breakdown times reported here match its
# decision.
BREAKDOWN_FRACTION = 0.10

# Below this ratio of median |Ip| at EFIT frames to the shot's 99.5th percentile of
# |Ip|, the Ip trace read at the EFIT frames is the pre-plasma noise floor.
IP_NEAR_ZERO_RATIO = 0.10

TIME_COLUMN_SUFFIXES = ("_time", "_times")


def _breakdown_ms(axis_ms: np.ndarray, ip_abs: np.ndarray, threshold: float) -> float:
    above = ip_abs > threshold
    if not above.any():
        return np.nan
    return float(axis_ms[int(np.argmax(above))])


def check_ip_time_axis(shot: Shot) -> dict:
    """Check 0.2: the plasma-current time axis after the data fix.

    On DIII-D the fix may re-origin ``magnetics_plasma_current_times``; the loader
    records the shipped axis in ``shot.fixes``. On MAST Ip sits on
    ``magnetics_time`` and only the coverage and magnitude metrics apply.

    Returns:
        ``ip_fix_applied``, ``ip_shift_ms`` (fixed minus shipped origin),
        ``ip_breakdown_shipped_ms``, ``ip_breakdown_fixed_ms``, ``ip_axis_min_ms``,
        ``ip_axis_max_ms``, ``ip_covers_efit`` (the fixed axis spans the EFIT
        window), ``ip_p995_abs_ka``, ``ip_median_abs_efit_ka`` (median |Ip|
        interpolated at the EFIT frames, fixed axis), ``ip_median_abs_efit_shipped_ka``
        (same on the shipped axis, DIII-D only), ``ip_median_over_p995`` (the ratio
        the fail rule uses), ``ip_near_zero`` (ratio below ``IP_NEAR_ZERO_RATIO``),
        ``ip_n_nan``, ``ip_n_samples``.
    """
    cfg = shot.machine
    ip = np.asarray(shot["magnetics_plasma_current"], dtype=np.float64)
    efit = np.asarray(shot["efit_times"], dtype=np.float64)
    out: dict = {"ip_n_samples": int(ip.size), "ip_n_nan": int(np.isnan(ip).sum())}

    if cfg.ip_has_own_time_axis:
        axis = np.asarray(shot["magnetics_plasma_current_times"], dtype=np.float64)
        shipped = shot.fixes.get("ip_times_shipped")
        shipped = axis if shipped is None else np.asarray(shipped, dtype=np.float64)
        out["ip_fix_applied"] = bool(shot.fixes.get("ip_fix_applied", False))
        out["ip_shift_ms"] = float(shot.fixes.get("ip_time_shift_ms", 0.0))
        if "ip_fix_error" in shot.fixes:
            out["ip_fix_error"] = shot.fixes["ip_fix_error"]
    else:
        axis = np.asarray(shot["magnetics_time"], dtype=np.float64)
        shipped = axis
        out["ip_fix_applied"] = False
        out["ip_shift_ms"] = 0.0

    finite = np.isfinite(ip) & np.isfinite(axis)
    ip_f, axis_f, shipped_f = ip[finite], axis[finite], shipped[finite]
    if ip_f.size == 0:
        out.update(ip_covers_efit=False, ip_p995_abs_ka=np.nan, ip_median_abs_efit_ka=np.nan,
                   ip_median_over_p995=np.nan, ip_near_zero=True)
        return out

    ip_abs = np.abs(ip_f)
    p995 = float(np.percentile(ip_abs, 99.5))
    threshold = BREAKDOWN_FRACTION * p995
    out["ip_p995_abs_ka"] = p995
    out["ip_breakdown_shipped_ms"] = _breakdown_ms(shipped_f, ip_abs, threshold)
    out["ip_breakdown_fixed_ms"] = _breakdown_ms(axis_f, ip_abs, threshold)
    out["ip_axis_min_ms"] = float(axis_f.min())
    out["ip_axis_max_ms"] = float(axis_f.max())

    lo, hi = efit_window_ms(efit)
    out["ip_covers_efit"] = bool(np.isfinite(lo) and axis_f.min() <= lo and axis_f.max() >= hi)

    order = np.argsort(axis_f)
    median_fixed = float(np.median(np.abs(np.interp(efit, axis_f[order], ip_f[order]))))
    out["ip_median_abs_efit_ka"] = median_fixed
    if cfg.ip_has_own_time_axis:
        order_s = np.argsort(shipped_f)
        out["ip_median_abs_efit_shipped_ka"] = float(
            np.median(np.abs(np.interp(efit, shipped_f[order_s], ip_f[order_s]))))
    ratio = median_fixed / p995 if p995 > 0 else np.nan
    out["ip_median_over_p995"] = ratio
    out["ip_near_zero"] = bool(not np.isfinite(ratio) or ratio < IP_NEAR_ZERO_RATIO)
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
        if shot.machine.ip_has_own_time_axis and col == "magnetics_plasma_current":
            axis_name = "magnetics_plasma_current_times"
        else:
            axis_name = "magnetics_time"
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
