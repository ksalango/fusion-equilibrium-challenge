"""Checks 0.3 (MAST Thomson channel alignment) and 0.7 (missing Thomson data)."""

from __future__ import annotations

import json

import numpy as np

from fusion_eq.io import Shot
from fusion_eq.qa.efit_window import nearest_distance_ms, window_mask
from fusion_eq.thomson import (SYSTEMS, VALID_PROFILE_FRACTION, missing_mask, valid_channel_mask,
                               valid_slice_mask)

__all__ = ["check_thomson_alignment", "check_thomson_missing", "missing_mask",
           "VALID_PROFILE_FRACTION", "MATCH_TOLERANCES_MS", "SYSTEMS"]

# Tolerances for matching a Thomson slice to an EFIT frame, in ms.
MATCH_TOLERANCES_MS = (10.0, 20.0)


def check_thomson_alignment(shot: Shot) -> dict:
    """Check 0.3: what the MAST Thomson core alignment fix did to this shot.

    The loader applies ``align_mast_thomson_core`` and records the result in
    ``shot.fixes``; this check reports it with the position count for comparison.

    Returns:
        ``ts_fix_applied``, ``ts_channels_before``, ``ts_channels_after``,
        ``ts_n_positions`` (length of ``thomson_core_R`` after the fix),
        ``ts_aligned`` (channels after equals positions),
        ``ts_dropped_nan_frac_te`` and ``ts_dropped_nan_frac_ne`` (NaN fraction of
        the dropped channel, only when a channel was dropped), ``ts_fix_error``.
    """
    fixes = shot.fixes
    r = shot.get("thomson_core_R")
    te = shot.get("thomson_core_Te")
    n_positions = int(len(r)) if r is not None else -1
    n_after = int(te.shape[1]) if isinstance(te, np.ndarray) and te.ndim == 2 else -1
    out = {
        "ts_fix_applied": bool(fixes.get("thomson_core_fix_applied", False)),
        "ts_channels_before": int(fixes.get("thomson_core_channels_before", n_after)),
        "ts_channels_after": n_after,
        "ts_n_positions": n_positions,
        "ts_aligned": bool(n_after == n_positions and n_after > 0),
    }
    for key in ("thomson_core_dropped_nan_frac_te", "thomson_core_dropped_nan_frac_ne"):
        if key in fixes:
            out["ts_dropped_nan_frac_" + key[-2:]] = float(fixes[key])
    if "thomson_core_fix_error" in fixes:
        out["ts_fix_error"] = fixes["thomson_core_fix_error"]
    return out


def _system_metrics(prefix: str, times, te, ne, efit: np.ndarray, convention: str) -> dict:
    out: dict = {}
    times = np.asarray(times, dtype=np.float64)
    out[f"{prefix}n_times"] = int(times.size)
    if not (isinstance(te, np.ndarray) and isinstance(ne, np.ndarray) and te.ndim == 2 and ne.ndim == 2):
        out[f"{prefix}error"] = "Te or ne is not a rectangular 2-D array"
        return out
    out[f"{prefix}n_channels"] = int(te.shape[1])
    out[f"{prefix}shape_matches_times"] = bool(te.shape[0] == times.size and ne.shape == te.shape)
    if times.size > 1:
        out[f"{prefix}dt_median_ms"] = float(np.median(np.diff(np.sort(times[np.isfinite(times)]))))
    if not out[f"{prefix}shape_matches_times"]:
        return out

    valid = valid_channel_mask(te, ne, convention)
    inside = window_mask(times, efit)
    out[f"{prefix}n_times_window"] = int(inside.sum())
    n_channels = te.shape[1]
    if inside.sum() == 0 or n_channels == 0:
        out[f"{prefix}missing_frac_mean"] = np.nan
        return out

    valid_w = valid[inside]
    missing_per_channel = 1.0 - valid_w.mean(axis=0)
    out[f"{prefix}missing_frac_mean"] = float(missing_per_channel.mean())
    out[f"{prefix}missing_frac_min"] = float(missing_per_channel.min())
    out[f"{prefix}missing_frac_max"] = float(missing_per_channel.max())
    out[f"{prefix}missing_frac_per_channel"] = json.dumps(
        [round(float(x), 4) for x in missing_per_channel])
    n_valid_per_slice = valid_w.sum(axis=1)
    is_profile = valid_slice_mask(valid_w)
    out[f"{prefix}slice_all_missing_frac"] = float((n_valid_per_slice == 0).mean())
    out[f"{prefix}slice_valid_profile_frac"] = float(is_profile.mean())

    t_w = times[inside]
    profile_times = t_w[is_profile]
    any_times = t_w[n_valid_per_slice >= 1]
    d_profile = nearest_distance_ms(efit, profile_times)
    d_any = nearest_distance_ms(efit, any_times)
    for tol in MATCH_TOLERANCES_MS:
        tag = f"{int(tol)}ms"
        out[f"{prefix}frames_valid_within_{tag}"] = float((d_profile <= tol).mean()) if efit.size else np.nan
        out[f"{prefix}frames_any_within_{tag}"] = float((d_any <= tol).mean()) if efit.size else np.nan
    return out


def check_thomson_missing(shot: Shot) -> dict:
    """Check 0.7: missing Thomson data inside the EFIT window, per system.

    A channel is valid at a time slice only when both Te and ne are present under
    the machine's convention (exact zero on DIII-D, NaN on MAST). A slice is a
    valid profile when at least ``VALID_PROFILE_FRACTION`` of its channels are
    valid. The ``frames_any_*`` metrics use one valid channel instead, so the
    threshold can be revisited from the stored table.

    Returns:
        For each system ``core`` and ``edge`` present, with prefix ``tsm_<sys>_``:
        ``n_times``, ``n_channels``, ``shape_matches_times`` (rows equal the time
        axis, Te and ne same shape), ``dt_median_ms``, ``n_times_window``,
        ``missing_frac_mean/min/max`` over channels, ``missing_frac_per_channel``
        (JSON list), ``slice_all_missing_frac``, ``slice_valid_profile_frac``,
        ``frames_valid_within_10ms/20ms`` and ``frames_any_within_10ms/20ms``
        (fractions of EFIT frames).
    """
    efit = np.asarray(shot["efit_times"], dtype=np.float64)
    out: dict = {}
    for system in SYSTEMS:
        times = shot.get(f"thomson_{system}_times")
        te = shot.get(f"thomson_{system}_Te")
        ne = shot.get(f"thomson_{system}_ne")
        if times is None or te is None or ne is None:
            continue
        out.update(_system_metrics(f"tsm_{system}_", times, te, ne, efit,
                                   shot.machine.thomson_missing))
    return out
