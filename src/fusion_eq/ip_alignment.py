"""Alignment of the DIII-D plasma-current time axis beyond the organizers' fix.

Background. Every DIII-D file ships the same Ip time axis. The organizers' fix
(``data_fixes.fix_d3d_ip_times``) re-origins that axis to ``magnetics_time[0]`` when the
shipped axis puts plasma breakdown outside -500 to 1000 ms, and leaves it untouched
otherwise. The full-dataset QA run showed that the untouched shots are the 2 kHz
magnetics population, whose breakdown then sits around 190 ms, while the corrected
20 kHz population breaks down at a median of 14.9 ms. The two populations record the
same machine, so the spread is an axis offset, not physics.

Rule B (applied by the loader). For a shot the organizers' fix leaves untouched, shift the
Ip time axis so that its breakdown time equals ``TARGET_BREAKDOWN_MS``. Breakdown uses
the organizers' definition: the first sample where |Ip| exceeds 10 percent of the shot's
99.5th percentile of |Ip|.

Rule A (diagnostic only). Anchor on the central-solenoid swing-down, the time
``magnetics_ECOILA`` leaves its pre-charge plateau. The difference between the shift rule
A would apply and the rule B shift is reported by QA check 0.2 to monitor whether the
breakdown-based alignment agrees with an Ip-independent marker.
"""

from __future__ import annotations

import numpy as np

# Fraction of the 99.5th percentile of |Ip| that marks plasma breakdown. This is the
# organizers' definition in data_fixes.fix_d3d_ip_times and must stay equal to it.
BREAKDOWN_FRACTION = 0.10

# Target breakdown time in ms for rule B: the median Ip breakdown of the DIII-D 20 kHz
# magnetics population after the organizers' fix, measured on the full training and public
# test sets (QA run of 2026-10-04, 5,518 shots). Hard-coded so that the alignment is a fixed
# rule of the pipeline and does not depend on which shots are loaded.
TARGET_BREAKDOWN_MS = 14.9

# Reference solenoid swing-down time in ms for the rule A diagnostic: the median anchor of the
# DIII-D 20 kHz magnetics population, measured on the full training and public test sets
# (notebooks/01_qa.ipynb, section 4, QA run of 2026-10-04).
TARGET_ANCHOR_MS = 1.5

# Rule A anchor parameters: moving-average width and plateau tolerance.
SWING_SMOOTH_MS = 5.0
SWING_PLATEAU_TOL = 0.05


def breakdown_time_ms(times_ms, ip_ka, fraction: float = BREAKDOWN_FRACTION) -> float:
    """Plasma breakdown time under the organizers' definition.

    Args:
        times_ms: Ip time axis in ms.
        ip_ka: Plasma current in kA (any consistent unit; only ratios matter).
        fraction: Fraction of the 99.5th percentile of |Ip| that marks breakdown.

    Returns:
        Time in ms of the first sample where |Ip| exceeds the threshold, or NaN when no
        sample does (the organizers' helper would return the first time stamp there).
        Samples whose time or current is not finite are skipped and the rest are taken in
        time order, as the resampling does: the MAST shots with merged clocks hold NaN on
        one sample in six (F20). On a sorted axis without NaN this is the organizers' rule.
    """
    t = np.asarray(times_ms, dtype=np.float64).ravel()
    ip = np.abs(np.asarray(ip_ka, dtype=np.float64)).ravel()
    if t.shape != ip.shape:
        raise ValueError(f"times and currents differ in length: {t.size} versus {ip.size}")
    keep = np.isfinite(t) & np.isfinite(ip)
    if not keep.any():
        return np.nan
    order = np.argsort(t[keep], kind="stable")
    t, ip = t[keep][order], ip[keep][order]
    threshold = fraction * np.percentile(ip, 99.5)
    above = ip > threshold
    if not above.any():
        return np.nan
    return float(t[int(np.argmax(above))])


def align_ip_times(times_ms, ip_ka, target_ms: float = TARGET_BREAKDOWN_MS):
    """Rule B: shifts an Ip time axis so that its breakdown time equals ``target_ms``.

    Args:
        times_ms: Ip time axis in ms, after the organizers' fix.
        ip_ka: Plasma current on that axis.
        target_ms: Breakdown time to impose, in ms.

    Returns:
        ``(aligned_times_ms, shift_ms, breakdown_before_ms)``. The shift is
        ``target_ms - breakdown_before_ms`` and is added to every time stamp. When no
        breakdown is found the axis is returned unchanged with a shift of 0 and a NaN
        breakdown.
    """
    t = np.asarray(times_ms, dtype=np.float64)
    before = breakdown_time_ms(t, ip_ka)
    if not np.isfinite(before):
        return t, 0.0, before
    shift = float(target_ms - before)
    return t + shift, shift, before


def moving_average(values, window: int):
    """Centred moving average over ``window`` samples; edges are padded with end values."""
    v = np.asarray(values, dtype=np.float64)
    if window <= 1 or v.size == 0:
        return v
    padded = np.pad(v, (window // 2, window - 1 - window // 2), mode="edge")
    cumsum = np.concatenate([[0.0], np.cumsum(padded)])
    return (cumsum[window:] - cumsum[:-window]) / window


def solenoid_swing_down_ms(times_ms, current_ka, smooth_ms: float = SWING_SMOOTH_MS,
                           tol: float = SWING_PLATEAU_TOL) -> float:
    """Rule A anchor: the time the solenoid current leaves its pre-charge plateau.

    The current is oriented so that the pre-charge is positive (the sign at the first
    sample exceeding half the record's maximum magnitude) and smoothed with a
    ``smooth_ms`` moving average. The plateau is the smoothed maximum. The steepest fall
    is the sample with the most negative slope, which is the inductive swing. The anchor
    is the last time at or before the steepest fall where the smoothed trace is still
    within ``tol`` of the plateau.

    Args:
        times_ms: Time axis in ms (``magnetics_time``).
        current_ka: Solenoid current (``magnetics_ECOILA``).

    Returns:
        Anchor time in ms, or NaN when the trace is empty or identically zero.
    """
    t = np.asarray(times_ms, dtype=np.float64)
    c = np.asarray(current_ka, dtype=np.float64)
    finite = np.isfinite(t) & np.isfinite(c)
    t, c = t[finite], c[finite]
    if t.size < 3:
        return np.nan
    peak = float(np.abs(c).max())
    if peak <= 0:
        return np.nan
    orientation = np.sign(c[int(np.argmax(np.abs(c) > 0.5 * peak))])
    dt = float(np.median(np.diff(t)))
    s = moving_average(orientation * c, max(1, int(round(smooth_ms / dt))))
    plateau = float(s.max())
    steepest = int(np.argmin(np.gradient(s, t)))
    near_plateau = np.flatnonzero(s[: steepest + 1] >= (1.0 - tol) * plateau)
    if near_plateau.size == 0:
        return np.nan
    return float(t[near_plateau[-1]])


def rule_a_diagnostics(anchor_ms: float, breakdown_fixed_ms: float,
                       target_breakdown_ms: float = TARGET_BREAKDOWN_MS,
                       target_anchor_ms: float = TARGET_ANCHOR_MS) -> dict:
    """Shift rule A would apply, the rule B shift, and their difference, in ms.

    Args:
        anchor_ms: Solenoid swing-down time of the shot.
        breakdown_fixed_ms: Ip breakdown after the organizers' fix, before alignment.

    Returns:
        ``lag_a_ms`` (breakdown minus anchor), ``shift_a_ms`` (makes the lag equal to the
        reference lag ``target_breakdown_ms - target_anchor_ms``), ``shift_b_ms``
        (makes breakdown equal to ``target_breakdown_ms``) and ``a_minus_b_ms``. All NaN
        when the anchor or the breakdown is NaN.
    """
    lag = breakdown_fixed_ms - anchor_ms
    shift_a = (target_breakdown_ms - target_anchor_ms) - lag
    shift_b = target_breakdown_ms - breakdown_fixed_ms
    return {"lag_a_ms": float(lag), "shift_a_ms": float(shift_a), "shift_b_ms": float(shift_b),
            "a_minus_b_ms": float(shift_a - shift_b)}
