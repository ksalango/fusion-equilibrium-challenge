"""Resampling of a sampled signal onto arbitrary target times.

Three forms serve the frame table and, later, lag features: the nearest valid
sample, the mean over a centred window, and the least-squares slope over a centred
window. Windows and gaps are given in milliseconds so that signals recorded at
different rates (20 kHz, 5 kHz, 2 kHz, the 0.5 ms Ip axis) are treated alike.

Non-finite samples and non-finite times are dropped before anything else, so a NaN
hole in a signal (the MAST shots of finding F20) is skipped rather than propagated.
All functions accept unsorted input and return one value per target time.
"""

from __future__ import annotations

import numpy as np

MS_PER_S = 1000.0

# Upper bound on the cells of the per-window matrix built by ``window_slope`` at a time.
_MAX_WINDOW_CELLS = 2_000_000


def _valid_sorted(times_ms, values) -> tuple[np.ndarray, np.ndarray]:
    """Finite (time, value) pairs as float64, sorted by time."""
    t = np.asarray(times_ms, dtype=np.float64).ravel()
    v = np.asarray(values, dtype=np.float64).ravel()
    if t.shape != v.shape:
        raise ValueError(f"times and values differ in length: {t.size} versus {v.size}")
    keep = np.isfinite(t) & np.isfinite(v)
    t, v = t[keep], v[keep]
    order = np.argsort(t, kind="stable")
    return t[order], v[order]


def _targets(target_ms) -> np.ndarray:
    return np.asarray(target_ms, dtype=np.float64)


def nearest_distance_ms(query_ms, reference_ms) -> np.ndarray:
    """Distance in ms from each query time to the nearest reference time.

    Args:
        query_ms: Times to look up, any shape.
        reference_ms: Reference times, unsorted and possibly containing NaN.

    Returns:
        Array with the shape of ``query_ms``; infinity when there is no finite
        reference time.
    """
    q = _targets(query_ms)
    ref = np.sort(np.asarray(reference_ms, dtype=np.float64).ravel())
    ref = ref[np.isfinite(ref)]
    if ref.size == 0:
        return np.full(q.shape, np.inf)
    idx = np.searchsorted(ref, q)
    left = ref[np.clip(idx - 1, 0, ref.size - 1)]
    right = ref[np.clip(idx, 0, ref.size - 1)]
    return np.minimum(np.abs(q - left), np.abs(q - right))


def nearest_index(sorted_times_ms: np.ndarray, target_ms: np.ndarray) -> np.ndarray:
    """Index of the nearest time in a sorted axis for each target; ties go to the earlier sample.

    Args:
        sorted_times_ms: Sorted finite times, at least one entry.
        target_ms: Target times, any shape.
    """
    idx = np.searchsorted(sorted_times_ms, target_ms)
    left = np.clip(idx - 1, 0, sorted_times_ms.size - 1)
    right = np.clip(idx, 0, sorted_times_ms.size - 1)
    use_right = np.abs(sorted_times_ms[right] - target_ms) < np.abs(target_ms - sorted_times_ms[left])
    return np.where(use_right, right, left)


def nearest_valid(times_ms, values, target_ms, max_gap_ms: float) -> np.ndarray:
    """Value of the nearest valid sample to each target time.

    Args:
        times_ms: Sample times in ms.
        values: Sample values, same length as ``times_ms``.
        target_ms: Target times, any shape.
        max_gap_ms: Targets farther than this from the nearest valid sample get NaN.

    Returns:
        Array with the shape of ``target_ms``, in the units of ``values``.
    """
    t, v = _valid_sorted(times_ms, values)
    target = _targets(target_ms)
    out = np.full(target.shape, np.nan)
    if t.size == 0:
        return out
    idx = nearest_index(t, target)
    within = np.abs(t[idx] - target) <= max_gap_ms
    out[within] = v[idx[within]]
    return out


def _window_bounds(t: np.ndarray, target: np.ndarray, window_ms: float) -> tuple[np.ndarray, np.ndarray]:
    """Index range [lo, hi) of the samples inside the closed centred window of each target."""
    half = 0.5 * window_ms
    lo = np.searchsorted(t, target - half, side="left")
    hi = np.searchsorted(t, target + half, side="right")
    return lo, hi


def _window_sums(cumulative: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Sum over [lo, hi) from a cumulative array with a leading zero."""
    return cumulative[hi] - cumulative[lo]


def window_mean(times_ms, values, target_ms, window_ms: float, max_gap_ms: float) -> np.ndarray:
    """Mean of the valid samples in a window centred on each target time.

    When the window holds no valid sample, the value is the linear interpolation
    between the valid samples on either side of the target, provided those two are
    at most ``max_gap_ms`` apart; otherwise NaN. Targets outside the span of the
    valid samples get NaN.

    Args:
        times_ms: Sample times in ms.
        values: Sample values, same length as ``times_ms``.
        target_ms: Target times, any shape.
        window_ms: Full window length in ms; the window is closed on both sides.
        max_gap_ms: Largest gap between bracketing samples that is interpolated over.

    Returns:
        Array with the shape of ``target_ms``, in the units of ``values``.
    """
    t, v = _valid_sorted(times_ms, values)
    target = _targets(target_ms)
    out = np.full(target.shape, np.nan)
    if t.size == 0:
        return out
    lo, hi = _window_bounds(t, target, window_ms)
    count = hi - lo
    cum_v = np.concatenate([[0.0], np.cumsum(v)])
    has = count > 0
    out[has] = _window_sums(cum_v, lo[has], hi[has]) / count[has]

    empty = ~has & (lo > 0) & (lo < t.size)
    if np.any(empty):
        left, right = lo[empty] - 1, lo[empty]
        gap = t[right] - t[left]
        ok = gap <= max_gap_ms
        frac = np.where(ok, (target[empty] - t[left]) / np.where(gap > 0, gap, 1.0), np.nan)
        interp = v[left] + frac * (v[right] - v[left])
        out[empty] = np.where(ok, interp, np.nan)
    return out


def window_slope(times_ms, values, target_ms, window_ms: float, min_samples: int = 3) -> np.ndarray:
    """Least-squares slope of the valid samples in a window centred on each target time.

    Args:
        times_ms: Sample times in ms.
        values: Sample values, same length as ``times_ms``.
        target_ms: Target times, any shape.
        window_ms: Full window length in ms; the window is closed on both sides.
        min_samples: Windows with fewer valid samples give NaN. Windows whose
            samples all share one time also give NaN.

    Returns:
        Array with the shape of ``target_ms``, in units of ``values`` per second.
    """
    t, v = _valid_sorted(times_ms, values)
    target = _targets(target_ms)
    flat = np.full(target.size, np.nan)
    if t.size == 0:
        return flat.reshape(target.shape)
    target_flat = target.ravel()
    lo, hi = _window_bounds(t, target_flat, window_ms)
    count = hi - lo
    enough = np.flatnonzero(count >= max(int(min_samples), 2))
    if enough.size == 0:
        return flat.reshape(target.shape)
    width = int(count[enough].max())
    # One row per target holding its window, with times centred on the target. Cumulative
    # sums over the whole axis would lose about six digits to cancellation at the time
    # offsets of these signals; the per-window form is exact to float64 precision. Rows
    # are processed in chunks so that the matrix stays small for long windows.
    chunk = max(1, int(_MAX_WINDOW_CELLS // max(width, 1)))
    offsets = np.arange(width)
    for start in range(0, enough.size, chunk):
        rows = enough[start:start + chunk]
        idx = lo[rows, None] + offsets[None, :]
        inside = offsets[None, :] < count[rows, None]
        idx = np.clip(idx, 0, t.size - 1)
        tw = np.where(inside, (t[idx] - target_flat[rows, None]) / MS_PER_S, np.nan)
        vw = np.where(inside, v[idx], np.nan)
        tw = tw - np.nanmean(tw, axis=1, keepdims=True)
        vw = vw - np.nanmean(vw, axis=1, keepdims=True)
        denominator = np.nansum(tw * tw, axis=1)
        numerator = np.nansum(tw * vw, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            slope = numerator / denominator
        # Zero spread in time (repeated samples) is reported as undefined, not infinite.
        slope[~(denominator > 0.0)] = np.nan
        flat[rows] = slope
    return flat.reshape(target.shape)


def linear_interpolate(times_ms, values, target_ms, hold_ends: bool = True) -> np.ndarray:
    """Linear interpolation of the valid samples at each target time.

    Args:
        times_ms: Sample times in ms.
        values: Sample values, same length as ``times_ms``.
        target_ms: Target times, any shape.
        hold_ends: When True, targets before the first or after the last valid sample
            take the end value; when False they get NaN.

    Returns:
        Array with the shape of ``target_ms``, in the units of ``values``; all NaN
        when no sample is valid.
    """
    t, v = _valid_sorted(times_ms, values)
    target = _targets(target_ms)
    if t.size == 0:
        return np.full(target.shape, np.nan)
    out = np.interp(target, t, v)
    if not hold_ends:
        out = np.where((target < t[0]) | (target > t[-1]), np.nan, out)
    return out
