"""Selection of samples inside the EFIT window of a shot.

The EFIT window is the closed interval from the first to the last entry of
``efit_times``. Input signals span far beyond it (pre-shot baseline, post-shot
decay), so statistics about the data a model would see are taken inside it.
"""

from __future__ import annotations

import numpy as np


def efit_window_ms(efit_times: np.ndarray) -> tuple[float, float]:
    """Returns the (first, last) finite EFIT time in ms."""
    t = np.asarray(efit_times, dtype=np.float64)
    t = t[np.isfinite(t)]
    if t.size == 0:
        return (np.nan, np.nan)
    return float(t.min()), float(t.max())


def window_mask(times_ms: np.ndarray, efit_times: np.ndarray, pad_ms: float = 0.0) -> np.ndarray:
    """Boolean mask of the samples whose time lies inside the EFIT window.

    Args:
        times_ms: Time axis of the signal in ms. NaN entries are excluded.
        efit_times: EFIT time base in ms.
        pad_ms: Widens the window on both sides, in ms.

    Returns:
        Mask with the shape of ``times_ms``; all False when the window is empty.
    """
    t = np.asarray(times_ms, dtype=np.float64)
    lo, hi = efit_window_ms(efit_times)
    if not np.isfinite(lo):
        return np.zeros(t.shape, dtype=bool)
    return np.isfinite(t) & (t >= lo - pad_ms) & (t <= hi + pad_ms)


def nearest_distance_ms(query_ms: np.ndarray, reference_ms: np.ndarray) -> np.ndarray:
    """Distance in ms from each query time to the nearest reference time.

    Args:
        query_ms: Times to look up, any shape.
        reference_ms: Reference times, unsorted and possibly containing NaN.

    Returns:
        Array with the shape of ``query_ms``; infinity when there is no finite
        reference time.
    """
    q = np.asarray(query_ms, dtype=np.float64)
    ref = np.sort(np.asarray(reference_ms, dtype=np.float64))
    ref = ref[np.isfinite(ref)]
    if ref.size == 0:
        return np.full(q.shape, np.inf)
    idx = np.searchsorted(ref, q)
    left = ref[np.clip(idx - 1, 0, ref.size - 1)]
    right = ref[np.clip(idx, 0, ref.size - 1)]
    return np.minimum(np.abs(q - left), np.abs(q - right))
