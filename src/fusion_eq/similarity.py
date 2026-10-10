"""Shot similarity from inputs: fingerprints, pairwise distances, neighbours and groups.

Two shots are compared through their current signals sampled at fixed times (the
fingerprint), scaled by each signal's typical range over the training frames. The
distance between two shots is the root mean square of the scaled differences over
the fingerprint entries present in both, so that a distance of 0.1 means the currents
differ by a tenth of each signal's range over the time the shots overlap. Near-repeat
pairs are those below a threshold and groups are complete-linkage clusters at that
threshold, so that every pair inside a group is a near-repeat pair.

Everything here reads inputs only, so it is defined for test shots as well. The
input-distribution comparison of analysis B8a lives here too, for the same reason.
Specification: ``docs/SPLIT.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import ks_2samp

from fusion_eq.frame_table import current_signals
from fusion_eq.machines import MachineConfig, machine_config

# Fingerprint grid: every 100 ms from 100 ms to 9,000 ms.
FINGERPRINT_TIMES_MS = np.arange(100.0, 9000.0 + 0.5, 100.0)
# A grid time inside a gap between frames longer than this is not interpolated.
MAX_GAP_MS = 200.0
# A pair has a distance when it shares at least this many grid times and at least this
# fraction of the grid times of the shorter shot.
MIN_COMMON_TIMES = 5
MIN_COMMON_FRACTION = 0.5
# Thresholds reported for near-repeat counts; the middle one is the provisional value.
THRESHOLDS = (0.05, 0.10, 0.20)
PROVISIONAL_THRESHOLD = 0.10
# Distance given to undefined pairs when clustering, far above any threshold.
UNDEFINED_DISTANCE = 1e6
# Frames with abs(Ip) above this fraction of the shot's peak count as flat-top (B8a).
FLAT_TOP_LEVEL = 0.9
# Signals whose scale is the interquartile range of the absolute value: the toroidal
# field is constant within a shot and takes both signs across shots.
ABSOLUTE_SCALE_SIGNALS = ("tf__mean",)


# -------------------------------------------------------------------------- signals
def similarity_signals(machine: str | MachineConfig = "DIII-D") -> list[str]:
    """The frame-table columns the fingerprint is built from: every current in ``__mean`` form."""
    cfg = machine if isinstance(machine, MachineConfig) else machine_config(machine)
    return [f"{name}__mean" for name in current_signals(cfg)]


# ---------------------------------------------------------------------- fingerprints
@dataclass
class Fingerprints:
    """Signals of many shots at the fixed grid times.

    Attributes:
        shots: Shot ids, ``(n_shots,)``.
        times_ms: Grid times, ``(n_times,)``.
        signals: Signal names, ``(n_signals,)``.
        values: ``(n_shots, n_times, n_signals)`` float64, NaN where undefined.
    """

    shots: np.ndarray
    times_ms: np.ndarray
    signals: tuple[str, ...]
    values: np.ndarray

    @property
    def present(self) -> np.ndarray:
        """``(n_shots, n_times)`` bool: grid times with at least one finite signal."""
        return np.isfinite(self.values).any(axis=2)

    @property
    def n_times_present(self) -> np.ndarray:
        """Grid times inside each shot's window, ``(n_shots,)``."""
        return self.present.sum(axis=1)

    def index_of(self, shot_ids: Iterable[str]) -> np.ndarray:
        lookup = {s: i for i, s in enumerate(self.shots.tolist())}
        return np.array([lookup[s] for s in shot_ids], dtype=int)


def interpolate_to_grid(time_ms, values, grid_ms, max_gap_ms: float = MAX_GAP_MS) -> np.ndarray:
    """Linear interpolation of ``(frames, n_signals)`` values onto grid times.

    A grid time outside ``[time_ms[0], time_ms[-1]]``, or inside a gap between
    consecutive frames longer than ``max_gap_ms``, gives NaN. A grid time that falls on
    a frame takes that frame's values. NaN frame values propagate to the grid times
    they bracket.

    Args:
        time_ms: Strictly increasing frame times, ``(frames,)``.
        values: ``(frames, n_signals)``.
        grid_ms: Grid times, ``(n_grid,)``.

    Returns:
        ``(n_grid, n_signals)`` float64.
    """
    t = np.asarray(time_ms, dtype=np.float64).ravel()
    v = np.asarray(values, dtype=np.float64)
    g = np.asarray(grid_ms, dtype=np.float64).ravel()
    if v.ndim != 2 or v.shape[0] != t.size:
        raise ValueError("values must be (frames, n_signals) with one row per time")
    out = np.full((g.size, v.shape[1]), np.nan)
    if t.size == 0:
        return out
    if t.size > 1 and np.any(np.diff(t) <= 0):
        raise ValueError("frame times must be strictly increasing")
    inside = (g >= t[0]) & (g <= t[-1])
    if not inside.any():
        return out
    gi = g[inside]
    hi = np.clip(np.searchsorted(t, gi, side="left"), 0, t.size - 1)
    lo = np.clip(hi - 1, 0, t.size - 1)
    exact = t[hi] == gi
    lo = np.where(exact, hi, lo)
    gap = t[hi] - t[lo]
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.where(gap > 0, (gi - t[lo]) / gap, 0.0)
    interp = v[lo] * (1.0 - w)[:, None] + v[hi] * w[:, None]
    interp[gap > max_gap_ms] = np.nan
    out[inside] = interp
    return out


def shot_fingerprints(frames: pd.DataFrame, signals: Sequence[str], times_ms=FINGERPRINT_TIMES_MS,
                      max_gap_ms: float = MAX_GAP_MS) -> Fingerprints:
    """Fingerprints of every shot in a frame table.

    Args:
        frames: Frame-table rows with ``shot``, ``time_ms`` and the signal columns.
            Frames of a shot may appear in any order.
        signals: Signal columns.
        times_ms: Grid times.
        max_gap_ms: Largest gap between frames interpolated over.

    Returns:
        Fingerprints with shots in sorted id order.
    """
    signals = tuple(signals)
    times = np.asarray(times_ms, dtype=np.float64)
    ids = np.array(sorted(frames["shot"].astype(str).unique()), dtype=object)
    values = np.full((ids.size, times.size, len(signals)), np.nan)
    ordered = frames.sort_values(["shot", "time_ms"], kind="stable")
    for i, (shot_id, rows) in enumerate(ordered.groupby(ordered["shot"].astype(str), sort=True)):
        values[i] = interpolate_to_grid(rows["time_ms"].to_numpy(), rows[list(signals)].to_numpy(dtype=np.float64),
                                        times, max_gap_ms)
    return Fingerprints(shots=ids, times_ms=times, signals=signals, values=values)


def signal_scales(frames: pd.DataFrame, signals: Sequence[str],
                  absolute: Sequence[str] = ABSOLUTE_SCALE_SIGNALS) -> np.ndarray:
    """Interquartile range of each signal over the given (training) frames.

    Signals listed in ``absolute`` are scaled by the interquartile range of their
    absolute value.

    Raises:
        ValueError: when a scale is not finite and positive.
    """
    scales = np.empty(len(signals))
    for j, name in enumerate(signals):
        x = frames[name].to_numpy(dtype=np.float64)
        if name in absolute:
            x = np.abs(x)
        x = x[np.isfinite(x)]
        q75, q25 = np.percentile(x, [75, 25]) if x.size else (np.nan, np.nan)
        scales[j] = q75 - q25
    bad = [name for name, s in zip(signals, scales) if not np.isfinite(s) or s <= 0]
    if bad:
        raise ValueError(f"signals without a positive interquartile range: {bad}")
    return scales


# ----------------------------------------------------------------------- distances
@dataclass
class PairwiseDistance:
    """Distances between every pair of shots.

    Attributes:
        shots: Shot ids, ``(n,)``.
        distance: ``(n, n)`` float64, NaN where the pair has no distance; 0 on the
            diagonal of every fingerprinted shot.
        n_common_times: ``(n, n)`` int32 grid times present in both shots.
    """

    shots: np.ndarray
    distance: np.ndarray
    n_common_times: np.ndarray

    def index_of(self, shot_ids: Iterable[str]) -> np.ndarray:
        lookup = {s: i for i, s in enumerate(self.shots.tolist())}
        return np.array([lookup[s] for s in shot_ids], dtype=int)


def pairwise_distance(fingerprints: Fingerprints, scales, min_common: int = MIN_COMMON_TIMES,
                      min_common_fraction: float = MIN_COMMON_FRACTION, chunk: int = 512) -> PairwiseDistance:
    """Root-mean-square scaled difference between every pair of shots.

    The sums over the entries present in both shots are formed with matrix products
    on the NaN-masked fingerprints, in row chunks, so no pair loop runs: with ``x`` the
    scaled values set to 0 where missing and ``m`` the presence mask,
    ``sum_k m_ik m_jk (x_ik - x_jk)^2 = (x_i^2 . m_j) + (m_i . x_j^2) - 2 (x_i . x_j)``.

    Args:
        fingerprints: Fingerprints of the shots.
        scales: Per-signal scales, ``(n_signals,)``.
        min_common: Fewest shared grid times for a pair to have a distance.
        min_common_fraction: Shared grid times must also be at least this fraction of
            the grid times of the shorter shot.
        chunk: Rows per block of the matrix products.
    """
    s = np.asarray(scales, dtype=np.float64)
    if s.shape != (len(fingerprints.signals),):
        raise ValueError("one scale per signal is required")
    n, n_times, n_signals = fingerprints.values.shape
    x = fingerprints.values / s
    # Centring each signal leaves every difference unchanged and keeps the squared terms
    # small, which limits the cancellation in ``a^2 + b^2 - 2ab`` for near-identical shots.
    x = (x - np.nanmean(x.reshape(-1, n_signals), axis=0)).reshape(n, n_times * n_signals)
    m = np.isfinite(x)
    x0 = np.where(m, x, 0.0)
    x0_sq = x0 * x0
    mf = m.astype(np.float64)
    present = fingerprints.present.astype(np.float64)
    n_present = present.sum(axis=1)
    distance = np.full((n, n), np.nan)
    common = np.zeros((n, n), dtype=np.int32)
    for start in range(0, n, chunk):
        rows = slice(start, start + chunk)
        sq = x0_sq[rows] @ mf.T + mf[rows] @ x0_sq.T - 2.0 * (x0[rows] @ x0.T)
        count = mf[rows] @ mf.T
        shared = present[rows] @ present.T
        shorter = np.minimum(n_present[rows][:, None], n_present[None, :])
        defined = (shared >= min_common) & (shared >= min_common_fraction * shorter) & (count > 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            d = np.sqrt(np.maximum(sq, 0.0) / count)
        distance[rows] = np.where(defined, d, np.nan)
        common[rows] = shared.astype(np.int32)
    # A fingerprinted shot is at distance 0 from itself, exactly.
    diag = n_present >= min_common
    distance[np.arange(n)[diag], np.arange(n)[diag]] = 0.0
    return PairwiseDistance(shots=fingerprints.shots.copy(), distance=distance, n_common_times=common)


def direct_distance(a, b, scales, min_common: int = MIN_COMMON_TIMES,
                    min_common_fraction: float = MIN_COMMON_FRACTION) -> float:
    """Distance between two fingerprints ``(n_times, n_signals)`` by a direct computation.

    The reference the matrix form is tested against.
    """
    a = np.asarray(a, dtype=np.float64) / np.asarray(scales, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64) / np.asarray(scales, dtype=np.float64)
    both = np.isfinite(a) & np.isfinite(b)
    shared = both.any(axis=1).sum()
    shorter = min(np.isfinite(a).any(axis=1).sum(), np.isfinite(b).any(axis=1).sum())
    if shared < min_common or shared < min_common_fraction * shorter or not both.any():
        return np.nan
    d = a[both] - b[both]
    return float(np.sqrt(np.mean(d * d)))


def nearest_neighbours(distance: np.ndarray, k: int, rows=None, columns=None) -> tuple[np.ndarray, np.ndarray]:
    """The ``k`` nearest shots of each row among the column shots, excluding itself.

    Args:
        distance: ``(n, n)`` distances with NaN for undefined pairs.
        k: Neighbours per row.
        rows: Row indices to report; all when None.
        columns: Candidate indices; all when None.

    Returns:
        ``(index, distance)`` each ``(len(rows), k)``; index is -1 and distance NaN
        where fewer than ``k`` defined neighbours exist. Ties keep index order.
    """
    d = np.asarray(distance, dtype=np.float64)
    n = d.shape[0]
    rows = np.arange(n) if rows is None else np.asarray(rows, dtype=int)
    columns = np.arange(n) if columns is None else np.asarray(columns, dtype=int)
    sub = d[np.ix_(rows, columns)].copy()
    self_hit = rows[:, None] == columns[None, :]
    sub[self_hit] = np.nan
    sub = np.where(np.isfinite(sub), sub, np.inf)
    k = min(k, columns.size)
    order = np.argsort(sub, axis=1, kind="stable")[:, :k]
    best = np.take_along_axis(sub, order, axis=1)
    idx = columns[order]
    idx = np.where(np.isfinite(best), idx, -1)
    best = np.where(np.isfinite(best), best, np.nan)
    return idx, best


def group_shots(distance: np.ndarray, threshold: float) -> np.ndarray:
    """Complete-linkage groups: every pair inside a group is below ``threshold``.

    Undefined pairs are given ``UNDEFINED_DISTANCE`` and so never merge; a shot with
    no defined distance to any other shot forms its own group.

    Returns:
        Group labels ``(n,)`` from 0, numbered by first appearance.
    """
    d = np.asarray(distance, dtype=np.float64)
    n = d.shape[0]
    if n == 0:
        return np.zeros(0, dtype=int)
    if n == 1:
        return np.zeros(1, dtype=int)
    full = np.where(np.isfinite(d), d, UNDEFINED_DISTANCE)
    full = 0.5 * (full + full.T)
    np.fill_diagonal(full, 0.0)
    condensed = squareform(full, checks=False)
    tree = linkage(condensed, method="complete")
    # fcluster merges at distances <= t; a near-repeat pair is strictly below the threshold.
    labels = fcluster(tree, t=np.nextafter(threshold, 0.0), criterion="distance")
    _, first = np.unique(labels, return_index=True)
    relabel = {labels[i]: rank for rank, i in enumerate(sorted(first))}
    return np.array([relabel[x] for x in labels], dtype=int)


def group_column(threshold: float) -> str:
    """Name of the group column of a threshold in ``shot_groups.parquet``, to two decimals:
    0.05 gives ``group_005``, 0.10 ``group_010``, 0.20 ``group_020``."""
    return f"group_{int(round(threshold * 100)):03d}"


def has_near_repeat_in(distance: PairwiseDistance, shot_ids: Iterable[str], reference_ids: Iterable[str],
                       threshold: float) -> np.ndarray:
    """Whether each shot has a near-repeat (distance below ``threshold``) among the
    reference shots, itself excluded. Returns a bool array aligned with ``shot_ids``."""
    rows = distance.index_of(shot_ids)
    cols = distance.index_of(reference_ids)
    sub = distance.distance[np.ix_(rows, cols)].copy()
    sub[rows[:, None] == cols[None, :]] = np.nan
    return np.nan_to_num(sub, nan=np.inf).min(axis=1) < threshold if cols.size else np.zeros(rows.size, dtype=bool)


# -------------------------------------------------------------- input distributions (B8a)
def flat_top_medians(frames: pd.DataFrame, quantities: Sequence[str], level: float = FLAT_TOP_LEVEL) -> pd.DataFrame:
    """Per shot, the median over flat-top frames (``ip_over_peak`` at or above ``level``)
    of each quantity. Shots without a flat-top frame get NaN. Indexed by shot id."""
    flat = frames[frames["ip_over_peak"] >= level]
    out = flat.groupby(flat["shot"].astype(str), sort=True)[list(quantities)].median()
    all_shots = sorted(frames["shot"].astype(str).unique())
    return out.reindex(all_shots)


def compare_distributions(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Two-sample comparison of each column between training and test shots.

    Returns:
        One row per column: ``ks`` (Kolmogorov-Smirnov statistic), ``p_value``,
        ``median_diff_iqr`` (test median minus training median over the training
        interquartile range), ``n_train``, ``n_test``; largest ``ks`` first.
    """
    rows = []
    for col in train.columns:
        a = train[col].to_numpy(dtype=np.float64)
        b = test[col].to_numpy(dtype=np.float64)
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        if a.size < 2 or b.size < 2:
            rows.append({"quantity": col, "ks": np.nan, "p_value": np.nan, "median_diff_iqr": np.nan,
                         "n_train": int(a.size), "n_test": int(b.size)})
            continue
        ks = ks_2samp(a, b)
        iqr = np.percentile(a, 75) - np.percentile(a, 25)
        rows.append({"quantity": col, "ks": float(ks.statistic), "p_value": float(ks.pvalue),
                     "median_diff_iqr": float((np.median(b) - np.median(a)) / iqr) if iqr > 0 else np.nan,
                     "n_train": int(a.size), "n_test": int(b.size)})
    return pd.DataFrame(rows).sort_values("ks", ascending=False, kind="stable").reset_index(drop=True)
