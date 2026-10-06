"""Thomson scattering validity rules and per-slice summaries.

Shared by the quality checks (check 0.7) and the frame table, so that both use one
definition of a valid channel and a valid time slice:

- a channel is valid in a time slice when both Te and ne are present under the
  machine's missing-value convention (exact zero on DIII-D, NaN on MAST);
- a slice is a valid profile when at least ``VALID_PROFILE_FRACTION`` of its
  channels are valid.

The summaries here are provisional descriptors used to test the time alignment of
Thomson against the EFIT frames. Profile descriptors on a normalized radius replace
them in a later phase.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fusion_eq.resample import linear_interpolate, nearest_distance_ms, nearest_index

# A Thomson time slice counts as a valid profile when at least this fraction of
# its channels has both Te and ne present.
VALID_PROFILE_FRACTION = 0.5

SYSTEMS = ("core", "edge")

# Electron pressure p_e = n_e * T_e * e, with T_e in eV and n_e in m^-3, gives Pa.
ELEMENTARY_CHARGE_J_PER_EV = 1.602e-19
PA_PER_KPA = 1000.0

# Percentile across valid channels used as the "peak" of a core profile; it is
# robust to a single outlying channel where the maximum is not.
PEAK_PERCENTILE = 90.0

# Summary names per system, without the ``ts_<system>_`` prefix of the frame table.
SUMMARY_NAMES = {
    "core": ("te_peak", "ne_peak", "pe_peak", "pe_mean"),
    "edge": ("te_mean", "ne_mean", "pe_mean"),
}


def missing_mask(values, convention: str) -> np.ndarray:
    """Marks missing Thomson points under a machine's convention.

    Args:
        values: Te or ne array, any shape.
        convention: ``"zero"`` (exact zero or non-finite is missing) or ``"nan"``
            (non-finite is missing).

    Raises:
        ValueError: for an unknown convention.
    """
    if convention not in ("zero", "nan"):
        raise ValueError(f"unknown Thomson missing-value convention {convention!r}")
    v = np.asarray(values, dtype=np.float64)
    missing = ~np.isfinite(v)
    if convention == "zero":
        missing |= v == 0
    return missing


def valid_channel_mask(te, ne, convention: str) -> np.ndarray:
    """Boolean ``(time, channel)`` mask of the points where both Te and ne are present."""
    te = np.asarray(te, dtype=np.float64)
    ne = np.asarray(ne, dtype=np.float64)
    if te.shape != ne.shape:
        raise ValueError(f"Te and ne differ in shape: {te.shape} versus {ne.shape}")
    return ~missing_mask(te, convention) & ~missing_mask(ne, convention)


def valid_slice_mask(valid_channels: np.ndarray, fraction: float = VALID_PROFILE_FRACTION) -> np.ndarray:
    """Boolean ``(time,)`` mask of the slices with enough valid channels to be a profile.

    Args:
        valid_channels: ``(time, channel)`` mask from ``valid_channel_mask``.
        fraction: Required fraction of valid channels; the threshold is the ceiling
            of ``fraction * n_channels``. With zero channels no slice is valid.
    """
    valid_channels = np.asarray(valid_channels, dtype=bool)
    if valid_channels.ndim != 2:
        raise ValueError("valid_channels must be a (time, channel) array")
    n_channels = valid_channels.shape[1]
    if n_channels == 0:
        return np.zeros(valid_channels.shape[0], dtype=bool)
    threshold = int(np.ceil(fraction * n_channels))
    return valid_channels.sum(axis=1) >= max(threshold, 1)


def pressure_kpa(te_ev, ne_m3) -> np.ndarray:
    """Electron pressure in kPa from Te in eV and ne in m^-3."""
    return np.asarray(ne_m3, dtype=np.float64) * np.asarray(te_ev, dtype=np.float64) \
        * ELEMENTARY_CHARGE_J_PER_EV / PA_PER_KPA


@dataclass
class SliceSummaries:
    """Summaries of the valid Thomson slices of one system of one shot.

    Attributes:
        system: ``"core"`` or ``"edge"``.
        times_ms: Times of the valid slices, sorted, in ms.
        n_valid: Valid channels in each valid slice.
        values: Summary name (see ``SUMMARY_NAMES``) to one value per valid slice.
            Units: Te in eV, ne in m^-3, pe in kPa.
    """

    system: str
    times_ms: np.ndarray
    n_valid: np.ndarray
    values: dict[str, np.ndarray]

    @property
    def n_slices(self) -> int:
        return int(self.times_ms.size)


def slice_summaries(times_ms, te, ne, convention: str, system: str) -> SliceSummaries:
    """Summaries per valid slice, from the valid channels only.

    Args:
        times_ms: Slice times in ms, length T.
        te: Electron temperature ``(T, C)`` in eV.
        ne: Electron density ``(T, C)`` in m^-3.
        convention: Missing-value convention, ``"zero"`` or ``"nan"``.
        system: ``"core"`` (peak and mean pressure descriptors) or ``"edge"`` (means).

    Returns:
        The summaries of the valid slices. Slices with non-finite times are dropped.
        With no valid slice every array is empty.

    Raises:
        ValueError: when the arrays are not rectangular ``(T, C)`` matching ``times_ms``.
    """
    if system not in SUMMARY_NAMES:
        raise ValueError(f"unknown Thomson system {system!r}")
    t = np.asarray(times_ms, dtype=np.float64).ravel()
    te = np.asarray(te, dtype=np.float64)
    ne = np.asarray(ne, dtype=np.float64)
    if te.ndim != 2 or te.shape != ne.shape or te.shape[0] != t.size:
        raise ValueError(
            f"{system}: Te {te.shape} and ne {ne.shape} must be (T, C) with T = {t.size}")
    valid = valid_channel_mask(te, ne, convention)
    keep = valid_slice_mask(valid) & np.isfinite(t)
    order = np.argsort(t[keep], kind="stable")
    rows = np.flatnonzero(keep)[order]
    names = SUMMARY_NAMES[system]
    if rows.size == 0:
        empty = np.zeros(0, dtype=np.float64)
        return SliceSummaries(system, empty, np.zeros(0, dtype=np.int64),
                              {name: empty.copy() for name in names})

    te_v = np.where(valid[rows], te[rows], np.nan)
    ne_v = np.where(valid[rows], ne[rows], np.nan)
    pe_v = pressure_kpa(te_v, ne_v)
    values: dict[str, np.ndarray] = {}
    if system == "core":
        values["te_peak"] = np.nanpercentile(te_v, PEAK_PERCENTILE, axis=1)
        values["ne_peak"] = np.nanpercentile(ne_v, PEAK_PERCENTILE, axis=1)
        values["pe_peak"] = np.nanpercentile(pe_v, PEAK_PERCENTILE, axis=1)
        values["pe_mean"] = np.nanmean(pe_v, axis=1)
    else:
        values["te_mean"] = np.nanmean(te_v, axis=1)
        values["ne_mean"] = np.nanmean(ne_v, axis=1)
        values["pe_mean"] = np.nanmean(pe_v, axis=1)
    return SliceSummaries(system, t[rows], valid[rows].sum(axis=1).astype(np.int64), values)


def bracket_gap_ms(sorted_times_ms: np.ndarray, target_ms: np.ndarray) -> np.ndarray:
    """Time between the last slice at or before each target and the first slice after it.

    Returns NaN where the target is not bracketed (before the first or at or after
    the last slice).
    """
    target = np.asarray(target_ms, dtype=np.float64)
    out = np.full(target.shape, np.nan)
    if sorted_times_ms.size < 2:
        return out
    idx = np.searchsorted(sorted_times_ms, target, side="right")
    inside = (idx >= 1) & (idx < sorted_times_ms.size)
    out[inside] = sorted_times_ms[idx[inside]] - sorted_times_ms[idx[inside] - 1]
    return out


def frame_columns(summaries: SliceSummaries, frame_times_ms, tol_ms: float,
                  prefix: str | None = None) -> dict[str, np.ndarray]:
    """Brings slice summaries onto frame times in the two forms the frame table stores.

    Args:
        summaries: Output of ``slice_summaries``.
        frame_times_ms: EFIT frame times in ms.
        tol_ms: A frame is marked valid when the nearest valid slice is within this
            distance.
        prefix: Column prefix; ``ts_<system>_`` when None.

    Returns:
        Column name to one value per frame: for each summary ``<s>``,
        ``<prefix><s>__interp`` (linear interpolation between the valid slices either
        side, end values held outside) and ``<prefix><s>__nearest`` (value at the
        nearest valid slice); ``<prefix>n_valid`` (valid channels in the nearest
        slice), ``<prefix>dt_ms`` (distance to the nearest slice), ``<prefix>gap_ms``
        (``bracket_gap_ms``) and ``<prefix>valid`` (bool). With no valid slice the
        numeric columns are NaN and ``valid`` is False.
    """
    prefix = f"ts_{summaries.system}_" if prefix is None else prefix
    frames = np.asarray(frame_times_ms, dtype=np.float64)
    t = summaries.times_ms
    out: dict[str, np.ndarray] = {}
    if t.size == 0:
        nan = np.full(frames.shape, np.nan)
        for name in summaries.values:
            out[f"{prefix}{name}__interp"] = nan.copy()
            out[f"{prefix}{name}__nearest"] = nan.copy()
        out[f"{prefix}n_valid"] = nan.copy()
        out[f"{prefix}dt_ms"] = nan.copy()
        out[f"{prefix}gap_ms"] = nan.copy()
        out[f"{prefix}valid"] = np.zeros(frames.shape, dtype=bool)
        return out
    idx = nearest_index(t, frames)
    for name, values in summaries.values.items():
        out[f"{prefix}{name}__interp"] = linear_interpolate(t, values, frames, hold_ends=True)
        out[f"{prefix}{name}__nearest"] = values[idx]
    out[f"{prefix}n_valid"] = summaries.n_valid[idx].astype(np.float64)
    dt = nearest_distance_ms(frames, t)
    out[f"{prefix}dt_ms"] = dt
    out[f"{prefix}gap_ms"] = bracket_gap_ms(t, frames)
    out[f"{prefix}valid"] = dt <= tol_ms
    return out
