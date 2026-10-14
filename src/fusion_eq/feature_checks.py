"""Calculations of the feature checks X1 to X16 (``docs/FEATURES.md``) and of the stretch-term
diagnostic for feature version 2 (notebook 06, section 6b).

The functions take arrays and tables and read no file: ``scripts/profile_features.py``
(X1 to X15), ``scripts/feature_linear_check.py`` (X16) and ``notebooks/06_features.ipynb``
read the feature tables, the frame table and the reports and call them.

Nothing is fitted on test inputs. Every comparison across splits takes the DIII-D
training frames (the ``train`` list of the split file) as the reference and reports the
other splits against it: the training 1st to 99th percentile band and full range, the
training median and interquartile range.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from fusion_eq.coil_flux import MU0
from fusion_eq.coil_pattern import TERM_NAMES, Layout, term_operator
from fusion_eq.coil_scales import FLAT_TOP_LEVEL, REFERENCE_PHASES, mirror_pairs
from fusion_eq.features import FeatureConfig, check_reads, frame_current_column
from fusion_eq.machines import MachineConfig, machine_config

A_PER_KA = 1000.0
STATUSES = ("pass", "fail", "review")
PHASES = ("all", *REFERENCE_PHASES)
# Splits compared against the training frames, in the order that sets the X10 seeds.
COMPARED_SPLITS = ("validation", "diii_d_public_test", "mast_public_test")
# Rule of a robust-scaled column without a scale group (in-memory variants).
ROBUST_RULE = {"scale": "robust", "scale_group": None, "scale_with": "source", "transform": None}
# Percentile band of the training frames that the range checks compare against.
BAND_PERCENTILES = (1.0, 99.0)
# A scaled value beyond this magnitude counts as an extreme (check X11).
EXTREME_LIMIT = 10.0
# X5: largest 95th percentile of the swap difference, as a share of the pattern's range,
# that passes without review. F51 expects about 0.02 from the asymmetry of the shipped coils.
SWAP_DIFFERENCE_PASS = 0.05


# ------------------------------------------------------------------------------ checks
@dataclass(frozen=True)
class Check:
    """One row of the check table.

    Attributes:
        id: Check id (``X1`` to ``X16``).
        title: Short name of the check.
        status: ``pass``, ``fail`` or ``review``.
        summary: One line with the numbers behind the status.
    """

    id: str
    title: str
    status: str
    summary: str

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"check {self.id}: status {self.status!r} is not one of {STATUSES}")

    def to_dict(self) -> dict:
        return asdict(self)


def check_table_markdown(checks: Iterable[Check]) -> str:
    """The check table as Markdown, one row per check."""
    lines = ["| Id | Check | Status | Summary |", "| --- | --- | --- | --- |"]
    for c in checks:
        lines.append(f"| {c.id} | {c.title} | {c.status} | {c.summary.replace('|', '/')} |")
    return "\n".join(lines)


def _markdown_cell(value, digits: int) -> str:
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return "nan" if not np.isfinite(value) else f"{value:.{digits}g}"
    return str(value).replace("|", "/")


def markdown_table(table: pd.DataFrame, digits: int = 4) -> str:
    """A table as Markdown, floats to ``digits`` significant digits, the index left out
    (reset it first to keep it)."""
    lines = ["| " + " | ".join(str(c) for c in table.columns) + " |", "| " + " | ".join("---" for _ in table.columns) + " |"]
    for row in table.itertuples(index=False):
        lines.append("| " + " | ".join(_markdown_cell(v, digits) for v in row) + " |")
    return "\n".join(lines)


# ------------------------------------------------------------------------------ phases
def frames_per_shot(shots) -> np.ndarray:
    """Number of rows of each row's shot, for rows that hold every frame of their shots."""
    s = pd.Series(np.asarray(shots))
    return s.groupby(s).transform("size").to_numpy()


def frame_phases(ip_over_peak, frame, n_frames, flat_top: float = FLAT_TOP_LEVEL) -> np.ndarray:
    """Phase of every frame, the rule of ``coil_scales.phase_of`` applied to arrays:
    ``flat_top`` at or above ``flat_top`` of the shot's peak Ip, otherwise ``rising`` in the
    first half of the shot and ``late`` in the second."""
    over = np.asarray(ip_over_peak, dtype=np.float64)
    frame = np.asarray(frame, dtype=np.float64)
    n = np.asarray(n_frames, dtype=np.float64)
    flat = np.isfinite(over) & (over >= flat_top)
    return np.where(flat, "flat_top", np.where(frame < n / 2.0, "rising", "late")).astype(object)


def profile_frame_columns(machine: str | MachineConfig) -> list[str]:
    """Frame-table columns the checks read besides the features: the frame time, Ip, its
    50 ms slope, the Ip rise and the coil currents (``__mean``). All pass the label guard."""
    cfg = machine if isinstance(machine, MachineConfig) else machine_config(machine)
    columns = ["time_ms", "ip__mean", "ip__rate50", "ip_rise_ms", "sol__mean",
               *[f"{c}__mean" for c in cfg.shaping_coil_columns]]
    check_reads(columns, cfg)
    return columns


def join_frame_columns(features: pd.DataFrame, frames: pd.DataFrame) -> pd.DataFrame:
    """Feature rows of one split with frame-table columns beside them, plus ``n_frames`` and
    ``phase`` (from ``ip_frac``, the frame index and the frames per shot).

    Args:
        features: A split's feature table (every frame of its shots, file order).
        frames: Frame-table rows of the same split with ``shot``, ``frame`` and the extra
            columns, in table order.

    Raises:
        ValueError: when the two tables differ in rows or keys (check X1), or share a
            column other than the keys.
    """
    same = len(features) == len(frames) and \
        np.array_equal(features["shot"].astype(str).to_numpy(), frames["shot"].astype(str).to_numpy()) and \
        np.array_equal(features["frame"].to_numpy(), frames["frame"].to_numpy())
    if not same:
        raise ValueError("feature rows and frame-table rows differ (X1); rebuild the features")
    extra = [c for c in frames.columns if c not in ("shot", "frame")]
    overlap = [c for c in extra if c in features.columns]
    if overlap:
        raise ValueError(f"columns in both tables: {overlap}")
    out = pd.concat([features.reset_index(drop=True), frames[extra].reset_index(drop=True)], axis=1)
    out["shot"] = out["shot"].astype(str)
    out["n_frames"] = frames_per_shot(out["shot"].to_numpy())
    out["phase"] = frame_phases(out["ip_frac"], out["frame"], out["n_frames"])
    return out


# -------------------------------------------------------------------------- completeness
def finite_shares(table: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    """Share of finite values per column (check X3)."""
    values = table[list(columns)].to_numpy(dtype=np.float64)
    shares = np.isfinite(values).mean(axis=0) if len(table) else np.full(len(columns), np.nan)
    return pd.Series(shares, index=list(columns))


def low_ip_counts(ip_ka, min_abs_ip_ka: float) -> dict:
    """Frames whose Ip cannot be divided by (check X3): missing, or below ``min_abs_ip_ka``
    in magnitude (kA). Every feature divided by Ip is NaN on them."""
    ip = np.asarray(ip_ka, dtype=np.float64)
    missing = ~np.isfinite(ip)
    below = np.isfinite(ip) & (np.abs(ip) < min_abs_ip_ka)
    n = int(ip.size)
    return {"n_frames": n, "n_missing_ip": int(missing.sum()), "n_below_min": int(below.sum()),
            "n_affected": int((missing | below).sum()), "share_affected": float((missing | below).mean()) if n else np.nan}


# -------------------------------------------------------------------------------- ranges
def reference_statistics(values) -> dict:
    """Percentiles 1, 25, 50, 75 and 99, minimum, maximum and count of the finite values."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"n": 0, **{k: np.nan for k in ("q01", "q25", "q50", "q75", "q99", "min", "max")}}
    q01, q25, q50, q75, q99 = np.percentile(v, [BAND_PERCENTILES[0], 25.0, 50.0, 75.0, BAND_PERCENTILES[1]])
    return {"n": int(v.size), "q01": float(q01), "q25": float(q25), "q50": float(q50), "q75": float(q75),
            "q99": float(q99), "min": float(v.min()), "max": float(v.max())}


def compare_ranges(reference: pd.DataFrame, splits: Mapping[str, pd.DataFrame], columns: Sequence[str],
                   phase: str = "all") -> pd.DataFrame:
    """Each split's values of each column against the reference frames (checks X8, X9).

    Args:
        reference: Feature rows of the reference (DIII-D training) frames.
        splits: Name to the feature rows of a compared split.
        columns: Feature columns.
        phase: Label written in the ``phase`` column.

    Returns:
        One row per column and split: the reference band (``ref_q01``, ``ref_q99``), range
        (``ref_min``, ``ref_max``), median and interquartile range; the split's count of
        finite values, its own median and band; ``share_outside_band`` and
        ``share_outside_range`` over the split's finite values; ``median_shift_iqr``, the
        split's median minus the reference median in reference interquartile ranges.
    """
    rows = []
    for column in columns:
        ref = reference_statistics(reference[column].to_numpy(dtype=np.float64))
        iqr = ref["q75"] - ref["q25"]
        for name, table in splits.items():
            v = table[column].to_numpy(dtype=np.float64)
            v = v[np.isfinite(v)]
            own = reference_statistics(v)
            row = {"feature": column, "phase": phase, "split": name, "n_finite": own["n"],
                   "median": own["q50"], "q01": own["q01"], "q99": own["q99"],
                   "ref_n": ref["n"], "ref_q01": ref["q01"], "ref_q99": ref["q99"], "ref_min": ref["min"],
                   "ref_max": ref["max"], "ref_median": ref["q50"], "ref_iqr": iqr,
                   "share_outside_band": np.nan, "share_outside_range": np.nan, "median_shift_iqr": np.nan}
            if v.size and ref["n"]:
                row["share_outside_band"] = float(((v < ref["q01"]) | (v > ref["q99"])).mean())
                row["share_outside_range"] = float(((v < ref["min"]) | (v > ref["max"])).mean())
                row["median_shift_iqr"] = float((own["q50"] - ref["q50"]) / iqr) if iqr > 0 else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def compare_ranges_by_phase(reference: pd.DataFrame, reference_phase, splits: Mapping[str, pd.DataFrame],
                            split_phases: Mapping[str, np.ndarray], columns: Sequence[str]) -> pd.DataFrame:
    """``compare_ranges`` over all frames and within each phase, the reference restricted to
    the same phase (check X9)."""
    reference_phase = np.asarray(reference_phase)
    parts = [compare_ranges(reference, splits, columns, "all")]
    for phase in REFERENCE_PHASES:
        ref = reference.loc[reference_phase == phase]
        sub = {k: t.loc[np.asarray(split_phases[k]) == phase] for k, t in splits.items()}
        parts.append(compare_ranges(ref, sub, columns, phase))
    return pd.concat(parts, ignore_index=True)


def extreme_shares(scaled: np.ndarray, columns: Sequence[str], limit: float = EXTREME_LIMIT) -> pd.Series:
    """Share of scaled values beyond ``limit`` in magnitude, per column (check X11)."""
    x = np.asarray(scaled, dtype=np.float64)
    shares = (np.abs(x) > limit).mean(axis=0) if x.shape[0] else np.full(len(columns), np.nan)
    return pd.Series(shares, index=list(columns))


# --------------------------------------------------------------------- combined distance
def sample_rows(n_rows: int, n: int, seed: int) -> np.ndarray:
    """Sorted indices of ``n`` rows drawn without replacement with a fixed seed; every row
    when ``n`` is at least ``n_rows``."""
    if n >= n_rows:
        return np.arange(n_rows)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_rows, size=n, replace=False))


def nearest_neighbours(reference: np.ndarray, queries: np.ndarray, chunk: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Euclidean distance from each query row to its nearest reference row, and that row's
    index, by exhaustive search in chunks of queries (check X10). Ties go to the lowest index.

    Raises:
        ValueError: when the two arrays differ in columns or the reference is empty.
    """
    ref = np.asarray(reference, dtype=np.float64)
    q = np.asarray(queries, dtype=np.float64)
    if ref.ndim != 2 or q.ndim != 2 or ref.shape[1] != q.shape[1]:
        raise ValueError("reference and queries must be 2-D with the same columns")
    if ref.shape[0] == 0:
        raise ValueError("empty reference")
    ref_sq = (ref * ref).sum(axis=1)
    index = np.empty(q.shape[0], dtype=np.int64)
    for start in range(0, q.shape[0], chunk):
        block = q[start:start + chunk]
        d2 = (block * block).sum(axis=1)[:, None] + ref_sq[None, :] - 2.0 * block @ ref.T
        index[start:start + chunk] = np.argmin(d2, axis=1)
    # The expanded form loses precision for near neighbours; the distance is taken directly.
    distance = np.sqrt(((q - ref[index]) ** 2).sum(axis=1))
    return distance, index


def group_shares(reference: np.ndarray, queries: np.ndarray, index: np.ndarray,
                 groups: Mapping[str, Sequence[int]]) -> dict[str, np.ndarray]:
    """Share of each query's squared distance to its neighbour that each column group
    carries; NaN where the distance is zero."""
    diff2 = (np.asarray(queries, dtype=np.float64) - np.asarray(reference, dtype=np.float64)[index]) ** 2
    total = diff2.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return {g: np.where(total > 0, diff2[:, list(cols)].sum(axis=1) / total, np.nan) for g, cols in groups.items()}


def x10_samples(n_train: int, n_compared: Mapping[str, int], n_train_frames: int, n_eval_frames: int,
                seed: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Rows drawn by check X10: ``n_train_frames`` training rows with ``seed``, and
    ``n_eval_frames`` rows of the k-th compared split (in the order of ``n_compared``) with
    ``seed + 1 + k``.

    Args:
        n_train: Rows of the training table.
        n_compared: Compared split to its number of rows.
    """
    train_rows = sample_rows(n_train, n_train_frames, seed)
    return train_rows, {split: sample_rows(n, n_eval_frames, seed + 1 + k) for k, (split, n) in enumerate(n_compared.items())}


def column_groups(columns: Sequence[str], group_of: Mapping[str, str]) -> dict[str, list[int]]:
    """Group to the positions of its columns in ``columns``, groups in order of appearance."""
    groups: dict[str, list[int]] = {}
    for j, c in enumerate(columns):
        groups.setdefault(group_of[c], []).append(j)
    return groups


def combined_distance(reference: np.ndarray, queries: Mapping[str, np.ndarray],
                      groups: Mapping[str, Sequence[int]]) -> pd.DataFrame:
    """Nearest-neighbour distance of every query row of each split to the reference rows
    (check X10).

    Returns:
        One row per query row, splits in the order of ``queries``: ``split``, ``distance``,
        ``nearest`` (row of ``reference``) and ``share_<group>`` per group.
    """
    parts = []
    for split, q in queries.items():
        distance, nearest = nearest_neighbours(reference, q)
        part = pd.DataFrame({"split": split, "distance": distance, "nearest": nearest})
        for g, v in group_shares(reference, q, nearest, groups).items():
            part[f"share_{g}"] = v
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def group_share_medians(table: pd.DataFrame, keys: Sequence[str] = ("set", "split")) -> pd.DataFrame:
    """Median of each ``share_<group>`` column per key, keys in order of appearance."""
    columns = [c for c in table.columns if c.startswith("share_")]
    return table.groupby(list(keys), sort=False)[columns].median().reset_index()


def distance_summary(table: pd.DataFrame, reference_split: str = "validation", percentiles=(50.0, 90.0, 99.0)) -> pd.DataFrame:
    """Percentiles of the nearest-neighbour distance per set and split, and the share of each
    split's frames beyond the ``reference_split`` 99th percentile of the same set.

    Args:
        table: Rows with ``set``, ``split`` and ``distance``.
    """
    rows = []
    for set_name, g in table.groupby("set", sort=False):
        ref = g.loc[g["split"] == reference_split, "distance"].to_numpy()
        limit = float(np.percentile(ref, 99.0)) if ref.size else np.nan
        for split, h in g.groupby("split", sort=False):
            d = h["distance"].to_numpy(dtype=np.float64)
            row = {"set": set_name, "split": split, "n": int(d.size), "max": float(d.max()) if d.size else np.nan,
                   f"{reference_split}_p99": limit,
                   f"share_beyond_{reference_split}_p99": float((d > limit).mean()) if d.size and np.isfinite(limit) else np.nan}
            for p in percentiles:
                row[f"p{p:g}"] = float(np.percentile(d, p)) if d.size else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- correlation
def correlation_long(table: pd.DataFrame, columns: Sequence[str], kind: str) -> pd.DataFrame:
    """Pearson correlation of every pair of columns over the rows where all are finite, in
    long form: ``kind``, ``a``, ``b``, ``r`` (check X12)."""
    x = table[list(columns)].to_numpy(dtype=np.float64)
    x = x[np.isfinite(x).all(axis=1)]
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.corrcoef(x, rowvar=False) if x.shape[0] > 1 else np.full((len(columns), len(columns)), np.nan)
    r = np.atleast_2d(r)
    a, b = np.meshgrid(np.arange(len(columns)), np.arange(len(columns)), indexing="ij")
    names = np.asarray(columns, dtype=object)
    return pd.DataFrame({"kind": kind, "a": names[a.ravel()], "b": names[b.ravel()], "r": r.ravel(), "n_rows": int(x.shape[0])})


def correlation_summary(long: pd.DataFrame, ip_columns: Sequence[str] = ()) -> dict:
    """Largest and median absolute correlation between distinct columns of one ``kind``,
    the Ip columns excluded, and for each Ip column the largest and median absolute
    correlation of the other columns with it."""
    ip_columns = list(ip_columns)
    pairs = long.loc[long["a"] < long["b"]]
    off = pairs.loc[~pairs["a"].isin(ip_columns) & ~pairs["b"].isin(ip_columns)]
    r = off["r"].abs()
    out = {"n_pairs": int(len(off)), "median_abs_r": float(r.median()) if len(r) else np.nan,
           "max_abs_r": float(r.max()) if len(r) else np.nan, "max_pair": None, "with_ip": {}}
    if len(r) and r.notna().any():
        best = off.loc[r.idxmax()]
        out["max_pair"] = [str(best["a"]), str(best["b"])]
    for ip in ip_columns:
        rows = long.loc[(long["a"] == ip) & ~long["b"].isin(ip_columns)]
        ri = rows["r"].abs()
        top = rows.loc[ri.idxmax(), "b"] if len(ri) and ri.notna().any() else None
        out["with_ip"][ip] = {"median_abs_r": float(ri.median()) if len(ri) else np.nan,
                              "max_abs_r": float(ri.max()) if len(ri) else np.nan, "max_column": top}
    return out


# ------------------------------------------------------------------------ Ip direction
def direction_comparison(table: pd.DataFrame, direction, columns: Sequence[str]) -> pd.DataFrame:
    """Distributions of columns on positive-Ip against negative-Ip frames (check X6).

    Returns:
        One row per column: counts, medians and interquartile ranges of each direction,
        ``median_shift_iqr`` (negative median minus positive median in positive
        interquartile ranges) and ``share_negative_in_positive_band`` (share of the
        negative-Ip values inside the positive 1st to 99th percentile band).
    """
    direction = np.asarray(direction)
    rows = []
    for column in columns:
        v = table[column].to_numpy(dtype=np.float64)
        pos = reference_statistics(v[direction == 1])
        neg_v = v[(direction == -1) & np.isfinite(v)]
        neg = reference_statistics(neg_v)
        iqr = pos["q75"] - pos["q25"]
        rows.append({"feature": column, "n_positive": pos["n"], "n_negative": neg["n"],
                     "median_positive": pos["q50"], "median_negative": neg["q50"], "iqr_positive": iqr,
                     "iqr_negative": neg["q75"] - neg["q25"],
                     "median_shift_iqr": (neg["q50"] - pos["q50"]) / iqr if iqr > 0 else np.nan,
                     "share_negative_in_positive_band": float(((neg_v >= pos["q01"]) & (neg_v <= pos["q99"])).mean())
                     if neg_v.size and pos["n"] else np.nan})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------- stretch terms
def vacuum_residual(table: pd.DataFrame, machine: str | MachineConfig, r0_m: float | None = None,
                    a0_m: float | None = None) -> np.ndarray:
    """``coil_c20 + coil_c02 - (a0 / (2 R0)) coil_c10`` per row, without units
    (diagnostic for feature version 2).

    A current-free field obeys the Grad-Shafranov equation without source,
    ``psi_RR - psi_R / R + psi_ZZ = 0``. In ``x = (R - R0) / a0``, ``y = Z / a0`` at the
    reference point this reads ``2 c20 + 2 c02 = (a0 / R0) c10`` for the Taylor
    coefficients. The coil flux is current-free away from the coils, so the relation is
    exact there; on the fitted terms the residual measures what the order-3 least-squares
    fit over the layout takes in from higher orders.

    Args:
        table: Rows with ``coil_c10``, ``coil_c20`` and ``coil_c02``.
        machine: Machine of the rows; R0 and a0 come from ``machines.py``.
        r0_m, a0_m: Radii the rows were built with, when they differ from the machine's
            (a diagnostic build with overridden radii).
    """
    cfg = machine if isinstance(machine, MachineConfig) else machine_config(machine)
    r0 = cfg.major_radius.value if r0_m is None else float(r0_m)
    a0 = cfg.minor_radius.value if a0_m is None else float(a0_m)
    c10, c20, c02 = (table[c].to_numpy(dtype=np.float64) for c in ("coil_c10", "coil_c20", "coil_c02"))
    return c20 + c02 - (a0 / (2.0 * r0)) * c10


def shaping(table: pd.DataFrame) -> np.ndarray:
    """``(coil_c02 - coil_c20) / 2`` per row: the stretch quantity left beside ``coil_c10``
    when the vacuum relation holds. Both terms are even under the up-down mirror, so the
    shaping is even."""
    return 0.5 * (table["coil_c02"].to_numpy(dtype=np.float64) - table["coil_c20"].to_numpy(dtype=np.float64))


def stretch_quantities(table: pd.DataFrame, machine: str | MachineConfig, r0_m: float | None = None,
                       a0_m: float | None = None) -> pd.DataFrame:
    """``vacuum_residual`` and ``shaping`` of every row, on the table's index."""
    return pd.DataFrame({"vacuum_residual": vacuum_residual(table, machine, r0_m, a0_m), "shaping": shaping(table)},
                        index=table.index)


def fitted_terms(layout: Layout, psi, r0_m: float, a0_m: float) -> dict[str, float]:
    """Named terms of the order-3 fit to a field given as a function, the route of the coil
    terms: the field sampled at the layout's points in metres, taken relative to the
    reference point, fitted with ``coil_pattern.term_operator``.

    Args:
        layout: Sample points in normalized coordinates.
        psi: Function of ``(R_m, Z_m)`` arrays returning the field at each point.
        r0_m, a0_m: Reference radii in metres.

    Returns:
        ``coil_<term>`` to the fitted coefficient, in the units of ``psi``.
    """
    pr, pz = layout.physical(r0_m, a0_m)
    values = np.asarray(psi(pr, pz), dtype=np.float64)
    coefficients = term_operator(layout) @ (values[:-1] - values[-1])
    return {f"coil_{name}": float(c) for name, c in zip(TERM_NAMES, coefficients)}


def relative_size(values, reference) -> float:
    """Median of ``abs(values)`` over the median of ``abs(reference)``, each over its finite
    values."""
    v = np.abs(np.asarray(values, dtype=np.float64))
    r = np.abs(np.asarray(reference, dtype=np.float64))
    v, r = v[np.isfinite(v)], r[np.isfinite(r)]
    if v.size == 0 or r.size == 0 or np.median(r) == 0:
        return np.nan
    return float(np.median(v) / np.median(r))


def variant_rules(config: FeatureConfig, columns: Sequence[str], unscaled: Sequence[str] = (),
                  extra: Mapping[str, Mapping] | None = None) -> dict[str, dict]:
    """Scaling rules of an in-memory feature variant, for ``RobustScaler.fit_rules``.

    Args:
        config: Configuration giving the rule of each configured column.
        columns: Columns of the variant, in order.
        unscaled: Columns given identity scaling (``scale: none``).
        extra: Rule of each column outside the configuration.

    Raises:
        KeyError: for a column neither configured nor in ``extra``, or an ``unscaled``
            column not among ``columns``.
    """
    extra = dict(extra or {})
    missing = [c for c in unscaled if c not in columns]
    if missing:
        raise KeyError(f"unscaled columns not in the variant: {missing}")
    specs = config.column_specs()
    rules = {}
    for c in columns:
        if c in extra:
            rules[c] = {**ROBUST_RULE, **extra[c]}
        elif c in specs:
            s = specs[c]
            rules[c] = {"scale": s.scale, "scale_group": s.scale_group, "scale_with": s.scale_with, "transform": s.transform}
        else:
            raise KeyError(f"column {c!r} is neither configured nor given a rule")
        if c in unscaled:
            rules[c].update(scale="none", scale_group=None)
    return rules


# ------------------------------------------------------------------------- force balance
def bz_ratio_from_c10(c10, r0_m: float, a0_m: float) -> np.ndarray:
    """``4 pi (R0 / a0) c10``: the vertical coil field at the reference point over
    ``mu0 Ip / (4 pi R0)``, the quantity of check K9, from the named term ``coil_c10``."""
    return 4.0 * np.pi * (r0_m / a0_m) * np.asarray(c10, dtype=np.float64)


def force_balance_check(ratios: Mapping[str, np.ndarray], k9_medians: Mapping[str, float]) -> dict:
    """Check X7: the medians of ``bz_ratio_from_c10`` per machine against the K9 medians.

    Passes when every machine's median has the sign of its K9 median and the machines
    come in the same order by value.

    Args:
        ratios: Machine to the per-frame ratios (flat-top frames).
        k9_medians: Machine to the K9 median of the Bz ratio.
    """
    machines = [m for m in k9_medians if m in ratios]
    medians = {}
    for m in machines:
        v = np.asarray(ratios[m], dtype=np.float64)
        v = v[np.isfinite(v)]
        medians[m] = float(np.median(v)) if v.size else np.nan
    signs = all(np.isfinite(medians[m]) and np.sign(medians[m]) == np.sign(k9_medians[m]) for m in machines)
    finite = all(np.isfinite(medians[m]) for m in machines)
    order = finite and sorted(machines, key=lambda m: medians[m]) == sorted(machines, key=lambda m: k9_medians[m])
    ok = bool(machines) and signs and order and len(machines) == len(k9_medians)
    return {"status": "pass" if ok else "fail", "medians": medians, "k9_medians": dict(k9_medians),
            "signs_match": bool(signs), "order_matches": bool(order),
            "n_frames": {m: int(np.isfinite(np.asarray(ratios[m], dtype=np.float64)).sum()) for m in machines}}


# --------------------------------------------------------------------------------- lines
def line_fit(x, y) -> dict:
    """Least-squares line ``y = intercept + slope x`` over the frames where both are finite
    and each lies inside its own 1st to 99th percentile band (check X14), with the
    correlation and the standard deviation of the residuals."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() >= 3:
        xl, xh = np.percentile(x[keep], BAND_PERCENTILES)
        yl, yh = np.percentile(y[keep], BAND_PERCENTILES)
        keep &= (x >= xl) & (x <= xh) & (y >= yl) & (y <= yh)
    n = int(keep.sum())
    if n < 3 or np.ptp(x[keep]) == 0:
        return {"n": n, "slope": np.nan, "intercept": np.nan, "r": np.nan, "residual_std": np.nan}
    slope, intercept = np.polyfit(x[keep], y[keep], 1)
    residual = y[keep] - (intercept + slope * x[keep])
    return {"n": n, "slope": float(slope), "intercept": float(intercept),
            "r": float(np.corrcoef(x[keep], y[keep])[0, 1]), "residual_std": float(residual.std())}


def median_offset(x, y, fit: Mapping[str, float]) -> float:
    """Median of ``y`` minus the line of ``fit`` over the frames where both are finite."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    keep = np.isfinite(x) & np.isfinite(y)
    if not keep.any() or not np.isfinite(fit["slope"]):
        return np.nan
    return float(np.median(y[keep] - (fit["intercept"] + fit["slope"] * x[keep])))


# ----------------------------------------------------------------------------- timescales
def band_table(values: Mapping[str, Mapping[str, np.ndarray]], reference: str = "DIII-D") -> pd.DataFrame:
    """1st, 50th and 99th percentiles and the band width of each quantity per machine, and
    the width over the reference machine's width (check X15).

    Args:
        values: Quantity to machine to values.
    """
    rows = []
    for quantity, by_machine in values.items():
        widths = {}
        for machine, v in by_machine.items():
            s = reference_statistics(v)
            widths[machine] = s["q99"] - s["q01"]
            rows.append({"quantity": quantity, "machine": machine, "n": s["n"], "q01": s["q01"], "q50": s["q50"],
                         "q99": s["q99"], "band_width": widths[machine]})
        for row in rows[-len(by_machine):]:
            ref = widths.get(reference, np.nan)
            row["width_over_reference"] = row["band_width"] / ref if np.isfinite(ref) and ref > 0 else np.nan
    return pd.DataFrame(rows)


def changing_current_shares(ip_rate_rel, phases, thresholds: Sequence[float]) -> dict:
    """Share of frames off the flat-top, and share with ``abs(ip_rate_rel)`` above each
    threshold (Ip changing by that fraction of itself per resistive time), over the frames
    where ``ip_rate_rel`` is finite for the latter."""
    rate = np.asarray(ip_rate_rel, dtype=np.float64)
    phases = np.asarray(phases)
    finite = np.isfinite(rate)
    out = {"n_frames": int(rate.size), "share_off_flat_top": float((phases != "flat_top").mean()) if rate.size else np.nan,
           "n_rate_finite": int(finite.sum())}
    for t in thresholds:
        out[f"share_abs_rate_rel_above_{t:g}"] = float((np.abs(rate[finite]) > t).mean()) if finite.any() else np.nan
    return out


# --------------------------------------------------------------------------------- mirror
def swap_mirror_currents(frames: pd.DataFrame, cfg: MachineConfig, table_columns: Sequence[str]) -> pd.DataFrame:
    """A copy of frame rows with the currents of every up-down coil pair exchanged (DIII-D
    ``A`` and ``B``, MAST ``u`` and ``l``; ``coil_scales.mirror_pairs``). Columns of
    ``table_columns`` are coil columns of a Green's or point table (check X5)."""
    out = frames.copy()
    for i, j in mirror_pairs(list(table_columns), cfg):
        a, b = frame_current_column(cfg, table_columns[i]), frame_current_column(cfg, table_columns[j])
        out[a], out[b] = frames[b].to_numpy(), frames[a].to_numpy()
    return out


def pattern_difference(a, b, reference=None) -> np.ndarray:
    """Per row, the largest absolute difference between two patterns over the range (max
    minus min) of ``reference`` (``a`` when None); NaN where a row is not finite or the
    range is zero."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ref = a if reference is None else np.asarray(reference, dtype=np.float64)
    rng = np.ptp(ref, axis=1)
    finite = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1) & np.isfinite(ref).all(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.abs(a - b).max(axis=1) / rng
    return np.where(finite & (rng > 0), out, np.nan)


def json_default(value):
    """``json.dumps`` fallback for numpy scalars and arrays."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not serializable: {type(value).__name__}")


def reference_differences(reference: Mapping, current: Mapping, keys: Sequence[str],
                          path_keys: Sequence[str] = ()) -> dict[str, list]:
    """Arguments in which an earlier version's run differs from the current run.

    A report shows an earlier version's numbers beside its own only when both runs used the
    same frames and samples; this lists what stands in the way.

    Args:
        reference: ``arguments`` of the earlier run's record (``dataset.run_info``).
        current: The current run's arguments (``vars(args)``).
        keys: Arguments that must agree.
        path_keys: Those of ``keys`` that are paths, compared after resolving.

    Returns:
        Key to ``[reference value, current value]`` for every key that differs or is
        missing from the reference; empty when the runs are comparable.
    """
    def norm(key: str, value):
        if value is None:
            return None
        return str(Path(value).resolve()) if key in path_keys else value

    out = {}
    for key in keys:
        ref = norm(key, reference.get(key)) if key in reference else "missing"
        cur = norm(key, current.get(key))
        if ref != cur:
            out[key] = [ref, cur]
    return out


def frames_equal(a: pd.DataFrame, b: pd.DataFrame, columns: Sequence[str]) -> bool:
    """Whether two tables hold the same values in the columns, NaN equal to NaN."""
    x = a[list(columns)].to_numpy(dtype=np.float64)
    y = b[list(columns)].to_numpy(dtype=np.float64)
    return x.shape == y.shape and bool(np.array_equal(x, y, equal_nan=True))


# ------------------------------------------------------------------------- linear check
def ip_usable(ip_ka, min_abs_ip_ka: float) -> np.ndarray:
    """Frames whose Ip (kA) is finite, non-zero and at least ``min_abs_ip_ka`` in magnitude,
    the rule of ``features.FeatureContext.ip_ok``."""
    ip = np.asarray(ip_ka, dtype=np.float64)
    return np.isfinite(ip) & (np.abs(ip) >= min_abs_ip_ka) & (ip != 0)


def target_per_ip(target, ip_ka, r0_m: float, min_abs_ip_ka: float) -> tuple[np.ndarray, np.ndarray]:
    """The plasma target over ``mu0 R0 abs(Ip)`` (check X16), without units.

    Args:
        target: ``(frames, nZ, nR)`` plasma target in Wb/rad.
        ip_ka: Ip per frame in kA.
        r0_m: Reference major radius in metres.
        min_abs_ip_ka: Frames below this magnitude, or without Ip, are set to zero.

    Returns:
        ``(scaled target, bool mask of the frames scaled)``.
    """
    t = np.asarray(target, dtype=np.float64)
    ip = np.asarray(ip_ka, dtype=np.float64)
    ok = ip_usable(ip, min_abs_ip_ka)
    denominator = np.where(ok, MU0 * r0_m * np.abs(ip) * A_PER_KA, 1.0)
    return np.where(ok[:, None, None], t / denominator[:, None, None], 0.0), ok


def target_from_per_ip(scaled, ip_ka, r0_m: float) -> np.ndarray:
    """Inverse of ``target_per_ip``: the target in Wb/rad (zero where Ip is missing)."""
    ip = np.nan_to_num(np.asarray(ip_ka, dtype=np.float64))
    return np.asarray(scaled, dtype=np.float64) * (MU0 * r0_m * np.abs(ip) * A_PER_KA)[:, None, None]


def pooled_r2(true, pred) -> float:
    """One minus the summed squared error over the summed variance of the true values,
    pooled over every column."""
    t = np.asarray(true, dtype=np.float64)
    p = np.asarray(pred, dtype=np.float64)
    if t.shape[0] < 2:
        return float("nan")
    total = float(((t - t.mean(axis=0)) ** 2).sum())
    return 1.0 - float(((t - p) ** 2).sum()) / total if total > 0 else float("nan")


def pooled_r2_by_group(true, pred, labels, groups: Sequence[str]) -> dict[str, float]:
    """``pooled_r2`` within each group of rows, each against its own group's mean; NaN for
    a group with fewer than two rows."""
    t = np.asarray(true, dtype=np.float64)
    p = np.asarray(pred, dtype=np.float64)
    labels = np.asarray(labels)
    return {g: pooled_r2(t[labels == g], p[labels == g]) for g in groups}


def variance_shares(true, labels, groups: Sequence[str]) -> dict[str, float]:
    """Share of the summed squared deviation from the overall mean (pooled over columns)
    held by each group's rows: where the variance a pooled R2 measures sits."""
    t = np.asarray(true, dtype=np.float64)
    labels = np.asarray(labels)
    d2 = ((t - t.mean(axis=0)) ** 2).reshape(t.shape[0], -1).sum(axis=1) if t.shape[0] else np.zeros(0)
    total = float(d2.sum())
    return {g: float(d2[labels == g].sum()) / total if total > 0 else float("nan") for g in groups}
