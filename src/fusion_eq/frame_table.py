"""Frame table: one row per EFIT frame with every input brought onto that frame's time.

The table is the single starting point of every later analysis and model. Shots are
read with ``fusion_eq.io.load_shot`` (organizers' fixes plus the project's Ip
alignment), currents are resampled onto ``efit_times`` in three forms (nearest valid
sample, windowed mean, windowed slope; a second, longer slope for Ip, the solenoid and
the shaping coils), Thomson summaries are brought to frame times in two forms
(interpolated, nearest), and shot-level flags and the time of the Ip rise are attached.
Labels are copied as stored, never interpolated, and only into the training table.

The column schema gives every column a role: ``key``, ``input``, ``flag`` or
``label``. ``input_columns`` and ``assert_input_columns`` are the guard against a
label reaching a model as a feature. Specification: ``docs/FRAME_TABLE.md``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from fusion_eq import dataset, io, ip_alignment, resample, thomson
from fusion_eq.machines import MachineConfig, machine_config

ROLES = ("key", "input", "flag", "label")
RESAMPLE_SUFFIXES = ("__nearest", "__mean", "__rate")
RATE_SUFFIX = "__rate"
# Slope windows in ms. The first gives the ``__rate`` columns of every current; each
# further window gives ``__rate<ms>`` columns for Ip, the solenoid and the shaping coils
# (F32: the 10 ms slope is too noisy on flat-top; the toroidal field is constant in a shot).
DEFAULT_RATE_WINDOWS_MS = (10.0, 50.0)
THOMSON_SUFFIXES = ("__interp", "__nearest")

# Columns copied from the shot file into the training table without any change.
COPIED_LABELS = ("efit_q95", "efit_beta_n", "efit_li", "efit_r_axis", "efit_z_axis", "efit_lcfs_n")
PSI_SUMMARY_LABELS = ("psi_min", "psi_max", "psi_mean")
# Columns withheld on test. Anything starting with ``efit_`` is a label too, except the
# time base and the grid, which do not enter the table under their own names.
LABEL_PREFIX = "efit_"
LABEL_EXCEPTIONS = ("efit_times", "efit_grid_R", "efit_grid_Z")
UNSCORED_LABELS = ("magnetics_dsep", "magnetics_dsep_times")

IP_PEAK_PERCENTILE = 99.5
RATE_ROUNDING_HZ = 100.0

ARROW_TYPES = {
    "string": pa.string(), "int8": pa.int8(), "int32": pa.int32(), "float32": pa.float32(),
    "float64": pa.float64(), "bool": pa.bool_(),
}


@dataclass(frozen=True)
class BuildParams:
    """Parameters of one frame-table build.

    Attributes:
        window_ms: Full length of the centred window of the ``__mean`` columns.
        rate_window_ms: Full lengths of the centred slope windows, in ms. The first gives
            the ``__rate`` columns; each further one the ``__rate<ms>`` columns of
            ``long_rate_signals`` (``rate_columns``). A single number is read as one window.
        max_gap_ms: Largest distance to the nearest valid sample for ``__nearest``, and
            largest gap interpolated over when a mean window is empty.
        thomson_tol_ms: Distance to the nearest valid Thomson slice under which a frame
            is marked ``ts_<system>_valid``.
        align_ip: Whether the loader applies the project's Ip alignment (rule B).
        psi_summary: Whether the training table carries ``psi_min/max/mean``, which
            requires reading the flux maps.
    """

    window_ms: float = 2.0
    rate_window_ms: tuple[float, ...] = DEFAULT_RATE_WINDOWS_MS
    max_gap_ms: float = 5.0
    thomson_tol_ms: float = 20.0
    align_ip: bool = True
    psi_summary: bool = True

    def __post_init__(self) -> None:
        windows = self.rate_window_ms
        windows = (float(windows),) if np.ndim(windows) == 0 else tuple(float(w) for w in windows)
        if not windows or any(not w > 0 for w in windows):
            raise ValueError(f"rate windows must be positive, got {self.rate_window_ms!r}")
        rate_suffixes(windows)
        object.__setattr__(self, "rate_window_ms", windows)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["rate_window_ms"] = list(self.rate_window_ms)
        return d


@dataclass(frozen=True)
class Column:
    """One column of the frame table.

    Attributes:
        name: Column name.
        role: ``key``, ``input``, ``flag`` or ``label``.
        dtype: Storage type, one of ``ARROW_TYPES``.
        units: Physical units, or an empty string.
        description: How the column is computed.
    """

    name: str
    role: str
    dtype: str
    units: str = ""
    description: str = ""


# ------------------------------------------------------------------------------ schema
def current_signals(cfg: MachineConfig) -> dict[str, str]:
    """Frame-table signal name to shot column for every current of a machine.

    Plasma current, solenoid and toroidal field share the names ``ip``, ``sol`` and
    ``tf`` on both machines; shaping coils and extra currents keep their column names.
    """
    signals = {"ip": cfg.ip_column, "sol": cfg.solenoid_column, "tf": cfg.tf_column}
    for col in (*cfg.shaping_coil_columns, *cfg.extra_current_columns):
        signals[col] = col
    return signals


def _current_units(cfg: MachineConfig, column: str) -> str:
    return "kA-turn" if column in cfg.shaping_coil_columns else "kA"


def long_rate_signals(cfg: MachineConfig) -> list[str]:
    """Signals that carry the slopes of the further rate windows: Ip, solenoid, shaping coils."""
    return ["ip", "sol", *cfg.shaping_coil_columns]


def rate_suffixes(rate_windows_ms=DEFAULT_RATE_WINDOWS_MS) -> list[tuple[str, float]]:
    """``(suffix, window_ms)`` of every slope column: ``__rate`` for the first window,
    ``__rate<ms>`` for each further one.

    Raises:
        ValueError: when a further window is not a whole number of ms or repeats a name.
    """
    windows = [float(w) for w in np.atleast_1d(rate_windows_ms)]
    out = [(RATE_SUFFIX, windows[0])]
    for w in windows[1:]:
        if w != round(w):
            raise ValueError(f"further rate windows must be whole ms, got {w}")
        out.append((f"{RATE_SUFFIX}{int(round(w))}", w))
    names = [name for name, _ in out]
    if len(set(names)) != len(names):
        raise ValueError(f"rate windows give repeated column names: {names}")
    return out


def table_schema(machine: str | MachineConfig, with_labels: bool,
                 rate_windows_ms=DEFAULT_RATE_WINDOWS_MS) -> list[Column]:
    """Ordered column schema of one split's table.

    Args:
        machine: Machine name or configuration.
        with_labels: Whether the label columns are included (training table only).
        rate_windows_ms: Slope windows of the build (``BuildParams.rate_window_ms``).
    """
    cfg = machine if isinstance(machine, MachineConfig) else machine_config(machine)
    long_signals = set(long_rate_signals(cfg))
    further_rates = rate_suffixes(rate_windows_ms)[1:]
    cols = [
        Column("machine", "key", "string", "", "Machine name"),
        Column("split", "key", "string", "", "Dataset split"),
        Column("shot", "key", "string", "", "Shot id"),
        Column("frame", "key", "int32", "", "Index of the frame within the shot, from 0"),
        Column("time_ms", "key", "float64", "ms", "efit_times[frame]"),
        Column("n_frames", "key", "int32", "", "Frames in the shot"),
        Column("dt_prev_ms", "key", "float32", "ms", "Time since the previous frame; NaN on the first"),
    ]
    for name, column in current_signals(cfg).items():
        units = _current_units(cfg, column)
        cols += [
            Column(f"{name}__nearest", "input", "float32", units,
                   f"Nearest valid sample of {column}; NaN beyond max_gap_ms"),
            Column(f"{name}__mean", "input", "float32", units,
                   f"Mean of {column} over the centred window_ms; interpolated when empty"),
            Column(f"{name}__rate", "input", "float32", f"{units}/s",
                   f"Least-squares slope of {column} over the centred rate_window_ms"),
        ]
        if name in long_signals:
            cols += [Column(f"{name}{suffix}", "input", "float32", f"{units}/s",
                            f"Least-squares slope of {column} over a centred {window:g} ms window")
                     for suffix, window in further_rates]
    cols += [
        Column("ip_rise_ms", "input", "float64", "ms",
               "Time of the Ip rise: first sample where abs(Ip) exceeds 10 percent of the shot's 99.5th "
               "percentile of abs(Ip) (ip_alignment.breakdown_time_ms), on the final Ip axis; one value per shot"),
        Column("sol_at_rise", "input", "float32", "kA",
               "Solenoid current at ip_rise_ms, mean over the centred window_ms; one value per shot"),
    ]
    for system in thomson.SYSTEMS:
        prefix = f"ts_{system}_"
        for summary in thomson.SUMMARY_NAMES[system]:
            units = {"te": "eV", "ne": "m^-3", "pe": "kPa"}[summary.split("_")[0]]
            what = f"{system} {summary.replace('_', ' ')}"
            cols += [
                Column(f"{prefix}{summary}__interp", "input", "float32", units,
                       f"{what}, linear interpolation between valid slices"),
                Column(f"{prefix}{summary}__nearest", "input", "float32", units,
                       f"{what}, nearest valid slice"),
            ]
        cols += [
            Column(f"{prefix}n_valid", "input", "float32", "count", "Valid channels in the nearest valid slice"),
            Column(f"{prefix}dt_ms", "input", "float32", "ms", "Distance to the nearest valid slice"),
            Column(f"{prefix}gap_ms", "input", "float32", "ms", "Time between the valid slices either side"),
        ]
    cols += [
        Column("mag_rate_hz", "flag", "int32", "Hz", "Recording rate of magnetics_time, rounded to 100 Hz"),
        Column("mag_has_nan", "flag", "bool", "", "A current column has NaN inside the EFIT window"),
        Column("ip_direction", "flag", "int8", "", "+1, -1 or 0 (mixed), Ip inside the EFIT window"),
        Column("tf_direction", "flag", "int8", "", "+1, -1 or 0 (mixed), toroidal field inside the EFIT window"),
        Column("ip_fix", "flag", "string", "", "organizers, alignment or none"),
        Column("ip_shift_ms", "flag", "float32", "ms", "Total shift applied to the Ip time axis"),
        Column("ip_over_peak", "flag", "float32", "",
               "abs(ip__mean) over the shot's 99.5th percentile of abs(Ip)"),
        Column("inputs_complete", "flag", "bool", "", "Every current __mean column is finite"),
        Column("ts_core_valid", "flag", "bool", "", "Nearest valid core slice within thomson_tol_ms"),
        Column("ts_edge_valid", "flag", "bool", "", "Nearest valid edge slice within thomson_tol_ms"),
    ]
    if with_labels:
        for name in COPIED_LABELS:
            cols.append(Column(name, "label", "float64", "", "Copied from the shot file"))
        for name in PSI_SUMMARY_LABELS:
            cols.append(Column(name, "label", "float32", "Wb/rad", f"{name[4:]} of the frame's flux map"))
    return cols


def arrow_schema(columns: Iterable[Column]) -> pa.Schema:
    """Arrow schema of a column list, used for deterministic Parquet output."""
    return pa.schema([pa.field(c.name, ARROW_TYPES[c.dtype]) for c in columns])


def is_label_column(name: str) -> bool:
    """True for a column that is withheld on the test splits."""
    if name in PSI_SUMMARY_LABELS or name in UNSCORED_LABELS:
        return True
    return name.startswith(LABEL_PREFIX) and name not in LABEL_EXCEPTIONS


def input_columns(machine: str | MachineConfig, include_flags: bool = False) -> list[str]:
    """Columns usable as model inputs for a machine. Never contains a label column.

    Args:
        machine: Machine name or configuration.
        include_flags: Whether to add the ``flag`` columns, which are also computed
            from inputs only.
    """
    roles = {"input", "flag"} if include_flags else {"input"}
    out = [c.name for c in table_schema(machine, with_labels=False) if c.role in roles]
    assert not any(is_label_column(c) for c in out)
    return out


def assert_input_columns(columns: Iterable[str], machine: str | MachineConfig) -> list[str]:
    """Checks that every requested column may serve as a model input.

    Args:
        columns: Requested column names.
        machine: Machine name or configuration.

    Returns:
        The requested columns as a list.

    Raises:
        ValueError: when a column is a label, or is not an input, flag or key column
            of the machine's schema.
    """
    requested = list(columns)
    labels = [c for c in requested if is_label_column(c)]
    if labels:
        raise ValueError(f"label columns cannot be model inputs: {labels}")
    allowed = {c.name for c in table_schema(machine, with_labels=False)}
    unknown = [c for c in requested if c not in allowed]
    if unknown:
        raise ValueError(f"not input columns of the {machine} frame table: {unknown}")
    return requested


# ----------------------------------------------------------------------- shot-level rules
def current_direction(values) -> int:
    """Sign class of a current trace: +1 when every finite value is at least 0, -1 when
    every finite value is at most 0, 0 when mixed or when no value is finite.

    Matches the positive, negative, mixed classification of the QA review notebook,
    which uses the minimum and maximum inside the EFIT window.
    """
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0
    if v.min() >= 0:
        return 1
    if v.max() <= 0:
        return -1
    return 0


def recording_rate_hz(times_ms) -> int:
    """Sampling rate of a time axis from its endpoints, rounded to the nearest 100 Hz.

    Computed as ``(n - 1) / (t_last - t_first)`` over the finite entries in float64,
    which is immune to the quantisation of a float32 axis that a median-step estimate
    suffers from. Returns 0 with fewer than two finite entries or a zero span.
    """
    t = np.asarray(times_ms, dtype=np.float64).ravel()
    t = t[np.isfinite(t)]
    if t.size < 2:
        return 0
    span_ms = float(t.max() - t.min())
    if span_ms <= 0:
        return 0
    hz = (t.size - 1) / span_ms * resample.MS_PER_S
    return int(round(hz / RATE_ROUNDING_HZ) * RATE_ROUNDING_HZ)


def _window_values(shot: io.Shot, column: str, efit: np.ndarray) -> np.ndarray:
    """Values of a current column at samples inside the closed EFIT window."""
    axis = np.asarray(shot[shot.machine.time_axis_column(column)], dtype=np.float64)
    values = np.asarray(shot[column], dtype=np.float64)
    if axis.shape != values.shape:
        raise ValueError(f"{column}: length {values.size} does not match its time axis {axis.size}")
    inside = np.isfinite(axis) & (axis >= efit.min()) & (axis <= efit.max())
    return values[inside]


def ip_fix_kind(fixes: dict) -> str:
    """``organizers``, ``alignment`` or ``none`` from a ``Shot.fixes`` record."""
    if fixes.get("ip_fix_applied", False):
        return "organizers"
    if fixes.get("ip_alignment_applied", False):
        return "alignment"
    return "none"


def shot_flags(shot: io.Shot) -> dict:
    """Shot-level flags: recording rate, NaN holes, current directions, Ip fix record.

    Returns:
        ``mag_rate_hz``, ``mag_has_nan``, ``ip_direction``, ``tf_direction``,
        ``ip_fix``, ``ip_shift_ms``.
    """
    cfg = shot.machine
    efit = np.asarray(shot["efit_times"], dtype=np.float64)
    has_nan = False
    for column in cfg.current_columns:
        if column in shot:
            has_nan |= bool(np.isnan(_window_values(shot, column, efit)).any())
    fixes = shot.fixes
    return {
        "mag_rate_hz": recording_rate_hz(shot["magnetics_time"]),
        "mag_has_nan": has_nan,
        "ip_direction": current_direction(_window_values(shot, cfg.ip_column, efit)),
        "tf_direction": current_direction(_window_values(shot, cfg.tf_column, efit)) if cfg.tf_column in shot else 0,
        "ip_fix": ip_fix_kind(fixes),
        "ip_shift_ms": float(fixes.get("ip_time_shift_ms", 0.0)) + float(fixes.get("ip_alignment_shift_ms", 0.0)),
    }


# ------------------------------------------------------------------------------- build
def columns_to_load(cfg: MachineConfig, with_labels: bool, psi_summary: bool) -> list[str]:
    """Shot columns the build reads. Columns absent from a file are ignored by the loader."""
    cols = ["efit_times", "magnetics_time", "magnetics_plasma_current_times", *cfg.current_columns]
    for system in thomson.SYSTEMS:
        cols += [f"thomson_{system}_times", f"thomson_{system}_Te", f"thomson_{system}_ne"]
    if with_labels:
        cols += list(COPIED_LABELS)
        if psi_summary:
            cols.append("efit_psirz")
    return cols


def _resampled_currents(shot: io.Shot, efit: np.ndarray, params: BuildParams) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    nan = np.full(efit.shape, np.nan)
    rates = rate_suffixes(params.rate_window_ms)
    long_signals = set(long_rate_signals(shot.machine))
    for name, column in current_signals(shot.machine).items():
        signal_rates = rates if name in long_signals else rates[:1]
        if column not in shot:
            out[f"{name}__nearest"], out[f"{name}__mean"] = nan, nan
            for suffix, _ in signal_rates:
                out[f"{name}{suffix}"] = nan
            continue
        axis = shot[shot.machine.time_axis_column(column)]
        values = shot[column]
        out[f"{name}__nearest"] = resample.nearest_valid(axis, values, efit, params.max_gap_ms)
        out[f"{name}__mean"] = resample.window_mean(axis, values, efit, params.window_ms, params.max_gap_ms)
        for suffix, window in signal_rates:
            out[f"{name}{suffix}"] = resample.window_slope(axis, values, efit, window)
    return out


def ip_rise(shot: io.Shot, params: BuildParams) -> dict[str, float]:
    """Time of the Ip rise and the solenoid current at that time, for one loaded shot.

    The rise is the 10 percent crossing of ``ip_alignment.breakdown_time_ms`` on the Ip
    axis as loaded (after the organizers' fix and the project's alignment), not the first
    departure from zero. The solenoid current is its mean over the centred ``window_ms``.

    Returns:
        ``ip_rise_ms`` (NaN when no crossing is found) and ``sol_at_rise`` in kA (NaN
        without a rise or a solenoid column).
    """
    cfg = shot.machine
    rise = np.nan
    if cfg.ip_column in shot:
        rise = ip_alignment.breakdown_time_ms(shot[cfg.time_axis_column(cfg.ip_column)], shot[cfg.ip_column])
    sol = np.nan
    if np.isfinite(rise) and cfg.solenoid_column in shot:
        sol = float(resample.window_mean(shot[cfg.time_axis_column(cfg.solenoid_column)], shot[cfg.solenoid_column],
                                         np.array([rise]), params.window_ms, params.max_gap_ms)[0])
    return {"ip_rise_ms": float(rise), "sol_at_rise": sol}


def _thomson_columns(shot: io.Shot, efit: np.ndarray, params: BuildParams) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for system in thomson.SYSTEMS:
        times = shot.get(f"thomson_{system}_times")
        te = shot.get(f"thomson_{system}_Te")
        ne = shot.get(f"thomson_{system}_ne")
        present = (times is not None and isinstance(te, np.ndarray) and isinstance(ne, np.ndarray)
                   and te.ndim == 2 and te.shape[1] > 0)
        if present:
            summaries = thomson.slice_summaries(times, te, ne, shot.machine.thomson_missing, system)
        else:
            summaries = thomson.SliceSummaries(
                system, np.zeros(0), np.zeros(0, dtype=np.int64),
                {name: np.zeros(0) for name in thomson.SUMMARY_NAMES[system]})
        out.update(thomson.frame_columns(summaries, efit, params.thomson_tol_ms))
    return out


def _label_columns(shot: io.Shot, n_frames: int, psi_summary: bool) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for name in COPIED_LABELS:
        if name not in shot:
            raise KeyError(f"label column {name} missing from a labelled shot")
        values = np.asarray(shot[name]).ravel()
        if values.size != n_frames:
            raise ValueError(f"{name}: {values.size} values for {n_frames} frames")
        # Copied, not cast: float32 and float64 sources both land exactly in float64.
        out[name] = values.astype(np.float64)
    nan = np.full(n_frames, np.nan)
    if psi_summary:
        psi = shot.get("efit_psirz")
        if not isinstance(psi, np.ndarray) or psi.ndim != 3 or psi.shape[0] != n_frames:
            raise ValueError("efit_psirz is not a (frames, 65, 65) array")
        psi64 = psi.astype(np.float64)
        out["psi_min"] = psi64.min(axis=(1, 2))
        out["psi_max"] = psi64.max(axis=(1, 2))
        out["psi_mean"] = psi64.mean(axis=(1, 2))
    else:
        out["psi_min"], out["psi_max"], out["psi_mean"] = nan, nan.copy(), nan.copy()
    return out


def build_shot_frames(shot: io.Shot, split: str, params: BuildParams, with_labels: bool) -> pd.DataFrame:
    """One shot's rows of the frame table, in schema order and storage types.

    Args:
        shot: Loaded shot (``load_shot`` with ``columns_to_load``).
        split: Split name written to the rows.
        params: Build parameters.
        with_labels: Whether to append the label columns (training split only).

    Raises:
        ValueError, KeyError: on malformed or missing columns; the caller records the
            error against the shot.
    """
    cfg = shot.machine
    efit = np.asarray(shot["efit_times"], dtype=np.float64).ravel()
    n = efit.size
    if n == 0:
        raise ValueError("efit_times is empty")
    if not np.all(np.isfinite(efit)) or np.any(np.diff(efit) <= 0):
        raise ValueError("efit_times is not finite and strictly increasing")

    data: dict[str, np.ndarray] = {
        "machine": np.full(n, cfg.name, dtype=object),
        "split": np.full(n, split, dtype=object),
        "shot": np.full(n, shot.shot_id, dtype=object),
        "frame": np.arange(n, dtype=np.int32),
        "time_ms": efit,
        "n_frames": np.full(n, n, dtype=np.int32),
        "dt_prev_ms": np.concatenate([[np.nan], np.diff(efit)]),
    }
    data.update(_resampled_currents(shot, efit, params))
    for key, value in ip_rise(shot, params).items():
        data[key] = np.full(n, value)
    ts = _thomson_columns(shot, efit, params)
    data.update(ts)

    flags = shot_flags(shot)
    for key, value in flags.items():
        data[key] = np.full(n, value)
    ip_abs = np.abs(np.asarray(shot[cfg.ip_column], dtype=np.float64))
    ip_abs = ip_abs[np.isfinite(ip_abs)]
    peak = float(np.percentile(ip_abs, IP_PEAK_PERCENTILE)) if ip_abs.size else np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        data["ip_over_peak"] = np.abs(data["ip__mean"]) / peak if peak > 0 else np.full(n, np.nan)
    mean_columns = [f"{name}__mean" for name in current_signals(cfg)]
    data["inputs_complete"] = np.all([np.isfinite(data[c]) for c in mean_columns], axis=0)
    for system in thomson.SYSTEMS:
        data[f"ts_{system}_valid"] = ts[f"ts_{system}_valid"]

    if with_labels:
        data.update(_label_columns(shot, n, params.psi_summary))

    schema = table_schema(cfg, with_labels, params.rate_window_ms)
    frame = pd.DataFrame({c.name: data[c.name] for c in schema})
    return cast_to_schema(frame, schema)


def cast_to_schema(frame: pd.DataFrame, schema: list[Column]) -> pd.DataFrame:
    """Casts the columns of a built frame to their storage types."""
    out = frame.copy()
    for c in schema:
        if c.dtype == "string":
            out[c.name] = out[c.name].astype("string")
        elif c.dtype == "bool":
            out[c.name] = out[c.name].astype(bool)
        else:
            out[c.name] = out[c.name].astype(c.dtype)
    return out


def to_arrow(frame: pd.DataFrame, schema: list[Column]) -> pa.Table:
    """Arrow table of a built frame under the fixed schema."""
    return pa.Table.from_pandas(frame, schema=arrow_schema(schema), preserve_index=False)


INDEX_COLUMNS = ("split", "machine", "shot", "path", "n_frames", "mag_rate_hz", "mag_has_nan",
                 "ip_direction", "tf_direction", "ip_fix", "ip_shift_ms", "ts_core_any_valid",
                 "ts_edge_any_valid", "inputs_complete_frac", "ip_rise_ms", "sol_at_rise", "error")


def build_one(path: Path | str, split: str, params: BuildParams) -> tuple[pd.DataFrame | None, dict]:
    """Loads one shot and builds its rows. Never raises.

    Returns:
        ``(frames, index_row)``. ``frames`` is None when the shot failed, in which
        case ``index_row["error"]`` holds the message. ``index_row`` always has the
        ``INDEX_COLUMNS`` keys.
    """
    path = Path(path)
    row: dict = {key: None for key in INDEX_COLUMNS}
    row.update(split=split, path=str(path), shot=path.stem)
    with_labels = split in dataset.LABELLED_SPLITS
    try:
        cfg, shot_id = io.parse_shot_filename(path)
        row.update(machine=cfg.name, shot=shot_id)
        shot = io.load_shot(path, columns_to_load(cfg, with_labels, params.psi_summary),
                            align_ip=params.align_ip)
        frames = build_shot_frames(shot, split, params, with_labels)
    except Exception as exc:  # noqa: BLE001 - recorded per shot, the run continues
        row["error"] = f"{type(exc).__name__}: {exc}"
        return None, row
    first = frames.iloc[0]
    row.update(
        n_frames=int(len(frames)),
        mag_rate_hz=int(first["mag_rate_hz"]), mag_has_nan=bool(first["mag_has_nan"]),
        ip_direction=int(first["ip_direction"]), tf_direction=int(first["tf_direction"]),
        ip_fix=str(first["ip_fix"]), ip_shift_ms=float(first["ip_shift_ms"]),
        ts_core_any_valid=bool(frames["ts_core_valid"].any()),
        ts_edge_any_valid=bool(frames["ts_edge_valid"].any()),
        inputs_complete_frac=float(frames["inputs_complete"].mean()),
        ip_rise_ms=float(first["ip_rise_ms"]), sol_at_rise=float(first["sol_at_rise"]),
    )
    return frames, row


def shot_index_frame(rows: list[dict]) -> pd.DataFrame:
    """Shot index table from ``build_one`` rows, sorted by split and shot."""
    df = pd.DataFrame(rows, columns=list(INDEX_COLUMNS))
    for col in ("split", "machine", "shot", "path", "ip_fix", "error"):
        df[col] = df[col].astype("string")
    for col in ("n_frames", "mag_rate_hz", "ip_direction", "tf_direction"):
        df[col] = df[col].astype("Int64")
    for col in ("mag_has_nan", "ts_core_any_valid", "ts_edge_any_valid"):
        df[col] = df[col].astype("boolean")
    for col in ("ip_shift_ms", "inputs_complete_frac", "ip_rise_ms", "sol_at_rise"):
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    return df.sort_values(["split", "shot"], kind="stable").reset_index(drop=True)


# ------------------------------------------------------------------------------ readers
DEFAULT_TABLE_ROOT = dataset.DEFAULT_PROCESSED_ROOT / "frame_table"


def table_path(split: str, root: Path | str = DEFAULT_TABLE_ROOT) -> Path:
    """File of one split's table under a build directory."""
    return Path(root) / f"{split}.parquet"


def read_frame_table(split: str, columns: Iterable[str] | None = None, shots: Iterable[str] | None = None,
                     root: Path | str = DEFAULT_TABLE_ROOT) -> pd.DataFrame:
    """Reads one split's table, or part of it.

    Args:
        split: Split name.
        columns: Columns to read; all when None.
        shots: Shot ids to keep; all when None. Rows keep the table's order (shots
            sorted by id, frames in time order).
        root: Build directory.

    Returns:
        The rows as a DataFrame.
    """
    cols = None if columns is None else list(dict.fromkeys(columns))
    filters = None
    if shots is not None:
        filters = [("shot", "in", sorted(set(shots)))]
        if cols is not None and "shot" not in cols:
            cols = cols + ["shot"]
    table = pq.read_table(table_path(split, root), columns=cols, filters=filters)
    frame = table.to_pandas()
    if shots is not None and columns is not None and "shot" not in columns:
        frame = frame.drop(columns=["shot"])
    return frame


def read_flux_maps(shots: Iterable[str], split: str = "diii_d_train",
                   data_root: Path | str = dataset.DEFAULT_DATA_ROOT) -> np.ndarray:
    """Flux maps of the listed shots in the frame table's row order.

    Reads ``efit_psirz`` from the raw shot files; the table does not store maps.
    Shots are taken in sorted id order with their frames in time order, which is the
    row order ``read_frame_table(split, shots=...)`` returns.

    Args:
        shots: Shot ids of a labelled split.
        split: Split name, ``diii_d_train`` by default.
        data_root: Raw data root.

    Returns:
        float32 array ``(total frames, 65, 65)`` indexed ``[row, iZ, iR]``.
    """
    maps = []
    for shot_id in sorted(set(shots)):
        shot = io.load_shot(dataset.shot_path(data_root, split, shot_id), ["efit_psirz"], apply=False)
        maps.append(np.asarray(shot["efit_psirz"], dtype=np.float32))
    if not maps:
        return np.zeros((0, 65, 65), dtype=np.float32)
    return np.concatenate(maps, axis=0)
