"""Shot file loading shared by every phase.

One challenge Parquet file holds one shot as a single row whose cells are nested
arrays. This module reads a chosen set of columns, converts the cells to numpy
arrays, and applies the organizers' data fixes from the starter kit:

- ``fix_d3d_ip_times`` corrects the DIII-D plasma-current time axis per shot.
- ``align_mast_thomson_core`` drops the uncalibrated MAST Thomson data channel 0.

Both fixes are imported from the starter kit, never reimplemented. After them the
loader applies the project's own Ip alignment (``fusion_eq.ip_alignment``, rule B):
DIII-D shots the organizers' fix leaves untouched get their Ip axis shifted so that
breakdown falls at the fixed target time. The effect of every step is recorded in
``Shot.fixes`` so that quality checks can report what changed.

Raw files are never written to.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from fusion_eq import ip_alignment, starter_kit
from fusion_eq.machines import MachineConfig, machine_from_filename

# Columns the Ip time-axis fix reads. Requesting any Ip column loads all three.
IP_FIX_COLUMNS = ("magnetics_plasma_current_times", "magnetics_plasma_current", "magnetics_time")
IP_COLUMNS = ("magnetics_plasma_current_times", "magnetics_plasma_current")

# Columns the MAST Thomson alignment fix reads. Requesting any loads all three.
THOMSON_FIX_COLUMNS = ("thomson_core_R", "thomson_core_Te", "thomson_core_ne")


@dataclass
class Shot:
    """One shot loaded from a Parquet file, after data fixes.

    Attributes:
        path: Source file.
        shot_id: File stem without the machine prefix, for example ``"203702"`` or
            the hash used in the full release.
        machine: Machine configuration inferred from the file name.
        data: Column name to value. Nested cells become numpy arrays; ragged cells
            become lists of arrays; scalar cells stay Python scalars.
        fixes: Record of what the data fixes did (see ``apply_fixes``).
        file_columns: All column names present in the file, whether loaded or not.
    """

    path: Path
    shot_id: str
    machine: MachineConfig
    data: dict[str, Any] = field(default_factory=dict)
    fixes: dict[str, Any] = field(default_factory=dict)
    file_columns: tuple[str, ...] = ()

    def __getitem__(self, column: str) -> Any:
        return self.data[column]

    def __contains__(self, column: str) -> bool:
        return column in self.data

    def get(self, column: str, default: Any = None) -> Any:
        return self.data.get(column, default)

    @property
    def columns(self) -> list[str]:
        """Loaded column names."""
        return list(self.data)

    @property
    def n_frames(self) -> int:
        """Number of EFIT timeslices, from ``efit_times``."""
        return int(len(self.data["efit_times"]))


def sha256_of_arrays(*arrays) -> str:
    """Hex SHA-256 over the exact bytes of several arrays, in order.

    Numeric arrays are hashed as float64; string arrays as UTF-8 joined by NUL.
    A missing array (None) contributes a fixed marker so the hash stays defined.
    Used to confirm that geometry and grids are identical across shots and builds.
    """
    h = hashlib.sha256()
    for a in arrays:
        if a is None:
            h.update(b"<none>")
            continue
        a = np.asarray(a)
        if a.dtype.kind in ("U", "S", "O"):
            h.update("\x00".join(str(x) for x in a.ravel()).encode())
        else:
            h.update(np.ascontiguousarray(a, dtype=np.float64).tobytes())
        h.update(b"|")
    return h.hexdigest()


def parse_shot_filename(path: Path | str) -> tuple[MachineConfig, str]:
    """Splits a shot file name into machine configuration and shot id."""
    stem = Path(path).stem
    cfg = machine_from_filename(stem)
    return cfg, stem[len(cfg.file_prefix):]


def file_columns(path: Path | str) -> tuple[str, ...]:
    """Returns every column name in the file from its metadata, without reading data."""
    return tuple(pq.ParquetFile(path).schema_arrow.names)


def read_table(path: Path | str, columns: Iterable[str] | None = None) -> pa.Table:
    """Reads the one-row table of a shot file.

    Args:
        path: Parquet file.
        columns: Column names to read; all columns when None. Names absent from the
            file are ignored, so callers can request machine-specific columns freely.

    Returns:
        The Arrow table, holding exactly one row.

    Raises:
        ValueError: when the file does not hold exactly one row.
    """
    if columns is not None:
        present = set(file_columns(path))
        columns = [c for c in dict.fromkeys(columns) if c in present]
    table = pq.read_table(path, columns=columns)
    if table.num_rows != 1:
        raise ValueError(f"{path}: expected one row (one shot), found {table.num_rows}")
    return table


def to_ndarray(cell: pa.Scalar) -> Any:
    """Converts one Arrow cell to a numpy array.

    Args:
        cell: Scalar taken from a one-row table column.

    Returns:
        A numpy array with the nesting levels as axes when every inner list has the
        same length; a list of per-row conversions when the lengths vary (ragged);
        the Python value for non-list cells; None for a null cell.
    """
    if not isinstance(cell, (pa.ListScalar, pa.LargeListScalar)):
        return cell.as_py()
    if not cell.is_valid:
        return None
    arr = cell.values
    shape = [len(arr)]
    while pa.types.is_list(arr.type) or pa.types.is_large_list(arr.type):
        lengths = np.diff(np.asarray(arr.offsets.to_numpy(zero_copy_only=False)))
        if lengths.size and np.any(lengths != lengths[0]):
            return [to_ndarray(inner) for inner in cell.values]
        shape.append(int(lengths[0]) if lengths.size else 0)
        arr = arr.flatten()
    if pa.types.is_string(arr.type) or pa.types.is_large_string(arr.type):
        return np.asarray(arr.to_pylist(), dtype=object).reshape(shape)
    return arr.to_numpy(zero_copy_only=False).reshape(shape)


def table_to_arrays(table: pa.Table) -> dict[str, Any]:
    """Converts every column of a one-row table with ``to_ndarray``."""
    return {name: to_ndarray(table.column(name)[0]) for name in table.column_names}


def apply_fixes(shot: Shot, on_error: str = "raise", align_ip: bool = True) -> None:
    """Applies the data fixes in place and records what they did.

    Order: the organizers' Ip time-axis fix, then the project's Ip alignment (rule B,
    ``fusion_eq.ip_alignment.align_ip_times``) on DIII-D shots the organizers' fix left
    untouched, then the organizers' MAST Thomson alignment.

    Args:
        shot: Shot whose ``data`` holds the raw columns.
        on_error: ``"raise"`` propagates a fix failure; ``"record"`` leaves the
            affected columns raw and stores the message under ``fixes[...]_error``.
        align_ip: Whether to apply rule B after the organizers' fix. False keeps the
            organizers' axis, for comparison.

    Records in ``shot.fixes`` when the Ip columns are loaded:
        ``ip_fix_applied`` (bool, the organizers' fix changed the axis),
        ``ip_time_shift_ms`` (organizers' shift, fixed minus shipped origin),
        ``ip_times_shipped`` (the raw axis in ms), ``ip_breakdown_before_ms``
        (breakdown after the organizers' fix), ``ip_alignment_applied`` (bool),
        ``ip_alignment_shift_ms`` (rule B shift, 0 when not applied),
        ``ip_breakdown_after_ms`` (breakdown on the final axis), ``ip_alignment_error``
        when no breakdown was found.
    When the Thomson core columns are loaded: ``thomson_core_fix_applied`` (bool),
        ``thomson_core_channels_before`` and ``..._after`` (int),
        ``thomson_core_dropped_nan_frac_te`` and ``..._ne`` (fraction of NaN in the
        dropped channel).
    """
    fixes = starter_kit.data_fixes()
    data = shot.data

    if all(c in data for c in IP_FIX_COLUMNS):
        shipped = np.asarray(data["magnetics_plasma_current_times"], dtype=np.float64)
        try:
            fixed = np.asarray(fixes.fix_d3d_ip_times(data), dtype=np.float64)
        except Exception as exc:  # noqa: BLE001 - the error itself is the QA finding
            if on_error != "record":
                raise
            shot.fixes["ip_fix_error"] = f"{type(exc).__name__}: {exc}"
        else:
            data["magnetics_plasma_current_times"] = fixed
            shot.fixes["ip_times_shipped"] = shipped
            shot.fixes["ip_time_shift_ms"] = float(fixed[0] - shipped[0])
            shot.fixes["ip_fix_applied"] = bool(fixed[0] != shipped[0])
            _align_ip(shot, align_ip)

    if all(c in data for c in THOMSON_FIX_COLUMNS):
        te_raw = data["thomson_core_Te"]
        n_before = int(te_raw.shape[1]) if isinstance(te_raw, np.ndarray) and te_raw.ndim == 2 else -1
        try:
            r_fixed, te_fixed, ne_fixed = fixes.align_mast_thomson_core(data)
        except Exception as exc:  # noqa: BLE001
            if on_error != "record":
                raise
            shot.fixes["thomson_core_fix_error"] = f"{type(exc).__name__}: {exc}"
            shot.fixes["thomson_core_channels_before"] = n_before
        else:
            n_after = int(te_fixed.shape[1])
            shot.fixes["thomson_core_fix_applied"] = n_after != n_before
            shot.fixes["thomson_core_channels_before"] = n_before
            shot.fixes["thomson_core_channels_after"] = n_after
            if n_after != n_before:
                ne_raw = data["thomson_core_ne"]
                shot.fixes["thomson_core_dropped_nan_frac_te"] = float(np.isnan(te_raw[:, 0]).mean())
                shot.fixes["thomson_core_dropped_nan_frac_ne"] = float(np.isnan(ne_raw[:, 0]).mean())
            data["thomson_core_R"] = r_fixed
            data["thomson_core_Te"] = te_fixed
            data["thomson_core_ne"] = ne_fixed


def _align_ip(shot: Shot, align_ip: bool) -> None:
    """Rule B on a shot whose Ip axis the organizers' fix left untouched (DIII-D only)."""
    data = shot.data
    ip = data["magnetics_plasma_current"]
    axis = data["magnetics_plasma_current_times"]
    before = ip_alignment.breakdown_time_ms(axis, ip)
    shot.fixes["ip_breakdown_before_ms"] = before
    shot.fixes["ip_alignment_applied"] = False
    shot.fixes["ip_alignment_shift_ms"] = 0.0
    shot.fixes["ip_breakdown_after_ms"] = before
    if not align_ip or shot.fixes.get("ip_fix_applied", False):
        return
    if not np.isfinite(before):
        shot.fixes["ip_alignment_error"] = "no Ip breakdown found"
        return
    aligned, shift, _ = ip_alignment.align_ip_times(axis, ip)
    data["magnetics_plasma_current_times"] = aligned
    shot.fixes["ip_alignment_applied"] = True
    shot.fixes["ip_alignment_shift_ms"] = shift
    shot.fixes["ip_breakdown_after_ms"] = ip_alignment.breakdown_time_ms(aligned, ip)


def shot_from_table(table: pa.Table, path: Path | str, apply: bool = True,
                    on_fix_error: str = "raise",
                    keep: Iterable[str] | None = None, align_ip: bool = True) -> Shot:
    """Builds a ``Shot`` from an already-read table.

    Args:
        table: One-row table.
        path: Source file, used for the machine and shot id.
        apply: Whether to run the data fixes.
        on_fix_error: Passed to ``apply_fixes``.
        keep: Columns to keep in ``data``; all table columns when None. Columns read
            only as fix dependencies are dropped here.
        align_ip: Passed to ``apply_fixes``.
    """
    path = Path(path)
    cfg, shot_id = parse_shot_filename(path)
    shot = Shot(path=path, shot_id=shot_id, machine=cfg, data=table_to_arrays(table),
                file_columns=file_columns(path))
    if apply:
        apply_fixes(shot, on_error=on_fix_error, align_ip=align_ip)
    if keep is not None:
        keep = set(keep)
        shot.data = {k: v for k, v in shot.data.items() if k in keep}
    return shot


def load_shot(path: Path | str, columns: Iterable[str] | None = None,
              apply: bool = True, on_fix_error: str = "raise", align_ip: bool = True) -> Shot:
    """Loads one shot with the data fixes applied.

    Args:
        path: Parquet file of the shot.
        columns: Columns to load; all when None. Columns absent from the file are
            ignored. Requesting a plasma-current column or a Thomson core column
            also reads the other columns the corresponding fix needs; those extra
            columns are not kept unless requested.
        apply: Whether to apply the data fixes. False returns the file as shipped.
        on_fix_error: ``"raise"`` or ``"record"``, see ``apply_fixes``.
        align_ip: Whether to apply the project's Ip alignment (rule B) after the
            organizers' fix. False keeps the organizers' axis, for comparison.

    Returns:
        The loaded shot.
    """
    if columns is None:
        return shot_from_table(read_table(path), path, apply, on_fix_error, align_ip=align_ip)
    requested = list(dict.fromkeys(columns))
    to_read = list(requested)
    if apply:
        if any(c in requested for c in IP_COLUMNS):
            to_read += [c for c in IP_FIX_COLUMNS if c not in to_read]
        if any(c in requested for c in THOMSON_FIX_COLUMNS):
            to_read += [c for c in THOMSON_FIX_COLUMNS if c not in to_read]
    table = read_table(path, to_read)
    return shot_from_table(table, path, apply, on_fix_error, keep=requested, align_ip=align_ip)
