"""Comparison of a rebuilt table with a stored one, for the determinism check (D7,
``scripts/check_determinism.py``).

Two measures per table, both independent of how the rows are laid out in the Parquet file
(row groups, chunking, file metadata):

- ``table_digest``: SHA-256 of the row bytes, taken column by column in table order. Each
  column contributes its name, its Arrow type, its null mask and its values (fixed-width types
  as their little-endian bytes with nulls set to zero, other types as their UTF-8 text with a
  length prefix). Equal digests mean bit-identical rows.
- ``compare_tables``: exact equality of values, with NaN equal to NaN and nulls equal only to
  nulls. Where the values are equal and the digests differ, the bytes differ without a
  difference of value (a NaN payload or the sign of a zero).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

KEY_COLUMNS = ("shot", "frame")
# Differing columns listed by name in a result; the rest are counted.
MAX_LISTED = 10


def _fixed_width(t: pa.DataType) -> bool:
    return pa.types.is_integer(t) or pa.types.is_floating(t) or pa.types.is_boolean(t)


def column_arrays(column: pa.ChunkedArray | pa.Array) -> tuple[np.ndarray, np.ndarray]:
    """``(values, nulls)`` of a column: fixed-width values as a numpy array with nulls set to
    zero (booleans as uint8), any other type as an object array of Python values; ``nulls`` is
    a bool array."""
    arr = column.combine_chunks() if isinstance(column, pa.ChunkedArray) else column
    nulls = np.asarray(arr.is_null().to_numpy(zero_copy_only=False), dtype=bool)
    if _fixed_width(arr.type):
        filled = pc.fill_null(arr, False if pa.types.is_boolean(arr.type) else 0) if arr.null_count else arr
        values = np.asarray(filled.to_numpy(zero_copy_only=False))
        return (values.astype(np.uint8) if values.dtype == bool else values), nulls
    return np.array(arr.to_pylist(), dtype=object), nulls


def _value_bytes(values: np.ndarray, nulls: np.ndarray) -> bytes:
    if values.dtype != object:
        return np.ascontiguousarray(values.astype(values.dtype.newbyteorder("<"), copy=False)).tobytes()
    parts = []
    for v, null in zip(values, nulls):
        text = b"" if null else str(v).encode("utf-8")
        parts.append(len(text).to_bytes(8, "little") + text)
    return b"".join(parts)


def table_digest(table: pa.Table) -> str:
    """SHA-256 of a table's row bytes (see the module docstring)."""
    h = hashlib.sha256()
    h.update(table.num_rows.to_bytes(8, "little"))
    for name, column in zip(table.column_names, table.columns):
        values, nulls = column_arrays(column)
        for part in (name.encode("utf-8"), str(column.type).encode("utf-8"), nulls.astype(np.uint8).tobytes(),
                     _value_bytes(values, nulls)):
            h.update(len(part).to_bytes(8, "little"))
            h.update(part)
    return h.hexdigest()


def differing_rows(stored: pa.ChunkedArray, rebuilt: pa.ChunkedArray) -> np.ndarray:
    """Bool per row: the values differ (NaN equals NaN; a null equals only a null)."""
    a, a_null = column_arrays(stored)
    b, b_null = column_arrays(rebuilt)
    if a.dtype == object or b.dtype == object:
        same = np.array([x == y for x, y in zip(a, b)], dtype=bool)
    else:
        same = a == b
        if a.dtype.kind == "f" and b.dtype.kind == "f":
            same |= np.isnan(a) & np.isnan(b)
    return (a_null != b_null) | (~a_null & ~same)


def _plain(value):
    """A value for the report (numpy scalars as Python numbers)."""
    return value.item() if isinstance(value, np.generic) else value


def _key_of(table: pa.Table, row: int, keys: Sequence[str]) -> dict:
    return {k: _plain(table.column(k)[row].as_py()) for k in keys if k in table.column_names}


def compare_tables(stored: pa.Table, rebuilt: pa.Table, keys: Sequence[str] = KEY_COLUMNS) -> dict:
    """Compares a rebuilt table with the stored rows of the same shots.

    Args:
        stored: Rows read from the stored table.
        rebuilt: Rows of the rebuild, in the same order (shots sorted by id, frames in order).
        keys: Key columns, reported with a differing row.

    Returns:
        ``identical`` (equal digests), ``values_equal``, the two digests and row counts, and
        when they differ ``reason`` and, for a difference of values, ``first`` (the first
        differing column in table order, its first differing row, that row's key and the two
        values, and the number of differing rows in that column) and ``differing_columns``.
    """
    out = {"rows_stored": stored.num_rows, "rows_rebuilt": rebuilt.num_rows, "digest_stored": table_digest(stored),
           "digest_rebuilt": table_digest(rebuilt)}
    out["identical"] = out["digest_stored"] == out["digest_rebuilt"]
    if stored.column_names != rebuilt.column_names:
        missing = [c for c in stored.column_names if c not in rebuilt.column_names]
        extra = [c for c in rebuilt.column_names if c not in stored.column_names]
        reason = (f"columns differ: missing {missing}, extra {extra}" if missing or extra else "column order differs")
        return {**out, "values_equal": False, "reason": reason}
    types = [c for c in stored.column_names if stored.schema.field(c).type != rebuilt.schema.field(c).type]
    if types:
        c = types[0]
        return {**out, "values_equal": False,
                "reason": f"type of {c} differs: stored {stored.schema.field(c).type}, rebuilt {rebuilt.schema.field(c).type}"}
    if stored.num_rows != rebuilt.num_rows:
        n = min(stored.num_rows, rebuilt.num_rows)
        first_key = next((r for r in range(n) if _key_of(stored, r, keys) != _key_of(rebuilt, r, keys)), n)
        return {**out, "values_equal": False,
                "reason": f"row counts differ: stored {stored.num_rows}, rebuilt {rebuilt.num_rows}; keys agree up to row {first_key}"}
    differing, first = [], None
    for name in stored.column_names:
        rows = np.flatnonzero(differing_rows(stored.column(name), rebuilt.column(name)))
        if rows.size == 0:
            continue
        differing.append(name)
        if first is None:
            r = int(rows[0])
            first = {"column": name, "row": r, "key": _key_of(stored, r, keys), "key_rebuilt": _key_of(rebuilt, r, keys),
                     "stored": _plain(stored.column(name)[r].as_py()), "rebuilt": _plain(rebuilt.column(name)[r].as_py()),
                     "n_rows": int(rows.size)}
    out["values_equal"] = not differing
    if differing:
        out.update(first=first, differing_columns=differing[:MAX_LISTED], n_differing_columns=len(differing),
                   reason=f"values differ in {len(differing)} columns")
    elif not out["identical"]:
        out["reason"] = "values equal, bytes differ (NaN payload or the sign of a zero)"
    return out


def read_rows(path: Path | str, shots: Iterable[str], columns: Sequence[str] | None = None) -> pa.Table:
    """Rows of the given shots from a Parquet table, in file order."""
    wanted = sorted({str(s) for s in shots})
    table = pq.read_table(path, columns=None if columns is None else list(columns))
    mask = pc.is_in(table.column("shot").cast(pa.string()), value_set=pa.array(wanted, pa.string()))
    return table.filter(mask)


def summary_line(result: dict) -> str:
    """One-line reading of a ``compare_tables`` result."""
    if result.get("skipped"):
        return f"skipped: {result['skipped']}"
    if result["identical"]:
        return "identical"
    first = result.get("first")
    if first is None:
        return f"differs: {result['reason']}"
    key = ", ".join(f"{k} {v}" for k, v in first["key"].items())
    more = result["n_differing_columns"] - 1
    return (f"differs: first column {first['column']}, row {first['row']} ({key}): stored {first['stored']!r}, rebuilt "
            f"{first['rebuilt']!r}; {first['n_rows']} rows in that column" + (f"; {more} further columns differ" if more else ""))
