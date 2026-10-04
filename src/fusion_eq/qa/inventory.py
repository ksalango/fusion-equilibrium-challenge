"""Checks 0.1 (file inventory) and 0.14 (column sets).

Both work from Parquet metadata and schema only, so they cost one small read per
file and run on every split. The null count needs the data and is computed from a
table the caller has already read for the other checks.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

# Columns whose stored element type is recorded per shot, for the README dtype
# contradiction (D4 in docs/data_dictionary.md).
DTYPE_COLUMNS = (
    "efit_times", "magnetics_time", "magnetics_plasma_current",
    "magnetics_plasma_current_times", "efit_psirz", "thomson_core_times",
)


def leaf_type(arrow_type: pa.DataType) -> pa.DataType:
    """Returns the innermost element type of a (possibly nested) list type."""
    while pa.types.is_list(arrow_type) or pa.types.is_large_list(arrow_type):
        arrow_type = arrow_type.value_type
    return arrow_type


def check_file_inventory(path: Path | str) -> dict:
    """Check 0.1: the file opens and holds exactly one row.

    Args:
        path: Shot Parquet file.

    Returns:
        ``inv_file_size_bytes``, ``inv_opens`` (bool), and when it opens
        ``inv_n_rows``, ``inv_n_columns``, ``inv_n_row_groups``; otherwise
        ``inv_open_error``.
    """
    path = Path(path)
    out: dict = {"inv_file_size_bytes": int(path.stat().st_size) if path.exists() else -1}
    try:
        meta = pq.ParquetFile(path).metadata
    except Exception as exc:  # noqa: BLE001 - the failure is the finding
        out["inv_opens"] = False
        out["inv_open_error"] = f"{type(exc).__name__}: {exc}"
        return out
    out["inv_opens"] = True
    out["inv_n_rows"] = int(meta.num_rows)
    out["inv_n_columns"] = int(meta.num_columns)
    out["inv_n_row_groups"] = int(meta.num_row_groups)
    return out


def check_column_set(path: Path | str) -> dict:
    """Check 0.14: the column list and the element types of key columns.

    Returns:
        ``col_n``, ``col_names`` (sorted, joined by ``|``), ``col_hash`` (SHA-256 of
        ``col_names``), and ``col_dtype_<name>`` for each of ``DTYPE_COLUMNS`` that is
        present.
    """
    schema = pq.ParquetFile(path).schema_arrow
    names = sorted(schema.names)
    joined = "|".join(names)
    out = {
        "col_n": len(names),
        "col_names": joined,
        "col_hash": hashlib.sha256(joined.encode()).hexdigest()[:16],
    }
    for name in DTYPE_COLUMNS:
        if name in schema.names:
            out[f"col_dtype_{name}"] = str(leaf_type(schema.field(name).type))
    return out


def null_leaf_count(table: pa.Table) -> dict:
    """Counts Parquet nulls at every nesting level of every column in the table.

    The README states the release has no Parquet nulls (gaps are NaN). This records
    the evidence (D8 in the data dictionary).

    Returns:
        ``inv_null_leaves``: total null count over the loaded columns, and
        ``inv_null_columns``: names of columns holding any null, joined by ``|``.
    """
    total = 0
    with_nulls: list[str] = []
    for name in table.column_names:
        arr = table.column(name).combine_chunks()
        count = arr.null_count
        while pa.types.is_list(arr.type) or pa.types.is_large_list(arr.type):
            arr = arr.flatten()
            count += arr.null_count
        total += count
        if count:
            with_nulls.append(name)
    return {"inv_null_leaves": int(total), "inv_null_columns": "|".join(with_nulls)}
