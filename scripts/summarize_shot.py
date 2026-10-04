"""Prints a per-column summary of one shot Parquet file.

Each challenge Parquet file holds one shot as a single row whose cells are
nested arrays. For every column the script reports the shape, the stored
element dtype, the min, max, NaN fraction and zero fraction of the values,
and, for time arrays, the median sampling interval in milliseconds. Every
other numeric column is matched by outer length against the time arrays of
the same shot, and the names of the matching time arrays are reported as
the candidate time base.

The file is described exactly as shipped. No data fixes are applied. Reading
goes through ``fusion_eq.io.read_table``, the shared loader.

Usage::

    python scripts/summarize_shot.py PATH [--columns NAME ...] [--csv]
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa

from fusion_eq.io import read_table
from fusion_eq.qa.inventory import leaf_type

# Columns that carry a time axis. Their lengths define the candidate time
# bases for every other column of the shot.
TIME_AXIS_SUFFIXES = ("_times", "magnetics_time")


@dataclass
class ColumnSummary:
    """Summary of one Parquet column of one shot.

    Attributes:
        name: Parquet column name.
        shape: Shape string, for example ``(65, 65)`` or ``(166, ragged 3..40)``.
        dtype: Leaf Arrow type as stored in the file.
        n_values: Number of leaf values.
        vmin: Minimum finite value, or None for text columns.
        vmax: Maximum finite value, or None for text columns.
        nan_frac: Fraction of leaf values that are NaN or null.
        zero_frac: Fraction of leaf values that equal zero exactly.
        dt_ms: Median sampling interval in ms; only set for time arrays.
        time_base: Names of time arrays whose length equals the outer length
            of this column, joined by ``|``; ``none`` when nothing matches;
            empty for time arrays and text columns.
        examples: First distinct text values, for text columns only.
    """

    name: str
    shape: str
    dtype: str
    n_values: int
    vmin: float | None
    vmax: float | None
    nan_frac: float | None
    zero_frac: float | None
    dt_ms: float | None = None
    time_base: str = ""
    examples: str = ""
    outer_len: int | None = None


def _unnest(array: pa.Array) -> tuple[list[str], pa.Array]:
    """Flattens a nested list array and records the shape at each level.

    Args:
        array: Arrow array holding one nesting level of the cell.

    Returns:
        The shape description per level (``"65"`` for a rectangular level,
        ``"ragged 3..40"`` when inner lengths vary) and the flat leaf array.
    """
    dims: list[str] = []
    current = array
    while pa.types.is_list(current.type) or pa.types.is_large_list(current.type):
        offsets = np.asarray(current.offsets.to_numpy(zero_copy_only=False))
        lengths = np.diff(offsets)
        if len(lengths) == 0:
            dims.append("0")
        elif lengths.min() == lengths.max():
            dims.append(str(int(lengths[0])))
        else:
            dims.append(f"ragged {int(lengths.min())}..{int(lengths.max())}")
        current = current.flatten()
    return dims, current


def summarize_column(name: str, column: pa.ChunkedArray) -> ColumnSummary:
    """Computes the summary of one column of a one-row Parquet table.

    Args:
        name: Column name.
        column: Chunked array with exactly one row.

    Returns:
        The column summary. Numeric statistics ignore NaN and null values.
    """
    scalar = column[0]
    leaf_type_ = leaf_type(column.type)

    if pa.types.is_list(column.type) or pa.types.is_large_list(column.type):
        if not scalar.is_valid:
            return ColumnSummary(name, "null", str(leaf_type_), 0, None, None, None, None)
        outer = scalar.values
        outer_len = len(outer)
        dims, leaf = _unnest(pa.array(outer))
        dims = [str(outer_len)] + dims
    else:
        outer_len = None
        dims = []
        leaf = pa.array([scalar.as_py()], type=column.type)

    shape = "scalar" if not dims else "(" + ", ".join(dims) + ")"

    if pa.types.is_string(leaf_type_) or pa.types.is_large_string(leaf_type_):
        values = leaf.to_pylist()
        n_null = sum(v is None for v in values)
        distinct: list[str] = []
        for v in values:
            if v is not None and v not in distinct:
                distinct.append(v)
            if len(distinct) >= 4:
                break
        examples = f"{len(set(values))} distinct: " + ", ".join(distinct)
        return ColumnSummary(
            name, shape, str(leaf_type_), len(values), None, None,
            n_null / max(len(values), 1), None, examples=examples,
            outer_len=outer_len,
        )

    data = leaf.to_numpy(zero_copy_only=False).astype(np.float64)
    n = data.size
    finite = data[np.isfinite(data)]
    nan_frac = float(np.isnan(data).sum() / n) if n else None
    zero_frac = float((data == 0).sum() / n) if n else None
    vmin = float(finite.min()) if finite.size else None
    vmax = float(finite.max()) if finite.size else None
    return ColumnSummary(
        name, shape, str(leaf_type_), n, vmin, vmax, nan_frac, zero_frac,
        outer_len=outer_len,
    )


def is_time_axis(name: str) -> bool:
    """Returns True for columns that hold a time axis in ms."""
    return name.endswith(TIME_AXIS_SUFFIXES[0]) or name == TIME_AXIS_SUFFIXES[1]


def median_interval_ms(column: pa.ChunkedArray) -> float | None:
    """Returns the median spacing of a time array in ms, ignoring NaN."""
    values = column[0].values.to_numpy(zero_copy_only=False).astype(np.float64)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return None
    return float(np.median(np.diff(values)))


def summarize_table(table: pa.Table, columns: list[str] | None = None) -> list[ColumnSummary]:
    """Summarizes every requested column and assigns candidate time bases.

    Args:
        table: One-row Parquet table for a single shot.
        columns: Column names to include; all columns when None.

    Returns:
        One summary per column, in file order.
    """
    names = [n for n in table.column_names if columns is None or n in columns]
    summaries = [summarize_column(n, table.column(n)) for n in names]

    # Time-base matching uses every time array in the file, not only the
    # selected ones, so a filtered run reports the same pairing as a full run.
    time_lengths: dict[str, int] = {}
    for n in table.column_names:
        if is_time_axis(n) and table.column(n)[0].is_valid:
            time_lengths[n] = len(table.column(n)[0].values)
    for s in summaries:
        if is_time_axis(s.name) and s.outer_len is not None:
            s.dt_ms = median_interval_ms(table.column(s.name))

    for s in summaries:
        if is_time_axis(s.name) or s.examples or s.outer_len is None:
            continue
        matches = [t for t, length in time_lengths.items() if length == s.outer_len]
        s.time_base = "|".join(matches) if matches else "none"
    return summaries


def _fmt(value: float | None, spec: str = ".4g") -> str:
    return "" if value is None else format(value, spec)


def print_table(summaries: list[ColumnSummary], out=sys.stdout) -> None:
    """Prints the summaries as an aligned text table."""
    header = ["column", "shape", "dtype", "min", "max", "nan_frac", "zero_frac", "dt_ms", "time_base"]
    rows = [header]
    for s in summaries:
        rows.append([
            s.name, s.shape, s.dtype, _fmt(s.vmin), _fmt(s.vmax),
            _fmt(s.nan_frac, ".3f"), _fmt(s.zero_frac, ".3f"), _fmt(s.dt_ms),
            s.examples or s.time_base,
        ])
    widths = [max(len(r[i]) for r in rows) for i in range(len(header))]
    for r in rows:
        print("  ".join(cell.ljust(w) for cell, w in zip(r, widths)).rstrip(), file=out)


def print_csv(summaries: list[ColumnSummary], out=sys.stdout) -> None:
    """Prints the summaries as CSV with full-precision numbers."""
    writer = csv.writer(out)
    writer.writerow(["column", "shape", "dtype", "n_values", "min", "max",
                     "nan_frac", "zero_frac", "dt_ms", "time_base", "examples"])
    for s in summaries:
        writer.writerow([s.name, s.shape, s.dtype, s.n_values, s.vmin, s.vmax,
                         s.nan_frac, s.zero_frac, s.dt_ms, s.time_base, s.examples])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("path", type=Path, help="Parquet file holding one shot")
    parser.add_argument("--columns", nargs="+", help="Restrict output to these columns")
    parser.add_argument("--csv", action="store_true", help="Emit CSV instead of a text table")
    args = parser.parse_args(argv)

    table = read_table(args.path)
    summaries = summarize_table(table, args.columns)
    if args.csv:
        print_csv(summaries)
    else:
        print(f"# {args.path.name}: {table.num_rows} row, {table.num_columns} columns")
        print_table(summaries)


if __name__ == "__main__":
    main()
