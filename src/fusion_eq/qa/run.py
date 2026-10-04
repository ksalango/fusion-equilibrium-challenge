"""Per-shot orchestration of the quality checks and assembly of the results table.

``run_shot`` is the unit of work the command-line runner distributes over worker
processes. It never raises: a failure in loading or in one check is recorded in the
row's ``error`` column and the remaining checks still run.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fusion_eq import io
from fusion_eq.qa.inventory import check_column_set, check_file_inventory, null_leaf_count
from fusion_eq.qa.registry import applicable_checks, columns_for

KEY_COLUMNS = ("split", "machine", "shot", "path", "error")


def run_shot(path: Path | str, split: str, with_sampled: bool = False) -> dict:
    """Runs every applicable check on one shot file.

    Args:
        path: Shot Parquet file.
        split: Split label written to the row (for example ``diii_d_train``).
        with_sampled: Whether to run the checks marked ``sampled`` (0.15).

    Returns:
        One flat row: ``split``, ``machine``, ``shot``, ``path``, ``error`` (None or
        the joined error messages, each prefixed by its check id), and the metrics
        of every check that ran.
    """
    path = Path(path)
    row: dict = {"split": split, "machine": None, "shot": path.stem, "path": str(path), "error": None}
    errors: list[str] = []
    try:
        cfg, shot_id = io.parse_shot_filename(path)
        row["machine"], row["shot"] = cfg.name, shot_id
    except ValueError as exc:
        errors.append(f"name: {exc}")

    row.update(check_file_inventory(path))
    if not row.get("inv_opens"):
        row["error"] = "; ".join(errors + ["0.1: file does not open"])
        return row
    try:
        row.update(check_column_set(path))
    except Exception as exc:  # noqa: BLE001
        errors.append(f"0.14: {type(exc).__name__}: {exc}")

    try:
        file_cols = io.file_columns(path)
        checks = applicable_checks(file_cols, with_sampled)
        columns = columns_for(checks, file_cols)
        table = io.read_table(path, columns)
        row.update(null_leaf_count(table))
        shot = io.shot_from_table(table, path, apply=True, on_fix_error="record")
        del table
    except Exception as exc:  # noqa: BLE001
        errors.append(f"load: {type(exc).__name__}: {exc}")
        row["error"] = "; ".join(errors)
        return row

    for check in checks:
        try:
            row.update(check.func(shot))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{check.check_id}: {type(exc).__name__}: {exc}")
    row["error"] = "; ".join(errors) if errors else None
    return row


def _coerce_column(values: pd.Series) -> pd.Series:
    """Gives a results column one storage type so it round-trips through Parquet."""
    non_null = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not non_null:
        return values.astype("float64")
    if all(isinstance(v, (bool, np.bool_)) for v in non_null):
        return pd.Series([None if v is None or (isinstance(v, float) and np.isnan(v)) else bool(v)
                          for v in values], index=values.index, dtype="boolean")
    if all(isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, (bool, np.bool_))
           for v in non_null):
        return pd.to_numeric(values, errors="coerce").astype("float64")
    return values.map(lambda v: None if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)) \
        .astype("string")


def rows_to_frame(rows: list[dict]) -> pd.DataFrame:
    """Builds the per-shot results table from the rows returned by ``run_shot``.

    Key columns come first; metric columns follow in alphabetical order. Rows are
    sorted by split and shot so that the output does not depend on completion order.
    """
    df = pd.DataFrame(rows)
    for key in KEY_COLUMNS:
        if key not in df:
            df[key] = None
    metrics = sorted(c for c in df.columns if c not in KEY_COLUMNS)
    df = df[list(KEY_COLUMNS) + metrics]
    for col in df.columns:
        df[col] = _coerce_column(df[col])
    return df.sort_values(["split", "shot"], kind="stable").reset_index(drop=True)
