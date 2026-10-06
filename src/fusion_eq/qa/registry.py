"""Table of the quality checks and the columns each one needs.

The runner resolves the column patterns against a file's schema, loads the union of
the columns of the checks that apply, and calls each check once. A check applies to
a shot when every pattern in ``required`` matches at least one column of the file,
so label checks skip the inputs-only test files without special cases.
"""

from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatch
from typing import Callable, Iterable

from fusion_eq.qa.currents import check_current_units, check_ip_time_axis
from fusion_eq.qa.geometry import check_coil_geometry, check_grid
from fusion_eq.qa.labels import check_boundary, check_dsep, check_flux_map, check_label_sanity
from fusion_eq.qa.scorer_check import check_against_scorer
from fusion_eq.qa.thomson import check_thomson_alignment, check_thomson_missing


@dataclass(frozen=True)
class Check:
    """One quality check on a loaded shot.

    Attributes:
        check_id: Id from ``docs/QA.md``, for example ``"0.2"``.
        name: Short name used in the report.
        func: ``func(shot) -> dict`` of metrics.
        columns: Glob patterns of the columns the check reads.
        required: Patterns that must match for the check to apply to a shot.
        sampled: True when the check runs on a sampled subset of shots only.
    """

    check_id: str
    name: str
    func: Callable
    columns: tuple[str, ...]
    required: tuple[str, ...]
    sampled: bool = False


CHECKS: tuple[Check, ...] = (
    Check("0.2", "Ip time axis", check_ip_time_axis,
          ("magnetics_plasma_current", "magnetics_plasma_current_times", "magnetics_time", "magnetics_ECOILA",
           "efit_times"),
          ("magnetics_plasma_current", "efit_times")),
    Check("0.3", "MAST Thomson alignment", check_thomson_alignment,
          ("thomson_core_R", "thomson_core_Te", "thomson_core_ne"),
          ("thomson_core_R", "thomson_core_Te", "thomson_core_ne")),
    Check("0.4", "Current units", check_current_units,
          ("magnetics_*", "efit_times"), ("efit_times",)),
    Check("0.5", "Flux map indexing", check_flux_map,
          ("efit_psirz", "efit_grid_R", "efit_grid_Z", "efit_r_axis", "efit_z_axis", "efit_times"),
          ("efit_psirz", "efit_r_axis", "efit_z_axis", "efit_grid_R", "efit_grid_Z")),
    Check("0.6", "Boundary contour", check_boundary,
          ("efit_lcfs_n", "efit_lcfs_r", "efit_lcfs_z", "efit_times"),
          ("efit_lcfs_n", "efit_lcfs_r", "efit_lcfs_z")),
    Check("0.7", "Missing Thomson", check_thomson_missing,
          ("thomson_core_times", "thomson_core_Te", "thomson_core_ne", "thomson_core_R",
           "thomson_edge_times", "thomson_edge_Te", "thomson_edge_ne", "efit_times"),
          ("efit_times", "thomson_core_times")),
    Check("0.8", "magnetics_dsep", check_dsep,
          ("magnetics_dsep", "magnetics_dsep_times", "efit_times"), ("magnetics_dsep",)),
    Check("0.9", "Coil geometry", check_coil_geometry, ("coil_*",), ("coil_name", "coil_input_column")),
    Check("0.10", "Grid", check_grid, ("efit_grid_R", "efit_grid_Z"), ("efit_grid_R", "efit_grid_Z")),
    Check("0.11", "Labels and time bases", check_label_sanity,
          ("efit_times", "magnetics_time", "magnetics_plasma_current", "magnetics_plasma_current_times",
           "magnetics_dsep_times", "thomson_core_times", "thomson_edge_times", "efit_psirz",
           "efit_q95", "efit_beta_n", "efit_li", "efit_r_axis", "efit_z_axis"),
          ("efit_times",)),
    Check("0.15", "Scorer comparison", check_against_scorer,
          ("efit_psirz", "efit_grid_R", "efit_grid_Z", "efit_r_axis", "efit_z_axis", "efit_li",
           "efit_lcfs_n", "efit_lcfs_r", "efit_lcfs_z", "efit_elongation"),
          ("efit_psirz", "efit_lcfs_n", "efit_r_axis"), sampled=True),
)

# Checks implemented as dataset-level aggregations in report.py, listed for the summary.
AGGREGATE_CHECKS = (
    ("0.1", "File inventory"),
    ("0.12", "Split integrity"),
    ("0.14", "Column sets"),
    ("0.11b", "Test frame totals"),
    ("0.13", "README against data"),
)


def matches(pattern: str, columns: Iterable[str]) -> list[str]:
    """Columns matching one glob pattern, in input order."""
    return [c for c in columns if fnmatch(c, pattern)]


def applicable_checks(file_columns: Iterable[str], with_sampled: bool) -> list[Check]:
    """Checks whose required columns are all present in the file."""
    cols = list(file_columns)
    out = []
    for check in CHECKS:
        if check.sampled and not with_sampled:
            continue
        if all(matches(p, cols) for p in check.required):
            out.append(check)
    return out


def columns_for(checks: Iterable[Check], file_columns: Iterable[str]) -> list[str]:
    """Union of the file columns the given checks read, in file order."""
    cols = list(file_columns)
    wanted: set[str] = set()
    for check in checks:
        for pattern in check.columns:
            wanted.update(matches(pattern, cols))
    return [c for c in cols if c in wanted]
