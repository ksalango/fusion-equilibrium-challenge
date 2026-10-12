"""Poloidal flux of the coils on the EFIT grid from coil geometry and coil currents.

Every coil element (a rectangle or a sheared parallelogram in the poloidal plane) is
divided into a grid of circular filaments that share its current. The flux per radian
of one filament at (Rc, Zc) carrying current I, evaluated at (R, Z), lengths in metres
and current in amperes, is

    psi = mu0 I / (2 pi) * sqrt(R Rc) / k * ((2 - k^2) K(k^2) - 2 E(k^2))
    k^2 = 4 R Rc / ((R + Rc)^2 + (Z - Zc)^2)

with K and E the complete elliptic integrals. The implementation writes
sqrt(R Rc) / k as sqrt((R + Rc)^2 + (Z - Zc)^2) / 2, which is finite at R = 0, where
the flux vanishes. A positive current gives positive flux, largest at the filament.

Near a filament the flux diverges logarithmically. The distance between a filament
and an evaluation point is floored at the geometric mean distance of the filament's
own sub-cell, 0.2235 (w + h) for a sub-cell of width w and height h, so that the
flux at a grid node inside a coil converges as the subdivision is refined.

A Green's table holds one 65 by 65 pattern per current column for 1 kA of that
column's value under the machine's turn convention (``fusion_eq.machines``). The
coil flux of any frame is then a weighted sum of patterns, which costs one matrix
product. Patterns are in Wb/rad per kA. Fitted multipliers (``CoilScales``) are
applied at that point, together with a flat solenoid term, a flux that is the same
at every node and proportional to the solenoid current, which stands in for the
part of the solenoid's flux that the lumped rectangle does not produce.

A point table (``PointTable``) holds the same patterns at a list of points instead of the
grid, for features that sample the coil flux at fixed positions; it applies the scales
and the flat term by the same code.

Skew convention (EFIT ``AF`` / ``AF2``, confirmed from the starter kit README, the
EFIT filament routine and the data): ``coil_angle1`` tilts the top and bottom sides
from the horizontal, shifting a point at radial offset u by ``u tan(angle1)`` in Z;
``coil_angle2`` is the angle of the left and right sides from the horizontal,
shifting a point at vertical offset v by ``v / tan(angle2)`` in R. Zero means no skew
and 90 degrees is the same rectangle. No shipped coil has both angles nonzero.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.special import ellipe, ellipk

from fusion_eq import resample
from fusion_eq.io import Shot, sha256_of_arrays
from fusion_eq.machines import MachineConfig, machine_config

MU0 = 4e-7 * np.pi
A_PER_KA = 1000.0
# Geometric mean distance of a rectangle from itself, as a fraction of (width + height).
GMD_FACTOR = 0.2235
# Filaments along the shorter side of an element. Chosen on 7 October 2026 from the K2
# convergence runs: between 16 and 32 the coil flux changes by 7.6e-5 of its range on DIII-D
# nodes away from the coils and 2.7e-4 within one cell of a coil (MAST 1.7e-6 and 1.5e-5);
# 4 failed the 1e-3 criterion on DIII-D and 8 passed it with 3.5e-4.
DEFAULT_N_FILAMENTS = 16
GEOMETRY_COLUMNS = ("coil_name", "coil_input_column", "coil_R", "coil_Z",
                    "coil_width", "coil_height", "coil_angle1", "coil_angle2")
GRID_COLUMNS = ("efit_grid_R", "efit_grid_Z")
# Filaments per chunk when evaluating the kernel, bounding the (nodes x filaments) matrix.
_FILAMENT_CHUNK = 256


# ----------------------------------------------------------------------- Green's function
def filament_flux(r, z, rc, zc, current_a=1.0, min_distance_m: float = 0.0) -> np.ndarray:
    """Flux per radian of circular filaments at evaluation points, in Wb/rad.

    Args:
        r, z: Evaluation points in metres; broadcast against ``rc``, ``zc``.
        rc, zc: Filament positions in metres.
        current_a: Filament current in amperes; broadcast like ``rc``.
        min_distance_m: Floor on the distance between filament and evaluation point.

    Returns:
        The broadcast array of flux values. ``r`` must be non-negative.
    """
    r = np.asarray(r, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    rc = np.asarray(rc, dtype=np.float64)
    zc = np.asarray(zc, dtype=np.float64)
    if np.any(r < 0) or np.any(rc <= 0):
        raise ValueError("evaluation radii must be non-negative and filament radii positive")
    rho2 = (r - rc) ** 2 + (z - zc) ** 2
    if min_distance_m > 0:
        rho2 = np.maximum(rho2, min_distance_m ** 2)
    four_rrc = 4.0 * r * rc
    denominator = four_rrc + rho2
    with np.errstate(divide="ignore", invalid="ignore"):
        k2 = np.where(denominator > 0, four_rrc / denominator, 0.0)
    k2 = np.clip(k2, 0.0, 1.0)
    bracket = (2.0 - k2) * ellipk(k2) - 2.0 * ellipe(k2)
    return MU0 * np.asarray(current_a, dtype=np.float64) / (2.0 * np.pi) * 0.5 * np.sqrt(denominator) * bracket


def filament_flux_at_points(points_r, points_z, rc, zc, current_a, min_distance_m: float = 0.0) -> np.ndarray:
    """Sum of the flux of many filaments at a list of points, as a ``(n_points,)`` array.

    Filaments are processed in chunks so the intermediate array stays small.
    """
    nodes_r = np.asarray(points_r, dtype=np.float64).ravel()[:, None]
    nodes_z = np.asarray(points_z, dtype=np.float64).ravel()[:, None]
    rc = np.asarray(rc, dtype=np.float64).ravel()
    zc = np.asarray(zc, dtype=np.float64).ravel()
    current = np.broadcast_to(np.asarray(current_a, dtype=np.float64), rc.shape)
    out = np.zeros(nodes_r.shape[0])
    for start in range(0, rc.size, _FILAMENT_CHUNK):
        sl = slice(start, start + _FILAMENT_CHUNK)
        out += filament_flux(nodes_r, nodes_z, rc[None, sl], zc[None, sl], current[None, sl],
                             min_distance_m).sum(axis=1)
    return out


def filament_flux_on_grid(grid_r, grid_z, rc, zc, current_a, min_distance_m: float = 0.0) -> np.ndarray:
    """Sum of the flux of many filaments on a grid, as a ``(len(grid_z), len(grid_r))`` map."""
    rr, zz = np.meshgrid(np.asarray(grid_r, dtype=np.float64), np.asarray(grid_z, dtype=np.float64))
    return filament_flux_at_points(rr, zz, rc, zc, current_a, min_distance_m).reshape(rr.shape)


# ----------------------------------------------------------------------------- geometry
def filament_counts(width: float, height: float, n: int) -> tuple[int, int]:
    """Filaments along R and along Z for ``n`` along the shorter side.

    The longer side gets proportionally more filaments, rounded, so that sub-cells are
    close to square. DIII-D F-coils are about six times taller than wide, so ``n`` by
    ``n`` would leave sub-cells taller than a grid step.
    """
    if n < 1:
        raise ValueError("n must be at least 1")
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    if height >= width:
        return n, max(n, int(round(n * height / width)))
    return max(n, int(round(n * width / height))), n


def element_filaments(rc: float, zc: float, width: float, height: float, angle1_deg: float,
                      angle2_deg: float, n: int) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Positions of the filaments of one coil element and the sub-cell size.

    ``n`` filaments span the shorter side and proportionally more the longer one
    (``filament_counts``). The element is an axis-aligned rectangle when both angles are
    zero, otherwise a parallelogram sheared under the EFIT convention (module
    docstring). Filaments sit at the centres of the sub-cells of the unsheared
    rectangle, then are sheared.

    Returns:
        ``(r, z, sub_width, sub_height)`` with ``r`` and ``z`` of equal length.

    Raises:
        ValueError: when both angles are nonzero (no shipped coil is), or ``n < 1``.
    """
    if angle1_deg != 0.0 and angle2_deg != 0.0:
        raise ValueError("an element with both skew angles nonzero is not supported")
    n_r, n_z = filament_counts(width, height, n)
    sub_w, sub_h = width / n_r, height / n_z
    u = -width / 2.0 + sub_w * (np.arange(n_r) + 0.5)
    v = -height / 2.0 + sub_h * (np.arange(n_z) + 0.5)
    uu, vv = np.meshgrid(u, v)
    uu, vv = uu.ravel(), vv.ravel()
    r = rc + uu
    z = zc + vv
    if angle1_deg != 0.0:
        z = z + uu * np.tan(np.radians(angle1_deg))
    elif angle2_deg != 0.0:
        r = r + vv / np.tan(np.radians(angle2_deg))
    return r, z, sub_w, sub_h


def element_outline(rc: float, zc: float, width: float, height: float, angle1_deg: float,
                    angle2_deg: float) -> np.ndarray:
    """Four corners of an element as an ``(4, 2)`` array of (R, Z), counter-clockwise."""
    corners_u = np.array([-0.5, 0.5, 0.5, -0.5]) * width
    corners_v = np.array([-0.5, -0.5, 0.5, 0.5]) * height
    r = rc + corners_u
    z = zc + corners_v
    if angle1_deg != 0.0:
        z = z + corners_u * np.tan(np.radians(angle1_deg))
    elif angle2_deg != 0.0:
        r = r + corners_v / np.tan(np.radians(angle2_deg))
    return np.column_stack([r, z])


@dataclass(frozen=True)
class CoilGeometry:
    """Coil elements of one machine, as shipped in every shot file.

    Attributes:
        names: Element names.
        input_columns: Current column of each element (the join key).
        r, z: Element centres in metres.
        width, height: Radial and vertical extents in metres.
        angle1, angle2: Skew angles in degrees.
    """

    names: np.ndarray
    input_columns: np.ndarray
    r: np.ndarray
    z: np.ndarray
    width: np.ndarray
    height: np.ndarray
    angle1: np.ndarray
    angle2: np.ndarray

    @classmethod
    def from_shot(cls, shot: Shot) -> "CoilGeometry":
        """Reads the eight ``coil_*`` columns of a loaded shot."""
        missing = [c for c in GEOMETRY_COLUMNS if c not in shot]
        if missing:
            raise KeyError(f"geometry columns missing from the shot: {missing}")
        names = np.asarray(shot["coil_name"]).astype(str).ravel()
        cols = np.asarray(shot["coil_input_column"]).astype(str).ravel()
        num = [np.asarray(shot[c], dtype=np.float64).ravel()
               for c in ("coil_R", "coil_Z", "coil_width", "coil_height", "coil_angle1", "coil_angle2")]
        if any(a.size != names.size for a in [cols, *num]):
            raise ValueError("geometry columns differ in length")
        return cls(names, cols, *num)

    @property
    def n_elements(self) -> int:
        return int(self.names.size)

    @property
    def columns(self) -> tuple[str, ...]:
        """Distinct current columns in order of first appearance."""
        return tuple(dict.fromkeys(self.input_columns.tolist()))

    def elements_of(self, column: str) -> np.ndarray:
        """Indices of the elements fed by one current column."""
        return np.flatnonzero(self.input_columns == column)

    def elements_per_column(self) -> dict[str, int]:
        return {c: int(self.elements_of(c).size) for c in self.columns}

    def hash(self) -> str:
        """SHA-256 over the eight geometry columns, matching QA check 0.9."""
        return sha256_of_arrays(self.names, self.input_columns, self.r, self.z, self.width,
                                self.height, self.angle1, self.angle2)

    def check_against_machine(self, cfg: MachineConfig, file_columns: Iterable[str] | None = None) -> list[str]:
        """Problems with the join between geometry and current columns; empty when none.

        Every input column must have a turn convention on the machine, and, when the
        file's column names are given, must exist in the file.
        """
        problems = []
        available = set(file_columns) if file_columns is not None else None
        for column in self.columns:
            try:
                cfg.coil_convention(column)
            except KeyError:
                problems.append(f"{column}: no turn convention on {cfg.name}")
            if available is not None and column not in available:
                problems.append(f"{column}: not a column of the shot file")
        return problems


def near_coil_mask(geometry: CoilGeometry, grid_r, grid_z, margin_cells: float = 1.0) -> np.ndarray:
    """Grid nodes within ``margin_cells`` grid cells of any coil element, as a bool map.

    A node is near an element when, in the element's sheared coordinates (the inverse
    of the filament shear), it lies inside the element outline widened by
    ``margin_cells`` times the grid step in R and in Z.
    """
    grid_r = np.asarray(grid_r, dtype=np.float64)
    grid_z = np.asarray(grid_z, dtype=np.float64)
    d_r = float(np.median(np.diff(grid_r))) * margin_cells
    d_z = float(np.median(np.diff(grid_z))) * margin_cells
    rr, zz = np.meshgrid(grid_r, grid_z)
    near = np.zeros(rr.shape, dtype=bool)
    for i in range(geometry.n_elements):
        u = rr - geometry.r[i]
        v = zz - geometry.z[i]
        if geometry.angle1[i] != 0.0:
            v = v - u * np.tan(np.radians(geometry.angle1[i]))
        elif geometry.angle2[i] != 0.0:
            u = u - v / np.tan(np.radians(geometry.angle2[i]))
        near |= (np.abs(u) <= geometry.width[i] / 2.0 + d_r) & (np.abs(v) <= geometry.height[i] / 2.0 + d_z)
    return near


# -------------------------------------------------------------------------- Green's table
def element_current_fraction(cfg: MachineConfig, column: str, n_elements: int) -> float:
    """Fraction of a column's value carried by each of its elements."""
    convention = cfg.coil_convention(column)
    if convention.turns == "shared":
        return convention.scale / n_elements
    if convention.turns == "per_element":
        return convention.scale
    raise ValueError(f"unknown turn convention {convention.turns!r}")


def column_pattern_at_points(cfg: MachineConfig, geometry: CoilGeometry, column: str, points_r, points_z,
                             n_filaments: int = DEFAULT_N_FILAMENTS) -> np.ndarray:
    """Flux in Wb/rad at a list of points for 1 kA of one current column, ``(n_points,)``."""
    idx = geometry.elements_of(column)
    if idx.size == 0:
        raise KeyError(f"no geometry element for {column}")
    fraction = element_current_fraction(cfg, column, idx.size)
    pattern = np.zeros(np.asarray(points_r).size)
    for i in idx:
        r, z, sub_w, sub_h = element_filaments(geometry.r[i], geometry.z[i], geometry.width[i],
                                               geometry.height[i], geometry.angle1[i], geometry.angle2[i],
                                               n_filaments)
        current = A_PER_KA * fraction / r.size
        pattern += filament_flux_at_points(points_r, points_z, r, z, current, GMD_FACTOR * (sub_w + sub_h))
    return pattern


def column_pattern(cfg: MachineConfig, geometry: CoilGeometry, column: str, grid_r, grid_z,
                   n_filaments: int = DEFAULT_N_FILAMENTS) -> np.ndarray:
    """Flux map in Wb/rad for 1 kA of one current column under the machine's convention."""
    rr, zz = np.meshgrid(np.asarray(grid_r, dtype=np.float64), np.asarray(grid_z, dtype=np.float64))
    return column_pattern_at_points(cfg, geometry, column, rr, zz, n_filaments).reshape(rr.shape)


def _scaled_currents(machine: str, columns: tuple[str, ...], currents_ka, scales) -> tuple[np.ndarray, np.ndarray, bool]:
    """Currents ``(frames, n_columns)`` with the scales applied, the flat solenoid term per
    frame in Wb/rad (computed from the unscaled solenoid current), and whether one frame
    was given. Shared by ``GreenTable.flux`` and ``PointTable.flux``."""
    c = np.nan_to_num(np.asarray(currents_ka, dtype=np.float64))
    single = c.ndim == 1
    c = np.atleast_2d(c)
    if c.shape[1] != len(columns):
        raise ValueError(f"expected {len(columns)} currents, got {c.shape[1]}")
    flat_term = np.zeros(c.shape[0])
    if isinstance(scales, CoilScales):
        if scales.machine != machine:
            raise ValueError(f"scales are for {scales.machine}, the table for {machine}")
        if scales.flat_wb_per_ka != 0.0:
            sol = list(columns).index(machine_config(machine).solenoid_column)
            flat_term = scales.flat_wb_per_ka * c[:, sol]
        c = c * scales.vector(columns)
    elif scales is not None:
        c = c * np.asarray(scales, dtype=np.float64)
    return c, flat_term, single


@dataclass
class GreenTable:
    """One flux pattern per current column of a machine.

    Attributes:
        machine: Machine name.
        columns: Current columns in pattern order.
        patterns: ``(n_columns, nZ, nR)`` flux in Wb/rad per kA of the column's value.
        grid_r, grid_z: Grid axes in metres.
        n_filaments: Filaments along the shorter side of each element (``filament_counts``).
        geometry_hash, grid_hash: SHA-256 of the inputs, for provenance.
        conventions: Column to turn convention name.
        elements_per_column: Column to element count.
    """

    machine: str
    columns: tuple[str, ...]
    patterns: np.ndarray
    grid_r: np.ndarray
    grid_z: np.ndarray
    n_filaments: int
    geometry_hash: str
    grid_hash: str
    conventions: dict[str, str]
    elements_per_column: dict[str, int]

    def flux(self, currents_ka, scales=None) -> np.ndarray:
        """Coil flux of one or many frames, in Wb/rad.

        Args:
            currents_ka: ``(n_columns,)`` or ``(frames, n_columns)`` currents in kA in
                ``columns`` order. NaN currents are treated as zero.
            scales: Optional per-column multipliers, ``(n_columns,)``, or a
                ``CoilScales`` record, whose flat solenoid term is added to every node.

        Returns:
            ``(nZ, nR)`` for one frame or ``(frames, nZ, nR)`` for many.
        """
        c, flat_term, single = _scaled_currents(self.machine, self.columns, currents_ka, scales)
        flat = c @ self.patterns.reshape(len(self.columns), -1)
        out = flat.reshape(c.shape[0], *self.patterns.shape[1:]) + flat_term[:, None, None]
        return out[0] if single else out

    def column_index(self, columns: Iterable[str]) -> np.ndarray:
        """Indices of the given columns in ``columns`` order; raises for unknown names."""
        lookup = {c: i for i, c in enumerate(self.columns)}
        return np.array([lookup[c] for c in columns], dtype=int)

    def save(self, path: Path | str) -> None:
        """Writes the table as an npz file with a JSON metadata entry."""
        meta = {"machine": self.machine, "columns": list(self.columns), "n_filaments": self.n_filaments,
                "geometry_hash": self.geometry_hash, "grid_hash": self.grid_hash,
                "conventions": self.conventions, "elements_per_column": self.elements_per_column,
                "units": "Wb/rad per kA", "gmd_factor": GMD_FACTOR}
        np.savez(path, patterns=self.patterns, grid_r=self.grid_r, grid_z=self.grid_z,
                 meta=np.array(json.dumps(meta, sort_keys=True)))

    @classmethod
    def load(cls, path: Path | str) -> "GreenTable":
        with np.load(path, allow_pickle=False) as f:
            meta = json.loads(str(f["meta"]))
            return cls(machine=meta["machine"], columns=tuple(meta["columns"]), patterns=f["patterns"],
                       grid_r=f["grid_r"], grid_z=f["grid_z"], n_filaments=int(meta["n_filaments"]),
                       geometry_hash=meta["geometry_hash"], grid_hash=meta["grid_hash"],
                       conventions=dict(meta["conventions"]),
                       elements_per_column={k: int(v) for k, v in meta["elements_per_column"].items()})


def build_green_table(machine: str | MachineConfig, geometry: CoilGeometry, grid_r, grid_z,
                      n_filaments: int = DEFAULT_N_FILAMENTS) -> GreenTable:
    """Builds the Green's table of a machine from its geometry and grid.

    Args:
        machine: Machine name or configuration.
        geometry: Coil elements.
        grid_r, grid_z: Grid axes in metres.
        n_filaments: Filaments along the shorter side of each element (``filament_counts``);
            the default is the subdivision chosen from the K2 convergence runs.

    Raises:
        ValueError: when a geometry column has no turn convention on the machine.
    """
    cfg = machine if isinstance(machine, MachineConfig) else machine_config(machine)
    problems = geometry.check_against_machine(cfg)
    if problems:
        raise ValueError("; ".join(problems))
    grid_r = np.asarray(grid_r, dtype=np.float64)
    grid_z = np.asarray(grid_z, dtype=np.float64)
    columns = geometry.columns
    patterns = np.stack([column_pattern(cfg, geometry, c, grid_r, grid_z, n_filaments) for c in columns])
    return GreenTable(machine=cfg.name, columns=columns, patterns=patterns, grid_r=grid_r, grid_z=grid_z,
                      n_filaments=n_filaments, geometry_hash=geometry.hash(),
                      grid_hash=sha256_of_arrays(grid_r, grid_z),
                      conventions={c: cfg.coil_convention(c).turns for c in columns},
                      elements_per_column=geometry.elements_per_column())


@dataclass
class PointTable:
    """One flux value per point and current column of a machine: a Green's table on a list
    of points instead of the grid.

    Attributes:
        machine: Machine name.
        columns: Current columns in pattern order.
        patterns: ``(n_columns, n_points)`` flux in Wb/rad per kA of the column's value.
        points_r, points_z: Point positions in metres.
        n_filaments: Filaments along the shorter side of each element.
        geometry_hash, points_hash: SHA-256 of the inputs, for provenance and the cache key.
    """

    machine: str
    columns: tuple[str, ...]
    patterns: np.ndarray
    points_r: np.ndarray
    points_z: np.ndarray
    n_filaments: int
    geometry_hash: str
    points_hash: str

    def flux(self, currents_ka, scales=None) -> np.ndarray:
        """Coil flux at the points in Wb/rad, ``(n_points,)`` for one frame or
        ``(frames, n_points)`` for many; arguments as ``GreenTable.flux``."""
        c, flat_term, single = _scaled_currents(self.machine, self.columns, currents_ka, scales)
        out = c @ self.patterns + flat_term[:, None]
        return out[0] if single else out

    def column_index(self, columns: Iterable[str]) -> np.ndarray:
        """Indices of the given columns in ``columns`` order; raises for unknown names."""
        lookup = {c: i for i, c in enumerate(self.columns)}
        return np.array([lookup[c] for c in columns], dtype=int)

    def save(self, path: Path | str) -> None:
        meta = {"machine": self.machine, "columns": list(self.columns), "n_filaments": self.n_filaments,
                "geometry_hash": self.geometry_hash, "points_hash": self.points_hash, "units": "Wb/rad per kA",
                "gmd_factor": GMD_FACTOR}
        np.savez(path, patterns=self.patterns, points_r=self.points_r, points_z=self.points_z,
                 meta=np.array(json.dumps(meta, sort_keys=True)))

    @classmethod
    def load(cls, path: Path | str) -> "PointTable":
        with np.load(path, allow_pickle=False) as f:
            meta = json.loads(str(f["meta"]))
            return cls(machine=meta["machine"], columns=tuple(meta["columns"]), patterns=f["patterns"],
                       points_r=f["points_r"], points_z=f["points_z"], n_filaments=int(meta["n_filaments"]),
                       geometry_hash=meta["geometry_hash"], points_hash=meta["points_hash"])


def build_point_table(machine: str | MachineConfig, geometry: CoilGeometry, points_r, points_z,
                      n_filaments: int = DEFAULT_N_FILAMENTS) -> PointTable:
    """Builds the point table of a machine from its geometry, with the code and filament
    subdivision of ``build_green_table``.

    Raises:
        ValueError: when a geometry column has no turn convention on the machine.
    """
    cfg = machine if isinstance(machine, MachineConfig) else machine_config(machine)
    problems = geometry.check_against_machine(cfg)
    if problems:
        raise ValueError("; ".join(problems))
    points_r = np.asarray(points_r, dtype=np.float64).ravel()
    points_z = np.asarray(points_z, dtype=np.float64).ravel()
    columns = geometry.columns
    patterns = np.stack([column_pattern_at_points(cfg, geometry, c, points_r, points_z, n_filaments) for c in columns])
    return PointTable(machine=cfg.name, columns=columns, patterns=patterns, points_r=points_r, points_z=points_z,
                      n_filaments=n_filaments, geometry_hash=geometry.hash(),
                      points_hash=sha256_of_arrays(points_r, points_z))


ADOPTED = "adopted"
RESOURCE_DIR = Path(__file__).resolve().parent / "resources"
# Adopted coil model per machine: the file is the ``CoilScales`` record chosen at the close
# of the coil-flux phase (DIII-D: shaping coils as stored plus the flat solenoid term, the
# ``solenoid_flat`` variant of the fit of 10 October 2026). A machine without a file uses
# the default record, every scale 1 and no flat term.
ADOPTED_FILES = {"DIII-D": RESOURCE_DIR / "coil_scales_d3d.json"}


def adopted_scales(machine: str, columns: Iterable[str]) -> "CoilScales":
    """The adopted ``CoilScales`` of a machine, or the default record when none is adopted.

    Raises:
        KeyError: when the adopted file lacks one of ``columns``.
    """
    columns = list(columns)
    path = ADOPTED_FILES.get(machine)
    if path is None or not path.is_file():
        return CoilScales.default(machine, columns)
    record = CoilScales.load(path)
    if record.machine != machine:
        raise ValueError(f"{path} is for {record.machine}, not {machine}")
    record.vector(columns)                   # every column must have a scale
    return record


def coil_flux(table: GreenTable, currents_ka, scales=ADOPTED) -> np.ndarray:
    """Coil flux from a table and currents in kA, under the machine's adopted coil model.

    ``scales`` defaults to ``adopted_scales(table.machine, table.columns)``; pass None for
    the unscaled patterns, an array of per-column multipliers, or a ``CoilScales``
    record. See ``GreenTable.flux``.
    """
    if isinstance(scales, str) and scales == ADOPTED:
        scales = adopted_scales(table.machine, table.columns)
    return table.flux(currents_ka, scales)


# -------------------------------------------------------------------- fitted scales
@dataclass
class CoilScales:
    """Fitted multipliers of a machine's current columns, applied by ``GreenTable.flux``.

    The values are fitted on DIII-D training shots (``fusion_eq.coil_scales``) and
    saved, so that the same record serves training, validation and test. A machine
    without a fit uses ``default``: every scale 1 and no flat term, which leaves the
    calculated flux unchanged; MAST is in that state.

    Attributes:
        machine: Machine name.
        scales: Current column to multiplier of its Green's pattern.
        flat_wb_per_ka: Flat solenoid term in Wb/rad per kA of the solenoid column,
            added at every node of the grid; 0 when not fitted.
        source: Free text on where the values come from (fit variant, run).
    """

    machine: str
    scales: dict[str, float]
    flat_wb_per_ka: float = 0.0
    source: str = ""

    @classmethod
    def default(cls, machine: str, columns: Iterable[str]) -> "CoilScales":
        """Every scale 1 and no flat term."""
        return cls(machine=machine, scales={c: 1.0 for c in columns}, flat_wb_per_ka=0.0, source="default")

    def vector(self, columns: Iterable[str]) -> np.ndarray:
        """Scales in the given column order.

        Raises:
            KeyError: for a column without a scale.
        """
        missing = [c for c in columns if c not in self.scales]
        if missing:
            raise KeyError(f"no scale for {missing} on {self.machine}")
        return np.array([self.scales[c] for c in columns], dtype=np.float64)

    def to_dict(self) -> dict:
        return {"machine": self.machine, "scales": dict(self.scales), "flat_wb_per_ka": float(self.flat_wb_per_ka),
                "source": self.source}

    @classmethod
    def from_dict(cls, d: dict) -> "CoilScales":
        return cls(machine=str(d["machine"]), scales={k: float(v) for k, v in d["scales"].items()},
                   flat_wb_per_ka=float(d.get("flat_wb_per_ka", 0.0)), source=str(d.get("source", "")))

    def save(self, path: Path | str) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=1, sort_keys=True) + "\n")

    @classmethod
    def load(cls, path: Path | str) -> "CoilScales":
        return cls.from_dict(json.loads(Path(path).read_text()))


def direct_coil_flux(machine: str | MachineConfig, geometry: CoilGeometry, currents_ka: dict[str, float],
                     grid_r, grid_z, n_filaments: int = DEFAULT_N_FILAMENTS) -> np.ndarray:
    """Coil flux of one frame summed element by element, without a table.

    Serves as the reference the table is tested against.
    """
    cfg = machine if isinstance(machine, MachineConfig) else machine_config(machine)
    grid_r = np.asarray(grid_r, dtype=np.float64)
    grid_z = np.asarray(grid_z, dtype=np.float64)
    out = np.zeros((len(grid_z), len(grid_r)))
    for column, value in currents_ka.items():
        if not np.isfinite(value) or value == 0:
            continue
        idx = geometry.elements_of(column)
        fraction = element_current_fraction(cfg, column, idx.size)
        for i in idx:
            r, z, sub_w, sub_h = element_filaments(geometry.r[i], geometry.z[i], geometry.width[i],
                                                   geometry.height[i], geometry.angle1[i], geometry.angle2[i],
                                                   n_filaments)
            current = A_PER_KA * value * fraction / r.size
            out += filament_flux_on_grid(grid_r, grid_z, r, z, current, GMD_FACTOR * (sub_w + sub_h))
    return out


# ------------------------------------------------------------------------ frame currents
def currents_at_frames(shot: Shot, columns: Iterable[str], window_ms: float = 2.0,
                       max_gap_ms: float = 5.0) -> np.ndarray:
    """Windowed-mean currents at the shot's EFIT frames, ``(frames, n_columns)`` in kA.

    Uses ``fusion_eq.resample.window_mean`` on each column's own time axis, the same
    rule as the frame table's ``__mean`` columns. A column absent from the shot gives
    NaN.
    """
    efit = np.asarray(shot["efit_times"], dtype=np.float64)
    columns = list(columns)
    out = np.full((efit.size, len(columns)), np.nan)
    for j, column in enumerate(columns):
        if column not in shot:
            continue
        axis = shot[shot.machine.time_axis_column(column)]
        out[:, j] = resample.window_mean(axis, shot[column], efit, window_ms, max_gap_ms)
    return out
