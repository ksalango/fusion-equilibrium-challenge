"""Synthetic MAST equilibria from the FreeGS free-boundary solver, and the ceiling of the
current basis on them.

The competition rules (update of 23 September 2026) permit synthetic equilibria from a
Grad-Shafranov solver. No MAST equilibrium label is read anywhere in this module: the machine
comes from the dataset's coil geometry and the adopted MAST coil model, the shape targets from
published MAST operating ranges, and the coil and toroidal-field current ranges from MAST test
inputs.

Machine. Each current column of the dataset geometry becomes one FreeGS coil
(``ColumnCoil``) holding FreeGS's own polygon quadrature points of every element of that
column (``freegs.quadrature.polygon_quad``) and FreeGS's Green's function. The coil's current
is the current of one element in amperes, so the column's total is that current times the
number of elements. The element current follows the coil model exactly as the Green's tables
do (``coil_flux.element_current_fraction`` and the ``CoilScales`` multiplier): a P-coil
column in kA-turn is shared over its elements, and under ``sol_half`` every solenoid element
carries half of the solenoid feed current (two-in-hand winding). FreeGS's vacuum flux is then
built from the same geometry and the same convention as the coil model, with an independent
discretisation of each element (quadrature points instead of filaments, no self-distance
floor), which is what the coil-model check compares.

A FreeGS ``Circuit`` of one ``ShapedCoil`` per element gives the same flux but evaluates the
812 elements in a Python loop at every constraint point of every Picard iteration;
``ColumnCoil`` evaluates the same points in one array operation.

Equilibria. A target fixes the plasma current, the toroidal field (``fvac = mu0 N I_tf /
(2 pi)`` from the TF conductor current and the 24 conductors of ``machines.py``), the solenoid
current (held fixed: the P-coils alone shape the plasma), the boundary shape (geometric
centre, minor radius, elongation, triangularity; a Miller-type contour), the topology (lower
single null or double null), poloidal beta and the current-profile peaking. The P-coil
currents are found by FreeGS's constraint controller from X-point and isoflux constraints on
the target boundary; the profile is ``ConstrainBetapIp`` and, when that fails to converge,
``ConstrainPaxisIp`` with the axis pressure that ``ConstrainBetapIp`` reached last.

Grids. FreeGS solves on a finer grid over the MAST flux grid's box. The plasma's own flux is
interpolated onto the native 65 by 65 nodes and the vacuum flux is evaluated directly at those
nodes, so the map is exact at nodes the two grids share and smooth elsewhere. Maps are in
Wb/rad in the physical frame, which on MAST with positive Ip is the stored frame
(``targets.PSI_SIGN_RULE_C``).

Ceiling. ``evaluate_equilibrium`` runs one synthetic frame through the target chain as if it
were labelled: coil flux of the adopted coil model from the currents FreeGS found, plasma flux,
current share inside the scorer's boundary, its coefficients on the stored basis, and the map
rebuilt by ``current_basis.compose_map``. The organizers' local scorer is called unmodified
with its machine set to MAST (``mast_local_score_module``).
"""

from __future__ import annotations

import copy
import importlib.util
import json
import math
import warnings
from dataclasses import asdict, dataclass, field, fields
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Mapping, Sequence

import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.stats import qmc

from fusion_eq import current_basis as cb, current_target as ct, scalars, starter_kit, targets
from fusion_eq.coil_flux import CoilGeometry, CoilScales, element_current_fraction, element_outline
from fusion_eq.lattice import Placement
from fusion_eq.machines import MachineConfig, machine_config

MACHINE = "MAST"
MU0 = 4e-7 * np.pi
A_PER_KA = 1000.0
TOPOLOGIES = ("LSN", "DN")
# Quadrature points per triangle of an element outline (FreeGS accepts 1, 3 or 6).
QUADRATURE_POINTS = 6
# Step of the centred differences of the field components, as in freegs.gradshafranov.
FIELD_STEP_M = 1e-3
# Quadrature points evaluated per chunk, bounding the (points x nodes) intermediate array.
_POINT_CHUNK = 64
# Evaluations at this many points or more are cached per coil (whole-grid evaluations).
_CACHE_MIN_POINTS = 1000
RECORD_VERSION = 1

# Target ranges. Ip, kappa, minor radius, beta_p: the request of 15 October 2026 (published MAST
# operating range). Geometric centre: stands in for the requested magnetic-axis range 0.85 to
# 1.0 m, which FreeGS cannot impose directly; the outward Shafranov shift of a spherical
# tokamak puts the axis about 0.05 to 0.12 m outboard of the boundary centre, and the achieved
# R_axis is recorded. Triangularity was not specified and spans the MAST range of Sykes et al.
# (2001). TF conductor current and solenoid current: 5th to 95th percentile of the MAST test
# inputs (frame table, 15 October 2026), magnitudes in kA. Peaking exponent: the scorer's li of
# the native map was 0.56, 0.70 and 0.90 at 0.25, 0.5 and 0.8 on calibration solves (double null,
# 700 kA, beta_p 0.1 to 0.6, little dependence on beta_p), so the range spans li about 0.5 to 1.0.
# FreeGS's internalInductance1/2/3 are other normalisations (0.60, 0.38 and 0.62 where the
# scorer gives 0.56). Upper gap: lower single null only, the height of the untied
# upper X-point above the mirror of the lower one (``shape_constraints``).
DEFAULT_RANGES: dict[str, tuple[float, float]] = {
    "ip_ka": (400.0, 900.0),
    "tf_ka": (77.7, 85.0),
    "sol_ka": (-44.0, 6.0),
    "kappa": (1.6, 2.2),
    "delta": (0.2, 0.5),
    "r_geo_m": (0.80, 0.92),
    "minor_radius_m": (0.50, 0.62),
    "beta_p": (0.1, 0.6),
    "alpha_n": (0.15, 1.0),
    "upper_gap_m": (0.02, 0.10),
}
DESIGN_KEYS = tuple(DEFAULT_RANGES)
# The design moves the boundary centre so that the inner midplane stays clear of the centre
# column (solenoid elements reach R = 0.157 m) and the outer midplane inside the P5 coils
# (R >= 1.585 m).
MIN_INNER_EDGE_M = 0.22
MAX_OUTER_EDGE_M = 1.48

# P-coil columns held at zero: MAST test inputs show P3 at 0.4 kA-turn (1st to 99th percentile
# -0.6 to 1.6) and P6 at 0 (-10.5 to 10.1) against hundreds on P2, P4 and P5; left free, the
# controller drives them to several hundred kA-turn.
DEFAULT_ZERO_COLUMNS = tuple(f"magnetics_p{i}{s}_current" for i in (3, 6) for s in "lu")
# Bounds of the controlled P-coil columns in kA-turn: minimum and maximum over the upper and
# lower coil of each pair in the MAST test inputs (frame table, 15 October 2026), so that every
# equilibrium is reached with coil currents MAST ran.
DEFAULT_CURRENT_LIMITS_KA: dict[str, tuple[float, float]] = {
    **{f"magnetics_p2{s}_current": (-418.8, 366.9) for s in "lu"},
    **{f"magnetics_p4{s}_current": (-287.9, -1.3) for s in "lu"},
    **{f"magnetics_p5{s}_current": (-177.2, 0.6) for s in "lu"},
}


# ------------------------------------------------------------------------------- machine
def _freegs():
    """Imports FreeGS on first use, so that the package imports without it."""
    import freegs  # noqa: F401  (registers the submodules used below)
    from freegs import coil, equilibrium, gradshafranov, jtor, machine, picard, quadrature, boundary, control

    return freegs, coil, equilibrium, gradshafranov, jtor, machine, picard, quadrature, boundary, control


def _column_coil_class():
    """``ColumnCoil`` derives from ``freegs.coil.Coil``; it is defined on first use."""
    _, coil, _, gs_freegs, *_ = _freegs()

    class ColumnCoil(coil.Coil):
        """All elements of one current column as one FreeGS coil.

        Attributes:
            rq, zq: Quadrature points of every element in metres.
            wq: Their weights; the weights of one element sum to one, so the flux per ampere
                of element current is the sum of ``wq`` times FreeGS's Green's function.
            n_elements: Elements of the column.
        """

        def __init__(self, rq, zq, wq, n_elements: int, control: bool = True) -> None:
            rq, zq, wq = (np.asarray(a, dtype=np.float64).ravel() for a in (rq, zq, wq))
            super().__init__(float(np.average(rq, weights=wq)), float(np.average(zq, weights=wq)), control=control)
            self.rq, self.zq, self.wq = rq, zq, wq
            self.n_elements = int(n_elements)
            self._cache: dict = {}

        def _weighted(self, name: str, kernel, r, z):
            r = np.asarray(r, dtype=np.float64)
            z = np.asarray(z, dtype=np.float64)
            rr, zz = np.broadcast_arrays(r, z)
            # FreeGS evaluates the coil fields on the whole solver grid at every Picard iteration
            # (the poloidal-beta integral of ConstrainBetapIp); the response per ampere depends on
            # the points only, so whole-grid evaluations are cached by the points' values.
            key = None
            if rr.size >= _CACHE_MIN_POINTS:
                key = (name, rr.shape, hash(rr.tobytes()), hash(zz.tobytes()))
                if key in self._cache:
                    return self._cache[key]
            flat_r, flat_z = rr.ravel(), zz.ravel()
            out = np.zeros(flat_r.size)
            for start in range(0, self.rq.size, _POINT_CHUNK):
                sl = slice(start, start + _POINT_CHUNK)
                out += (kernel(self.rq[sl, None], self.zq[sl, None], flat_r[None], flat_z[None]) * self.wq[sl, None]).sum(axis=0)
            if not rr.ndim:
                return float(out[0])
            out = out.reshape(rr.shape)
            if key is not None:
                out.setflags(write=False)
                self._cache[key] = out
            return out

        def controlPsi(self, R, Z):  # noqa: N802, N803  (FreeGS interface)
            return self._weighted("psi", gs_freegs.Greens, R, Z)

        def controlBr(self, R, Z):  # noqa: N802, N803
            return self._weighted("br", lambda rc, zc, r, z: gs_freegs.GreensBr(rc, zc, r, z, FIELD_STEP_M), R, Z)

        def controlBz(self, R, Z):  # noqa: N802, N803
            return self._weighted("bz", lambda rc, zc, r, z: gs_freegs.GreensBz(rc, zc, r, z, FIELD_STEP_M), R, Z)

        def __repr__(self) -> str:
            return f"ColumnCoil(n_elements={self.n_elements}, points={self.rq.size}, current={self.current:.1f} A)"

    return ColumnCoil


@lru_cache(maxsize=1)
def column_coil_class():
    """The ``ColumnCoil`` class (a subclass of ``freegs.coil.Coil``)."""
    return _column_coil_class()


def column_quadrature(geometry: CoilGeometry, column: str, n_points: int = QUADRATURE_POINTS) -> np.ndarray:
    """FreeGS quadrature points of every element of one column, ``(n, 3)`` of (R, Z, weight);
    each element's weights sum to one."""
    quadrature = _freegs()[7]
    points = []
    for i in geometry.elements_of(column):
        outline = element_outline(geometry.r[i], geometry.z[i], geometry.width[i], geometry.height[i],
                                  geometry.angle1[i], geometry.angle2[i])
        points += quadrature.polygon_quad([tuple(map(float, p)) for p in outline], n=n_points)
    if not points:
        raise KeyError(f"no geometry element for {column}")
    return np.asarray(points, dtype=np.float64)


def element_current_a(cfg: MachineConfig, geometry: CoilGeometry, scales: CoilScales, column: str, current_ka: float) -> float:
    """Current of one element in amperes for a column value in the dataset's units, under the
    coil model (the factor the Green's table applies to its pattern)."""
    fraction = element_current_fraction(cfg, column, geometry.elements_of(column).size)
    return float(current_ka) * A_PER_KA * fraction * scales.scales[column]


def column_current_ka(cfg: MachineConfig, geometry: CoilGeometry, scales: CoilScales, column: str, element_a: float) -> float:
    """Inverse of ``element_current_a``: the column value in the dataset's units."""
    fraction = element_current_fraction(cfg, column, geometry.elements_of(column).size)
    return float(element_a) / (A_PER_KA * fraction * scales.scales[column])


@dataclass
class FreegsMachine:
    """A FreeGS machine built from the dataset geometry under one coil model.

    Attributes:
        machine: The ``freegs.machine.Machine``; coil labels are the dataset's current columns.
        geometry: The dataset coil geometry.
        scales: The coil model.
        cfg: Machine conventions.
        fixed_columns: Columns whose current the controller does not change.
    """

    machine: object
    geometry: CoilGeometry
    scales: CoilScales
    cfg: MachineConfig
    fixed_columns: tuple[str, ...]

    @property
    def columns(self) -> tuple[str, ...]:
        return tuple(label for label, _ in self.machine.coils)

    def set_currents_ka(self, currents_ka: Mapping[str, float], tokamak=None) -> None:
        """Sets column currents given in the dataset's units on this machine or on ``tokamak``
        (the machine held by an equilibrium)."""
        tokamak = self.machine if tokamak is None else tokamak
        for column, value in currents_ka.items():
            tokamak[column].current = element_current_a(self.cfg, self.geometry, self.scales, column, value)

    def currents_ka(self, tokamak=None) -> dict[str, float]:
        """Column currents in the dataset's units (kA-turn for P-coils, kA for the solenoid)."""
        tokamak = self.machine if tokamak is None else tokamak
        return {label: column_current_ka(self.cfg, self.geometry, self.scales, label, coil.current)
                for label, coil in tokamak.coils}

    def column_patterns(self, grid_r, grid_z) -> np.ndarray:
        """FreeGS vacuum flux per kA of each column's value at the nodes of a grid,
        ``(n_columns, nZ, nR)`` in Wb/rad per kA."""
        rr, zz = np.meshgrid(np.asarray(grid_r, dtype=np.float64), np.asarray(grid_z, dtype=np.float64))
        return np.stack([coil.controlPsi(rr, zz) * element_current_a(self.cfg, self.geometry, self.scales, label, 1.0)
                         for label, coil in self.machine.coils])


def freegs_machine(geometry: CoilGeometry, scales: CoilScales, machine: str | MachineConfig = MACHINE,
                   fixed_columns: Sequence[str] | None = None, n_points: int = QUADRATURE_POINTS) -> FreegsMachine:
    """The FreeGS machine of the dataset geometry under a coil model.

    Args:
        geometry: Coil elements of the machine.
        scales: The coil model (``coil_flux.coil_model``).
        machine: Machine conventions.
        fixed_columns: Columns held at their set current; the solenoid and
            ``DEFAULT_ZERO_COLUMNS`` when None.
        n_points: Quadrature points per triangle.

    Raises:
        ValueError: when the geometry does not join the machine's columns or the coil model
            lacks a column.
    """
    cfg = machine if isinstance(machine, MachineConfig) else machine_config(machine)
    problems = geometry.check_against_machine(cfg)
    if problems:
        raise ValueError("; ".join(problems))
    scales.vector(geometry.columns)
    fixed = (cfg.solenoid_column, *DEFAULT_ZERO_COLUMNS) if fixed_columns is None else tuple(fixed_columns)
    coil_cls = column_coil_class()
    machine_mod = _freegs()[5]
    coils = []
    for column in geometry.columns:
        q = column_quadrature(geometry, column, n_points)
        coils.append((column, coil_cls(q[:, 0], q[:, 1], q[:, 2], geometry.elements_of(column).size,
                                       control=column not in fixed)))
    return FreegsMachine(machine=machine_mod.Machine(coils), geometry=geometry, scales=scales, cfg=cfg, fixed_columns=fixed)


GEOMETRY_SPLIT = "mast_public_test"


def load_geometry(data_root: Path | str, split: str = GEOMETRY_SPLIT) -> tuple[CoilGeometry, np.ndarray, np.ndarray, str]:
    """MAST coil geometry and flux grid from the first shot file of a MAST test split, reading
    the ``coil_*`` and ``efit_grid_R/Z`` columns only (geometry is identical in every shot, QA
    check 0.9). The labelled MAST sample shots are refused.

    Returns:
        ``(geometry, grid_r, grid_z, file name)``.

    Raises:
        ValueError: for a split that is not a MAST test split.
    """
    from fusion_eq import dataset, io
    from fusion_eq.coil_flux import GEOMETRY_COLUMNS, GRID_COLUMNS

    if dataset.SPLIT_MACHINE.get(split) != MACHINE or split in dataset.LABELLED_SPLITS:
        raise ValueError(f"geometry is read from a MAST test split, not {split!r}")
    path = dataset.list_shots(data_root, split)[0]
    shot = io.load_shot(path, [*GEOMETRY_COLUMNS, *GRID_COLUMNS], apply=False)
    return (CoilGeometry.from_shot(shot), np.asarray(shot["efit_grid_R"], dtype=np.float64),
            np.asarray(shot["efit_grid_Z"], dtype=np.float64), path.name)


def freegs_provenance() -> dict:
    """Installed FreeGS version and, for an install from a repository, its URL and commit."""
    from importlib import metadata

    out = {"version": metadata.version("freegs")}
    try:
        direct = json.loads(metadata.distribution("freegs").read_text("direct_url.json") or "{}")
        out["url"] = direct.get("url")
        out["commit"] = direct.get("vcs_info", {}).get("commit_id")
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return out


# -------------------------------------------------------------------------------- targets
@dataclass(frozen=True)
class EquilibriumTarget:
    """One synthetic equilibrium to solve.

    Attributes:
        index: Position in the design; names the output file.
        topology: ``LSN`` (lower single null) or ``DN`` (double null).
        ip_ka: Plasma current, kA (positive).
        tf_ka: Current of one toroidal-field conductor, kA (magnitude).
        sol_ka: Solenoid feed current, kA, held fixed.
        kappa, delta: Elongation and triangularity of the target boundary.
        r_geo_m, minor_radius_m: Geometric centre and minor radius of the target boundary, m.
        beta_p: Poloidal beta.
        alpha_n: Exponent of the current profile, ``(1 - psi_n^alpha_m)^alpha_n``.
        upper_gap_m: Lower single null only: height in metres of the upper X-point above the
            mirror of the lower one; 0 for double null.
        alpha_m: Inner exponent of the current profile.
    """

    index: int
    topology: str
    ip_ka: float
    tf_ka: float
    sol_ka: float
    kappa: float
    delta: float
    r_geo_m: float
    minor_radius_m: float
    beta_p: float
    alpha_n: float
    upper_gap_m: float = 0.0
    alpha_m: float = 1.0

    def __post_init__(self) -> None:
        if self.topology not in TOPOLOGIES:
            raise ValueError(f"topology {self.topology!r} not in {TOPOLOGIES}")
        if not (self.ip_ka > 0 and self.minor_radius_m > 0 and self.kappa > 0 and -1 < self.delta < 1):
            raise ValueError(f"invalid target {self}")
        if self.r_geo_m - self.minor_radius_m <= 0:
            raise ValueError("the target boundary crosses R = 0")
        if self.topology == "DN" and self.upper_gap_m != 0.0:
            raise ValueError("a double-null target has no upper gap")
        if self.topology == "LSN" and self.upper_gap_m <= 0.0:
            raise ValueError("a lower-single-null target needs a positive upper gap")

    @property
    def aspect_ratio(self) -> float:
        return self.r_geo_m / self.minor_radius_m

    def fvac(self, cfg: MachineConfig | None = None) -> float:
        """Vacuum ``R B_t`` in T m from the conductor current and the machine's TF turns."""
        cfg = cfg or machine_config(MACHINE)
        return MU0 * cfg.tf_turns.value * abs(self.tf_ka) * A_PER_KA / (2.0 * np.pi)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping) -> "EquilibriumTarget":
        names = {f.name for f in fields(cls)}
        return cls(**{k: (int(v) if k == "index" else str(v) if k == "topology" else float(v)) for k, v in d.items() if k in names})


def target_design(n_per_topology: int, seed: int = 0, ranges: Mapping[str, tuple[float, float]] | None = None,
                  topologies: Sequence[str] = TOPOLOGIES) -> list[EquilibriumTarget]:
    """A space-filling design of targets: a scrambled Sobol sequence of ``DESIGN_KEYS`` over
    ``ranges`` per topology, deterministic for a seed.

    The geometric centre is clipped to ``[a + MIN_INNER_EDGE_M, MAX_OUTER_EDGE_M - a]``.

    Args:
        n_per_topology: Targets per topology.
        seed: Seed of the Sobol scrambling (each topology uses ``seed`` plus its position).
        ranges: Key to (low, high); ``DEFAULT_RANGES`` for keys left out.
        topologies: Topologies to include, in order.
    """
    bounds = {**DEFAULT_RANGES, **(ranges or {})}
    unknown = set(bounds) - set(DESIGN_KEYS)
    if unknown:
        raise KeyError(f"unknown design keys {sorted(unknown)}")
    low = np.array([bounds[k][0] for k in DESIGN_KEYS], dtype=np.float64)
    high = np.array([bounds[k][1] for k in DESIGN_KEYS], dtype=np.float64)
    if np.any(high < low):
        raise ValueError("every range needs low <= high")
    out = []
    for t, topology in enumerate(topologies):
        sampler = qmc.Sobol(d=len(DESIGN_KEYS), scramble=True, seed=seed + t)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")            # a count that is not a power of two is fine here
            unit = sampler.random(int(n_per_topology))
        values = qmc.scale(unit, low, high) if np.any(high > low) else np.broadcast_to(low, unit.shape)
        for row in np.atleast_2d(values):
            params = dict(zip(DESIGN_KEYS, map(float, row)))
            if topology == "DN":
                params["upper_gap_m"] = 0.0
            a = params["minor_radius_m"]
            params["r_geo_m"] = float(np.clip(params["r_geo_m"], a + MIN_INNER_EDGE_M, MAX_OUTER_EDGE_M - a))
            out.append(EquilibriumTarget(index=len(out), topology=topology, **params))
    return out


def miller_boundary(target: EquilibriumTarget, theta) -> tuple[np.ndarray, np.ndarray]:
    """Target boundary ``R = R0 + a cos(theta + arcsin(delta) sin(theta))``,
    ``Z = kappa a sin(theta)``, in metres."""
    th = np.asarray(theta, dtype=np.float64)
    a = target.minor_radius_m
    return target.r_geo_m + a * np.cos(th + np.arcsin(target.delta) * np.sin(th)), target.kappa * a * np.sin(th)


# X-points sit a little beyond the boundary tips: their height is this factor times kappa a,
# their radius is the tip's radius moved inward by this factor times delta a.
XPOINT_HEIGHT_FACTOR = 1.05
XPOINT_INWARD_FACTOR = 0.1
# Boundary points (angles from the outer midplane) tied by isoflux to the outer midplane.
ISOFLUX_ANGLES = (np.pi / 4, 3 * np.pi / 4, np.pi, -np.pi / 4, -3 * np.pi / 4)


def shape_constraints(target: EquilibriumTarget) -> tuple[list[tuple[float, float]], list[tuple[float, float, float, float]]]:
    """X-points and isoflux pairs of a target.

    Double null: an X-point at each tip, both tied to the outer midplane. Lower single null: the
    lower X-point tied to the outer midplane, and an upper X-point ``upper_gap_m`` above the
    mirror of the lower one that is not tied, so that it lies outside the boundary (the
    near-double-null configuration MAST mostly ran). Tying the upper tip as an ordinary
    boundary point instead gave elongations near 1.5 and negative upper triangularity, or no
    convergence, within the coil-current limits. Every other constraint ties a boundary point
    (``ISOFLUX_ANGLES``) to the outer midplane point.

    Returns:
        ``(xpoints, isoflux)`` in FreeGS's form: ``[(R, Z)]`` and ``[(R1, Z1, R2, Z2)]``.
    """
    a = target.minor_radius_m
    r_x = target.r_geo_m - (1.0 + XPOINT_INWARD_FACTOR) * target.delta * a
    z_x = XPOINT_HEIGHT_FACTOR * target.kappa * a
    r_out, z_out = target.r_geo_m + a, 0.0
    r_b, z_b = miller_boundary(target, ISOFLUX_ANGLES)
    lower = (float(r_x), float(-z_x))
    upper = (float(r_x), float(z_x + target.upper_gap_m))
    xpoints = [lower, upper]
    isoflux = [(lower[0], lower[1], r_out, z_out)]
    if target.topology == "DN":
        isoflux.append((upper[0], upper[1], r_out, z_out))
    isoflux += [(float(r), float(z), r_out, z_out) for r, z in zip(r_b, z_b)]
    return xpoints, isoflux


def initial_psi(target: EquilibriumTarget, r2d, z2d) -> np.ndarray:
    """Starting plasma flux for FreeGS: a peak at the target centre with the target's widths,
    zero on the domain border, in Wb/rad of the order of the plasma's own flux."""
    a = target.minor_radius_m
    amplitude = MU0 * target.ip_ka * A_PER_KA * target.r_geo_m / (2.0 * np.pi)
    psi = amplitude * np.exp(-((r2d - target.r_geo_m) / a) ** 2 - (z2d / (target.kappa * a)) ** 2)
    psi[0, :] = psi[-1, :] = psi[:, 0] = psi[:, -1] = 0.0
    return psi


# ------------------------------------------------------------------------------- solving
@dataclass(frozen=True)
class SolverGrid:
    """The FreeGS grid: a box in metres and the node counts along R and Z."""

    r_min: float
    r_max: float
    z_min: float
    z_max: float
    n_r: int
    n_z: int

    @classmethod
    def refined(cls, grid_r, grid_z, factor: int = 2) -> "SolverGrid":
        """The box of a native grid with ``factor`` times finer steps, so that every native
        node is a solver node."""
        r = np.asarray(grid_r, dtype=np.float64)
        z = np.asarray(grid_z, dtype=np.float64)
        return cls(float(r[0]), float(r[-1]), float(z[0]), float(z[-1]), factor * (r.size - 1) + 1, factor * (z.size - 1) + 1)

    def axes(self) -> tuple[np.ndarray, np.ndarray]:
        return np.linspace(self.r_min, self.r_max, self.n_r), np.linspace(self.z_min, self.z_max, self.n_z)


def template_equilibrium(fm: FreegsMachine, grid: SolverGrid):
    """A FreeGS equilibrium on ``grid`` with free boundary, built once and copied per target
    so that the coil Green's functions on the grid are computed once."""
    _, _, equilibrium, *_unused, boundary, _ = _freegs()
    return equilibrium.Equilibrium(tokamak=fm.machine, Rmin=grid.r_min, Rmax=grid.r_max, Zmin=grid.z_min, Zmax=grid.z_max,
                                   nx=grid.n_r, ny=grid.n_z, boundary=boundary.freeBoundaryHagenow)


@dataclass
class SolveOutcome:
    """Result of one target.

    Attributes:
        target: The target.
        converged: Whether a solution passed every check.
        reason: Empty when converged, otherwise why the target was rejected.
        profile: Profile constraint that produced the solution (``betap_ip`` or ``paxis_ip``).
        equilibrium: The FreeGS equilibrium (None when the solve raised).
        iterations_s: Wall time of the solve, s.
    """

    target: EquilibriumTarget
    converged: bool
    reason: str
    profile: str = ""
    equilibrium: object = None
    iterations_s: float = 0.0


# Checks of a solution.
IP_RTOL = 0.01
MAX_XPOINT_DISTANCE_M = 0.15


def current_limits_a(fm: FreegsMachine, limits_ka: Mapping[str, tuple[float, float]] | None) -> list | None:
    """FreeGS's ``current_lims``: per coil of the machine, the bounds of its element current in
    amperes (unbounded for a column without limits; ignored by FreeGS for fixed columns)."""
    if not limits_ka:
        return None
    out = []
    for column in fm.columns:
        if column in limits_ka:
            lo, hi = (element_current_a(fm.cfg, fm.geometry, fm.scales, column, v) for v in limits_ka[column])
            out.append((min(lo, hi), max(lo, hi)))
        else:
            out.append((-np.inf, np.inf))
    return out


def _solve_once(template, fm: FreegsMachine, target: EquilibriumTarget, profile_kind: str, paxis_pa: float | None,
                rtol: float, maxits: int, limits_ka: Mapping[str, tuple[float, float]] | None = None):
    _, _, _, _, jtor, _, picard, _, _, control = _freegs()
    eq = copy.deepcopy(template)
    eq._updatePlasmaPsi(initial_psi(target, eq.R, eq.Z))
    fm.set_currents_ka({c: 0.0 for c in fm.columns}, eq.tokamak)
    fm.set_currents_ka({fm.cfg.solenoid_column: target.sol_ka}, eq.tokamak)
    fvac = target.fvac(fm.cfg)
    ip_a = target.ip_ka * A_PER_KA
    if profile_kind == "betap_ip":
        profile = jtor.ConstrainBetapIp(eq, target.beta_p, ip_a, fvac, alpha_m=target.alpha_m, alpha_n=target.alpha_n,
                                        Raxis=target.r_geo_m)
    else:
        profile = jtor.ConstrainPaxisIp(eq, paxis_pa, ip_a, fvac, alpha_m=target.alpha_m, alpha_n=target.alpha_n,
                                        Raxis=target.r_geo_m)
    xpoints, isoflux = shape_constraints(target)
    controller = control.constrain(xpoints=xpoints, isoflux=isoflux, gamma=1e-12,
                                   current_lims=current_limits_a(fm, limits_ka))
    error = ""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            picard.solve(eq, profile, controller, rtol=rtol, maxits=maxits, show=False)
    except Exception as exc:  # FreeGS raises RuntimeError, ValueError and others on failure
        error = f"{type(exc).__name__}: {exc}"[:200]
    return eq, profile, error


def check_solution(eq, target: EquilibriumTarget) -> str:
    """Empty when the solution is acceptable, otherwise the first failed check: plasma current
    within ``IP_RTOL`` of the target, an O-point, the target's X-points found within
    ``MAX_XPOINT_DISTANCE_M``, and a finite map."""
    from freegs import critical

    psi = eq.psi()
    if not np.all(np.isfinite(psi)):
        return "non-finite flux"
    ip = eq.plasmaCurrent()
    if not np.isfinite(ip) or abs(ip - target.ip_ka * A_PER_KA) > IP_RTOL * target.ip_ka * A_PER_KA:
        return f"plasma current {ip / A_PER_KA:.1f} kA, target {target.ip_ka:.1f} kA"
    opoints, xpoints = critical.find_critical(eq.R, eq.Z, psi)
    if not opoints:
        return "no O-point"
    wanted, _ = shape_constraints(target)
    for wr, wz in wanted:
        d = min((math.hypot(x[0] - wr, x[1] - wz) for x in xpoints), default=np.inf)
        if d > MAX_XPOINT_DISTANCE_M:
            return f"no X-point within {MAX_XPOINT_DISTANCE_M} m of ({wr:.2f}, {wz:.2f})"
    return ""


def solve_target(template, fm: FreegsMachine, target: EquilibriumTarget, rtol: float = 1e-4, maxits: int = 80,
                 limits_ka: Mapping[str, tuple[float, float]] | None = None) -> SolveOutcome:
    """Solves one target: ``ConstrainBetapIp`` first, ``ConstrainPaxisIp`` at the axis pressure
    the first attempt reached when the first attempt fails. ``limits_ka`` bounds the controlled
    columns (``DEFAULT_CURRENT_LIMITS_KA``); None leaves them free."""
    import time

    t0 = time.time()
    eq, profile, error = _solve_once(template, fm, target, "betap_ip", None, rtol, maxits, limits_ka)
    reason = error or check_solution(eq, target)
    if not reason:
        return SolveOutcome(target, True, "", "betap_ip", eq, time.time() - t0)
    paxis = None
    try:
        paxis = float(eq.pressure(0.0))
    except Exception:  # no profile parameters were fitted before the failure
        paxis = None
    if paxis is not None and np.isfinite(paxis) and paxis > 0:
        eq2, _, error2 = _solve_once(template, fm, target, "paxis_ip", paxis, rtol, maxits, limits_ka)
        reason2 = error2 or check_solution(eq2, target)
        if not reason2:
            return SolveOutcome(target, True, "", "paxis_ip", eq2, time.time() - t0)
        reason = f"betap_ip: {reason}; paxis_ip: {reason2}"
    return SolveOutcome(target, False, reason, "", None, time.time() - t0)


# ----------------------------------------------------------------------- native-grid maps
def interpolate_to_grid(values_rz, solver_r, solver_z, grid_r, grid_z) -> np.ndarray:
    """A map on the solver grid (FreeGS layout ``[iR, iZ]``) at the nodes of a native grid,
    returned in the package layout ``[iZ, iR]``; bicubic, exact at shared nodes."""
    spline = RectBivariateSpline(np.asarray(solver_r, dtype=np.float64), np.asarray(solver_z, dtype=np.float64),
                                 np.asarray(values_rz, dtype=np.float64), kx=3, ky=3)
    return spline(np.asarray(grid_r, dtype=np.float64), np.asarray(grid_z, dtype=np.float64)).T


def native_maps(eq, fm: FreegsMachine, grid_r, grid_z) -> tuple[np.ndarray, np.ndarray]:
    """``(psi, psi_vacuum)`` on the native grid, ``[iZ, iR]`` in Wb/rad: the plasma's own flux
    interpolated from the solver grid plus the vacuum flux evaluated at the native nodes."""
    rr, zz = np.meshgrid(np.asarray(grid_r, dtype=np.float64), np.asarray(grid_z, dtype=np.float64))
    vacuum = np.asarray(eq.tokamak.psi(rr, zz), dtype=np.float64)
    plasma = interpolate_to_grid(eq.plasma_psi, eq.R[:, 0], eq.Z[0, :], grid_r, grid_z)
    return plasma + vacuum, vacuum


def _safe(fn, default=np.nan) -> float:
    try:
        v = float(fn())
        return v if np.isfinite(v) else default
    except Exception:  # FreeGS diagnostics raise on unusual shapes; the value is then missing
        return default


def equilibrium_record(outcome: SolveOutcome, fm: FreegsMachine, grid_r, grid_z, meta: Mapping | None = None) -> dict:
    """The arrays and scalars written per converged equilibrium.

    Keys: ``psi`` and ``psi_vacuum`` (native grid, Wb/rad), ``grid_r``, ``grid_z`` (m),
    ``columns`` and ``currents_ka`` (FreeGS's coil currents in the dataset's units), ``ip_ka``,
    ``tf_ka``, ``beta_p``, ``li``, ``li1``, ``li3`` (FreeGS ``internalInductance2``, ``1`` and
    ``3``; the scorer's li, computed from the map, is none of them), ``q95`` (``scalars.q95_from_map`` on the native map; FreeGS's ``q`` fails under the
    pinned NumPy), ``beta_n`` (FreeGS, percent m T / MA), ``r_axis_m``, ``z_axis_m``, ``kappa``,
    ``delta_upper``, ``delta_lower``, ``target`` (JSON), ``profile``, ``meta`` (JSON).
    """
    eq = outcome.equilibrium
    psi, vacuum = native_maps(eq, fm, grid_r, grid_z)
    currents = fm.currents_ka(eq.tokamak)
    columns = list(fm.columns)
    return {
        "version": np.array(RECORD_VERSION),
        "psi": psi, "psi_vacuum": vacuum,
        "grid_r": np.asarray(grid_r, dtype=np.float64), "grid_z": np.asarray(grid_z, dtype=np.float64),
        "columns": np.array(columns), "currents_ka": np.array([currents[c] for c in columns], dtype=np.float64),
        "ip_ka": np.array(eq.plasmaCurrent() / A_PER_KA), "tf_ka": np.array(outcome.target.tf_ka),
        "beta_p": np.array(_safe(eq.poloidalBeta)), "li": np.array(_safe(eq.internalInductance2)),
        "li1": np.array(_safe(eq.internalInductance1)), "li3": np.array(_safe(eq.internalInductance3)),
        "q95": np.array(_safe(lambda: scalars.q95_from_map(psi, grid_r, grid_z, outcome.target.tf_ka,
                                                            fm.cfg.tf_turns.value, fm.cfg))),
        "beta_n": np.array(_safe(eq.betaN)),
        "r_axis_m": np.array(_safe(eq.Rmagnetic)), "z_axis_m": np.array(_safe(eq.Zmagnetic)),
        "kappa": np.array(_safe(eq.elongation)), "delta_upper": np.array(_safe(eq.triangularityUpper)),
        "delta_lower": np.array(_safe(eq.triangularityLower)),
        "target": np.array(json.dumps(outcome.target.to_dict(), sort_keys=True)), "profile": np.array(outcome.profile),
        "meta": np.array(json.dumps(dict(meta or {}), sort_keys=True)),
    }


def record_path(folder: Path | str, index: int) -> Path:
    return Path(folder) / f"eq_{int(index):05d}.npz"


def write_record(path: Path | str, record: Mapping[str, np.ndarray]) -> None:
    """Writes one record as an npz without pickled objects, atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **record)
    tmp.replace(path)


def load_record(path: Path | str) -> dict:
    """Reads a record written by ``write_record``; ``target`` and ``meta`` as dicts, strings as
    str, scalars as float."""
    out = {}
    with np.load(path, allow_pickle=False) as f:
        for key in f.files:
            v = f[key]
            if key in ("target", "meta"):
                out[key] = json.loads(str(v))
            elif key == "profile":
                out[key] = str(v)
            elif key == "columns":
                out[key] = [str(c) for c in v]
            elif v.ndim == 0:
                out[key] = float(v)
            else:
                out[key] = v
    return out


def list_records(folder: Path | str) -> list[Path]:
    """Record files of a folder in index order."""
    return sorted(Path(folder).glob("eq_[0-9][0-9][0-9][0-9][0-9].npz"))


# -------------------------------------------------------------------------------- scoring
@lru_cache(maxsize=1)
def mast_local_score_module() -> ModuleType:
    """A separate import of the organizers' ``local_score.py``, unmodified on disk, with its
    module-level ``MACHINE`` set to MAST, so that ``build_reference`` and ``score_shot`` use the
    MAST axis sign; the envelope mask is passed by the caller."""
    starter_kit.scoring()
    path = starter_kit.require() / "local_score.py"
    spec = importlib.util.spec_from_file_location("starter_local_score_mast", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.MACHINE = MACHINE
    return module


@dataclass
class SyntheticTruth:
    """The scorer's view of one synthetic frame, scored as a one-frame shot.

    Attributes:
        psi: ``(1, nZ, nR)`` true map, stored frame.
        q95, beta_n: ``(1,)`` true scalars.
        reference: ``(contours, cons, cmask, rgeo)`` from ``local_score.build_reference``.
        stats: The pooled sums (``baseline.reference_stats``).
    """

    psi: np.ndarray
    q95: np.ndarray
    beta_n: np.ndarray
    reference: tuple
    stats: dict

    def scorer_dict(self) -> dict:
        return {"psi": self.psi, "q95": self.q95, "betaN": self.beta_n}


def synthetic_truth(psi, q95: float, beta_n: float) -> SyntheticTruth:
    """The scorer's reference of one synthetic frame on the MAST grid."""
    from fusion_eq import baseline as bl

    ls = mast_local_score_module()
    env = starter_kit.envelope_mask(MACHINE)
    psi = np.asarray(psi, dtype=np.float64).reshape(1, env.grid_Z.size, env.grid_R.size)
    q = np.array([q95], dtype=np.float64)
    b = np.array([beta_n], dtype=np.float64)
    ref = ls.build_reference(psi, env.grid_R, env.grid_Z, env.mask_coarse, env.mask_f)
    stats = bl.reference_stats({"psi": psi, "q95": q, "betaN": b}, ref[1], ref[2])
    return SyntheticTruth(psi=psi, q95=q, beta_n=b, reference=ref, stats=stats)


def means_of(truths: Sequence[SyntheticTruth]) -> tuple:
    """``(mean_psi, mean_scal, mean_cons)`` over a set of truths, the scorer's fill values."""
    tot = {k: sum(np.asarray(t.stats[k], dtype=np.float64) for t in truths) for k in truths[0].stats}
    scal_n, cons_n = tot["scal_n"], tot["cons_n"]
    return (float(tot["psi_sum"] / tot["psi_n"]) if tot["psi_n"] else 0.0,
            np.where(scal_n > 0, tot["scal_sum"] / np.where(scal_n > 0, scal_n, 1), 0.0),
            np.where(cons_n > 0, tot["cons_sum"] / np.where(cons_n > 0, cons_n, 1), 0.0))


def global_psi_sign(truths: Sequence[SyntheticTruth], predictions: Sequence[np.ndarray], mean_psi: float) -> float:
    """The scorer's one global flux sign over a set of frames (its probe pass)."""
    ls = mast_local_score_module()
    totals = {s: 0.0 for s in ls.PSI_SIGNS}
    for t, p in zip(truths, predictions):
        rr, _, _ = ls.psi_residuals(t.psi, np.asarray(p, dtype=np.float64).reshape(t.psi.shape), mean_psi)
        for s in ls.PSI_SIGNS:
            totals[s] += rr[s]
    return min(ls.PSI_SIGNS, key=lambda s: totals[s])


def score_part(truth: SyntheticTruth, psi_pred, psi_sign: float, means: tuple, q95=None, beta_n=None) -> dict:
    """``local_score.score_shot`` of one frame on MAST; the true scalars unless given."""
    ls = mast_local_score_module()
    env = starter_kit.envelope_mask(MACHINE)
    pred = {"psirz": np.asarray(psi_pred, dtype=np.float64).reshape(truth.psi.shape),
            "q95": truth.q95 if q95 is None else np.atleast_1d(np.asarray(q95, dtype=np.float64)),
            "betaN": truth.beta_n if beta_n is None else np.atleast_1d(np.asarray(beta_n, dtype=np.float64))}
    return ls.score_shot(truth.scorer_dict(), truth.reference, pred, env.grid_R, env.grid_Z, env.mask_coarse, env.mask_f,
                         psi_sign, means)


def frame_metrics(truth: SyntheticTruth, psi_pred, psi_sign: float) -> dict:
    """Per-frame quantities behind the pooled terms, from the scorer's functions: the boundary
    distance in metres (``d_lcfs_m``, symmetric Hausdorff; NaN when either boundary is missing),
    the same over the reference's major radius (``d_lcfs_rel``, capped at one as the scorer
    does, one when the predicted boundary is missing), and the seven derived scalars of the
    predicted map (``pred_<name>``) beside the truth's (``true_<name>``)."""
    ls = mast_local_score_module()
    env = starter_kit.envelope_mask(MACHINE)
    sk = starter_kit.scoring()
    psi = np.asarray(psi_pred, dtype=np.float64).reshape(truth.psi.shape)[0]
    contours, cons, _, rgeo = truth.reference
    axis_sign = ls.AXIS_SIGN[MACHINE] * psi_sign
    ex, _ = ls.extract_lcfs_with_sign(psi, env.grid_R, env.grid_Z, MACHINE, env.mask_coarse, env.mask_f, n_iter=ls.N_ITER,
                                      axis_sign=axis_sign)
    ref = contours[0]
    out = {"d_lcfs_m": np.nan, "d_lcfs_rel": np.nan}
    if ref is not None:
        if ex is None:
            out["d_lcfs_rel"] = 1.0
        else:
            d = float(sk.contour.symmetric_hausdorff(ex, np.asarray(ref, dtype=np.float64)))
            out["d_lcfs_m"] = d
            out["d_lcfs_rel"] = min(1.0, d / rgeo) if rgeo > 0 else 1.0
    values = ls.derive_frame(psi, env.grid_R, env.grid_Z, MACHINE, env.mask_coarse, env.mask_f, contour=ex, axis_sign=axis_sign)
    for j, name in enumerate(ls.CONS_SCALARS):
        out[f"pred_{name}"] = float(values[name])
        out[f"true_{name}"] = float(cons[0, j])
    return out


def finalize_parts(parts: Sequence[dict], stats: Sequence[dict], psi_sign: float, mask=None) -> dict:
    """S, its four terms and the per-scalar R2 over the selected frames
    (``baseline.LocalScore.finalize`` and ``baseline.score_terms``)."""
    from fusion_eq import baseline as bl

    ids = [str(i) for i in range(len(parts))]
    result = bl.LocalScore(shot_ids=ids, parts=list(parts), ref_stats=list(stats), psi_sign=psi_sign, frame_stride=1).finalize(mask)
    return bl.score_terms(result)


# --------------------------------------------------------------------------- the ceiling
def project_share(share, basis: cb.CurrentBasis, place: Placement) -> np.ndarray:
    """Coefficients whose placed share is closest to a native share in the least-squares sense,
    ``argmin_c || P (s_mean + B^T c) - s ||``: the analogue on another machine's grid of
    ``c = B (s - s_mean)``, which it equals where placement is the identity."""
    s = np.asarray(share, dtype=np.float64).reshape(-1)
    mean = place.apply(basis.mean.reshape(basis.shape)).ravel()
    comps = place.apply(basis.components.reshape(-1, *basis.shape)).reshape(basis.n_components, -1)
    c, *_ = np.linalg.lstsq(comps.T, s - mean, rcond=None)
    return c


def project_flux(plasma_flux, ip_ka: float, flux_basis: cb.MachineFluxBasis) -> np.ndarray:
    """Coefficients whose flux ``Ip (phi_mean + Phi c)`` is closest to a plasma flux over every
    node: the best map the basis can give with an exact coil flux (an upper bound of the
    share-space projection for R2 of the flux)."""
    y = np.asarray(plasma_flux, dtype=np.float64).ravel() / (float(ip_ka) * A_PER_KA) - flux_basis.phi_mean
    c, *_ = np.linalg.lstsq(flux_basis.phi.T, y, rcond=None)
    return c


def lattice_footprint(place: Placement) -> np.ndarray:
    """Native cells that receive weight from the reference lattice, ``(nZ, nR)`` bool."""
    return np.outer(place.wy.sum(axis=1) > 0, place.wx.sum(axis=1) > 0)


VARIANTS = ("truth", "exact_share", "basis_share", "basis_flux", "mean_share")


@dataclass
class FrameEvaluation:
    """One synthetic frame through the target chain.

    Attributes:
        maps: Variant name to the rebuilt stored-frame map ``(nZ, nR)``.
        coefficients: ``basis_share`` and ``basis_flux`` coefficients.
        diagnostics: Share total, absolute share outside the boundary (all nodes, and nodes not
            in ``exclude``), share outside the reference lattice's footprint, relative share
            reconstruction error, boundary found.
    """

    maps: dict[str, np.ndarray]
    coefficients: dict[str, np.ndarray]
    diagnostics: dict[str, float] = field(default_factory=dict)


def evaluate_equilibrium(psi_stored, coil_flux_ours, ip_ka: float, grid_r, grid_z, basis: cb.CurrentBasis,
                         flux_basis: cb.MachineFluxBasis, place: Placement, response: ct.PlasmaResponse,
                         machine: str = MACHINE, exclude=None) -> FrameEvaluation:
    """Runs one labelled synthetic frame through the target chain.

    Variants: ``truth`` (the map itself, a scorer self-check), ``exact_share`` (coil flux plus
    the flux of the exact masked share: what the mask and the response keep), ``basis_share``
    (the share projected onto the basis through the placement, ``project_share``),
    ``basis_flux`` (``project_flux``) and ``mean_share`` (the basis mean alone). Every variant
    but ``truth`` is rebuilt from ``coil_flux_ours`` and has no remainder, as on MAST.

    Args:
        psi_stored: ``(nZ, nR)`` true map, stored frame.
        coil_flux_ours: ``(nZ, nR)`` coil flux of the adopted coil model with the frame's
            currents, physical frame, Wb/rad.
        ip_ka: Signed plasma current, kA.
        grid_r, grid_z: Native grid in metres.
        basis, flux_basis, place, response: The stored target objects of the machine.
        machine: Machine with an established sign rule.
        exclude: ``(nZ, nR)`` bool nodes left out of ``outside_abs_share_kept`` (the nodes near
            coil elements, where the current density of the plasma flux holds the difference
            between two coil discretisations).
    """
    psi = np.asarray(psi_stored, dtype=np.float64)
    coil = np.asarray(coil_flux_ours, dtype=np.float64)
    ip = np.array([float(ip_ka)])
    share_res = cb.current_share(psi[None], coil[None], ip, grid_r, grid_z, machine)
    share = share_res.share[0]
    plasma = targets.to_physical(psi, ip[0], machine) - coil
    c_share = project_share(share, basis, place)
    c_flux = project_flux(plasma, ip[0], flux_basis)
    maps = {
        "truth": psi.copy(),
        "exact_share": targets.to_stored(coil + ip[0] * A_PER_KA * cb.cell_flux(response, share), ip[0], machine),
        "basis_share": cb.compose_map(machine, coil[None], ip, c_share[None], flux_basis)[0],
        "basis_flux": cb.compose_map(machine, coil[None], ip, c_flux[None], flux_basis)[0],
        "mean_share": cb.compose_map(machine, coil[None], ip, np.zeros((1, flux_basis.n_components)), flux_basis)[0],
    }
    placed = (place.apply(basis.mean.reshape(basis.shape)) + place.apply(
        (c_share @ basis.components).reshape(basis.shape)))
    footprint = lattice_footprint(place)
    abs_total = float(np.abs(share).sum())
    j_abs = np.abs(ct.plasma_current_from_stored(psi[None], coil[None], grid_r, grid_z, ip, machine)[0])
    kept = ~share_res.mask[0] if exclude is None else ~share_res.mask[0] & ~np.asarray(exclude, dtype=bool)
    kept_total = float(j_abs[share_res.mask[0]].sum() + j_abs[kept].sum())
    diagnostics = {
        "share_total": float(share_res.total[0]),
        "outside_abs_share": float(share_res.outside_abs_share[0]),
        "outside_abs_share_kept": float(j_abs[kept].sum()) / kept_total if kept_total > 0 else np.nan,
        "outside_footprint_abs_share": float(np.abs(share[~footprint]).sum()) / abs_total if abs_total > 0 else np.nan,
        "share_rel_error": float(np.linalg.norm(placed - share) / np.linalg.norm(share)) if abs_total > 0 else np.nan,
        "boundary_found": float(share_res.boundary_found[0]),
    }
    return FrameEvaluation(maps=maps, coefficients={"basis_share": c_share, "basis_flux": c_flux}, diagnostics=diagnostics)


def rms_over_range(difference, reference) -> float:
    """RMS of a difference map over the range (max minus min) of a reference map."""
    d = np.asarray(difference, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    span = float(np.nanmax(ref) - np.nanmin(ref))
    return float(np.sqrt(np.nanmean(d ** 2)) / span) if span > 0 else np.nan


def contour_aspect_ratio(contour) -> float:
    """``(R_max + R_min) / (R_max - R_min)`` of a boundary contour ``(n, 2)`` of (R, Z)."""
    if contour is None or len(contour) == 0:
        return np.nan
    r = np.asarray(contour, dtype=np.float64)[:, 0]
    width = float(r.max() - r.min())
    return float(r.max() + r.min()) / width if width > 0 else np.nan


def bin_labels(values, edges: Sequence[float]) -> np.ndarray:
    """Interval labels ``[lo, hi)`` of values for bin edges; NaN values get ``missing``."""
    v = np.asarray(values, dtype=np.float64)
    e = list(edges)
    out = np.full(v.shape, "missing", dtype=object)
    for lo, hi in zip(e[:-1], e[1:]):
        inside = (v >= lo) & (v < hi)
        out[inside] = f"{lo:g}-{hi:g}"
    out[np.isfinite(v) & (v < e[0])] = f"<{e[0]:g}"
    out[np.isfinite(v) & (v >= e[-1])] = f">={e[-1]:g}"
    return out
