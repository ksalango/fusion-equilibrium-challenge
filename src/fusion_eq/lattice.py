"""Reference lattice of the current share and its placement on each machine's grid.

Scaled coordinates on each machine: ``x = (R - R0) / a0``, ``y = Z / a0``, with R0 and a0
from ``machines.py``. The reference lattice is the DIII-D flux grid in DIII-D's scaled
coordinates (65 by 65 nodes, each node owning one cell). The current share and its basis
live on it (``docs/TARGET.md``, "Reference lattice and placement").

Placement maps a share (a current per cell, without units) from the reference lattice to the
native cells of a machine by overlap of cell areas in scaled coordinates: each reference cell
gives each native cell the fraction of its own area that the native cell covers. The map is
linear and separable, ``P s = W_y s W_x^T``, and conserves the total of any share lying inside
the native grid; native cells outside the reference lattice receive nothing. On DIII-D the
native lattice is the reference lattice and placement is the identity.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fusion_eq import starter_kit
from fusion_eq.machines import MachineConfig, machine_config

REFERENCE_MACHINE = "DIII-D"
# Largest deviation of a lattice spacing from the median spacing, relative to the median.
UNIFORM_RTOL = 1e-5


@dataclass(frozen=True)
class Lattice:
    """Nodes of a uniform lattice in scaled coordinates; each node owns the cell centred on it.

    Attributes:
        x: ``(nR,)`` node positions along R, scaled.
        y: ``(nZ,)`` node positions along Z, scaled.
    """

    x: np.ndarray
    y: np.ndarray

    def __post_init__(self) -> None:
        for name in ("x", "y"):
            a = np.asarray(getattr(self, name), dtype=np.float64).ravel()
            d = np.diff(a)
            # The grids are stored as float32 values: the DIII-D R spacing varies by 9e-6 of
            # its median, so uniformity is judged relative to the median spacing.
            step = float(np.median(d)) if d.size else 0.0
            if a.size < 2 or step <= 0 or np.abs(d - step).max() > UNIFORM_RTOL * step:
                raise ValueError(f"lattice axis {name} must be increasing and uniform")
            object.__setattr__(self, name, a)

    @property
    def shape(self) -> tuple[int, int]:
        """``(nZ, nR)``, the order of the maps."""
        return self.y.size, self.x.size

    @property
    def dx(self) -> float:
        return float(np.diff(self.x).mean())

    @property
    def dy(self) -> float:
        return float(np.diff(self.y).mean())

    def x_edges(self) -> np.ndarray:
        return np.concatenate([self.x - 0.5 * self.dx, [self.x[-1] + 0.5 * self.dx]])

    def y_edges(self) -> np.ndarray:
        return np.concatenate([self.y - 0.5 * self.dy, [self.y[-1] + 0.5 * self.dy]])

    def in_metres(self, r0_m: float, a0_m: float) -> tuple[np.ndarray, np.ndarray]:
        """``(R, Z)`` node axes in metres for the radii given."""
        return r0_m + a0_m * self.x, a0_m * self.y


def machine_grid(machine: str | MachineConfig) -> tuple[np.ndarray, np.ndarray]:
    """``(grid_R, grid_Z)`` in metres of a machine's flux grid, from the scorer's envelope file."""
    envelope = starter_kit.envelope_mask(machine)
    return np.asarray(envelope.grid_R, dtype=np.float64), np.asarray(envelope.grid_Z, dtype=np.float64)


def _radii(cfg: MachineConfig, r0_m: float | None, a0_m: float | None) -> tuple[float, float]:
    return (cfg.major_radius.value if r0_m is None else float(r0_m)), (cfg.minor_radius.value if a0_m is None else float(a0_m))


def scaled_lattice(grid_r, grid_z, r0_m: float, a0_m: float) -> Lattice:
    """A grid in metres expressed in scaled coordinates."""
    return Lattice((np.asarray(grid_r, dtype=np.float64) - r0_m) / a0_m, np.asarray(grid_z, dtype=np.float64) / a0_m)


def reference_lattice(grid_r=None, grid_z=None) -> Lattice:
    """The DIII-D flux grid in DIII-D's scaled coordinates (``grid_r``, ``grid_z`` in metres;
    the scorer's DIII-D grid when None)."""
    cfg = machine_config(REFERENCE_MACHINE)
    if grid_r is None or grid_z is None:
        grid_r, grid_z = machine_grid(cfg)
    return scaled_lattice(grid_r, grid_z, cfg.major_radius.value, cfg.minor_radius.value)


def native_lattice(machine: str | MachineConfig, grid_r=None, grid_z=None, r0_m: float | None = None,
                   a0_m: float | None = None) -> Lattice:
    """A machine's flux grid in its own scaled coordinates.

    Args:
        machine: Machine name or configuration.
        grid_r, grid_z: Grid axes in metres; the scorer's grid of the machine when None.
        r0_m, a0_m: Radii used instead of the machine's constants (sensitivity checks).
    """
    cfg = machine if isinstance(machine, MachineConfig) else machine_config(machine)
    if grid_r is None or grid_z is None:
        grid_r, grid_z = machine_grid(cfg)
    return scaled_lattice(grid_r, grid_z, *_radii(cfg, r0_m, a0_m))


def overlap_matrix(source_edges, target_edges) -> np.ndarray:
    """``(n_target, n_source)`` fraction of each source cell's length that lies in each target
    cell, along one axis. A column sums to the share of its source cell inside the target span."""
    s = np.asarray(source_edges, dtype=np.float64)
    t = np.asarray(target_edges, dtype=np.float64)
    lo = np.maximum(t[:-1, None], s[None, :-1])
    hi = np.minimum(t[1:, None], s[None, 1:])
    return np.clip(hi - lo, 0.0, None) / np.diff(s)[None, :]


@dataclass
class Placement:
    """Conservative remap of a share between two lattices, separable in x and y.

    Attributes:
        machine: Machine of the native lattice.
        wx: ``(nR_target, nR_source)`` overlap fractions along x.
        wy: ``(nZ_target, nZ_source)`` overlap fractions along y.
        r0_m, a0_m: Radii of the native lattice.
        direction: ``to_native`` (reference to native) or ``to_reference``.
        identity: True when the two lattices coincide (DIII-D); ``apply`` then returns its input.
    """

    machine: str
    wx: np.ndarray
    wy: np.ndarray
    r0_m: float
    a0_m: float
    direction: str = "to_native"
    identity: bool = False

    def apply(self, shares) -> np.ndarray:
        """Placed shares ``(..., nZ_target, nR_target)`` of shares ``(..., nZ_source, nR_source)``."""
        s = np.asarray(shares, dtype=np.float64)
        if self.identity:
            return s.copy()
        if s.shape[-2:] != (self.wy.shape[1], self.wx.shape[1]):
            raise ValueError(f"share shape {s.shape[-2:]} does not match the placement {(self.wy.shape[1], self.wx.shape[1])}")
        return np.einsum("ij,...jk,lk->...il", self.wy, s, self.wx, optimize=True)

    def retained(self, shares) -> np.ndarray:
        """Per share, the total after placement over the total before (1 when nothing is lost);
        NaN for a share whose total is zero."""
        s = np.asarray(shares, dtype=np.float64)
        before = s.sum(axis=(-2, -1))
        after = self.apply(s).sum(axis=(-2, -1))
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(before != 0, after / before, np.nan)

    def save(self, path: Path | str) -> None:
        np.savez(path, machine=np.array(self.machine), wx=self.wx, wy=self.wy, r0_m=self.r0_m, a0_m=self.a0_m,
                 direction=np.array(self.direction), identity=self.identity)

    @classmethod
    def load(cls, path: Path | str) -> "Placement":
        with np.load(path, allow_pickle=False) as f:
            return cls(machine=str(f["machine"]), wx=f["wx"], wy=f["wy"], r0_m=float(f["r0_m"]), a0_m=float(f["a0_m"]),
                       direction=str(f["direction"]), identity=bool(f["identity"]))


def placement(machine: str | MachineConfig, reference: Lattice | None = None, native: Lattice | None = None,
              r0_m: float | None = None, a0_m: float | None = None) -> Placement:
    """Placement P_m of the reference lattice onto a machine's native cells.

    Args:
        machine: Machine name or configuration.
        reference: Reference lattice; ``reference_lattice()`` when None.
        native: Native lattice; ``native_lattice(machine, r0_m=..., a0_m=...)`` when None.
        r0_m, a0_m: Radii used instead of the machine's constants (sensitivity checks).
    """
    cfg = machine if isinstance(machine, MachineConfig) else machine_config(machine)
    r0, a0 = _radii(cfg, r0_m, a0_m)
    reference = reference_lattice() if reference is None else reference
    native = native_lattice(cfg, r0_m=r0, a0_m=a0) if native is None else native
    same = native.shape == reference.shape and np.array_equal(native.x, reference.x) and np.array_equal(native.y, reference.y)
    if same:
        return Placement(cfg.name, np.eye(native.x.size), np.eye(native.y.size), r0, a0, identity=True)
    return Placement(cfg.name, overlap_matrix(reference.x_edges(), native.x_edges()),
                     overlap_matrix(reference.y_edges(), native.y_edges()), r0, a0)


def back_placement(forward: Placement, reference: Lattice, native: Lattice) -> Placement:
    """The conservative remap from the native cells back to the reference lattice (the
    pessimistic placement test of ceiling variant i: placed and brought back)."""
    if forward.identity:
        return Placement(forward.machine, forward.wx, forward.wy, forward.r0_m, forward.a0_m, "to_reference", True)
    return Placement(forward.machine, overlap_matrix(native.x_edges(), reference.x_edges()),
                     overlap_matrix(native.y_edges(), reference.y_edges()), forward.r0_m, forward.a0_m, "to_reference")
