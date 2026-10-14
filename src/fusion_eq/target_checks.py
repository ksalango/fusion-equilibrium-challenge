"""Calculations of the target checks Y1 to Y16 (``docs/TARGET.md``). The functions take arrays
and fitted objects and read no file; ``scripts/profile_targets.py`` reads the tables, the
objects and the maps and calls them.

Y9 compares the plasma response with the continuous filament Green's function. The response is
a discrete solver (F89): the difference is discretisation error, and the pass rule is set at
that level (beyond five cells, the largest difference over the largest reference value there,
at most 2e-2 on DIII-D and 1e-1 on MAST; beyond ten cells, the largest relative difference per
node, at most 5e-3 and 3e-2; symmetry in the two arguments to 1e-3 between source cells outside
the first three interior rings; the Z flip exact to rounding). The asymmetry of cells in those
rings is a property of the border treatment (F89); the guard that goes with it is the share of
the masked current of the fit sample in those rings, at most 1e-4 on each machine's grid.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

import numpy as np

from fusion_eq import current_basis as cb, current_target as ct, grad_shafranov as gs
from fusion_eq.coil_flux import CoilGeometry, filament_flux
from fusion_eq.coil_pattern import clearances
from fusion_eq.lattice import Placement

STATUSES = ("pass", "fail", "review")
# Y9 pass rule per machine (F89).
Y9_THRESHOLDS = {"DIII-D": {"beyond_5": 2e-2, "beyond_10": 5e-3}, "MAST": {"beyond_5": 1e-1, "beyond_10": 3e-2}}
Y9_SYMMETRY = 1e-3
# Symmetry is ruled between source cells outside the first Y9_RINGS interior rings; the share of
# the fit sample's absolute current in those rings is guarded at Y9_RING_SHARE.
Y9_RINGS = 3
Y9_RING_SHARE = 1e-4
# The Z flip is exact up to rounding of the linear solve.
Y9_FLIP = 1e-12


@dataclass(frozen=True)
class Check:
    """One row of the check table: id, title, status (pass, fail or review), summary."""

    id: str
    title: str
    status: str
    summary: str

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"check {self.id}: status {self.status!r} is not one of {STATUSES}")

    def to_dict(self) -> dict:
        return asdict(self)


def check_table(checks: Iterable[Check]) -> str:
    lines = ["| Id | Check | Status | Summary |", "| --- | --- | --- | --- |"]
    for c in checks:
        lines.append(f"| {c.id} | {c.title} | {c.status} | {c.summary.replace('|', '/')} |")
    return "\n".join(lines)


def quantiles(values, qs: Sequence[float] = (5, 50, 95)) -> dict:
    """Count and percentiles of the finite values."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    out = {"n": int(v.size)}
    for q in qs:
        out[f"q{q:g}"] = float(np.percentile(v, q)) if v.size else float("nan")
    return out


# ----------------------------------------------------------------------------- Y4, Y5
def parity_errors(basis: cb.CurrentBasis) -> dict:
    """Largest departure of each component from its parity and of the mean from evenness."""
    comps = basis.components.reshape(-1, *basis.shape)
    errors = np.abs(comps[:, ::-1] - basis.parity[:, None, None] * comps).reshape(len(comps), -1).max(axis=1)
    mean = basis.mean.reshape(basis.shape)
    return {"component_max": float(errors.max()) if errors.size else 0.0, "mean_max": float(np.abs(mean[::-1] - mean).max()),
            "even": int((basis.parity > 0).sum()), "odd": int((basis.parity < 0).sum())}


def mirror_errors(basis: cb.CurrentBasis, shares) -> dict:
    """Coefficients of the flipped shares against ``c`` times the parity, relative to the largest
    coefficient, and whether mirroring twice is the identity."""
    s = np.asarray(shares, dtype=np.float64)
    c = basis.transform(s)
    flipped = basis.transform(cb.mirror_share(s))
    scale = float(np.abs(c).max()) or 1.0
    twice = np.array_equal(cb.mirror_share(cb.mirror_share(s)), s) and np.array_equal(
        cb.mirror_coeffs(cb.mirror_coeffs(c, basis.parity), basis.parity), c)
    return {"max_relative": float(np.abs(flipped - cb.mirror_coeffs(c, basis.parity)).max() / scale), "twice_identity": bool(twice),
            "n_frames": int(len(s))}


# ---------------------------------------------------------------------------- Y7, Y8
def flux_basis_error(basis: cb.CurrentBasis, flux: cb.MachineFluxBasis, response: ct.PlasmaResponse, place: Placement,
                     coefficients) -> float:
    """Largest difference between ``phi_mean + Phi c`` and ``G P`` applied to the reconstructed
    shares, over the largest value of the latter (per ampere of Ip)."""
    c = np.asarray(coefficients, dtype=np.float64)
    direct = cb.cell_flux(response, place.apply(basis.inverse(c)))
    via_basis = flux.flux(np.full(len(c), 1e-3), c)            # 1e-3 kA: one ampere
    scale = float(np.abs(direct).max()) or 1.0
    return float(np.abs(via_basis - direct).max() / scale)


def placement_balance(place: Placement, shares) -> dict:
    """Totals of shares before and after placement and the weight that falls outside the
    target grid; ``max_imbalance`` is the largest of ``|after + outside - before|`` over the
    largest absolute total."""
    s = np.asarray(shares, dtype=np.float64)
    s = s[None] if s.ndim == 2 else s
    before = s.sum(axis=(1, 2))
    after = place.apply(s).sum(axis=(1, 2))
    kept = np.outer(place.wy.sum(axis=0), place.wx.sum(axis=0))
    outside = (s * (1.0 - kept)).sum(axis=(1, 2))
    scale = float(np.abs(before).max()) or 1.0
    return {"max_imbalance": float(np.abs(after + outside - before).max() / scale),
            "outside_share": quantiles(np.where(before != 0, outside / np.where(before != 0, before, 1.0), np.nan)),
            "n": int(len(s))}


# --------------------------------------------------------------------------------- Y9
def interior_sources(shape: tuple[int, int], step: int = 6) -> list[tuple[int, int]]:
    """Source cells on a lattice of the interior nodes (every ``step``-th node)."""
    return [(i, j) for i in range(1, shape[0] - 1, step) for j in range(1, shape[1] - 1, step)]


def border_distance(shape: tuple[int, int]) -> np.ndarray:
    """``(nZ, nR)`` distance of each cell from the grid border in cells: 0 on the border, k on
    the k-th interior ring."""
    i = np.arange(shape[0])[:, None]
    j = np.arange(shape[1])[None, :]
    return np.minimum(np.minimum(i, shape[0] - 1 - i), np.minimum(j, shape[1] - 1 - j))


def ring_shares(shares, rings: int = Y9_RINGS) -> dict[str, np.ndarray]:
    """Per frame, the share of the absolute current on the border cells and in the first
    ``rings`` interior rings, and the absolute total.

    Args:
        shares: ``(frames, nZ, nR)`` or ``(nZ, nR)`` shares on a machine's grid.
        rings: Number of interior rings counted.

    Returns:
        ``border`` and ``rings`` (fractions of the absolute total, NaN where it is zero) and
        ``total`` (the absolute total).
    """
    s = np.abs(np.asarray(shares, dtype=np.float64))
    s = s[None] if s.ndim == 2 else s
    d = border_distance(s.shape[1:])
    total = s.sum(axis=(1, 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        return {"border": np.where(total > 0, s[:, d == 0].sum(axis=1) / total, np.nan),
                "rings": np.where(total > 0, s[:, (d >= 1) & (d <= rings)].sum(axis=1) / total, np.nan), "total": total}


def ring_guard(parts: Sequence[dict[str, np.ndarray]], rings: int = Y9_RINGS) -> dict:
    """Summary of ``ring_shares`` over frames (several batches): the largest per-frame share in
    the first ``rings`` interior rings (the guarded value), its 99th percentile, the pooled
    share over all frames, and the largest per-frame share on the border cells."""
    frac = np.concatenate([p["rings"] for p in parts]) if parts else np.zeros(0)
    border = np.concatenate([p["border"] for p in parts]) if parts else np.zeros(0)
    total = np.concatenate([p["total"] for p in parts]) if parts else np.zeros(0)
    ok = np.isfinite(frac)
    pooled = float((frac[ok] * total[ok]).sum() / total[ok].sum()) if ok.any() and total[ok].sum() > 0 else float("nan")
    largest = float(frac[ok].max()) if ok.any() else float("nan")
    return {"n_frames": int(ok.sum()), "rings": rings, "max": largest,
            "q99": float(np.percentile(frac[ok], 99)) if ok.any() else float("nan"), "pooled": pooled,
            "border_max": float(np.nanmax(border)) if np.isfinite(border).any() else float("nan"),
            "rule": Y9_RING_SHARE, "pass": bool(ok.any() and largest <= Y9_RING_SHARE)}


def response_against_filaments(response: ct.PlasmaResponse, sources: Sequence[tuple[int, int]], rings: int = Y9_RINGS) -> dict:
    """Response to one ampere in each source cell against a filament at the cell's node.

    Returns the largest, over the sources, of: beyond five cells (Chebyshev distance in cells),
    the largest difference over the largest reference value there; beyond ten cells, the
    largest relative difference per node. Also the largest asymmetry of the mutual flux between
    two source cells outside the first ``rings`` interior rings (``symmetry``, the ruled value)
    and between any two source cells (``symmetry_all_sources``), and the largest relative
    difference between the response to the flipped currents and the flipped response.
    """
    r, z = response.grid_r, response.grid_z
    d_r, d_z = gs.grid_steps(r, z)
    rr, zz = np.meshgrid(r, z)
    shape = rr.shape
    units = np.zeros((len(sources), *shape))
    for k, (i, j) in enumerate(sources):
        units[k, i, j] = 1.0
    flux = cb.cell_flux(response, units)
    beyond_5, beyond_10 = 0.0, 0.0
    for k, (i, j) in enumerate(sources):
        ref = filament_flux(rr, zz, r[j], z[i], 1.0)
        dist = np.maximum(np.abs(rr - r[j]) / d_r, np.abs(zz - z[i]) / d_z)
        far5, far10 = dist > 5, dist > 10
        diff = np.abs(flux[k] - ref)
        if far5.any():
            beyond_5 = max(beyond_5, float(diff[far5].max() / np.abs(ref[far5]).max()))
        if far10.any():
            beyond_10 = max(beyond_10, float((diff[far10] / np.abs(ref[far10])).max()))
    mutual = np.array([[flux[a][sources[b]] for b in range(len(sources))] for a in range(len(sources))])
    with np.errstate(divide="ignore", invalid="ignore"):
        asym = np.abs(mutual - mutual.T) / np.maximum(np.abs(mutual), np.abs(mutual.T))
    distance = border_distance(shape)
    inner = np.array([distance[i, j] > rings for i, j in sources], dtype=bool)
    flipped = cb.cell_flux(response, units[:, ::-1])
    flip = float(np.abs(flipped - flux[:, ::-1]).max() / np.abs(flux).max())
    return {"n_sources": len(sources), "beyond_5": beyond_5, "beyond_10": beyond_10,
            "symmetry": float(np.nanmax(asym[np.ix_(inner, inner)])) if inner.sum() > 1 else float("nan"),
            "n_symmetry_sources": int(inner.sum()), "symmetry_rings_excluded": rings,
            "symmetry_all_sources": float(np.nanmax(asym)), "flip": flip}


def y9_status(machine: str, measured: dict) -> bool:
    """Whether the response meets the Y9 rule of F89 (the ring guard is judged separately)."""
    t = Y9_THRESHOLDS[machine]
    return (measured["beyond_5"] <= t["beyond_5"] and measured["beyond_10"] <= t["beyond_10"]
            and measured["symmetry"] <= Y9_SYMMETRY and measured["flip"] <= Y9_FLIP)


# ---------------------------------------------------------------------------- Y10, Y11
def support(shares, fraction: float = 0.99) -> np.ndarray:
    """Per share, the cells that hold ``fraction`` of its absolute total, largest first."""
    s = np.abs(np.asarray(shares, dtype=np.float64))
    s = s[None] if s.ndim == 2 else s
    flat = s.reshape(len(s), -1)
    order = np.argsort(-flat, axis=1, kind="stable")
    sorted_ = np.take_along_axis(flat, order, axis=1)
    cum = np.cumsum(sorted_, axis=1)
    total = cum[:, -1:]
    needed = (cum - sorted_) < fraction * total             # cells before the running total reaches the fraction
    out = np.zeros_like(flat, dtype=bool)
    np.put_along_axis(out, order, needed & (sorted_ > 0), axis=1)
    return out.reshape(s.shape)


def envelope(shares, fraction: float = 0.99, min_frames: float = 0.01) -> np.ndarray:
    """Cells inside the ``fraction`` support of at least ``min_frames`` of the shares: the
    envelope holding 99 percent of the training shares, the rarest cells left out."""
    occupancy = support(shares, fraction).mean(axis=0)
    return occupancy >= min_frames


def landing(mean_share, region, grid_r, grid_z, geometry: CoilGeometry) -> dict:
    """Where a region of cells and a mean share lie on a grid, in metres.

    Args:
        mean_share: ``(nZ, nR)`` placed mean share, for the centroid.
        region: ``(nZ, nR)`` bool cells (an envelope or a support).
        grid_r, grid_z: Node axes of the grid in metres.
        geometry: Coil geometry of the machine.

    Returns:
        Inner and outer radius and the height of the region (cell edges), the centroid of the
        mean share, the distance from the region's nodes to the nearest coil element and to the
        grid edge, and the coil elements whose centre lies in a region cell.
    """
    r = np.asarray(grid_r, dtype=np.float64)
    z = np.asarray(grid_z, dtype=np.float64)
    d_r, d_z = gs.grid_steps(r, z)
    rr, zz = np.meshgrid(r, z)
    m = np.asarray(mean_share, dtype=np.float64)
    out = {"centroid_r_m": float((m * rr).sum() / m.sum()), "centroid_z_m": float((m * zz).sum() / m.sum()),
           "n_cells": int(region.sum())}
    if not region.any():
        return {**out, "inner_r_m": float("nan"), "outer_r_m": float("nan"), "height_m": float("nan"),
                "coil_distance_m": float("nan"), "edge_distance_m": float("nan"), "coils_inside": []}
    pr, pz = rr[region], zz[region]
    dist, nearest = clearances(pr, pz, geometry)
    edge = np.minimum.reduce([pr - r[0], r[-1] - pr, pz - z[0], z[-1] - pz])
    i = np.clip(np.rint((geometry.z - z[0]) / d_z).astype(int), 0, z.size - 1)
    j = np.clip(np.rint((geometry.r - r[0]) / d_r).astype(int), 0, r.size - 1)
    on_grid = (geometry.r >= r[0] - d_r / 2) & (geometry.r <= r[-1] + d_r / 2) & (geometry.z >= z[0] - d_z / 2) & (geometry.z <= z[-1] + d_z / 2)
    inside = on_grid & region[i, j]
    return {**out, "inner_r_m": float(pr.min() - d_r / 2), "outer_r_m": float(pr.max() + d_r / 2),
            "height_m": float(pz.max() - pz.min() + d_z), "coil_distance_m": float(dist.min()),
            "nearest_coil": str(geometry.input_columns[nearest[int(np.argmin(dist))]]),
            "edge_distance_m": float(edge.min()),
            "coils_inside": sorted({str(c) for c in geometry.input_columns[inside]}), "n_elements_inside": int(inside.sum())}
