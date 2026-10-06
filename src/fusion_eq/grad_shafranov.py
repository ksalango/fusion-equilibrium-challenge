"""Grad-Shafranov operators on the EFIT grid.

With psi the poloidal flux per radian in Wb/rad, R in metres and J the toroidal
current density in A/m^2:

    delta_star(psi) = R d/dR((1/R) dpsi/dR) + d2psi/dZ2 = -mu0 R J
    Bz = (1/R) dpsi/dR                                    (vertical field, T)

The R part of ``delta_star`` is written in conservative form, as the difference of
``(1/R) dpsi/dR`` at the half-nodes, so that the current density summed over a region
depends only on the flux gradient along its boundary. The integrated current then
recovers the enclosed current to second order wherever the flux is smooth on the
boundary, which is what the current-consistency test relies on. Border nodes have no
centred stencil and are NaN.

``flux_from_current`` turns a current-density map back into flux at target nodes by
treating every source cell as one filament carrying ``J dR dZ``, through the same
Green's function the coil flux uses.
"""

from __future__ import annotations

import numpy as np
from matplotlib.path import Path as PolygonPath

from fusion_eq.coil_flux import GMD_FACTOR, MU0, filament_flux

__all__ = ["MU0", "delta_star", "current_density", "total_current", "vertical_field",
           "boundary_mask", "flux_from_current", "grid_steps"]


def grid_steps(grid_r, grid_z) -> tuple[float, float]:
    """(dR, dZ) of a uniform grid in metres.

    Raises:
        ValueError: when either axis is not uniform to 1e-5 m or has fewer than 3 nodes.
    """
    out = []
    for axis in (grid_r, grid_z):
        a = np.asarray(axis, dtype=np.float64).ravel()
        d = np.diff(a)
        if a.size < 3 or not np.allclose(d, d[0], rtol=0, atol=1e-5):
            raise ValueError("grid axis must be uniform with at least 3 nodes")
        out.append(float(d.mean()))
    return out[0], out[1]


def delta_star(psi, grid_r, grid_z) -> np.ndarray:
    """``R d/dR((1/R) dpsi/dR) + d2psi/dZ2`` on the interior nodes; NaN on the border.

    Args:
        psi: ``(nZ, nR)`` flux map, or ``(frames, nZ, nR)``.
        grid_r, grid_z: Uniform grid axes in metres.
    """
    p = np.asarray(psi, dtype=np.float64)
    single = p.ndim == 2
    p = p[None] if single else p
    r = np.asarray(grid_r, dtype=np.float64)
    d_r, d_z = grid_steps(grid_r, grid_z)
    out = np.full(p.shape, np.nan)
    # Conservative R part: (1/R) dpsi/dR at the half-nodes, differenced, times R.
    r_half = 0.5 * (r[1:] + r[:-1])                                  # (nR - 1,)
    g_half = (p[:, :, 1:] - p[:, :, :-1]) / d_r / r_half             # (F, nZ, nR - 1)
    radial = r[None, None, 1:-1] * (g_half[:, :, 1:] - g_half[:, :, :-1]) / d_r
    vertical = (p[:, 2:, :] - 2.0 * p[:, 1:-1, :] + p[:, :-2, :]) / d_z ** 2
    out[:, 1:-1, 1:-1] = radial[:, 1:-1, :] + vertical[:, :, 1:-1]
    return out[0] if single else out


def current_density(psi, grid_r, grid_z) -> np.ndarray:
    """Toroidal current density ``J = -delta_star(psi) / (mu0 R)`` in A/m^2; NaN on the border."""
    r = np.asarray(grid_r, dtype=np.float64)
    return -delta_star(psi, grid_r, grid_z) / (MU0 * r[None, :])


def total_current(j, grid_r, grid_z, mask=None) -> float:
    """Current in amperes of a density map summed over ``mask`` (all finite nodes when None)."""
    d_r, d_z = grid_steps(grid_r, grid_z)
    jj = np.asarray(j, dtype=np.float64)
    if mask is not None:
        jj = np.where(np.asarray(mask, dtype=bool), jj, 0.0)
    return float(np.nansum(jj) * d_r * d_z)


def vertical_field(psi, grid_r, grid_z) -> np.ndarray:
    """``Bz = (1/R) dpsi/dR`` in tesla, second-order differences including the R ends."""
    p = np.asarray(psi, dtype=np.float64)
    r = np.asarray(grid_r, dtype=np.float64)
    dpsi_dr = np.gradient(p, r, axis=-1, edge_order=2)
    return dpsi_dr / r


def boundary_mask(contour_rz, grid_r, grid_z) -> np.ndarray:
    """Grid nodes strictly inside a closed (R, Z) polygon, as a ``(nZ, nR)`` bool map.

    Returns an all-False map when the contour is None or has fewer than 3 points.
    """
    grid_r = np.asarray(grid_r, dtype=np.float64)
    grid_z = np.asarray(grid_z, dtype=np.float64)
    rr, zz = np.meshgrid(grid_r, grid_z)
    if contour_rz is None or len(contour_rz) < 3:
        return np.zeros(rr.shape, dtype=bool)
    path = PolygonPath(np.asarray(contour_rz, dtype=np.float64))
    return path.contains_points(np.column_stack([rr.ravel(), zz.ravel()])).reshape(rr.shape)


def flux_from_current(j, grid_r, grid_z, source_mask, target_mask=None,
                      min_distance_m: float | None = None) -> np.ndarray:
    """Flux at target nodes from the current in the source cells, in Wb/rad.

    Each source cell is one filament at its node carrying ``J dR dZ``. Targets default
    to every node outside the source mask; nodes that are not targets are NaN. The
    distance floor defaults to the cell's geometric mean distance, which only acts
    when a target is also a source.

    Args:
        j: ``(nZ, nR)`` current density in A/m^2; non-finite entries count as zero.
        grid_r, grid_z: Uniform grid axes in metres.
        source_mask: Bool map of the source cells.
        target_mask: Bool map of the nodes to evaluate; ``~source_mask`` when None.
        min_distance_m: Distance floor; cell geometric mean distance when None.
    """
    grid_r = np.asarray(grid_r, dtype=np.float64)
    grid_z = np.asarray(grid_z, dtype=np.float64)
    d_r, d_z = grid_steps(grid_r, grid_z)
    source = np.asarray(source_mask, dtype=bool)
    target = ~source if target_mask is None else np.asarray(target_mask, dtype=bool)
    jj = np.nan_to_num(np.asarray(j, dtype=np.float64))
    rr, zz = np.meshgrid(grid_r, grid_z)
    src_r, src_z, src_i = rr[source], zz[source], jj[source] * d_r * d_z
    out = np.full(rr.shape, np.nan)
    if src_r.size == 0:
        out[target] = 0.0
        return out
    floor = GMD_FACTOR * (d_r + d_z) if min_distance_m is None else min_distance_m
    tgt_r, tgt_z = rr[target][:, None], zz[target][:, None]
    values = np.zeros(tgt_r.shape[0])
    chunk = 256
    for start in range(0, src_r.size, chunk):
        sl = slice(start, start + chunk)
        values += filament_flux(tgt_r, tgt_z, src_r[None, sl], src_z[None, sl], src_i[None, sl], floor).sum(axis=1)
    out[target] = values
    return out
