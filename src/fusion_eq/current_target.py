"""The current-distribution target: toroidal current density from a flux map, the plasma
mask, and the response that rebuilds the plasma's flux from a current density.

In the physical frame (``targets.to_physical``) the stored map is the coil flux of the
adopted coil model plus the plasma's own flux, ``psi_phys = coil + P``. The plasma current
density is ``J = -delta_star(P) / (mu0 R)`` (``grad_shafranov``), which integrates to the
measured Ip inside the scorer's last closed surface (F46); outside that surface it holds
the coil model's residual and numerical noise, which the plasma mask removes.

``PlasmaResponse`` inverts the operator on one machine's grid: the flux on the border nodes
is the Green's function of every interior cell's current (the same filament kernel the coil
flux uses), and the interior solves ``delta_star(psi) = -mu0 R J`` with those border values,
through one sparse LU factorisation of the conservative stencil. ``response(delta_star(P))``
returns ``P`` for any flux that is sourced inside the grid. The border Green's matrix is
cached under the processed-data root, keyed by the grid.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy.sparse as sps
from scipy.sparse.linalg import splu

from fusion_eq import dataset, grad_shafranov as gs, starter_kit, targets
from fusion_eq.coil_flux import GMD_FACTOR, MU0, filament_flux
from fusion_eq.io import sha256_of_arrays
from fusion_eq.machines import MachineConfig, machine_config

A_PER_KA = 1000.0
DEFAULT_CACHE_DIR = dataset.DEFAULT_PROCESSED_ROOT / "current_target"


# ------------------------------------------------------------------ current density
def current_from_map(psi_physical, grid_r, grid_z) -> np.ndarray:
    """Toroidal current density in A/m^2 of a flux map in the physical frame, border nodes 0.

    Args:
        psi_physical: ``(nZ, nR)`` or ``(frames, nZ, nR)`` flux in Wb/rad.
        grid_r, grid_z: Uniform grid axes in metres.
    """
    j = gs.current_density(psi_physical, grid_r, grid_z)
    return np.nan_to_num(j, nan=0.0)


def plasma_current_from_stored(psi_stored, coil, grid_r, grid_z, ip_ka, machine: str | MachineConfig = "DIII-D") -> np.ndarray:
    """Plasma current density of stored maps: ``current_from_map(to_physical(psi) - coil)``.

    Args:
        psi_stored: ``(frames, nZ, nR)`` or one map, as stored.
        coil: Coil flux of the same frames under the adopted coil model, Wb/rad.
        grid_r, grid_z: Grid axes in metres.
        ip_ka: Plasma current per frame in kA (sign gives the frame).
        machine: Machine with an established sign rule.
    """
    plasma = targets.to_physical(psi_stored, ip_ka, machine) - np.asarray(coil, dtype=np.float64)
    return current_from_map(plasma, grid_r, grid_z)


# ---------------------------------------------------------------------- plasma mask
def plasma_mask(psi_stored, grid_r, grid_z, machine: str | MachineConfig = "DIII-D", envelope=None, scorer=None) -> np.ndarray:
    """Grid nodes inside the scorer's last closed surface of one stored map (all False when
    no boundary is found)."""
    cfg = machine if isinstance(machine, MachineConfig) else machine_config(machine)
    scorer = scorer or starter_kit.scoring()
    envelope = envelope or starter_kit.envelope_mask(cfg)
    r = np.asarray(grid_r, dtype=np.float64)
    z = np.asarray(grid_z, dtype=np.float64)
    contour = scorer.lcfs.extract_lcfs(np.asarray(psi_stored, dtype=np.float64), r, z, cfg.name, envelope.mask_coarse, envelope.mask_f)
    return gs.boundary_mask(contour, r, z)


def mask_statistics(j, mask, grid_r, grid_z, ip_ka: float) -> dict:
    """Per-frame record of what the mask does to a current density.

    Returns ``dropped_abs_share`` (sum of ``|J|`` outside the mask over the total),
    ``masked_total_over_ip`` (masked current in A over Ip in A, NaN without Ip),
    ``n_inside`` and ``boundary_found``.
    """
    jj = np.nan_to_num(np.asarray(j, dtype=np.float64))
    m = np.asarray(mask, dtype=bool)
    total_abs = float(np.abs(jj).sum())
    dropped = float(np.abs(jj[~m]).sum()) / total_abs if total_abs > 0 else np.nan
    masked_a = gs.total_current(jj, grid_r, grid_z, m)
    ip_a = float(ip_ka) * A_PER_KA if np.isfinite(ip_ka) else np.nan
    return {"dropped_abs_share": dropped, "masked_total_over_ip": masked_a / ip_a if np.isfinite(ip_a) and ip_a != 0 else np.nan,
            "n_inside": int(m.sum()), "boundary_found": bool(m.any())}


# ------------------------------------------------------------------ plasma response
def _border_index(n_z: int, n_r: int) -> tuple[np.ndarray, np.ndarray]:
    """Row and column indices of the border nodes, in a fixed order."""
    rows, cols = np.meshgrid(np.arange(n_z), np.arange(n_r), indexing="ij")
    border = (rows == 0) | (rows == n_z - 1) | (cols == 0) | (cols == n_r - 1)
    return rows[border], cols[border]


def border_green_matrix(grid_r, grid_z) -> np.ndarray:
    """Flux at the border nodes per unit current density in each interior cell, Wb/rad per (A/m^2).

    Each interior cell is one filament at its node carrying ``J dR dZ`` (``flux_from_current``),
    with the cell's geometric mean distance as the floor. Shape ``(n_border, n_interior)``,
    interior cells in row-major order of the ``(nZ - 2, nR - 2)`` interior.
    """
    r = np.asarray(grid_r, dtype=np.float64)
    z = np.asarray(grid_z, dtype=np.float64)
    d_r, d_z = gs.grid_steps(r, z)
    b_rows, b_cols = _border_index(z.size, r.size)
    rr, zz = np.meshgrid(r[1:-1], z[1:-1])
    src_r, src_z = rr.ravel(), zz.ravel()
    out = np.zeros((b_rows.size, src_r.size))
    tgt_r, tgt_z = r[b_cols][:, None], z[b_rows][:, None]
    floor = GMD_FACTOR * (d_r + d_z)
    chunk = 256
    for start in range(0, src_r.size, chunk):
        sl = slice(start, start + chunk)
        out[:, sl] = filament_flux(tgt_r, tgt_z, src_r[None, sl], src_z[None, sl], d_r * d_z, floor)
    return out


def interior_operator(grid_r, grid_z) -> tuple[sps.csc_matrix, dict]:
    """The conservative ``delta_star`` stencil on the interior nodes as a sparse matrix, plus
    the coefficients that multiply the border neighbours (moved to the right-hand side).

    Returns ``(A, border_terms)`` with ``A`` of shape ``(n_interior, n_interior)`` acting on the
    interior flux in row-major order, and ``border_terms`` a dict ``(side) -> (interior index,
    border row, border col, coefficient)`` arrays for the four sides.
    """
    r = np.asarray(grid_r, dtype=np.float64)
    z = np.asarray(grid_z, dtype=np.float64)
    d_r, d_z = gs.grid_steps(r, z)
    n_z, n_r = z.size, r.size
    m_z, m_r = n_z - 2, n_r - 2
    r_half = 0.5 * (r[1:] + r[:-1])
    ii, jj = np.meshgrid(np.arange(1, n_z - 1), np.arange(1, n_r - 1), indexing="ij")
    ii, jj = ii.ravel(), jj.ravel()
    k = (ii - 1) * m_r + (jj - 1)
    c_east = r[jj] / (d_r ** 2 * r_half[jj])
    c_west = r[jj] / (d_r ** 2 * r_half[jj - 1])
    c_ns = np.full(k.size, 1.0 / d_z ** 2)
    diag = -(c_east + c_west) - 2.0 / d_z ** 2
    rows, cols, vals = [k], [k], [diag]
    border_terms = {}
    for name, di, dj, coef in (("east", 0, 1, c_east), ("west", 0, -1, c_west), ("north", 1, 0, c_ns), ("south", -1, 0, c_ns)):
        ni, nj = ii + di, jj + dj
        inside = (ni >= 1) & (ni <= n_z - 2) & (nj >= 1) & (nj <= n_r - 2)
        rows.append(k[inside]); cols.append((ni[inside] - 1) * m_r + (nj[inside] - 1)); vals.append(coef[inside])
        border_terms[name] = (k[~inside], ni[~inside], nj[~inside], coef[~inside])
    a = sps.csc_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(m_z * m_r, m_z * m_r))
    return a, border_terms


@dataclass
class PlasmaResponse:
    """Flux on the whole grid from a toroidal current density on the grid.

    Attributes:
        machine: Machine name.
        grid_r, grid_z: Grid axes in metres.
        green: Border Green's matrix (``border_green_matrix``).
    """

    machine: str
    grid_r: np.ndarray
    grid_z: np.ndarray
    green: np.ndarray
    _lu: object = field(default=None, repr=False, compare=False)
    _border_terms: dict = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.grid_r = np.asarray(self.grid_r, dtype=np.float64)
        self.grid_z = np.asarray(self.grid_z, dtype=np.float64)
        a, self._border_terms = interior_operator(self.grid_r, self.grid_z)
        self._lu = splu(a)

    @property
    def grid_hash(self) -> str:
        return sha256_of_arrays(self.grid_r, self.grid_z)

    @classmethod
    def build(cls, machine: str, grid_r, grid_z) -> "PlasmaResponse":
        return cls(machine=machine, grid_r=grid_r, grid_z=grid_z, green=border_green_matrix(grid_r, grid_z))

    @classmethod
    def load_or_build(cls, machine: str, grid_r, grid_z, cache_dir: Path | str = DEFAULT_CACHE_DIR) -> "PlasmaResponse":
        """The response of a grid from the cache (``response_<machine>_<grid hash>.npz``), built and
        written there when absent."""
        key = sha256_of_arrays(grid_r, grid_z)[:16]
        path = Path(cache_dir) / f"response_{machine_config(machine).file_prefix.split('_')[0]}_{key}.npz"
        if path.is_file():
            with np.load(path, allow_pickle=False) as f:
                return cls(machine=str(f["machine"]), grid_r=f["grid_r"], grid_z=f["grid_z"], green=f["green"])
        out = cls.build(machine, grid_r, grid_z)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.npz")
        np.savez(tmp, machine=np.array(machine), grid_r=out.grid_r, grid_z=out.grid_z, green=out.green)
        tmp.replace(path)
        return out

    def flux(self, j) -> np.ndarray:
        """Flux in Wb/rad on every node from a current density in A/m^2 on the interior nodes.

        Args:
            j: ``(nZ, nR)`` or ``(frames, nZ, nR)``; border values and non-finite entries are
                ignored (count as zero current).
        """
        jj = np.nan_to_num(np.asarray(j, dtype=np.float64))
        single = jj.ndim == 2
        jj = jj[None] if single else jj
        n_f, n_z, n_r = jj.shape
        if (n_z, n_r) != (self.grid_z.size, self.grid_r.size):
            raise ValueError(f"current density shape {(n_z, n_r)} does not match the grid")
        interior = jj[:, 1:-1, 1:-1].reshape(n_f, -1)                       # (F, n_interior)
        out = np.zeros((n_f, n_z, n_r))
        b_rows, b_cols = _border_index(n_z, n_r)
        out[:, b_rows, b_cols] = interior @ self.green.T                      # border from the Green's function
        rhs = -(MU0 * self.grid_r[None, None, 1:-1] * jj[:, 1:-1, 1:-1]).reshape(n_f, -1)
        for k, ni, nj, coef in self._border_terms.values():
            if k.size:
                np.add.at(rhs, (slice(None), k), -coef[None, :] * out[:, ni, nj])
        out[:, 1:-1, 1:-1] = self._lu.solve(rhs.T).T.reshape(n_f, n_z - 2, n_r - 2)
        return out[0] if single else out
