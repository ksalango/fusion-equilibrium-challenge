"""The current share, its basis with exact up-down parity, the flux it produces on each machine,
and the mirror on the target side (``docs/TARGET.md``, Parts 1, 2 and 4).

With ``psi_plasma = psi_phys - psi_coil`` (physical frame, adopted coil model) and ``J`` its
toroidal current density (``current_target.plasma_current_from_stored``), the share of a
frame is the masked cell current over the measured plasma current,

    s = J dR dZ mask / Ip,

without units and positive where the current flows with Ip. Its total over the grid is the
masked current over the measured Ip, not rescaled. Inside the scorer's boundary there is no
coil, so the current density of ``psi_plasma`` is that of ``psi_phys`` there (the definition
of the specification); the masked current is the one validated by variant e of F79.

The basis is the principal components of the share over originals and their mirrors. With
the mirrors included the covariance splits exactly into an even and an odd part, which are
diagonalised separately on the upper half of the grid and embedded by copying rows, so that
every component is exactly even or odd in Z and the mean is exactly even.

The flux of a share on machine ``m`` is ``Ip G_m P_m s``: placement onto the native cells
(``lattice.placement``) and the existing plasma response (``current_target.PlasmaResponse``),
which takes a current density; a cell current ``s Ip`` is the density ``s Ip / (dR dZ)``. The
response ignores current on the border nodes of the grid, as the target build does (the
current density is not defined there). The flux basis per machine holds
``phi_mean = G P s_mean`` and ``Phi = G P B`` in Wb/rad per ampere of Ip, so that the flux of
a predicted current is ``Ip (phi_mean + Phi c)``. ``compose_map`` assembles the stored map
``k sign(Ip) (psi_coil + Ip (phi_mean + Phi c) + r)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from fusion_eq import current_target as ct, grad_shafranov as gs, targets
from fusion_eq.coil_flux import GreenTable
from fusion_eq.lattice import Placement
from fusion_eq.machines import MachineConfig, machine_config

A_PER_KA = 1000.0
SQRT2 = np.sqrt(2.0)
# Label columns the mirror changes: the axis height changes sign, the triangularities exchange.
Z_COLUMNS = ("Z_axis", "z_axis", "efit_z_axis")
EXCHANGED_PAIRS = (("tri_top", "tri_bot"),)


# ------------------------------------------------------------------------------ share
@dataclass
class ShareResult:
    """Current share of frames and its diagnostics.

    Attributes:
        share: ``(frames, nZ, nR)`` masked cell current over Ip, no units.
        total: Per frame, the share summed over the grid (the masked current over Ip).
        outside_abs_share: Per frame, the share of the absolute current outside the boundary.
        boundary_found: Per frame, whether the boundary was found (otherwise the share is zero).
        mask: ``(frames, nZ, nR)`` bool, the nodes inside the boundary.
    """

    share: np.ndarray
    total: np.ndarray
    outside_abs_share: np.ndarray
    boundary_found: np.ndarray
    mask: np.ndarray


def current_share(psi_stored, coil, ip_ka, grid_r, grid_z, machine: str | MachineConfig = "DIII-D", masks=None,
                  envelope=None, scorer=None) -> ShareResult:
    """The share of every frame, from the stored maps, wrapping the masked current of
    ``current_target``.

    Args:
        psi_stored: ``(frames, nZ, nR)`` or one map, as stored.
        coil: Coil flux of the same frames in the physical frame, Wb/rad.
        ip_ka: Signed plasma current per frame in kA; frames without a finite, non-zero Ip
            get a zero share.
        grid_r, grid_z: Grid axes in metres.
        machine: Machine with an established sign rule.
        masks: Bool masks of the plasma; the scorer's boundary of each stored map
            (``current_target.plasma_mask``) when None.
        envelope, scorer: Passed to ``plasma_mask``.
    """
    psi = np.asarray(psi_stored, dtype=np.float64)
    psi = psi[None] if psi.ndim == 2 else psi
    ip = np.atleast_1d(np.asarray(ip_ka, dtype=np.float64))
    coil = np.asarray(coil, dtype=np.float64).reshape(psi.shape)
    if masks is None:
        masks = np.stack([ct.plasma_mask(p, grid_r, grid_z, machine, envelope, scorer) for p in psi])
    masks = np.asarray(masks, dtype=bool).reshape(psi.shape)
    j = ct.plasma_current_from_stored(psi, coil, grid_r, grid_z, ip, machine)
    d_r, d_z = gs.grid_steps(grid_r, grid_z)
    ip_a = ip * A_PER_KA
    ok = np.isfinite(ip_a) & (ip_a != 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        share = np.where(ok[:, None, None], j * masks * d_r * d_z / np.where(ok, ip_a, 1.0)[:, None, None], 0.0)
    stats = [ct.mask_statistics(j[k], masks[k], grid_r, grid_z, ip[k]) for k in range(len(ip))]
    return ShareResult(share=share, total=share.sum(axis=(1, 2)),
                       outside_abs_share=np.array([s["dropped_abs_share"] for s in stats], dtype=np.float64),
                       boundary_found=np.array([s["boundary_found"] for s in stats], dtype=bool), mask=masks)


# ------------------------------------------------------------------------- parity helpers
def _half_rows(n_z: int) -> tuple[int, bool]:
    """Rows strictly above the midplane, and whether a midplane row exists."""
    return n_z // 2, bool(n_z % 2)


def _even_coords(maps) -> np.ndarray:
    """Even parts ``(s + flip s) / 2`` of maps ``(F, nZ, nR)`` in weighted half coordinates:
    the rows above the midplane times sqrt(2) and the midplane row, so that inner products
    equal those of the full even maps."""
    m = np.asarray(maps, dtype=np.float64)
    n_half, mid = _half_rows(m.shape[1])
    top = 0.5 * (m[:, :n_half] + m[:, ::-1][:, :n_half]) * SQRT2
    parts = [top.reshape(len(m), -1)]
    if mid:
        parts.append(m[:, n_half])
    return np.concatenate(parts, axis=1)


def _odd_coords(maps) -> np.ndarray:
    """Odd parts ``(s - flip s) / 2`` in weighted half coordinates (the midplane row is zero)."""
    m = np.asarray(maps, dtype=np.float64)
    n_half, _ = _half_rows(m.shape[1])
    return (0.5 * (m[:, :n_half] - m[:, ::-1][:, :n_half]) * SQRT2).reshape(len(m), -1)


def _embed(u, shape: tuple[int, int], parity: int) -> np.ndarray:
    """Full maps ``(k, nZ, nR)`` from weighted half coordinates, exactly even (``parity`` +1)
    or odd (-1) by copying the rows."""
    u = np.atleast_2d(np.asarray(u, dtype=np.float64))
    n_z, n_r = shape
    n_half, mid = _half_rows(n_z)
    top = u[:, :n_half * n_r].reshape(len(u), n_half, n_r) / SQRT2
    out = np.zeros((len(u), n_z, n_r))
    out[:, :n_half] = top
    out[:, n_z - 1 - np.arange(n_half)] = parity * top
    if mid and parity > 0:
        out[:, n_half] = u[:, n_half * n_r:]
    return out


# ---------------------------------------------------------------------------- basis
@dataclass
class CurrentBasis:
    """Principal components of the share over originals and mirrors, with exact parity.

    Attributes:
        mean: ``(nZ * nR,)`` mean share, exactly even.
        components: ``(k, nZ * nR)`` orthonormal rows ordered by variance.
        variances: ``(k,)`` variance along each component (originals and mirrors).
        parity: ``(k,)`` +1 for an even component, -1 for an odd one.
        total_variance: Total variance of the shares, for the explained share.
        n_frames: Original frames of the fit (the mirrors double the rows).
        shape: ``(nZ, nR)`` of the maps.
    """

    mean: np.ndarray
    components: np.ndarray
    variances: np.ndarray
    parity: np.ndarray
    total_variance: float
    n_frames: int
    shape: tuple[int, int]

    @property
    def n_components(self) -> int:
        return int(self.components.shape[0])

    @property
    def explained_ratio(self) -> np.ndarray:
        return self.variances / self.total_variance if self.total_variance > 0 else np.zeros_like(self.variances)

    def truncated(self, k: int) -> "CurrentBasis":
        """The first ``k`` components."""
        if not 1 <= k <= self.n_components:
            raise ValueError(f"{k} components requested of {self.n_components}")
        return CurrentBasis(self.mean, self.components[:k], self.variances[:k], self.parity[:k], self.total_variance,
                            self.n_frames, self.shape)

    def transform(self, shares) -> np.ndarray:
        """Coefficients ``c = B (s - s_mean)``, ``(frames, k)``."""
        s = np.asarray(shares, dtype=np.float64).reshape(-1, self.mean.size)
        return (s - self.mean) @ self.components.T

    def inverse(self, coefficients) -> np.ndarray:
        """Shares ``(frames, nZ, nR)`` from coefficients ``(frames, k)``."""
        c = np.asarray(coefficients, dtype=np.float64).reshape(-1, self.n_components)
        return (c @ self.components + self.mean).reshape(-1, *self.shape)

    def save(self, path: Path | str) -> None:
        np.savez(path, mean=self.mean, components=self.components, variances=self.variances, parity=self.parity,
                 total_variance=self.total_variance, n_frames=self.n_frames, shape=np.asarray(self.shape))

    @classmethod
    def load(cls, path: Path | str) -> "CurrentBasis":
        with np.load(path, allow_pickle=False) as f:
            return cls(f["mean"], f["components"], f["variances"], f["parity"].astype(np.int64), float(f["total_variance"]),
                       int(f["n_frames"]), tuple(int(v) for v in f["shape"]))

    @classmethod
    def fit(cls, shares: Iterable[np.ndarray], n_components: int) -> "CurrentBasis":
        """The basis of shares given as an iterable of ``(frames, nZ, nR)`` blocks; the mirrors
        are included implicitly."""
        acc = None
        for block in shares:
            block = np.asarray(block, dtype=np.float64)
            acc = acc or CurrentBasisAccumulator(block.shape[1:])
            acc.add(block)
        if acc is None:
            raise ValueError("no share to fit")
        return acc.finish(n_components)


class CurrentBasisAccumulator:
    """Mean and the even and odd covariances of shares and their mirrors, block by block.

    With the mirror of every frame included, the covariance of the stacked data is the sum
    of the covariance of the even parts and that of the odd parts; the odd parts have mean
    zero and the mean is the mean of the even parts.
    """

    def __init__(self, shape) -> None:
        self.shape = (int(shape[0]), int(shape[1]))
        n_half, mid = _half_rows(self.shape[0])
        n_even = (n_half + int(mid)) * self.shape[1]
        n_odd = n_half * self.shape[1]
        self.sum_even = np.zeros(n_even)
        self.gram_even = np.zeros((n_even, n_even))
        self.gram_odd = np.zeros((n_odd, n_odd))
        self.n = 0

    def add(self, shares) -> None:
        s = np.asarray(shares, dtype=np.float64)
        s = s[None] if s.ndim == 2 else s
        if s.shape[1:] != self.shape:
            raise ValueError(f"share shape {s.shape[1:]} differs from {self.shape}")
        e, o = _even_coords(s), _odd_coords(s)
        self.sum_even += e.sum(axis=0)
        self.gram_even += e.T @ e
        self.gram_odd += o.T @ o
        self.n += s.shape[0]

    def finish(self, n_components: int) -> CurrentBasis:
        if self.n < 2:
            raise ValueError("at least two frames are needed to fit a basis")
        mean_e = self.sum_even / self.n
        cov_e = self.gram_even / self.n - np.outer(mean_e, mean_e)
        cov_o = self.gram_odd / self.n
        found = []
        for cov, parity in ((cov_e, 1), (cov_o, -1)):
            values, vectors = np.linalg.eigh(0.5 * (cov + cov.T))
            found += [(float(values[i]), parity, vectors[:, i]) for i in range(values.size)]
        found.sort(key=lambda t: -t[0])
        k = min(int(n_components), len(found))
        comps = np.zeros((k, self.shape[0] * self.shape[1]))
        parity = np.zeros(k, dtype=np.int64)
        for i, (_, p, v) in enumerate(found[:k]):
            full = _embed(v, self.shape, p).ravel()
            # Sign convention: the largest-magnitude entry is positive (first occurrence).
            full *= 1.0 if full[np.argmax(np.abs(full))] >= 0 else -1.0
            comps[i], parity[i] = full, p
        mean = _embed(mean_e, self.shape, 1).ravel()
        return CurrentBasis(mean=mean, components=comps, variances=np.maximum([t[0] for t in found[:k]], 0.0),
                            parity=parity, total_variance=float(np.trace(cov_e) + np.trace(cov_o)), n_frames=self.n,
                            shape=self.shape)


# ------------------------------------------------------------------------------ mirror
def mirror_share(maps) -> np.ndarray:
    """Mirror of shares or any maps on a grid symmetric about Z = 0: a flip of the Z index."""
    m = np.asarray(maps)
    return m[..., ::-1, :].copy()


def mirror_coeffs(coefficients, parity) -> np.ndarray:
    """Coefficients of the mirrored share: odd components change sign."""
    return np.asarray(coefficients, dtype=np.float64) * np.asarray(parity, dtype=np.float64)


def mirror_labels(labels):
    """Labels of the mirrored frame: the axis height changes sign, ``tri_top`` and ``tri_bot``
    exchange, every other column is unchanged.

    Args:
        labels: DataFrame or mapping of columns.
    """
    out = labels.copy() if isinstance(labels, pd.DataFrame) else dict(labels)
    for column in Z_COLUMNS:
        if column in out:
            out[column] = -np.asarray(labels[column], dtype=np.float64)
    for a, b in EXCHANGED_PAIRS:
        if a in out and b in out:
            out[a], out[b] = np.asarray(labels[b]), np.asarray(labels[a])
        elif a in out or b in out:
            raise KeyError(f"mirroring needs both {a} and {b}")
    return out


# ------------------------------------------------------------------------- flux basis
def cell_flux(response: ct.PlasmaResponse, shares) -> np.ndarray:
    """Flux in Wb/rad per ampere of Ip of shares on the response's grid: each cell carries the
    current ``s`` (per ampere), the density ``s / (dR dZ)``. Border cells carry nothing."""
    d_r, d_z = gs.grid_steps(response.grid_r, response.grid_z)
    return response.flux(np.asarray(shares, dtype=np.float64) / (d_r * d_z))


@dataclass
class MachineFluxBasis:
    """``phi_mean = G P s_mean`` and ``Phi = G P B`` of one machine, in Wb/rad per ampere of Ip.

    Attributes:
        machine: Machine name.
        phi_mean: ``(nZ * nR,)`` flux of the mean share.
        phi: ``(k, nZ * nR)`` flux of each component.
        shape: ``(nZ, nR)`` of the machine's grid.
        r0_m, a0_m: Radii of the placement.
        mean_retained: Total of the placed mean share over its total before placement.
        mean_border_weight: Share of the placed mean's total on the grid's border cells,
            where the response carries no current.
    """

    machine: str
    phi_mean: np.ndarray
    phi: np.ndarray
    shape: tuple[int, int]
    r0_m: float
    a0_m: float
    mean_retained: float = 1.0
    mean_border_weight: float = 0.0

    @property
    def n_components(self) -> int:
        return int(self.phi.shape[0])

    def flux(self, ip_ka, coefficients) -> np.ndarray:
        """Physical flux ``Ip (phi_mean + Phi c)`` of predicted currents, ``(frames, nZ, nR)``
        in Wb/rad; zero where Ip is not finite."""
        ip_a = np.nan_to_num(np.atleast_1d(np.asarray(ip_ka, dtype=np.float64))) * A_PER_KA
        c = np.asarray(coefficients, dtype=np.float64).reshape(-1, self.n_components)
        if len(c) != len(ip_a):
            raise ValueError(f"{len(c)} coefficient rows for {len(ip_a)} frames")
        return (ip_a[:, None] * (self.phi_mean + c @ self.phi)).reshape(-1, *self.shape)

    def save(self, path: Path | str) -> None:
        np.savez(path, machine=np.array(self.machine), phi_mean=self.phi_mean, phi=self.phi, shape=np.asarray(self.shape),
                 r0_m=self.r0_m, a0_m=self.a0_m, mean_retained=self.mean_retained, mean_border_weight=self.mean_border_weight)

    @classmethod
    def load(cls, path: Path | str) -> "MachineFluxBasis":
        with np.load(path, allow_pickle=False) as f:
            return cls(str(f["machine"]), f["phi_mean"], f["phi"], tuple(int(v) for v in f["shape"]), float(f["r0_m"]),
                       float(f["a0_m"]), float(f["mean_retained"]), float(f["mean_border_weight"]))


def flux_basis(machine: str | MachineConfig, basis: CurrentBasis, response: ct.PlasmaResponse,
               place: Placement) -> MachineFluxBasis:
    """The flux basis of one machine from the current basis, its placement and its response.

    Raises:
        ValueError: when the placement, the response and the machine disagree.
    """
    cfg = machine if isinstance(machine, MachineConfig) else machine_config(machine)
    if place.machine != cfg.name or response.machine != cfg.name:
        raise ValueError(f"placement for {place.machine} and response for {response.machine}, machine {cfg.name}")
    shape = (response.grid_z.size, response.grid_r.size)
    mean = place.apply(basis.mean.reshape(basis.shape))
    comps = place.apply(basis.components.reshape(-1, *basis.shape))
    if mean.shape != shape:
        raise ValueError(f"placed share shape {mean.shape} differs from the response grid {shape}")
    border = np.zeros(shape, dtype=bool)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    total_before = float(basis.mean.sum())
    total_after = float(mean.sum())
    return MachineFluxBasis(machine=cfg.name, phi_mean=cell_flux(response, mean).ravel(),
                            phi=cell_flux(response, comps).reshape(len(comps), -1), shape=shape, r0_m=place.r0_m,
                            a0_m=place.a0_m, mean_retained=total_after / total_before if total_before else float("nan"),
                            mean_border_weight=float(mean[border].sum()) / total_after if total_after else float("nan"))


def compose_map(machine: str | MachineConfig, psi_coil, ip_ka, coefficients, basis: MachineFluxBasis,
                remainder=None) -> np.ndarray:
    """The stored-frame map ``k sign(Ip) (psi_coil + Ip (phi_mean + Phi c) + r)``.

    Args:
        machine: Machine with an established sign rule.
        psi_coil: ``(frames, nZ, nR)`` coil flux in the physical frame, Wb/rad.
        ip_ka: Signed plasma current per frame in kA.
        coefficients: ``(frames, k)`` current coefficients.
        basis: The machine's flux basis.
        remainder: ``(frames, nZ, nR)`` remainder in the physical frame (``psi_plasma - psi_J``),
            Wb/rad; zero when None.
    """
    psi_coil = np.asarray(psi_coil, dtype=np.float64)
    psi_coil = psi_coil[None] if psi_coil.ndim == 2 else psi_coil
    physical = psi_coil + basis.flux(ip_ka, coefficients)
    if remainder is not None:
        physical = physical + np.asarray(remainder, dtype=np.float64).reshape(physical.shape)
    return targets.to_stored(physical, np.atleast_1d(np.asarray(ip_ka, dtype=np.float64)), machine)


def machine_response(machine: str | MachineConfig, green_dir: Path | str, cache_dir: Path | str = ct.DEFAULT_CACHE_DIR,
                     write: bool = True) -> ct.PlasmaResponse:
    """The plasma response of a machine on the grid of its coil Green's table
    (``green_<machine>.npz`` in ``green_dir``), from the response cache, built and cached there
    when absent (``write`` False builds it in memory only)."""
    cfg = machine if isinstance(machine, MachineConfig) else machine_config(machine)
    table = GreenTable.load(Path(green_dir) / f"green_{cfg.file_prefix.split('_')[0]}.npz")
    return ct.PlasmaResponse.load_or_build(cfg.name, table.grid_r, table.grid_z, cache_dir, write=write)


def response_record(response: ct.PlasmaResponse) -> Mapping:
    """Identity of a response for build records: machine, grid hash and shape."""
    return {"machine": response.machine, "grid_hash": response.grid_hash,
            "shape": [int(response.grid_z.size), int(response.grid_r.size)]}
