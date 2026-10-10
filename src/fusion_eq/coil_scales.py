"""Tests of the coil-flux calculation against true DIII-D flux maps, and the scale fit.

Per frame (``analyze_frame``): the plasma current density follows from the true map
through ``delta_star`` inside the scorer's boundary; its integral against the measured
Ip gives the ratio of check K3 and the orientation of the stored map; the plasma's own
flux at the nodes outside the boundary is subtracted from the oriented true map, and
what remains is compared with the calculated coil flux (K5). Nodes within one grid
cell of a coil outline are reported separately and excluded from the fit.

Across frames (``ScaleFit``): least squares for one scale per current column, with or
without a per-frame constant, accumulated as normal equations so that any number of
frames fits in memory (D4, K6). ``sign_rule`` turns the per-shot ratios into the one
rule, stated in terms of the direction of Ip, that orients every stored map (K4).
``mirror_difference`` is check K8 and ``reference_ratios`` the label-free MAST checks
of D5 (K9). ``reference_shots`` and ``reference_frames`` choose the eight DIII-D shots
and the three frames per shot of the single-frame tests (D3) from the frame table, so
that the review notebook and the fit script look at the same frames. Labels enter only
as test targets, never the calculation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from fusion_eq import grad_shafranov as gs
from fusion_eq.coil_flux import MU0, GreenTable
from fusion_eq.machines import MachineConfig

A_PER_KA = 1000.0
# Recording rates of the DIII-D magnetics (frame table ``mag_rate_hz``).
FAST_RATE_HZ = 20_000
SLOW_RATE_HZ = 2_000
# A frame is on the flat-top above this fraction of the shot's peak Ip; the rise is taken
# to have started, and the ramp-down not to have finished, above ``RISE_LEVEL``.
FLAT_TOP_LEVEL = 0.9
RISE_LEVEL = 0.5
REFERENCE_PHASES = ("rising", "flat_top", "late")


# ----------------------------------------------------------- reference shots and frames
def reference_shots(index: pd.DataFrame, n_fast_positive: int = 3, n_negative: int = 3, n_slow: int = 2,
                    min_frames: int = 30) -> dict[str, list[str]]:
    """The DIII-D reference shots of analysis D3 from the frame table's shot index.

    Three kinds, each the first shots in sorted id order among the training shots that
    built without error and have at least ``min_frames`` frames: positive Ip at the
    fast recording rate, negative Ip at any rate, and positive Ip at the slow rate. A
    shot is listed under one kind only.

    Args:
        index: ``shot_index.parquet`` of a frame-table build (columns ``split``,
            ``shot``, ``n_frames``, ``mag_rate_hz``, ``ip_direction``, ``error``).

    Returns:
        ``{"positive_20khz": [...], "negative_ip": [...], "rate_2khz": [...]}``.
    """
    train = index[(index["split"] == "diii_d_train") & index["error"].isna() & (index["n_frames"] >= min_frames)]
    rate = train["mag_rate_hz"].astype("Int64")
    direction = train["ip_direction"].astype("Int64")
    kinds = {
        "positive_20khz": (train[(rate == FAST_RATE_HZ) & (direction == 1)], n_fast_positive),
        "negative_ip": (train[direction == -1], n_negative),
        "rate_2khz": (train[(rate == SLOW_RATE_HZ) & (direction == 1)], n_slow),
    }
    out: dict[str, list[str]] = {}
    taken: set[str] = set()
    for kind, (sub, n) in kinds.items():
        ids = [s for s in sorted(sub["shot"].astype(str)) if s not in taken][:n]
        taken.update(ids)
        out[kind] = ids
    return out


def reference_frames(ip_over_peak, flat_top: float = FLAT_TOP_LEVEL, rise: float = RISE_LEVEL) -> dict[str, int]:
    """Three frames of one shot for the single-frame tests: rising, mid flat-top, late.

    ``rising`` is the first frame at or above ``rise`` of the peak, ``flat_top`` the
    middle of the frames at or above ``flat_top``, and ``late`` the last frame at or
    above ``rise``, so that every frame still holds a plasma the scorer can bound. When
    the trace never reaches a level, the frame of its maximum stands in; when two
    choices coincide, the three frames fall back to the first, middle and last frame.

    Args:
        ip_over_peak: ``(frames,)`` absolute Ip over the shot's peak, in frame order
            (frame table column ``ip_over_peak``). NaN entries never qualify.

    Returns:
        Frame indices keyed by ``REFERENCE_PHASES``.

    Raises:
        ValueError: with fewer than three frames.
    """
    x = np.asarray(ip_over_peak, dtype=np.float64).ravel()
    n = x.size
    if n < 3:
        raise ValueError("a reference shot needs at least three frames")
    finite = np.where(np.isfinite(x), x, -np.inf)
    peak = int(np.argmax(finite))
    above_rise = np.flatnonzero(finite >= rise)
    flat = np.flatnonzero(finite >= flat_top)
    rising = int(above_rise[0]) if above_rise.size else peak
    late = int(above_rise[-1]) if above_rise.size else peak
    mid = int(flat[flat.size // 2]) if flat.size else peak
    frames = (rising, mid, late)
    if len(set(frames)) < 3:
        frames = (0, n // 2, n - 1)
    return dict(zip(REFERENCE_PHASES, frames))


# -------------------------------------------------------------------------- per frame
@dataclass
class FrameAnalysis:
    """Results of ``analyze_frame`` for one frame.

    Attributes:
        orientation: Sign ``s`` applied to the stored map so that the current density
            inside the boundary integrates to a current with the sign of the measured
            Ip: the physical orientation, in which the coil flux from the measured
            coil currents is directly comparable.
        boundary_found: Whether the scorer returned a closed boundary.
        contour: The scorer's boundary as an ``(n, 2)`` array of (R, Z) in metres, or
            None when no boundary was found.
        inside: Bool map of nodes inside the boundary.
        fit_mask: Nodes outside the boundary and not near a coil, with finite target.
        near_coil_mask: Nodes outside the boundary and within one cell of a coil outline.
        current_a: Integral of the current density inside the boundary, in amperes, from
            the stored (unoriented) map; its sign is the orientation evidence.
        ratio: ``current_a / (Ip in amperes)``; NaN without Ip.
        oriented_psi: ``s * psi_stored``, the true map in the physical orientation.
        target: ``s * psi_stored - plasma_flux`` outside the boundary, NaN inside.
        coil_maps: ``(n_columns, nZ, nR)`` unscaled flux of each column at its current.
        flux_range: Range of the oriented true map, the normaliser of K5.
    """

    orientation: float
    boundary_found: bool
    contour: np.ndarray | None
    inside: np.ndarray
    fit_mask: np.ndarray
    near_coil_mask: np.ndarray
    current_a: float
    ratio: float
    oriented_psi: np.ndarray
    target: np.ndarray
    coil_maps: np.ndarray
    flux_range: float

    @property
    def coil_total(self) -> np.ndarray:
        return self.coil_maps.sum(axis=0)

    def scaled_coil_flux(self, scales=None) -> np.ndarray:
        if scales is None:
            return self.coil_total
        return np.tensordot(np.asarray(scales, dtype=np.float64), self.coil_maps, axes=1)


def analyze_frame(psi_stored, grid_r, grid_z, cfg: MachineConfig, envelope, table: GreenTable,
                  currents_ka, ip_ka: float, near_coil, scorer, orientation: float | None = None) -> FrameAnalysis:
    """Runs the per-frame decomposition of one true flux map.

    Args:
        psi_stored: ``(nZ, nR)`` true map as stored.
        grid_r, grid_z: Grid axes in metres.
        cfg: Machine configuration.
        envelope: Scorer envelope mask namespace (``starter_kit.envelope_mask``).
        table: Green's table of the machine.
        currents_ka: Currents in kA in ``table.columns`` order.
        ip_ka: Measured plasma current in kA at the frame.
        near_coil: Bool map of nodes near a coil outline (``coil_flux.near_coil_mask``).
        scorer: ``starter_kit.scoring()`` namespace.
        orientation: Sign to apply to the stored map; when None, the sign of the
            integrated current density times the sign of Ip is used, which puts the
            map in the physical orientation whatever the direction of the discharge.
            Without a finite nonzero Ip the sign of the current alone is used.
    """
    psi = np.asarray(psi_stored, dtype=np.float64)
    grid_r = np.asarray(grid_r, dtype=np.float64)
    grid_z = np.asarray(grid_z, dtype=np.float64)
    contour = scorer.lcfs.extract_lcfs(psi, grid_r, grid_z, cfg.name, envelope.mask_coarse, envelope.mask_f)
    inside = gs.boundary_mask(contour, grid_r, grid_z)
    j_stored = gs.current_density(psi, grid_r, grid_z)
    current_a = gs.total_current(j_stored, grid_r, grid_z, inside) if inside.any() else np.nan
    ip_a = float(ip_ka) * A_PER_KA if np.isfinite(ip_ka) else np.nan
    ratio = current_a / ip_a if np.isfinite(current_a) and np.isfinite(ip_a) and ip_a != 0 else np.nan
    if orientation is None:
        if np.isfinite(ratio) and ratio != 0:
            orientation = float(np.sign(ratio))
        else:
            orientation = float(np.sign(current_a)) if np.isfinite(current_a) and current_a != 0 else 1.0
    oriented = orientation * psi
    plasma = gs.flux_from_current(orientation * j_stored, grid_r, grid_z, inside) if inside.any() \
        else np.zeros_like(psi)
    target = oriented - plasma
    outside = ~inside
    fit_mask = outside & ~np.asarray(near_coil, dtype=bool) & np.isfinite(target)
    near_mask = outside & np.asarray(near_coil, dtype=bool) & np.isfinite(target)
    c = np.nan_to_num(np.asarray(currents_ka, dtype=np.float64))
    coil_maps = table.patterns * c[:, None, None]
    return FrameAnalysis(orientation=orientation, boundary_found=contour is not None,
                         contour=None if contour is None else np.asarray(contour, dtype=np.float64), inside=inside,
                         fit_mask=fit_mask, near_coil_mask=near_mask, current_a=float(current_a),
                         ratio=float(ratio), oriented_psi=oriented, target=target, coil_maps=coil_maps,
                         flux_range=float(np.nanmax(oriented) - np.nanmin(oriented)))


def rms_fraction(target, model, mask, flux_range: float, constant: float = 0.0) -> float:
    """Root-mean-square of ``target - model - constant`` over ``mask``, divided by ``flux_range``."""
    m = np.asarray(mask, dtype=bool)
    if not m.any() or not np.isfinite(flux_range) or flux_range <= 0:
        return np.nan
    d = np.asarray(target, dtype=np.float64)[m] - np.asarray(model, dtype=np.float64)[m] - constant
    return float(np.sqrt(np.mean(d * d)) / flux_range)


# -------------------------------------------------------------------------- sign rule
def sign_rule(ratios, ip_directions) -> dict:
    """The orientation rule from per-shot current ratios and Ip directions (K4).

    The rule is ``psi_physical = c * sign(Ip) * psi_stored`` with one constant ``c``.
    For each shot ``c_shot = sign(ratio) * sign(Ip)``, which is the sign of the
    current the stored map carries; the rule holds when every finite ``c_shot``
    agrees. The per-frame orientation of ``analyze_frame``, ``sign(ratio)``, then
    equals ``orientation_from_rule(c, Ip)`` on every frame that obeys the rule.

    Returns:
        ``c`` (+1, -1, or 0 when inconsistent or empty), ``consistent``,
        ``n_shots``, ``n_positive_ip``, ``n_negative_ip``, ``ratio_sign_by_direction``.
    """
    r = np.asarray(ratios, dtype=np.float64)
    d = np.sign(np.asarray(ip_directions, dtype=np.float64))
    ok = np.isfinite(r) & (r != 0) & (d != 0)
    c_shot = np.sign(r[ok]) * d[ok]
    consistent = c_shot.size > 0 and bool(np.all(c_shot == c_shot[0]))
    by_direction = {int(k): sorted(set(int(x) for x in np.sign(r[ok][d[ok] == k]))) for k in (1, -1)}
    return {"c": int(c_shot[0]) if consistent else 0, "consistent": consistent, "n_shots": int(ok.sum()),
            "n_positive_ip": int((d[ok] > 0).sum()), "n_negative_ip": int((d[ok] < 0).sum()),
            "ratio_sign_by_direction": by_direction}


def orientation_from_rule(c: float, ip_ka: float) -> float:
    """Sign that orients a stored map under the rule ``psi_physical = c sign(Ip) psi_stored``."""
    return float(c) * (1.0 if ip_ka >= 0 else -1.0)


# -------------------------------------------------------------------------- scale fit
@dataclass
class ScaleFit:
    """Least-squares scales, one per current column, accumulated frame by frame.

    Minimises ``sum_frames sum_mask (target - sum_c s_c coil_c [- b_frame])^2``. With
    ``with_constant`` the per-frame constants are eliminated exactly by subtracting
    the mask means of target and of every coil map before accumulation.
    """

    columns: tuple[str, ...]
    with_constant: bool = False
    ata: np.ndarray = field(init=False)
    atb: np.ndarray = field(init=False)
    n_nodes: int = field(init=False, default=0)
    n_frames: int = field(init=False, default=0)
    sum_sq_target: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        n = len(self.columns)
        self.ata = np.zeros((n, n))
        self.atb = np.zeros(n)

    def add(self, target, coil_maps, mask) -> None:
        """Adds one frame. ``coil_maps`` is ``(n_columns, nZ, nR)`` unscaled."""
        m = np.asarray(mask, dtype=bool)
        if m.sum() < len(self.columns) + 1:
            return
        y = np.asarray(target, dtype=np.float64)[m]
        a = np.asarray(coil_maps, dtype=np.float64)[:, m]
        if self.with_constant:
            y = y - y.mean()
            a = a - a.mean(axis=1, keepdims=True)
        self.ata += a @ a.T
        self.atb += a @ y
        self.n_nodes += int(m.sum())
        self.n_frames += 1
        self.sum_sq_target += float(y @ y)

    def solve(self) -> np.ndarray:
        """Scales in ``columns`` order; NaN for columns with no signal. Uses the pseudo-inverse."""
        if self.n_frames == 0:
            return np.full(len(self.columns), np.nan)
        diag = np.diag(self.ata)
        active = diag > 0
        scales = np.full(len(self.columns), np.nan)
        if active.any():
            sub = self.ata[np.ix_(active, active)]
            scales[active] = np.linalg.pinv(sub) @ self.atb[active]
        return scales

    def constant_for(self, target, coil_maps, mask, scales) -> float:
        """Per-frame constant implied by the solved scales; 0 without constants."""
        if not self.with_constant:
            return 0.0
        m = np.asarray(mask, dtype=bool)
        if not m.any():
            return np.nan
        model = np.tensordot(np.nan_to_num(np.asarray(scales, dtype=np.float64)),
                             np.asarray(coil_maps, dtype=np.float64), axes=1)
        return float(np.mean(np.asarray(target, dtype=np.float64)[m] - model[m]))


# -------------------------------------------------------------------- remainder (K7)
def remainder_metrics(oriented_psi, coil_flux_scaled, grid_r, grid_z, cfg: MachineConfig, envelope, scorer,
                      r_axis: float, z_axis: float, ip_ka: float) -> dict:
    """Where the extremum of ``oriented_psi - coil_flux`` sits and how large it is.

    In the physical orientation the plasma's flux hill has the sign of Ip, so the
    remainder is multiplied by ``sign(Ip)`` to make the hill a maximum before its
    O-point is found with the scorer's finder inside the envelope.

    Returns:
        ``rem_axis_dist_m`` (O-point to stored axis), ``rem_amplitude`` (remainder at
        the O-point minus its mean on the envelope rim, positive for a hill),
        ``rem_amplitude_over_ip`` (per kA of ``abs(Ip)``).
    """
    direction = -1.0 if np.isfinite(ip_ka) and ip_ka < 0 else 1.0
    phi = direction * (np.asarray(oriented_psi, dtype=np.float64) - np.asarray(coil_flux_scaled, dtype=np.float64))
    r_o, z_o, iz, ir, _ = scorer.o_point.find_o_point(phi, grid_r, grid_z, envelope.mask_coarse)
    rim = envelope.mask_coarse & ~scorer.o_point._erode(envelope.mask_coarse, 1)
    base = float(np.mean(phi[rim])) if rim.any() else 0.0
    amplitude = float(phi[iz, ir] - base)
    return {"rem_axis_dist_m": float(np.hypot(r_o - r_axis, z_o - z_axis)), "rem_amplitude": amplitude,
            "rem_amplitude_over_ip": amplitude / abs(ip_ka) if np.isfinite(ip_ka) and ip_ka != 0 else np.nan}


# ----------------------------------------------------------------------- mirror (K8)
def mirror_pairs(columns, cfg: MachineConfig) -> list[tuple[int, int]]:
    """Index pairs of up-down mirror coils (DIII-D ``A``/``B``, MAST ``u``/``l``)."""
    lookup = {c: i for i, c in enumerate(columns)}
    pairs = []
    for c in columns:
        if cfg.name == "DIII-D" and c.endswith("A"):
            partner = c[:-1] + "B"
        elif cfg.name == "MAST" and c.endswith("u_current"):
            partner = c.replace("u_current", "l_current")
        else:
            continue
        if partner in lookup:
            pairs.append((lookup[c], lookup[partner]))
    return pairs


def mirror_difference(table: GreenTable, currents_ka, cfg: MachineConfig) -> float:
    """K8: largest difference between the flipped coil flux and the swapped-current coil flux,
    as a fraction of the flux range, for one frame of currents."""
    c = np.nan_to_num(np.asarray(currents_ka, dtype=np.float64)).copy()
    swapped = c.copy()
    for i, j in mirror_pairs(table.columns, cfg):
        swapped[i], swapped[j] = c[j], c[i]
    original = table.flux(c)
    flipped = original[::-1, :]
    other = table.flux(swapped)
    rng = float(original.max() - original.min())
    return float(np.max(np.abs(flipped - other)) / rng) if rng > 0 else np.nan


# ------------------------------------------------------------------ MAST checks (K9)
def reference_ratios(table: GreenTable, currents_ka, ip_ka, cfg: MachineConfig, grid_r, grid_z,
                     start_frame: int, flat_frame: int) -> dict:
    """Label-free coil-flux ratios at the machine's reference point (D5).

    Args:
        table: Green's table.
        currents_ka: ``(frames, n_columns)`` currents in kA.
        ip_ka: ``(frames,)`` plasma current in kA.
        cfg: Machine configuration (reference radius).
        grid_r, grid_z: Grid axes.
        start_frame: Frame at the start of the Ip rise.
        flat_frame: Frame on the flat-top.

    Returns:
        ``bz_ratio``: vertical coil field at (R0, 0) over ``mu0 Ip / (4 pi R0)`` at the
        flat frame; ``sol_flux_ratio``: change of solenoid flux at (R0, 0) between the
        two frames over ``mu0 R0 Ip`` at the flat frame.
    """
    grid_r = np.asarray(grid_r, dtype=np.float64)
    grid_z = np.asarray(grid_z, dtype=np.float64)
    r0 = cfg.reference_radius_m
    ir = int(np.argmin(np.abs(grid_r - r0)))
    iz = int(np.argmin(np.abs(grid_z)))
    c = np.nan_to_num(np.asarray(currents_ka, dtype=np.float64))
    ip_flat_a = float(ip_ka[flat_frame]) * A_PER_KA
    flux_flat = table.flux(c[flat_frame])
    bz = gs.vertical_field(flux_flat, grid_r, grid_z)[iz, ir]
    bz_ratio = bz / (MU0 * ip_flat_a / (4.0 * np.pi * r0)) if ip_flat_a != 0 else np.nan
    sol = table.column_index([cfg.solenoid_column])[0]
    sol_only = np.zeros_like(c[flat_frame])
    sol_only[sol] = c[flat_frame, sol] - c[start_frame, sol]
    delta_sol = table.flux(sol_only)[iz, ir]
    sol_ratio = delta_sol / (MU0 * r0 * ip_flat_a) if ip_flat_a != 0 else np.nan
    return {"bz_ratio": float(bz_ratio), "sol_flux_ratio": float(sol_ratio), "ir": ir, "iz": iz}
