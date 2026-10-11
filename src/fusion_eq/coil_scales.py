"""Tests of the coil-flux calculation against true DIII-D flux maps, and the scale fit.

Per frame (``analyze_frame``): the plasma current density follows from the true map
through ``delta_star`` inside the scorer's boundary; its integral against the measured
Ip gives the ratio of check K3 and the orientation of the stored map; the plasma's own
flux at the nodes outside the boundary is subtracted from the oriented true map, and
what remains is compared with the calculated coil flux (K5). Nodes within one grid
cell of a coil outline are reported separately and excluded from the fit.

Across frames (``ScaleFit``): least squares of one ``FitVariant`` (per-coil scales,
the solenoid alone, a shared shaping scale, a flat solenoid term, each with or without
a per-frame constant), accumulated from per-frame ``FrameNormal`` sums so that any
number of frames fits in memory and every variant's residual on any frame follows
from the same sums (D4, K5, K6). ``sign_rule`` turns the per-shot ratios into the one
rule, stated in terms of the direction of Ip, that orients every stored map (K4).
``mirror_difference`` is check K8 and ``reference_ratios`` the label-free MAST checks
of D5 (K9). ``reference_shots`` and ``reference_frames`` choose the eight DIII-D shots
and the three frames per shot of the single-frame tests (D3) from the frame table, so
that the review notebook and the fit script look at the same frames. Labels enter only
as test targets, never the calculation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

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


def phase_of(ip_over_peak: float, frame: int, n_frames: int, flat_top: float = FLAT_TOP_LEVEL) -> str:
    """Phase of a frame: ``flat_top`` at or above ``flat_top`` of the peak Ip, otherwise
    ``rising`` in the first half of the shot and ``late`` in the second."""
    if not np.isfinite(ip_over_peak) or ip_over_peak < flat_top:
        return "rising" if frame < n_frames / 2 else "late"
    return "flat_top"


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
FLAT_PARAMETER = "flat"
SHARED_PARAMETER = "shared"


@dataclass(frozen=True)
class FitVariant:
    """One linear model of the remainder outside the plasma.

    The model at a node is ``sum_c s_c coil_c + a I_sol [+ b_frame]`` with the coil
    maps at the frame's currents. Each column is in exactly one of ``free`` (its own
    scale), ``shared`` (one scale for all of them) or ``fixed`` (scale 1), or in none
    (scale 0, the pattern left out).

    Attributes:
        name: Variant name.
        free: Columns with their own scale.
        shared: Columns sharing one scale.
        fixed: Columns held at scale 1.
        flat: Whether the flat solenoid term ``a I_sol``, the same at every node, is
            fitted (one coefficient for all frames, in Wb/rad per kA).
        constant: Whether every frame has its own free constant, eliminated exactly
            by centring the frame's sums on its fit nodes.
    """

    name: str
    free: tuple[str, ...] = ()
    shared: tuple[str, ...] = ()
    fixed: tuple[str, ...] = ()
    flat: bool = False
    constant: bool = False

    @property
    def parameters(self) -> tuple[str, ...]:
        names = list(self.free)
        if self.shared:
            names.append(SHARED_PARAMETER)
        if self.flat:
            names.append(FLAT_PARAMETER)
        return tuple(names)

    def check(self, columns) -> None:
        groups = [set(self.free), set(self.shared), set(self.fixed)]
        for a in range(3):
            for b in range(a + 1, 3):
                if groups[a] & groups[b]:
                    raise ValueError(f"{self.name}: a column is in two groups: {groups[a] & groups[b]}")
        unknown = (groups[0] | groups[1] | groups[2]) - set(columns)
        if unknown:
            raise ValueError(f"{self.name}: unknown columns {sorted(unknown)}")


def standard_variants(columns, solenoid_column: str) -> dict[str, FitVariant]:
    """The eight variants of the coil-scale fit (D4), by name.

    ``per_coil`` is the report's headline variant. The ``*_constant`` variants add a
    free constant per frame. ``solenoid_flat`` replaces the solenoid's rectangle
    pattern by the flat term; ``solenoid_flat_plus_pattern`` fits both.
    """
    columns = tuple(columns)
    if solenoid_column not in columns:
        raise ValueError(f"{solenoid_column} is not a column")
    shaping = tuple(c for c in columns if c != solenoid_column)
    sol = (solenoid_column,)
    return {
        "per_coil": FitVariant("per_coil", free=columns),
        "per_coil_constant": FitVariant("per_coil_constant", free=columns, constant=True),
        "solenoid_only": FitVariant("solenoid_only", free=sol, fixed=shaping),
        "solenoid_only_constant": FitVariant("solenoid_only_constant", free=sol, fixed=shaping, constant=True),
        "shared_plus_solenoid": FitVariant("shared_plus_solenoid", free=sol, shared=shaping),
        "shared_plus_solenoid_constant": FitVariant("shared_plus_solenoid_constant", free=sol, shared=shaping, constant=True),
        "solenoid_flat": FitVariant("solenoid_flat", fixed=shaping, flat=True),
        "solenoid_flat_plus_pattern": FitVariant("solenoid_flat_plus_pattern", free=sol, fixed=shaping, flat=True),
    }


@dataclass
class FrameNormal:
    """Sums over the nodes of one frame of the base regressors against the target.

    The base regressors are the coil maps at the frame's currents, in column order,
    then the flat solenoid map (``I_sol`` at every node) and a map of ones. Every
    variant's normal equations, residual and per-frame constant follow from these
    sums, so the maps need not be kept.

    Attributes:
        gram: ``(n_base, n_base)`` sums of products of the regressors.
        moment: ``(n_base,)`` sums of regressor times target.
        sum_sq: Sum of the squared target.
        n: Nodes summed over.
        flux_range: Range of the frame's oriented true map (the K5 normaliser).
    """

    gram: np.ndarray
    moment: np.ndarray
    sum_sq: float
    n: int
    flux_range: float

    @classmethod
    def from_frame(cls, analysis: FrameAnalysis, sol_ka: float, mask=None) -> "FrameNormal":
        """Sums over ``mask`` (the fit mask when None) of one analysed frame."""
        m = analysis.fit_mask if mask is None else np.asarray(mask, dtype=bool)
        y = analysis.target[m]
        n = int(m.sum())
        maps = np.asarray(analysis.coil_maps, dtype=np.float64)[:, m]
        sol = float(sol_ka) if np.isfinite(sol_ka) else 0.0
        base = np.vstack([maps, np.full((1, n), sol), np.ones((1, n))])
        return cls(gram=base @ base.T, moment=base @ y, sum_sq=float(y @ y), n=n, flux_range=analysis.flux_range)

    @property
    def n_base(self) -> int:
        return int(self.gram.shape[0])

    def centred(self) -> "FrameNormal":
        """The same sums with the frame's mean removed from the target and every regressor."""
        if self.n == 0:
            return self
        g = self.gram[:, -1]
        sum_y = self.moment[-1]
        return FrameNormal(gram=self.gram - np.outer(g, g) / self.n, moment=self.moment - g * sum_y / self.n,
                           sum_sq=self.sum_sq - sum_y * sum_y / self.n, n=self.n, flux_range=self.flux_range)

    def to_arrays(self) -> dict:
        return {"gram": self.gram, "moment": self.moment, "sum_sq": self.sum_sq, "n": self.n, "flux_range": self.flux_range}


def design(variant: FitVariant, columns) -> tuple[np.ndarray, np.ndarray]:
    """``(basis, fixed)`` of a variant over the base regressors.

    The model's base coefficients are ``fixed + basis @ theta``: ``basis`` is
    ``(n_base, n_parameters)`` and ``fixed`` has 1 at the fixed columns.
    """
    columns = tuple(columns)
    variant.check(columns)
    n_base = len(columns) + 2
    basis = np.zeros((n_base, len(variant.parameters)))
    fixed = np.zeros(n_base)
    for j, c in enumerate(variant.free):
        basis[columns.index(c), j] = 1.0
    if variant.shared:
        j = variant.parameters.index(SHARED_PARAMETER)
        for c in variant.shared:
            basis[columns.index(c), j] = 1.0
    if variant.flat:
        basis[len(columns), variant.parameters.index(FLAT_PARAMETER)] = 1.0
    for c in variant.fixed:
        fixed[columns.index(c)] = 1.0
    return basis, fixed


@dataclass
class FitSolution:
    """Solved coefficients of one variant.

    Attributes:
        variant: The variant.
        columns: Current columns in order.
        theta: Coefficients in ``variant.parameters`` order; NaN for a parameter
            without signal.
    """

    variant: FitVariant
    columns: tuple[str, ...]
    theta: np.ndarray

    @property
    def base(self) -> np.ndarray:
        """Base coefficients ``fixed + basis @ theta`` (NaN parameters count as 0)."""
        basis, fixed = design(self.variant, self.columns)
        return fixed + basis @ np.nan_to_num(self.theta)

    @property
    def scales(self) -> np.ndarray:
        """Per-column scales, ``(n_columns,)``: fitted, shared, 1 for fixed, 0 for left out;
        NaN where a free column had no signal."""
        out = self.base[:len(self.columns)]
        for j, c in enumerate(self.variant.free):
            if np.isnan(self.theta[j]):
                out[self.columns.index(c)] = np.nan
        return out

    @property
    def flat_wb_per_ka(self) -> float:
        """The flat solenoid coefficient, 0 when the variant has none."""
        return float(self.base[len(self.columns)]) if self.variant.flat else 0.0

    def coefficients(self) -> dict[str, float]:
        return {name: float(v) for name, v in zip(self.variant.parameters, self.theta)}

    def model(self, coil_maps, sol_ka: float) -> np.ndarray:
        """The model map from a frame's coil maps and solenoid current, before any constant."""
        maps = np.asarray(coil_maps, dtype=np.float64)
        out = np.tensordot(np.nan_to_num(self.scales), maps, axes=1)
        if self.variant.flat:
            out = out + self.flat_wb_per_ka * (float(sol_ka) if np.isfinite(sol_ka) else 0.0)
        return out

    def residual_sums(self, normal: FrameNormal) -> tuple[float, float, int]:
        """``(sum of squared residuals, sum of residuals, n)`` of a frame under this variant;
        with a per-frame constant the sums are those of the centred residual."""
        nm = normal.centred() if self.variant.constant else normal
        b = self.base
        ss = nm.sum_sq - 2.0 * b @ nm.moment + b @ nm.gram @ b
        total = (normal.moment[-1] - b @ normal.gram[:, -1])
        return float(max(ss, 0.0)), float(total), nm.n

    def frame_rms(self, normal: FrameNormal) -> float:
        """Root-mean-square residual of a frame over its flux range (K5); NaN without nodes."""
        ss, _, n = self.residual_sums(normal)
        if n == 0 or not np.isfinite(normal.flux_range) or normal.flux_range <= 0:
            return np.nan
        return float(np.sqrt(ss / n) / normal.flux_range)

    def frame_mean_residual(self, normal: FrameNormal) -> float:
        """Mean of ``target - model`` over the frame's nodes, in Wb/rad, before any constant.
        For a variant with a constant this is the frame's constant."""
        _, total, n = self.residual_sums(normal)
        return float(total / n) if n else np.nan

    def to_coil_scales(self, machine: str, source: str = "") -> "CoilScales":
        """The record ``GreenTable.flux`` applies: per-column scales and the flat term."""
        from fusion_eq.coil_flux import CoilScales
        return CoilScales(machine=machine, scales={c: float(np.nan_to_num(s)) for c, s in zip(self.columns, self.scales)},
                          flat_wb_per_ka=self.flat_wb_per_ka, source=source or self.variant.name)


class ScaleFit:
    """Least squares of one variant, accumulated frame by frame from ``FrameNormal`` sums.

    Minimises ``sum_frames sum_nodes (target - model)^2`` with the model of
    ``FitVariant``. A frame with fewer nodes than parameters plus one is skipped.
    """

    def __init__(self, columns, variant: FitVariant) -> None:
        self.columns = tuple(columns)
        self.variant = variant
        self.basis, self.fixed = design(variant, self.columns)
        p = self.basis.shape[1]
        self.ata = np.zeros((p, p))
        self.atb = np.zeros(p)
        self.n_nodes = 0
        self.n_frames = 0

    def add(self, normal: FrameNormal) -> None:
        if normal.n < self.basis.shape[1] + 1:
            return
        nm = normal.centred() if self.variant.constant else normal
        self.ata += self.basis.T @ nm.gram @ self.basis
        self.atb += self.basis.T @ (nm.moment - nm.gram @ self.fixed)
        self.n_nodes += normal.n
        self.n_frames += 1

    def solve(self) -> FitSolution:
        """Coefficients by the pseudo-inverse; NaN for a parameter with no signal."""
        p = self.basis.shape[1]
        theta = np.full(p, np.nan)
        if self.n_frames > 0:
            active = np.diag(self.ata) > 0
            if active.any():
                theta[active] = np.linalg.pinv(self.ata[np.ix_(active, active)]) @ self.atb[active]
        return FitSolution(variant=self.variant, columns=self.columns, theta=theta)


def fit_variants(normals, columns, variants: dict[str, FitVariant]) -> dict[str, FitSolution]:
    """Solves every variant over the same frames (an iterable of ``FrameNormal``)."""
    fits = {name: ScaleFit(columns, v) for name, v in variants.items()}
    for normal in normals:
        for fit in fits.values():
            fit.add(normal)
    return {name: fit.solve() for name, fit in fits.items()}


# ------------------------------------------------------------------ X-point (D9)
def xpoint_class(contour, oriented_psi, grid_r, grid_z, ratio: float = 2.0) -> str:
    """Position of the X-point of a frame, as a proxy: ``lower``, ``upper``, ``both`` or ``none``.

    The scorer has no X-point finder. The X-point is a null of the poloidal field, so
    the proxy takes the boundary point with the smallest ``|grad psi| / R``; the
    class is ``both`` when the half of the boundary on the other side of Z = 0 holds a
    minimum within ``ratio`` of the smallest, ``none`` without a boundary.
    """
    if contour is None or len(contour) < 3:
        return "none"
    c = np.asarray(contour, dtype=np.float64)
    grid_r = np.asarray(grid_r, dtype=np.float64)
    grid_z = np.asarray(grid_z, dtype=np.float64)
    dpsi_dz, dpsi_dr = np.gradient(np.asarray(oriented_psi, dtype=np.float64), grid_z, grid_r)
    pts = c[:, ::-1]
    br = RegularGridInterpolator((grid_z, grid_r), dpsi_dz, bounds_error=False, fill_value=np.nan)(pts)
    bz = RegularGridInterpolator((grid_z, grid_r), dpsi_dr, bounds_error=False, fill_value=np.nan)(pts)
    bp = np.hypot(br, bz) / c[:, 0]
    if not np.isfinite(bp).any():
        return "none"
    i0 = int(np.nanargmin(bp))
    z0 = c[i0, 1]
    other = c[:, 1] * np.sign(z0) < 0
    if other.any() and np.isfinite(bp[other]).any():
        second = np.nanmin(bp[other])
        # A second null within the ratio, or an exact null on both sides.
        if second < ratio * bp[i0] or second <= 1e-9 * np.nanmax(bp):
            return "both"
    return "lower" if z0 < 0 else "upper"


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
                     start_frame: int, flat_frame: int, scales=None) -> dict:
    """Label-free coil-flux ratios at the machine's reference point (D5).

    Args:
        table: Green's table.
        currents_ka: ``(frames, n_columns)`` currents in kA.
        ip_ka: ``(frames,)`` plasma current in kA, signed: the ratios then carry the
            same sign on both directions of the discharge.
        cfg: Machine configuration (reference radius).
        grid_r, grid_z: Grid axes.
        start_frame: Frame at the start of the Ip rise.
        flat_frame: Frame on the flat-top.
        scales: Coil model applied to the flux (``GreenTable.flux``); None for the
            unscaled patterns.

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
    flux_flat = table.flux(c[flat_frame], scales)
    bz = gs.vertical_field(flux_flat, grid_r, grid_z)[iz, ir]
    bz_ratio = bz / (MU0 * ip_flat_a / (4.0 * np.pi * r0)) if ip_flat_a != 0 else np.nan
    sol = table.column_index([cfg.solenoid_column])[0]
    sol_only = np.zeros_like(c[flat_frame])
    sol_only[sol] = c[flat_frame, sol] - c[start_frame, sol]
    delta_sol = table.flux(sol_only, scales)[iz, ir]
    sol_ratio = delta_sol / (MU0 * r0 * ip_flat_a) if ip_flat_a != 0 else np.nan
    return {"bz_ratio": float(bz_ratio), "sol_flux_ratio": float(sol_ratio), "ir": ir, "iz": iz}
