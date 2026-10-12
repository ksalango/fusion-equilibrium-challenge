"""Physics formulas for q95 and betaN, boundary shapes from flux maps and the learned
correction on top of the formulas.

The scalar route is a physics formula times a learned correction. The formula carries
the units and the machine size, so that it can transfer to MAST; the correction is a
function of quantities without units only (elongation, triangularities, inverse aspect
ratio, the Thomson peak-to-mean ratio, Ip over its peak).

Shapes come from the scorer's own boundary extraction and shape functions, imported
through ``fusion_eq.starter_kit``, so that kappa and the triangularities here are the
quantities the scorer derives (with its plausibility clamping to NaN). ``R_geo`` and
``a`` follow the scorer's definitions: half the sum and half the difference of the
extreme major radii of the boundary.

Formulas (``Ip`` in MA, ``Bt`` in T, lengths in m, pressure in Pa; the sizes of Ip and
Bt are used, q95 and betaN being stored positive):

    Bt(R)  = 2e-7 n_turns |I_tf| / R                       I_tf in A
    q95    = 5 a^2 Bt / (R Ip) (1 + kappa^2 (1 + 2 delta^2 - 1.2 delta^3)) / 2
             (1.17 - 0.65 eps) / (1 - eps^2)^2             delta = (tri_top + tri_bot) / 2
    betaN  = 100 (2 mu0 p / Bt^2) a Bt / Ip

The toroidal-field turn count is a per-machine parameter. The DIII-D value of 144 is
provisional, to verify against a published source; section 2 of notebook 05 fits the
multiplier the labels imply and the turn count it corresponds to.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from fusion_eq import baseline as bl, starter_kit
from fusion_eq.machines import MachineConfig, machine_config

MU0 = 4e-7 * np.pi
A_PER_KA = 1000.0
MA_PER_KA = 1e-3
PA_PER_KPA = 1000.0
# Toroidal field coil turns per machine. DIII-D: 144, provisional, to verify against a
# published source. MAST: unknown, left out on purpose until a source is found.
TF_TURNS = {"DIII-D": 144}
SHAPE_COLUMNS = ("R_geo", "a", "kappa", "tri_top", "tri_bot", "eps")
# Inputs of the correction model: quantities without units only.
CORRECTION_FEATURES = ("kappa", "tri_top", "tri_bot", "eps", "peak_to_mean", "ip_over_peak")
DEFAULT_PRESSURE_COLUMN = "ts_core_pe_mean__interp"


# ----------------------------------------------------------------------------- formulas
def toroidal_field(tf_ka, r_m, n_turns: float) -> np.ndarray:
    """Vacuum toroidal field in tesla, ``2e-7 n_turns |I| / R`` with ``I`` in amperes.

    Args:
        tf_ka: Toroidal field coil current in kA (sign ignored).
        r_m: Major radius of evaluation in metres.
        n_turns: Turns of the toroidal field coil set.
    """
    current_a = np.abs(np.asarray(tf_ka, dtype=np.float64)) * A_PER_KA
    return 2e-7 * float(n_turns) * current_a / np.asarray(r_m, dtype=np.float64)


def q95_formula(a_m, r_geo_m, kappa, delta, bt_t, ip_ma) -> np.ndarray:
    """q95 from the boundary shape, the toroidal field and the plasma current.

    Args:
        a_m: Minor radius in metres.
        r_geo_m: Geometric major radius in metres.
        kappa: Elongation.
        delta: Triangularity, the mean of top and bottom.
        bt_t: Toroidal field in tesla (sign ignored).
        ip_ma: Plasma current in MA (sign ignored).
    """
    a = np.asarray(a_m, dtype=np.float64)
    r = np.asarray(r_geo_m, dtype=np.float64)
    k = np.asarray(kappa, dtype=np.float64)
    d = np.asarray(delta, dtype=np.float64)
    bt = np.abs(np.asarray(bt_t, dtype=np.float64))
    ip = np.abs(np.asarray(ip_ma, dtype=np.float64))
    eps = a / r
    shape = (1.0 + k ** 2 * (1.0 + 2.0 * d ** 2 - 1.2 * d ** 3)) / 2.0
    aspect = (1.17 - 0.65 * eps) / (1.0 - eps ** 2) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        return 5.0 * a ** 2 * bt / (r * ip) * shape * aspect


def beta_n_formula(p_pa, a_m, bt_t, ip_ma) -> np.ndarray:
    """betaN in percent metre tesla per MA: ``100 (2 mu0 p / Bt^2) a Bt / Ip``.

    Args:
        p_pa: Plasma pressure in Pa (the core electron pressure summary when called from
            the frame table, so a constant factor to the total pressure is expected).
        a_m: Minor radius in metres.
        bt_t: Toroidal field in tesla (sign ignored).
        ip_ma: Plasma current in MA (sign ignored).
    """
    p = np.asarray(p_pa, dtype=np.float64)
    a = np.asarray(a_m, dtype=np.float64)
    bt = np.abs(np.asarray(bt_t, dtype=np.float64))
    ip = np.abs(np.asarray(ip_ma, dtype=np.float64))
    with np.errstate(divide="ignore", invalid="ignore"):
        return 100.0 * (2.0 * MU0 * p / bt ** 2) * a * bt / ip


# ------------------------------------------------------------------------------- shapes
def shape_from_map(psi, grid_r, grid_z, machine: str | MachineConfig = "DIII-D", envelope=None, scorer=None) -> dict:
    """Boundary shape of one stored flux map with the scorer's functions.

    Returns ``R_geo`` and ``a`` (metres), ``kappa``, ``tri_top``, ``tri_bot`` (the scorer's
    derived values, NaN when its plausibility check fails), ``eps = a / R_geo`` and
    ``boundary_found``. Every value is NaN when no closed boundary is found.
    """
    cfg = machine if isinstance(machine, MachineConfig) else machine_config(machine)
    scorer = scorer or starter_kit.scoring()
    envelope = envelope or starter_kit.envelope_mask(cfg)
    r = np.asarray(grid_r, dtype=np.float64)
    z = np.asarray(grid_z, dtype=np.float64)
    out = {k: np.nan for k in SHAPE_COLUMNS}
    contour = scorer.lcfs.extract_lcfs(np.asarray(psi, dtype=np.float64), r, z, cfg.name, envelope.mask_coarse, envelope.mask_f)
    out["boundary_found"] = contour is not None and len(contour) >= 8
    if not out["boundary_found"]:
        return out
    derived = scorer.derive.derive_frame(psi, r, z, cfg.name, envelope.mask_coarse, envelope.mask_f, contour=contour, with_li=False)
    r_max, r_min = float(contour[:, 0].max()), float(contour[:, 0].min())
    out["R_geo"] = scorer.lcfs.major_radius(contour)
    out["a"] = 0.5 * (r_max - r_min)
    for k in ("kappa", "tri_top", "tri_bot"):
        out[k] = float(derived[k])
    out["eps"] = out["a"] / out["R_geo"] if out["R_geo"] > 0 else np.nan
    return out


def shape_frames(shot_id: str, frame_index, maps, grid_r, grid_z, machine: str = "DIII-D") -> pd.DataFrame:
    """Shapes of one shot's maps as rows ``shot, frame, R_geo, a, kappa, tri_top, tri_bot,
    eps, boundary_found``. Runs in a worker process."""
    cfg = machine_config(machine)
    scorer = starter_kit.scoring()
    envelope = starter_kit.envelope_mask(cfg)
    rows = []
    for frame, psi in zip(np.asarray(frame_index), np.asarray(maps)):
        rows.append({"shot": shot_id, "frame": int(frame), **shape_from_map(psi, grid_r, grid_z, cfg, envelope, scorer)})
    return pd.DataFrame(rows, columns=["shot", "frame", *SHAPE_COLUMNS, "boundary_found"])


def shape_table(items: Iterable[tuple[str, np.ndarray, np.ndarray]], grid_r, grid_z, machine: str = "DIII-D",
                workers: int | None = None) -> pd.DataFrame:
    """Shapes of many shots' maps, in the order given.

    Args:
        items: ``(shot_id, frame_index, maps)`` per shot, consumed lazily.
        grid_r, grid_z: Grid axes in metres.
        machine: Machine name.
        workers: Processes; the CPU count minus two when None, one for a single process.
    """
    r = np.asarray(grid_r, dtype=np.float64)
    z = np.asarray(grid_z, dtype=np.float64)
    parts = list(bl._map_ordered(shape_frames, ((s, f, m, r, z, machine) for s, f, m in items), workers))
    if not parts:
        return pd.DataFrame(columns=["shot", "frame", *SHAPE_COLUMNS, "boundary_found"])
    return pd.concat(parts, ignore_index=True)


def cached_shape_table(path: Path | str, items_fn: Callable[[], Iterable], grid_r, grid_z, machine: str = "DIII-D",
                       workers: int | None = None) -> pd.DataFrame:
    """``shape_table`` read from ``path`` when it exists, computed and written there otherwise."""
    path = Path(path)
    if path.is_file():
        return pd.read_parquet(path)
    table = shape_table(items_fn(), grid_r, grid_z, machine, workers)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.parquet")
    table.to_parquet(tmp, index=False)
    tmp.replace(path)
    return table


# ----------------------------------------------------------------- q95 from the map
Q95_SURFACE = 0.95


def _bilinear(field, grid_r, grid_z, rq, zq) -> np.ndarray:
    """Bilinear sample of a ``(nZ, nR)`` field at points ``(rq, zq)`` in metres (clipped to the grid)."""
    r = np.asarray(grid_r, dtype=np.float64)
    z = np.asarray(grid_z, dtype=np.float64)
    fr = np.clip((np.asarray(rq, dtype=np.float64) - r[0]) / (r[1] - r[0]), 0, r.size - 1 - 1e-9)
    fz = np.clip((np.asarray(zq, dtype=np.float64) - z[0]) / (z[1] - z[0]), 0, z.size - 1 - 1e-9)
    ir, iz = np.floor(fr).astype(int), np.floor(fz).astype(int)
    tr, tz = fr - ir, fz - iz
    f = np.asarray(field, dtype=np.float64)
    return ((1 - tz) * (1 - tr) * f[iz, ir] + (1 - tz) * tr * f[iz, ir + 1]
            + tz * (1 - tr) * f[iz + 1, ir] + tz * tr * f[iz + 1, ir + 1])


def q95_from_map(psi, grid_r, grid_z, tf_ka, n_turns: float, machine: str | MachineConfig = "DIII-D",
                 psi_n: float = Q95_SURFACE, envelope=None, scorer=None, use_envelope: bool = True) -> float:
    """Safety factor of the ``psi_n`` surface, integrated along the surface of one stored map.

    The axis and the boundary come from the scorer's functions: the boundary is its LCFS
    (``extract_lcfs_with_sign``, with the flux sign under which it was found), the axis its
    O-point under that sign; ``psi_axis`` is the flux of the O-point cell, ``psi_boundary``
    the flux sampled along the LCFS. The surface is the closed contour of
    ``psi_axis + psi_n (psi_boundary - psi_axis)`` enclosing the axis, and

        q = F / (2 pi) * closed integral of dl / (R |grad psi|),   F = 2e-7 n_turns |I_tf|

    with ``I_tf`` in amperes, psi in Wb/rad, so that ``|grad psi| / R`` is the poloidal
    field. The gradient is by central differences, sampled bilinearly at the midpoints of
    the contour segments. The result is positive and the same for ``psi`` and ``-psi``.

    Args:
        psi: ``(nZ, nR)`` stored flux map in Wb/rad.
        grid_r, grid_z: Grid axes in metres (uniform).
        tf_ka: Toroidal field coil current in kA (sign ignored).
        n_turns: Turns of the toroidal field coil set.
        machine: Machine name or configuration (flux sign convention and envelope).
        psi_n: Normalized flux of the surface, 0.95 for q95.
        envelope, scorer: ``starter_kit.envelope_mask`` and ``starter_kit.scoring``; loaded
            when None.
        use_envelope: Whether the boundary search is confined to the scorer's plasma
            envelope; False for synthetic maps on another grid.

    Returns:
        q of the surface, or NaN when no boundary, no closed surface or no finite field.
    """
    cfg = machine if isinstance(machine, MachineConfig) else machine_config(machine)
    scorer = scorer or starter_kit.scoring()
    r = np.asarray(grid_r, dtype=np.float64)
    z = np.asarray(grid_z, dtype=np.float64)
    p = np.asarray(psi, dtype=np.float64)
    if not np.isfinite(p).all() or not np.isfinite(tf_ka):
        return np.nan
    mask_coarse = mask_f = None
    if use_envelope:
        envelope = envelope or starter_kit.envelope_mask(cfg)
        mask_coarse, mask_f = envelope.mask_coarse, envelope.mask_f
    lcfs, sign = scorer.lcfs.extract_lcfs_with_sign(p, r, z, cfg.name, mask_coarse, mask_f)
    if lcfs is None or len(lcfs) < 8:
        return np.nan
    _, _, iz, ir = scorer.derive.magnetic_axis(p, r, z, cfg.name, mask_coarse, axis_sign=sign)
    psi_axis = float(p[iz, ir])
    psi_boundary = float(np.mean(_bilinear(p, r, z, lcfs[:, 0], lcfs[:, 1])))
    if not psi_boundary != psi_axis:
        return np.nan
    level = psi_axis + psi_n * (psi_boundary - psi_axis)
    surface = None
    for c in scorer.contour.find_contours(p, level):
        if len(c) >= 4 and scorer.contour.is_closed(c) and scorer.contour.encloses_point(c, iz, ir):
            surface = scorer.contour.grid_to_physical(c, r, z)
            break
    if surface is None:
        return np.nan
    dpsi_dz, dpsi_dr = np.gradient(p, z, r)
    grad = np.hypot(dpsi_dr, dpsi_dz)
    mid = 0.5 * (surface[1:] + surface[:-1])
    dl = np.hypot(*(surface[1:] - surface[:-1]).T)
    g = _bilinear(grad, r, z, mid[:, 0], mid[:, 1])
    if not (g > 0).all():
        return np.nan
    f_t = 2e-7 * float(n_turns) * abs(float(tf_ka)) * A_PER_KA
    return float(f_t / (2.0 * np.pi) * np.sum(dl / (mid[:, 0] * g)))


def q95_frames(shot_id: str, frame_index, maps, tf_ka, grid_r, grid_z, n_turns: float, machine: str = "DIII-D",
               levels: Sequence[float] = (Q95_SURFACE,)) -> pd.DataFrame:
    """``q95_from_map`` of one shot's maps at each level, as rows ``shot, frame, q95_calc_<level>``
    (``q95_calc_095`` for 0.95). Runs in a worker process."""
    cfg = machine_config(machine)
    scorer = starter_kit.scoring()
    envelope = starter_kit.envelope_mask(cfg)
    rows = []
    for frame, psi, tf in zip(np.asarray(frame_index), np.asarray(maps), np.asarray(tf_ka, dtype=np.float64)):
        row = {"shot": shot_id, "frame": int(frame)}
        for level in levels:
            row[q95_column(level)] = q95_from_map(psi, grid_r, grid_z, tf, n_turns, cfg, level, envelope, scorer)
        rows.append(row)
    return pd.DataFrame(rows, columns=["shot", "frame", *[q95_column(level) for level in levels]])


def q95_column(level: float) -> str:
    return f"q95_calc_{int(round(level * 100)):03d}"


def q95_table(items: Iterable[tuple[str, np.ndarray, np.ndarray, np.ndarray]], grid_r, grid_z, n_turns: float,
              machine: str = "DIII-D", levels: Sequence[float] = (Q95_SURFACE,), workers: int | None = None) -> pd.DataFrame:
    """Calculated q95 of many shots' maps, in the order given.

    Args:
        items: ``(shot_id, frame_index, maps, tf_ka)`` per shot, ``tf_ka`` one value per frame.
        grid_r, grid_z: Grid axes in metres.
        n_turns: Toroidal field coil turns.
        machine: Machine name.
        levels: Normalized flux of the surfaces to integrate on.
        workers: Processes; the CPU count minus two when None, one for a single process.
    """
    r = np.asarray(grid_r, dtype=np.float64)
    z = np.asarray(grid_z, dtype=np.float64)
    args = ((s, f, m, t, r, z, float(n_turns), machine, tuple(levels)) for s, f, m, t in items)
    parts = list(bl._map_ordered(q95_frames, args, workers))
    if not parts:
        return pd.DataFrame(columns=["shot", "frame", *[q95_column(level) for level in levels]])
    return pd.concat(parts, ignore_index=True)


def cached_q95_table(path: Path | str, items_fn: Callable[[], Iterable], grid_r, grid_z, n_turns: float,
                     machine: str = "DIII-D", levels: Sequence[float] = (Q95_SURFACE,), workers: int | None = None) -> pd.DataFrame:
    """``q95_table`` read from ``path`` when it exists, computed and written there otherwise."""
    path = Path(path)
    if path.is_file():
        return pd.read_parquet(path)
    table = q95_table(items_fn(), grid_r, grid_z, n_turns, machine, levels, workers)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.parquet")
    table.to_parquet(tmp, index=False)
    tmp.replace(path)
    return table


# --------------------------------------------------------------------- frame quantities
def frame_quantities(frames: pd.DataFrame, shapes: pd.DataFrame, machine: str | MachineConfig = "DIII-D",
                     n_turns: float | None = None, pressure_column: str = DEFAULT_PRESSURE_COLUMN) -> pd.DataFrame:
    """Joins frame-table rows with boundary shapes and adds the formula inputs and outputs.

    Added columns: ``ip_ma`` (signed), ``bt`` (tesla at ``R_geo``), ``delta``,
    ``peak_to_mean`` (core ``pe_peak`` over ``pe_mean``), ``p_pa`` (``pressure_column`` in
    kPa times 1000), ``q95_formula`` and ``betan_formula``. Rows keep the frame order;
    frames without a shape get NaN.
    """
    cfg = machine if isinstance(machine, MachineConfig) else machine_config(machine)
    n_turns = TF_TURNS[cfg.name] if n_turns is None else n_turns
    out = frames.merge(shapes, on=["shot", "frame"], how="left", validate="one_to_one")
    out["ip_ma"] = out["ip__mean"].to_numpy(dtype=np.float64) * MA_PER_KA
    out["bt"] = toroidal_field(out["tf__mean"].to_numpy(dtype=np.float64), out["R_geo"].to_numpy(dtype=np.float64), n_turns)
    out["delta"] = 0.5 * (out["tri_top"].to_numpy(dtype=np.float64) + out["tri_bot"].to_numpy(dtype=np.float64))
    with np.errstate(divide="ignore", invalid="ignore"):
        out["peak_to_mean"] = (out["ts_core_pe_peak__interp"].to_numpy(dtype=np.float64)
                               / out["ts_core_pe_mean__interp"].to_numpy(dtype=np.float64))
    out["p_pa"] = out[pressure_column].to_numpy(dtype=np.float64) * PA_PER_KPA
    out["q95_formula"] = q95_formula(out["a"], out["R_geo"], out["kappa"], out["delta"], out["bt"], out["ip_ma"])
    out["betan_formula"] = beta_n_formula(out["p_pa"], out["a"], out["bt"], out["ip_ma"])
    return out


def candidate_rules(frames: pd.DataFrame, tf_min_ka: float = 50.0, ip_over_peak_min: float = 0.1,
                    q95_band: tuple[float, float] = (2.91, 11.76), betan_band: tuple[float, float] = (0.08, 3.31),
                    excluded_shots: Sequence[str] = ("5df1630aa7", "955c35c962")) -> dict[str, np.ndarray]:
    """Bool masks of the frames each candidate label-filter rule would remove.

    ``valid_label`` is the frames with both labels finite (the base filter, always applied by
    the notebook); the other rules are candidates whose frame cost the notebook reports.
    """
    q95 = frames["efit_q95"].to_numpy(dtype=np.float64)
    betan = frames["efit_beta_n"].to_numpy(dtype=np.float64)
    tf = np.abs(frames["tf__mean"].to_numpy(dtype=np.float64))
    over = frames["ip_over_peak"].to_numpy(dtype=np.float64)
    shots = frames["shot"].astype(str).to_numpy()
    with np.errstate(invalid="ignore"):
        return {
            "invalid_label": ~(np.isfinite(q95) & np.isfinite(betan)),
            "tf_under_threshold": tf < tf_min_ka,
            "ip_under_fraction_of_peak": over < ip_over_peak_min,
            "q95_outside_band": np.isfinite(q95) & ((q95 < q95_band[0]) | (q95 > q95_band[1])),
            "betan_outside_band": np.isfinite(betan) & ((betan < betan_band[0]) | (betan > betan_band[1])),
            "excluded_shots": np.isin(shots, list(excluded_shots)),
        }


# -------------------------------------------------------------------------------- scores
def pooled_r2(y_true, y_pred) -> float:
    """One R2 over all frames, as the scorer computes it: ``1 - SS_res / SS_tot`` with the total
    sum of squares about the pooled mean, unclipped. Frames with a non-finite prediction count
    as error against the pooled mean of the labels, as the scorer fills them; frames with a
    non-finite label are dropped."""
    y = np.asarray(y_true, dtype=np.float64).ravel()
    p = np.asarray(y_pred, dtype=np.float64).ravel()
    keep = np.isfinite(y)
    y, p = y[keep], p[keep]
    if y.size == 0:
        return np.nan
    p = np.where(np.isfinite(p), p, y.mean())
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - float(np.sum((y - p) ** 2)) / ss_tot if ss_tot > 0 else 0.0


def best_multiplier(label, formula) -> float:
    """The scalar ``c`` minimizing ``sum (label - c formula)^2`` over the finite pairs."""
    y = np.asarray(label, dtype=np.float64).ravel()
    f = np.asarray(formula, dtype=np.float64).ravel()
    keep = np.isfinite(y) & np.isfinite(f)
    denominator = float(np.sum(f[keep] ** 2))
    return float(np.sum(y[keep] * f[keep]) / denominator) if denominator > 0 else np.nan


# ---------------------------------------------------------------------------- correction
@dataclass
class CorrectionModel:
    """Multiplicative correction of a formula, learned from quantities without units.

    The target is ``log(label / formula)`` over the training frames with a finite, positive
    ratio; the prediction is the factor ``exp(model(x))``. Missing features are filled with
    the training median. ``method`` is ``"ridge"`` (standardized inputs, fixed ``alpha``) or
    ``"gbm"`` (a small histogram gradient-boosted model, no early stopping, fixed seed).

    Attributes:
        features: Input columns.
        method: ``"ridge"`` or ``"gbm"``.
        alpha: Ridge regularization.
        max_iter, max_depth, learning_rate: Gradient boosting size.
        seed: Random seed of the boosted model.
    """

    features: tuple[str, ...] = CORRECTION_FEATURES
    method: str = "ridge"
    alpha: float = 1.0
    max_iter: int = 200
    max_depth: int = 3
    learning_rate: float = 0.1
    seed: int = 0
    fill: dict[str, float] = field(default_factory=dict)
    mean: np.ndarray | None = None
    std: np.ndarray | None = None
    model: object = None
    n_train: int = 0

    def _matrix(self, table: pd.DataFrame, fit: bool) -> np.ndarray:
        x = np.column_stack([table[c].to_numpy(dtype=np.float64) for c in self.features])
        if fit:
            self.fill = {c: float(np.nanmedian(x[:, j])) for j, c in enumerate(self.features)}
        for j, c in enumerate(self.features):
            col = x[:, j]
            x[:, j] = np.where(np.isfinite(col), col, self.fill.get(c, 0.0))
        if fit:
            self.mean = x.mean(axis=0)
            std = x.std(axis=0)
            self.std = np.where(std > 0, std, 1.0)
        return (x - self.mean) / self.std

    def fit(self, table: pd.DataFrame, label, formula) -> "CorrectionModel":
        """Fits on the rows whose label and formula are finite and positive."""
        y = np.asarray(label, dtype=np.float64)
        f = np.asarray(formula, dtype=np.float64)
        keep = np.isfinite(y) & np.isfinite(f) & (y > 0) & (f > 0)
        if keep.sum() < 2:
            raise ValueError("at least two frames with a positive label and formula are needed")
        x = self._matrix(table.loc[keep], fit=True)
        target = np.log(y[keep] / f[keep])
        if self.method == "ridge":
            from sklearn.linear_model import Ridge
            self.model = Ridge(alpha=self.alpha).fit(x, target)
        elif self.method == "gbm":
            from sklearn.ensemble import HistGradientBoostingRegressor
            self.model = HistGradientBoostingRegressor(max_iter=self.max_iter, max_depth=self.max_depth,
                                                       learning_rate=self.learning_rate, early_stopping=False,
                                                       random_state=self.seed).fit(x, target)
        else:
            raise ValueError(f"unknown method {self.method!r}")
        self.n_train = int(keep.sum())
        return self

    def factor(self, table: pd.DataFrame) -> np.ndarray:
        """The multiplicative correction of every row."""
        if self.model is None:
            raise RuntimeError("fit the model first")
        return np.exp(self.model.predict(self._matrix(table, fit=False)))

    def predict(self, table: pd.DataFrame, formula) -> np.ndarray:
        """``formula`` times the learned factor, NaN where the formula is not finite."""
        return np.asarray(formula, dtype=np.float64) * self.factor(table)
