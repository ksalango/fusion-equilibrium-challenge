"""The betaN heads of the prediction chain (``docs/PIPELINE.md``, Part 5 step 4; F99), and the
pieces of the betaN head study (``scripts/betan_heads.py``) they are built from.

One head per machine:

- DIII-D, ``ridge_raw``: a ridge on the 25 raw inputs of ``hybrid_ridge`` with the currents in
  the physical frame (``targets.signed_frames``), standardized by ``baseline.InputScaler``. It
  reads no map, so no refit concerns it.
- MAST, ``formula_gbm``: the geometry-explicit base ``100 (2 mu0 p_e / Bt0^2) a Bt0 / abs(Ip)``
  (``Bt0`` the vacuum field at the machine's R0 with its toroidal-field turns, ``a`` the minor
  radius of the predicted map's boundary) times one constant times a gradient-boosted
  correction on the transferable inputs (``HEAD_FEATURES``: unscaled v2 feature values,
  ``q_cyl`` clipped to the DIII-D training band, the boundary's ``kappa`` and ``a / a0``).
  The constant and the correction are fitted on DIII-D training frames with the DIII-D
  constants in the base and applied on MAST with the MAST constants.

Stage-2 refit (``refit_mast``): before a submission that uses the MAST head, the base constant
and the correction are refitted on the shapes of the maps the submission's model predicts for
the training frames. The DIII-D head is carried over unchanged (the same object).

Training frames are every ``stride``-th frame of each training shot that passes the lighter
label filter (``label_filter``, F77). Labels enter only as regression targets and through the
filter.
"""

from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from fusion_eq import baseline as bl, feature_checks as fc, label_filter as lf, scalars as sc, targets
from fusion_eq.machines import MachineConfig, machine_config

MACHINE = "DIII-D"
LABEL = "efit_beta_n"
BASE = "betan_base"
PRESSURE_COLUMN = sc.DEFAULT_PRESSURE_COLUMN
# Feature-table columns of the transferable list; q_cyl enters clipped.
FEATURE_COLUMNS = ("ts_pe_peaking", "ts_valid", "shafranov_lambda", "greenwald_frac", "ip_frac", "ohmic_flux_used",
                   "shaping", "q_cyl")
CLIPPED_COLUMN = "q_cyl"
CLIPPED_NAME = "q_cyl_clipped"
HEAD_FEATURES = (*[CLIPPED_NAME if c == CLIPPED_COLUMN else c for c in FEATURE_COLUMNS], "kappa", "a_over_a0")
# Frame-table columns the heads read at prediction time (inputs only).
FRAME_INPUT_COLUMNS = tuple(dict.fromkeys([*bl.input_columns(MACHINE), "tf__mean", "ip__mean", PRESSURE_COLUMN]))
# The F75 settings of the gradient-boosted correction.
GBM_SETTINGS = {"max_iter": 200, "max_depth": 3, "learning_rate": 0.1, "seed": 0}
HEAD_NAMES = {"DIII-D": "ridge_raw", "MAST": "formula_gbm"}


# -------------------------------------------------------------------------------- frames
def strided(frames: pd.DataFrame, stride: int) -> pd.DataFrame:
    """Every ``stride``-th frame of each shot, in shot and frame order (the rule of the scalar study)."""
    rows = frames.assign(shot=frames["shot"].astype(str)).sort_values(["shot", "frame"])
    return rows.loc[rows.groupby("shot", sort=False).cumcount().to_numpy() % int(stride) == 0].reset_index(drop=True)


# ----------------------------------------------------------------------- band and corner
def training_statistics(q_cyl, beta_p) -> dict:
    """Percentiles of the DIII-D training frames used by the clip and the transfer proxy.

    Returns:
        ``{"q_cyl": {n, q01, q10, q50, q99}, "ts_beta_p": {n, q50, q75}}`` over the finite values;
        q01 and q99 are the band of X8 (``feature_checks.reference_statistics``).
    """
    q = np.asarray(q_cyl, dtype=np.float64)
    b = np.asarray(beta_p, dtype=np.float64)
    sq, sb = fc.reference_statistics(q), fc.reference_statistics(b)
    finite_q = q[np.isfinite(q)]
    q10 = float(np.percentile(finite_q, 10.0)) if finite_q.size else np.nan
    return {"q_cyl": {"n": sq["n"], "q01": sq["q01"], "q10": q10, "q50": sq["q50"], "q99": sq["q99"]},
            "ts_beta_p": {"n": sb["n"], "q50": sb["q50"], "q75": sb["q75"]}}


def clip_to_band(values, band: tuple[float, float]) -> np.ndarray:
    """Values clipped to ``[band[0], band[1]]``; non-finite values stay NaN."""
    v = np.asarray(values, dtype=np.float64)
    lo, hi = float(band[0]), float(band[1])
    if not lo <= hi:
        raise ValueError(f"band lower edge {lo} above upper edge {hi}")
    return np.where(np.isfinite(v), np.clip(v, lo, hi), np.nan)


# ---------------------------------------------------------------------------------- base
def toroidal_field_at_r0(tf_ka, machine: str | MachineConfig = MACHINE) -> np.ndarray:
    """Vacuum toroidal field in tesla at the machine's reference radius R0.

    Raises:
        KeyError: when the machine's toroidal-field turn count is not verified.
    """
    cfg = machine if isinstance(machine, MachineConfig) else machine_config(machine)
    return sc.toroidal_field(tf_ka, cfg.major_radius.value, sc.TF_TURNS[cfg.name])


def betan_base(p_e_kpa, a_m, tf_ka, ip_ka, machine: str | MachineConfig = MACHINE) -> np.ndarray:
    """The geometry-explicit base: ``100 (2 mu0 p_e / Bt0^2) a Bt0 / abs(Ip)``.

    Args:
        p_e_kpa: Core electron pressure summary in kPa.
        a_m: Minor radius in metres.
        tf_ka: Toroidal field coil current in kA (sign ignored).
        ip_ka: Plasma current in kA (sign ignored).
        machine: Machine name or configuration (R0 and the turn count).

    Returns:
        betaN in percent metre tesla per MA, NaN where an input is missing.
    """
    p_pa = np.asarray(p_e_kpa, dtype=np.float64) * sc.PA_PER_KPA
    ip_ma = np.asarray(ip_ka, dtype=np.float64) * sc.MA_PER_KA
    return sc.beta_n_formula(p_pa, a_m, toroidal_field_at_r0(tf_ka, machine), ip_ma)


def head_table(frames: pd.DataFrame, shapes: pd.DataFrame, q_cyl_band: tuple[float, float],
               machine: str | MachineConfig = MACHINE) -> pd.DataFrame:
    """Frames joined with one stage's shapes, with the base and the head features added.

    Added columns: ``a``, ``kappa`` and ``boundary_found`` from ``shapes`` (NaN and False for
    a frame without a shape row), ``q_cyl_clipped``, ``a_over_a0`` and ``betan_base``.
    """
    cfg = machine if isinstance(machine, MachineConfig) else machine_config(machine)
    part = shapes[["shot", "frame", "a", "kappa", "boundary_found"]].assign(shot=lambda t: t["shot"].astype(str))
    out = frames.merge(part, on=["shot", "frame"], how="left", validate="one_to_one")
    out["boundary_found"] = out["boundary_found"].eq(True).to_numpy()
    out[CLIPPED_NAME] = clip_to_band(out[CLIPPED_COLUMN], q_cyl_band)
    out["a_over_a0"] = out["a"].to_numpy(dtype=np.float64) / cfg.minor_radius.value
    out[BASE] = betan_base(out[PRESSURE_COLUMN], out["a"], out["tf__mean"], out["ip__mean"], cfg)
    return out


# ------------------------------------------------------------------------------ DIII-D head
@dataclass
class RawRidge:
    """Ridge on the raw inputs of ``hybrid_ridge`` with the currents in the physical frame
    (``targets.signed_frames``), standardized by ``baseline.InputScaler`` as in F75."""

    alpha: float = 1.0
    machine: str = MACHINE
    scaler: bl.InputScaler | None = None
    weights: np.ndarray | None = None
    intercept: float = 0.0
    n_train: int = 0

    def fit(self, table: pd.DataFrame, label, keep) -> "RawRidge":
        signed = targets.signed_frames(table, self.machine)
        keep = np.asarray(keep, dtype=bool)
        self.scaler = bl.InputScaler.fit(signed.loc[keep], bl.input_columns(self.machine))
        x, complete = self.scaler.transform(signed)
        y = np.asarray(label, dtype=np.float64)
        rows = keep & complete & np.isfinite(y)
        w, b = bl.ridge_fit(x[rows], y[rows], self.alpha)
        self.weights, self.intercept, self.n_train = w[:, 0], float(b[0]), int(rows.sum())
        return self

    def predict(self, table: pd.DataFrame) -> np.ndarray:
        x, _ = self.scaler.transform(targets.signed_frames(table, self.machine))
        return x @ self.weights + self.intercept


# ------------------------------------------------------------------------- production heads
@dataclass
class BetanHeads:
    """The two betaN heads of a submission.

    Attributes:
        d3d: The DIII-D ridge on the raw inputs.
        mast_constant: Least-squares multiplier of the base on the training frames; None when
            the MAST head is not fitted (a prediction without MAST).
        mast_correction: The gradient-boosted correction of ``constant * base``, or None.
        q_cyl_band: DIII-D training band ``(q01, q99)`` of ``q_cyl``.
        record: How each head was fitted (lists, stride, frame counts, shape source).
    """

    d3d: RawRidge
    mast_constant: float | None
    mast_correction: sc.CorrectionModel | None
    q_cyl_band: tuple[float, float]
    record: dict = field(default_factory=dict)

    def predict(self, machine: str, frames: pd.DataFrame, shapes: pd.DataFrame | None = None) -> np.ndarray:
        """betaN of every row of ``frames``.

        Args:
            machine: ``DIII-D`` or ``MAST``.
            frames: Rows with ``shot``, ``frame``, the frame-table inputs of
                ``FRAME_INPUT_COLUMNS`` and, on MAST, the unscaled ``FEATURE_COLUMNS``.
            shapes: On MAST, the boundary shapes of the predicted maps (``scalars.shape_table``
                rows: ``shot``, ``frame``, ``a``, ``kappa``, ``boundary_found``).

        Returns:
            One value per row; NaN on MAST where the predicted map has no boundary.
        """
        if machine == "DIII-D":
            return self.d3d.predict(frames)
        if machine == "MAST":
            if self.mast_correction is None:
                raise ValueError("the MAST head is not fitted")
            if shapes is None:
                raise ValueError("the MAST head needs the shapes of the predicted maps")
            table = head_table(frames, shapes, self.q_cyl_band, "MAST")
            return self.mast_correction.predict(table, self.mast_constant * table[BASE].to_numpy(dtype=np.float64))
        raise ValueError(f"unknown machine {machine!r}")

    def save(self, path: Path | str) -> str:
        """Writes the heads (pickle, the format of the fitted scikit-learn model) atomically;
        returns the file's SHA-256."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(pickle.dumps(self, protocol=5))
        tmp.replace(path)
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @classmethod
    def load(cls, path: Path | str) -> "BetanHeads":
        heads = pickle.loads(Path(path).read_bytes())
        if not isinstance(heads, cls):
            raise TypeError(f"{path} does not hold betaN heads")
        return heads

    def summary(self) -> dict:
        """JSON-serialisable description for run records."""
        mast = self.mast_correction
        return {"heads": dict(HEAD_NAMES), "d3d_n_train": self.d3d.n_train, "d3d_alpha": self.d3d.alpha,
                "mast_fitted": mast is not None, "mast_constant": self.mast_constant,
                "mast_n_train": None if mast is None else mast.n_train, "mast_features": list(HEAD_FEATURES),
                "gbm": dict(GBM_SETTINGS), "q_cyl_band": list(self.q_cyl_band), **self.record}


def fit_d3d_head(frames: pd.DataFrame, alpha: float = 1.0) -> RawRidge:
    """The DIII-D ridge on the training rows passing the lighter label filter.

    Args:
        frames: Strided training frame-table rows with ``FRAME_INPUT_COLUMNS``, the label
            filter's columns and the label.
    """
    keep = lf.lighter_label_filter(frames)
    return RawRidge(alpha=alpha, machine=MACHINE).fit(frames, frames[LABEL], keep)


def fit_mast_head(table: pd.DataFrame, keep, alpha: float = 1.0) -> tuple[float, sc.CorrectionModel]:
    """The base constant and the gradient-boosted correction on DIII-D training rows.

    Args:
        table: ``head_table`` rows of the training frames (DIII-D constants in the base).
        keep: Bool per row, the frames passing the lighter label filter.
        alpha: Ridge regularization (unused by the boosted model; kept for the record).

    Raises:
        ValueError: when fewer than two training frames have a finite base and label.
    """
    keep = np.asarray(keep, dtype=bool)
    y = table[LABEL].to_numpy(dtype=np.float64)
    base = table[BASE].to_numpy(dtype=np.float64)
    usable = keep & np.isfinite(y) & np.isfinite(base)
    if usable.sum() < 2:
        raise ValueError(f"{int(usable.sum())} training frames with a finite base and label; the MAST head needs at least two")
    constant = sc.best_multiplier(y[keep], base[keep])
    rows = table.loc[keep]
    model = sc.CorrectionModel(features=HEAD_FEATURES, method="gbm", alpha=alpha, max_iter=GBM_SETTINGS["max_iter"],
                               max_depth=GBM_SETTINGS["max_depth"], learning_rate=GBM_SETTINGS["learning_rate"],
                               seed=GBM_SETTINGS["seed"])
    model.fit(rows, rows[LABEL], constant * rows[BASE].to_numpy(dtype=np.float64))
    return float(constant), model


def fit_heads(frames: pd.DataFrame, shapes: pd.DataFrame | None, q_cyl_band: tuple[float, float],
              record: Mapping | None = None, alpha: float = 1.0) -> BetanHeads:
    """Both heads on strided DIII-D training rows and the shapes of the given maps.

    Args:
        frames: Training rows: frame-table inputs, filter columns, the label and the unscaled
            ``FEATURE_COLUMNS``.
        shapes: Boundary shapes of the maps the MAST head is fitted on; None fits the DIII-D
            head alone.
        q_cyl_band: DIII-D training band of ``q_cyl``.
        record: Description of the fit for the run record.
    """
    keep = lf.lighter_label_filter(frames)
    d3d = RawRidge(alpha=alpha, machine=MACHINE).fit(frames, frames[LABEL], keep)
    rec = {"n_train_frames": int(len(frames)), "n_train_kept": int(keep.sum()), **dict(record or {})}
    if shapes is None:
        return BetanHeads(d3d=d3d, mast_constant=None, mast_correction=None, q_cyl_band=tuple(map(float, q_cyl_band)), record=rec)
    table = head_table(frames, shapes, q_cyl_band, MACHINE)
    constant, correction = fit_mast_head(table, keep, alpha)
    rec["n_train_without_boundary"] = int((~table["boundary_found"].to_numpy()).sum())
    return BetanHeads(d3d=d3d, mast_constant=constant, mast_correction=correction, q_cyl_band=tuple(map(float, q_cyl_band)),
                      record=rec)


def refit_mast(heads: BetanHeads, frames: pd.DataFrame, shapes: pd.DataFrame, record: Mapping | None = None) -> BetanHeads:
    """The stage-2 refit: the MAST constant and correction refitted on the shapes of the
    submission model's maps of the training frames; the DIII-D head and the band unchanged.

    Args:
        heads: The heads before the refit.
        frames: Training rows as for ``fit_heads``.
        shapes: Boundary shapes of the submission model's maps of those rows.
        record: Description of the refit (model checkpoints, lists, stride).
    """
    keep = lf.lighter_label_filter(frames)
    table = head_table(frames, shapes, heads.q_cyl_band, MACHINE)
    constant, correction = fit_mast_head(table, keep, 1.0 if heads.mast_correction is None else heads.mast_correction.alpha)
    rec = dict(heads.record)
    rec["stage2_refit"] = {"n_train_frames": int(len(frames)), "n_train_kept": int(keep.sum()),
                           "n_train_without_boundary": int((~table["boundary_found"].to_numpy()).sum()),
                           "constant_before": heads.mast_constant, "constant_after": constant, **dict(record or {})}
    return replace(heads, mast_constant=constant, mast_correction=correction, record=rec)
