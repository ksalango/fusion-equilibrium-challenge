"""betaN head comparison (diagnostic): four ways of predicting ``efit_beta_n`` on the working
lists, in the two stages of the scalar study (F73 to F75, ``notebooks/05_scalars.ipynb``).

Frames: every ``--frame-stride``-th frame (4) of each shot of the working lists of
``splits/split_v1.json`` (300 training, 100 validation shots), as in the scalar study.
Training frames pass the lighter label filter (``fusion_eq.label_filter``, F77): both labels
finite and positive, ``abs(tf) >= 50 kA``, ``ip_over_peak >= 0.1``, shots 5df1630aa7 and
955c35c962 left out. Validation frames are those with both labels finite, as in F73.

Stages: stage 1 takes the boundary shape (``a``, ``kappa``) from the true maps, stage 2 from
the maps of ``hybrid_ridge``. The shape tables of the scalar study are read from
``--shape-dir`` when present and computed there with the scorer's functions otherwise
(stage 2 then needs the fitted models of ``compare_baselines.py --working``). The base
constant and every model are refitted per stage. Stage 1 is the label-side bound: its
shape inputs come from the true maps and cannot be computed for a test shot.

Arms, all predicting ``efit_beta_n``:

- ``ridge_raw``: ridge (``--alpha``, 1) on the 25 raw inputs of ``hybrid_ridge`` (currents
  multiplied by sign(Ip), the three core Thomson summaries, the Thomson flag); the
  reference. It reads no shape, so the two stages agree.
- ``ridge_features``: ridge on the transferable feature list, direct prediction.
- ``formula_ridge``: the base times a constant times a ridge correction on the same list.
- ``formula_gbm``: the same base and constant times a gradient-boosted correction with the
  settings of the F75 GBM (``scalars.CorrectionModel``: 200 iterations, depth 3, rate 0.1,
  seed 0).

Base, from the definition of betaN with the vacuum field at the machine's reference radius:

    Bt0    = 2e-7 N_tf abs(I_tf) / R0                     I_tf in A, R0 and N_tf from machines.py
    base   = 100 (2 mu0 p_e / Bt0^2) a Bt0 / abs(Ip)      = 200 mu0 p_e a / (Bt0 abs(Ip))

with ``p_e = ts_core_pe_mean__interp`` in Pa, ``a`` the stage's minor radius in m and Ip in
MA. Elongation does not enter the base; it enters the corrections. One constant ``c``
(least squares of the label on the base over the training frames) multiplies the base.

Transferable feature list: v2 columns where MAST sits inside the DIII-D band (X8) plus map
shapes: ``ts_pe_peaking``, ``ts_valid``, ``shafranov_lambda``, ``greenwald_frac``,
``ip_frac``, ``ohmic_flux_used``, ``shaping``, ``q_cyl`` clipped to the DIII-D training 1st
to 99th percentile band, the stage's ``kappa`` and ``a / a0``. Values are unscaled feature
values; each model fills missing values with the training median and standardizes.

Scores: pooled R2 on validation (scorer's convention, unclipped) over all frames, the
negative-Ip frames, and a label-free transfer proxy: frames with ``q_cyl`` below the DIII-D
training 10th percentile and ``ts_beta_p`` below the training median (relaxed to the 75th
percentile when fewer than ``--min-corner-frames``, 300, qualify). The band, the 10th
percentile and the median are taken over every frame of the full training list of the v2
feature table. Each arm's training R2 and each stage's base constant are reported.

Writes ``betan_heads.md`` and ``betan_heads.json`` to ``--out`` (``reports/scalars``).
Usage::

    python scripts/betan_heads.py
    python scripts/betan_heads.py --workers 8      # only used when a shape table is computed
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Mapping

import numpy as np
import pandas as pd

from fusion_eq import (baseline as bl, dataset, feature_checks as fc, features as fe, frame_table as ft,
                       label_filter as lf, scalars as sc, split as sp, targets)
from fusion_eq.dataset import default_data_root, run_info
from fusion_eq.machines import MachineConfig, machine_config

DEFAULT_SPLIT = dataset.REPO_ROOT / "splits" / "split_v1.json"
DEFAULT_CONFIG = fe.DEFAULT_CONFIG.with_name("features_v2.yaml")
DEFAULT_SHAPE_DIR = dataset.DEFAULT_PROCESSED_ROOT / "scalars"
DEFAULT_OUT = dataset.REPO_ROOT / "reports" / "scalars"
DEFAULT_GREEN_DIR = dataset.DEFAULT_PROCESSED_ROOT / "green"
SPLIT_NAME = "diii_d_train"
MACHINE = "DIII-D"
LABEL = "efit_beta_n"
OTHER_LABEL = "efit_q95"
BASE = "betan_base"
PRESSURE_COLUMN = sc.DEFAULT_PRESSURE_COLUMN
# Feature-table columns of the transferable list; q_cyl enters clipped.
FEATURE_COLUMNS = ("ts_pe_peaking", "ts_valid", "shafranov_lambda", "greenwald_frac", "ip_frac", "ohmic_flux_used",
                   "shaping", "q_cyl")
CLIPPED_COLUMN = "q_cyl"
CLIPPED_NAME = "q_cyl_clipped"
# Feature-table columns of the transfer proxy only (not model inputs).
CORNER_COLUMNS = ("q_cyl", "ts_beta_p")
HEAD_FEATURES = (*[CLIPPED_NAME if c == CLIPPED_COLUMN else c for c in FEATURE_COLUMNS], "kappa", "a_over_a0")
# Frame-table columns read as inputs by the raw ridge or the base.
FRAME_INPUTS = tuple(dict.fromkeys([*bl.input_columns(MACHINE), "tf__mean", "ip__mean", PRESSURE_COLUMN]))
FRAME_COLUMNS = list(dict.fromkeys([*FRAME_INPUTS, "ip_over_peak", "ip_direction", OTHER_LABEL, LABEL]))
ARMS = ("ridge_raw", "ridge_features", "formula_ridge", "formula_gbm")
CONTEXT_ARM = "base_x_constant"
REFERENCE_ARM = "ridge_raw"
SUBSETS = ("all", "negative_ip", "corner")
F75_RIDGE_R2 = 0.9409
MIN_CORNER_FRAMES = 300


def log(message: str) -> None:
    print(message, flush=True)


def display_path(path: Path | str) -> str:
    """A path relative to the repository when possible, so that the report carries no
    absolute home path."""
    return os.path.relpath(Path(path).resolve(), dataset.REPO_ROOT)


# --------------------------------------------------------------------------------- guard
def guard_inputs(frame_inputs: Iterable[str], feature_inputs: Iterable[str], config: fe.FeatureConfig,
                 machine: str = MACHINE) -> dict[str, list[str]]:
    """The label guard over every input of the study.

    Frame-table inputs go through ``frame_table.assert_input_columns``; each feature must be
    built by the configuration and the columns its function reads go through
    ``features.check_reads``.

    Returns:
        Feature name to the frame-table columns it reads.

    Raises:
        ValueError: when an input is a label, unknown, or a feature the configuration lacks.
    """
    ft.assert_input_columns(list(frame_inputs), machine)
    reads = fe.feature_reads(config, machine)
    missing = [c for c in feature_inputs if c not in reads]
    if missing:
        raise ValueError(f"features not built by version {config.version}: {missing}")
    return {c: fe.check_reads(reads[c], machine) for c in feature_inputs}


# -------------------------------------------------------------------------------- frames
def strided(frames: pd.DataFrame, stride: int) -> pd.DataFrame:
    """Every ``stride``-th frame of each shot, in shot and frame order (the rule of the scalar study)."""
    rows = frames.assign(shot=frames["shot"].astype(str)).sort_values(["shot", "frame"])
    return rows.loc[rows.groupby("shot", sort=False).cumcount().to_numpy() % int(stride) == 0].reset_index(drop=True)


def training_filter(frames: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    """The lighter label filter on training frames.

    Returns:
        ``(keep, counts)``: bool per frame, and one row per rule with the frames and shots it
        removes on its own, then ``removed (union)`` and ``kept``.
    """
    rules = lf.lighter_filter_rules(frames)
    keep = ~rules.any(axis=1).to_numpy()
    shots = frames["shot"].astype(str).to_numpy()
    rows = [{"rule": r, "frames": int(rules[r].sum()), "shots": int(len(set(shots[rules[r].to_numpy()])))} for r in lf.RULES]
    rows.append({"rule": "removed (union)", "frames": int((~keep).sum()), "shots": int(len(set(shots[~keep])))})
    rows.append({"rule": "kept", "frames": int(keep.sum()), "shots": int(len(set(shots[keep])))})
    return keep, pd.DataFrame(rows)


def validation_mask(frames: pd.DataFrame) -> np.ndarray:
    """Validation frames scored: both labels finite (F73)."""
    return (np.isfinite(frames[LABEL].to_numpy(dtype=np.float64))
            & np.isfinite(frames[OTHER_LABEL].to_numpy(dtype=np.float64)))


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


def corner_mask(q_cyl, beta_p, stats: Mapping, eligible, min_frames: int = MIN_CORNER_FRAMES) -> tuple[np.ndarray, dict]:
    """The MAST-like corner of DIII-D validation: ``q_cyl`` below the training 10th percentile
    and ``ts_beta_p`` below the training median, or below the 75th percentile when fewer than
    ``min_frames`` eligible frames qualify. Frames with a missing value are outside.

    Returns:
        ``(mask, record)``; the mask is restricted to ``eligible``.
    """
    q = np.asarray(q_cyl, dtype=np.float64)
    b = np.asarray(beta_p, dtype=np.float64)
    eligible = np.asarray(eligible, dtype=bool)
    with np.errstate(invalid="ignore"):
        low_q = q < stats["q_cyl"]["q10"]
        strict = eligible & low_q & (b < stats["ts_beta_p"]["q50"])
        relaxed_mask = eligible & low_q & (b < stats["ts_beta_p"]["q75"])
    record = {"q_cyl_below": stats["q_cyl"]["q10"], "n_strict": int(strict.sum()), "min_frames": int(min_frames)}
    if strict.sum() >= min_frames:
        return strict, {**record, "relaxed": False, "ts_beta_p_below": stats["ts_beta_p"]["q50"],
                        "ts_beta_p_percentile": 50, "n_frames": int(strict.sum())}
    return relaxed_mask, {**record, "relaxed": True, "ts_beta_p_below": stats["ts_beta_p"]["q75"],
                          "ts_beta_p_percentile": 75, "n_frames": int(relaxed_mask.sum())}


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


# ---------------------------------------------------------------------------------- arms
@dataclass
class StandardRidge:
    """Ridge on standardized columns; missing values take the training median.

    Attributes:
        columns: Input columns.
        alpha: Ridge regularization.
    """

    columns: tuple[str, ...]
    alpha: float = 1.0
    fill: dict[str, float] = field(default_factory=dict)
    mean: np.ndarray | None = None
    std: np.ndarray | None = None
    weights: np.ndarray | None = None
    intercept: float = 0.0
    n_train: int = 0

    def _matrix(self, table: pd.DataFrame, fit: bool) -> np.ndarray:
        x = np.column_stack([table[c].to_numpy(dtype=np.float64) for c in self.columns])
        if fit:
            self.fill = {c: float(np.nanmedian(x[:, j])) for j, c in enumerate(self.columns)}
        for j, c in enumerate(self.columns):
            x[:, j] = np.where(np.isfinite(x[:, j]), x[:, j], self.fill[c])
        if fit:
            self.mean = x.mean(axis=0)
            std = x.std(axis=0)
            self.std = np.where(std > 0, std, 1.0)
        return (x - self.mean) / self.std

    def fit(self, table: pd.DataFrame, label, keep) -> "StandardRidge":
        """Fits on the rows of ``keep`` whose label is finite."""
        y = np.asarray(label, dtype=np.float64)
        rows = np.asarray(keep, dtype=bool) & np.isfinite(y)
        x = self._matrix(table.loc[rows], fit=True)
        w, b = bl.ridge_fit(x, y[rows], self.alpha)
        self.weights, self.intercept, self.n_train = w[:, 0], float(b[0]), int(rows.sum())
        return self

    def predict(self, table: pd.DataFrame) -> np.ndarray:
        return self._matrix(table, fit=False) @ self.weights + self.intercept


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


@dataclass
class StageFit:
    """One stage's fitted constant and every arm's predictions.

    Attributes:
        constant: Least-squares multiplier of the base on the training frames.
        train, val: Arm name to its predictions on the training and validation rows.
        n_fit: Arm name to the frames its fit used.
        n_inputs: Arm name to its input count.
    """

    constant: float
    train: dict[str, np.ndarray]
    val: dict[str, np.ndarray]
    n_fit: dict[str, int]
    n_inputs: dict[str, int]


def fit_stage(train: pd.DataFrame, keep, val: pd.DataFrame, alpha: float = 1.0,
              features: tuple[str, ...] = HEAD_FEATURES, machine: str = MACHINE) -> StageFit:
    """Fits the base constant and the four arms on the training rows of ``keep``.

    Args:
        train, val: ``head_table`` rows of the training and validation frames.
        keep: Bool per training row, the frames fit for training.
        alpha: Ridge regularization of the ridge arms and the ridge correction.
        features: Inputs of ``ridge_features`` and of the corrections.
    """
    keep = np.asarray(keep, dtype=bool)
    y = train[LABEL].to_numpy(dtype=np.float64)
    base_train = train[BASE].to_numpy(dtype=np.float64)
    base_val = val[BASE].to_numpy(dtype=np.float64)
    constant = sc.best_multiplier(y[keep], base_train[keep])
    out = StageFit(constant=constant, train={}, val={}, n_fit={}, n_inputs={})

    raw = RawRidge(alpha=alpha, machine=machine).fit(train, y, keep)
    direct = StandardRidge(columns=tuple(features), alpha=alpha).fit(train, y, keep)
    for arm, model, n_inputs in (("ridge_raw", raw, len(bl.input_columns(machine))), ("ridge_features", direct, len(features))):
        out.train[arm], out.val[arm] = model.predict(train), model.predict(val)
        out.n_fit[arm], out.n_inputs[arm] = model.n_train, n_inputs

    rows = train.loc[keep]
    for arm, method in (("formula_ridge", "ridge"), ("formula_gbm", "gbm")):
        model = sc.CorrectionModel(features=tuple(features), method=method, alpha=alpha)
        model.fit(rows, rows[LABEL], constant * rows[BASE].to_numpy(dtype=np.float64))
        out.train[arm] = model.predict(train, constant * base_train)
        out.val[arm] = model.predict(val, constant * base_val)
        out.n_fit[arm], out.n_inputs[arm] = model.n_train, len(features)

    out.train[CONTEXT_ARM], out.val[CONTEXT_ARM] = constant * base_train, constant * base_val
    out.n_fit[CONTEXT_ARM] = int((keep & np.isfinite(y) & np.isfinite(base_train)).sum())
    out.n_inputs[CONTEXT_ARM] = 0
    return out


def score_stage(stage: int, fit: StageFit, y_train, keep, y_val, masks: Mapping[str, np.ndarray]) -> list[dict]:
    """One row per arm: pooled R2 on each validation subset, training R2 and counts."""
    y_train = np.asarray(y_train, dtype=np.float64)
    y_val = np.asarray(y_val, dtype=np.float64)
    keep = np.asarray(keep, dtype=bool)
    rows = []
    for arm in (*ARMS, CONTEXT_ARM):
        p_val, p_train = np.asarray(fit.val[arm], dtype=np.float64), np.asarray(fit.train[arm], dtype=np.float64)
        row = {"stage": stage, "arm": arm, "n_inputs": fit.n_inputs[arm], "n_fit": fit.n_fit[arm],
               "r2_train": sc.pooled_r2(y_train[keep], p_train[keep]), "constant": fit.constant,
               "n_missing_pred": int((~np.isfinite(p_val[masks["all"]])).sum())}
        for name in SUBSETS:
            m = np.asarray(masks[name], dtype=bool)
            row[f"r2_{name}"] = sc.pooled_r2(y_val[m], p_val[m]) if m.any() else np.nan
            row[f"n_{name}"] = int(m.sum())
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------- shape tables
def shape_path(shape_dir: Path | str, frames: pd.DataFrame, tag: str, stride: int) -> Path:
    """Cache path of a shape table, named as in the scalar study."""
    return Path(shape_dir) / f"shapes_{tag}_{bl.hash_of_ids(frames['shot'].unique())}_s{stride}.parquet"


def stage_tags(stage: int, train_hash: str) -> tuple[str, str]:
    """Cache tags of a stage's training and validation shape tables."""
    return ("true_train", "true_val") if stage == 1 else (f"hybrid{train_hash}_train", f"hybrid{train_hash}_val")


class MapSource:
    """Maps for shape tables that are not cached yet: true maps from the raw files, predicted
    maps from ``hybrid_ridge``. Nothing is loaded until a table must be computed."""

    def __init__(self, args, train_ids: list[str]):
        self.args = args
        self.train_ids = train_ids
        self._table = None
        self._hybrid = None

    def grid(self) -> tuple[np.ndarray, np.ndarray]:
        from fusion_eq.coil_flux import GreenTable
        if self._table is None:
            self._table = GreenTable.load(Path(self.args.green_dir) / "green_d3d.npz")
        return self._table.grid_r, self._table.grid_z

    def true_items(self, frames: pd.DataFrame):
        for shot_id, g in frames.groupby("shot", sort=False):
            maps = ft.read_flux_maps([shot_id], SPLIT_NAME, self.args.data_root)
            idx = g["frame"].to_numpy()
            yield shot_id, idx, maps[idx]

    def hybrid_items(self, frames: pd.DataFrame):
        model = self.hybrid()
        for shot_id, g in frames.groupby("shot", sort=False):
            yield shot_id, g["frame"].to_numpy(), model.predict(g).psirz

    def hybrid(self):
        """``hybrid_ridge`` of the training list, assembled as in notebook 05."""
        if self._hybrid is None:
            from fusion_eq import models
            from fusion_eq.coil_flux import adopted_scales
            from fusion_eq.predictors import model_files
            files = model_files(self.args.cache_dir, self.train_ids)
            missing = [str(p) for k, p in files.items() if k != "ridge_raw" and not p.is_file()]
            if missing:
                raise SystemExit(f"stage 2 shapes need the fitted models of compare_baselines.py --working: {missing}")
            self.grid()
            hybrid = bl.PcaRidge.load(files["hybrid_ridge"])
            signed = bl.PcaRidge.load(files["ridge_signed"])
            with np.load(files["mean_shape"], allow_pickle=False) as f:
                mean_shape, n_mean = f["mean"], int(f["n"])
            model_set = models.ModelSet(table=self._table, scales=adopted_scales(MACHINE, self._table.columns),
                                        basis_psi=signed.basis, basis_plasma=hybrid.basis, ridge_raw=signed,
                                        ridge_signed=signed, hybrid=hybrid, mean_shape=mean_shape, n_mean_frames=n_mean)
            self._hybrid = model_set.baselines()["hybrid_ridge"]
        return self._hybrid


def load_shapes(path: Path, items_fn: Callable[[], Iterable], source: MapSource, workers: int | None) -> tuple[pd.DataFrame, bool]:
    """A shape table from its cache, or computed and cached. Returns ``(table, reused)``."""
    if path.is_file():
        return pd.read_parquet(path), True
    grid_r, grid_z = source.grid()
    return sc.cached_shape_table(path, items_fn, grid_r, grid_z, MACHINE, workers), False


# -------------------------------------------------------------------------------- report
ARM_LABELS = {"ridge_raw": "ridge_raw", "ridge_features": "ridge_features", "formula_ridge": "formula_ridge",
              "formula_gbm": "formula_gbm", CONTEXT_ARM: "base x constant (context)"}


def _fmt(v, digits: int = 4) -> str:
    return "-" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{digits}f}"


def _signed(v) -> str:
    return "-" if v is None or not np.isfinite(v) else f"{v:+.4f}"


def render(results: pd.DataFrame, info: dict) -> str:
    """The Markdown report: summary table, one section per stage, frames and runtime."""
    by = {(int(r.stage), r.arm): r for r in results.itertuples(index=False)}
    stages = sorted({int(s) for s in results["stage"]})
    last = stages[-1]
    corner = info["corner"]
    corner_rule = (f"q_cyl < {corner['q_cyl_below']:.3f} (training 10th percentile) and ts_beta_p < "
                   f"{corner['ts_beta_p_below']:.4f} (training {corner['ts_beta_p_percentile']}th percentile)")
    lines = [f"# betaN heads, {info['lists']} lists of `{info['split_name']}`, stride {info['frame_stride']}", "",
             f"{info['n_train_shots']} training shots ({info['n_train_kept']} of {info['n_train_frames']} frames kept by the "
             f"lighter label filter), {info['n_val_shots']} validation shots ({info['n_val']} frames with both labels "
             f"finite). Features: version {info['feature_version']} (`{info['feature_dir']}`). Ridge alpha {info['alpha']:g}; "
             f"the GBM correction uses the F75 settings. Stage 1 takes `a` and `kappa` from the true maps (label-side bound), "
             f"stage 2 from the `hybrid_ridge` maps. Diagnostic only.", "",
             f"Base: `betaN_base = 100 (2 mu0 p_e / Bt0^2) a Bt0 / abs(Ip)`, with `p_e = {PRESSURE_COLUMN}` (Pa), "
             f"`Bt0 = 2e-7 N_tf abs(I_tf) / R0` (N_tf {info['n_tf']:g}, R0 {info['r0_m']:g} m, machines.py), `a` the stage's "
             f"minor radius (m), Ip in MA; elongation enters the corrections, not the base. Predictions of the formula arms: "
             f"`c * betaN_base * correction`, `c` fitted per stage by least squares on the training frames "
             + ", ".join(f"(stage {s}: c = {by[(s, ARMS[0])].constant:.4f})" for s in stages) + ".", "",
             f"Transfer proxy (corner): {corner_rule}; {corner['n_frames']} validation frames"
             + (f" (relaxed: {corner['n_strict']} frames under the median rule, fewer than {corner['min_frames']})"
                if corner["relaxed"] else "") + ".", "",
             f"| Status | Arm | Inputs | " + " | ".join(f"Stage {s} R2 all" for s in stages)
             + f" | Stage {last} R2 negative Ip | Stage {last} R2 corner | Stage {last} R2 all - ridge_raw |",
             "| --- " * (6 + len(stages)) + "|"]
    for arm in ARMS:
        r = by[(last, arm)]
        ref = by[(last, REFERENCE_ARM)]
        status = "reference" if arm == REFERENCE_ARM else "review"
        lines.append(f"| {status} | {arm} | {r.n_inputs} | " + " | ".join(_fmt(by[(s, arm)].r2_all) for s in stages)
                     + f" | {_fmt(r.r2_negative_ip)} | {_fmt(r.r2_corner)} | {_signed(r.r2_all - ref.r2_all)} |")
    lines += ["", f"ridge_raw is expected near F75's {F75_RIDGE_R2} (ridge on inputs, refit on the training frames); the "
              "lighter filter here removes more training frames than F75's finite-label rule.", ""]

    for s in stages:
        ref = by[(s, REFERENCE_ARM)]
        lines += [f"## Stage {s}", "", f"Base constant c = {ref.constant:.4f}. Frames: all {ref.n_all}, negative Ip "
                  f"{ref.n_negative_ip}, corner {ref.n_corner}.", "",
                  "| Arm | Inputs | R2 all | R2 negative Ip | R2 corner | R2 train | Fit frames | Missing pred | R2 all - ridge_raw |",
                  "| --- " * 9 + "|"]
        for arm in (*ARMS, CONTEXT_ARM):
            r = by[(s, arm)]
            lines.append(f"| {ARM_LABELS[arm]} | {r.n_inputs} | {_fmt(r.r2_all)} | {_fmt(r.r2_negative_ip)} | {_fmt(r.r2_corner)} | "
                         f"{_fmt(r.r2_train)} | {r.n_fit} | {r.n_missing_pred} | {_signed(r.r2_all - ref.r2_all)} |")
        shapes = info["shapes"][str(s)]
        lines += ["", f"Shapes: training {shapes['train']['file']} ({'reused' if shapes['train']['reused'] else 'computed'}, "
                  f"{shapes['train']['n_without_boundary']} frames without a boundary, {shapes['train']['n_without_row']} without a "
                  f"row); validation {shapes['val']['file']} ({'reused' if shapes['val']['reused'] else 'computed'}, "
                  f"{shapes['val']['n_without_boundary']} without a boundary, {shapes['val']['n_without_row']} without a row). "
                  "A missing base counts as the pooled mean in R2.", ""]

    stats = info["training_statistics"]
    lines += ["## Frames and inputs", "", "Lighter label filter on the training frames (each rule alone, then together):", "",
              "| Rule | Frames | Shots |", "| --- | --- | --- |"]
    lines += [f"| {r['rule']} | {r['frames']} | {r['shots']} |" for r in info["filter_counts"]]
    lines += ["", f"Training statistics, every frame of the {info['n_stat_shots']} training shots of the v{info['feature_version']} "
              f"feature table: q_cyl band (1st to 99th percentile) {stats['q_cyl']['q01']:.3f} to {stats['q_cyl']['q99']:.3f}, "
              f"10th percentile {stats['q_cyl']['q10']:.3f} ({stats['q_cyl']['n']} frames); ts_beta_p median "
              f"{stats['ts_beta_p']['q50']:.4f}, 75th percentile {stats['ts_beta_p']['q75']:.4f} ({stats['ts_beta_p']['n']} frames).",
              "", f"Inputs of ridge_features and the corrections: {', '.join(f'`{c}`' for c in HEAD_FEATURES)}. "
              "Every frame-table input passes `frame_table.assert_input_columns` and every feature's reads pass "
              "`features.check_reads`; labels enter only as the regression target and the label filter.", "",
              f"Runtime {info['runtime_s']:.0f} s.", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------------- main
def read_frames(ids: list[str], args, config: fe.FeatureConfig) -> pd.DataFrame:
    """Strided frame-table rows of the shots joined with their feature columns.

    Raises:
        SystemExit: when a frame has no feature row.
    """
    frames = strided(bl.frames_of_shots(ids, FRAME_COLUMNS, args.table_dir, SPLIT_NAME), args.frame_stride)
    columns = list(dict.fromkeys([*FEATURE_COLUMNS, *CORNER_COLUMNS]))
    feats = fe.read_features(SPLIT_NAME, config.version, columns, ids, args.feature_root)
    feats = feats.assign(shot=feats["shot"].astype(str))
    out = frames.merge(feats, on=["shot", "frame"], how="left", validate="one_to_one", indicator=True)
    absent = int((out.pop("_merge") != "both").sum())
    if absent:
        raise SystemExit(f"{absent} frames have no row in the version {config.version} feature table")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--full", action="store_true", help="full training and validation lists instead of the working lists")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--feature-root", type=Path, default=fe.DEFAULT_FEATURE_ROOT)
    parser.add_argument("--table-dir", type=Path, default=ft.DEFAULT_TABLE_ROOT)
    parser.add_argument("--data-root", type=Path, default=default_data_root())
    parser.add_argument("--green-dir", type=Path, default=DEFAULT_GREEN_DIR)
    parser.add_argument("--cache-dir", type=Path, default=bl.DEFAULT_CACHE_DIR, help="fitted models of compare_baselines.py")
    parser.add_argument("--shape-dir", type=Path, default=DEFAULT_SHAPE_DIR, help="shape tables of the scalar study")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--frame-stride", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--min-corner-frames", type=int, default=MIN_CORNER_FRAMES)
    parser.add_argument("--workers", type=int, default=None, help="processes for a shape table that must be computed")
    args = parser.parse_args(argv)

    started = time.time()
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    cfg = machine_config(MACHINE)
    config = fe.FeatureConfig.load(args.config)
    guard_inputs(FRAME_INPUTS, list(dict.fromkeys([*FEATURE_COLUMNS, *CORNER_COLUMNS])), config, MACHINE)
    split = sp.read_split(args.split)
    train_ids = sorted(split["train" if args.full else "working_train"])
    val_ids = sorted(split["validation" if args.full else "working_validation"])
    full_train_ids = sorted(split["train"])
    train_hash = bl.hash_of_ids(train_ids)

    stat_rows = fe.read_features(SPLIT_NAME, config.version, list(CORNER_COLUMNS), full_train_ids, args.feature_root)
    stats = training_statistics(stat_rows["q_cyl"], stat_rows["ts_beta_p"])
    band = (stats["q_cyl"]["q01"], stats["q_cyl"]["q99"])
    log(f"training statistics over {len(stat_rows)} frames of {stat_rows['shot'].nunique()} shots: q_cyl band "
        f"{band[0]:.3f} to {band[1]:.3f}, 10th percentile {stats['q_cyl']['q10']:.3f}; ts_beta_p median {stats['ts_beta_p']['q50']:.4f}")

    train = read_frames(train_ids, args, config)
    val = read_frames(val_ids, args, config)
    keep, filter_counts = training_filter(train)
    scored = validation_mask(val)
    negative = scored & (val["ip_direction"].to_numpy() == -1)
    corner, corner_info = corner_mask(val["q_cyl"], val["ts_beta_p"], stats, scored, args.min_corner_frames)
    masks = {"all": scored, "negative_ip": negative, "corner": corner}
    log(f"{len(train)} training frames ({int(keep.sum())} kept), {len(val)} validation frames ({int(scored.sum())} scored, "
        f"{int(negative.sum())} negative Ip, {int(corner.sum())} in the corner{', relaxed' if corner_info['relaxed'] else ''})")

    source = MapSource(args, train_ids)
    results, shapes_info = [], {}
    for stage in (1, 2):
        tags = stage_tags(stage, train_hash)
        items_fn = source.true_items if stage == 1 else source.hybrid_items
        tables, shapes_info[str(stage)] = {}, {}
        for part, frames, tag in (("train", train, tags[0]), ("val", val, tags[1])):
            path = shape_path(args.shape_dir, frames, tag, args.frame_stride)
            t0 = time.time()
            table, reused = load_shapes(path, lambda f=frames, fn=items_fn: fn(f), source, args.workers)
            tables[part] = head_table(frames, table, band, cfg)
            keys = set(zip(table["shot"].astype(str), table["frame"].astype(int)))
            shapes_info[str(stage)][part] = {
                "file": path.name, "reused": reused,
                "n_without_row": int(sum((s, int(f)) not in keys for s, f in zip(frames["shot"], frames["frame"]))),
                "n_without_boundary": int((~tables[part]["boundary_found"]).sum())}
            log(f"stage {stage} {part}: {'reused' if reused else 'computed'} {path.name} ({time.time() - t0:.0f} s)")
        fit = fit_stage(tables["train"], keep, tables["val"], args.alpha, HEAD_FEATURES, MACHINE)
        rows = score_stage(stage, fit, tables["train"][LABEL], keep, tables["val"][LABEL], masks)
        results += rows
        log(f"stage {stage}: c = {fit.constant:.4f}; " + ", ".join(f"{r['arm']} {r['r2_all']:.4f}" for r in rows))

    table = pd.DataFrame(results)
    args.out.mkdir(parents=True, exist_ok=True)
    info = {"lists": "full" if args.full else "working", "split_name": args.split.name, "frame_stride": args.frame_stride,
            "n_train_shots": len(train_ids), "n_val_shots": len(val_ids), "n_train_frames": int(len(train)),
            "n_train_kept": int(keep.sum()), "n_val": int(scored.sum()), "feature_version": config.version,
            "feature_dir": display_path(fe.features_dir(config.version, args.feature_root)), "alpha": args.alpha,
            "n_tf": float(sc.TF_TURNS[MACHINE]), "r0_m": cfg.major_radius.value, "a0_m": cfg.minor_radius.value,
            "corner": corner_info, "training_statistics": stats, "n_stat_shots": int(stat_rows["shot"].nunique()),
            "filter_counts": filter_counts.to_dict(orient="records"), "shapes": shapes_info, "head_features": list(HEAD_FEATURES),
            "constants": {str(int(r["stage"])): r["constant"] for r in results if r["arm"] == REFERENCE_ARM}}
    runtime_s = time.time() - started
    info["runtime_s"] = runtime_s
    (args.out / "betan_heads.md").write_text(render(table, info))
    # run_info takes the runtime as an argument; a second copy in **info would duplicate the keyword.
    info.pop("runtime_s")
    record = run_info(argv, vars(args), timestamp, runtime_s, **info, results=table.to_dict(orient="records"))
    (args.out / "betan_heads.json").write_text(json.dumps(record, indent=2, default=float) + "\n")
    log(f"wrote {display_path(args.out / 'betan_heads.md')} and betan_heads.json ({runtime_s:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
