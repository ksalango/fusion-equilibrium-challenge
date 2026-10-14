"""Linear check of the features: check X16 of ``docs/FEATURES.md``.

Asks whether a linear model on the features keeps what a linear model takes from the raw
inputs. For each feature set of the configuration, a ridge (``--alpha``, 1) on the scaled
features predicts ``--n-components`` (50) principal components of the plasma target
divided by ``mu0 R0 abs(Ip)``. The map is rebuilt: the decoded target times
``mu0 R0 abs(Ip)``, plus the coil flux under the adopted DIII-D coil model, in the stored
frame. It is scored with the organizers' scorer, with q95 and betaN from the scalar
ridge of ``ridge_signed``. ``hybrid_ridge`` (``fusion_eq.models``) is fitted on the same
training shots and scored on the same frames in the same run as the reference
(0.9186 on the working lists in F61), so the arms differ in the flux map alone.

Training uses every frame of the training shots, as ``compare_baselines.py`` does; frames
with ``abs(Ip)`` below the configuration's ``min_abs_ip_ka`` are left out of the feature
ridges' fit and set to zero in the basis fit. Validation uses every ``--frame-stride``-th
frame of each validation shot. The bases, the coefficients and the scorer references are
cached under ``--cache-dir`` with those of the baseline comparison.

Besides the scores, the run reports the pooled R2 of the predicted coefficients on the
validation frames for each set, for each set with one feature group left out, and for a
ridge on the raw inputs of ``hybrid_ridge`` (Ip-signed currents and the Thomson
summaries) predicting the same coefficients. No scoring is run for those.

Writes ``linear_check.md``, ``linear_check.json`` and ``linear_check.parquet`` to
``<report-root>/v<version>/``.

Follow-up options (diagnostic). Each adds a variant computed in the same run on the same
frames and reported beside version 1; with any of them the run writes
``linear_check_followup.{md,json,parquet}`` only and leaves ``linear_check.*`` unchanged:

- ``--by-phase``: every arm's scores and every coefficient R2 also by phase (rising,
  flat_top, late; ``coil_scales.phase_of`` on ``ip_over_peak``, the rule of X9). A phase
  is scored as if only its frames existed (``baseline.subset_references``), under the flux
  sign chosen over all frames; the coefficient R2 tables add each phase's share of the
  target's variance.
- ``--target masked_current_flux``: the target is the flux of the current inside the
  scorer's boundary, the plasma target minus the vacuum remainder
  (``current_target.vacuum_remainder``, variant e of F79), over ``mu0 R0 abs(Ip)``. Masks
  come from the true maps frame by frame, in worker processes, and are kept in memory.
  ``hybrid_ridge`` is refitted on the same target, basis and frames from its own inputs.
  Scoring adds the true remainder back, so the arms differ only in the predicted
  masked-current flux.
- ``--ip-frac-unscaled``: ``ip_frac`` enters with identity scaling, every other statistic
  unchanged. ``hybrid_ridge`` does not read it; its version-1 score is the reference.

Each variant reports the coefficient R2 with each feature group left out. A follow-up run
only reads the caches of the default run and writes nothing outside the report folder; it
stops when a cache entry is missing, so the default run with the same lists and
parameters comes first. Usage::

    python scripts/feature_linear_check.py              # working lists of splits/split_v1.json
    python scripts/feature_linear_check.py --full       # the full training and validation lists
    python scripts/feature_linear_check.py --by-phase --target masked_current_flux --ip-frac-unscaled
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from fusion_eq import (baseline as bl, current_target as ct, dataset, feature_checks as fc, features as fe,
                       frame_table as ft, models, split as sp, targets)
from fusion_eq.coil_flux import GreenTable, adopted_scales
from fusion_eq.coil_scales import REFERENCE_PHASES
from fusion_eq.dataset import default_data_root, run_info
from fusion_eq.feature_scaler import RobustScaler
from fusion_eq.machines import machine_config

DEFAULT_SPLIT = dataset.REPO_ROOT / "splits" / "split_v1.json"
SPLIT_NAME = "diii_d_train"
MACHINE = "DIII-D"
# Reference score of hybrid_ridge on the working lists (F61); the run scores it again.
F61_REFERENCE_S = 0.9186
TERMS = ("S", "r2_psi", "r2_qb", "one_minus_dlcfs", "consistency")
SHAPE_TERMS = ("r2_R_axis", "r2_Z_axis", "r2_kappa", "r2_tri_top", "r2_tri_bot", "r2_volume", "r2_li")
FOLLOWUP_STEM = "linear_check_followup"
VERSION_1 = "version 1"
MASKED = "masked_current_flux"
IP_FRAC_UNSCALED = "ip_frac_unscaled"
TARGET_CHOICES = ("plasma", MASKED)
RAW_ROW = "raw inputs of hybrid_ridge"


def log(message: str) -> None:
    print(message, flush=True)


@dataclass
class FeatureRidge:
    """Ridge from scaled inputs to the coefficients of a per-Ip target.

    Attributes:
        name: Arm name.
        columns: Feature columns in order (empty for the raw inputs of ``hybrid_ridge``).
        weights, intercept: Ridge solution, ``(p, k)`` and ``(k,)``.
        n_train_frames: Frames of the fit.
    """

    name: str
    columns: list[str]
    weights: np.ndarray
    intercept: np.ndarray
    n_train_frames: int

    def coefficients(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=np.float64) @ self.weights + self.intercept


def fit_feature_ridge(name: str, x: np.ndarray, columns: list[str], coefficients: np.ndarray, keep: np.ndarray,
                      alpha: float) -> FeatureRidge:
    """Ridge on the rows of ``keep``."""
    w, b = bl.ridge_fit(x[keep], np.asarray(coefficients, dtype=np.float64)[keep], alpha)
    return FeatureRidge(name=name, columns=list(columns), weights=w, intercept=b, n_train_frames=int(keep.sum()))


@dataclass
class TargetSpace:
    """Coefficients of one per-Ip target in its basis.

    Attributes:
        name: ``plasma`` (the plasma target) or ``masked_current_flux``.
        basis: Basis of the per-Ip target.
        train: Coefficients of every training frame, in frame-table order.
        val: Coefficients of the strided validation frames, in the order of their rows.
        remainder: True vacuum remainder in Wb/rad per validation shot on its strided
            frames, added back when a map is rebuilt; None for the plasma target.
        info: Record of how the space was built.
    """

    name: str
    basis: bl.FluxBasis
    train: np.ndarray
    val: np.ndarray
    remainder: dict[str, np.ndarray] | None = None
    info: dict = field(default_factory=dict)


def feature_table(shot_ids: list[str], config: fe.FeatureConfig, columns, root: Path) -> pd.DataFrame:
    """Unscaled feature rows of the listed training shots, in file order."""
    return fe.read_features(SPLIT_NAME, config.version, columns, shot_ids, root)


def scaled_rows(table: pd.DataFrame, scaler: RobustScaler) -> pd.DataFrame:
    """``shot``, ``frame`` and every scaled column of the scaler."""
    out = pd.DataFrame(scaler.transform(table, MACHINE), columns=list(scaler.columns))
    out.insert(0, "frame", table["frame"].to_numpy())
    out.insert(0, "shot", table["shot"].astype(str).to_numpy())
    return out


def aligned(frames: pd.DataFrame, scaled: pd.DataFrame) -> pd.DataFrame:
    """The scaled feature rows of ``frames``' keys, in the order of ``frames``.

    Raises:
        ValueError: when a frame has no feature row.
    """
    keys = frames[["shot", "frame"]].astype({"shot": str}).reset_index(drop=True)
    out = keys.merge(scaled, on=["shot", "frame"], how="left", validate="one_to_one")
    if out[list(scaled.columns[2:])].isna().any().any():
        raise ValueError("frame rows without a feature row; rebuild the features (X1)")
    return out


def per_ip_target_fn(table: GreenTable, frames_by_shot: dict[str, pd.DataFrame], scales, r0_m: float, min_abs_ip_ka: float):
    """``(shot_id, stored maps) -> T / (mu0 R0 abs(Ip))``, zero on frames below the Ip threshold."""
    base = models.plasma_target_fn(table, frames_by_shot, scales, MACHINE)

    def fn(shot_id: str, maps: np.ndarray) -> np.ndarray:
        ip = frames_by_shot[shot_id]["ip__mean"].to_numpy(dtype=np.float64)
        return fc.target_per_ip(base(shot_id, maps), ip, r0_m, min_abs_ip_ka)[0]
    return fn


def split_masked(table: GreenTable, rows: pd.DataFrame, maps: np.ndarray, packed: np.ndarray, response: ct.PlasmaResponse,
                 scales) -> tuple[np.ndarray, np.ndarray]:
    """``(inside, remainder)`` in Wb/rad of one shot's maps (``current_target.vacuum_remainder``)
    with its packed scorer masks.

    Raises:
        ValueError: when the rows, maps and masks differ in frames.
    """
    cfg = machine_config(MACHINE)
    masks = ct.unpack_masks(packed, maps.shape[1:])
    if not len(rows) == maps.shape[0] == masks.shape[0]:
        raise ValueError(f"{len(rows)} rows, {maps.shape[0]} maps and {masks.shape[0]} masks")
    coil = targets.coil_flux_of_frames(table, rows, scales, cfg)
    return ct.vacuum_remainder(np.asarray(maps, dtype=np.float64), coil, rows["ip__mean"].to_numpy(dtype=np.float64),
                               masks, response, cfg)


def masked_per_ip_target_fn(table: GreenTable, frames_by_shot: dict[str, pd.DataFrame], masks: dict[str, np.ndarray],
                            response: ct.PlasmaResponse, scales, r0_m: float, min_abs_ip_ka: float):
    """``(shot_id, stored maps) -> masked-current flux / (mu0 R0 abs(Ip))``, zero on frames
    below the Ip threshold."""
    def fn(shot_id: str, maps: np.ndarray) -> np.ndarray:
        rows = frames_by_shot[shot_id]
        inside, _ = split_masked(table, rows, maps, masks[shot_id], response, scales)
        return fc.target_per_ip(inside, rows["ip__mean"].to_numpy(dtype=np.float64), r0_m, min_abs_ip_ka)[0]
    return fn


def masked_space(train_ids: list[str], frames: pd.DataFrame, val_by_shot: dict[str, pd.DataFrame], table: GreenTable,
                 scales, r0_m: float, min_abs_ip_ka: float, params: bl.BaselineParams, args, workers: int) -> TargetSpace:
    """The masked-current target: scorer masks of every training frame and of the strided
    validation frames (worker processes), the basis and the coefficients, all in memory."""
    t0 = time.time()
    response = ct.PlasmaResponse.load_or_build(MACHINE, table.grid_r, table.grid_z, args.response_cache, write=False)
    grid = (table.grid_r, table.grid_z)
    train_items = [(s, str(args.data_root), SPLIT_NAME, 1, *grid, MACHINE) for s in train_ids]
    train_masks = dict(zip(train_ids, bl._map_ordered(ct.shot_masks, train_items, workers)))
    val_ids = list(val_by_shot)
    val_items = [(s, str(args.data_root), SPLIT_NAME, args.frame_stride, *grid, MACHINE) for s in val_ids]
    val_masks = dict(zip(val_ids, bl._map_ordered(ct.shot_masks, val_items, workers)))
    masks_s = time.time() - t0
    log(f"scorer masks: {sum(len(m) for m in train_masks.values())} training and "
        f"{sum(len(m) for m in val_masks.values())} validation frames ({masks_s:.0f} s)")

    frames_by_shot = {s: g for s, g in frames.groupby("shot", sort=False)}
    target_fn = masked_per_ip_target_fn(table, frames_by_shot, train_masks, response, scales, r0_m, min_abs_ip_ka)
    order = bl.basis_shot_order(train_ids, params.seed)
    basis = bl.fit_flux_basis((target_fn(s, m) for s, m in bl.shot_maps(order, args.data_root, SPLIT_NAME)),
                              params.n_components, params.pca_frames)
    train = np.concatenate([basis.encode(target_fn(s, m)) for s, m in bl.shot_maps(sorted(train_ids), args.data_root, SPLIT_NAME)])
    if len(train) != len(frames):
        raise ValueError(f"{len(train)} masked-target rows for {len(frames)} training frames")

    val_parts, remainder = [], {}
    for s, rows in val_by_shot.items():
        maps = bl.load_truth(s, args.data_root, SPLIT_NAME, args.frame_stride)["psi"]
        inside, remainder[s] = split_masked(table, rows, maps, val_masks[s], response, scales)
        val_parts.append(basis.encode(fc.target_per_ip(inside, rows["ip__mean"].to_numpy(dtype=np.float64), r0_m,
                                                       min_abs_ip_ka)[0]))

    def no_boundary(masks: dict[str, np.ndarray]) -> float:
        found = np.concatenate([m.any(axis=1) for m in masks.values()])
        return float((~found).mean()) if found.size else float("nan")

    info = {"n_components": basis.n_components, "explained": float(basis.explained_ratio.sum()), "basis_frames": basis.n_frames,
            "share_without_boundary": {"train": no_boundary(train_masks), "validation": no_boundary(val_masks)},
            "masks_s": masks_s, "total_s": time.time() - t0}
    log(f"masked-current target: {basis.n_components} components, {info['explained']:.4f} of the variance; frames without "
        f"a boundary: training {info['share_without_boundary']['train']:.4f}, validation {info['share_without_boundary']['validation']:.4f}")
    return TargetSpace(MASKED, basis, train, np.concatenate(val_parts), remainder, info)


def ridge_baseline(name: str, ridge: FeatureRidge, basis: bl.FluxBasis, x_of: Callable[[str, pd.DataFrame], np.ndarray],
                   table: GreenTable, scales, r0_m: float, scalars, remainder: dict[str, np.ndarray] | None = None) -> models.Baseline:
    """A ``models.Baseline`` whose map is the coil flux plus the per-Ip target rebuilt from a
    ridge (plus the true remainder when given), with the scalars of ``scalars``."""
    cfg = machine_config(MACHINE)

    def flux(frames: pd.DataFrame) -> np.ndarray:
        shots = frames["shot"].astype(str).unique()
        if len(shots) != 1:
            raise ValueError("the ridge arms predict one shot at a time")
        shot = str(shots[0])
        ip = frames["ip__mean"].to_numpy(dtype=np.float64)
        target = fc.target_from_per_ip(basis.decode(ridge.coefficients(x_of(shot, frames))), ip, r0_m)
        if remainder is not None:
            if remainder[shot].shape != target.shape:
                raise ValueError(f"{shot}: remainder of {remainder[shot].shape[0]} frames for {target.shape[0]}")
            target = target + remainder[shot]
        coil = targets.coil_flux_of_frames(table, frames, scales, cfg)
        return targets.stored_from_target(target, coil, ip, cfg)

    return models.Baseline(name, flux, scalars, scalars_from="ridge_signed")


def strided(frames: pd.DataFrame, stride: int) -> dict[str, pd.DataFrame]:
    return {s: g.sort_values("frame").iloc[::stride] for s, g in frames.groupby("shot", sort=False)}


def phases_of(frames: pd.DataFrame) -> np.ndarray:
    """Phase of each frame-table row by the rule of X9."""
    return fc.frame_phases(frames["ip_over_peak"], frames["frame"], frames["n_frames"])


def coefficient_r2(space: TargetSpace, inputs: dict[str, tuple[np.ndarray, np.ndarray, int]], keep: np.ndarray,
                   val_keep: np.ndarray, alpha: float, phases: tuple[np.ndarray, np.ndarray] | None) -> dict:
    """Pooled R2 of the coefficients predicted by a ridge per input set, training and validation.

    Args:
        inputs: Row name to ``(training inputs, validation inputs of the strided rows, count)``.
        phases: ``(training phases, validation phases)`` to add the R2 by phase, or None.
    """
    out = {}
    for name, (x_train, x_val, n_inputs) in inputs.items():
        ridge = fit_feature_ridge(name, x_train, [], space.train, keep, alpha)
        pred_train, pred_val = ridge.coefficients(x_train[keep]), ridge.coefficients(x_val[val_keep])
        row = {"n_inputs": n_inputs, "train": fc.pooled_r2(space.train[keep], pred_train),
               "validation": fc.pooled_r2(space.val[val_keep], pred_val)}
        if phases is not None:
            row["train_by_phase"] = fc.pooled_r2_by_group(space.train[keep], pred_train, phases[0][keep], REFERENCE_PHASES)
            row["validation_by_phase"] = fc.pooled_r2_by_group(space.val[val_keep], pred_val, phases[1][val_keep], REFERENCE_PHASES)
        out[name] = row
    return out


def render(table: pd.DataFrame, info: dict) -> str:
    lines = [f"# Linear check of the features (X16), version {info['version']}", "",
             f"{info['n_train_shots']} training shots ({info['n_train_frames']} frames), {info['n_val_shots']} validation shots "
             f"scored at frame stride {info['arguments']['frame_stride']} ({'full lists' if info['arguments']['full'] else 'working lists'}). "
             f"Ridge alpha {info['arguments']['alpha']}, {info['n_components']} components of the plasma target over "
             f"mu0 R0 abs(Ip) (R0 {info['r0_m']} m; frames below {info['min_abs_ip_ka']:g} kA left out of the fit). Every arm "
             f"takes q95 and betaN from the scalar ridge of `ridge_signed`. Reference: `hybrid_ridge`, scored in this run "
             f"(F61: {F61_REFERENCE_S} on the working lists).", "",
             "| Status | Arm | Inputs | S - hybrid_ridge |", "| --- | --- | --- | --- |"]
    for arm, s in info["status"].items():
        lines.append(f"| {s['status']} | {arm} | {s['n_inputs']} | {s['delta_s']:+.4f} |")
    for subset, g in table.groupby("subset", sort=False):
        lines += ["", f"## {subset} ({int(g['n_shots'].iloc[0])} shots)", "",
                  "| Arm | S | R2 psi | R2 q95,betaN | 1 - D_LCFS | Consistency | " + " | ".join(t[3:] for t in SHAPE_TERMS) + " |",
                  "| --- " * (6 + len(SHAPE_TERMS)) + "|"]
        for r in g.itertuples(index=False):
            lines.append(f"| {r.model} | " + " | ".join(f"{getattr(r, t):.4f}" for t in (*TERMS, *SHAPE_TERMS)) + " |")
    coef = info["coefficient_r2"]
    lines += ["", "## Coefficient R2 (validation frames, no scoring)", "",
              "Pooled R2 of the predicted coefficients of the per-Ip plasma target, frames above the Ip threshold.", "",
              "| Inputs | Inputs count | Training | Validation |", "| --- | --- | --- | --- |"]
    for name, r in coef.items():
        lines.append(f"| {name} | {r['n_inputs']} | {r['train']:.4f} | {r['validation']:.4f} |")
    lines += ["", f"Runtime: {info['runtime_s']:.0f} s (" + ", ".join(f"{k} {v:.0f}" for k, v in info["timing"].items()) + ")", ""]
    return "\n".join(lines)


def _f(v, digits: int = 4, signed: bool = False) -> str:
    if v is None or not np.isfinite(v):
        return "nan"
    return f"{v:+.{digits}f}" if signed else f"{v:.{digits}f}"


def render_followup(scores: pd.DataFrame, info: dict) -> str:
    a = info["arguments"]
    lines = [f"# Linear check follow-up (X16), version {info['version']}", "",
             f"{info['n_train_shots']} training shots ({info['n_train_frames']} frames), {info['n_val_shots']} validation shots "
             f"scored at frame stride {a['frame_stride']} ({'full lists' if a['full'] else 'working lists'}); ridge alpha "
             f"{a['alpha']}, {info['n_components']} components; frames below {info['min_abs_ip_ka']:g} kA left out of every "
             f"ridge fit. Every arm takes q95 and betaN from the scalar ridge of `ridge_signed`. Variants beside version 1: "
             + (", ".join(f"`{v}`" for v in info["variants"] if v != VERSION_1) or "none") + "; by phase: "
             + ("yes" if a["by_phase"] else "no") + ". Diagnostic only: version 1 and its outputs are unchanged.", ""]
    notes = {VERSION_1: "the plasma target over mu0 R0 abs(Ip), as in `linear_check.md`; `hybrid_ridge` predicts the plasma "
                        "target itself from its 25 inputs.",
             MASKED: "the flux of the current inside the scorer's boundary (plasma target minus the vacuum remainder, variant "
                     "e of F79) over mu0 R0 abs(Ip), its own basis; `hybrid_ridge` refitted on this target, basis and frames "
                     "from its own inputs; the true remainder added back before scoring.",
             IP_FRAC_UNSCALED: "the version-1 target; `ip_frac` with identity scaling. `hybrid_ridge` does not read `ip_frac`; "
                               "its version-1 score is the reference."}
    for v in info["variants"]:
        lines.append(f"- `{v}`: {notes[v]}")
    if MASKED in info.get("spaces", {}):
        m = info["spaces"][MASKED]
        lines.append(f"- Masked-current basis: {m['n_components']} components, {_f(m['explained'])} of the variance over "
                     f"{m['basis_frames']} frames; frames without a boundary (target zero): training "
                     f"{_f(m['share_without_boundary']['train'])}, validation {_f(m['share_without_boundary']['validation'])}.")
    header = ("| Variant | Arm | S | S - v1 same arm | S - reference | R2 psi | R2 q95,betaN | 1 - D_LCFS | Consistency | "
              + " | ".join(t[3:] for t in SHAPE_TERMS) + " |")
    rule = "| --- " * (9 + len(SHAPE_TERMS)) + "|"

    def table_rows(g: pd.DataFrame) -> list[str]:
        out = []
        v1 = g[g["variant"] == VERSION_1].set_index("model")["S"]
        for r in g.itertuples(index=False):
            ref = g[(g["variant"] == r.variant) & (g["model"] == "hybrid_ridge")]
            ref_s = ref["S"].iloc[0] if len(ref) else v1.get("hybrid_ridge", np.nan)
            out.append(f"| {r.variant} | {r.model} | {_f(r.S)} | {_f(r.S - v1.get(r.model, np.nan), signed=True)} | "
                       f"{_f(r.S - ref_s, signed=True)} | " + " | ".join(_f(getattr(r, t)) for t in (*TERMS[1:], *SHAPE_TERMS)) + " |")
        return out

    lines += ["", "S - reference: against `hybrid_ridge` of the same variant (of version 1 for `ip_frac_unscaled`)."]
    for subset, g in scores[scores["phase"] == "all"].groupby("subset", sort=False):
        lines += ["", f"## Scores, {subset} ({int(g['n_shots'].iloc[0])} shots)", "", header, rule, *table_rows(g)]
    if a["by_phase"]:
        counts = info["phase_frames"]
        for phase in REFERENCE_PHASES:
            g = scores[scores["phase"] == phase]
            if g.empty:
                continue
            lines += ["", f"## Scores, phase {phase} ({counts[phase]['frames']} validation frames of {counts[phase]['shots']} "
                      f"shots)", "", header, rule, *table_rows(g)]
    for v, coef in info["coefficient_r2"].items():
        lines += ["", f"## Coefficient R2, {v}", "", "Pooled R2 of the predicted coefficients of the variant's per-Ip target, "
                  "frames above the Ip threshold (validation: the strided frames).", ""]
        by_phase = a["by_phase"]
        cols = ["Inputs", "Count", "Training", "Validation"]
        if by_phase:
            cols += [f"{p} {s}" for s in ("training", "validation") for p in REFERENCE_PHASES]
        lines += ["| " + " | ".join(cols) + " |", "| --- " * len(cols) + "|"]
        for name, r in coef.items():
            cells = [name, str(r["n_inputs"]), _f(r["train"]), _f(r["validation"])]
            if by_phase:
                cells += [_f(r[f"{s}_by_phase"][p]) for s in ("train", "validation") for p in REFERENCE_PHASES]
            lines.append("| " + " | ".join(cells) + " |")
        shares = info.get("variance_shares", {}).get(v)
        if shares:
            lines += ["", "Share of the target's variance (sum of squared deviations from the overall mean) by phase: "
                      + "; ".join(f"{s} " + ", ".join(f"{p} {_f(shares[s][p], 3)}" for p in REFERENCE_PHASES)
                                  for s in ("train", "validation")) + "."]
    lines += ["", f"Runtime: {info['runtime_s']:.0f} s (" + ", ".join(f"{k} {v:.0f}" for k, v in info["timing"].items()) + ")", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, default=fe.DEFAULT_CONFIG)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--full", action="store_true", help="use the full training and validation lists instead of the working ones")
    parser.add_argument("--frame-stride", type=int, default=4)
    parser.add_argument("--n-components", type=int, default=bl.BaselineParams.n_components)
    parser.add_argument("--alpha", type=float, default=bl.BaselineParams.alpha)
    parser.add_argument("--pca-frames", type=int, default=bl.BaselineParams.pca_frames)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--data-root", type=Path, default=default_data_root())
    parser.add_argument("--table-dir", type=Path, default=ft.DEFAULT_TABLE_ROOT)
    parser.add_argument("--feature-root", type=Path, default=fe.DEFAULT_FEATURE_ROOT)
    parser.add_argument("--report-root", type=Path, default=fe.DEFAULT_REPORT_ROOT)
    parser.add_argument("--green-dir", type=Path, default=fe.DEFAULT_GREEN_DIR)
    parser.add_argument("--cache-dir", type=Path, default=bl.DEFAULT_CACHE_DIR)
    parser.add_argument("--by-phase", action="store_true", help="follow-up: scores and coefficient R2 also by phase")
    parser.add_argument("--target", choices=TARGET_CHOICES, default="plasma",
                        help="follow-up: masked_current_flux adds the masked-current target as a variant")
    parser.add_argument("--ip-frac-unscaled", action="store_true", help="follow-up: add the variant with ip_frac unscaled")
    parser.add_argument("--response-cache", type=Path, default=ct.DEFAULT_CACHE_DIR,
                        help="plasma response of the grid (read; built in memory when absent)")
    args = parser.parse_args(argv)
    followup = args.by_phase or args.target == MASKED or args.ip_frac_unscaled
    try:
        return run(args, argv, followup)
    except bl.CacheMissing as exc:
        raise SystemExit(f"{exc}. A follow-up run reads the caches of the default run: run python "
                         f"scripts/feature_linear_check.py with the same lists and parameters first.") from exc


def run(args, argv, followup: bool) -> int:
    started = time.time()
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    timing = {}
    config = fe.FeatureConfig.load(args.config)
    out_dir = Path(args.report_root) / f"v{config.version}"
    out_dir.mkdir(parents=True, exist_ok=True)
    scaler = RobustScaler.load(fe.scaler_path(config.version, args.report_root))
    workers = bl.default_workers() if args.workers is None else args.workers
    params = bl.BaselineParams(n_components=args.n_components, alpha=args.alpha, pca_frames=args.pca_frames, seed=args.seed)
    cfg = machine_config(MACHINE)
    r0_m = cfg.major_radius.value
    read_only = followup

    split = sp.read_split(args.split)
    train_ids = sorted(split["train" if args.full else "working_train"])
    val_ids = sorted(split["validation" if args.full else "working_validation"])
    table = GreenTable.load(args.green_dir / "green_d3d.npz")
    scales = adopted_scales(MACHINE, table.columns)
    log(f"{len(train_ids)} training and {len(val_ids)} validation shots, stride {args.frame_stride}, {workers} workers"
        + ("; follow-up run, caches read only" if followup else ""))

    # The reference models of the comparison, and the scalar ridge every arm uses.
    t0 = time.time()
    columns = bl.input_columns()
    frames = bl.frames_of_shots(train_ids, [*columns, *bl.SCALAR_TARGETS.values(), "n_frames", "ip_over_peak"], args.table_dir)
    frames["shot"] = frames["shot"].astype(str)
    model_set = models.fit_model_set(frames, train_ids, args.data_root, table, params, args.cache_dir, scales, log=log,
                                     read_only=read_only)
    baselines = model_set.baselines()
    timing["reference_fit_s"] = time.time() - t0

    # The per-Ip plasma target: basis and training coefficients.
    t0 = time.time()
    frames_by_shot = {s: g for s, g in frames.groupby("shot", sort=False)}
    target_fn = per_ip_target_fn(table, frames_by_shot, scales, r0_m, config.min_abs_ip_ka)
    tag = f"plasma_per_ip_r{r0_m:g}_min{config.min_abs_ip_ka:g}_{scales.source.split(',')[0] or 'scales'}"
    basis, seconds = bl.cached_basis(train_ids, params, args.data_root, args.cache_dir, split=SPLIT_NAME, target=target_fn, tag=tag,
                                     read_only=read_only)
    coefs = bl.encode_shots(train_ids, basis, args.data_root, args.cache_dir, SPLIT_NAME, target=target_fn, tag=tag,
                            read_only=read_only)
    if len(coefs) != len(frames):
        raise ValueError(f"{len(coefs)} coefficient rows for {len(frames)} training frames")
    keep = fc.ip_usable(frames["ip__mean"], config.min_abs_ip_ka)
    timing["per_ip_basis_s"] = time.time() - t0
    log(f"per-Ip target basis: {basis.n_components} components over {basis.n_frames} frames ({seconds:.0f} s fitting), "
        f"{basis.explained_ratio.sum():.4f} of the variance; {int(keep.sum())} of {len(frames)} training frames above the Ip threshold")

    # Validation rows, the strided frames scored, and the true coefficients of those frames.
    val_frames = bl.frames_of_shots(val_ids, [*columns, "n_frames", "ip_over_peak"], args.table_dir)
    val_frames["shot"] = val_frames["shot"].astype(str)
    val_by_shot = strided(val_frames, args.frame_stride)
    val_rows = np.concatenate([val_frames.index.get_indexer(g.index) for g in val_by_shot.values()])
    val_target_fn = per_ip_target_fn(table, {s: g for s, g in val_frames.groupby("shot", sort=False)}, scales, r0_m,
                                     config.min_abs_ip_ka)
    val_coefs = bl.encode_shots(val_ids, basis, args.data_root, args.cache_dir, SPLIT_NAME, target=val_target_fn, tag=tag,
                                read_only=read_only)
    if len(val_coefs) != len(val_frames):
        raise ValueError(f"{len(val_coefs)} coefficient rows for {len(val_frames)} validation frames")
    val_keep = fc.ip_usable(val_frames["ip__mean"].to_numpy()[val_rows], config.min_abs_ip_ka)
    spaces = {"plasma": TargetSpace("plasma", basis, coefs, val_coefs[val_rows])}
    if args.target == MASKED:
        t0 = time.time()
        spaces[MASKED] = masked_space(train_ids, frames, val_by_shot, table, scales, r0_m, config.min_abs_ip_ka, params, args, workers)
        timing["masked_target_s"] = time.time() - t0

    # Scaled inputs: the features under each scaler, and the raw inputs of hybrid_ridge.
    scalers = {VERSION_1: scaler}
    if args.ip_frac_unscaled:
        scalers[IP_FRAC_UNSCALED] = scaler.with_identity(["ip_frac"])
    train_table = feature_table(train_ids, config, scaler.columns, args.feature_root)
    val_table = feature_table(val_ids, config, scaler.columns, args.feature_root)
    train_x, val_x, x_by_shot = {}, {}, {}
    for name, s in scalers.items():
        train_x[name] = aligned(frames, scaled_rows(train_table, s))
        val_x[name] = aligned(val_frames, scaled_rows(val_table, s))
        x_by_shot[name] = {shot: g.set_index("frame") for shot, g in val_x[name].groupby("shot", sort=False)}
    raw_scaler = model_set.ridge_signed.scaler
    raw_train, _ = raw_scaler.transform(targets.signed_frames(frames, cfg))
    raw_val, _ = raw_scaler.transform(targets.signed_frames(val_frames, cfg))

    # Variants: (target space, scaler, whether hybrid_ridge is refitted on the target).
    variants = {VERSION_1: ("plasma", VERSION_1)}
    if args.target == MASKED:
        variants[MASKED] = (MASKED, VERSION_1)
    if args.ip_frac_unscaled:
        variants[IP_FRAC_UNSCALED] = ("plasma", IP_FRAC_UNSCALED)

    # Coefficient R2 on the strided validation frames above the Ip threshold.
    t0 = time.time()
    phases = (phases_of(frames), phases_of(val_frames)[val_rows]) if args.by_phase else None
    specs = config.column_specs()
    coefficient_r2_all, variance_shares = {}, {}
    for variant, (space_name, scaler_name) in variants.items():
        space, tx, vx = spaces[space_name], train_x[scaler_name], val_x[scaler_name].iloc[val_rows]
        inputs = {}
        for set_name in config.sets:
            set_columns = config.set_columns(set_name)
            inputs[f"features_{set_name}"] = (tx[set_columns].to_numpy(), vx[set_columns].to_numpy(), len(set_columns))
            for group in dict.fromkeys(specs[c].group for c in set_columns):
                kept = [c for c in set_columns if specs[c].group != group]
                inputs[f"features_{set_name} without {group}"] = (tx[kept].to_numpy(), vx[kept].to_numpy(), len(kept))
        inputs[RAW_ROW] = (raw_train, raw_val[val_rows], len(columns))
        coefficient_r2_all[variant] = coefficient_r2(space, inputs, keep, val_keep, args.alpha, phases)
        if phases is not None:
            variance_shares[variant] = {"train": fc.variance_shares(space.train[keep], phases[0][keep], REFERENCE_PHASES),
                                        "validation": fc.variance_shares(space.val[val_keep], phases[1][val_keep], REFERENCE_PHASES)}
        for name, r in coefficient_r2_all[variant].items():
            log(f"coefficient R2 [{variant}] {name}: training {r['train']:.4f}, validation {r['validation']:.4f}")
    timing["coefficient_r2_s"] = time.time() - t0

    # Arms of each variant.
    scalars = baselines["ridge_signed"].scalars
    arms: dict[str, dict[str, models.Baseline]] = {}
    ridges: dict[str, dict[str, FeatureRidge]] = {}
    for variant, (space_name, scaler_name) in variants.items():
        space = spaces[space_name]
        arms[variant], ridges[variant] = {}, {}
        if variant == VERSION_1:
            arms[variant]["hybrid_ridge"] = baselines["hybrid_ridge"]
        elif space_name == MASKED:
            hybrid = fit_feature_ridge("hybrid_ridge", raw_train, [], space.train, keep, args.alpha)

            def raw_x(shot: str, rows: pd.DataFrame) -> np.ndarray:
                return raw_scaler.transform(targets.signed_frames(rows, cfg))[0]
            arms[variant]["hybrid_ridge"] = ridge_baseline("hybrid_ridge", hybrid, space.basis, raw_x, table, scales, r0_m,
                                                           scalars, space.remainder)
        for set_name in config.sets:
            set_columns = config.set_columns(set_name)
            name = f"features_{set_name}"
            ridge = fit_feature_ridge(name, train_x[scaler_name][set_columns].to_numpy(), set_columns, space.train, keep, args.alpha)

            def feature_x(shot: str, rows: pd.DataFrame, by_shot=x_by_shot[scaler_name], cols=set_columns) -> np.ndarray:
                return by_shot[shot].loc[rows["frame"].to_numpy(), cols].to_numpy(dtype=np.float64)
            ridges[variant][name] = ridge
            arms[variant][name] = ridge_baseline(name, ridge, space.basis, feature_x, table, scales, r0_m, scalars, space.remainder)

    # Scoring: every arm on the same frames, and by phase on subsets of the same references.
    t0 = time.time()
    references = bl.build_references(val_ids, args.data_root, SPLIT_NAME, args.frame_stride, workers=workers,
                                     cache_dir=args.cache_dir / "references", read_only=read_only)
    timing["references_s"] = time.time() - t0
    index = pd.read_parquet(Path(args.table_dir) / "shot_index.parquet")
    direction = index.loc[index["split"] == SPLIT_NAME].assign(shot=lambda t: t["shot"].astype(str)) \
        .set_index("shot")["ip_direction"].reindex(val_ids).to_numpy()
    subsets = {"all": np.ones(len(val_ids), dtype=bool), "ip_positive": direction == 1, "ip_negative": direction == -1}
    subsets = {k: v for k, v in subsets.items() if v.any()}
    phase_refs, phase_frames = {}, {}
    if args.by_phase:
        t0 = time.time()
        for phase in REFERENCE_PHASES:
            chosen = {s: np.flatnonzero(phases_of(g) == phase) for s, g in val_by_shot.items()}
            phase_refs[phase] = bl.subset_references(references, chosen, args.data_root, SPLIT_NAME)
            phase_frames[phase] = {"frames": int(sum(len(v) for v in chosen.values())), "shots": len(phase_refs[phase].shot_ids)}
        timing["phase_references_s"] = time.time() - t0
    rows = []
    for variant, variant_arms in arms.items():
        for name, model in variant_arms.items():
            t0 = time.time()
            predict = (lambda s, m=model: m.predict(val_by_shot[s]))
            result = bl.score_local(predict, val_ids, args.data_root, SPLIT_NAME, args.frame_stride, references, workers)
            for subset, mask in subsets.items():
                rows.append({"variant": variant, "phase": "all", "model": name, "subset": subset, **bl.score_terms(result.finalize(mask))})
            overall = rows[-len(subsets)]["S"]
            for phase, ref in phase_refs.items():
                if not ref.shot_ids:
                    continue
                part = bl.score_local(predict, ref.shot_ids, args.data_root, SPLIT_NAME, args.frame_stride, ref, workers,
                                      psi_sign=result.psi_sign)
                rows.append({"variant": variant, "phase": phase, "model": name, "subset": "all", **bl.score_terms(part.finalize())})
            key = name if variant == VERSION_1 else f"{variant}: {name}"
            timing[f"score_{key}_s"] = time.time() - t0
            log(f"{key}: S {overall:.4f} ({timing[f'score_{key}_s']:.0f} s)")
    scores = pd.DataFrame(rows)

    common = dict(version=config.version, config_hash=config.hash(), workers=workers, n_train_shots=len(train_ids),
                  n_train_frames=int(len(frames)), n_val_shots=len(val_ids),
                  n_val_frames=int(sum(len(g) for g in val_by_shot.values())), n_components=basis.n_components,
                  per_ip_basis_explained=float(basis.explained_ratio.sum()), r0_m=r0_m, min_abs_ip_ka=config.min_abs_ip_ka,
                  coil_model=scales.source, references_from_cache=references.from_cache, f61_reference_s=F61_REFERENCE_S)
    if not followup:
        v1 = scores[scores["variant"] == VERSION_1].drop(columns=["variant", "phase"]).reset_index(drop=True)
        v1.to_parquet(out_dir / "linear_check.parquet", index=False)
        overall = v1.loc[v1["subset"] == "all"].set_index("model")["S"]
        status = {name: {"status": "pass" if overall[name] >= overall["hybrid_ridge"] else "review",
                         "delta_s": float(overall[name] - overall["hybrid_ridge"]), "n_inputs": len(r.columns)}
                  for name, r in ridges[VERSION_1].items()}
        status["hybrid_ridge"] = {"status": "reference", "delta_s": 0.0, "n_inputs": len(columns)}
        info = run_info(argv, vars(args), timestamp, time.time() - started, **common, status=status,
                        coefficient_r2=coefficient_r2_all[VERSION_1], timing=timing, scores=v1.to_dict(orient="records"))
        (out_dir / "linear_check.json").write_text(json.dumps(info, indent=2, default=fc.json_default) + "\n")
        report = render(v1, info)
        (out_dir / "linear_check.md").write_text(report)
        print(report)
        return 0

    scores.to_parquet(out_dir / f"{FOLLOWUP_STEM}.parquet", index=False)
    info = run_info(argv, vars(args), timestamp, time.time() - started, **common, variants=list(variants),
                    spaces={k: v.info for k, v in spaces.items() if v.info}, phase_frames=phase_frames,
                    coefficient_r2=coefficient_r2_all, variance_shares=variance_shares, timing=timing,
                    scores=scores.to_dict(orient="records"))
    (out_dir / f"{FOLLOWUP_STEM}.json").write_text(json.dumps(info, indent=2, default=fc.json_default) + "\n")
    report = render_followup(scores, info)
    (out_dir / f"{FOLLOWUP_STEM}.md").write_text(report)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
