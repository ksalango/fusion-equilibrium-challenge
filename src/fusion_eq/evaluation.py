"""Evaluation of a training run (``docs/PIPELINE.md``, Part 6): the sections of
``reports/runs/<run_id>/eval.md``.

Validation maps are those of the prediction chain (``fusion_eq.prediction``): mirror averaging
as configured, q95 from the map and betaN from the DIII-D head (the raw-input ridge, fitted on
the run's training list), on every ``scoring.stride``-th frame of every shot of the run's
validation list, scored with the organizers' scorer.

Sections:

- Score terms (S, R2 psi pooled and per-shot mean, R2 of q95 and betaN, 1 - D_LCFS, consistency
  and its seven values) for all shots, the shots without a near-repeat, the negative-Ip shots
  and the MAST-like corner (F99's proxy), and by phase (rising, flat-top, late) for all shots
  and the no-near-repeat shots. Shot slices are finalized from one scoring pass; frame slices
  (corner, phases) are scored as if only their frames existed, under the flux sign of the pass.
- The score identity ``S = 0.55 R2 psi + 0.15 scalars + 0.10 (1 - D_LCFS) + 0.20 consistency``
  (negative terms clipped) beside every row, with the MAST floor of F100 as the comparison line.
- Worst 20 validation shots by R2 psi, with D_LCFS.
- Overfit panel: train and validation loss at the best and the stopping epoch, the all-shots
  against no-near-repeat gap of the trained model beside the warm start's (F58), the seed spread
  of S over the given runs; every S through the chain (each seed run's best checkpoint scored
  here), so runs trained without the scorer are covered.
- Clip rates per feature and machine (validation, DIII-D test, MAST test).
- The warm-start score beside the trained score (training-time scorer, ``run.json``) and the
  chain score with mirror averaging on and off.
- Grouped permutation importance on validation per score term (each feature group's columns
  permuted together across frames, seeded).
- The label-free MAST sensitivity check: each group permuted on MAST test inputs, with the
  change in the predicted maps and in the boundary and axis the scorer's functions derive from
  them. No MAST label is read.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from fusion_eq import (baseline as bl, dataset, features as fe, frame_table as ft, loader as ld, model_scoring as ms,
                       prediction as pr, scalar_heads as sh, split as sp, starter_kit, target_build as tb)
from fusion_eq.coil_scales import phase_of
from fusion_eq.feature_scaler import RobustScaler
from fusion_eq.machines import machine_config
from fusion_eq.model_config import PHASES, ModelConfig

SPLIT_NAME = tb.LABELLED_SPLIT
WEIGHTS = {"r2_psi": 0.55, "r2_qb": 0.15, "one_minus_dlcfs": 0.10, "consistency": 0.20}
# F100: mean_current_v0 on MAST (Codabench 988040).
MAST_FLOOR = {"S": 0.4833, "r2_psi": 0.7447, "d_lcfs_m": 0.2626}
CONS_NAMES = ("R_axis", "Z_axis", "kappa", "tri_top", "tri_bot", "volume", "li")
TERM_KEYS = ("S", "r2_psi", "r2_psi_per_shot_mean", "r2_q95", "r2_betaN", "r2_qb", "one_minus_dlcfs", "consistency",
             *[f"r2_{c}" for c in CONS_NAMES])
SHOT_SLICES = ("all", "no_near_repeat", "ip_negative")
DERIVED = ("R_axis", "Z_axis", "kappa", "R_geo", "a")


# ---------------------------------------------------------------------------- terms
def terms_of(result: Mapping) -> dict:
    """Score terms of a finalized scorer result, with the per-shot mean R2 psi and the
    boundary failure share."""
    out = {k: float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v for k, v in bl.score_terms(dict(result)).items()}
    if np.isfinite(out.get("S", np.nan)):
        out["r2_psi_per_shot_mean"] = float(result.get("r2_psi_per_shot_mean", np.nan))
        out["lcfs_fail_frac"] = float(result.get("lcfs_fail_frac", np.nan))
    return out


def score_identity(terms: Mapping) -> dict:
    """``0.55 max(0, R2 psi) + 0.15 max(0, R2 scalars) + 0.10 (1 - D_LCFS) + 0.20 consistency``
    from the terms, beside the reported S (consistency is already the mean of clipped values)."""
    if not np.isfinite(terms.get("S", np.nan)):
        return {"S": float("nan"), "reconstructed": float("nan"), "difference": float("nan")}
    parts = {"r2_psi": WEIGHTS["r2_psi"] * max(0.0, terms["r2_psi"]), "r2_qb": WEIGHTS["r2_qb"] * max(0.0, terms["r2_qb"]),
             "one_minus_dlcfs": WEIGHTS["one_minus_dlcfs"] * terms["one_minus_dlcfs"],
             "consistency": WEIGHTS["consistency"] * terms["consistency"]}
    total = float(sum(parts.values()))
    return {"S": float(terms["S"]), "reconstructed": total, "difference": float(terms["S"]) - total, "parts": parts}


def mast_floor_line() -> str:
    return (f"MAST floor (F100, mean_current_v0 on Codabench): S {MAST_FLOOR['S']:.4f}, R2 psi {MAST_FLOOR['r2_psi']:.4f}, "
            f"D_LCFS {MAST_FLOOR['d_lcfs_m']:.3f} m.")


# ---------------------------------------------------------------------------- data
@dataclass
class EvalSet:
    """The scored validation frames.

    Attributes:
        shot_ids: Validation shots in order.
        rows: ``merge_inputs`` rows the model reads: every stride-th frame, or, for a sequence
            model, every frame with the stride-th ones flagged ``scored``.
        scored: The scored rows (every stride-th frame), shots in order; slices index them.
        stride: Frame stride.
        references: Scorer references of the shots at the stride.
        shot_slices: Slice name to a bool per shot.
        frame_masks: Frame-slice name to a bool per row.
        frame_refs: Frame-slice name to its subset references (built on first use).
        corner: Corner thresholds and whether they were relaxed.
    """

    shot_ids: list[str]
    rows: pd.DataFrame
    scored: pd.DataFrame
    stride: int
    references: bl.References
    shot_slices: dict[str, np.ndarray]
    frame_masks: dict[str, np.ndarray]
    corner: dict = field(default_factory=dict)
    frame_refs: dict[str, bl.References] = field(default_factory=dict)

    def frames_of(self, mask: np.ndarray) -> dict[str, np.ndarray]:
        """Shot id to the indices of the masked frames among its strided frames."""
        out = {}
        for shot, g in self.scored.groupby("shot", sort=False):
            idx = np.flatnonzero(np.asarray(mask)[g.index.to_numpy()])
            if idx.size:
                out[str(shot)] = idx
        return out

    def subset_references(self, name: str, data_root: Path) -> bl.References | None:
        if name not in self.frame_refs:
            frames = self.frames_of(self.frame_masks[name])
            self.frame_refs[name] = bl.subset_references(self.references, frames, data_root, SPLIT_NAME) if frames else None
        return self.frame_refs[name]


def build_eval_set(source: pr.TableSource, config: ModelConfig, data_paths: ld.DataPaths, val_list: str, train_list: str,
                   data_root: Path, references_dir: Path, workers: int | None, log: Callable[[str], None] = print,
                   mark: bool = False) -> EvalSet:
    """The validation rows, slices and references of a run (every frame kept and the scored ones
    flagged with ``mark``, for sequence models)."""
    ld.require_readable(val_list)
    split = sp.read_split(data_paths.split)
    shot_ids = sorted(split[val_list])
    stride = int(config.get("scoring.stride"))
    model_rows = pr.read_rows(source, SPLIT_NAME, shot_ids, stride, mark=mark)
    rows = pr.scored(model_rows)
    extra = ft.read_frame_table(SPLIT_NAME, ["shot", "frame", "ip_over_peak", "n_frames", "ip_direction"], shot_ids,
                                root=source.table_root)
    extra["shot"] = extra["shot"].astype(str)
    rows = rows.merge(extra.drop(columns=[c for c in ("ip_over_peak", "n_frames", "ip_direction") if c in rows.columns]),
                      on=["shot", "frame"], how="left", validate="one_to_one")
    if "ts_beta_p" not in rows.columns:
        b = fe.read_features(SPLIT_NAME, int(config.get("features.version")), ["ts_beta_p"], shot_ids, data_paths.feature_root)
        rows = rows.merge(b.assign(shot=b["shot"].astype(str)), on=["shot", "frame"], how="left", validate="one_to_one")
    shot_ids = [s for s in shot_ids if s in set(rows["shot"])]
    phase = np.array([phase_of(float(o), int(f), int(n)) for o, f, n in
                      zip(rows["ip_over_peak"], rows["frame"], rows["n_frames"])], dtype=object)
    thresholds = ms.corner_thresholds(sorted(split[train_list]), config, data_paths)
    corner, relaxed = ms.corner_mask(rows["q_cyl"], rows["ts_beta_p"], thresholds, int(config.get("scoring.corner_min_frames")))
    near = ld.near_repeat_flags(split, np.array(shot_ids))
    direction = rows.groupby("shot", sort=False)["ip_direction"].first().reindex(shot_ids).to_numpy()
    nnr_frame = ~ld.near_repeat_flags(split, rows["shot"].to_numpy())
    frame_masks = {"corner": corner}
    for p in PHASES:
        frame_masks[f"phase_{p}"] = phase == p
        frame_masks[f"no_near_repeat_phase_{p}"] = (phase == p) & nnr_frame
    references = bl.build_references(shot_ids, data_root, SPLIT_NAME, stride, workers=workers, cache_dir=references_dir)
    log(f"evaluation set: {len(shot_ids)} {val_list} shots at stride {stride}, {len(rows)} frames, {int(corner.sum())} corner frames"
        f"{' (relaxed)' if relaxed else ''}")
    return EvalSet(shot_ids=shot_ids, rows=model_rows, scored=rows, stride=stride, references=references,
                   shot_slices={"all": np.ones(len(shot_ids), dtype=bool), "no_near_repeat": ~near, "ip_negative": direction == -1},
                   frame_masks=frame_masks, corner={"thresholds": thresholds, "relaxed": relaxed, "n_frames": int(corner.sum())})


# ------------------------------------------------------------------------- scoring
def predictions(chain: pr.PredictionChain, rows: pd.DataFrame, machine: str = "DIII-D", with_scalars: bool = True) -> dict:
    """Shot id to its ``Prediction`` (float32 maps), one shot at a time."""
    return {s: p.as_prediction() for s, p in pr.predict_shots(chain, rows, machine, with_scalars).items()}


def score(preds: Mapping[str, bl.Prediction], es: EvalSet, data_root: Path, workers: int | None,
          frame_slices: Sequence[str] = ()) -> tuple[dict, bl.LocalScore]:
    """Terms of every shot slice and of the requested frame slices."""
    result = bl.score_local(preds, es.shot_ids, data_root, SPLIT_NAME, es.stride, es.references, workers)
    out = {name: terms_of(result.finalize(mask)) if mask.any() else {"S": float("nan"), "n_shots": 0}
           for name, mask in es.shot_slices.items()}
    for name in frame_slices:
        refs = es.subset_references(name, data_root)
        if refs is None:
            out[name] = {"S": float("nan"), "n_shots": 0}
            continue
        part = bl.score_local(preds, refs.shot_ids, data_root, SPLIT_NAME, es.stride, refs, workers, psi_sign=result.psi_sign)
        out[name] = terms_of(part.finalize())
    return out, result


def worst_shots(result: bl.LocalScore, n: int = 20) -> list[dict]:
    """The ``n`` shots with the lowest R2 psi (under the pass's flux sign), with D_LCFS."""
    rows = [{"shot": s, "r2_psi": float(p["shot_r2_psi"][result.psi_sign]),
             "d_lcfs": float("nan") if p["shot_dlcfs"] is None else float(p["shot_dlcfs"]),
             "lcfs_fails": int(p["lcfs_fails"]), "lcfs_frames": int(p["lcfs_frames"])}
            for s, p in zip(result.shot_ids, result.parts)]
    return sorted(rows, key=lambda r: (r["r2_psi"], r["shot"]))[:n]


# ------------------------------------------------------------------------- permutation
def feature_groups(fconfig: fe.FeatureConfig, columns: Sequence[str]) -> dict[str, list[str]]:
    """Model columns by feature group, groups in column order."""
    specs = fconfig.column_specs()
    out: dict[str, list[str]] = {}
    for c in columns:
        out.setdefault(specs[c].group, []).append(c)
    return out


def permuted(rows: pd.DataFrame, columns: Sequence[str], seed: int) -> pd.DataFrame:
    """A copy of ``rows`` with the given columns permuted together across all rows."""
    out = rows.copy()
    idx = np.random.default_rng(seed).permutation(len(rows))
    out[list(columns)] = rows[list(columns)].to_numpy()[idx]
    return out


def permutation_importance(chain: pr.PredictionChain, es: EvalSet, groups: Mapping[str, Sequence[str]], base: Mapping,
                           data_root: Path, workers: int | None, seed: int = 0, log: Callable[[str], None] = print) -> dict:
    """Per group: the terms on all validation shots with the group permuted, and the drop of
    each term from ``base``."""
    out = {}
    for k, (group, columns) in enumerate(groups.items()):
        terms, _ = score(predictions(chain, permuted(es.rows, columns, seed + k)), es, data_root, workers)
        drop = {t: base[t] - terms["all"][t] for t in TERM_KEYS if t in base and t in terms["all"]}
        out[group] = {"columns": list(columns), "terms": terms["all"], "drop": drop}
        log(f"  permutation {group}: S {terms['all']['S']:.4f} (drop {drop.get('S', float('nan')):+.4f})")
    return out


# --------------------------------------------------------------------- MAST sensitivity
def derived_of_maps(machine: str, psi: np.ndarray) -> np.ndarray:
    """Per map ``boundary_found, R_axis, Z_axis, kappa, R_geo, a`` with the scorer's functions
    (NaN without a boundary). Runs in a worker process."""
    scorer = starter_kit.scoring()
    cfg = machine_config(machine)
    env = starter_kit.envelope_mask(cfg)
    out = np.full((len(psi), 1 + len(DERIVED)), np.nan)
    for k, p in enumerate(np.asarray(psi, dtype=np.float64)):
        contour = scorer.lcfs.extract_lcfs(p, env.grid_R, env.grid_Z, cfg.name, env.mask_coarse, env.mask_f)
        out[k, 0] = float(contour is not None and len(contour) >= 8)
        if not out[k, 0]:
            continue
        vals = scorer.derive.derive_frame(p, env.grid_R, env.grid_Z, cfg.name, env.mask_coarse, env.mask_f, contour=contour,
                                          with_li=False)
        out[k, 1:4] = [vals["R_axis"], vals["Z_axis"], vals["kappa"]]
        out[k, 4] = scorer.lcfs.major_radius(contour)
        out[k, 5] = 0.5 * (float(contour[:, 0].max()) - float(contour[:, 0].min()))
    return out


def derived_table(maps: Mapping[str, np.ndarray], machine: str, workers: int | None) -> np.ndarray:
    parts = list(bl._map_ordered(derived_of_maps, ((machine, m) for m in maps.values()), workers))
    return np.concatenate(parts) if parts else np.zeros((0, 1 + len(DERIVED)))


def mast_sensitivity(chain: pr.PredictionChain, rows: pd.DataFrame, groups: Mapping[str, Sequence[str]], workers: int | None,
                     seed: int = 0, log: Callable[[str], None] = print) -> dict:
    """Each group permuted on MAST test inputs: change of the maps and of the derived boundary and
    axis against the unpermuted prediction. Reads inputs only."""
    pr.assert_mast_sign()
    n_frames = len(pr.scored(rows))
    base_maps = {s: p.psi for s, p in pr.predict_shots(chain, rows, "MAST", with_scalars=False).items()}
    base = derived_table(base_maps, "MAST", workers)
    stacked = np.concatenate(list(base_maps.values()))
    rms = float(np.sqrt(np.mean(stacked ** 2)))
    out = {"n_shots": int(rows["shot"].nunique()), "n_frames": int(n_frames), "map_rms": rms,
           "boundary_share": float(np.mean(base[:, 0])) if len(base) else float("nan"),
           "base_medians": {k: float(np.nanmedian(base[:, j + 1])) if np.isfinite(base[:, j + 1]).any() else float("nan")
                            for j, k in enumerate(DERIVED)}, "groups": {}}
    for k, (group, columns) in enumerate(groups.items()):
        maps = {s: p.psi for s, p in pr.predict_shots(chain, permuted(rows, columns, seed + k), "MAST", with_scalars=False).items()}
        der = derived_table(maps, "MAST", workers)
        change = np.concatenate(list(maps.values())) - stacked
        both = (base[:, 0] > 0) & (der[:, 0] > 0)
        entry = {"columns": list(columns), "map_rms_change": float(np.sqrt(np.mean(change ** 2))),
                 "map_rms_change_relative": float(np.sqrt(np.mean(change ** 2)) / rms) if rms > 0 else float("nan"),
                 "boundary_share": float(np.mean(der[:, 0])) if len(der) else float("nan"), "n_both": int(both.sum())}
        for j, name in enumerate(DERIVED):
            d = np.abs(der[both, j + 1] - base[both, j + 1])
            entry[f"median_abs_change_{name}"] = float(np.nanmedian(d)) if np.isfinite(d).any() else float("nan")
        out["groups"][group] = entry
        log(f"  MAST sensitivity {group}: map change {entry['map_rms_change_relative']:.3f} of the map RMS, "
            f"boundary on {entry['boundary_share']:.3f} of frames")
    return out


# ---------------------------------------------------------------------- clip rates
def clip_rate_table(columns: Sequence[str], scaler: RobustScaler, bound: float, feature_root: Path, version: int,
                    val_ids: Sequence[str]) -> dict:
    """Share of scaled values beyond the bound per column, on validation (DIII-D), the DIII-D
    test fold and the MAST test fold (the folds whose feature tables exist)."""
    sources = {"DIII-D validation": ("diii_d_train", list(val_ids)), "DIII-D test": ("diii_d_public_test", None),
               "MAST test": ("mast_public_test", None)}
    out = {}
    for label, (split, shots) in sources.items():
        if not fe.features_path(split, version, feature_root).is_file():
            continue
        table = fe.read_features(split, version, list(columns), shots, feature_root)
        x = scaler.transform(table, dataset.SPLIT_MACHINE[split], list(columns))
        rates = ld.clip_rates(x, bound)
        out[label] = {"n_frames": int(len(table)), "rates": {c: float(r) for c, r in zip(columns, rates)}}
    return out


# ------------------------------------------------------------------------ run record
def read_metrics(run_dir: Path) -> list[dict]:
    path = Path(run_dir) / "metrics.csv"
    if not path.is_file():
        return []
    with path.open() as f:
        return [{k: (v if k == "best" else float(v)) for k, v in row.items()} for row in csv.DictReader(f)]


def overfit_panel(record: Mapping, metrics: Sequence[Mapping], seeds: Sequence[Mapping], trained: Mapping | None = None,
                  warm: Mapping | None = None) -> dict:
    """Losses at the best and stopping epochs, the all against no-near-repeat gaps of the trained
    model and of the warm start, and the seed spread of S.

    Args:
        record, metrics: The run record and its per-epoch metrics.
        seeds: One entry per seed run, ``{run_id, seed, S, S_nnr}`` (chain scores).
        trained, warm: Chain score terms per slice of the best and warm-start checkpoints; the
            training-time scores of the run record when None.
    """
    es = record.get("early_stopping", {})

    def at(epoch):
        row = next((m for m in metrics if int(m["epoch"]) == int(epoch)), None)
        if row is None:
            return None
        return {"epoch": int(epoch), "train": row.get("train_total"), "validation": row.get("val_total"),
                "validation_no_near_repeat": row.get("val_nnr_total")}

    def gap(score):
        if not score:
            return None
        return {"all": score["all"]["S"], "no_near_repeat": score["no_near_repeat"]["S"],
                "gap": score["all"]["S"] - score["no_near_repeat"]["S"]}

    seeds = list(seeds)
    s = np.array([x["S"] for x in seeds], dtype=np.float64)
    return {"best_epoch": at(es.get("best_epoch", -1)), "stopped_epoch": at(es.get("stopped_epoch", -1)),
            "stopped_early": es.get("stopped_early"),
            "trained_gap": gap(trained if trained is not None else (record.get("best") or {}).get("score")),
            "warm_start_gap": gap(warm if warm is not None else (record.get("warm_start") or {}).get("score")), "seeds": seeds,
            "seed_spread": {"n": int(s.size), "mean": float(s.mean()) if s.size else float("nan"),
                            "std": float(s.std(ddof=1)) if s.size > 1 else float("nan"),
                            "min": float(s.min()) if s.size else float("nan"), "max": float(s.max()) if s.size else float("nan")}}


def warm_chain_comparison(warm: Mapping | None, trained: Mapping) -> dict:
    """The chain scores of the warm-start checkpoint beside the trained model's, per slice, and
    check Z4's comparison on them."""
    if not warm:
        return {"available": False}
    out = {"available": True}
    for name in ("all", "no_near_repeat", "ip_negative", "corner"):
        w, t = warm.get(name, {}).get("S", float("nan")), trained.get(name, {}).get("S", float("nan"))
        out[name] = {"warm_start": w, "trained": t, "difference": t - w}
    out["z4_trained_at_or_above"] = bool(out["all"]["difference"] >= 0 and out["no_near_repeat"]["difference"] >= 0)
    return out


def warm_against_trained(record: Mapping) -> dict:
    """The training-time scores (fixed ridge scalars) of the warm start and the best checkpoint,
    and check Z4's comparison."""
    warm = (record.get("warm_start") or {}).get("score")
    best = (record.get("best") or {}).get("score")
    if not warm or not best:
        return {"available": False}
    out = {"available": True}
    for name in ("all", "no_near_repeat", "ip_negative", "corner"):
        out[name] = {"warm_start": warm[name]["S"], "trained": best[name]["S"], "difference": best[name]["S"] - warm[name]["S"]}
    out["z4_trained_at_or_above"] = bool(best["all"]["S"] >= warm["all"]["S"] and best["no_near_repeat"]["S"] >= warm["no_near_repeat"]["S"])
    return out


# --------------------------------------------------------------------------- report
def _f(v, digits: int = 4) -> str:
    return "-" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{digits}f}"


def terms_table(rows: Mapping[str, Mapping], order: Sequence[str]) -> list[str]:
    head = ["Slice", "Shots", "S", "Identity", "R2 psi", "R2 psi per shot", "R2 q95", "R2 betaN", "1 - D_LCFS", "Consistency",
            *CONS_NAMES]
    lines = ["| " + " | ".join(head) + " |", "| " + " | ".join("---" for _ in head) + " |"]
    for name in order:
        t = rows.get(name)
        if t is None:
            continue
        ident = score_identity(t)
        lines.append("| " + " | ".join([name, str(int(t.get("n_shots", 0))), _f(t.get("S")), _f(ident["reconstructed"]),
                                        _f(t.get("r2_psi")), _f(t.get("r2_psi_per_shot_mean")), _f(t.get("r2_q95")),
                                        _f(t.get("r2_betaN")), _f(t.get("one_minus_dlcfs")), _f(t.get("consistency")),
                                        *[_f(t.get(f"r2_{c}")) for c in CONS_NAMES]]) + " |")
    return lines


def render(result: Mapping) -> str:
    """The Markdown report of ``evaluate`` results."""
    r = result
    lines = [f"# Evaluation of run {r['run_id']}", "",
             f"Validation list `{r['lists']['validation']}` ({r['eval_set']['n_shots']} shots, {r['eval_set']['n_frames']} frames at "
             f"stride {r['eval_set']['stride']}); chain: checkpoints {', '.join(r['chain']['runs'])}, mirror averaging "
             f"{'on' if r['chain']['mirror_average'] else 'off'}, q95 from the map, betaN from the DIII-D raw-input ridge "
             f"(fitted on `{r['lists']['train']}` at stride {r['heads']['stride']}). Corner: q_cyl < "
             f"{r['eval_set']['corner']['thresholds']['q_cyl']:.3f} and ts_beta_p < "
             f"{r['eval_set']['corner']['thresholds']['ts_beta_p_relaxed' if r['eval_set']['corner']['relaxed'] else 'ts_beta_p']:.4f}"
             f"{' (relaxed)' if r['eval_set']['corner']['relaxed'] else ''}, {r['eval_set']['corner']['n_frames']} frames.", "",
             mast_floor_line(), "", "## Checks", "", "| Id | Check | Status |", "| --- | --- | --- |"]
    for c in r["checks"]:
        lines.append(f"| {c['id']} | {c['check']} | {c['status']} |")
    lines += ["", "## Score terms by slice", "",
              "Identity: `0.55 max(0, R2 psi) + 0.15 max(0, R2 scalars) + 0.10 (1 - D_LCFS) + 0.20 consistency`, beside the "
              "scorer's S.", "", *terms_table(r["scores"], [*SHOT_SLICES, "corner"]), "", mast_floor_line(), "",
              "## Score terms by phase", "", *terms_table(r["scores"], [f"{g}phase_{p}" for g in ("", "no_near_repeat_") for p in PHASES]),
              "", mast_floor_line(), "", "## Worst 20 validation shots by R2 psi", "", "| Shot | R2 psi | D_LCFS | Boundary failures |",
              "| --- | --- | --- | --- |"]
    for w in r["worst_20"]:
        lines.append(f"| {w['shot']} | {_f(w['r2_psi'])} | {_f(w['d_lcfs'])} | {w['lcfs_fails']} of {w['lcfs_frames']} |")
    o = r["overfit"]
    lines += ["", "## Overfit panel", "", "| Epoch | Train loss | Validation loss | No-near-repeat loss |", "| --- | --- | --- | --- |"]
    for key in ("best_epoch", "stopped_epoch"):
        e = o.get(key)
        if e:
            lines.append(f"| {e['epoch']} ({key.replace('_', ' ')}) | {_f(e['train'], 5)} | {_f(e['validation'], 5)} | "
                         f"{_f(e['validation_no_near_repeat'], 5)} |")
    for label, g in (("Trained", o.get("trained_gap")), ("Warm start (ridge)", o.get("warm_start_gap"))):
        if g:
            lines.append("")
            lines.append(f"{label}: S all {_f(g['all'])}, no near-repeat {_f(g['no_near_repeat'])}, gap {_f(g['gap'])}.")
    ss = o["seed_spread"]
    lines += ["", f"Seed spread of S over {ss['n']} run(s): mean {_f(ss['mean'])}, standard deviation {_f(ss['std'])}, "
              f"range {_f(ss['min'])} to {_f(ss['max'])}."]
    for s in o["seeds"]:
        lines.append(f"- {s['run_id']} (seed {s['seed']}): S {_f(s['S'])}, no near-repeat {_f(s['S_nnr'])}")
    w = r["warm_start"]
    wc = r.get("warm_chain", {"available": False})
    lines += ["", "## Warm start beside the trained model", ""]
    if wc.get("available"):
        lines += ["Prediction chain (warm-start checkpoint `warm.pt` against the best checkpoint; q95 from the map, betaN "
                  "from the DIII-D head, mirror averaging on):", "", "| Slice | Warm start | Trained | Difference |",
                  "| --- | --- | --- | --- |"]
        for name in ("all", "no_near_repeat", "ip_negative", "corner"):
            lines.append(f"| {name} | {_f(wc[name]['warm_start'])} | {_f(wc[name]['trained'])} | {_f(wc[name]['difference'])} |")
        lines.append("")
    if w.get("available"):
        lines += ["Training-time scorer (fixed ridge scalars):", "", "| Slice | Warm start | Trained | Difference |", "| --- | --- | --- | --- |"]
        for name in ("all", "no_near_repeat", "ip_negative", "corner"):
            lines.append(f"| {name} | {_f(w[name]['warm_start'])} | {_f(w[name]['trained'])} | {_f(w[name]['difference'])} |")
    else:
        lines.append("The run record holds no training-time warm-start or best score (trained with --no-score).")
    m = r["mirror"]
    lines += ["", f"Chain score, mirror averaging on: S {_f(m['on']['S'])} (no near-repeat {_f(m['on_nnr'])}); off: "
              f"S {_f(m['off']['S'])} (no near-repeat {_f(m['off_nnr'])})."]
    lines += ["", "## Clip rates", "", f"Share of scaled values beyond {r['clip_bound']:g} per column."]
    for label, entry in r["clip_rates"].items():
        top = sorted(entry["rates"].items(), key=lambda kv: -kv[1])
        lines += ["", f"{label} ({entry['n_frames']} frames): " + ", ".join(f"`{c}` {v:.4f}" for c, v in top if v > 0)
                  + ("" if any(v > 0 for _, v in top) else "no value beyond the bound") + "."]
    lines += ["", "## Grouped permutation importance (validation, all shots)", ""]
    if r.get("permutation"):
        head = ["Group", "Columns", "S", *[f"drop {t}" for t in ("S", "r2_psi", "r2_qb", "one_minus_dlcfs", "consistency")]]
        lines += ["| " + " | ".join(head) + " |", "| " + " | ".join("---" for _ in head) + " |"]
        for g, e in r["permutation"].items():
            lines.append(f"| {g} | {len(e['columns'])} | {_f(e['terms'].get('S'))} | " + " | ".join(
                _f(e["drop"].get(t)) for t in ("S", "r2_psi", "r2_qb", "one_minus_dlcfs", "consistency")) + " |")
        lines += ["", "Drop of each consistency value:", "", "| Group | " + " | ".join(CONS_NAMES) + " |",
                  "| --- " * (1 + len(CONS_NAMES)) + "|"]
        for g, e in r["permutation"].items():
            lines.append(f"| {g} | " + " | ".join(_f(e["drop"].get(f"r2_{c}")) for c in CONS_NAMES) + " |")
    else:
        lines.append("Not run (--no-permutation).")
    lines += ["", "## MAST sensitivity (label-free)", ""]
    ms_ = r.get("mast_sensitivity")
    if ms_:
        lines += [f"{ms_['n_shots']} MAST test shots, {ms_['n_frames']} frames; unpermuted maps: boundary on "
                  f"{_f(ms_['boundary_share'], 3)} of frames, map RMS {ms_['map_rms']:.4g} Wb/rad; medians "
                  + ", ".join(f"{k} {_f(v, 3)}" for k, v in ms_["base_medians"].items()) + ". No MAST label is read.", "",
                  "| Group | Map change / RMS | Boundary share | Frames with both | " + " | ".join(f"median abs change {k}" for k in DERIVED) + " |",
                  "| --- " * (4 + len(DERIVED)) + "|"]
        for g, e in ms_["groups"].items():
            lines.append(f"| {g} | {_f(e['map_rms_change_relative'], 3)} | {_f(e['boundary_share'], 3)} | {e['n_both']} | "
                         + " | ".join(_f(e[f'median_abs_change_{k}'], 4) for k in DERIVED) + " |")
    else:
        lines.append("Not run (no MAST test tables, or --no-mast).")
    lines += ["", mast_floor_line(), ""]
    return "\n".join(lines)


def checks(result: Mapping) -> list[dict]:
    """Status of the checks the evaluation reports (Z4, Z7, Z8, Z11, Z12)."""
    scores = result["scores"]
    slices_present = all(k in scores for k in (*SHOT_SLICES, "corner", *[f"phase_{p}" for p in PHASES]))
    z7 = slices_present and bool(result.get("worst_20")) and bool(result.get("overfit"))
    w = result.get("warm_chain") if (result.get("warm_chain") or {}).get("available") else result["warm_start"]
    z4 = "review: trained {} the warm start (all {} against {}, no near-repeat {} against {})".format(
        "at or above" if w["z4_trained_at_or_above"] else "below", _f(w["all"]["trained"]), _f(w["all"]["warm_start"]),
        _f(w["no_near_repeat"]["trained"]), _f(w["no_near_repeat"]["warm_start"])) if w.get("available") else \
        "review: no warm-start score (no warm.pt, no score in run.json)"
    m = result["mirror"]
    rates = [v for e in result["clip_rates"].values() for v in e["rates"].values()]
    out = [{"id": "Z4", "check": "Warm start", "status": z4},
           {"id": "Z7", "check": "Slices present", "status": "pass" if z7 else "fail"},
           {"id": "Z8", "check": "Clip rates", "status": f"review: largest share {max(rates):.4f}" if rates else "review: no table"},
           {"id": "Z11", "check": "Mirror averaging", "status": f"review: on {_f(m['on']['S'])}, off {_f(m['off']['S'])}"}]
    if result.get("mast_sensitivity"):
        out.append({"id": "Z12", "check": "MAST sign", "status": "pass (asserted before the MAST maps)"})
    return out


# ---------------------------------------------------------------------------- driver
@dataclass
class EvalPaths:
    """Locations an evaluation reads."""

    run_root: Path = pr.DEFAULT_RUN_ROOT
    chain: pr.ChainPaths = field(default_factory=pr.ChainPaths)
    split: Path = dataset.REPO_ROOT / "splits" / "split_v1.json"
    references_dir: Path = bl.DEFAULT_REFERENCE_DIR


def evaluate(run_id: str, paths: EvalPaths = EvalPaths(), seed_runs: Sequence[str] = (), workers: int | None = None,
             scalar_workers: int | None = None, device: str | None = "cpu", permutation: bool = True, mast: bool = True,
             mast_shot_step: int = 20, heads_stride: int = 4, seed: int = 0, log: Callable[[str], None] = print) -> dict:
    """Every section of ``eval.md`` for one run (module docstring)."""
    record = json.loads((Path(paths.run_root) / run_id / "run.json").read_text())
    lists = record["lists"]
    for name in lists.values():
        ld.require_readable(name)
    chain, chain_record = pr.build_chain([run_id], paths.chain, mirror_average=True, seed_average=False, device=device,
                                         workers=scalar_workers)
    config = chain.coefficients.members[0].config
    version = chain_record["feature_version"]
    data_paths = ld.DataPaths(feature_root=paths.chain.feature_root, feature_config=paths.chain.feature_config,
                              scaler=paths.chain.scaler, target_root=paths.chain.targets_root, split=paths.split)
    source = pr.TableSource(paths.chain.table_root, paths.chain.feature_root, version,
                            {m: chain.frame_columns(m) for m in pr.MACHINES}, chain.feature_columns())

    train_rows = pr.heads_training_rows(source, paths.split, lists["train"], heads_stride)
    chain.heads = sh.fit_heads(train_rows, None, (float("nan"), float("nan")), {"train_list": lists["train"], "stride": heads_stride})
    del train_rows
    seq = chain.coefficients.sequence
    es = build_eval_set(source, config, data_paths, lists["validation"], lists["train"], paths.chain.data_root,
                        paths.references_dir, workers, log, mark=seq)
    frame_slices = list(es.frame_masks)
    preds = predictions(chain, es.rows)
    scores, result = score(preds, es, paths.chain.data_root, workers, frame_slices)
    log(f"  chain score: S {scores['all']['S']:.4f} (no near-repeat {scores['no_near_repeat']['S']:.4f}, corner {scores['corner']['S']:.4f})")

    chain.coefficients.mirror_average = False
    off, _ = score(predictions(chain, es.rows), es, paths.chain.data_root, workers)
    chain.coefficients.mirror_average = True
    mirror = {"on": scores["all"], "on_nnr": scores["no_near_repeat"]["S"], "off": off["all"], "off_nnr": off["no_near_repeat"]["S"]}

    warm_scores = None
    if (Path(paths.chain.checkpoint_root) / run_id / "warm.pt").is_file():
        warm_chain, _ = pr.build_chain([run_id], paths.chain, which="warm", mirror_average=True, seed_average=False, device=device,
                                       workers=scalar_workers)
        warm_chain.heads = chain.heads
        warm_scores, _ = score(predictions(warm_chain, es.rows), es, paths.chain.data_root, workers, ["corner"])
        warm_chain.close()
        log(f"  warm start through the chain: S {warm_scores['all']['S']:.4f} (no near-repeat {warm_scores['no_near_repeat']['S']:.4f})")

    fconfig = chain.coefficients.fconfig
    groups = feature_groups(fconfig, chain.coefficients.columns)
    perm = permutation_importance(chain, es, groups, scores["all"], paths.chain.data_root, workers, seed, log) if permutation else None
    mast_part = None
    if mast and ft.table_path("mast_public_test", paths.chain.table_root).is_file() \
            and fe.features_path("mast_public_test", version, paths.chain.feature_root).is_file():
        ids = sorted(ft.read_frame_table("mast_public_test", ["shot"], root=paths.chain.table_root)["shot"].astype(str).unique())
        chosen = ids[::max(1, int(mast_shot_step))]
        rows = pr.read_rows(source, "mast_public_test", chosen, es.stride, mark=seq)
        mast_part = mast_sensitivity(chain, rows, groups, workers, seed, log)

    seeds = [{"run_id": run_id, "seed": record.get("seed"), "S": scores["all"]["S"], "S_nnr": scores["no_near_repeat"]["S"]}]
    for other in seed_runs:
        other_chain, _ = pr.build_chain([other], paths.chain, mirror_average=True, seed_average=False, device=device,
                                        workers=scalar_workers)
        other_chain.heads = chain.heads
        terms, _ = score(predictions(other_chain, es.rows), es, paths.chain.data_root, workers)
        other_chain.close()
        other_record = json.loads((Path(paths.run_root) / other / "run.json").read_text())
        seeds.append({"run_id": other, "seed": other_record.get("seed"), "S": terms["all"]["S"], "S_nnr": terms["no_near_repeat"]["S"]})
        log(f"  seed run {other}: S {terms['all']['S']:.4f} (no near-repeat {terms['no_near_repeat']['S']:.4f})")
    out = {"run_id": run_id, "lists": lists, "chain": chain_record, "heads": {"stride": heads_stride, **chain.heads.summary()},
           "eval_set": {"n_shots": len(es.shot_ids), "n_frames": int(len(es.scored)), "stride": es.stride, "corner": es.corner,
                        "slice_shots": {k: int(v.sum()) for k, v in es.shot_slices.items()},
                        "frame_slices": {k: int(v.sum()) for k, v in es.frame_masks.items()}},
           "scores": scores, "identity": {k: score_identity(v) for k, v in scores.items()}, "mast_floor": MAST_FLOOR,
           "worst_20": worst_shots(result), "overfit": overfit_panel(record, read_metrics(Path(paths.run_root) / run_id), seeds, scores, warm_scores),
           "warm_start": warm_against_trained(record), "warm_chain": warm_chain_comparison(warm_scores, scores), "mirror": mirror, "clip_bound": float(config.get("loader.clip_bound")),
           "clip_rates": clip_rate_table(chain.coefficients.columns, chain.coefficients.scaler, float(config.get("loader.clip_bound")),
                                         paths.chain.feature_root, version, es.shot_ids),
           "permutation": perm, "mast_sensitivity": mast_part}
    out["checks"] = checks(out)
    chain.close()
    return out
