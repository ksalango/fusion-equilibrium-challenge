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
``<report-root>/v<version>/``. Usage::

    python scripts/feature_linear_check.py              # working lists of splits/split_v1.json
    python scripts/feature_linear_check.py --full       # the full training and validation lists
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from fusion_eq import baseline as bl, dataset, feature_checks as fc, features as fe, frame_table as ft, models, split as sp, targets
from fusion_eq.coil_flux import GreenTable, adopted_scales
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


def log(message: str) -> None:
    print(message, flush=True)


@dataclass
class FeatureRidge:
    """Ridge from scaled features to the coefficients of the per-Ip plasma target.

    Attributes:
        name: Arm name.
        columns: Feature columns in order.
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


def scaled_feature_rows(shot_ids: list[str], config: fe.FeatureConfig, scaler: RobustScaler, root: Path) -> pd.DataFrame:
    """``shot``, ``frame`` and every scaled feature column of the listed training shots, in
    file order (shots sorted, frames in order)."""
    table = fe.read_features(SPLIT_NAME, config.version, scaler.columns, shot_ids, root)
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


def feature_baseline(ridge: FeatureRidge, basis: bl.FluxBasis, x_by_shot: dict[str, pd.DataFrame], table: GreenTable,
                     scales, r0_m: float, scalars) -> models.Baseline:
    """A ``models.Baseline`` whose map is the coil flux plus the target rebuilt from the
    feature ridge, with the scalars of ``scalars``."""
    cfg = machine_config(MACHINE)

    def flux(frames: pd.DataFrame) -> np.ndarray:
        shots = frames["shot"].astype(str).unique()
        if len(shots) != 1:
            raise ValueError("the feature arms predict one shot at a time")
        x = x_by_shot[str(shots[0])].loc[frames["frame"].to_numpy(), ridge.columns].to_numpy(dtype=np.float64)
        ip = frames["ip__mean"].to_numpy(dtype=np.float64)
        target = fc.target_from_per_ip(basis.decode(ridge.coefficients(x)), ip, r0_m)
        coil = targets.coil_flux_of_frames(table, frames, scales, cfg)
        return targets.stored_from_target(target, coil, ip, cfg)

    return models.Baseline(ridge.name, flux, scalars, scalars_from="ridge_signed")


def strided(frames: pd.DataFrame, stride: int) -> dict[str, pd.DataFrame]:
    return {s: g.sort_values("frame").iloc[::stride] for s, g in frames.groupby("shot", sort=False)}


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
    args = parser.parse_args(argv)

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

    split = sp.read_split(args.split)
    train_ids = sorted(split["train" if args.full else "working_train"])
    val_ids = sorted(split["validation" if args.full else "working_validation"])
    table = GreenTable.load(args.green_dir / "green_d3d.npz")
    scales = adopted_scales(MACHINE, table.columns)
    log(f"{len(train_ids)} training and {len(val_ids)} validation shots, stride {args.frame_stride}, {workers} workers")

    # The reference models of the comparison, and the scalar ridge every arm uses.
    t0 = time.time()
    columns = bl.input_columns()
    frames = bl.frames_of_shots(train_ids, [*columns, *bl.SCALAR_TARGETS.values()], args.table_dir)
    frames["shot"] = frames["shot"].astype(str)
    model_set = models.fit_model_set(frames, train_ids, args.data_root, table, params, args.cache_dir, scales, log=log)
    baselines = model_set.baselines()
    timing["reference_fit_s"] = time.time() - t0

    # The per-Ip plasma target: basis and training coefficients.
    t0 = time.time()
    frames_by_shot = {s: g for s, g in frames.groupby("shot", sort=False)}
    target_fn = per_ip_target_fn(table, frames_by_shot, scales, r0_m, config.min_abs_ip_ka)
    tag = f"plasma_per_ip_r{r0_m:g}_min{config.min_abs_ip_ka:g}_{scales.source.split(',')[0] or 'scales'}"
    basis, seconds = bl.cached_basis(train_ids, params, args.data_root, args.cache_dir, split=SPLIT_NAME, target=target_fn, tag=tag)
    coefs = bl.encode_shots(train_ids, basis, args.data_root, args.cache_dir, SPLIT_NAME, target=target_fn, tag=tag)
    if len(coefs) != len(frames):
        raise ValueError(f"{len(coefs)} coefficient rows for {len(frames)} training frames")
    keep = fc.ip_usable(frames["ip__mean"], config.min_abs_ip_ka)
    timing["per_ip_basis_s"] = time.time() - t0
    log(f"per-Ip target basis: {basis.n_components} components over {basis.n_frames} frames ({seconds:.0f} s fitting), "
        f"{basis.explained_ratio.sum():.4f} of the variance; {int(keep.sum())} of {len(frames)} training frames above the Ip threshold")

    # Feature ridges, one per set.
    train_x = aligned(frames, scaled_feature_rows(train_ids, config, scaler, args.feature_root))
    ridges = {}
    for set_name in config.sets:
        set_columns = config.set_columns(set_name)
        ridges[f"features_{set_name}"] = fit_feature_ridge(f"features_{set_name}", train_x[set_columns].to_numpy(),
                                                           set_columns, coefs, keep, args.alpha)

    # Validation rows, features and true coefficients.
    val_frames = bl.frames_of_shots(val_ids, [*columns, "n_frames", "ip_over_peak"], args.table_dir)
    val_frames["shot"] = val_frames["shot"].astype(str)
    val_x = aligned(val_frames, scaled_feature_rows(val_ids, config, scaler, args.feature_root))
    x_by_shot = {s: g.set_index("frame") for s, g in val_x.groupby("shot", sort=False)}
    val_by_shot = strided(val_frames, args.frame_stride)

    # Coefficient R2 on the strided validation frames above the Ip threshold.
    t0 = time.time()
    val_target_fn = per_ip_target_fn(table, {s: g for s, g in val_frames.groupby("shot", sort=False)}, scales, r0_m,
                                     config.min_abs_ip_ka)
    val_coefs = bl.encode_shots(val_ids, basis, args.data_root, args.cache_dir, SPLIT_NAME, target=val_target_fn, tag=tag)
    if len(val_coefs) != len(val_frames):
        raise ValueError(f"{len(val_coefs)} coefficient rows for {len(val_frames)} validation frames")
    pick = np.zeros(len(val_frames), dtype=bool)
    for s, g in val_by_shot.items():
        pick[val_frames.index.get_indexer(g.index)] = True
    val_keep = pick & fc.ip_usable(val_frames["ip__mean"], config.min_abs_ip_ka)
    coefficient_r2 = {}

    def r2_row(name, x_train, x_val, n_inputs):
        ridge = fit_feature_ridge(name, x_train, [], coefs, keep, args.alpha)
        coefficient_r2[name] = {"n_inputs": n_inputs,
                                "train": fc.pooled_r2(coefs[keep], ridge.coefficients(x_train[keep])),
                                "validation": fc.pooled_r2(val_coefs[val_keep], ridge.coefficients(x_val[val_keep]))}

    specs = config.column_specs()
    for set_name in config.sets:
        set_columns = config.set_columns(set_name)
        r2_row(f"features_{set_name}", train_x[set_columns].to_numpy(), val_x[set_columns].to_numpy(), len(set_columns))
        for group in dict.fromkeys(specs[c].group for c in set_columns):
            kept = [c for c in set_columns if specs[c].group != group]
            r2_row(f"features_{set_name} without {group}", train_x[kept].to_numpy(), val_x[kept].to_numpy(), len(kept))
    raw_scaler = model_set.ridge_signed.scaler
    raw_train, _ = raw_scaler.transform(targets.signed_frames(frames, cfg))
    raw_val, _ = raw_scaler.transform(targets.signed_frames(val_frames, cfg))
    r2_row("raw inputs of hybrid_ridge", raw_train, raw_val, len(columns))
    timing["coefficient_r2_s"] = time.time() - t0
    for name, r in coefficient_r2.items():
        log(f"coefficient R2 {name}: training {r['train']:.4f}, validation {r['validation']:.4f}")

    # Scoring: the reference and the feature arms on the same frames.
    t0 = time.time()
    references = bl.build_references(val_ids, args.data_root, SPLIT_NAME, args.frame_stride, workers=workers,
                                     cache_dir=args.cache_dir / "references")
    timing["references_s"] = time.time() - t0
    index = pd.read_parquet(Path(args.table_dir) / "shot_index.parquet")
    direction = index.loc[index["split"] == SPLIT_NAME].assign(shot=lambda t: t["shot"].astype(str)) \
        .set_index("shot")["ip_direction"].reindex(val_ids).to_numpy()
    subsets = {"all": np.ones(len(val_ids), dtype=bool), "ip_positive": direction == 1, "ip_negative": direction == -1}
    subsets = {k: v for k, v in subsets.items() if v.any()}
    scalars = baselines["ridge_signed"].scalars
    arms = {"hybrid_ridge": baselines["hybrid_ridge"],
            **{name: feature_baseline(r, basis, x_by_shot, table, scales, r0_m, scalars) for name, r in ridges.items()}}
    rows = []
    for name, model in arms.items():
        t0 = time.time()
        result = bl.score_local(lambda s, m=model: m.predict(val_by_shot[s]), val_ids, args.data_root, SPLIT_NAME,
                                args.frame_stride, references, workers)
        for subset, mask in subsets.items():
            rows.append({"model": name, "subset": subset, **bl.score_terms(result.finalize(mask))})
        timing[f"score_{name}_s"] = time.time() - t0
        log(f"{name}: S {rows[-len(subsets)]['S']:.4f} ({timing[f'score_{name}_s']:.0f} s)")
    scores = pd.DataFrame(rows)
    scores.to_parquet(out_dir / "linear_check.parquet", index=False)

    overall = scores.loc[scores["subset"] == "all"].set_index("model")["S"]
    status = {name: {"status": "pass" if overall[name] >= overall["hybrid_ridge"] else "review",
                     "delta_s": float(overall[name] - overall["hybrid_ridge"]), "n_inputs": len(r.columns)}
              for name, r in ridges.items()}
    status["hybrid_ridge"] = {"status": "reference", "delta_s": 0.0, "n_inputs": len(columns)}
    info = run_info(argv, vars(args), timestamp, time.time() - started, version=config.version, config_hash=config.hash(),
                    workers=workers, n_train_shots=len(train_ids), n_train_frames=int(len(frames)), n_val_shots=len(val_ids),
                    n_val_frames=int(sum(len(g) for g in val_by_shot.values())), n_components=basis.n_components,
                    per_ip_basis_explained=float(basis.explained_ratio.sum()), r0_m=r0_m, min_abs_ip_ka=config.min_abs_ip_ka,
                    coil_model=scales.source, references_from_cache=references.from_cache, f61_reference_s=F61_REFERENCE_S,
                    status=status, coefficient_r2=coefficient_r2, timing=timing,
                    scores=scores.to_dict(orient="records"))
    (out_dir / "linear_check.json").write_text(json.dumps(info, indent=2, default=fc.json_default) + "\n")
    report = render(scores, info)
    (out_dir / "linear_check.md").write_text(report)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
