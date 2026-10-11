"""Fits four baselines on the training list of a split and scores them on its validation list.

The models (``fusion_eq.models``): ``ridge_raw`` (the plain PCA and ridge baseline),
``ridge_signed`` (the same with the current inputs in the physical frame),
``physics_mean`` (coil flux under the adopted DIII-D coil model plus Ip times the mean
plasma shape per kA, no learned flux model) and ``hybrid_ridge`` (PCA and ridge of the
plasma target, added to the coil flux), plus two ceilings with no learned flux model:
the true validation maps encoded and decoded with the training basis of the stored map,
and with the basis of the plasma target with the coil flux added back.
``physics_mean``, ``hybrid_ridge`` and the two ceilings take q95 and betaN from the
scalar ridge of ``ridge_signed``, so that they differ from it in the flux map alone.

Every model is scored with the challenge scorer on the same validation frames (every
``--val-frame-stride``-th frame of each validation shot, no frame filter), in
``--workers`` processes with the scorer references cached. Writes to ``--out``
(``reports/baseline/``): ``compare_baselines.parquet`` (S and its terms per model and
subset: all validation shots, those with and without a near-repeat in training, and by
Ip direction), ``compare_baselines.md``, ``compare_baselines.json``, and per model the
residual diagnostics: ``residual_maps.npz`` and ``residual_<model>.png`` (mean absolute
residual per pixel over all validation frames and per phase), ``frame_residuals.parquet``
(one row per validation frame with the rms residual, the phase, ``ip_over_peak``, the
solenoid current, the Ip direction and the shot's distance to its nearest shot of the
training list from ``--neighbours``) and ``score_parts.parquet`` (the scorer's per-shot
parts). Fitted models are saved under ``--cache-dir``. Usage::

    python scripts/compare_baselines.py --working
    python scripts/compare_baselines.py
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from fusion_eq import baseline as bl, frame_table as ft, models, split as sp
from fusion_eq.coil_flux import GreenTable, adopted_scales
from fusion_eq.dataset import REPO_ROOT, default_data_root, run_info

DEFAULT_SPLIT = REPO_ROOT / "splits" / "split_v1.json"
DEFAULT_OUT = REPO_ROOT / "reports" / "baseline"
DEFAULT_GREEN_DIR = REPO_ROOT.parent / "data" / "processed" / "green"
DEFAULT_NEIGHBOURS = REPO_ROOT / "reports" / "split" / "shot_neighbours.parquet"
TERMS = ("S", "r2_psi", "r2_qb", "one_minus_dlcfs", "consistency")


def nearest_training_distance(val_ids: list[str], train_ids: list[str], neighbours_path: Path) -> dict[str, float]:
    """Distance of each validation shot to its nearest shot of the training list, from the
    neighbour table of ``find_near_repeats.py`` (NaN when none of the listed neighbours is in
    the training list, or without the table)."""
    if not neighbours_path.is_file():
        return {s: np.nan for s in val_ids}
    nn = pd.read_parquet(neighbours_path).set_index("shot")
    train = set(train_ids)
    k = max(int(c[2:-5]) for c in nn.columns if c.startswith("nn") and c.endswith("_shot"))
    out = {}
    for s in val_ids:
        out[s] = np.nan
        if s not in nn.index:
            continue
        for j in range(1, k + 1):
            if nn.loc[s, f"nn{j}_shot"] in train:
                out[s] = float(nn.loc[s, f"nn{j}_distance"])
                break
    return out


def residual_figure(summary: models.ResidualSummary, path: Path) -> None:
    """One PNG per model: the mean absolute residual per pixel, all frames and per phase."""
    fig, axes = plt.subplots(1, len(models.RESIDUAL_GROUPS), figsize=(4.2 * len(models.RESIDUAL_GROUPS), 4.6))
    finite = [m[np.isfinite(m)] for m in summary.mean_abs.values()]
    finite = [v for v in finite if v.size]
    vmax = float(max(v.max() for v in finite)) if finite else 1.0
    vmax = vmax if vmax > 0 else 1.0
    for ax, group in zip(axes, models.RESIDUAL_GROUPS):
        im = ax.imshow(summary.mean_abs[group], origin="lower", cmap="viridis", vmin=0.0, vmax=vmax, aspect="equal")
        ax.set_title(f"{group} ({summary.counts[group]} frames)", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=axes.tolist(), fraction=0.02, pad=0.02, label="mean |residual| (Wb/rad)")
    fig.suptitle(f"{summary.model}: mean absolute residual per pixel, validation frames", fontsize=11)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def subsets_of(val_ids: list[str], split: dict, index: pd.DataFrame) -> dict[str, np.ndarray]:
    """Validation subsets as bool masks over ``val_ids``."""
    near = np.array([bool(split["shots"].get(s, {}).get("near_repeat_in_train") or False) for s in val_ids])
    known = np.array([split["shots"].get(s, {}).get("near_repeat_in_train") is not None for s in val_ids])
    direction = index.set_index("shot")["ip_direction"].reindex(val_ids).to_numpy()
    out = {"all": np.ones(len(val_ids), dtype=bool)}
    if known.all():
        out["with_near_repeat"] = near
        out["without_near_repeat"] = ~near
    out["ip_positive"] = direction == 1
    out["ip_negative"] = direction == -1
    return {k: v for k, v in out.items() if v.any()}


def render_report(table: pd.DataFrame, args, info: dict) -> str:
    lines = [f"# Baseline comparison, split `{args.split}`", "",
             f"{info['n_train_shots']} training shots ({info['n_train_frames']} frames), {info['n_val_shots']} validation shots "
             f"scored on every {args.val_frame_stride}{'st' if args.val_frame_stride == 1 else 'th'} frame "
             f"({'working subsets' if args.working else 'full lists'}); {info['workers']} workers; "
             f"references {'from the cache' if info['references_from_cache'] else 'built'}.", "",
             f"Models: `ridge_raw`, `ridge_signed`, `physics_mean`, `hybrid_ridge` (`fusion_eq.models`), and two "
             f"ceilings with no learned flux model: `ceiling_psi_basis` (the true validation maps encoded and decoded "
             f"with the training basis of the stored map) and `ceiling_plasma_basis` (the same with the basis of the "
             f"plasma target, the coil flux added back). `physics_mean`, `hybrid_ridge` and the two ceilings take q95 "
             f"and betaN from the scalar ridge of `ridge_signed`, so their differences from `ridge_signed` come from "
             f"the flux map alone. "
             f"Coil model: {info['coil_model']}; flat solenoid term {info['flat_wb_per_ka']:.5f} Wb/rad per kA; "
             f"mean plasma shape over {info['n_mean_frames']} frames; {info['params']['n_components']} components.", ""]
    for subset, g in table.groupby("subset", sort=False):
        lines += [f"## {subset} ({int(g['n_shots'].iloc[0])} shots)", "",
                  "| Model | S | R2 psi | R2 q95,betaN | 1 - D_LCFS | Consistency | psi sign |", "| --- | --- | --- | --- | --- | --- | --- |"]
        for r in g.itertuples(index=False):
            lines.append(f"| {r.model} | {r.S:.4f} | {r.r2_psi:.4f} | {r.r2_qb:.4f} | {r.one_minus_dlcfs:.4f} | "
                         f"{r.consistency:.4f} | {int(r.psi_sign):+d} |")
        lines.append("")
    lines += ["Runtime (s): " + ", ".join(f"{k} {v:.0f}" for k, v in info["timing"].items()), ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--working", action="store_true", help="use the working subsets")
    parser.add_argument("--val-frame-stride", type=int, default=4)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--n-components", type=int, default=bl.BaselineParams.n_components)
    parser.add_argument("--alpha", type=float, default=bl.BaselineParams.alpha)
    parser.add_argument("--pca-frames", type=int, default=bl.BaselineParams.pca_frames)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--table-dir", type=Path, default=ft.DEFAULT_TABLE_ROOT)
    parser.add_argument("--data-root", type=Path, default=default_data_root())
    parser.add_argument("--green-dir", type=Path, default=DEFAULT_GREEN_DIR)
    parser.add_argument("--cache-dir", type=Path, default=bl.DEFAULT_CACHE_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--neighbours", type=Path, default=DEFAULT_NEIGHBOURS,
                        help="shot_neighbours.parquet of find_near_repeats.py, for the nearest-training-shot distance")
    args = parser.parse_args(argv)

    started = time.time()
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    args.out.mkdir(parents=True, exist_ok=True)
    params = bl.BaselineParams(n_components=args.n_components, alpha=args.alpha, pca_frames=args.pca_frames, seed=args.seed)
    workers = bl.default_workers() if args.workers is None else args.workers

    split = sp.read_split(args.split)
    train_ids = sorted(split["working_train" if args.working else "train"])
    val_ids = sorted(split["working_validation" if args.working else "validation"])
    index = pd.read_parquet(args.table_dir / "shot_index.parquet")
    index["shot"] = index["shot"].astype(str)
    table = GreenTable.load(args.green_dir / "green_d3d.npz")
    scales = adopted_scales("DIII-D", table.columns)
    columns = bl.input_columns()
    timing = {}

    t0 = time.time()
    frames = bl.frames_of_shots(train_ids, [*columns, *bl.SCALAR_TARGETS.values()], args.table_dir)
    frames["shot"] = frames["shot"].astype(str)
    timing["frames_s"] = time.time() - t0
    print(f"{len(train_ids)} training shots, {len(frames)} frames, {len(val_ids)} validation shots; {workers} workers", flush=True)

    t0 = time.time()
    model_set = models.fit_model_set(frames, train_ids, args.data_root, table, params, args.cache_dir, scales, log=lambda m: print(m, flush=True))
    timing["fit_s"] = time.time() - t0
    for name, model in (("ridge_raw", model_set.ridge_raw), ("ridge_signed", model_set.ridge_signed), ("hybrid_ridge", model_set.hybrid)):
        model.save(args.cache_dir / f"model_{name}_{bl.hash_of_ids(train_ids)}.npz")
    np.savez(args.cache_dir / f"mean_shape_{bl.hash_of_ids(train_ids)}_adopted.npz", mean=model_set.mean_shape, n=model_set.n_mean_frames)

    val_frames = bl.frames_of_shots(val_ids, [*columns, "n_frames", "ip_over_peak"], args.table_dir)
    val_frames["shot"] = val_frames["shot"].astype(str)
    val_by_shot = {s: g.sort_values("frame").iloc[::args.val_frame_stride] for s, g in val_frames.groupby("shot")}
    t0 = time.time()
    references = bl.build_references(val_ids, args.data_root, frame_stride=args.val_frame_stride, workers=workers,
                                     cache_dir=args.cache_dir / "references")
    timing["references_s"] = time.time() - t0
    print(f"scorer references for {len(val_ids)} shots {'read from the cache' if references.from_cache else 'built'}, "
          f"{timing['references_s']:.0f} s", flush=True)

    subsets = subsets_of(val_ids, split, index)
    nearest = nearest_training_distance(val_ids, train_ids, args.neighbours)
    rows, frame_rows, part_rows, residual_maps = [], [], [], {}
    candidates = {**model_set.baselines(), **models.ceiling_baselines(model_set, args.data_root, args.val_frame_stride)}
    for name, model in candidates.items():
        t0 = time.time()
        result = bl.score_local(lambda s, m=model: m.predict(val_by_shot[s]), val_ids, args.data_root,
                                frame_stride=args.val_frame_stride, references=references, workers=workers)
        timing[f"score_{name}_s"] = time.time() - t0
        for subset, mask in subsets.items():
            terms = bl.score_terms(result.finalize(mask))
            rows.append({"model": name, "subset": subset, "scalars_from": model.scalars_from or name, **terms})
        print(f"{name}: S = {rows[-len(subsets)]['S']:.4f} on all validation shots, {timing[f'score_{name}_s']:.0f} s", flush=True)
        t0 = time.time()
        summary = models.residual_summary(name, model, val_by_shot, args.data_root, args.val_frame_stride, result.psi_sign)
        summary.frames["nn_train_distance"] = summary.frames["shot"].map(nearest)
        frame_rows.append(summary.frames)
        part_rows.append(models.score_parts_table(name, result))
        for group, m in summary.mean_abs.items():
            residual_maps[f"{name}_{group}"] = m
        residual_figure(summary, args.out / f"residual_{name}.png")
        timing[f"residuals_{name}_s"] = time.time() - t0
    table_out = pd.DataFrame(rows)
    table_out.to_parquet(args.out / "compare_baselines.parquet", index=False)
    pd.concat(frame_rows, ignore_index=True).to_parquet(args.out / "frame_residuals.parquet", index=False)
    pd.concat(part_rows, ignore_index=True).to_parquet(args.out / "score_parts.parquet", index=False)
    np.savez(args.out / "residual_maps.npz", **residual_maps, grid_r=table.grid_r, grid_z=table.grid_z)
    info = run_info(argv, vars(args), timestamp, time.time() - started, params=params.to_dict(), workers=workers,
                    n_train_shots=len(train_ids), n_train_frames=int(len(frames)), n_val_shots=len(val_ids),
                    references_from_cache=references.from_cache, coil_model=scales.source,
                    flat_wb_per_ka=scales.flat_wb_per_ka, n_mean_frames=model_set.n_mean_frames, timing=timing,
                    subsets={k: int(v.sum()) for k, v in subsets.items()})
    (args.out / "compare_baselines.json").write_text(json.dumps(info, indent=2, default=float) + "\n")
    report = render_report(table_out, args, info)
    (args.out / "compare_baselines.md").write_text(report)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
