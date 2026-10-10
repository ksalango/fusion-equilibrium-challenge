"""Leak test: how much of a local score depends on near-repeat shots (B8c, S7).

With the split given by ``--split``, the PCA and ridge baseline (``fusion_eq.baseline``)
is fitted three ways and scored on the same validation shots with the challenge scorer:

- A: on the whole training part;
- B: on the training part with every near-repeat of a validation shot removed, at
  ``--threshold`` on the distance matrix of ``find_near_repeats.py``;
- C: on the training part with the same number of shots removed at random, one arm
  per seed (``--n-random-arms``).

The flux basis and the input scaler are fitted once on the whole training part and
shared, so the arms differ only in the rows given to the ridge. Scores are reported
for all validation shots, for those with a near-repeat in training and for those
without. ``--working`` uses the working subsets to record the runtime first;
``--val-frame-stride`` scores every n-th frame of each validation shot when scoring
all frames is slow. Writes ``reports/split/leak_test.parquet``, appends S7 to
``split_report.md`` and caches the basis and the coefficients under
``../data/processed/baseline/``. Usage::

    python scripts/leak_test.py --split reports/split/split_provisional.json --working
    python scripts/leak_test.py --split reports/split/split_provisional.json
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from fusion_eq import baseline as bl, frame_table as ft, similarity as sim, split as sp
from fusion_eq.dataset import REPO_ROOT, default_data_root, run_info

DEFAULT_OUT = REPO_ROOT / "reports" / "split"
DEFAULT_DISTANCE = bl.dataset.DEFAULT_PROCESSED_ROOT / "split" / "distance.npz"
SUBSETS = ("all", "with_near_repeat", "without_near_repeat")


def load_distance(path: Path) -> sim.PairwiseDistance:
    with np.load(path, allow_pickle=False) as f:
        return sim.PairwiseDistance(shots=f["shots"].astype(object), distance=f["distance"].astype(np.float64),
                                    n_common_times=f["n_common_times"].astype(np.int32))


def fit_basis(train_ids: list[str], params: bl.BaselineParams, data_root: Path, cache_dir: Path) -> tuple[bl.FluxBasis, float]:
    """The basis of a training set, fitted once and cached by the ids and parameters."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"basis_{bl.hash_of_ids(train_ids)}_k{params.n_components}_f{params.pca_frames}_s{params.seed}.npz"
    if path.is_file():
        return bl.FluxBasis.load(path), 0.0
    t0 = time.time()
    order = bl.basis_shot_order(train_ids, params.seed)
    basis = bl.fit_flux_basis((maps for _, maps in bl.shot_maps(order, data_root)), params.n_components, params.pca_frames)
    basis.save(path)
    return basis, time.time() - t0


def render_section(rows: pd.DataFrame, args, timing: dict, n_removed: int, n_train: int, n_val: int, n_with: int) -> str:
    lines = [f"## S7 Leak test (review), split `{args.split}`", "",
             f"Threshold {args.threshold}; {'working subsets' if args.working else 'full parts'}; "
             f"validation frame stride {args.val_frame_stride}; {n_train} training shots, {n_val} validation shots "
             f"of which {n_with} have a near-repeat in training; arm B removes {n_removed} training shots.", "",
             f"Runtime: basis {timing['basis_s']:.0f} s, coefficients {timing['encode_s']:.0f} s, "
             f"references {timing['references_s']:.0f} s, per arm fit {timing['fit_s_per_arm']:.0f} s and "
             f"scoring {timing['score_s_per_arm']:.0f} s.", "",
             "| Arm | Seed | Subset | Shots | S | R2 psi | R2 q95,betaN | 1 - D_LCFS | Consistency | psi sign |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows.itertuples(index=False):
        lines.append(f"| {r.arm} | {r.seed} | {r.subset} | {r.n_shots} | {r.S:.4f} | {r.r2_psi:.4f} | {r.r2_qb:.4f} | "
                     f"{r.one_minus_dlcfs:.4f} | {r.consistency:.4f} | {int(r.psi_sign):+d} |")
    a = rows[(rows["arm"] == "A") & (rows["subset"] == "all")]["S"]
    b = rows[(rows["arm"] == "B") & (rows["subset"] == "all")]["S"]
    c = rows[(rows["arm"] == "C") & (rows["subset"] == "all")]["S"]
    if len(a) and len(b) and len(c):
        lines += ["", f"Leak A minus B on all validation shots: {a.iloc[0] - b.iloc[0]:+.4f}; "
                      f"spread of the C arms (max minus min): {c.max() - c.min():.4f}, A minus mean C: {a.iloc[0] - c.mean():+.4f}."]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=sim.PROVISIONAL_THRESHOLD)
    parser.add_argument("--working", action="store_true", help="use the working subsets")
    parser.add_argument("--val-frame-stride", type=int, default=1)
    parser.add_argument("--n-random-arms", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0, help="seed of the random removals (arm C) and of the basis fit")
    parser.add_argument("--n-components", type=int, default=bl.BaselineParams.n_components)
    parser.add_argument("--alpha", type=float, default=bl.BaselineParams.alpha)
    parser.add_argument("--pca-frames", type=int, default=bl.BaselineParams.pca_frames)
    parser.add_argument("--table-dir", type=Path, default=ft.DEFAULT_TABLE_ROOT)
    parser.add_argument("--data-root", type=Path, default=default_data_root())
    parser.add_argument("--distance", type=Path, default=DEFAULT_DISTANCE)
    parser.add_argument("--cache-dir", type=Path, default=bl.DEFAULT_CACHE_DIR)
    parser.add_argument("--workers", type=int, default=None,
                        help="processes for the scorer references and the scoring; CPU count minus two by default")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    started = time.time()
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    args.out.mkdir(parents=True, exist_ok=True)
    params = bl.BaselineParams(n_components=args.n_components, alpha=args.alpha, pca_frames=args.pca_frames, seed=args.seed)

    split = sp.read_split(args.split)
    train_ids = sorted(split["working_train" if args.working else "train"])
    val_ids = sorted(split["working_validation" if args.working else "validation"])
    dist = load_distance(args.distance)
    leak_ids = sorted(np.asarray(train_ids)[sim.has_near_repeat_in(dist, train_ids, val_ids, args.threshold)].tolist())
    val_with = sim.has_near_repeat_in(dist, val_ids, train_ids, args.threshold)
    print(f"{len(train_ids)} training shots, {len(val_ids)} validation shots ({int(val_with.sum())} with a near-repeat "
          f"in training); arm B removes {len(leak_ids)} training shots", flush=True)

    basis, basis_s = fit_basis(train_ids, params, args.data_root, args.cache_dir)
    print(f"basis: {basis.n_components} components over {basis.n_frames} frames, "
          f"{basis.explained_ratio.sum():.4%} of the variance, {basis_s:.0f} s", flush=True)
    t0 = time.time()
    coefs = bl.encode_shots(train_ids, basis, args.data_root, args.cache_dir)
    encode_s = time.time() - t0
    columns = bl.input_columns()
    frames = bl.frames_of_shots(train_ids, [*columns, *bl.SCALAR_TARGETS.values()], args.table_dir)
    frames["shot"] = frames["shot"].astype(str)
    if len(frames) != coefs.shape[0] or frames["shot"].drop_duplicates().tolist() != train_ids:
        raise SystemExit("frame table rows and cached coefficients do not line up")
    scaler = bl.InputScaler.fit(frames, columns)
    print(f"{len(frames)} training frames encoded, {encode_s:.0f} s", flush=True)

    val_frames = bl.frames_of_shots(val_ids, columns, args.table_dir)
    val_frames["shot"] = val_frames["shot"].astype(str)
    val_by_shot = {s: f.sort_values("frame").iloc[::args.val_frame_stride] for s, f in val_frames.groupby("shot")}
    workers = bl.default_workers() if args.workers is None else args.workers
    t0 = time.time()
    references = bl.build_references(val_ids, args.data_root, frame_stride=args.val_frame_stride, workers=workers,
                                     cache_dir=args.cache_dir / "references")
    references_s = time.time() - t0
    print(f"scorer references for {len(val_ids)} shots {'read from the cache' if references.from_cache else 'built'}, "
          f"{references_s:.0f} s, {workers} workers", flush=True)

    rng = np.random.default_rng(args.seed)
    arms = [("A", 0, []), ("B", 0, leak_ids)]
    for k in range(args.n_random_arms):
        arms.append(("C", k, sorted(rng.choice(train_ids, size=len(leak_ids), replace=False).tolist()) if leak_ids else []))
    rows, fit_s, score_s = [], [], []
    for arm, seed, removed in arms:
        t0 = time.time()
        keep = ~frames["shot"].isin(set(removed)).to_numpy()
        model = bl.fit_pca_ridge(frames[keep], coefs[keep], basis, params, scaler)
        fit_s.append(time.time() - t0)
        t0 = time.time()
        result = bl.score_local(lambda s: bl.predict(model, val_by_shot[s]), val_ids, args.data_root,
                                frame_stride=args.val_frame_stride, references=references, workers=workers)
        score_s.append(time.time() - t0)
        for subset, mask in (("all", None), ("with_near_repeat", val_with), ("without_near_repeat", ~val_with)):
            terms = bl.score_terms(result.finalize(mask))
            rows.append({"arm": arm, "seed": seed, "subset": subset, "n_removed": len(removed),
                         "n_train_shots": len(train_ids) - len(removed), "n_train_frames": model.n_train_frames, **terms})
        print(f"arm {arm} seed {seed}: S = {rows[-3]['S']:.4f} on all, fit {fit_s[-1]:.0f} s, score {score_s[-1]:.0f} s", flush=True)

    table = pd.DataFrame(rows)
    table.to_parquet(args.out / "leak_test.parquet", index=False)
    timing = {"basis_s": basis_s, "encode_s": encode_s, "references_s": references_s, "workers": workers,
              "references_from_cache": references.from_cache,
              "fit_s_per_arm": float(np.mean(fit_s)), "score_s_per_arm": float(np.mean(score_s))}
    section = render_section(table, args, timing, len(leak_ids), len(train_ids), len(val_ids), int(val_with.sum()))
    with (args.out / "split_report.md").open("a") as f:
        f.write("\n" + section)
    info = run_info(argv, vars(args), timestamp, time.time() - started, params=params.to_dict(), timing=timing,
                    basis_hash=basis.hash(), n_removed=len(leak_ids), removed_arm_b=leak_ids)
    (args.out / "leak_test.json").write_text(json.dumps(info, indent=2, default=float) + "\n")
    print(section)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
