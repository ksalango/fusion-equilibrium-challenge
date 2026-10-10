"""Fingerprints, pairwise distances, nearest neighbours and groups of every DIII-D shot.

Analyses B8a and B8b of ``docs/SPLIT.md``. Reads the frame table of the DIII-D training
and public test splits (inputs only), builds one fingerprint per shot, the distance
between every pair, each shot's nearest training shots and the complete-linkage
groups at the reported thresholds, and compares the flat-top input distributions of
training and test shots. Writes to ``--out`` (``reports/split/``):

- ``shot_neighbours.parquet``: one row per shot with its split, the number of grid
  times in its window and its ``--k`` nearest training shots (id, split, distance).
- ``shot_groups.parquet``: group id per shot at each threshold (``group_005`` ...).
- ``input_distributions.parquet``: the B8a comparison per quantity.
- ``split_report.md``: checks S1 to S4 (``make_split.py`` and ``leak_test.py`` append
  to it), and ``near_repeats.json`` with the run's provenance.

The full distance matrix is written to ``--processed-out``
(``../data/processed/split/distance.npz``) for ``make_split.py`` and ``leak_test.py``.
Test shots are compared and never fitted: the scales come from training frames. Usage::

    python scripts/find_near_repeats.py
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from fusion_eq import frame_table as ft, similarity as sim
from fusion_eq.baseline import THOMSON_INPUTS
from fusion_eq.dataset import DEFAULT_PROCESSED_ROOT, REPO_ROOT, run_info

DEFAULT_OUT = REPO_ROOT / "reports" / "split"
DEFAULT_PROCESSED_OUT = DEFAULT_PROCESSED_ROOT / "split"
SPLITS = ("diii_d_train", "diii_d_public_test")
TRAIN, TEST = SPLITS


def neighbour_table(fp: sim.Fingerprints, dist: sim.PairwiseDistance, split_of: np.ndarray, k: int) -> pd.DataFrame:
    """One row per shot with its k nearest training shots."""
    train_idx = np.flatnonzero(split_of == TRAIN)
    idx, d = sim.nearest_neighbours(dist.distance, k, columns=train_idx)
    if idx.shape[1] < k:                     # fewer training shots than neighbours asked for
        pad = k - idx.shape[1]
        idx = np.pad(idx, ((0, 0), (0, pad)), constant_values=-1)
        d = np.pad(d, ((0, 0), (0, pad)), constant_values=np.nan)
    out = pd.DataFrame({"shot": fp.shots.astype(str), "split": split_of, "n_grid_times": fp.n_times_present})
    for j in range(k):
        hit = idx[:, j] >= 0
        out[f"nn{j + 1}_shot"] = np.where(hit, fp.shots[np.maximum(idx[:, j], 0)].astype(str), "")
        out[f"nn{j + 1}_split"] = np.where(hit, TRAIN, "")
        out[f"nn{j + 1}_distance"] = d[:, j]
    return out


def quantiles(x: np.ndarray) -> dict:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {q: np.nan for q in ("p05", "p25", "p50", "p75", "p95")}
    return dict(zip(("p05", "p25", "p50", "p75", "p95"), np.percentile(x, [5, 25, 50, 75, 95])))


def render_report(index: pd.DataFrame, neighbours: pd.DataFrame, groups: pd.DataFrame, comparison: pd.DataFrame,
                  thresholds: list[float], min_common: int) -> str:
    lines = ["# Split report", "", "Written by `scripts/find_near_repeats.py`; `make_split.py` and `leak_test.py` append.",
             "", "## S1 Fingerprint coverage (review)", "", "| Split | Shots | Fingerprinted | Too short |",
             "| --- | --- | --- | --- |"]
    for split in SPLITS:
        sub = neighbours[neighbours["split"] == split]
        ok = int((sub["n_grid_times"] >= min_common).sum())
        lines.append(f"| {split} | {len(sub)} | {ok} | {len(sub) - ok} |")
    lines += ["", "## S2 Input distributions, training against test (review)", "",
              "Per-shot medians over flat-top frames; largest Kolmogorov-Smirnov statistic first.", "",
              "| Quantity | KS | p-value | Median difference (train IQR) | n train | n test |", "| --- | --- | --- | --- | --- | --- |"]
    for r in comparison.itertuples(index=False):
        lines.append(f"| {r.quantity} | {r.ks:.3f} | {r.p_value:.2e} | {r.median_diff_iqr:+.3f} | {r.n_train} | {r.n_test} |")
    train_nn = neighbours.loc[neighbours["split"] == TRAIN, "nn1_distance"].to_numpy()
    test_nn = neighbours.loc[neighbours["split"] == TEST, "nn1_distance"].to_numpy()
    lines += ["", "## S3 Nearest-neighbour distance among training shots (review)", "",
              "| Shots | p05 | p25 | p50 | p75 | p95 | undefined |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for name, x in (("training (other training shots)", train_nn), ("test (training shots)", test_nn)):
        q = quantiles(x)
        lines.append(f"| {name} | {q['p05']:.4f} | {q['p25']:.4f} | {q['p50']:.4f} | {q['p75']:.4f} | {q['p95']:.4f} | "
                     f"{int((~np.isfinite(x)).sum())} |")
    a, b = train_nn[np.isfinite(train_nn)], test_nn[np.isfinite(test_nn)]
    if a.size > 1 and b.size > 1:
        ks = ks_2samp(a, b)
        lines.append(f"\nKolmogorov-Smirnov statistic between the two: {ks.statistic:.4f} (p = {ks.pvalue:.2e}).")
    lines += ["", "## S4 Near-repeat prevalence (review)", "",
              "| Threshold | Training shots with a near-repeat in training | Test shots with a near-repeat in training | "
              "Groups (all shots) | Shots in a group of 2 or more | Largest group |",
              "| --- | --- | --- | --- | --- | --- |"]
    for t in thresholds:
        col = sim.group_column(t)
        sizes = groups[col].value_counts()
        grouped_shots = int(sizes[sizes >= 2].sum())
        lines.append(f"| {t:.2f} | {np.nanmean(train_nn < t):.3%} | {np.nanmean(test_nn < t):.3%} | {sizes.size} | "
                     f"{grouped_shots} | {int(sizes.max())} |")
    lines += ["", f"Shots in the index: {len(index)}; thresholds reported: {thresholds}; provisional {sim.PROVISIONAL_THRESHOLD}.", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--table-dir", type=Path, default=ft.DEFAULT_TABLE_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--processed-out", type=Path, default=DEFAULT_PROCESSED_OUT)
    parser.add_argument("--k", type=int, default=10, help="nearest training shots kept per shot")
    parser.add_argument("--thresholds", type=float, nargs="+", default=list(sim.THRESHOLDS))
    args = parser.parse_args(argv)

    started = time.time()
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    args.out.mkdir(parents=True, exist_ok=True)
    args.processed_out.mkdir(parents=True, exist_ok=True)

    index = pd.read_parquet(args.table_dir / "shot_index.parquet")
    index = index[index["split"].isin(SPLITS) & index["error"].isna()].copy()
    index["shot"] = index["shot"].astype(str)
    signals = sim.similarity_signals()
    quantities = [*signals, *THOMSON_INPUTS]
    columns = ["shot", "time_ms", "ip_over_peak", *quantities]
    frames = {split: ft.read_frame_table(split, columns, root=args.table_dir) for split in SPLITS}
    for split in SPLITS:
        frames[split]["shot"] = frames[split]["shot"].astype(str)
    print(f"frames read: {', '.join(f'{s} {len(f)}' for s, f in frames.items())}, {time.time() - started:.0f} s", flush=True)

    fp = sim.shot_fingerprints(pd.concat(frames.values(), ignore_index=True), signals)
    scales = sim.signal_scales(frames[TRAIN], signals)
    dist = sim.pairwise_distance(fp, scales)
    print(f"fingerprints and distances for {fp.shots.size} shots, {time.time() - started:.0f} s", flush=True)
    split_of = index.set_index("shot").loc[fp.shots.astype(str), "split"].to_numpy().astype(str)

    neighbours = neighbour_table(fp, dist, split_of, args.k)
    groups = pd.DataFrame({"shot": fp.shots.astype(str), "split": split_of})
    for t in args.thresholds:
        groups[sim.group_column(t)] = sim.group_shots(dist.distance, t)
    print(f"neighbours and groups, {time.time() - started:.0f} s", flush=True)

    medians = {split: sim.flat_top_medians(frames[split], quantities) for split in SPLITS}
    comparison = sim.compare_distributions(medians[TRAIN], medians[TEST])

    neighbours.to_parquet(args.out / "shot_neighbours.parquet", index=False)
    groups.to_parquet(args.out / "shot_groups.parquet", index=False)
    comparison.to_parquet(args.out / "input_distributions.parquet", index=False)
    np.savez(args.processed_out / "distance.npz", shots=fp.shots.astype(str), distance=dist.distance.astype(np.float32),
             n_common_times=dist.n_common_times.astype(np.int16), scales=scales, signals=np.array(signals))
    report = render_report(index, neighbours, groups, comparison, args.thresholds, sim.MIN_COMMON_TIMES)
    (args.out / "split_report.md").write_text(report)
    info = run_info(argv, vars(args), timestamp, time.time() - started, n_shots=int(fp.shots.size),
                    signals=signals, scales=dict(zip(signals, scales.tolist())),
                    fingerprint_times_ms=[float(sim.FINGERPRINT_TIMES_MS[0]), float(sim.FINGERPRINT_TIMES_MS[-1]),
                                          float(sim.FINGERPRINT_TIMES_MS[1] - sim.FINGERPRINT_TIMES_MS[0])],
                    max_gap_ms=sim.MAX_GAP_MS, min_common_times=sim.MIN_COMMON_TIMES,
                    min_common_fraction=sim.MIN_COMMON_FRACTION)
    (args.out / "near_repeats.json").write_text(json.dumps(info, indent=2, default=float) + "\n")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
