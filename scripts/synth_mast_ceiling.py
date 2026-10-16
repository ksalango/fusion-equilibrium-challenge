"""Ceiling of the DIII-D-fitted current basis on synthetic MAST equilibria.

Each equilibrium written by ``synth_mast_equilibria.py`` is run through the target chain as if
it were a labelled MAST frame (``synthetic_mast.evaluate_equilibrium``): coil flux from the
adopted MAST coil model (``sol_half``) with the coil currents FreeGS found, plasma flux,
current share inside the scorer's boundary, its coefficients on the stored 50-component basis
(``targets/v1``), and the map rebuilt by ``current_basis.compose_map`` with no remainder, as on
MAST. Every map is scored with the organizers' local scorer (machine set to MAST) against the
synthetic truth, with the true q95 and betaN, so that differences in S come from the map alone.

Variants (one global flux sign each, as separate submissions would have):

- ``truth``: the true map (scorer self-check, S = 1).
- ``exact_share``: coil flux plus the flux of the exact masked share (what the mask and the
  MAST response keep without any basis).
- ``basis_share``: the share projected onto the basis through the MAST placement; the
  analogue of the DIII-D target ``c = B (s - s_mean)``. The ceiling of the request.
- ``basis_flux``: the coefficients whose flux best fits the plasma flux over the grid (the best
  map 50 coefficients can give).
- ``mean_share``: the basis mean alone (``mean_current_v0`` with an exact coil flux).

Coil-model check: FreeGS's vacuum flux against the coil model's for the same currents, RMS over
the frame's flux range, over all nodes, nodes within one cell of a coil and the rest, and per
coil group (P2 to P6, solenoid) from ``freegs_vacuum_patterns.npz``.

Writes ``ceiling.md``, ``ceiling.json`` and ``per_frame.parquet`` to ``--out``
(``reports/synthetic/mast_v1/``). No MAST equilibrium label is read. Usage::

    python scripts/synth_mast_ceiling.py
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from fusion_eq import current_basis as cb, synthetic_mast as sm, target_build as tb
from fusion_eq.coil_flux import GreenTable, coil_model, near_coil_mask
from fusion_eq.dataset import DEFAULT_PROCESSED_ROOT, REPO_ROOT, default_data_root, run_info

DEFAULT_IN = DEFAULT_PROCESSED_ROOT / "synthetic" / "mast_v1"
DEFAULT_OUT = REPO_ROOT / "reports" / "synthetic" / "mast_v1"
DEFAULT_TARGETS = tb.DEFAULT_TARGET_ROOT / "v1"
DEFAULT_GREEN_DIR = DEFAULT_PROCESSED_ROOT / "green"
DIII_D_CEILING = REPO_ROOT / "reports" / "current_target" / "ceiling.json"
DIII_D_VARIANTS = {"h_symmetric_50": "DIII-D F90 (h at 50: basis, 10-component remainder)",
                   "a_masked": "DIII-D a_masked (exact share, no remainder)"}
TERMS = ("S", "r2_psi", "r2_qb", "one_minus_dlcfs", "consistency")
SHAPE_TERMS = ("r2_R_axis", "r2_Z_axis", "r2_kappa", "r2_tri_top", "r2_tri_bot", "r2_volume", "r2_li")
BINNED_VARIANTS = ("basis_share", "exact_share", "basis_flux")
BINS = {"kappa": (1.4, 1.7, 1.85, 2.0, 2.3), "aspect_ratio": (1.2, 1.45, 1.6, 1.75, 2.1),
        "beta_p": (0.0, 0.2, 0.35, 0.5, 0.8), "li": (0.3, 0.6, 0.8, 1.0, 1.4)}
COIL_GROUPS = {"P2": "p2", "P3": "p3", "P4": "p4", "P5": "p5", "P6": "p6", "solenoid": "sol"}

# Written into every report: how the synthetic set departs from the request and from real MAST.
FLAGS = (
    "Magnetic axis range. FreeGS cannot impose R_axis; the design samples the boundary centre (0.80 to 0.92 m, "
    "clipped so the inner edge stays at least 0.22 m and the outer midplane at most 1.48 m out) and the achieved "
    "R_axis is reported under Coverage against the requested 0.85 to 1.0 m.",
    "Lower single null. A single tied lower X-point with the upper tip as an ordinary boundary point gave "
    "elongations near 1.5 and negative upper triangularity, or no convergence, within MAST's coil currents. LSN "
    "targets are therefore lower-dominant near-double-nulls: the upper X-point is constrained 2 to 10 cm above the "
    "mirror of the lower one and not tied to the boundary flux. Truly single-null shapes are not covered.",
    "Coil currents. P3 and P6 are held at zero and P2, P4 and P5 are bounded to the range of the MAST test inputs; "
    "without the bounds the first double null used P4 at -970 kA-turn (test range -288 to -1) and P6 at +550. The "
    "set describes plasmas MAST's coils can make, not every shape in the requested ranges; achieved shapes are "
    "reported and binned.",
    "Current profiles. Two-parameter FreeGS profiles (ConstrainBetapIp, alpha_m = 1, alpha_n 0.15 to 1.0, "
    "calibrated against the scorer's li, which is about 0.9 times FreeGS's li1 and none of FreeGS's li1, li2, li3; "
    "the scorer's docstring calls its li ITER li(2) while its formula normalises by the poloidal perimeter). No "
    "pedestal or edge current, no hollow ramp-up profile, no limited (inner-wall) plasma. Each of these would test "
    "the basis harder; a profile family through ProfilesPprimeFfprime with an edge term is the next step if the "
    "ceiling here is high.",
    "No passive currents. The synthetic truth is coils plus plasma exactly, so the chain without a remainder is "
    "complete here; real MAST maps carry vessel currents (largest in the ramp-up) that the chain sets to zero. "
    "This ceiling is an upper bound in that respect and says nothing about the size of MAST's remainder.",
    "Coil-model check. FreeGS's machine is built from the same geometry and the same convention (sol_half), so the "
    "check confirms the implementation (turns, sharing, signs, the solenoid factor applied once) and the "
    "discretisation; it cannot say whether half the feed current is the right physics.",
    "Projection. basis_share fits the placed basis to the native share by least squares, the analogue of the "
    "DIII-D target. The model predicts coefficients from inputs, so a high ceiling here places the MAST "
    "shortfall of the submissions in the input-to-coefficient map (transfer), not in the basis.",
    "Software. FreeGS 0.8.2 (PyPI) fails under NumPy 2.5 in find_critical; FreeGS is pinned to commit 9fbec20 "
    "(30 December 2025), which carries the upstream fix. FreeGS's own q fails under the same NumPy, so q95 comes "
    "from scalars.q95_from_map on the native map.",
)

_W: dict = {}


def log(message: str) -> None:
    print(message, flush=True)


def _init(targets_dir: str, green_dir: str, coil_model_name: str, in_dir: str, data_root: str) -> None:
    import warnings

    warnings.simplefilter("ignore")
    objects = tb.TargetObjects.load(targets_dir)
    table = GreenTable.load(Path(green_dir) / "green_mast.npz")
    geometry, _, _, _ = sm.load_geometry(data_root)
    with np.load(Path(in_dir) / "freegs_vacuum_patterns.npz", allow_pickle=False) as f:
        fg = {"patterns": f["patterns"], "columns": [str(c) for c in f["columns"]]}
    _W.update(objects=objects, table=table, scales=coil_model(sm.MACHINE, table.columns, coil_model_name),
              response=cb.machine_response(sm.MACHINE, green_dir), freegs=fg,
              near=near_coil_mask(geometry, table.grid_r, table.grid_z))


def coil_check(rec: dict, coil: np.ndarray, currents: np.ndarray) -> dict:
    """FreeGS's vacuum flux against the coil model's, RMS over the frame's flux range."""
    table, scales, fg, near = _W["table"], _W["scales"], _W["freegs"], _W["near"]
    diff = rec["psi_vacuum"] - coil
    span = float(rec["psi"].max() - rec["psi"].min())
    out = {"vac_rms_all": sm.rms_over_range(diff, rec["psi"]),
           "vac_rms_far": float(np.sqrt(np.mean(diff[~near] ** 2)) / span),
           "vac_rms_near": float(np.sqrt(np.mean(diff[near] ** 2)) / span) if near.any() else np.nan}
    order = [fg["columns"].index(c) for c in table.columns]
    fg_parts = fg["patterns"][order] * currents[:, None, None]
    ours = table.patterns * (currents * scales.vector(table.columns))[:, None, None]
    out["vac_pattern_sum_error"] = sm.rms_over_range(fg_parts.sum(axis=0) - rec["psi_vacuum"], rec["psi"])
    for group, key in COIL_GROUPS.items():
        idx = [j for j, c in enumerate(table.columns) if f"_{key}" in c]
        d = (fg_parts[idx] - ours[idx]).sum(axis=0)
        out[f"vac_rms_{group}_all"] = sm.rms_over_range(d, rec["psi"])
        out[f"vac_rms_{group}_far"] = float(np.sqrt(np.mean(d[~near] ** 2)) / span)
    return out


def _evaluate(path: str) -> dict:
    """Pass 1 in a worker: one equilibrium through the chain."""
    rec = sm.load_record(path)
    objects, table, scales = _W["objects"], _W["table"], _W["scales"]
    if not (np.allclose(rec["grid_r"], table.grid_r) and np.allclose(rec["grid_z"], table.grid_z)):
        raise ValueError(f"{path}: grid differs from the Green's table")
    currents = np.array([rec["currents_ka"][rec["columns"].index(c)] for c in table.columns])
    coil = table.flux(currents, scales)
    ev = sm.evaluate_equilibrium(rec["psi"], coil, rec["ip_ka"], table.grid_r, table.grid_z, objects.current_basis,
                                 objects.flux_bases[sm.MACHINE], objects.placements[sm.MACHINE], _W["response"],
                                 exclude=_W["near"])
    truth = sm.synthetic_truth(rec["psi"], rec["q95"], rec["beta_n"])
    contour = truth.reference[0][0]
    row = {"index": int(Path(path).stem.split("_")[1]), "topology": rec["target"]["topology"], "profile": rec["profile"],
           "ip_ka": rec["ip_ka"], "beta_p": rec["beta_p"], "li2_freegs": rec["li"], "li1_freegs": rec["li1"], "q95": rec["q95"], "beta_n": rec["beta_n"],
           "aspect_ratio": sm.contour_aspect_ratio(contour), "boundary_found_truth": contour is not None,
           **{f"target_{k}": v for k, v in rec["target"].items() if k not in ("index", "topology")},
           **{f"current_{c.split('_')[1]}": v for c, v in zip(rec["columns"], rec["currents_ka"])},
           **ev.diagnostics, **coil_check(rec, coil, currents)}
    return {"row": row, "maps": ev.maps, "truth": truth,
            "coefficients": {k: v.tolist() for k, v in ev.coefficients.items()}}


def _score(truth: sm.SyntheticTruth, maps: dict, signs: dict, means: tuple) -> dict:
    """Pass 2 in a worker: the scorer's part of every variant and the per-frame metrics."""
    parts, metrics = {}, {}
    for v in sm.VARIANTS:
        parts[v] = sm.score_part(truth, maps[v], signs[v], means)
        metrics[v] = sm.frame_metrics(truth, maps[v], signs[v])
    return {"parts": parts, "metrics": metrics}


def _pool(workers: int, initargs: tuple):
    return ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"), initializer=_init, initargs=initargs)


def quantiles(values) -> dict:
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if not v.size:
        return {"n": 0}
    return {"n": int(v.size), "p5": float(np.percentile(v, 5)), "median": float(np.median(v)),
            "p95": float(np.percentile(v, 95)), "min": float(v.min()), "max": float(v.max())}


def fmt(x, digits: int = 4) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{digits}f}"


def diii_d_reference() -> dict:
    if not DIII_D_CEILING.is_file():
        return {}
    scores = json.loads(DIII_D_CEILING.read_text()).get("scores", {})
    return {label: scores[k] for k, label in DIII_D_VARIANTS.items() if k in scores}


def write_report(out: Path, frame: pd.DataFrame, overall: dict, binned: dict, build: dict, info: dict) -> None:
    lines = ["# Ceiling of the current basis on synthetic MAST equilibria (mast_v1)", ""]
    lines += [f"Run of {info['timestamp']}, repository commit {info['repo_commit']}. {len(frame)} converged FreeGS "
              f"equilibria of {build.get('n_targets', 'n/a')} targets ({build.get('by_topology', {})}); FreeGS "
              f"{build.get('meta', {}).get('freegs', {})}; coil model {info['arguments']['coil_model']}; basis "
              f"{info['basis']}. No MAST equilibrium label read. True q95 and betaN in every variant.", ""]
    lines += ["## Score terms", "", "One global flux sign per variant. DIII-D rows from `reports/current_target/ceiling.json` "
              "(validation frames of the working lists, true q95 and betaN).", ""]
    head = "| Variant | " + " | ".join(TERMS) + " | " + " | ".join(t[3:] for t in SHAPE_TERMS) + " |"
    lines += [head, "|" + " --- |" * (1 + len(TERMS) + len(SHAPE_TERMS))]
    for name, terms in overall.items():
        lines.append(f"| {name} | " + " | ".join(fmt(terms.get(t)) for t in (*TERMS, *SHAPE_TERMS)) + " |")
    lines += ["", "## Per-frame boundary distance and share diagnostics", "",
              "| Variant | D_LCFS median (cm) | p95 (cm) | predicted boundary missing |", "| --- | --- | --- | --- |"]
    for v in sm.VARIANTS:
        d = frame[f"{v}_d_lcfs_m"] * 100
        lines.append(f"| {v} | {fmt(d.median(), 2)} | {fmt(d.quantile(0.95), 2)} | "
                     f"{int((frame[f'{v}_d_lcfs_rel'] == 1.0).sum())} |")
    lines += ["", "| Quantity | p5 | median | p95 |", "| --- | --- | --- | --- |"]
    for k, label in (("share_total", "Share total (masked current over Ip)"),
                     ("outside_abs_share", "Absolute current outside the boundary, share of the total"),
                     ("outside_abs_share_kept", "The same without nodes within one cell of a coil"),
                     ("outside_footprint_abs_share", "Absolute share outside the placed reference lattice"),
                     ("share_rel_error", "Relative error of the projected share (native grid)")):
        q = quantiles(frame[k])
        lines.append(f"| {label} | {fmt(q.get('p5'))} | {fmt(q.get('median'))} | {fmt(q.get('p95'))} |")
    lines += ["", "## By shape and profile (basis_share unless stated)", "",
              "Pooled R2 within a bin uses that bin's own spread, so a term whose quantity varies little inside the "
              "bin (kappa inside a kappa bin) reads lower than overall; the boundary distance in cm compares bins "
              "directly.", ""]
    for key, table in binned.items():
        lines += [f"### {key}", "", "| Bin | Variant | n | S | r2_psi | 1 - D_LCFS | consistency | D_LCFS median (cm) |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for row in table:
            lines.append(f"| {row['bin']} | {row['variant']} | {row['n']} | {fmt(row.get('S'))} | {fmt(row.get('r2_psi'))} | "
                         f"{fmt(row.get('one_minus_dlcfs'))} | {fmt(row.get('consistency'))} | {fmt(row['d_lcfs_cm'], 2)} |")
        lines.append("")
    lines += ["## Coil-model check (FreeGS vacuum flux against the coil model, same currents)", "",
              "RMS of the difference over the frame's flux range. Near: nodes within one grid cell of a coil element "
              "(FreeGS has no self-distance floor there).", "", "| Nodes | p5 | median | p95 | max |",
              "| --- | --- | --- | --- | --- |"]
    for k, label in (("vac_rms_all", "all"), ("vac_rms_far", "away from coils"), ("vac_rms_near", "near coils"),
                     *[(f"vac_rms_{g}_far", f"{g}, away from coils") for g in COIL_GROUPS],
                     ("vac_pattern_sum_error", "FreeGS record vs its own column patterns")):
        q = quantiles(frame[k])
        lines.append(f"| {label} | {fmt(q.get('p5'), 6)} | {fmt(q.get('median'), 6)} | {fmt(q.get('p95'), 6)} | {fmt(q.get('max'), 6)} |")
    lines += ["", "## Coverage of the target ranges (achieved)", "", "| Quantity | requested | p5 | median | p95 | min | max |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    requested = {"ip_ka": "400-900", "true_kappa": "1.6-2.2", "true_R_axis": "0.85-1.0", "beta_p": "0.1-0.6",
                 "true_li": "0.5-1.0", "aspect_ratio": "a 0.5-0.62", "true_tri_top": "", "true_tri_bot": "",
                 "true_volume": "", "q95": "", "beta_n": ""}
    for k, req in requested.items():
        col = f"truth_{k[5:]}" if k.startswith("true_") else k
        q = quantiles(frame[col]) if col in frame else {"n": 0}
        lines.append(f"| {k.replace('true_', '')} | {req} | {fmt(q.get('p5'), 3)} | {fmt(q.get('median'), 3)} | "
                     f"{fmt(q.get('p95'), 3)} | {fmt(q.get('min'), 3)} | {fmt(q.get('max'), 3)} |")
    lines += ["", "Coil currents found (kA-turn; solenoid kA, fixed):", "", "| Column | p5 | median | p95 |", "| --- | --- | --- | --- |"]
    for c in [c for c in frame.columns if c.startswith("current_")]:
        q = quantiles(frame[c])
        lines.append(f"| {c[8:]} | {fmt(q.get('p5'), 1)} | {fmt(q.get('median'), 1)} | {fmt(q.get('p95'), 1)} |")
    lines += ["", "## Method notes and flags", ""] + [f"{k}. {t}" for k, t in enumerate(FLAGS, 1)] + [
        "", "## Reading", "", "Left for the review of the full run.", ""]
    (out / "ceiling.md").write_text("\n".join(lines))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--in", dest="in_dir", type=Path, default=DEFAULT_IN, help="folder of synth_mast_equilibria.py")
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS, help="target build holding the basis")
    parser.add_argument("--green-dir", type=Path, default=DEFAULT_GREEN_DIR)
    parser.add_argument("--data-root", type=Path, default=default_data_root(), help="raw data root (geometry for the near-coil mask)")
    parser.add_argument("--coil-model", default="sol_half")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 3) - 2))
    args = parser.parse_args(argv)
    t_start = time.time()
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")

    paths = [str(p) for p in sm.list_records(args.in_dir)]
    if not paths:
        raise FileNotFoundError(f"no equilibrium record in {args.in_dir}")
    build = json.loads((args.in_dir / "build.json").read_text()) if (args.in_dir / "build.json").is_file() else {}
    rec_model = build.get("meta", {}).get("coil_model")
    if rec_model and rec_model != args.coil_model:
        raise ValueError(f"equilibria were built under {rec_model}, the ceiling asked for {args.coil_model}")
    initargs = (str(args.targets), str(args.green_dir), args.coil_model, str(args.in_dir), str(args.data_root))
    log(f"{len(paths)} equilibria from {args.in_dir}; pass 1 (chain) with {args.workers} workers")
    with _pool(args.workers, initargs) as pool:
        first = list(pool.map(_evaluate, paths, chunksize=4))
    truths = [r["truth"] for r in first]
    means = sm.means_of(truths)
    signs = {v: sm.global_psi_sign(truths, [r["maps"][v] for r in first], means[0]) for v in sm.VARIANTS}
    log(f"Pass 2 (scorer), flux signs {signs}")
    with _pool(args.workers, initargs) as pool:
        second = list(pool.map(_score, truths, [r["maps"] for r in first], [signs] * len(first), [means] * len(first),
                               chunksize=4))

    stats = [t.stats for t in truths]
    overall = {}
    for label, terms in diii_d_reference().items():
        overall[label] = terms
    for v in sm.VARIANTS:
        overall[f"MAST synthetic {v}"] = sm.finalize_parts([s["parts"][v] for s in second], stats, signs[v])
    self_check = overall["MAST synthetic truth"]["S"]
    if not np.isclose(self_check, 1.0):
        log(f"WARNING: the truth variant scores S {self_check}, not 1")

    rows = []
    for r, s in zip(first, second):
        row = dict(r["row"])
        for v in sm.VARIANTS:
            for k, val in s["metrics"][v].items():
                if k.startswith("true_"):
                    row[f"truth_{k[5:]}"] = val
                else:
                    row[f"{v}_{k}"] = val
        rows.append(row)
    frame = pd.DataFrame(rows)

    binned = {}
    groups = {"topology": frame["topology"].to_numpy(dtype=object),
              "kappa": sm.bin_labels(frame["truth_kappa"], BINS["kappa"]),
              "aspect_ratio": sm.bin_labels(frame["aspect_ratio"], BINS["aspect_ratio"]),
              "beta_p": sm.bin_labels(frame["beta_p"], BINS["beta_p"]),
              "li": sm.bin_labels(frame["truth_li"], BINS["li"])}
    for key, labels in groups.items():
        table = []
        for label in sorted(set(labels)):
            mask = labels == label
            for v in BINNED_VARIANTS:
                terms = sm.finalize_parts([s["parts"][v] for s in second], stats, signs[v], mask)
                table.append({"bin": label, "variant": v, "n": int(mask.sum()),
                              "d_lcfs_cm": float(np.nanmedian(frame.loc[mask, f"{v}_d_lcfs_m"])) * 100, **terms})
        binned[key] = table

    args.out.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.out / "per_frame.parquet", index=False)
    objects = tb.TargetObjects.load(args.targets)
    info = run_info(argv, vars(args), timestamp, round(time.time() - t_start, 1), n_equilibria=len(paths),
                    basis={"n_components": objects.current_basis.n_components,
                           "file_sha256": tb.sha256_of(args.targets / tb.OBJECT_FILES["current_basis"])},
                    psi_signs=signs, synthetic_build=build.get("meta", {}))
    write_report(args.out, frame, overall, binned, build, info)
    (args.out / "ceiling.json").write_text(json.dumps({**info, "scores": overall, "binned": binned}, indent=1, default=float) + "\n")
    for name in overall:
        t = overall[name]
        log(f"{name}: S {fmt(t.get('S'))}, r2_psi {fmt(t.get('r2_psi'))}, 1 - D_LCFS {fmt(t.get('one_minus_dlcfs'))}, "
            f"consistency {fmt(t.get('consistency'))}")
    log(f"Wrote {args.out} ({time.time() - t_start:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
