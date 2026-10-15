"""Profiles a target build: checks Y1 to Y16 of ``docs/TARGET.md``.

Reads the target tables and fitted objects of ``scripts/build_targets.py`` (with
``build.json``), the frame table, the plasma responses, the stored maps of a fixed-seed sample
of training shots (Y2, Y5), the MAST coil geometry (Y10, Y11), the ceiling report
(``reports/current_target/ceiling.json``, Y15) and, for Y16, the fitted ``baseline_v0`` scalar
ridge, the scorer references of the validation list and the inputs of the public test shots.
No MAST equilibrium label is read: every MAST quantity comes from geometry, machine constants
and test inputs.

Writes ``<report-root>/v<version>/target_report.md`` and ``target_summary.json``. Usage::

    python scripts/profile_targets.py
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from fusion_eq import (baseline as bl, coil_pattern as cp, current_basis as cb, current_target as ct, dataset,
                       frame_table as ft, lattice as lt, predictors, split as sp, starter_kit, target_build as tb,
                       target_checks as tc, targets)
from fusion_eq.coil_flux import GreenTable, adopted_scales
from fusion_eq.coil_scales import REFERENCE_PHASES
from fusion_eq.dataset import default_data_root, run_info
from fusion_eq.machines import machine_config

DEFAULT_SPLIT = dataset.REPO_ROOT / "splits" / "split_v1.json"
DEFAULT_REPORT_ROOT = dataset.REPO_ROOT / "reports" / "targets"
DEFAULT_CEILING = dataset.REPO_ROOT / "reports" / "current_target" / "ceiling.json"
# Reference values of F79 for Y12 to Y14.
F79 = {"remainder_share": {"all": 0.0105, "rising": 0.0193, "flat_top": 0.0093, "late": 0.0143},
       "explained": {5: 0.9978, 10: 0.9998, 20: 1.0000}, "ridge": {"train": 0.8885, "validation": 0.8842},
       "f": {"S": 0.9982, "consistency": 0.9944, "r2_volume": 0.998}, "g": {"S": 0.9973, "consistency": 0.9911, "r2_volume": 0.991}}
SCORE_KEYS = ("S", "r2_psi", "r2_qb", "one_minus_dlcfs", "consistency", "r2_R_axis", "r2_Z_axis", "r2_kappa", "r2_tri_top",
              "r2_tri_bot", "r2_volume", "r2_li")
DERIVED = ("R_axis", "Z_axis", "kappa")
# Size of the found boundary (tc.boundary_extent), in metres.
EXTENT = ("inner_R", "outer_R", "width", "height")
Y16_LABELS = {"R_axis": "R_axis (m)", "Z_axis": "Z_axis (m)", "kappa": "kappa", "inner_R": "Inner R (m)",
              "outer_R": "Outer R (m)", "width": "Width (m)", "height": "Height (m)"}


def log(message: str) -> None:
    print(message, flush=True)


def display_path(path: Path | str) -> str:
    return os.path.relpath(Path(path).resolve(), dataset.REPO_ROOT)


def strided_rows(split_name: str, ids, columns, table_dir: Path, stride: int) -> dict[str, pd.DataFrame]:
    rows = ft.read_frame_table(split_name, list(dict.fromkeys(["shot", "frame", *columns])), shots=list(ids), root=table_dir)
    rows["shot"] = rows["shot"].astype(str)
    return {s: g.sort_values("frame").iloc[::stride].reset_index(drop=True) for s, g in rows.groupby("shot", sort=True)}


# --------------------------------------------------------------------------------- build
def check_y1(tables: dict[str, pd.DataFrame], split: dict, record: dict, args) -> tuple[tc.Check, dict]:
    out, ok = {}, True
    for name, table in tables.items():
        keys = strided_rows(args.split_name, sorted(split[name]), [], args.table_dir, record["stride"])
        expected = pd.concat(list(keys.values()), ignore_index=True) if keys else pd.DataFrame(columns=["shot", "frame"])
        same = len(expected) == len(table) and np.array_equal(expected["shot"].to_numpy(), table["shot"].astype(str).to_numpy()) \
            and np.array_equal(expected["frame"].to_numpy(), table["frame"].to_numpy())
        out[name] = {"rows": int(len(table)), "frame_table_rows": int(len(expected)), "keys_match": bool(same)}
        ok &= same
    summary = "; ".join(f"{n} {v['rows']} rows, keys {'match' if v['keys_match'] else 'differ'}" for n, v in out.items())
    return tc.Check("Y1", "Rows and keys", "pass" if ok and out else "fail", summary or "no table"), out


def sample_frames(fit_rows: dict[str, pd.DataFrame], n_shots: int, n_frames: int, seed: int) -> dict[str, np.ndarray]:
    """Up to ``n_frames`` strided frames of up to ``n_shots`` training shots in a fixed-seed order."""
    shots = list(fit_rows)
    order = np.random.default_rng(seed).permutation(len(shots))[:n_shots]
    per_shot = max(1, int(np.ceil(n_frames / max(1, len(order)))))
    chosen, total = {}, 0
    for k in order:
        s = shots[k]
        n = min(len(fit_rows[s]), per_shot, n_frames - total)
        if n <= 0:
            break
        chosen[s] = np.linspace(0, len(fit_rows[s]) - 1, n).round().astype(int)
        total += n
    return chosen


def check_y2_y5(objects: tb.TargetObjects, fit_rows, chosen, d3d: dict, args, stride: int) -> tuple[tc.Check, tc.Check, dict]:
    table, scales, response = d3d["table"], d3d["scales"], d3d["response"]
    envelope, scorer = starter_kit.envelope_mask(machine_config("DIII-D")), starter_kit.scoring()
    basis, flux = objects.current_basis, objects.flux_bases["DIII-D"]
    chain, compose, shares = 0.0, 0.0, []
    for s, idx in chosen.items():
        rows = fit_rows[s].iloc[idx].reset_index(drop=True)
        psi = tb.read_truth(s, args.data_root, args.split_name, stride)["psi"][idx].astype(np.float64)
        share, rem_target = tb.split_frames(psi, rows, table, scales, response, envelope, scorer)
        ip = rows["ip__mean"].to_numpy(dtype=np.float64)
        sign = targets.sign_of_ip(ip)[:, None, None]
        coil = targets.coil_flux_of_frames(table, rows, scales, "DIII-D")
        phys = targets.to_physical(psi, ip, "DIII-D")
        flux_s = np.nan_to_num(ip)[:, None, None] * 1e3 * cb.cell_flux(response, share.share)
        rng = (psi.max(axis=(1, 2)) - psi.min(axis=(1, 2)))[:, None, None]
        chain = max(chain, float((np.abs(coil + flux_s + sign * rem_target - phys) / rng).max()))
        c = basis.transform(share.share)
        remainder_for_c = phys - coil - flux.flux(ip, c)
        composed = cb.compose_map("DIII-D", coil, ip, c, flux, remainder_for_c)
        compose = max(compose, float((np.abs(composed - psi) / rng).max()))
        shares.append(share.share)
    shares = np.concatenate(shares) if shares else np.zeros((0, *basis.shape))
    mirror = tc.mirror_errors(basis, shares)
    n = int(len(shares))
    y2 = tc.Check("Y2", "Bookkeeping", "pass" if n and chain <= 1e-10 and compose <= 1e-10 else "fail",
                  f"{n} frames of {len(chosen)} training shots: coil + Ip G s + r against psi_phys {chain:.2e}, compose_map "
                  f"against the stored map {compose:.2e} of the frame's flux range (rule 1e-10)")
    y5 = tc.Check("Y5", "Mirror", "pass" if n and mirror["max_relative"] <= 1e-8 and mirror["twice_identity"] else "fail",
                  f"{n} frames: coefficients of the flipped share against c times p {mirror['max_relative']:.2e} of the largest "
                  f"coefficient (rule 1e-8); mirroring twice {'is' if mirror['twice_identity'] else 'is not'} the identity")
    return y2, y5, {"n_frames": n, "chain_max": chain, "compose_max": compose, "mirror": mirror}


def check_y3(tables: dict[str, pd.DataFrame]) -> tuple[tc.Check, dict]:
    out = {}
    for name, t in tables.items():
        found = t["boundary_found"].to_numpy(dtype=bool)
        out[name] = {"all": tc.quantiles(t.loc[found, "share_total"], (1, 5, 50, 95, 99)), "no_boundary": int((~found).sum()),
                     **{p: tc.quantiles(t.loc[found & (t["phase"] == p).to_numpy(), "share_total"], (5, 50, 95)) for p in REFERENCE_PHASES}}
    medians = [v["all"]["q50"] for v in out.values()]
    ok = bool(medians) and all(0.99 <= m <= 1.005 for m in medians)
    summary = "; ".join(f"{n} median {v['all']['q50']:.4f} (5th to 95th {v['all']['q5']:.4f} to {v['all']['q95']:.4f}), "
                        f"{v['no_boundary']} frames without a boundary" for n, v in out.items())
    return tc.Check("Y3", "Share total", "pass" if ok else "fail", summary + " (rule: median within 0.99 to 1.005)"), out


def check_y6(tables: dict[str, pd.DataFrame]) -> tuple[tc.Check, dict]:
    out = {}
    for name, t in tables.items():
        found = t["boundary_found"].to_numpy(dtype=bool)
        out[name] = {kind: {"all": tc.quantiles(t.loc[found, col]),
                            **{p: tc.quantiles(t.loc[found & (t["phase"] == p).to_numpy(), col]) for p in REFERENCE_PHASES}}
                     for kind, col in (("original", "recon_error"), ("mirror", "recon_error_mirror"))}
    summary = "; ".join(f"{n} median {v['original']['all']['q50']:.4f} (mirrors {v['mirror']['all']['q50']:.4f})" for n, v in out.items())
    return tc.Check("Y6", "Reconstruction", "review", "relative error of the share through the basis: " + summary), out


# ------------------------------------------------------------------------- placement, MAST
def reconstructed_shares(objects: tb.TargetObjects, table: pd.DataFrame, n: int, seed: int) -> np.ndarray:
    c_columns = [f"c_{j:02d}" for j in range(objects.current_basis.n_components)]
    usable = table.loc[table["boundary_found"].to_numpy(dtype=bool) & table["passes_filter"].to_numpy(dtype=bool)]
    rows = usable.iloc[np.sort(np.random.default_rng(seed).permutation(len(usable))[:n])]
    return objects.current_basis.inverse(rows[c_columns].to_numpy(dtype=np.float64))


def check_y7_y8(objects, shares: np.ndarray, responses: dict) -> tuple[tc.Check, tc.Check, dict]:
    basis = objects.current_basis
    c = basis.transform(shares)
    y7 = {m: tc.flux_basis_error(basis, objects.flux_bases[m], responses[m], objects.placements[m], c) for m in tb.MACHINES}
    place = objects.placements["MAST"]
    mean = basis.mean.reshape(basis.shape)
    y8 = {"mean": tc.placement_balance(place, mean), "shares": tc.placement_balance(place, shares),
          "border_weight_of_mean": objects.flux_bases["MAST"].mean_border_weight}
    c7 = tc.Check("Y7", "Flux basis", "pass" if all(v <= 1e-10 for v in y7.values()) else "fail",
                  f"{len(c)} reconstructed training shares: phi_mean + Phi c against G P s, " + ", ".join(
                      f"{m} {v:.2e}" for m, v in y7.items()) + " of the largest value (rule 1e-10)")
    worst = max(y8["mean"]["max_imbalance"], y8["shares"]["max_imbalance"])
    c8 = tc.Check("Y8", "Conservation", "pass" if worst <= 1e-10 else "fail",
                  f"placement on MAST: placed total plus the weight outside the grid against the total, mean share and "
                  f"{y8['shares']['n']} reconstructed shares, worst {worst:.2e} (rule 1e-10); weight outside the MAST grid, "
                  f"median {y8['shares']['outside_share']['q50']:.2e}, 95th {y8['shares']['outside_share']['q95']:.2e}; "
                  f"mean share on MAST border cells {y8['border_weight_of_mean']:.2e}")
    return c7, c8, {"Y7": y7, "Y8": y8}


def check_y9(responses: dict, step: int, record: dict) -> tuple[tc.Check, dict]:
    """The response against the filament Green's function (rule of F89) and the ring guard of
    the fit sample, read from the build record. Without the guard in the record the status is
    review."""
    guards = record.get("sample", {}).get("border_rings", {})
    out = {}
    for m, r in responses.items():
        measured = tc.response_against_filaments(r, tc.interior_sources((r.grid_z.size, r.grid_r.size), step))
        guard = guards.get(m)
        out[m] = {**measured, "thresholds": {**tc.Y9_THRESHOLDS[m], "symmetry": tc.Y9_SYMMETRY, "flip": tc.Y9_FLIP,
                                             "ring_share": tc.Y9_RING_SHARE},
                  "ring_guard": guard, "response_pass": tc.y9_status(m, measured),
                  "pass": tc.y9_status(m, measured) and bool(guard and guard["pass"])}
    parts = []
    for m, v in out.items():
        g = v["ring_guard"]
        guard_text = (f"share of the fit sample's current in the first {g['rings']} interior rings, largest {g['max']:.1e} "
                      f"(rule {tc.Y9_RING_SHARE:g}, {g['n_frames']} frames)" if g else "ring guard not in the build record; rebuild")
        parts.append(f"{m}: beyond 5 cells {v['beyond_5']:.2e} (rule {v['thresholds']['beyond_5']:g}), beyond 10 cells "
                     f"{v['beyond_10']:.2e} (rule {v['thresholds']['beyond_10']:g}), symmetry outside the first "
                     f"{v['symmetry_rings_excluded']} interior rings {v['symmetry']:.2e} (rule {tc.Y9_SYMMETRY:g}; all sources "
                     f"{v['symmetry_all_sources']:.2e}), Z flip {v['flip']:.1e}; {guard_text}")
    if all(v["pass"] for v in out.values()):
        status = "pass"
    elif all(v["response_pass"] and (v["ring_guard"] is None or v["ring_guard"]["pass"]) for v in out.values()):
        status = "review"
    else:
        status = "fail"
    return tc.Check("Y9", "Response", status, "; ".join(parts)), out


def landing_cases(objects, shares: np.ndarray, mast_grid: GreenTable, d3d_grid: GreenTable, geometry, chunk: int = 2000) -> dict:
    """Y10 at the MAST constants of ``machines.py`` and Y11 at moved constants."""
    cfg = machine_config("MAST")
    r0, a0 = cfg.major_radius.value, cfg.minor_radius.value
    cases = {"nominal": (r0, a0), "R0 +5%": (1.05 * r0, a0), "R0 -5%": (0.95 * r0, a0), "a0 +5%": (r0, 1.05 * a0),
             "a0 -5%": (r0, 0.95 * a0), "Morris 1998 (0.70, 0.50)": (0.70, 0.50)}
    reference = lt.reference_lattice(d3d_grid.grid_r, d3d_grid.grid_z)
    mean = objects.current_basis.mean.reshape(objects.current_basis.shape)
    out = {}
    for name, (r, a) in cases.items():
        native = lt.native_lattice("MAST", mast_grid.grid_r, mast_grid.grid_z, r, a)
        place = lt.placement("MAST", reference, native, r, a)
        occupancy = np.zeros((mast_grid.grid_z.size, mast_grid.grid_r.size))
        for start in range(0, len(shares), chunk):
            occupancy += tc.support(place.apply(shares[start:start + chunk])).sum(axis=0)
        envelope = occupancy / max(1, len(shares)) >= 0.01
        placed_mean = place.apply(mean)
        out[name] = {"r0_m": r, "a0_m": a, "retained_mean": float(placed_mean.sum() / mean.sum()),
                     "envelope": tc.landing(placed_mean, envelope, mast_grid.grid_r, mast_grid.grid_z, geometry),
                     "mean_support": tc.landing(placed_mean, tc.support(placed_mean)[0], mast_grid.grid_r, mast_grid.grid_z, geometry)}
    return out


def landing_line(name: str, v: dict) -> str:
    e = v["envelope"]
    return (f"{name}: centroid R {e['centroid_r_m']:.3f} m, Z {e['centroid_z_m']:+.3f} m; envelope R {e['inner_r_m']:.3f} to "
            f"{e['outer_r_m']:.3f} m, height {e['height_m']:.3f} m; nearest coil {e['coil_distance_m']:.3f} m, grid edge "
            f"{e['edge_distance_m']:.3f} m; coil elements inside {e.get('n_elements_inside', 0)}")


# ----------------------------------------------------------------------------- remainder
def check_y12_to_y14(tables: dict[str, pd.DataFrame], objects, record: dict) -> tuple[list[tc.Check], dict]:
    y12 = {}
    for name, t in tables.items():
        found = t["boundary_found"].to_numpy(dtype=bool)
        y12[name] = {"all": tc.quantiles(t.loc[found, "remainder_rms_over_range"]),
                     **{p: tc.quantiles(t.loc[found & (t["phase"] == p).to_numpy(), "remainder_rms_over_range"]) for p in REFERENCE_PHASES}}
    cum = objects.remainder_model.explained_cumulative
    y13 = {k: float(cum[k - 1]) for k in (5, 10, 20) if k <= len(cum)}
    y14 = {w: dict(r.scores) for w, r in objects.remainder_model.ridges.items()}
    checks = [
        tc.Check("Y12", "Remainder size", "review", "median RMS over the flux range: " + "; ".join(
            f"{n} {v['all']['q50']:.4f} (" + ", ".join(f"{p} {v[p]['q50']:.4f}" for p in REFERENCE_PHASES) + ")" for n, v in y12.items())
            + f"; F79 {F79['remainder_share']['all']} (rising {F79['remainder_share']['rising']}, flat-top "
              f"{F79['remainder_share']['flat_top']}, late {F79['remainder_share']['late']})"),
        tc.Check("Y13", "Remainder components", "review", "variance held: " + ", ".join(f"{k} {v:.4f} (F79 {F79['explained'][k]})"
                                                                                       for k, v in y13.items())),
        tc.Check("Y14", "Remainder ridge", "review", "; ".join(
            f"{w}: training {v.get('train', float('nan')):.4f}, validation {v.get('validation', float('nan')):.4f}" for w, v in y14.items())
            + f" (F79: {F79['ridge']['train']}, {F79['ridge']['validation']}); default window {objects.remainder_model.default_window}; "
              "the 50 ms ridge keeps the toroidal field's 10 ms rate"),
    ]
    return checks, {"Y12": y12, "Y13": y13, "Y14": y14}


# ------------------------------------------------------------------------- ceiling, Y16
def check_y15(path: Path) -> tuple[tc.Check, dict]:
    if not path.is_file():
        return tc.Check("Y15", "Ceiling variants h to k", "review", f"{display_path(path)} not found; run python scripts/current_ceiling.py"), {}
    scores = json.loads(path.read_text()).get("scores", {})
    keep = {v: {k: s.get(k) for k in SCORE_KEYS} for v, s in scores.items() if v[:2] in ("f_", "g_", "h_", "i_", "j_", "k_")}
    if not any(v[:2] in ("h_", "i_", "j_", "k_") for v in keep):
        return tc.Check("Y15", "Ceiling variants h to k", "review", "the ceiling report has no variant h to k; rerun the ceiling"), keep
    summary = ", ".join(f"{v} {s['S']:.4f}" for v, s in keep.items() if s.get("S") is not None)
    return tc.Check("Y15", "Ceiling variants h to k", "review", f"S: {summary} (F79: f {F79['f']['S']}, g {F79['g']['S']})"), keep


def derived_on_maps(maps: np.ndarray, machine: str) -> dict:
    """Share of maps on which the scorer finds a closed boundary; over those maps, the derived
    R_axis, Z_axis and kappa (scorer functions, the machine's envelope) and the boundary's inner
    and outer radius, width and height in metres: medians, and count, 5th, 50th and 95th
    percentiles under ``spread``."""
    scorer = starter_kit.scoring()
    env = starter_kit.envelope_mask(machine_config(machine))
    found, values = 0, {k: [] for k in (*DERIVED, *EXTENT)}
    for psi in maps:
        p = np.asarray(psi, dtype=np.float64)
        contour = scorer.lcfs.extract_lcfs(p, env.grid_R, env.grid_Z, machine, env.mask_coarse, env.mask_f)
        if contour is None:
            continue
        found += 1
        vals = scorer.derive.derive_frame(p, env.grid_R, env.grid_Z, machine, env.mask_coarse, env.mask_f, contour=contour)
        vals = {**{k: vals[k] for k in DERIVED}, **tc.boundary_extent(contour)}
        for k, v in vals.items():
            values[k].append(v)
    n = len(maps)
    return {"n_frames": n, "boundary_share": found / n if n else float("nan"),
            **{f"median_{k}": float(np.nanmedian(v)) if v else float("nan") for k, v in values.items()},
            "spread": {k: tc.quantiles(v) for k, v in values.items()}}


def spread_text(q: dict, digits: int = 3) -> str:
    """``median (5th to 95th)`` of a ``tc.quantiles`` result."""
    return f"{q['q50']:.{digits}f} ({q['q5']:.{digits}f} to {q['q95']:.{digits}f})"


def check_y16(objects, split: dict, record: dict, args, workers: int) -> tuple[tc.Check, dict]:
    d3d_table = predictors.green_table(args.green_dir, "DIII-D")
    d3d_scales = adopted_scales("DIII-D", d3d_table.columns)
    mast_table = predictors.green_table(args.green_dir, "MAST")
    mast_scales = adopted_scales("MAST", mast_table.columns)
    ridge_columns = [c for r in objects.remainder_model.ridges.values() for c in r.columns]
    d3d_columns = [*targets.current_columns_of(d3d_table, "DIII-D"), "ip__mean", *ridge_columns, *bl.input_columns()]
    out = {}
    # DIII-D validation: every score term, with and without the remainder; baseline_v0 scalars.
    val_name = record.get("validation_list")
    if val_name and split.get(val_name):
        baseline = predictors.BaselineV0(args.split, args.cache_dir, args.green_dir, args.table_dir, args.data_root)
        baseline.setup(log=log)
        scalars = baseline.model_set.baselines()["ridge_signed"].scalars
        val_ids = sorted(split[val_name])
        val_rows = strided_rows(args.split_name, val_ids, d3d_columns, args.table_dir, args.y16_stride)
        references = bl.build_references(val_ids, args.data_root, args.split_name, args.y16_stride, workers=workers,
                                         cache_dir=args.cache_dir / "references")
        for label, with_remainder in (("with_remainder", True), ("without_remainder", False)):
            def predict(shot_id, wr=with_remainder):
                rows = val_rows[shot_id]
                maps = tb.mean_current_maps("DIII-D", rows, d3d_table, d3d_scales, objects, with_remainder=wr)
                q95, beta_n = scalars(rows)
                return bl.Prediction(maps.astype(np.float32), np.asarray(q95), np.asarray(beta_n))
            result = bl.score_local(predict, val_ids, args.data_root, args.split_name, args.y16_stride, references, workers)
            out[f"validation_{label}"] = {k: v for k, v in bl.score_terms(result.finalize()).items() if k in SCORE_KEYS}
            log(f"  Y16 validation {label}: S {out[f'validation_{label}']['S']:.4f}")
    # Test frames: inputs only.
    rng = np.random.default_rng(args.seed)
    for machine, split_name, table, scales, columns in (
            ("DIII-D", "diii_d_public_test", d3d_table, d3d_scales, [*targets.current_columns_of(d3d_table, "DIII-D"), "ip__mean", *ridge_columns]),
            ("MAST", "mast_public_test", mast_table, mast_scales, [*targets.current_columns_of(mast_table, "MAST"), "ip__mean"])):
        rows = ft.read_frame_table(split_name, list(dict.fromkeys(["shot", "frame", *columns])), root=args.table_dir)
        rows = rows.iloc[np.sort(rng.permutation(len(rows))[:args.y16_frames])].reset_index(drop=True)
        maps = tb.mean_current_maps(machine, rows, table, scales, objects, with_remainder=True)
        out[f"test_{machine}"] = derived_on_maps(maps, machine)
    parts = []
    for key in ("validation_with_remainder", "validation_without_remainder"):
        if key in out:
            parts.append(f"{key.replace('_', ' ')} S {out[key]['S']:.4f}, R2 psi {out[key]['r2_psi']:.4f}")
    for m in tb.MACHINES:
        v = out[f"test_{m}"]
        parts.append(f"{m} test ({v['n_frames']} frames): boundary on {v['boundary_share']:.3f}, median R_axis "
                     f"{v['median_R_axis']:.3f} m, Z_axis {v['median_Z_axis']:+.3f} m, kappa {v['median_kappa']:.3f}, "
                     f"boundary inner R {v['median_inner_R']:.3f} m, outer R {v['median_outer_R']:.3f} m, "
                     f"width {v['median_width']:.3f} m, height {v['median_height']:.3f} m")
    return tc.Check("Y16", "Mean current", "review", "; ".join(parts)), out


# -------------------------------------------------------------------------------- report
def render(checks: list[tc.Check], info: dict) -> str:
    lines = [f"# Target profile, version {info['version']}", "",
             f"Target build `{info['target_dir']}` (configuration hash `{info['config_hash'][:12]}`, lists "
             f"{', '.join(info['tables'])}, stride {info['stride']}). No MAST equilibrium label is read.", "",
             tc.check_table(checks), ""]
    y10 = info.get("Y10_Y11", {})
    if y10:
        lines += ["## Y10 and Y11. Where the current lands on MAST", "",
                  "Envelope: the cells inside the 99 percent support (largest cells first) of at least 1 percent of the "
                  f"{info['n_envelope_shares']} reconstructed training shares, placed on MAST. Centroid of the placed mean share.", ""]
        lines += [f"- {landing_line(name, v)}" for name, v in y10.items()]
        nominal = y10.get("nominal", {}).get("envelope", {})
        if nominal.get("coils_inside"):
            lines.append(f"- Coil columns with an element inside the nominal envelope: {', '.join(nominal['coils_inside'])}")
        lines.append("")
    y9 = info.get("Y9", {})
    if y9:
        rings = next(iter(y9.values()))["symmetry_rings_excluded"]
        lines += ["## Y9. Response against the filament Green's function (F89)", "",
                  f"Symmetry is ruled between source cells outside the first {rings} interior rings; the value over all "
                  "source cells is shown beside it. The asymmetry of cells in those rings is a property of the border "
                  "treatment (F89); the ring guard is the share of the absolute current of the fit sample in those rings "
                  "(the share as it is on DIII-D, placed on MAST), largest over the frames.", "",
                  "| Machine | Beyond 5 cells | Rule | Beyond 10 cells | Rule | Symmetry | Rule | Symmetry, all sources | "
                  "Z flip | Rule | Ring share, largest | 99th percentile | Pooled | Rule |", "| --- " * 14 + "|"]
        for m, v in y9.items():
            t, g = v["thresholds"], v.get("ring_guard") or {}
            guard = " | ".join(f"{g[k]:.1e}" if k in g else "not recorded" for k in ("max", "q99", "pooled"))
            lines.append(f"| {m} | {v['beyond_5']:.2e} | {t['beyond_5']:g} | {v['beyond_10']:.2e} | {t['beyond_10']:g} | "
                         f"{v['symmetry']:.2e} ({v['n_symmetry_sources']} sources) | {t['symmetry']:g} | "
                         f"{v['symmetry_all_sources']:.2e} ({v['n_sources']} sources) | {v['flip']:.1e} | {t['flip']:g} | "
                         f"{guard} | {t['ring_share']:g} |")
        lines.append("")
    y15 = info.get("Y15", {})
    if y15:
        lines += ["## Y15. Ceiling variants", "", "| Variant | " + " | ".join(SCORE_KEYS) + " |", "| --- " * (len(SCORE_KEYS) + 1) + "|"]
        for v, s in y15.items():
            lines.append(f"| {v} | " + " | ".join("nan" if s.get(k) is None else f"{s[k]:.4f}" for k in SCORE_KEYS) + " |")
        lines.append("")
    y16 = info.get("Y16", {})
    if any(k.startswith("validation") for k in y16):
        lines += ["## Y16. Mean current on DIII-D validation", "", "| Arm | " + " | ".join(SCORE_KEYS) + " |",
                  "| --- " * (len(SCORE_KEYS) + 1) + "|"]
        for k, s in y16.items():
            if k.startswith("validation"):
                lines.append(f"| {k} | " + " | ".join(f"{s.get(x, float('nan')):.4f}" for x in SCORE_KEYS) + " |")
        lines.append("")
    tests = {k[len("test_"):]: v for k, v in y16.items() if k.startswith("test_") and "spread" in v}
    if tests:
        lines += ["## Y16. Mean current on the test frames", "",
                  "Inputs only. Over the frames on which the scorer finds a closed boundary in the composed map: the "
                  "derived R_axis, Z_axis and kappa and the boundary's inner and outer radius, width and height, as "
                  "median (5th to 95th percentile).", "",
                  "| Machine | Frames | Boundary found | " + " | ".join(Y16_LABELS.values()) + " |",
                  "| --- " * (3 + len(Y16_LABELS)) + "|"]
        for m, v in tests.items():
            lines.append(f"| {m} | {v['n_frames']} | {v['boundary_share']:.3f} | "
                         + " | ".join(spread_text(v["spread"][k]) for k in Y16_LABELS) + " |")
        lines.append("")
    lines += [f"Runtime {info['runtime_s']:.0f} s.", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, default=tb.DEFAULT_CONFIG)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--split-name", default=tb.LABELLED_SPLIT)
    parser.add_argument("--targets", type=Path, default=tb.DEFAULT_TARGET_ROOT)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--ceiling", type=Path, default=DEFAULT_CEILING)
    parser.add_argument("--data-root", type=Path, default=default_data_root())
    parser.add_argument("--table-dir", type=Path, default=ft.DEFAULT_TABLE_ROOT)
    parser.add_argument("--green-dir", type=Path, default=dataset.DEFAULT_PROCESSED_ROOT / "green")
    parser.add_argument("--response-cache", type=Path, default=ct.DEFAULT_CACHE_DIR)
    parser.add_argument("--cache-dir", type=Path, default=bl.DEFAULT_CACHE_DIR, help="baseline cache: baseline_v0 models and scorer references")
    parser.add_argument("--check-shots", type=int, default=50)
    parser.add_argument("--check-frames", type=int, default=1000)
    parser.add_argument("--envelope-frames", type=int, default=20000)
    parser.add_argument("--source-step", type=int, default=6, help="Y9: every n-th interior node is a source cell")
    parser.add_argument("--y16-frames", type=int, default=2000)
    parser.add_argument("--y16-stride", type=int, default=4)
    parser.add_argument("--skip-y16", action="store_true", help="leave out Y16 (scoring of the mean current)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args(argv)

    started = time.time()
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    try:
        tb.require_labelled(args.split_name)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    config = tb.TargetConfig.load(args.config)
    folder = tb.target_dir(config.version, args.targets)
    if not (folder / "build.json").is_file():
        raise SystemExit(f"{folder / 'build.json'} not found; run python scripts/build_targets.py first")
    objects = tb.TargetObjects.load(folder)
    record = objects.record
    workers = bl.default_workers() if args.workers is None else args.workers
    split = sp.read_split(args.split)
    tables = {n: pd.read_parquet(folder / f) for n, f in record["tables"].items()}
    out_dir = Path(args.report_root) / f"v{config.version}"
    out_dir.mkdir(parents=True, exist_ok=True)
    responses = {m: cb.machine_response(m, args.green_dir, args.response_cache) for m in tb.MACHINES}
    d3d_grid = GreenTable.load(args.green_dir / "green_d3d.npz")
    mast_grid = GreenTable.load(args.green_dir / "green_mast.npz")
    d3d = {"table": d3d_grid, "scales": adopted_scales("DIII-D", d3d_grid.columns), "response": responses["DIII-D"]}
    info = {"version": config.version, "config_hash": record.get("config_hash", ""), "target_dir": display_path(folder),
            "tables": list(tables), "stride": record["stride"]}
    checks = []

    c, info["Y1"] = check_y1(tables, split, record, args); checks.append(c)
    fit_name = record["fit_list"]
    fit_rows = strided_rows(args.split_name, sorted(split[fit_name]), [*targets.current_columns_of(d3d_grid, "DIII-D"), "ip__mean"],
                            args.table_dir, record["stride"])
    chosen = sample_frames(fit_rows, args.check_shots, args.check_frames, args.seed)
    y2, y5, info["Y2_Y5"] = check_y2_y5(objects, fit_rows, chosen, d3d, args, record["stride"])
    c, info["Y3"] = check_y3(tables); checks += [y2, c]
    p = tc.parity_errors(objects.current_basis)
    info["Y4"] = p
    checks.append(tc.Check("Y4", "Parity", "pass" if p["component_max"] <= 1e-8 and p["mean_max"] <= 1e-8 else "fail",
                           f"{p['even']} even and {p['odd']} odd components; largest departure {p['component_max']:.1e}, "
                           f"mean {p['mean_max']:.1e} (rule 1e-8)"))
    checks.append(y5)
    c, info["Y6"] = check_y6(tables); checks.append(c)
    log("Y1 to Y6 done")

    shares = reconstructed_shares(objects, tables[fit_name], args.check_frames, args.seed)
    c7, c8, info["Y7_Y8"] = check_y7_y8(objects, shares, responses); checks += [c7, c8]
    c, info["Y9"] = check_y9(responses, args.source_step, record); checks.append(c)
    geometry, _ = cp.geometry_from_split(args.data_root, "MAST")
    envelope_shares = reconstructed_shares(objects, tables[fit_name], args.envelope_frames, args.seed + 1)
    info["n_envelope_shares"] = int(len(envelope_shares))
    info["Y10_Y11"] = landing_cases(objects, envelope_shares, mast_grid, d3d_grid, geometry)
    checks.append(tc.Check("Y10", "Where the current lands", "review", landing_line("MAST constants", info["Y10_Y11"]["nominal"])))
    checks.append(tc.Check("Y11", "Sensitivity to the constants", "review", "; ".join(
        f"{n}: centroid R {v['envelope']['centroid_r_m']:.3f} m, envelope R {v['envelope']['inner_r_m']:.3f} to "
        f"{v['envelope']['outer_r_m']:.3f} m, nearest coil {v['envelope']['coil_distance_m']:.3f} m"
        for n, v in info["Y10_Y11"].items() if n != "nominal")))
    log("Y7 to Y11 done")

    rem_checks, info["Y12_Y14"] = check_y12_to_y14(tables, objects, record); checks += rem_checks
    c, info["Y15"] = check_y15(args.ceiling); checks.append(c)
    if args.skip_y16:
        checks.append(tc.Check("Y16", "Mean current", "review", "not run (--skip-y16)"))
        info["Y16"] = {}
    else:
        c, info["Y16"] = check_y16(objects, split, record, args, workers); checks.append(c)
    log("Y12 to Y16 done")

    checks.sort(key=lambda k: int(k.id[1:]))
    info["checks"] = [k.to_dict() for k in checks]
    info.update(run_info(argv, vars(args), timestamp, time.time() - started))
    (out_dir / "target_summary.json").write_text(json.dumps(info, indent=2, default=lambda v: v.tolist() if hasattr(v, "tolist") else str(v)) + "\n")
    (out_dir / "target_report.md").write_text(render(checks, info))
    print(tc.check_table(checks))
    log(f"wrote {out_dir} in {time.time() - started:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
