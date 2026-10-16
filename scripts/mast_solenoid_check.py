"""MAST solenoid scale check (diagnostic, inputs only).

Tests whether the MAST solenoid flux of the coil model (656 single-turn elements, each
carrying the full ``magnetics_sol_current``) is off by a constant factor, with DIII-D as the
calibrated reference (adopted coil model: shaping coils at scale 1 plus the flat solenoid
term of F48). Report: ``reports/coil_flux/mast_solenoid_check.md``.

No equilibrium label is read. The shots are the two public test splits, whose files carry no
label; every raw column requested is checked against ``frame_table.is_label_column``, and the
three labelled MAST sample shots are not opened. The factor ``f`` multiplies the MAST solenoid column of the
``unit`` coil model (``CHECK_COIL_MODELS``), whatever the adopted MAST model is.

Sections of the output:

1. Geometry and units. Solenoid winding (layers, conductors, pitch), its flux per kA at the
   reference point and in the bore against the long-solenoid formula, the P-coil element
   counts, and the solenoid current range of the data. The flux swing implied by that range
   under ``f`` = 1 and 0.5 is compared with the published MAST swing.
2. Flux consumption, per shot from the raw magnetics. Start: the last Ip sample within
   ``START_FRACTION`` of the peak above the pre-shot baseline before the 10 percent crossing,
   less ``START_LEAD_MS`` (before the MAST merging-compression P3 swing ends). End: the first
   time the 2 ms mean of Ip reaches ``FLAT_TOP_FRACTION`` of the peak. The coil flux consumed
   between the two, ``-2 pi sign(Ip) (psi(end) - psi(start))``, is divided by
   ``mu0 R0 Ip(end)`` and split into the solenoid and the shaping-coil parts. The inductive
   need is ``ell (1 - Ip(start) / Ip(end))`` with ``ell`` the plasma self-inductance over
   ``mu0 R0``, either large-aspect-ratio ``ln(8 R0 / a0) + li / 2 - 2`` or Hirshman-Neilson
   external inductance plus ``li / 2``. The implied Ejima coefficient is consumed minus need.
   The solenoid factor that gives MAST the DIII-D median coefficient is reported per shot.
   The same is repeated from the 10 percent crossing (``ip_rise_ms``) as a sensitivity case.
3. P-coil checks. Vertical field of the coils at (R0, 0) at the flat-top start over
   ``mu0 Ip / (4 pi R0)``, split into solenoid and shaping coils, against the large-aspect
   force-balance value ``-(ln(8 R0 / a0) + beta_p + li / 2 - 3 / 2)``. On MAST, the solenoid
   factor that minimizes the coil poloidal field over a breakdown region at the start time
   (least squares): at the start on shots whose P3 coils are idle there, and ``NULL_PRE_MS``
   before the start on shots whose P3 coils are idle then (before a merging-compression pulse).
4. Impact of ``f`` = 0.5 on MAST: change of the coil-point features, the named terms,
   ``shaping``, ``shafranov_lambda`` and ``ohmic_flux_used`` in units of the DIII-D scaler's
   interquartile range, the share of MAST frames outside the DIII-D test 1st to 99th
   percentile before and after, and the sum of squares of the map change against the pooled
   sum of squares of a submission's MAST maps.

Writes ``shots.parquet`` (one row per shot, both machines), ``impact_features.parquet`` and
``summary.json`` to ``--out``, and prints the summary. Usage::

    python scripts/mast_solenoid_check.py
    python scripts/mast_solenoid_check.py --max-shots 20 --workers 2 --no-impact
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from fusion_eq import coil_pattern as cp
from fusion_eq import dataset, features, frame_table, io, resample
from fusion_eq.coil_flux import (A_PER_KA, GEOMETRY_COLUMNS, MU0, CoilGeometry, GreenTable,
                                 adopted_scales, build_point_table)
from fusion_eq.feature_scaler import RobustScaler
from fusion_eq.ip_alignment import breakdown_time_ms
from fusion_eq.machines import MAST, MachineConfig, machine_config

SPLITS = {"DIII-D": "diii_d_public_test", "MAST": "mast_public_test"}
GREEN_FILES = {"DIII-D": "green_d3d.npz", "MAST": "green_mast.npz"}
DEFAULT_GREEN_DIR = features.DEFAULT_GREEN_DIR
DEFAULT_OUT = dataset.REPO_ROOT / "reports" / "coil_flux" / "mast_solenoid_check"
DEFAULT_PREDICTIONS = dataset.REPO_ROOT / "submissions" / "2026-10-15_arm_g_v1" / "mast_public_test.npz"
FEATURE_VERSION = 2

# Windowed means of the currents, as in the frame table (resample.window_mean).
WINDOW_MS = 2.0
MAX_GAP_MS = 5.0
# Shot timing.
RISE_FRACTION = 0.10            # the organizers' breakdown fraction (ip_rise_ms)
START_FRACTION = 0.02           # of the peak, above the pre-shot baseline
START_LEAD_MS = 1.0             # half the averaging window, so the start mean is pre-rise
BASELINE_WINDOW_MS = (-100.0, -30.0)   # relative to the 10 percent crossing
FLAT_TOP_FRACTION = 0.90        # coil_scales.FLAT_TOP_LEVEL
FLAT_SEARCH_STEP_MS = 0.5
FLAT_SEARCH_SPAN_MS = 3000.0
MIN_PEAK_KA = 100.0
MIN_RAMP_MS = 5.0
# Finite-difference half step in metres for Bz at the reference point.
BZ_STEP_M = 0.01
# Nominal internal inductance per machine (the investigation's values), and a nominal
# elongation for the Hirshman-Neilson external inductance (sensitivity case only).
LI_NOMINAL = {"DIII-D": 1.0, "MAST": 0.7}
KAPPA_NOMINAL = {"DIII-D": 1.8, "MAST": 1.8}
BETA_P_BAND = (0.1, 1.0)
# MAST start-up class: merging-compression when the mean P3 current at the start exceeds this
# (kA-turn); the P3 coils carry about 120 kA-turn before a merging-compression start.
P3_ACTIVE_KA = 20.0
P3_COLUMNS = ("magnetics_p3l_current", "magnetics_p3u_current")
# Breakdown region of the field-null fit on MAST, metres, clear of every coil.
NULL_REGION_R = (0.30, 1.30)
NULL_REGION_Z = (-0.80, 0.80)
MIN_SOL_FOR_NULL_KA = 10.0
NULL_PRE_MS = 10.0
SOLENOID_FACTORS = (1.0, 0.5)
# The coil models the factor f multiplies: MAST unit (every element at the full feed current,
# the model under test) and the adopted DIII-D model as the calibrated reference.
CHECK_COIL_MODELS = {"DIII-D": "solenoid_flat", "MAST": "unit"}
IMPACT_FACTOR = 0.5
BOOTSTRAP_SAMPLES = 2000
SEED = 0
# Published MAST solenoid flux swing in V s (Wb). Sources in the report.
PUBLISHED_SWING_WB = {"mast_u_paper_original_mast": 0.9, "mast_startup_study_available": 0.8}


# --------------------------------------------------------------------------- physics
def large_aspect_inductance(r0_m: float, a0_m: float, li: float) -> float:
    """Plasma self-inductance over ``mu0 R0``, ``ln(8 R0 / a0) + li / 2 - 2`` (circular,
    large aspect ratio), dimensionless."""
    return float(np.log(8.0 * r0_m / a0_m) + li / 2.0 - 2.0)


def hirshman_neilson_inductance(r0_m: float, a0_m: float, li: float, kappa: float) -> float:
    """Plasma self-inductance over ``mu0 R0``: the Hirshman-Neilson (1986) external
    inductance of a shaped plasma at inverse aspect ratio ``a0 / R0`` and elongation
    ``kappa``, plus ``li / 2``, dimensionless."""
    eps = a0_m / r0_m
    s = np.sqrt(eps)
    a = (1.0 + 1.81 * s + 2.05 * eps) * np.log(8.0 / eps) - (2.0 + 9.25 * s - 1.21 * eps)
    b = 0.73 * s * (1.0 + 2.0 * eps ** 4 - 6.0 * eps ** 5 + 3.7 * eps ** 6)
    return float(a * (1.0 - eps) / (1.0 - eps + b * kappa) + li / 2.0)


def inductances(cfg: MachineConfig) -> dict[str, float]:
    r0, a0, li = cfg.major_radius.value, cfg.minor_radius.value, LI_NOMINAL[cfg.name]
    return {"la": large_aspect_inductance(r0, a0, li),
            "hn": hirshman_neilson_inductance(r0, a0, li, KAPPA_NOMINAL[cfg.name])}


def force_balance_bz_ratio(cfg: MachineConfig, beta_p: float) -> float:
    """Large-aspect vertical field over ``mu0 Ip / (4 pi R0)`` for force balance, signed as
    the coil field opposing a positive Ip: ``-(ln(8 R0 / a0) + beta_p + li / 2 - 3 / 2)``."""
    r0, a0 = cfg.major_radius.value, cfg.minor_radius.value
    return float(-(np.log(8.0 * r0 / a0) + beta_p + LI_NOMINAL[cfg.name] / 2.0 - 1.5))


def field_patterns(table: GreenTable, region_r, region_z) -> np.ndarray:
    """``(columns, 2, nodes)`` poloidal field (BR, BZ) in T per kA of each column at the grid
    nodes inside the region, from ``BR = -(1/R) dpsi/dZ`` and ``BZ = (1/R) dpsi/dR``."""
    r = np.asarray(table.grid_r, dtype=np.float64)
    z = np.asarray(table.grid_z, dtype=np.float64)
    rr, zz = np.meshgrid(r, z)
    inside = ((rr >= region_r[0]) & (rr <= region_r[1]) & (zz >= region_z[0]) & (zz <= region_z[1])).ravel()
    out = []
    for p in table.patterns:
        dpsi_dz, dpsi_dr = np.gradient(p, z, r)
        out.append(np.stack([(-dpsi_dz / rr).ravel()[inside], (dpsi_dr / rr).ravel()[inside]]))
    return np.stack(out)


# ------------------------------------------------------------------------------ tables
def evaluation_points(cfg: MachineConfig) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    r0, a0 = cfg.major_radius.value, cfg.minor_radius.value
    names = ("ref", "inner", "outer", "bz_minus", "bz_plus")
    pr = np.array([r0, r0 - a0, r0 + a0, r0 - BZ_STEP_M, r0 + BZ_STEP_M])
    return pr, np.zeros(pr.size), names


def machine_tables(machine: str, data_root: Path, green_dir: Path) -> dict:
    """Point table at the evaluation points, adopted coil model and the field patterns of
    one machine, from the geometry of the first shot of its test split."""
    cfg = machine_config(machine)
    path = dataset.list_shots(data_root, SPLITS[machine])[0]
    assert_no_labels(GEOMETRY_COLUMNS)
    shot = io.load_shot(path, GEOMETRY_COLUMNS)
    geometry = CoilGeometry.from_shot(shot)
    grid = GreenTable.load(green_dir / GREEN_FILES[machine])
    if geometry.hash() != grid.geometry_hash:
        raise ValueError(f"geometry of {path.name} differs from {GREEN_FILES[machine]}")
    pr, pz, names = evaluation_points(cfg)
    point = build_point_table(cfg, geometry, pr, pz, grid.n_filaments)
    return {"machine": machine, "geometry": geometry, "grid": grid, "point": point, "point_names": names,
            "scales": adopted_scales(machine, point.columns, CHECK_COIL_MODELS[machine]),
            "fields": field_patterns(grid, NULL_REGION_R, NULL_REGION_Z), "geometry_source": path.name}


# ---------------------------------------------------------------------------- per shot
def assert_no_labels(columns) -> None:
    """Raises when a requested raw column is withheld on the test splits."""
    labels = [c for c in columns if frame_table.is_label_column(c)]
    if labels:
        raise ValueError(f"label columns requested: {labels}")


def _flux_parts(tables: dict, currents_ka: np.ndarray) -> dict[str, np.ndarray]:
    """Coil flux in Wb/rad at the evaluation points under the adopted model, for the
    solenoid alone and the shaping coils alone; ``currents_ka`` in point-table order."""
    point, cfg = tables["point"], machine_config(tables["machine"])
    sol = point.columns.index(cfg.solenoid_column)
    only_sol = np.zeros_like(currents_ka)
    only_sol[sol] = currents_ka[sol]
    only_pf = currents_ka.copy()
    only_pf[sol] = 0.0
    return {"sol": np.asarray(point.flux(only_sol, tables["scales"])),
            "pf": np.asarray(point.flux(only_pf, tables["scales"]))}


def _flat_top_time(t_ip, ip_abs, t_from: float, level_ka: float) -> float:
    grid = np.arange(t_from, t_from + FLAT_SEARCH_SPAN_MS, FLAT_SEARCH_STEP_MS)
    mean = resample.window_mean(t_ip, ip_abs, grid, WINDOW_MS, MAX_GAP_MS)
    hit = np.flatnonzero(np.nan_to_num(mean) >= level_ka)
    return float(grid[hit[0]]) if hit.size else np.nan


def shot_timing(t_ip, ip_ka) -> dict:
    """Peak, baseline, start, 10 percent and flat-top times of one shot (ms, kA)."""
    t = np.asarray(t_ip, dtype=np.float64)
    ip = np.asarray(ip_ka, dtype=np.float64)
    keep = np.isfinite(t) & np.isfinite(ip)
    order = np.argsort(t[keep], kind="stable")
    t, ip = t[keep][order], ip[keep][order]
    out = {"peak_ka": np.nan, "baseline_ka": np.nan, "t_start_ms": np.nan, "t10_ms": np.nan, "t_ft_ms": np.nan}
    if t.size == 0:
        return out
    peak = float(np.percentile(np.abs(ip), 99.5))
    t10 = breakdown_time_ms(t, ip, RISE_FRACTION)
    out.update(peak_ka=peak, t10_ms=t10)
    if not np.isfinite(t10):
        return out
    base = (t >= t10 + BASELINE_WINDOW_MS[0]) & (t <= t10 + BASELINE_WINDOW_MS[1])
    baseline = float(np.median(ip[base])) if base.any() else 0.0
    before = np.flatnonzero((t < t10) & (np.abs(ip - baseline) <= START_FRACTION * peak))
    t_cross = float(t[before[-1]]) if before.size else t10
    out.update(baseline_ka=baseline, t_start_ms=t_cross - START_LEAD_MS,
               t_ft_ms=_flat_top_time(t, np.abs(ip), t10, FLAT_TOP_FRACTION * peak))
    return out


def shot_balance(path: Path, tables: dict) -> dict:
    """Flux consumption, vertical field and breakdown-null quantities of one shot."""
    machine = tables["machine"]
    cfg = machine_config(machine)
    point = tables["point"]
    columns = list(dict.fromkeys([cfg.ip_column, cfg.time_axis_column(cfg.ip_column), *point.columns, "magnetics_time"]))
    assert_no_labels(columns)
    shot = io.load_shot(path, columns)
    row: dict = {"machine": machine, "shot": shot.shot_id, "error": ""}
    t_ip = shot[cfg.time_axis_column(cfg.ip_column)]
    ip = shot[cfg.ip_column]
    row.update(shot_timing(t_ip, ip))
    t_mag = shot["magnetics_time"]
    sol_raw = np.asarray(shot[cfg.solenoid_column], dtype=np.float64)
    row["sol_record_min_ka"] = float(np.nanmin(sol_raw))
    row["sol_record_max_ka"] = float(np.nanmax(sol_raw))
    if not (np.isfinite(row["t_start_ms"]) and np.isfinite(row["t_ft_ms"])):
        row["error"] = "no start or flat-top time"
        return row
    if row["peak_ka"] < MIN_PEAK_KA or row["t_ft_ms"] - row["t_start_ms"] < MIN_RAMP_MS:
        row["error"] = "peak or ramp too small"
        return row
    times = np.array([row["t_start_ms"], row["t10_ms"], row["t_ft_ms"]])
    ip_at = resample.window_mean(t_ip, ip, times, WINDOW_MS, MAX_GAP_MS)
    currents = np.column_stack([resample.window_mean(t_mag, shot[c], times, WINDOW_MS, MAX_GAP_MS)
                                for c in point.columns])
    if not (np.isfinite(ip_at).all() and np.isfinite(currents).all()):
        row["error"] = "missing current at an evaluation time"
        return row
    in_plasma = (t_mag >= row["t10_ms"]) & (t_mag <= row["t_ft_ms"])
    sol_idx = point.columns.index(cfg.solenoid_column)
    row.update(ip_start_ka=float(ip_at[0]), ip_t10_ka=float(ip_at[1]), ip_ft_ka=float(ip_at[2]),
               sol_start_ka=float(currents[0, sol_idx]), sol_t10_ka=float(currents[1, sol_idx]),
               sol_ft_ka=float(currents[2, sol_idx]),
               sol_ramp_min_ka=float(np.nanmin(sol_raw[in_plasma])) if in_plasma.any() else np.nan,
               ramp_ms=float(row["t_ft_ms"] - row["t_start_ms"]))
    if machine == "MAST":
        p3 = [point.columns.index(c) for c in P3_COLUMNS]
        row["p3_start_ka"] = float(np.mean(currents[0, p3]))
        row["p3_t10_ka"] = float(np.mean(currents[1, p3]))
        row["merging_compression"] = bool(abs(row["p3_start_ka"]) > P3_ACTIVE_KA)
    sign = float(np.sign(ip_at[2]))
    denominator = MU0 * cfg.major_radius.value * abs(ip_at[2]) * A_PER_KA
    names = tables["point_names"]
    parts = [_flux_parts(tables, currents[k]) for k in range(3)]
    for label, k0 in (("start", 0), ("t10", 1)):
        for comp in ("sol", "pf"):
            consumed = -2.0 * np.pi * sign * (parts[2][comp] - parts[k0][comp]) / denominator
            row[f"c_{comp}_ref_{label}"] = float(consumed[names.index("ref")])
            row[f"c_{comp}_edge_{label}"] = float(0.5 * (consumed[names.index("inner")] + consumed[names.index("outer")]))
        row[f"ip_frac_{label}"] = float(ip_at[k0] / ip_at[2])
    # Vertical field at (R0, 0) at the flat-top start, over mu0 Ip / (4 pi R0), signed by Ip.
    r0 = cfg.major_radius.value
    bz_unit = MU0 * ip_at[2] * A_PER_KA / (4.0 * np.pi * r0)
    for comp in ("sol", "pf"):
        psi = parts[2][comp]
        bz = (psi[names.index("bz_plus")] - psi[names.index("bz_minus")]) / (2.0 * BZ_STEP_M * r0)
        row[f"bz_ratio_{comp}"] = float(bz / bz_unit)
    # Breakdown null (MAST): least-squares solenoid factor over the region at the start, and
    # NULL_PRE_MS earlier, before the merging-compression P3 pulse.
    if machine == "MAST":
        row.update(null_fit(tables, point.columns, currents[0], ""))
        pre = np.array([resample.window_mean(t_mag, shot[c], np.array([row["t_start_ms"] - NULL_PRE_MS]), WINDOW_MS,
                                             MAX_GAP_MS)[0] for c in point.columns])
        p3 = [point.columns.index(c) for c in P3_COLUMNS]
        row["p3_pre_ka"] = float(np.mean(pre[p3]))
        row["sol_pre_ka"] = float(pre[sol_idx])
        if np.isfinite(pre).all():
            row.update(null_fit(tables, point.columns, pre, "_pre"))
    return row


def null_fit(tables: dict, columns, currents_ka: np.ndarray, suffix: str) -> dict:
    """Solenoid factor ``f*`` minimizing the summed squared coil poloidal field over the null
    region, ``-<B_pf, B_sol> / <B_sol, B_sol>``, and the rms field in gauss at the factors of
    ``SOLENOID_FACTORS`` and for the solenoid alone."""
    cfg = machine_config(tables["machine"])
    grid_cols = list(tables["grid"].columns)
    c = np.array([currents_ka[list(columns).index(name)] for name in grid_cols])
    fields = tables["fields"]
    g_sol = grid_cols.index(cfg.solenoid_column)
    b_sol = c[g_sol] * fields[g_sol]
    b_pf = np.tensordot(np.where(np.arange(len(grid_cols)) == g_sol, 0.0, c), fields, axes=1)
    ss_sol = float(np.sum(b_sol ** 2))
    out = {f"null{suffix}_f_star": float(-np.sum(b_pf * b_sol) / ss_sol) if ss_sol > 0 else np.nan,
           f"null{suffix}_rms_gauss_sol_only": float(np.sqrt(np.mean(np.sum(b_sol ** 2, axis=0))) * 1e4)}
    for f in SOLENOID_FACTORS:
        out[f"null{suffix}_rms_gauss_f{f:g}"] = float(np.sqrt(np.mean(np.sum((b_pf + f * b_sol) ** 2, axis=0))) * 1e4)
    return out


_TABLES: dict = {}


def _init_worker(tables: dict) -> None:
    global _TABLES
    _TABLES = tables


def _worker(job: tuple[str, str]) -> dict:
    machine, path = job
    try:
        return shot_balance(Path(path), _TABLES[machine])
    except Exception as exc:                      # recorded per shot, the run continues
        return {"machine": machine, "shot": Path(path).stem.split("_")[-1], "error": repr(exc)}


# ---------------------------------------------------------------------------- summaries
def describe(values) -> dict:
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"n": 0}
    q = np.percentile(v, [5, 25, 50, 75, 95])
    return {"n": int(v.size), "p05": float(q[0]), "p25": float(q[1]), "median": float(q[2]), "p75": float(q[3]),
            "p95": float(q[4])}


def bootstrap_median_ratio(num, den, rng: np.random.Generator) -> dict:
    """Ratio of medians with a 90 percent bootstrap interval over shots."""
    a = np.asarray(num, dtype=np.float64)
    b = np.asarray(den, dtype=np.float64)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return {"ratio": np.nan}
    draws = np.array([np.median(rng.choice(a, a.size)) / np.median(rng.choice(b, b.size))
                      for _ in range(BOOTSTRAP_SAMPLES)])
    return {"ratio": float(np.median(a) / np.median(b)), "p05": float(np.percentile(draws, 5)),
            "p95": float(np.percentile(draws, 95))}


def balance_summary(shots: pd.DataFrame, rng: np.random.Generator) -> dict:
    """Implied Ejima coefficients per machine and the MAST solenoid factor per case."""
    ok = shots[shots["error"] == ""]
    d3d, mast = ok[ok["machine"] == "DIII-D"], ok[ok["machine"] == "MAST"]
    ell = {m: inductances(machine_config(m)) for m in SPLITS}
    classes = {"all": mast, "solenoid_start": mast[~mast["merging_compression"].astype(bool)],
               "merging_compression": mast[mast["merging_compression"].astype(bool)]}
    out: dict = {"inductance_over_mu0_r0": ell, "cases": {}}
    for start in ("start", "t10"):
        for where in ("ref", "edge"):
            for form in ("la", "hn"):
                key = f"{start}/{where}/{form}"
                case: dict = {}
                need = {m: ell[m][form] * (1.0 - df[f"ip_frac_{start}"]) for m, df in (("DIII-D", d3d), ("MAST", mast))}
                sol = {m: df[f"c_sol_{where}_{start}"] for m, df in (("DIII-D", d3d), ("MAST", mast))}
                pf = {m: df[f"c_pf_{where}_{start}"] for m, df in (("DIII-D", d3d), ("MAST", mast))}
                ce_sol = {m: sol[m] - need[m] for m in sol}
                ce_tot = {m: sol[m] + pf[m] - need[m] for m in sol}
                ref_sol, ref_tot = float(np.nanmedian(ce_sol["DIII-D"])), float(np.nanmedian(ce_tot["DIII-D"]))
                f_sol = (need["MAST"] + ref_sol) / sol["MAST"]
                f_tot = (need["MAST"] + ref_tot - pf["MAST"]) / sol["MAST"]
                for m in SPLITS:
                    case[m] = {"c_sol": describe(sol[m]), "c_pf": describe(pf[m]), "need": describe(need[m]),
                               "ce_solenoid_only": describe(ce_sol[m]), "ce_total": describe(ce_tot[m])}
                case["ratio_mast_over_d3d_ce_solenoid_only"] = bootstrap_median_ratio(ce_sol["MAST"], ce_sol["DIII-D"], rng)
                case["ratio_mast_over_d3d_ce_total"] = bootstrap_median_ratio(ce_tot["MAST"], ce_tot["DIII-D"], rng)
                case["mast_f_star_solenoid_only"] = {c: describe(f_sol.loc[df.index]) for c, df in classes.items()}
                case["mast_f_star_total"] = {c: describe(f_tot.loc[df.index]) for c, df in classes.items()}
                out["cases"][key] = case
    return out


def pcoil_summary(shots: pd.DataFrame) -> dict:
    ok = shots[shots["error"] == ""]
    out: dict = {}
    for m in SPLITS:
        df = ok[ok["machine"] == m]
        cfg = machine_config(m)
        out[m] = {"bz_ratio_sol": describe(df["bz_ratio_sol"]), "bz_ratio_pf": describe(df["bz_ratio_pf"]),
                  "bz_ratio_total_f1": describe(df["bz_ratio_sol"] + df["bz_ratio_pf"]),
                  "force_balance_band": [force_balance_bz_ratio(cfg, b) for b in BETA_P_BAND]}
        if m == "MAST":
            out[m]["bz_ratio_total_f0.5"] = describe(0.5 * df["bz_ratio_sol"] + df["bz_ratio_pf"])
            usable = df[(~df["merging_compression"].astype(bool)) & (df["sol_start_ka"].abs() >= MIN_SOL_FOR_NULL_KA)]
            out[m]["null"] = {"n_shots": int(len(usable)), "f_star": describe(usable["null_f_star"]),
                              **{f"rms_gauss_f{f:g}": describe(usable[f"null_rms_gauss_f{f:g}"]) for f in SOLENOID_FACTORS},
                              "rms_gauss_sol_only": describe(usable["null_rms_gauss_sol_only"])}
            out[m]["null_merging_compression"] = describe(df.loc[df["merging_compression"].astype(bool), "null_f_star"])
            pre = df[(df["p3_pre_ka"].abs() <= P3_ACTIVE_KA) & (df["sol_pre_ka"].abs() >= MIN_SOL_FOR_NULL_KA)]
            out[m]["null_pre"] = {"n_shots": int(len(pre)), "f_star": describe(pre["null_pre_f_star"]),
                                  **{f"rms_gauss_f{f:g}": describe(pre[f"null_pre_rms_gauss_f{f:g}"]) for f in SOLENOID_FACTORS},
                                  "rms_gauss_sol_only": describe(pre["null_pre_rms_gauss_sol_only"])}
    return out


def geometry_summary(tables: dict) -> dict:
    """Winding of the solenoid and flux per kA, per machine."""
    out: dict = {}
    for m, t in tables.items():
        cfg = machine_config(m)
        g = t["geometry"]
        idx = g.elements_of(cfg.solenoid_column)
        r, z = g.r[idx], g.z[idx]
        layers = np.unique(np.round(r, 5))
        point = t["point"]
        unit = np.zeros(len(point.columns))
        unit[point.columns.index(cfg.solenoid_column)] = 1.0
        psi_unit = np.asarray(point.flux(unit, t["scales"]))
        grid = t["grid"]
        sol_pattern = grid.patterns[list(grid.columns).index(cfg.solenoid_column)]
        rec = {"geometry_source": t["geometry_source"], "solenoid_elements": int(idx.size),
               "layers_r_m": [float(v) for v in layers],
               "per_layer": [int(np.sum(np.round(r, 5) == v)) for v in layers],
               "z_min_m": float(z.min()), "z_max_m": float(z.max()),
               "element_width_m": sorted({float(v) for v in np.round(g.width[idx], 5)}),
               "element_height_m": sorted({float(v) for v in np.round(g.height[idx], 5)}),
               "duplicate_positions": int(pd.DataFrame({"r": np.round(r, 6), "z": np.round(z, 6)}).duplicated().sum()),
               "model_psi_ref_wb_per_rad_per_ka": float(psi_unit[t["point_names"].index("ref")]),
               "model_psi_inner_wb_per_rad_per_ka": float(psi_unit[t["point_names"].index("inner")]),
               "grid_pattern_wb_per_rad_per_ka": {"min": float(sol_pattern.min()), "max": float(sol_pattern.max())},
               "elements_per_column": {c: int(n) for c, n in g.elements_per_column().items()}}
        if idx.size > 1:
            dz = np.diff(np.sort(z[np.round(r, 5) == layers[0]]))
            length = float(z.max() - z.min() + np.median(dz))
            rec["axial_pitch_m"] = float(np.median(dz))
            rec["winding_length_m"] = length
            # Long-solenoid flux per radian per kA if every element carries 1 kA:
            # mu0 (n_layer / L) I r_layer^2 / 2 summed over layers.
            rec["long_solenoid_psi_wb_per_rad_per_ka"] = float(
                sum(MU0 * (np.sum(np.round(r, 5) == v) / length) * A_PER_KA * v ** 2 / 2.0 for v in layers))
        out[m] = rec
    return out


def current_ranges(shots: pd.DataFrame, geometry: dict) -> dict:
    """Solenoid current range of the data and the flux swing it implies at R0."""
    out: dict = {}
    for m in SPLITS:
        df = shots[shots["machine"] == m]
        geo = geometry[m]
        rec = {"record_min_ka": describe(df["sol_record_min_ka"]), "record_max_ka": describe(df["sol_record_max_ka"]),
               "start_ka": describe(df.get("sol_start_ka", pd.Series(dtype=float))),
               "flat_top_ka": describe(df.get("sol_ft_ka", pd.Series(dtype=float))),
               "ramp_min_ka": describe(df.get("sol_ramp_min_ka", pd.Series(dtype=float))),
               "extreme_min_ka": float(np.nanmin(df["sol_record_min_ka"])) if len(df) else np.nan,
               "extreme_max_ka": float(np.nanmax(df["sol_record_max_ka"])) if len(df) else np.nan}
        span = np.nanpercentile(df["sol_record_max_ka"], 99) - np.nanpercentile(df["sol_record_min_ka"], 1)
        per_ka = 2.0 * np.pi * geo["model_psi_ref_wb_per_rad_per_ka"]
        rec["swing_span_ka_p1_p99"] = float(span)
        rec["swing_wb_at_ref"] = {f"f{f:g}": float(f * per_ka * span) for f in SOLENOID_FACTORS}
        per_ka_bore = 2.0 * np.pi * geo["model_psi_inner_wb_per_rad_per_ka"]
        rec["swing_wb_at_inner_edge"] = {f"f{f:g}": float(f * per_ka_bore * span) for f in SOLENOID_FACTORS}
        out[m] = rec
    out["published_mast_swing_wb"] = PUBLISHED_SWING_WB
    return out



# ------------------------------------------------------------------------------ impact
def impact(predictions: Path | None) -> tuple[dict, pd.DataFrame]:
    """Change of the MAST coil features and maps under the solenoid factor ``IMPACT_FACTOR``."""
    config = features.FeatureConfig.load(dataset.REPO_ROOT / "configs" / f"features_v{FEATURE_VERSION}.yaml")
    res = features.load_resources("MAST", config, coil_model=CHECK_COIL_MODELS["MAST"])
    table = res.point_table
    frames = frame_table.read_frame_table(SPLITS["MAST"], ["shot", "frame", "ip__mean", "sol__mean"])
    coil_cols = config.columns_of("coil_pt")
    term_cols = [f"coil_{t}" for t in cp.TERM_NAMES]
    other = ["shaping", "shafranov_lambda", "ohmic_flux_used"]
    wanted = ["shot", "frame", *coil_cols, *term_cols, *other]
    feats = {s: features.read_features(s, FEATURE_VERSION, columns=wanted) for s in SPLITS.values()}
    mast = frames.merge(feats[SPLITS["MAST"]], on=["shot", "frame"], how="inner", validate="one_to_one")
    currents = np.zeros((len(mast), len(table.columns)))
    currents[:, table.column_index([MAST.solenoid_column])[0]] = (IMPACT_FACTOR - 1.0) * mast["sol__mean"].to_numpy()
    delta = {}
    d_pt = cp.coil_point_values(table, currents, mast["ip__mean"].to_numpy(), res.r0_m, res.scales,
                                config.min_abs_ip_ka)
    for j, c in enumerate(coil_cols):
        delta[c] = d_pt[:, j]
    d_terms = d_pt @ cp.term_operator(config.layout).T
    for j, c in enumerate(term_cols):
        delta[c] = d_terms[:, j]
    delta["shaping"] = 0.5 * (delta["coil_c02"] - delta["coil_c20"])
    delta["shafranov_lambda"] = -4.0 * np.pi * (res.r0_m / res.a0_m) * delta["coil_c10"]
    delta["ohmic_flux_used"] = (IMPACT_FACTOR - 1.0) * mast["ohmic_flux_used"].to_numpy()
    scaler = RobustScaler.load(features.scaler_path(FEATURE_VERSION))
    d3d = feats[SPLITS["DIII-D"]]
    rows = []
    for c in [*coil_cols, *term_cols, *other]:
        centre, iqr = scaler.statistics(c, "MAST")
        before = mast[c].to_numpy(dtype=np.float64)
        after = before + delta[c]
        lo, hi = np.nanpercentile(d3d[c], [1, 99])
        fin = np.isfinite(before)
        rows.append({"feature": c, "iqr": iqr, "in_set_a": c in config.set_columns("set_a"),
                     "in_set_b": c in config.set_columns("set_b"),
                     "median_abs_shift_over_iqr": float(np.nanmedian(np.abs(delta[c])) / iqr),
                     "p95_abs_shift_over_iqr": float(np.nanpercentile(np.abs(delta[c]), 95) / iqr),
                     "mast_scaled_median_f1": float(np.nanmedian((before - centre) / iqr)),
                     f"mast_scaled_median_f{IMPACT_FACTOR:g}": float(np.nanmedian((after - centre) / iqr)),
                     "d3d_test_scaled_median": float(np.nanmedian((d3d[c] - centre) / iqr)),
                     "mast_outside_d3d_p1_p99_f1": float(np.mean((before[fin] < lo) | (before[fin] > hi))),
                     f"mast_outside_d3d_p1_p99_f{IMPACT_FACTOR:g}": float(np.mean((after[fin] < lo) | (after[fin] > hi)))})
    table_out = pd.DataFrame(rows)
    # Map change: (f - 1) * sol * pattern on every frame, against the predicted maps.
    grid = GreenTable.load(DEFAULT_GREEN_DIR / GREEN_FILES["MAST"])
    pattern = grid.patterns[list(grid.columns).index(MAST.solenoid_column)]
    sol = np.nan_to_num(frames["sol__mean"].to_numpy(dtype=np.float64))
    ss_delta = float((IMPACT_FACTOR - 1.0) ** 2 * np.sum(sol ** 2) * np.sum(pattern ** 2))
    mean_delta = float((IMPACT_FACTOR - 1.0) * np.mean(sol) * np.mean(pattern))
    n_elements = len(frames) * pattern.size
    summary = {"factor": IMPACT_FACTOR, "frames": int(len(frames)), "feature_rows_matched": int(len(mast)),
               "map_ss_delta": ss_delta, "map_ss_delta_about_its_mean": ss_delta - n_elements * mean_delta ** 2,
               "map_rms_delta_wb_per_rad": float(np.sqrt(ss_delta / n_elements)),
               "solenoid_pattern_at_ref_wb_per_rad_per_ka": float(
                   pattern[np.argmin(np.abs(grid.grid_z)), np.argmin(np.abs(grid.grid_r - MAST.major_radius.value))])}
    if predictions is not None and Path(predictions).is_file():
        total, total_sq, n = 0.0, 0.0, 0
        with np.load(predictions, mmap_mode="r") as z:
            for k in sorted(k for k in z.files if k.endswith("_psirz")):
                p = np.asarray(z[k], dtype=np.float64)
                total += p.sum()
                total_sq += np.sum(p ** 2)
                n += p.size
        ss_pred = total_sq - total ** 2 / n
        summary.update(predictions=str(predictions), prediction_elements=int(n), prediction_ss_about_mean=float(ss_pred),
                       map_ss_delta_over_prediction_ss=float(ss_delta / ss_pred),
                       elements_match_frames=bool(n == n_elements))
    return summary, table_out



# -------------------------------------------------------------------------------- print
def _fmt(d: dict, keys=("n", "p05", "p25", "median", "p75", "p95")) -> str:
    return "  ".join(f"{k} {d[k]:.3f}" if isinstance(d.get(k), float) else f"{k} {d.get(k)}" for k in keys if k in d)


def print_summary(summary: dict, impact_table: pd.DataFrame | None) -> None:
    print("\n== 1. Geometry and units")
    for m, g in summary["geometry"].items():
        print(f"{m}: solenoid elements {g['solenoid_elements']}, layers {len(g['layers_r_m'])} x {g['per_layer']}, "
              f"Z {g['z_min_m']:.3f} to {g['z_max_m']:.3f} m, element {g['element_width_m']} x {g['element_height_m']} m, "
              f"duplicates {g['duplicate_positions']}")
        print(f"   psi per kA at R0 {g['model_psi_ref_wb_per_rad_per_ka']:.4e}, at R0-a0 "
              f"{g['model_psi_inner_wb_per_rad_per_ka']:.4e} Wb/rad; long-solenoid "
              f"{g.get('long_solenoid_psi_wb_per_rad_per_ka', float('nan')):.4e}")
    for m in SPLITS:
        r = summary["current_ranges"][m]
        print(f"{m}: sol record min [{_fmt(r['record_min_ka'])}]")
        print(f"   sol record max [{_fmt(r['record_max_ka'])}]; extremes {r['extreme_min_ka']:.1f} / {r['extreme_max_ka']:.1f} kA")
        print(f"   sol at start [{_fmt(r['start_ka'])}]; at flat-top [{_fmt(r['flat_top_ka'])}]")
        print(f"   span p1-p99 {r['swing_span_ka_p1_p99']:.1f} kA -> swing at R0 {r['swing_wb_at_ref']}, "
              f"at R0-a0 {r['swing_wb_at_inner_edge']} Wb")
    print(f"published MAST swing {summary['current_ranges']['published_mast_swing_wb']} Wb")
    print("\n== 2. Flux consumption (consumed coil flux over mu0 R0 Ip)")
    b = summary["balance"]
    print(f"inductance over mu0 R0: {b['inductance_over_mu0_r0']}")
    for key, case in b["cases"].items():
        print(f"-- case {key}")
        for m in SPLITS:
            c = case[m]
            print(f"  {m}: c_sol [{_fmt(c['c_sol'], ('n', 'p25', 'median', 'p75'))}]  c_pf [{_fmt(c['c_pf'], ('median',))}]  "
                  f"need [{_fmt(c['need'], ('median',))}]")
            print(f"  {m}: C_E solenoid-only [{_fmt(c['ce_solenoid_only'], ('p25', 'median', 'p75'))}]  "
                  f"C_E total [{_fmt(c['ce_total'], ('p25', 'median', 'p75'))}]")
        for k in ("ratio_mast_over_d3d_ce_solenoid_only", "ratio_mast_over_d3d_ce_total"):
            r = case[k]
            print(f"  {k}: {r.get('ratio', float('nan')):.3f} (90% {r.get('p05', float('nan')):.3f} to {r.get('p95', float('nan')):.3f})")
        for k in ("mast_f_star_solenoid_only", "mast_f_star_total"):
            for cls, d in case[k].items():
                print(f"  {k} [{cls}]: {_fmt(d, ('n', 'p25', 'median', 'p75'))}")
    print("\n== 3. P-coil checks")
    for m, p in summary["pcoil"].items():
        print(f"{m}: Bz ratio sol [{_fmt(p['bz_ratio_sol'], ('median',))}]  pf [{_fmt(p['bz_ratio_pf'], ('p25', 'median', 'p75'))}]  "
              f"total f=1 [{_fmt(p['bz_ratio_total_f1'], ('p25', 'median', 'p75'))}]  force balance {p['force_balance_band']}")
        if m == "MAST":
            print(f"   total f=0.5 [{_fmt(p['bz_ratio_total_f0.5'], ('p25', 'median', 'p75'))}]")
            nl = p["null"]
            print(f"   null shots {nl['n_shots']}: f* [{_fmt(nl['f_star'])}]")
            print(f"   null rms gauss f=1 [{_fmt(nl['rms_gauss_f1'], ('median',))}]  f=0.5 [{_fmt(nl['rms_gauss_f0.5'], ('median',))}]"
                  f"  solenoid alone [{_fmt(nl['rms_gauss_sol_only'], ('median',))}]")
            print(f"   null f* on merging-compression shots [{_fmt(p['null_merging_compression'], ('n', 'p25', 'median', 'p75'))}]")
            nl = p["null_pre"]
            print(f"   null {NULL_PRE_MS:g} ms before the start, P3 idle, shots {nl['n_shots']}: f* [{_fmt(nl['f_star'])}]")
            print(f"   null rms gauss f=1 [{_fmt(nl['rms_gauss_f1'], ('median',))}]  f=0.5 [{_fmt(nl['rms_gauss_f0.5'], ('median',))}]"
                  f"  solenoid alone [{_fmt(nl['rms_gauss_sol_only'], ('median',))}]")
    print("\n== 0. Shots")
    print(summary["shots"])
    if "impact" in summary:
        print("\n== 4. Impact of the solenoid factor on MAST")
        print({k: v for k, v in summary["impact"].items() if k != "predictions"})
        with pd.option_context("display.width", 200, "display.max_columns", 20, "display.precision", 3):
            print(impact_table.to_string(index=False))


# --------------------------------------------------------------------------------- main
def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data-root", type=Path, default=dataset.DEFAULT_DATA_ROOT)
    parser.add_argument("--green-dir", type=Path, default=DEFAULT_GREEN_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-shots", type=int, default=None, help="First N shots per machine (sorted), for a smoke run")
    parser.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 2)))
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS,
                        help="MAST submission npz for the map comparison; the comparison is skipped if absent")
    parser.add_argument("--no-impact", action="store_true", help="Skip section 4")
    args = parser.parse_args(argv)

    started = time.time()
    tables = {m: machine_tables(m, args.data_root, args.green_dir) for m in SPLITS}
    geometry = geometry_summary(tables)
    jobs = []
    for m, split in SPLITS.items():
        paths = dataset.list_shots(args.data_root, split)
        jobs += [(m, str(p)) for p in (paths if args.max_shots is None else paths[:args.max_shots])]
    print(f"{len(jobs)} shots, {args.workers} workers")
    worker_tables = {m: {k: v for k, v in t.items() if k != "geometry"} for m, t in tables.items()}
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker, initargs=(worker_tables,)) as pool:
            rows = list(pool.map(_worker, jobs, chunksize=8))
    else:
        _init_worker(worker_tables)
        rows = [_worker(j) for j in jobs]
    shots = pd.DataFrame(rows).sort_values(["machine", "shot"], kind="stable").reset_index(drop=True)
    if "merging_compression" in shots:
        shots["merging_compression"] = shots["merging_compression"].fillna(False).astype(bool)
    rng = np.random.default_rng(SEED)
    summary = {"script": "scripts/mast_solenoid_check.py", "splits": SPLITS,
               "parameters": {"window_ms": WINDOW_MS, "rise_fraction": RISE_FRACTION, "start_fraction": START_FRACTION,
                              "start_lead_ms": START_LEAD_MS, "flat_top_fraction": FLAT_TOP_FRACTION,
                              "li": LI_NOMINAL, "kappa": KAPPA_NOMINAL, "p3_active_ka": P3_ACTIVE_KA,
                              "null_region_r_m": NULL_REGION_R, "null_region_z_m": NULL_REGION_Z,
                              "r0_a0_m": {m: [machine_config(m).major_radius.value, machine_config(m).minor_radius.value]
                                          for m in SPLITS},
                              "coil_model": {m: t["scales"].to_dict() for m, t in tables.items()}},
               "shots": {m: {"total": int((shots["machine"] == m).sum()),
                             "used": int(((shots["machine"] == m) & (shots["error"] == "")).sum()),
                             "errors": shots.loc[(shots["machine"] == m) & (shots["error"] != ""), "error"]
                             .value_counts().to_dict()} for m in SPLITS},
               "geometry": geometry, "current_ranges": current_ranges(shots, geometry)}
    mast_ok = shots[(shots["machine"] == "MAST") & (shots["error"] == "")]
    summary["shots"]["MAST"]["merging_compression"] = int(mast_ok["merging_compression"].sum())
    summary["balance"] = balance_summary(shots, rng)
    summary["pcoil"] = pcoil_summary(shots)
    impact_table = None
    if not args.no_impact:
        summary["impact"], impact_table = impact(args.predictions)
    args.out.mkdir(parents=True, exist_ok=True)
    shots.to_parquet(args.out / "shots.parquet", index=False)
    if impact_table is not None:
        impact_table.to_parquet(args.out / "impact_features.parquet", index=False)
    summary["seconds"] = round(time.time() - started, 1)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=float) + "\n")
    print_summary(summary, impact_table)
    print(f"\nwritten to {args.out} in {summary['seconds']} s")


if __name__ == "__main__":
    main()
