"""Boundary diagnostic of predicted MAST maps, without any MAST label
(``fusion_eq.diagnostics.boundary``).

Three map sets per frame: the MAST maps of two submissions (``--arm-g``, ``--set-b``, float16 as
written by the submission writer) and a physics-only control computed here, the mean-current map
(``c = 0``) on the coil flux of the ``sol_half`` coil model through ``compose_map``. For every
frame and set, the organizers' boundary and axis functions (machine set to MAST) give the
boundary, and the module records its geometry, axis, saddles, limiting and failure. Flags F1 to
F5 use the frozen thresholds of ``--config``; per shot, the share of flagged frames, the phase
breakdown and input measures from the frame table and ``--features`` (Ip, ip_rate_rel, solenoid
swing, P-coil ranges, clip counts under each arm's own scaler and clip bound). With
``--known-shots`` (shot ids, or a CSV with ``shot`` and optionally ``d_lcfs`` from the Codabench
worst-shot tables), the overlap of each flag with those shots is reported as a reading, after
the flags are computed; the known shots never enter a threshold. Known shots are given as
submission keys (``shot_0678``, mapped to MAST shot numbers through the writing order recorded in
the submission's ``run.json``, the order the writer used) or as MAST shot numbers; the run stops
when none matches. Every known shot that was analysed is plotted, in order of ``d_lcfs``, with a
per-frame table of all three sets.

Flags follow ``--config`` (version 2: F2 is the change of side of the defining saddle only);
the flags of ``--compare-config`` (version 1, with the near-double-null bound) are computed from
the same analysis and reported beside them. The two configurations must agree on every entry the
analysis reads.

Inputs refused: the labelled MAST sample shots, raw shot files, and any Parquet file with a
label column (``boundary.refuse_label_paths``). Writes to ``--out``
(``reports/diagnostics/mast_boundary_v1/``): ``report.md``, ``per_frame.parquet`` (one row per
set and frame), ``per_shot.csv``, ``figures/``. Usage::

    python scripts/diag_mast_boundary.py --stride 8 --shots 40      # quick run, first 40 shots
    python scripts/diag_mast_boundary.py --known-shots known.csv     # every frame of every shot

``known.csv``: a header line with ``key`` (or ``shot``) and optionally ``d_lcfs``, one known
shot per line, or one key per line without a header.
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

from fusion_eq import features as fe, frame_table as ft, starter_kit, target_build as tb, targets
from fusion_eq.coil_flux import GreenTable, coil_model
from fusion_eq.dataset import DEFAULT_PROCESSED_ROOT, REPO_ROOT, run_info
from fusion_eq.diagnostics import boundary as bd
from fusion_eq.feature_scaler import RobustScaler
from fusion_eq.machines import machine_config

SPLIT = "mast_public_test"
DEFAULT_ARMS = {"arm_g": "2026-10-15_arm_g_solhalf", "set_b": "2026-10-15_arm_setb_g_solhalf"}
DEFAULT_CONFIG = REPO_ROOT / "configs" / "diag_mast_boundary_v2.yaml"
DEFAULT_COMPARE = REPO_ROOT / "configs" / "diag_mast_boundary_v1.yaml"
DEFAULT_OUT = REPO_ROOT / "reports" / "diagnostics" / "mast_boundary_v2"
CONTROL_COIL_MODEL = "sol_half"
COIL_COLUMNS = tuple(f"magnetics_p{i}{s}_current__mean" for i in range(2, 7) for s in "lu")
FRAME_COLUMNS = ("time_ms", "n_frames", "ip_over_peak", "ip__mean", "sol__mean", "tf__mean", *COIL_COLUMNS)
# Figure selection (not thresholds): board D_LCFS ranges of the two failure clusters
# (boundary.D_LCFS_GROUPS), used only to choose among flagged shots when --known-shots carries
# d_lcfs, and the shot counts.
CLUSTERS = (("cluster 0.6-0.86 m", 0.60, 0.86, 6), ("cluster 0.3-0.45 m", 0.30, 0.45, 4))
KNOWN_TABLE_COLUMNS = (("boundary_type", "type"), ("a", "a"), ("r0", "R0"), ("kappa", "kappa"), ("dpsi_sep", "dpsi_sep"),
                       ("limited_on", "limited"))
N_GOOD = 4
FIGURE_FRAMES = (0.1, 0.5, 0.9)
SET_STYLE = {"arm_g": ("tab:red", "-"), "set_b": ("tab:blue", "--"), "control": ("k", ":")}

_W: dict = {}


def log(message: str) -> None:
    print(message, flush=True)


# ---------------------------------------------------------------------------- inputs
def npz_of(path: Path) -> Path:
    return path / f"{SPLIT}.npz" if path.is_dir() else path


def shot_order(record: dict) -> list[str]:
    checks = record["folds"]["MAST"]["shot_checks"]
    if [int(c["index"]) for c in checks] != list(range(len(checks))):
        raise ValueError("shot_checks indices are not 0 to n - 1 in order")
    return [str(c["shot"]) for c in checks]


def clip_setup(record: dict, runs_root: Path) -> dict:
    """The arm's model columns, its stored scaler (checked against the run record) and its
    clip bound (training run record)."""
    ident = record["predictor_identifiers"]
    version = int(ident["feature_version"])
    path = fe.scaler_path(version)
    sha = __import__("hashlib").sha256(path.read_bytes()).hexdigest()
    if sha != ident["scaler_sha256"]:
        raise ValueError(f"{path} differs from the scaler of the submission ({ident['scaler_sha256'][:12]})")
    run = json.loads((runs_root / ident["runs"][0] / "run.json").read_text())
    return {"columns": list(ident["columns"]), "scaler": RobustScaler.load(path),
            "bound": float(run["config"]["loader"]["clip_bound"]), "run": ident["runs"][0]}


def read_known(spec: str | None) -> pd.DataFrame:
    """Known-shot entries (``key``, ``d_lcfs``) from a file (CSV with a ``key`` or ``shot``
    column and optional ``d_lcfs``, or one entry per line) or a comma-separated list. Entries are
    submission keys (``shot_0678``) or MAST shot numbers; ``boundary.map_known`` resolves them."""
    if not spec:
        return pd.DataFrame(columns=["key", "d_lcfs"])
    path = Path(spec)
    if path.is_file():
        bd.refuse_label_paths([path])
        text = path.read_text()
        if "," in text.splitlines()[0]:
            known = pd.read_csv(path, dtype=str)
            known = known.rename(columns={"shot": "key"}) if "key" not in known else known
        else:
            known = pd.DataFrame({"key": [s.strip() for s in text.split() if s.strip()]})
    else:
        known = pd.DataFrame({"key": [s.strip() for s in spec.split(",") if s.strip()]})
    known["d_lcfs"] = pd.to_numeric(known["d_lcfs"], errors="coerce") if "d_lcfs" in known else np.nan
    return known[["key", "d_lcfs"]]


# --------------------------------------------------------------------------- workers
def _init(npz_paths: dict, green: str, targets_dir: str, config: str) -> None:
    import warnings

    warnings.simplefilter("ignore")
    thr = bd.load_thresholds(config)
    env = starter_kit.envelope_mask(bd.MACHINE)
    table = GreenTable.load(green)
    _W.update(npz={k: np.load(v, allow_pickle=False) for k, v in npz_paths.items()}, thr=thr, env=env,
              wall=bd.scorer_wall(env, thr.centre_column_max_r_m), table=table,
              scales=coil_model(bd.MACHINE, table.columns, CONTROL_COIL_MODEL), objects=tb.TargetObjects.load(targets_dir))


def set_maps(shot: str, index: int, rows: pd.DataFrame) -> dict[str, np.ndarray]:
    """The three map sets of the given frames of one shot, aligned to ``rows``."""
    out = {}
    n = int(rows["n_frames"].iloc[0])
    for name, npz in _W["npz"].items():
        maps = bd.read_submission_maps(npz, index)
        if maps.shape[0] != n:
            raise ValueError(f"{shot}: {maps.shape[0]} maps in {name} for {n} frames")
        out[name] = maps[rows["frame"].to_numpy()]
    out["control"] = bd.control_maps(rows, _W["table"], _W["scales"], _W["objects"])
    return out


def _analyze_shot(shot: str, index: int, rows: pd.DataFrame) -> list[dict]:
    env, thr = _W["env"], _W["thr"]
    maps = set_maps(shot, index, rows)
    out = []
    for name, stack in maps.items():
        for k, frame in enumerate(rows["frame"].to_numpy()):
            rec, _ = bd.analyze_frame(stack[k], env.grid_R, env.grid_Z, env, _W["wall"], thr)
            out.append({"set": name, "shot": shot, "frame": int(frame), **rec})
    return out


# --------------------------------------------------------------------------- figures
def plot_shot(shot: str, index: int, rows: pd.DataFrame, flags: pd.DataFrame, group: str, out_dir: Path,
              prefix: str = "") -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    env, thr, wall = _W["env"], _W["thr"], _W["wall"]
    picks = sorted({int(round(q * (len(rows) - 1))) for q in FIGURE_FRAMES})
    sub = rows.iloc[picks]
    maps = set_maps(shot, index, sub)
    fig, axes = plt.subplots(1, len(picks), figsize=(4.2 * len(picks), 6.5), squeeze=False)
    for ax, k in zip(axes[0], range(len(picks))):
        psi = maps["arm_g"][k]
        ax.contour(env.grid_R, env.grid_Z, psi, 30, linewidths=0.6, cmap="viridis")
        ax.plot(*wall.points.T, color="0.5", lw=1)
        ax.plot(*wall.points[wall.centre_column].T, ".", color="0.3", ms=1.5)
        for name in bd.SET_NAMES:
            rec, contour = bd.analyze_frame(maps[name][k], env.grid_R, env.grid_Z, env, wall, thr)
            color, style = SET_STYLE[name]
            if contour is not None:
                ax.plot(contour[:, 0], contour[:, 1], color=color, ls=style, lw=1.6, label=name)
            ax.plot(rec["R_axis"], rec["Z_axis"], "+", color=color, ms=10)
            if name == "arm_g":
                for r_x, z_x, inside in zip(rec["xpt_r"], rec["xpt_z"], rec["xpt_in_envelope"]):
                    ax.plot(r_x, z_x, "x", color="m" if inside else "0.6", ms=7)
        frame = int(sub["frame"].iloc[k])
        ax.set_title(f"frame {frame}, t {sub['time_ms'].iloc[k]:.0f} ms")
        ax.set_aspect("equal")
        ax.set_xlabel("R (m)")
    axes[0][0].set_ylabel("Z (m)")
    axes[0][0].legend(loc="lower right", fontsize=7)
    fig.suptitle(f"{shot} ({group}): arm_g contours; x saddles (magenta inside the envelope); grey wall, dots centre column")
    fig.tight_layout()
    names = [f"{prefix}{shot}_maps.png"]
    fig.savefig(out_dir / names[0], dpi=90)
    plt.close(fig)

    fig, axes = plt.subplots(6, 1, figsize=(8, 12), sharex=True)
    t_of = dict(zip(rows["frame"], rows["time_ms"]))
    for name in bd.SET_NAMES:
        f = flags[(flags["set"] == name) & (flags["shot"] == shot)].sort_values("frame")
        t = f["frame"].map(t_of)
        color, style = SET_STYLE[name]
        for ax, col in zip(axes[:4], ("a", "r0", "kappa", "dpsi_sep")):
            ax.plot(t, f[col], color=color, ls=style, label=name)
            ax.set_ylabel(col)
            ax.ticklabel_format(axis="y", useOffset=False)
        side = f["def_xpt_side"].map({"upper": 1, "lower": -1}).where(f["boundary_type"] == "diverted")
        axes[4].plot(t, side, "o", color=color, ms=3)
    if thr.dpsi_sep_max is not None:
        axes[3].axhspan(-thr.dpsi_sep_max, thr.dpsi_sep_max, color="0.9")
    axes[4].set_yticks([-1, 1], ["lower", "upper"])
    axes[4].set_ylabel("defining X")
    axes[5].plot(rows["time_ms"], rows["ip__mean"] / 100, "k", label="Ip / 100 (kA)")
    for c in COIL_COLUMNS:
        if np.nanmax(np.abs(rows[c])) > thr.coil_active_ka:
            axes[5].plot(rows["time_ms"], rows[c], lw=0.8, label=c.replace("magnetics_", "").replace("_current__mean", ""))
    axes[5].plot(rows["time_ms"], rows["sol__mean"], lw=0.8, label="sol")
    axes[5].legend(fontsize=6, ncol=4)
    axes[5].set_xlabel("time (ms)")
    axes[0].legend(fontsize=7)
    fig.suptitle(f"{shot} ({group})")
    fig.tight_layout()
    names.append(f"{prefix}{shot}_traces.png")
    fig.savefig(out_dir / names[1], dpi=90)
    plt.close(fig)
    return names


def pick_figure_shots(per_shot: pd.DataFrame, known: pd.DataFrame) -> list[tuple[str, str]]:
    """Shots to plot: shots flagged (F1, F2 or F3) in both arms, ranked by the share of F5
    frames, then by the arms' mean share of flagged frames, then by id; known shots and their
    d_lcfs only choose among them. Unflagged shots are spread over the id order."""
    both = per_shot[per_shot["shot_arm_g_any_flag"] & per_shot["shot_set_b_any_flag"]].assign(
        arm_share=lambda d: 0.5 * (d["frac_arm_g_any_flag"] + d["frac_set_b_any_flag"]))
    flagged = both.sort_values(["frac_F5", "arm_share", "shot"], ascending=[False, False, True])["shot"].tolist()
    d = dict(zip(known["shot"], known["d_lcfs"]))
    picks, used = [], set()
    for label, lo, hi, n in CLUSTERS:
        chosen = [s for s in flagged if s not in used and lo <= d.get(s, np.nan) <= hi][:n]
        chosen += [s for s in flagged if s not in used and s not in chosen][:n - len(chosen)]
        picks += [(s, label if lo <= d.get(s, np.nan) <= hi else "flagged") for s in chosen]
        used |= set(chosen)
    any_cols = [c for c in per_shot.columns if c.startswith("shot_")]
    clean = per_shot[~per_shot[any_cols].any(axis=1)]["shot"].sort_values().tolist()
    if clean:
        picks += [(clean[int(i)], "unflagged") for i in np.linspace(0, len(clean) - 1, min(N_GOOD, len(clean)))]
    return picks


# ------------------------------------------------------------------------------ flags
FLAG_COLUMNS = (*bd.FLAGS, "any_flag", "F4", "F5", "xpt_switch")


def flag_frames(raw: pd.DataFrame, thr: bd.Thresholds) -> pd.DataFrame:
    """F1 to F5 of every set's frames under one configuration."""
    flagged = pd.concat([bd.set_flags(raw[raw["set"] == n].reset_index(drop=True), thr) for n in bd.SET_NAMES],
                        ignore_index=True)
    pair = bd.arm_flags(*(flagged[flagged["set"] == n] for n in bd.SET_NAMES), thr)
    return flagged.merge(pair, on=["shot", "frame"], how="left")


def shot_tables(flagged: pd.DataFrame, thr: bd.Thresholds, suffix: str = "") -> pd.DataFrame:
    """Per shot, the share of frames per flag and the shot flags of every set and of the arm pair;
    column names carry ``suffix`` (``frac_arm_g_F1_v1``)."""
    out = None
    for name in bd.SET_NAMES:
        f = flagged[flagged["set"] == name].assign(fail=lambda d: ~d["found"].astype(bool))
        t = bd.shot_flag_table(f, [*bd.FLAGS, "any_flag", "fail"], thr).set_index("shot")
        t.columns = [c.replace("frac_", f"frac_{name}_").replace("shot_", f"shot_{name}_") for c in t.columns]
        out = t if out is None else out.join(t)
    out = out.join(bd.shot_flag_table(flagged[flagged["set"] == "arm_g"], ["F4", "F5"], thr).set_index("shot"))
    return out.add_suffix(suffix)


def flag_share_tables(flagged: pd.DataFrame, per_shot: pd.DataFrame, suffix: str = "") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Frame share per flag and set, and shots flagged per flag, for one configuration."""
    rows = [{"set": n, **{c: flagged.loc[flagged["set"] == n, c + suffix].mean() for c in (*bd.FLAGS, "any_flag")}}
            for n in bd.SET_NAMES]
    arm = flagged[flagged["set"] == "arm_g"]
    rows.append({"set": "arm pair", "F4": arm["F4" + suffix].mean(), "F5": arm["F5" + suffix].mean()})
    cols = [c for c in per_shot.columns if c.startswith("shot_") and c.endswith(suffix) and
            (suffix or not c.endswith(tuple(f"_v{v}" for v in range(1, 10))))]
    shots = pd.DataFrame([{"flag": c[5:len(c) - len(suffix)] if suffix else c[5:], "shots": int(per_shot[c].sum())}
                          for c in cols])
    return pd.DataFrame(rows), shots


def known_frame_table(flagged: pd.DataFrame, shot: str) -> pd.DataFrame:
    """Per frame of one shot: time, phase and, for every set, boundary type, a, R0, kappa,
    dpsi_sep and the limiting surface."""
    base = None
    for name in bd.SET_NAMES:
        f = flagged[(flagged["set"] == name) & (flagged["shot"] == shot)].sort_values("frame")
        part = f[["frame", "time_ms", "phase"]].copy() if base is None else f[["frame"]].copy()
        for col, label in KNOWN_TABLE_COLUMNS:
            part[f"{name} {label}"] = f[col].replace("", "-").to_numpy() if col == "limited_on" else f[col].to_numpy()
        base = part if base is None else base.merge(part, on="frame")
    return base


def known_order(known: pd.DataFrame) -> pd.DataFrame:
    """Matched known shots ordered by d_lcfs (missing last), then key, with their rank."""
    k = known[known["matched"]].copy()
    k["_d"] = pd.to_numeric(k["d_lcfs"], errors="coerce").fillna(np.inf)
    k = k.sort_values(["_d", "key"]).drop(columns="_d").reset_index(drop=True)
    k["rank"] = np.arange(1, len(k) + 1)
    return k


# ---------------------------------------------------------------------------- report
def fmt(x, digits: int = 3) -> str:
    if isinstance(x, (bool, np.bool_)):
        return str(bool(x))
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    try:
        return "n/a" if not np.isfinite(x) else f"{x:.{digits}f}"
    except TypeError:
        return str(x)


def md_table(df: pd.DataFrame, digits: int = 3) -> list[str]:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + " --- |" * len(cols)]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(fmt(row[c], digits) for c in cols) + " |")
    return lines


def config_lines(thr: bd.Thresholds, role: str) -> list[str]:
    note = thr.note or "Written before any comparison with known shots."
    return [f"{role}: `{Path(thr.source).name}` (version {thr.version}), SHA-256 `{thr.sha256}`. {note}", ""]


def f2_text(thr: bd.Thresholds) -> str:
    if thr.dpsi_sep_max is None:
        return "F2 topology: the defining saddle changes side (upper or lower) between consecutive frames."
    return (f"F2 topology: |dpsi_sep| below {thr.dpsi_sep_max:g}, or the defining saddle changes side between "
            "consecutive frames.")


def write_report(out: Path, info: dict, thr: bd.Thresholds, cmp: bd.Thresholds | None, frames: pd.DataFrame,
                 per_shot: pd.DataFrame, compare: pd.DataFrame, phase_tab: pd.DataFrame, dpsi_tab: pd.DataFrame,
                 known: pd.DataFrame, known_rows: list[dict], overlap_tab: pd.DataFrame | None, figures: list) -> None:
    L = [f"# MAST boundary diagnostic (flags v{thr.version})", "",
         f"Run of {info['timestamp']}, repository commit {info['repo_commit']}. {info['n_shots']} shots, "
         f"{info['n_frames']} frames per set at stride {info['arguments']['stride']}. Map sets: arm_g "
         f"(`{info['inputs']['arm_g']}`), set_b (`{info['inputs']['set_b']}`), control (coil flux of "
         f"{CONTROL_COIL_MODEL} plus Ip phi_mean, c = 0, no remainder). No MAST equilibrium label read.", "",
         "## Method", "",
         "- Boundary and axis: the organizers' `extract_lcfs_with_sign` and `derive_frame`, unmodified, machine MAST, "
         f"axis sign {thr.axis_sign:+g}, on the float16 maps read as float64 (as the scorer reads them).",
         "- Wall: the 0.5 isoline of the scorer's MAST envelope mask (the boundary must stay inside it); centre column: "
         f"its points at R <= {thr.centre_column_max_r_m} m. `machines.py` holds no wall or centre column.",
         "- Saddles: zeros of the gradient of a bicubic spline of the map (Newton from minima of |grad psi|^2), "
         "classified by the Hessian; the scorer has no saddle finder. dpsi_sep uses the two saddles inside the "
         "envelope closest in flux to the boundary.",
         "- Phase: `coil_scales.phase_of` on the frame table's ip_over_peak (flat-top at or above its level of peak Ip; "
         "rising and falling by position in the shot).",
         "- Clip counts: each arm's model columns scaled by its stored DIII-D scaler, beyond its training clip bound.",
         "- Known shots: submission keys mapped to MAST shot numbers through the writing order of the submission's "
         "run record (`folds.MAST.shot_checks`); they choose what is plotted and enter the overlap only.", "",
         "## Flag definitions and frozen thresholds", ""]
    L += config_lines(thr, "Flags")
    if cmp is not None:
        L += config_lines(cmp, "Comparison")
    rec = thr.record()
    cmp_rec = cmp.record() if cmp is not None else {}
    L += md_table(pd.DataFrame([{"threshold": k, f"v{thr.version}": str(v),
                                 **({f"v{cmp.version}": str(cmp_rec[k])} if cmp is not None else {})}
                                for k, v in rec.items() if k not in ("source", "sha256", "note", "version")]))
    L += ["", "- F1 size: |robust z| of a or area above robust_z_max (over the set's frames), or a or area outside its band.",
          f"- {f2_text(thr)}" + (f" (v{cmp.version}: {f2_text(cmp)})" if cmp is not None else ""),
          "- F3 limiting: limited on the centre column, within grid_edge_m of the grid edge, or no boundary found.",
          "- F4 arm disagreement: |a_g - a_b| above da_max_m or |R0_g - R0_b| above dr0_max_m, or only one found.",
          "- F5 control agreement: F1, F2 or F3 in arm_g, in set_b and in the control.",
          f"- A shot is flagged for a flag when at least {thr.flag_fraction:g} of its frames carry it.", ""]
    L += ["## Boundaries per set", ""]
    rows = []
    for name in bd.SET_NAMES:
        f = frames[frames["set"] == name]
        rows.append({"set": name, "frames": len(f), "found": f["found"].mean(),
                     **{f"type_{t}": (f["boundary_type"] == t).mean() for t in ("diverted", "limited", "unresolved", "none")},
                     "limited_cc": (f["limited_on"] == "centre_column").mean(),
                     "fail_axis_on_rim": (f["fail_reason"] == "axis_on_rim").mean(),
                     "fail_open_at_axis": (f["fail_reason"] == "open_at_axis").mean(),
                     "two_peaks": (f["n_maxima"] > 1).mean(), "second_gap_median": f["second_max_gap"].median(),
                     "xpt_switch": f["xpt_switch"].mean(),
                     "a_median_m": f["a"].median(), "r0_median_m": f["r0"].median(), "kappa_median": f["kappa"].median()})
    L += md_table(pd.DataFrame(rows))
    configs = [(thr, "")] + ([(cmp, f"_v{cmp.version}")] if cmp is not None else [])
    for c, suffix in configs:
        share, shots = flag_share_tables(frames, per_shot, suffix)
        L += ["", f"### Flags v{c.version}", "", "Frame share per flag:", ""]
        L += md_table(share)
        L += ["", "Shots flagged per flag:", ""]
        L += md_table(shots)
    L += ["", "## dpsi_sep by set and phase", "",
          "|dpsi_sep| = |psi_X2 - psi_X1| / |psi_axis - psi_boundary| of the two saddles inside the envelope closest in "
          "flux to the boundary; with_value is the share of frames with two such saddles.", ""]
    L += md_table(dpsi_tab, 4)
    L += ["", f"## Flags v{thr.version} by phase (share of frames)", ""]
    L += md_table(phase_tab)
    L += ["", f"## Flagged against unflagged shots, flags v{thr.version} (medians of shot-level inputs)", ""]
    L += md_table(compare)
    if len(known):
        n_matched = int(known["matched"].sum())
        analysed = sum(r["analysed"] for r in known_rows)
        L += ["", "## Known shots", "",
              f"{len(known)} entries, {n_matched} matched to a shot of the submission, {analysed} analysed in this run "
              f"(the others lie outside --shots). Ordered by the board's d_lcfs; groups "
              + ", ".join(f"{lab} ({lo}-{hi} m)" for lo, hi, lab in bd.D_LCFS_GROUPS) + ".", ""]
        unmatched = known[~known["matched"]]["key"].tolist()
        if unmatched:
            L += [f"Unmatched entries: {', '.join(unmatched)}.", ""]
        if known_rows:
            L += md_table(pd.DataFrame([{k: v for k, v in r.items() if k != "analysed"} for r in known_rows]))
    if overlap_tab is not None:
        L += ["", f"## Overlap with the known shots, flags v{thr.version} and v{cmp.version if cmp else thr.version} (reading)", "",
              "Computed after the flags were frozen; no threshold was tuned to these shots.", ""]
        L += md_table(overlap_tab)
    L += ["", "## Figures (flag-based selection)", ""]
    for shot, group, names in figures:
        L += [f"- {shot} ({group}): " + ", ".join(f"[{n}](figures/{n})" for n in names)]
    L += ["", "## Readings", "", "Left for the review of the full run.", ""]
    (out / "report.md").write_text("\n".join(L))


# ------------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--arm-g", type=Path, default=REPO_ROOT / "submissions" / DEFAULT_ARMS["arm_g"])
    parser.add_argument("--set-b", type=Path, default=REPO_ROOT / "submissions" / DEFAULT_ARMS["set_b"])
    parser.add_argument("--records", type=Path, default=REPO_ROOT / "reports" / "submissions",
                        help="folder of the submissions' run.json (one sub-folder per submission name)")
    parser.add_argument("--runs-root", type=Path, default=REPO_ROOT / "reports" / "runs", help="training run records")
    parser.add_argument("--features", type=Path, default=DEFAULT_PROCESSED_ROOT / "features" / "v2_solhalf")
    parser.add_argument("--table-dir", type=Path, default=ft.DEFAULT_TABLE_ROOT)
    parser.add_argument("--green", type=Path, default=DEFAULT_PROCESSED_ROOT / "green" / "green_mast.npz")
    parser.add_argument("--targets", type=Path, default=tb.DEFAULT_TARGET_ROOT / "v1")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="thresholds of the flags")
    parser.add_argument("--compare-config", default=str(DEFAULT_COMPARE),
                        help="thresholds whose flags are reported beside (none to skip)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--stride", type=int, default=1, help="every stride-th frame of each shot")
    parser.add_argument("--shots", default=None, help="first N shots (an integer) or comma-separated shot ids")
    parser.add_argument("--known-shots", default=None,
                        help="file or comma list of submission keys (shot_0678) or shot numbers (reading only)")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 3) - 2))
    args = parser.parse_args(argv)
    t_start = time.time()
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")

    import warnings

    warnings.simplefilter("ignore")
    thr = bd.load_thresholds(args.config)
    cmp = None if str(args.compare_config).lower() == "none" else bd.load_thresholds(args.compare_config)
    if cmp is not None and cmp.analysis_record() != thr.analysis_record():
        raise ValueError(f"{args.compare_config} differs from {args.config} in entries the analysis reads; "
                         "the two flag sets cannot share one analysis")
    arms = {"arm_g": npz_of(args.arm_g), "set_b": npz_of(args.set_b)}
    features_path = args.features / f"{SPLIT}.parquet"
    table_path = ft.table_path(SPLIT, args.table_dir)
    bd.refuse_label_paths([*arms.values(), features_path, table_path])
    records = {k: json.loads((args.records / v.parent.name / "run.json").read_text()) for k, v in arms.items()}
    orders = {k: shot_order(r) for k, r in records.items()}
    if orders["arm_g"] != orders["set_b"]:
        raise ValueError("the two submissions list their MAST shots in different orders")
    order = orders["arm_g"]
    for k, r in records.items():
        model = r["predictor_identifiers"]["coil_models"]["MAST"]
        if model != CONTROL_COIL_MODEL:
            log(f"note: {k} maps use the MAST coil model {model}; the control uses {CONTROL_COIL_MODEL}")
    if args.shots is None:
        shots = order
    elif args.shots.isdigit():
        shots = order[:int(args.shots)]
    else:
        shots = [s.strip().zfill(10) for s in args.shots.split(",")]
        unknown = sorted(set(shots) - set(order))
        if unknown:
            raise ValueError(f"shots not in the submission: {unknown}")
    index = {s: i for i, s in enumerate(order)}
    entries = read_known(args.known_shots)
    known = bd.map_known(entries, order) if len(entries) else pd.DataFrame(
        {"key": pd.Series(dtype=str), "shot": pd.Series(dtype=str), "d_lcfs": pd.Series(dtype=float),
         "matched": pd.Series(dtype=bool), "group": pd.Series(dtype=str)})
    if len(known):
        log(f"Known shots: {int(known['matched'].sum())} of {len(known)} entries matched to shots of the submission")

    table = GreenTable.load(args.green)
    control_cols = targets.current_columns_of(table, bd.MACHINE)
    columns = list(dict.fromkeys([*FRAME_COLUMNS, *control_cols]))
    ft.assert_input_columns(columns, bd.MACHINE)
    frames = ft.read_frame_table(SPLIT, ["shot", "frame", *columns], shots=shots, root=args.table_dir)
    frames["shot"] = frames["shot"].astype(str)
    clip = {k: clip_setup(r, args.runs_root) for k, r in records.items()}
    feat_cols = sorted({c for v in clip.values() for c in v["columns"]} | {"ip_rate_rel"})
    feats = fe.read_features_at(features_path, feat_cols, shots=shots)
    feats["shot"] = feats["shot"].astype(str)
    frames = frames.merge(feats, on=["shot", "frame"], how="left", validate="one_to_one")
    for k, v in clip.items():
        frames[f"clip_{k}"] = bd.clip_counts(frames, v["scaler"], v["columns"], v["bound"])
    frames["phase"] = bd.phases(frames)
    strided = frames[frames.groupby("shot", sort=False).cumcount().to_numpy() % max(1, args.stride) == 0]
    n_frames = len(strided)
    log(f"{len(shots)} shots, {n_frames} frames at stride {args.stride} (of {len(frames)}), 3 map sets")

    initargs = ({k: str(v) for k, v in arms.items()}, str(args.green), str(args.targets), str(args.config))
    _init(*initargs)
    probe_shot = shots[0]
    probe_rows = strided[strided["shot"] == probe_shot]
    _analyze_shot(probe_shot, index[probe_shot], probe_rows.iloc[:1])          # warm-up: imports and caches
    t0 = time.time()
    _analyze_shot(probe_shot, index[probe_shot], probe_rows)
    per_frame_s = (time.time() - t0) / max(1, len(probe_rows))
    log(f"Runtime estimate: {per_frame_s * 1000:.0f} ms per frame for the three sets on {probe_shot}; "
        f"{per_frame_s * n_frames / 60:.1f} CPU min, about {per_frame_s * n_frames / args.workers / 60:.1f} min "
        f"on {args.workers} workers, plus figures")

    tasks = [(s, index[s], g) for s, g in strided.groupby("shot", sort=False)]
    results = []
    pool = None
    if args.workers > 1:
        pool = ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn"), initializer=_init,
                                   initargs=initargs)
        stream = pool.map(_analyze_shot, *zip(*tasks), chunksize=2)
    else:
        stream = (_analyze_shot(*t) for t in tasks)
    try:
        for k, recs in enumerate(stream, 1):
            results += recs
            if k % 50 == 0 or k == len(tasks):
                log(f"  {k}/{len(tasks)} shots ({time.time() - t_start:.0f} s)")
    finally:
        if pool is not None:
            pool.shutdown()
    raw = pd.DataFrame(results)
    keep = ["shot", "frame", "time_ms", "phase", "ip__mean", "ip_over_peak", "clip_arm_g", "clip_set_b"]
    flagged = flag_frames(raw, thr).merge(strided[keep], on=["shot", "frame"], how="left")
    cmp_suffix = ""
    if cmp is not None:
        cmp_suffix = f"_v{cmp.version}"
        other = flag_frames(raw, cmp)[["set", "shot", "frame", *FLAG_COLUMNS]]
        flagged = flagged.merge(other.rename(columns={c: c + cmp_suffix for c in FLAG_COLUMNS}),
                                on=["set", "shot", "frame"], how="left", validate="one_to_one")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "figures").mkdir(exist_ok=True)
    flagged.to_parquet(args.out / "per_frame.parquet", index=False)

    per_shot = shot_tables(flagged, thr)
    if cmp is not None:
        per_shot = per_shot.join(shot_tables(flagged.drop(columns=list(FLAG_COLUMNS)).rename(
            columns={c + cmp_suffix: c for c in FLAG_COLUMNS}), cmp, cmp_suffix))
    ph = flagged[flagged["set"] == "arm_g"].groupby("shot")["phase"].value_counts(normalize=True).unstack(fill_value=0)
    per_shot = per_shot.join(ph.add_prefix("phase_share_"))
    inputs = bd.shot_inputs(strided, COIL_COLUMNS, thr).set_index("shot")
    inputs["clip_rate_arm_g"] = strided.groupby("shot")["clip_arm_g"].apply(lambda v: float((v > 0).mean()))
    inputs["clip_rate_set_b"] = strided.groupby("shot")["clip_set_b"].apply(lambda v: float((v > 0).mean()))
    per_shot = per_shot.join(inputs).reset_index()
    if len(known):
        k = known[known["matched"]].drop_duplicates("shot").set_index("shot")
        per_shot["known_key"] = per_shot["shot"].map(k["key"]).fillna("")
        per_shot["known_d_lcfs"] = per_shot["shot"].map(k["d_lcfs"])
    per_shot.to_csv(args.out / "per_shot.csv", index=False)

    measures = ["ip_mean_ka", "ip_max_ka", "ip_rate_rel_max", "sol_swing_ka", "clip_rate_arm_g", "clip_rate_set_b"]
    primary_cols = [c for c in per_shot.columns if c.startswith("shot_") and not (cmp_suffix and c.endswith(cmp_suffix))]
    all_shot_cols = [c for c in per_shot.columns if c.startswith("shot_")]
    compare = []
    for c in primary_cols:
        f = per_shot[c].astype(bool)
        compare.append({"flag": c[5:], "n_flagged": int(f.sum()),
                        **{f"{m} flagged / not": f"{fmt(per_shot.loc[f, m].median())} / {fmt(per_shot.loc[~f, m].median())}"
                           for m in measures}})
    compare = pd.DataFrame(compare)
    phase_rows = []
    for name in bd.SET_NAMES:
        f = flagged[flagged["set"] == name]
        for phase, g in f.groupby("phase"):
            phase_rows.append({"set": name, "phase": phase, "frames": len(g),
                               **{c: g[c].mean() for c in (*bd.FLAGS, "F4", "F5")}, "fail": (~g["found"].astype(bool)).mean()})
    phase_tab = pd.DataFrame(phase_rows)
    dpsi_tab = pd.concat([bd.dpsi_sep_distribution(flagged, ("set",)).assign(phase="all"),
                          bd.dpsi_sep_distribution(flagged, ("set", "phase"))], ignore_index=True)
    dpsi_tab = dpsi_tab[["set", "phase", "frames", "with_value", "p5", "p25", "p50", "p75", "p95"]]

    analysed_shots = set(flagged["shot"])
    overlap_tab = None
    if len(known):
        hits = known.loc[known["matched"] & known["shot"].isin(analysed_shots), "shot"]
        overlap_tab = bd.overlap(per_shot, hits, all_shot_cols) if len(hits) else None
    figures, known_rows = [], []
    known_dir = args.out / "figures" / "known"
    if len(known):
        known_dir.mkdir(exist_ok=True)
    for _, kr in (known_order(known).iterrows() if len(known) else []):
        shot = kr["shot"]
        row = {"rank": int(kr["rank"]), "key": kr["key"], "shot": shot, "d_lcfs": kr["d_lcfs"], "group": kr["group"],
               "analysed": shot in analysed_shots}
        if row["analysed"]:
            f = flagged[(flagged["shot"] == shot)]
            for name in bd.SET_NAMES:
                g = f[f["set"] == name]
                row[f"{name} diverted/limited/none"] = "/".join(
                    f"{(g['boundary_type'] == t).mean():.2f}" for t in ("diverted", "limited", "none"))
            g = f[f["set"] == "arm_g"]
            row.update({"a_g median": g["a"].median(), "R0_g median": g["r0"].median(), "|dpsi_sep|_g median":
                        g["dpsi_sep"].abs().median(), "F3_g": g["F3"].mean(), "F4": g["F4"].mean(), "F5": g["F5"].mean()})
            stem = f"k{int(kr['rank']):02d}_{kr['key']}_{shot}"
            table = known_frame_table(flagged, shot)
            (known_dir / f"{stem}_frames.md").write_text(
                f"# {kr['key']} ({shot}), board d_lcfs {fmt(kr['d_lcfs'])}, {kr['group']}\n\n" + "\n".join(md_table(table)) + "\n")
            links = [f"[frames](figures/known/{stem}_frames.md)"]
            if not args.no_figures:
                names = plot_shot(shot, index[shot], frames[frames["shot"] == shot], flagged,
                                  f"{kr['key']}, d_lcfs {fmt(kr['d_lcfs'])}, {kr['group']}", known_dir, prefix=f"k{int(kr['rank']):02d}_{kr['key']}_")
                links += [f"[{n.rsplit('_', 1)[-1][:-4]}](figures/known/{n})" for n in names]
            row["files"] = ", ".join(links)
        known_rows.append(row)
    if not args.no_figures:
        for shot, group in pick_figure_shots(per_shot, known[known["matched"] & known["shot"].isin(analysed_shots)]):
            rows = frames[frames["shot"] == shot]
            names = plot_shot(shot, index[shot], rows, flagged, group, args.out / "figures")
            figures.append((shot, group, names))
    info = run_info(argv, vars(args), timestamp, round(time.time() - t_start, 1), n_shots=len(shots), n_frames=n_frames,
                    inputs={k: str(v) for k, v in arms.items()}, thresholds=thr.record(),
                    compare_thresholds=cmp.record() if cmp is not None else None,
                    known={"entries": len(known), "matched": int(known["matched"].sum()) if len(known) else 0,
                           "analysed": sum(r["analysed"] for r in known_rows)},
                    clip={k: {"run": v["run"], "bound": v["bound"], "n_columns": len(v["columns"])} for k, v in clip.items()},
                    control={"coil_model": CONTROL_COIL_MODEL, "targets": str(args.targets)},
                    machine=machine_config(bd.MACHINE).constants_record())
    write_report(args.out, info, thr, cmp, flagged, per_shot, compare, phase_tab, dpsi_tab, known, known_rows,
                 overlap_tab, figures)
    (args.out / "run.json").write_text(json.dumps(info, indent=1, default=str) + "\n")
    log(f"Wrote {args.out} ({time.time() - t_start:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
