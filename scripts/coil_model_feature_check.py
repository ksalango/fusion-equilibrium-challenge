"""Diagnostic: the MAST test features under another coil model, beside the version's own build,
against DIII-D training (label-free confirmation of a coil-model change).

Reads the version's MAST table (``<feature-root>/v<version>/mast_public_test.parquet``) and a
variant build of it (``<feature-root>/v<version>_<variant>``, written by
``build_features.py --splits mast_public_test --coil-model MAST=<name> --variant <variant>``),
with the coil model each build records. Nothing is rebuilt, and DIII-D, the configuration and
``scaler.json`` are only read. The cross-machine checks of the feature profile are repeated on
both MAST tables against the ``train`` list of ``--split``:

1. Unchanged columns: every feature that does not read the coil currents must be identical in
   the two MAST tables (``ip_rate_rel`` among them); the coil features are listed with the
   largest change.
2. X8 ranges of every feature and of the vacuum residual (F82), all frames and by phase (the
   X9 rule: the reference restricted to the same phase).
3. X9 stretch terms and the vacuum relation of F82: ``coil_c20``, ``coil_c02``, ``shaping`` and
   ``vacuum_residual`` (median, interquartile range, share outside the training band), and the
   residual relative to ``coil_c20`` beside the same ratio for the current-free field
   R^4 - 4 R^2 Z^2.
4. X14: ``shafranov_lambda`` against ``ts_beta_p``, all frames and flat-top, with MAST's slope
   relative to DIII-D training's (F85).
5. X10 combined distance for set_a and set_b with the version's scaler (``--nn-train-frames``,
   ``--nn-eval-frames``, ``--seed``), both MAST tables on the same frames.
6. A summary of whether each MAST anomaly moves toward DIII-D: the even coil terms (F81),
   ``shaping`` and ``shafranov_lambda``, the vacuum-residual offset (F82), the X14 slope
   (F85), the X10 distances.

No equilibrium label is read: the MAST test files have none, the DIII-D rows are feature rows
and frame keys. Writes ``<report-root>/v<version>_<variant>/coil_model_check.md`` and ``.json``.
Usage::

    python scripts/coil_model_feature_check.py --variant solhalf
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

from fusion_eq import dataset, feature_checks as fc, features as fe, frame_table as ft, split as sp
from fusion_eq.coil_scales import REFERENCE_PHASES
from fusion_eq.dataset import run_info
from fusion_eq.feature_scaler import RobustScaler
from fusion_eq.machines import machine_config

DEFAULT_SPLIT = dataset.REPO_ROOT / "splits" / "split_v1.json"
DEFAULT_CONFIG = dataset.REPO_ROOT / "configs" / "features_v2.yaml"
MAST_SPLIT = "mast_public_test"
REPORT_NAME = "coil_model_check"
STRETCH = ["vacuum_residual"]
# Even coil terms under the up-down mirror (F81) and the quantities built from them.
EVEN_TERMS = ["coil_c10", "coil_c20", "coil_c02", "coil_c30", "coil_c12", "shaping", "shafranov_lambda"]
BASE, VARIANT = "mast_base", "mast_variant"
SETS = ("set_a", "set_b")


def log(message: str) -> None:
    print(message, flush=True)


def display_path(path: Path | str) -> str:
    """A path relative to the repository, so that the report carries no home path."""
    return os.path.relpath(Path(path).resolve(), dataset.REPO_ROOT)


# ------------------------------------------------------------------------------- load
def joined(features: pd.DataFrame, split: str, args, shots=None) -> pd.DataFrame:
    """Feature rows with ``n_frames`` and ``phase`` (X9 rule), keys checked against the frame table."""
    frames = ft.read_frame_table(split, ["shot", "frame"], shots=shots, root=args.table_dir)
    try:
        return fc.join_frame_columns(features, frames)
    except ValueError as exc:
        raise SystemExit(f"{split}: {exc}") from exc


def build_records(config: fe.FeatureConfig, args) -> dict:
    """Folder, configuration hash and MAST coil model of the two builds."""
    out = {}
    for key, variant in ((BASE, None), (VARIANT, args.variant)):
        folder = fe.features_dir(config.version, args.feature_root, variant)
        path = folder / "build.json"
        if not path.is_file():
            raise SystemExit(f"{path} not found")
        record = json.loads(path.read_text())
        out[key] = {"folder": display_path(folder), "config_hash": record.get("config_hash"),
                    "coil_model": fe.build_coil_model(folder, "MAST"), "repo_commit": record.get("repo_commit"),
                    "built_at": record.get("timestamp")}
    if out[BASE]["config_hash"] != out[VARIANT]["config_hash"]:
        raise SystemExit("the two builds were made with different feature configurations")
    if out[BASE]["coil_model"] == out[VARIANT]["coil_model"]:
        raise SystemExit(f"both builds record the MAST coil model {out[BASE]['coil_model']!r}; nothing to compare")
    return out


def load_tables(config: fe.FeatureConfig, args) -> dict[str, pd.DataFrame]:
    columns = config.all_columns()
    split = sp.read_split(args.split)
    d3d = joined(fe.read_features("diii_d_train", config.version, columns, None, args.feature_root), "diii_d_train", args)
    out = {"train": d3d[d3d["shot"].isin(set(split["train"]))].reset_index(drop=True),
           "validation": d3d[d3d["shot"].isin(set(split["validation"]))].reset_index(drop=True)}
    del d3d
    out[BASE] = joined(fe.read_features(MAST_SPLIT, config.version, columns, None, args.feature_root), MAST_SPLIT, args)
    out[VARIANT] = joined(fe.read_features(MAST_SPLIT, config.version, columns, None, args.feature_root, args.variant),
                          MAST_SPLIT, args)
    if not (out[BASE][["shot", "frame"]].equals(out[VARIANT][["shot", "frame"]])
            and np.array_equal(out[BASE]["phase"].to_numpy(), out[VARIANT]["phase"].to_numpy())):
        raise SystemExit("MAST rows or phases differ between the two builds")
    for key, t in out.items():
        machine = "MAST" if key in (BASE, VARIANT) else "DIII-D"
        t["vacuum_residual"] = fc.stretch_quantities(t, machine)["vacuum_residual"].to_numpy()
    return out


# ----------------------------------------------------------------------- comparisons
def coil_columns(config: fe.FeatureConfig) -> list[str]:
    """Feature columns whose function reads the coil currents (``features.coil_reads``)."""
    reads = fe.feature_reads(config, "MAST")
    coil = {"sol__mean", *[f"{c}__mean" for c in machine_config("MAST").shaping_coil_columns]}
    return [c for c, spec in config.column_specs().items() if coil & set(reads[spec.name])]


def unchanged_check(tables: dict[str, pd.DataFrame], config: fe.FeatureConfig) -> tuple[pd.DataFrame, list[str]]:
    """Largest absolute change per column between the two MAST tables; the non-coil columns
    that are not identical."""
    coil = set(coil_columns(config))
    rows, failing = [], []
    for c in config.all_columns():
        a, b = tables[BASE][c].to_numpy(dtype=np.float64), tables[VARIANT][c].to_numpy(dtype=np.float64)
        same_nan = np.array_equal(np.isnan(a), np.isnan(b))
        diff = float(np.nanmax(np.abs(a - b))) if np.isfinite(a - b).any() else 0.0
        identical = same_nan and np.array_equal(a[~np.isnan(a)], b[~np.isnan(b)])
        rows.append({"feature": c, "reads coil currents": c in coil, "identical": identical, "largest abs change": diff})
        if c not in coil and not identical:
            failing.append(c)
    return pd.DataFrame(rows), failing


def residual_ratios(tables: dict[str, pd.DataFrame], config: fe.FeatureConfig) -> pd.DataFrame:
    """Median abs(vacuum_residual) over median abs(coil_c20) by phase, beside the ratio of the
    current-free field R^4 - 4 R^2 Z^2 through the same fit at the machine's radii."""
    rows = []
    for key, label in (("train", "DIII-D train"), (BASE, "MAST, base"), (VARIANT, "MAST, variant")):
        machine = "DIII-D" if key == "train" else "MAST"
        cfg = machine_config(machine)
        r0, a0 = cfg.major_radius.value, cfg.minor_radius.value
        quartic = pd.DataFrame([fc.fitted_terms(config.layout, lambda r, z: r ** 4 - 4 * r ** 2 * z ** 2, r0, a0)])
        benchmark = abs(fc.vacuum_residual(quartic, cfg, r0, a0)[0]) / abs(quartic["coil_c20"].iloc[0])
        t = tables[key]
        for phase in fc.PHASES:
            p = t if phase == "all" else t[t["phase"] == phase]
            rows.append({"split": label, "phase": phase, "n": int(p["vacuum_residual"].notna().sum()),
                         "ratio": fc.relative_size(p["vacuum_residual"], p["coil_c20"]), "ratio for R^4 - 4 R^2 Z^2": benchmark})
    return pd.DataFrame(rows)


def stretch_statistics(tables: dict[str, pd.DataFrame], ranges: pd.DataFrame, labels: dict) -> pd.DataFrame:
    rows = []
    for q in ["coil_c20", "coil_c02", "shaping", "vacuum_residual"]:
        for key in ("train", BASE, VARIANT):
            s = fc.reference_statistics(tables[key][q])
            row = {"quantity": q, "split": labels[key], "n": s["n"], "median": s["q50"], "IQR": s["q75"] - s["q25"]}
            if key != "train":
                r = ranges[(ranges["feature"] == q) & (ranges["phase"] == "all") & (ranges["split"] == key)].iloc[0]
                row.update({"outside band": r["share_outside_band"], "median shift (IQR)": r["median_shift_iqr"]})
            rows.append(row)
    return pd.DataFrame(rows)


def x14(tables: dict[str, pd.DataFrame], labels: dict) -> pd.DataFrame:
    rows = []
    for subset in ("all", "flat_top"):
        reference = None
        for key in ("train", BASE, VARIANT):
            t = tables[key] if subset == "all" else tables[key][tables[key]["phase"] == "flat_top"]
            fit = fc.line_fit(t["ts_beta_p"], t["shafranov_lambda"])
            reference = fit if key == "train" else reference
            rows.append({"frames": subset, "split": labels[key], "n": fit["n"], "slope": fit["slope"],
                         "slope over training": fit["slope"] / reference["slope"], "intercept": fit["intercept"], "r": fit["r"],
                         "residual std": fit["residual_std"],
                         "median offset from training line": fc.median_offset(t["ts_beta_p"], t["shafranov_lambda"], reference)})
    return pd.DataFrame(rows)


def x10(tables: dict[str, pd.DataFrame], config: fe.FeatureConfig, scaler: RobustScaler, labels: dict, args):
    """Combined distance of set_a and set_b with the version's scaler, both MAST tables on the same frames."""
    train = tables["train"]
    group_of = {c: s.group for c, s in config.column_specs().items()}
    compared = {"validation": len(tables["validation"]), MAST_SPLIT: len(tables[BASE])}
    train_rows, eval_rows = fc.x10_samples(len(train), compared, args.nn_train_frames, args.nn_eval_frames, args.seed)
    eval_rows[BASE] = eval_rows[VARIANT] = eval_rows.pop(MAST_SPLIT)
    parts = []
    for name in SETS:
        columns = config.set_columns(name)
        reference = scaler.transform(train.iloc[train_rows], "DIII-D", columns)
        queries = {labels[k]: scaler.transform(tables[k].iloc[rows], "DIII-D" if k == "validation" else "MAST", columns)
                   for k, rows in eval_rows.items()}
        parts.append(fc.combined_distance(reference, queries, fc.column_groups(columns, group_of)).assign(set=name))
        log(f"  X10 {name}: {len(columns)} columns, {len(reference)} training frames")
    table = pd.concat(parts, ignore_index=True)
    info = {"n_train_sample": int(len(train_rows)), "n_eval": {labels[k]: int(len(v)) for k, v in eval_rows.items()}}
    return fc.distance_summary(table, reference_split=labels["validation"]), fc.group_share_medians(table), info


def toward(before: float, after: float) -> str:
    """``toward``, ``away`` or ``unchanged`` for a distance from DIII-D before and after."""
    if not (np.isfinite(before) and np.isfinite(after)):
        return "n/a"
    if abs(after - before) <= 1e-3 * max(abs(before), 1e-12):
        return "unchanged"
    return "toward" if abs(after) < abs(before) else "away"


def verdicts(ranges: pd.DataFrame, lines14: pd.DataFrame, ratios: pd.DataFrame, nn: pd.DataFrame, labels: dict) -> pd.DataFrame:
    """Whether each MAST anomaly moves toward DIII-D under the variant's coil model."""
    rows = []
    allp = ranges[ranges["phase"] == "all"]
    for q in [*EVEN_TERMS, "vacuum_residual", "ip_rate_rel"]:
        g = allp[allp["feature"] == q]
        b, v = g[g["split"] == BASE].iloc[0], g[g["split"] == VARIANT].iloc[0]
        rows.append({"check": f"X8 {q}: median shift (training IQR)", "base": b["median_shift_iqr"], "variant": v["median_shift_iqr"],
                     "moves": toward(b["median_shift_iqr"], v["median_shift_iqr"])})
        rows.append({"check": f"X8 {q}: share outside the training band", "base": b["share_outside_band"],
                     "variant": v["share_outside_band"], "moves": toward(b["share_outside_band"], v["share_outside_band"])})
    for subset in ("all", "flat_top"):
        g = lines14[lines14["frames"] == subset]
        b, v = g[g["split"] == labels[BASE]].iloc[0], g[g["split"] == labels[VARIANT]].iloc[0]
        rows.append({"check": f"X14 slope over DIII-D training ({subset})", "base": b["slope over training"],
                     "variant": v["slope over training"], "moves": toward(b["slope over training"] - 1.0, v["slope over training"] - 1.0)})
        rows.append({"check": f"X14 median offset from the training line ({subset})", "base": b["median offset from training line"],
                     "variant": v["median offset from training line"],
                     "moves": toward(b["median offset from training line"], v["median offset from training line"])})
    train_ratio = ratios[(ratios["split"] == "DIII-D train") & (ratios["phase"] == "all")]["ratio"].iloc[0]
    rb = ratios[(ratios["split"] == "MAST, base") & (ratios["phase"] == "all")]["ratio"].iloc[0]
    rv = ratios[(ratios["split"] == "MAST, variant") & (ratios["phase"] == "all")]["ratio"].iloc[0]
    rows.append({"check": "vacuum residual over coil_c20 (DIII-D training " + f"{train_ratio:.3f})", "base": rb, "variant": rv,
                 "moves": toward(rb - train_ratio, rv - train_ratio)})
    for name in SETS:
        g = nn[nn["set"] == name]
        b, v = g[g["split"] == labels[BASE]].iloc[0], g[g["split"] == labels[VARIANT]].iloc[0]
        rows.append({"check": f"X10 {name}: median distance", "base": b["p50"], "variant": v["p50"], "moves": toward(b["p50"], v["p50"])})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------ report
def render(info: dict, unchanged: pd.DataFrame, failing: list[str], ranges: pd.DataFrame, stretch: pd.DataFrame,
           ratios: pd.DataFrame, lines14: pd.DataFrame, nn: pd.DataFrame, shares: pd.DataFrame, verdict: pd.DataFrame,
           labels: dict) -> str:
    b = info["builds"]
    lines = [f"# MAST features under the coil model `{b[VARIANT]['coil_model']}`, beside `{b[BASE]['coil_model']}`", "",
             f"Diagnostic, label-free. MAST public test features of `{b[BASE]['folder']}` (MAST coil model "
             f"`{b[BASE]['coil_model']}`) and of `{b[VARIANT]['folder']}` (`{b[VARIANT]['coil_model']}`), same configuration "
             f"(`{info['config_hash'][:12]}`), against the `train` list of `{info['split_file']}` (DIII-D, read only). The "
             "version's scaler is used for X10 on both.", "",
             "## 1. Summary: does MAST move toward DIII-D?", "",
             "`toward` when the distance from DIII-D (median shift, share outside the band, slope ratio minus 1, distance) "
             "shrinks under the variant's coil model.", "", fc.markdown_table(verdict), ""]
    status = "pass" if not failing else "fail: " + ", ".join(failing)
    coil = unchanged[unchanged["reads coil currents"]].sort_values("largest abs change", ascending=False)
    lines += ["## 2. Columns that do not read the coil currents", "",
              f"Status **{status}**: every feature without a coil read is identical in the two MAST tables "
              f"({int((~unchanged['reads coil currents']).sum())} columns, `ip_rate_rel` among them). Coil features, largest "
              "absolute change:", "", fc.markdown_table(coil.head(12)), ""]
    allp = ranges[ranges["phase"] == "all"]
    rows = []
    for feature, g in allp.groupby("feature", sort=False):
        r0, r1 = g[g["split"] == BASE].iloc[0], g[g["split"] == VARIANT].iloc[0]
        rows.append({"feature": feature, "train 1st": r0["ref_q01"], "train 99th": r0["ref_q99"], "base median": r0["median"],
                     "variant median": r1["median"], "base outside band": r0["share_outside_band"],
                     "variant outside band": r1["share_outside_band"], "base shift (IQR)": r0["median_shift_iqr"],
                     "variant shift (IQR)": r1["median_shift_iqr"]})
    lines += ["## 3. Ranges against DIII-D training (X8), all frames", "", fc.markdown_table(pd.DataFrame(rows)), "",
              "### By phase (X9 rule), even coil terms and the vacuum residual", ""]
    rows = []
    for (feature, phase), g in ranges[(ranges["phase"] != "all") & ranges["feature"].isin([*EVEN_TERMS, "vacuum_residual"])].groupby(
            ["feature", "phase"], sort=False):
        r0, r1 = g[g["split"] == BASE].iloc[0], g[g["split"] == VARIANT].iloc[0]
        rows.append({"feature": feature, "phase": phase, "MAST frames": r0["n_finite"], "base outside band": r0["share_outside_band"],
                     "variant outside band": r1["share_outside_band"], "base shift (IQR)": r0["median_shift_iqr"],
                     "variant shift (IQR)": r1["median_shift_iqr"]})
    by_phase = pd.DataFrame(rows)
    order = {f: i for i, f in enumerate([*EVEN_TERMS, "vacuum_residual"])}
    by_phase = by_phase.assign(_f=by_phase["feature"].map(order), _p=by_phase["phase"].map(
        {p: i for i, p in enumerate(REFERENCE_PHASES)})).sort_values(["_f", "_p"], kind="stable").drop(columns=["_f", "_p"])
    lines += [fc.markdown_table(by_phase), "",
              "## 4. Stretch terms and the vacuum relation (X9, F82)", "", fc.markdown_table(stretch), "",
              "Median abs(`vacuum_residual`) over median abs(`coil_c20`), beside the current-free field R^4 - 4 R^2 Z^2 "
              "fitted the same way at the same radii:", "", fc.markdown_table(ratios), "",
              "## 5. Thomson against force balance (X14, F85)", "",
              "Line `shafranov_lambda = intercept + slope ts_beta_p` over frames inside both 1st to 99th percentile bands of "
              "each split; median offset from the DIII-D training line of the same frames.", "", fc.markdown_table(lines14), ""]
    x = info["x10"]
    beyond = f"share_beyond_{labels['validation']}_p99"
    table = nn.rename(columns={"p50": "median", "p90": "90th", "p99": "99th", beyond: "beyond validation 99th"})
    lines += ["## 6. Combined distance (X10)", "",
              f"{x['n_train_sample']} DIII-D training frames (seed {info['arguments']['seed']}); "
              + ", ".join(f"{k} {n}" for k, n in x["n_eval"].items()) + " frames (both MAST tables on the same frames).", "",
              fc.markdown_table(table[["set", "split", "n", "median", "90th", "99th", "max", "beyond validation 99th"]]), "",
              "Median share of the squared distance per feature group:", "",
              fc.markdown_table(shares.rename(columns={c: c[6:] for c in shares.columns if c.startswith("share_")})), "",
              f"Runtime {info['runtime_s']:.0f} s.", ""]
    return "\n".join(lines)


# -------------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--variant", default="solhalf", help="variant build v<version>_<variant> of the MAST features")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--feature-root", type=Path, default=fe.DEFAULT_FEATURE_ROOT)
    parser.add_argument("--report-root", type=Path, default=fe.DEFAULT_REPORT_ROOT)
    parser.add_argument("--table-dir", type=Path, default=ft.DEFAULT_TABLE_ROOT)
    parser.add_argument("--nn-train-frames", type=int, default=200_000)
    parser.add_argument("--nn-eval-frames", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    started = time.time()
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    config = fe.FeatureConfig.load(args.config)
    builds = build_records(config, args)
    labels = {"train": "DIII-D train", "validation": "DIII-D validation",
              BASE: f"MAST, {builds[BASE]['coil_model']}", VARIANT: f"MAST, {builds[VARIANT]['coil_model']}"}
    log(f"MAST coil models: {builds[BASE]['folder']} {builds[BASE]['coil_model']}, "
        f"{builds[VARIANT]['folder']} {builds[VARIANT]['coil_model']}")
    scaler = RobustScaler.load(fe.scaler_path(config.version, args.report_root))
    tables = load_tables(config, args)
    unchanged, failing = unchanged_check(tables, config)
    columns = [*config.all_columns(), *STRETCH]
    compared = (BASE, VARIANT)
    ranges = fc.compare_ranges_by_phase(tables["train"][columns], tables["train"]["phase"],
                                        {k: tables[k][columns] for k in compared}, {k: tables[k]["phase"] for k in compared},
                                        columns)
    ratios = residual_ratios(tables, config)
    stretch = stretch_statistics(tables, ranges, labels)
    lines14 = x14(tables, labels)
    log("X8, X9, F82 and X14 done")
    nn, shares, x10_info = x10(tables, config, scaler, labels, args)
    verdict = verdicts(ranges, lines14, ratios, nn, labels)
    info = {"version": config.version, "config_hash": config.hash(), "split_file": display_path(args.split), "builds": builds,
            "x10": x10_info, "unchanged_failing": failing,
            **run_info(argv, vars(args), timestamp, time.time() - started)}
    out_dir = Path(args.report_root) / fe.features_dir(config.version, args.feature_root, args.variant).name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{REPORT_NAME}.md").write_text(render(info, unchanged, failing, ranges, stretch, ratios, lines14, nn, shares,
                                                      verdict, labels))
    (out_dir / f"{REPORT_NAME}.json").write_text(json.dumps({**info, "verdicts": verdict.to_dict(orient="records"),
                                                             "x14": lines14.to_dict(orient="records"),
                                                             "residual_ratios": ratios.to_dict(orient="records")},
                                                            indent=2, default=fc.json_default) + "\n")
    log(verdict.to_string(index=False))
    log(f"wrote {out_dir / (REPORT_NAME + '.md')} in {time.time() - started:.0f} s")
    return 1 if failing else 0


if __name__ == "__main__":
    raise SystemExit(main())
