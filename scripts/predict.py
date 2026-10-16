"""Predicts the public-test folds with a trained model and writes them through the existing
submission writer (``docs/PIPELINE.md``, Part 5).

Build (default)::

    python scripts/predict.py --predictor rung1_v1 --run <run_id> --machines d3d mast --source tables --name <label>
    python scripts/predict.py --predictor rung2_v1 --run <rung-2 run_id> --name <label>

1. The chain of the run's best checkpoint (``--run`` repeated for seed averaging, which then
   follows ``predict.seed_average`` or ``--seed-average``); mirror averaging per
   ``predict.mirror_average`` unless ``--no-mirror-average``.
2. betaN heads (F99), unless ``--heads`` names a stored file: the DIII-D ridge on the raw
   inputs and, when MAST is predicted, the MAST head refitted on the shapes of this model's maps
   of the training list at ``--refit-stride`` (stage 2, Part 5 step 4). When the betaN study's
   cached ``hybrid_ridge`` shape table exists for the same list and stride, the head fitted on
   it is the one before the refit, and both are scored on the validation list with the model's
   shapes (all frames and the MAST-like corner; check Z10). The heads are written to
   ``submissions/<date>_<name>/betan_heads.pkl`` and summarized in the run record.
3. Each fold through ``submission.predict_fold`` and ``NpzWriter`` (float16 maps, float32
   scalars, the pipeline's fallbacks), validated against the raw folds, with the MAST input
   checks; the MAST sign constant +1 is asserted before any MAST map is composed.
4. The local validation score (Part 5 step 5): the run's validation list (``--working`` for the
   working list) through the same writer, read back and scored with the organizers' scorer at
   stride 4; ``--no-score-local`` skips it.

MAST inputs and coil model: ``--mast-feature-dir`` reads the MAST features from another feature
folder (a variant build, for example ``<feature-root>/v2_solhalf`` of ``build_features.py
--variant solhalf``); the coil model of each machine's maps is the one its feature build
records, so the corrected MAST path needs only the folder::

    python scripts/predict.py --run arm_g_s0 --name arm_g_solhalf --mast-feature-dir ../data/processed/features/v2_solhalf

``--mast-coil-model NAME`` names the MAST model explicitly; it must match the record unless
``--allow-coil-model-mismatch`` is given (maps under one model, features under another: a
diagnostic). The run record names the coil model and the feature folder of each machine.

The run record ``reports/submissions/<date>_<name>/run.json`` has the layout of
``make_submission.py build``, so its ``validate``, ``push`` and ``pointer`` subcommands apply.
Nothing is submitted or pushed here.

Determinism (check Z5)::

    python scripts/predict.py --determinism --run <run_id> --heads submissions/<date>_<name>/betan_heads.pkl

predicts the D7 subset (the shots of ``scripts/check_determinism.py``, less any ``local_test``
shot) three times into a temporary directory: twice from the stored tables, once from the raw
files. Every array must be byte-identical across the two table runs and between the tables and
the raw files. Writes ``reports/determinism/prediction_check.md`` and ``.json``; exit status 1
when any array differs.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
import time
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np

from fusion_eq import (baseline as bl, coil_flux as cf, dataset, features as fe, frame_table as ft, io, loader as ld,
                       model_scoring as ms, prediction as pr, predictors, scalar_heads as sh, split as sp, starter_kit,
                       submission as sub, target_build as tb)
from fusion_eq.dataset import REPO_ROOT, default_data_root

DEFAULT_SUBMISSIONS_DIR = REPO_ROOT / "submissions"
DEFAULT_REPORTS_DIR = REPO_ROOT / "reports" / "submissions"
DEFAULT_DETERMINISM_DIR = REPO_ROOT / "reports" / "determinism"
DEFAULT_SHAPE_DIR = dataset.DEFAULT_PROCESSED_ROOT / "scalars"
RECORD_NAME = "run.json"
HEADS_NAME = "betan_heads.pkl"


def log(message: str) -> None:
    print(message, flush=True)


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(f"not JSON serialisable: {type(value).__name__}")


def save_record(report_dir: Path, record: dict) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / RECORD_NAME
    path.write_text(json.dumps(record, indent=2, default=json_default) + "\n")
    return path


# ---------------------------------------------------------------------------- inputs
def feature_dirs(args) -> dict:
    """Machine to a feature folder replacing ``<feature-root>/v<version>`` (``--mast-feature-dir``)."""
    return {} if args.mast_feature_dir is None else {"MAST": args.mast_feature_dir}


def coil_models(args) -> dict:
    """Machine to an explicitly named coil model (``--mast-coil-model``)."""
    return {} if args.mast_coil_model is None else {"MAST": args.mast_coil_model}


def chain_paths(args) -> pr.ChainPaths:
    return pr.ChainPaths(table_root=args.table_root, feature_root=args.feature_root, feature_config=args.feature_config,
                         scaler=args.scaler, targets_root=args.targets_root, green_dir=args.green_dir, data_root=args.data_root,
                         checkpoint_root=args.checkpoint_root, feature_dirs=feature_dirs(args) or None)


def run_lists(args) -> dict:
    """The training and validation lists of the first run (its run record), or the full lists."""
    path = args.run_root / args.run[0] / "run.json"
    if path.is_file():
        return json.loads(path.read_text())["lists"]
    log(f"{path} not found; the full train and validation lists are used")
    return {"train": "train", "validation": "validation"}


def table_source(chain: pr.PredictionChain, args, version: int) -> pr.TableSource:
    return pr.TableSource(args.table_root, args.feature_root, version, {m: chain.frame_columns(m) for m in pr.MACHINES},
                          chain.feature_columns(), feature_dirs(args))


def q_cyl_statistics(args, version: int) -> dict:
    """``training_statistics`` over every frame of the full training list (the band of F99)."""
    ids = sorted(sp.read_split(args.split)["train"])
    rows = fe.read_features(tb.LABELLED_SPLIT, version, ["q_cyl", "ts_beta_p"], ids, args.feature_root)
    return sh.training_statistics(rows["q_cyl"], rows["ts_beta_p"])


def study_shapes(args, train_ids: list[str]):
    """The betaN study's cached ``hybrid_ridge`` shape table of the training list, or None."""
    h = bl.hash_of_ids(train_ids)
    path = args.shape_dir / f"shapes_hybrid{h}_train_{h}_s{args.refit_stride}.parquet"
    if not path.is_file():
        return None, path
    import pandas as pd
    return pd.read_parquet(path), path


# ----------------------------------------------------------------------------- heads
def fit_heads(args, lists: dict, machines: list[str]) -> tuple[sh.BetanHeads, dict]:
    """The betaN heads of this submission and the Z10 record (see the module docstring)."""
    t0 = time.time()
    chain, record = pr.build_chain(args.run, chain_paths(args), mirror_average=args.mirror_average,
                                   seed_average=args.seed_average, device=args.device, workers=args.workers,
                                   coil_models=coil_models(args), allow_coil_model_mismatch=args.allow_coil_model_mismatch)
    version = record["feature_version"]
    source = table_source(chain, args, version)
    train_list = args.refit_list or lists["train"]
    stats = q_cyl_statistics(args, version)
    band = (stats["q_cyl"]["q01"], stats["q_cyl"]["q99"])
    seq = chain.coefficients.sequence
    marked = pr.heads_training_rows(source, args.split, train_list, args.refit_stride, mark=seq)
    rows = pr.scored(marked)
    train_ids = sorted(sp.read_split(args.split)[train_list])
    log(f"betaN heads: {len(rows)} frames of {train_list} at stride {args.refit_stride}; q_cyl band {band[0]:.3f} to {band[1]:.3f}")
    base_record = {"train_list": train_list, "stride": args.refit_stride, "q_cyl_statistics": stats}
    z10: dict = {"status": "not run (no MAST prediction)"}
    if "MAST" not in machines:
        heads = sh.fit_heads(rows, None, band, base_record)
        chain.close()
        return heads, {"z10": z10, "seconds": time.time() - t0}

    shapes = pr.model_shapes(chain, marked)
    log(f"  shapes of the model's training maps: {int(shapes['boundary_found'].sum())} of {len(shapes)} with a boundary "
        f"({time.time() - t0:.0f} s)")
    refit_record = {"checkpoints": record["checkpoints"], "mirror_average": record["mirror_average"],
                    "seed_average": record["seed_average"]}
    before_shapes, before_path = study_shapes(args, train_ids)
    if before_shapes is not None:
        before = sh.fit_heads(rows, before_shapes, band, {**base_record, "shapes": f"hybrid_ridge ({before_path.name})"})
        heads = sh.refit_mast(before, rows, shapes, refit_record)
    else:
        before = None
        heads = sh.fit_heads(rows, shapes, band, {**base_record, "shapes": "submission model (no study shape table found)"})
        heads.record["stage2_refit"] = {**refit_record, "note": f"no head before the refit: {before_path.name} not found"}

    val_list = lists["validation"]
    val_ids = sorted(sp.read_split(args.split)[val_list])
    val_marked = pr.read_rows(source, tb.LABELLED_SPLIT, val_ids, args.refit_stride, ("efit_q95", sh.LABEL), mark=seq)
    val_shapes = pr.model_shapes(chain, val_marked)
    val = pr.scored(val_marked)
    for column in ("q_cyl", "ts_beta_p"):
        if column not in val.columns:
            extra = fe.read_features(tb.LABELLED_SPLIT, version, [column], val_ids, args.feature_root)
            val = val.merge(extra.assign(shot=extra["shot"].astype(str)), on=["shot", "frame"], how="left")
    thresholds = {"q_cyl": stats["q_cyl"]["q10"], "ts_beta_p": stats["ts_beta_p"]["q50"], "ts_beta_p_relaxed": stats["ts_beta_p"]["q75"]}
    corner, relaxed = ms.corner_mask(val["q_cyl"], val["ts_beta_p"], thresholds, args.min_corner_frames)
    z10 = {"validation_list": val_list, "stride": args.refit_stride, "corner_thresholds": thresholds, "corner_relaxed": relaxed,
           "after": pr.heads_r2(heads, val, val_shapes, corner),
           "before": pr.heads_r2(before, val, val_shapes, corner) if before is not None else None,
           "before_note": None if before is not None else f"{before_path.name} not found (scripts/betan_heads.py writes it)"}
    log("  Z10, MAST head on the model's validation shapes: after the refit R2 {:.4f} (corner {:.4f})".format(
        z10["after"]["r2_all"], z10["after"]["r2_corner"]) + ("" if before is None else "; before {:.4f} (corner {:.4f})".format(
            z10["before"]["r2_all"], z10["before"]["r2_corner"])))
    chain.close()
    return heads, {"z10": z10, "seconds": time.time() - t0}


# ------------------------------------------------------------------------------ folds
def predictor_coil_model(predictor, machine: str) -> str | None:
    """The coil model of a set-up predictor's maps on a machine (None: the default)."""
    return (getattr(predictor, "identifiers", None) or {}).get("coil_models", {}).get(machine)


def fallbacks(args, machine: str, coil_model: str | None = None) -> sub.Fallbacks:
    """The pipeline's fallbacks: coil flux (under ``coil_model``, the default when None) plus the
    training mean shape (DIII-D, when cached), nearest frame, constant; scalars from the nearest
    frame, then the training medians."""
    medians = predictors.training_scalar_medians(args.split, args.cache_dir, args.table_root)
    table = predictors.green_table(args.green_dir, machine)
    mean_shape = None
    if machine == "DIII-D":
        path = predictors.model_files(args.cache_dir, sorted(sp.read_split(args.split)["train"]))["mean_shape"]
        if path.is_file():
            with np.load(path, allow_pickle=False) as f:
                mean_shape = f["mean"]
    return sub.Fallbacks(machine=machine, scalar_medians=medians, table=table,
                         scales=cf.adopted_scales(machine, table.columns, coil_model), mean_shape=mean_shape)


def predict_files(predictor, files: list[Path], machine: str, path: Path, args, checks: bool = False) -> sub.FoldRecord:
    if isinstance(getattr(predictor, "source", None), pr.TableSource):
        split = files[0].parent.name if files else None
        if split:
            predictor.source.prefetch(split, [io.parse_shot_filename(p)[1] for p in files])
    fb = fallbacks(args, machine, predictor_coil_model(predictor, machine))
    check = None
    if checks and machine == "MAST":
        table = predictors.green_table(args.green_dir, "MAST")
        check = lambda i, shot: sub.mast_shot_checks(shot, table, fb.scales)  # noqa: E731
    with sub.NpzWriter(path) as writer:
        return sub.predict_fold(predictor, files, machine, fb, writer.add, shot_check=check, log=log)


def make_predictor(args, heads, source: str | None = None):
    kwargs = dict(split_path=args.split, cache_dir=args.cache_dir, green_dir=args.green_dir, table_root=args.table_root,
                  data_root=args.data_root)
    if args.predictor in ("rung1_v1", "rung2_v1"):
        kwargs.update(runs=args.run, source=source or args.source, heads=heads, feature_root=args.feature_root,
                      feature_config=args.feature_config, scaler=args.scaler, targets_root=args.targets_root,
                      checkpoint_root=args.checkpoint_root, mirror_average=args.mirror_average, seed_average=args.seed_average,
                      device=args.device, workers=args.workers, feature_dirs=feature_dirs(args), coil_models=coil_models(args),
                      allow_coil_model_mismatch=args.allow_coil_model_mismatch)
    return sub.make_predictor(args.predictor, **kwargs)


def score_local(predictor, args, lists: dict, sub_dir: Path) -> dict:
    """The validation list through the writer, read back and scored at stride 4."""
    t0 = time.time()
    part = "working_validation" if args.working else lists["validation"]
    ld.require_readable(part)
    val_ids = sorted(sp.read_split(args.split)[part])
    files = [dataset.shot_path(args.data_root, tb.LABELLED_SPLIT, s) for s in val_ids]
    path = sub_dir / f"local_{part}.npz"
    log(f"local score: predicting {len(val_ids)} {part} shots into {path.name}")
    fold = predict_files(predictor, files, "DIII-D", path, args)
    arrays = sub.read_npz(path)
    workers = bl.default_workers() if args.score_workers is None else args.score_workers
    references = bl.build_references(val_ids, args.data_root, tb.LABELLED_SPLIT, args.frame_stride, workers=workers,
                                     cache_dir=args.cache_dir / "references")
    index = {s: i for i, s in enumerate(val_ids)}
    result = bl.score_local(lambda s: sub.read_prediction(arrays, index[s], args.frame_stride), val_ids, args.data_root,
                            tb.LABELLED_SPLIT, args.frame_stride, references, workers)
    terms = bl.score_terms(result.finalize())
    log("  S {S:.4f}  R2 psi {r2_psi:.4f}  R2 q95,betaN {r2_qb:.4f}  1 - D_LCFS {one_minus_dlcfs:.4f}  consistency {consistency:.4f}"
        .format(**terms))
    return {"scored_at": now(), "part": part, "n_shots": len(val_ids), "frame_stride": args.frame_stride, "file": str(path),
            "file_sha256": sub.sha256_of_file(path), "fold": fold.to_dict(), "terms": terms, "seconds": time.time() - t0}


def cmd_build(args) -> int:
    started = time.time()
    run_label = f"{args.date}_{args.name}"
    sub_dir, report_dir = args.submissions_dir / run_label, args.reports_dir / run_label
    if (sub_dir / f"{sub.TEST_CONFIGS['DIII-D']}.npz").exists() and not args.overwrite:
        log(f"{sub_dir} already holds a build; pass --overwrite to rebuild")
        return 1
    machines = [pr.MACHINE_KEYS[m] for m in args.machines]
    record = {"run": run_label, "name": args.name, "built_at": now(), "repo_commit": starter_kit.git_commit(REPO_ROOT),
              "starter_commit": starter_kit.git_commit(starter_kit.starter_dir()), "args": " ".join(sys.argv[1:]),
              "split_file": str(args.split), "predictor": args.predictor, "source": args.source, "machines": machines,
              "data_root": str(args.data_root), "folds": {}, "files": {}}
    heads = None
    if args.predictor in ("rung1_v1", "rung2_v1"):
        if not args.run:
            log(f"{args.predictor} needs --run")
            return 2
        lists = run_lists(args)
        record["training_lists"] = lists
        if args.heads is not None:
            heads = sh.BetanHeads.load(args.heads)
            record["heads"] = {"file": str(args.heads), "sha256": pr.sha256_of(args.heads), "summary": heads.summary(),
                               "note": "stored heads; no refit in this build"}
            if "MAST" in machines and heads.mast_correction is None:
                log(f"{args.heads} has no MAST head; drop --heads to fit it")
                return 2
        else:
            heads, info = fit_heads(args, lists, machines)
            path = sub_dir / HEADS_NAME
            record["heads"] = {"file": str(path), "sha256": heads.save(path), "summary": heads.summary(), **info}
    predictor = make_predictor(args, heads)
    record["predictor_identifiers"] = predictor.setup(log)
    ident = record["predictor_identifiers"]
    record["coil_models"] = ident.get("coil_models", {m: cf.DEFAULT_COIL_MODELS[m] for m in pr.MACHINES})
    record["coil_model_sources"] = ident.get("coil_model_sources", {})
    record["feature_dirs"] = ident.get("feature_dirs", {})
    for machine in machines:
        files = sub.test_shot_files(args.data_root, machine)
        if args.max_shots:
            files = files[:args.max_shots]
        path = sub_dir / f"{sub.TEST_CONFIGS[machine]}.npz"
        log(f"{machine}: predicting {len(files)} shots with {args.predictor} ({args.source})")
        t0 = time.time()
        fold = predict_files(predictor, files, machine, path, args, checks=True)
        entry = fold.to_dict()
        entry["seconds"] = time.time() - t0
        if machine == "MAST":
            entry["mast_checks_summary"] = sub.summarize_mast_checks(fold.shot_checks)
            entry["mast_sign_asserted"] = pr.assert_mast_sign()
        record["folds"][machine] = entry
        record["files"][path.name] = {"path": str(path), "sha256": sub.sha256_of_file(path), "bytes": path.stat().st_size}
        log(f"{machine}: {fold.counts.n_shots} shots, {fold.counts.n_frames} frames in {entry['seconds']:.0f} s; "
            f"fallbacks {fold.counts.counts}; predictor errors {len(fold.counts.predictor_errors)}")
    if args.max_shots:
        record["max_shots"] = args.max_shots
        log("partial build (--max-shots): validation against the full folds is skipped")
    else:
        record["validation"] = {}
        for machine in machines:
            result = sub.validate_npz(sub_dir / f"{sub.TEST_CONFIGS[machine]}.npz", sub.test_shot_files(args.data_root, machine), machine)
            record["validation"][machine] = result.to_dict()
            log(f"{machine} validation: {'ok' if result.ok else f'{len(result.errors)} problem(s)'}")
    # A partial build scores the working validation list only (--working), never the full list.
    if args.score_local and (not args.max_shots or args.working):
        record["score_local"] = score_local(predictor, args, record.get("training_lists", {"validation": "validation"}), sub_dir)
    if hasattr(predictor, "close"):
        predictor.close()
    record["build_seconds"] = time.time() - started
    log(f"run record: {save_record(report_dir, record)}")
    ok = bool(args.max_shots) or all(v["ok"] for v in record["validation"].values())
    return 0 if ok else 1


# ------------------------------------------------------------------------ determinism
def d7_subset(args) -> tuple[list[tuple[str, str]], list[str]]:
    """``[(split, shot)]`` of the D7 subset (or of ``--shots-file``), less any ``local_test``
    shot, and the shots left out."""
    if args.shots_file is not None:
        listed = {split: list(ids) for split, ids in json.loads(Path(args.shots_file).read_text()).items()}
    else:
        spec = importlib.util.spec_from_file_location("check_determinism", REPO_ROOT / "scripts" / "check_determinism.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        listed = {"diii_d_train": list(module.D3D_SHOTS), "mast_public_test": list(module.MAST_SHOTS)}
    protected = set(sp.read_split(args.split).get("local_test", []))
    shots = [(split, s) for split, ids in listed.items() for s in ids if not (split == tb.LABELLED_SPLIT and s in protected)]
    return shots, sorted(s for s in listed.get(tb.LABELLED_SPLIT, []) if s in protected)


def compare_npz(a: Path, b: Path) -> dict:
    """Keys whose arrays differ in bytes, dtype or shape between two npz files."""
    x, y = sub.read_npz(a), sub.read_npz(b)
    differing = [k for k in sorted(set(x) | set(y))
                 if k not in x or k not in y or x[k].dtype != y[k].dtype or x[k].shape != y[k].shape or x[k].tobytes() != y[k].tobytes()]
    return {"n_keys": len(set(x) | set(y)), "differing": differing, "identical": not differing}


def cmd_determinism(args) -> int:
    started = time.time()
    if args.heads is None:
        log("--determinism needs --heads (the betaN heads file of a build)")
        return 2
    shots, left_out = d7_subset(args)
    present = [(split, s) for split, s in shots if dataset.shot_path(args.data_root, split, s).is_file()]
    missing = sorted(set(shots) - set(present))
    results, files, identity = {}, {}, {}
    with tempfile.TemporaryDirectory(prefix="prediction_check_") as tmp:
        tmp = Path(tmp)
        for label, source in (("tables_1", "tables"), ("tables_2", "tables"), ("raw", "raw")):
            predictor = make_predictor(args, args.heads, source)
            ident = predictor.setup(log)
            identity[label] = {k: ident.get(k) for k in ("coil_models", "feature_coil_models", "feature_dirs")}
            for split in sorted({s for s, _ in present}):
                machine = dataset.SPLIT_MACHINE[split]
                paths = [dataset.shot_path(args.data_root, split, s) for sp_, s in present if sp_ == split]
                out = tmp / label / f"{split}.npz"
                predict_files(predictor, paths, machine, out, args)
                files.setdefault(split, {})[label] = {"sha256": sub.sha256_of_file(out)}
            predictor.close()
        for split in files:
            results[split] = {"repeat": compare_npz(tmp / "tables_1" / f"{split}.npz", tmp / "tables_2" / f"{split}.npz"),
                              "raw_vs_tables": compare_npz(tmp / "tables_1" / f"{split}.npz", tmp / "raw" / f"{split}.npz")}
    passed = bool(results) and all(r["repeat"]["identical"] and r["raw_vs_tables"]["identical"] for r in results.values())
    info = {"checked_at": now(), "repo_commit": starter_kit.git_commit(REPO_ROOT), "runs": args.run, "heads": str(args.heads),
            "heads_sha256": pr.sha256_of(args.heads), "n_shots": len(present), "missing": missing,
            "left_out_local_test": left_out, "coil_models": identity.get("tables_1", {}).get("coil_models"),
            "feature_dirs": identity.get("tables_1", {}).get("feature_dirs"), "sources": identity,
            "files": files, "results": results, "pass": passed, "seconds": time.time() - started}
    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / "prediction_check.json").write_text(json.dumps(info, indent=2, default=json_default) + "\n")
    models = info["coil_models"] or {}
    lines = ["# Prediction determinism (Z5)", "", f"Runs {', '.join(args.run)}; heads `{Path(args.heads).name}`; {len(present)} shots of "
             f"the D7 subset ({len(left_out)} local_test shot left out, {len(missing)} not found). Coil models "
             + ", ".join(f"{m} {v}" for m, v in models.items()) + "; feature folders "
             + ", ".join(f"{m} `{Path(v).name}`" for m, v in (info["feature_dirs"] or {}).items())
             + f". Status: **{'pass' if passed else 'fail'}**.", "", "| Split | Repeat (tables twice) | Raw against tables | Keys |", "| --- | --- | --- | --- |"]
    for split, r in results.items():
        word = lambda c: "identical" if c["identical"] else f"{len(c['differing'])} keys differ (first {c['differing'][0]})"  # noqa: E731
        lines.append(f"| {split} | {word(r['repeat'])} | {word(r['raw_vs_tables'])} | {r['repeat']['n_keys']} |")
    (args.report_dir / "prediction_check.md").write_text("\n".join(lines) + "\n")
    log(f"Z5 {'pass' if passed else 'FAIL'}: {args.report_dir / 'prediction_check.md'}")
    return 0 if passed else 1


# ------------------------------------------------------------------------------- main
def build_parser() -> argparse.ArgumentParser:
    import fusion_eq.predictors  # noqa: F401 - fills the registry before the parser lists it
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--predictor", default="rung1_v1", choices=sorted(sub.REGISTRY))
    parser.add_argument("--run", action="append", default=[], help="training run of a checkpoint; repeat for seed averaging")
    parser.add_argument("--machines", nargs="+", choices=sorted(pr.MACHINE_KEYS), default=["d3d", "mast"])
    parser.add_argument("--source", choices=pr.SOURCES, default="tables")
    parser.add_argument("--name", default=None, help="label of the build; the folder is <date>_<name>")
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--heads", type=Path, default=None, help="stored betaN heads (skips the fit and the refit)")
    parser.add_argument("--refit-list", default=None, help="training list of the heads; the run's training list when omitted")
    parser.add_argument("--refit-stride", type=int, default=4)
    parser.add_argument("--min-corner-frames", type=int, default=300)
    parser.add_argument("--no-mirror-average", dest="mirror_average", action="store_false", default=None)
    parser.add_argument("--seed-average", dest="seed_average", action="store_true", default=None)
    parser.add_argument("--max-shots", type=int, default=0, help="cap shots per fold for a smoke run (0 = all)")
    parser.add_argument("--no-score-local", dest="score_local", action="store_false",
                        help="skip writing and scoring the validation list (the run record then has no local score)")
    parser.add_argument("--working", action="store_true", help="score the working validation list")
    parser.add_argument("--frame-stride", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--determinism", action="store_true", help="check Z5 on the D7 subset instead of building")
    parser.add_argument("--shots-file", type=Path, default=None, help="JSON {split: [shot ids]} replacing the D7 subset")
    parser.add_argument("--device", default="cpu", help="torch device of the forward pass (cpu keeps outputs bit-reproducible)")
    parser.add_argument("--workers", type=int, default=None, help="processes for q95 and shapes")
    parser.add_argument("--score-workers", type=int, default=None)
    parser.add_argument("--submissions-dir", type=Path, default=DEFAULT_SUBMISSIONS_DIR)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_DETERMINISM_DIR, help="output of --determinism")
    parser.add_argument("--data-root", type=Path, default=default_data_root())
    parser.add_argument("--split", type=Path, default=predictors.DEFAULT_SPLIT)
    parser.add_argument("--cache-dir", type=Path, default=bl.DEFAULT_CACHE_DIR)
    parser.add_argument("--green-dir", type=Path, default=predictors.DEFAULT_GREEN_DIR)
    parser.add_argument("--table-root", type=Path, default=ft.DEFAULT_TABLE_ROOT)
    parser.add_argument("--feature-root", type=Path, default=fe.DEFAULT_FEATURE_ROOT)
    parser.add_argument("--feature-config", type=Path, default=None)
    parser.add_argument("--mast-feature-dir", type=Path, default=None,
                        help="MAST feature folder replacing <feature-root>/v<version> (a variant build)")
    parser.add_argument("--mast-coil-model", default=None, choices=cf.coil_model_names("MAST"),
                        help="MAST coil model of the maps; the MAST feature build's record when omitted")
    parser.add_argument("--allow-coil-model-mismatch", action="store_true",
                        help="accept a --mast-coil-model that differs from the MAST feature build (diagnostic)")
    parser.add_argument("--scaler", type=Path, default=None)
    parser.add_argument("--targets-root", type=Path, default=tb.DEFAULT_TARGET_ROOT)
    parser.add_argument("--run-root", type=Path, default=pr.DEFAULT_RUN_ROOT)
    parser.add_argument("--checkpoint-root", type=Path, default=pr.DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--shape-dir", type=Path, default=DEFAULT_SHAPE_DIR, help="shape tables of the betaN study")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.determinism:
        return cmd_determinism(args)
    if not args.name:
        log("--name is required for a build")
        return 2
    return cmd_build(args)


if __name__ == "__main__":
    raise SystemExit(main())
