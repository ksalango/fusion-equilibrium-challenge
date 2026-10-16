"""Builds, validates, scores, pushes and packages a challenge submission.

Subcommands, each reading and extending the run record
``reports/submissions/<date>_<name>/run.json`` (never holding the token):

- ``build --predictor <name> --name <label>``: predicts both public-test folds in stream
  order with the pipeline's fallbacks, writes ``submissions/<date>_<name>/diii_d_public_test.npz``
  and ``mast_public_test.npz``, validates them against the raw folds and writes the run
  record (commit, split, predictor and coil-model identifiers, frame and fallback counts,
  validation results, MAST input checks).
- ``validate --run <date>_<name>``: validates the two files again and updates the record.
- ``score-local --run <date>_<name>``: predicts the validation shots of the split through
  the same writer, reads the file back and scores it with the organizers' local scorer on
  every ``--frame-stride``-th frame. ``baseline_v0`` on the validation list of
  ``split_v1`` at stride 4 is expected to give S = 0.9219 (``reports/baseline``).
- ``push --run <date>_<name> --repo <user>/<repo>``: uploads the two npz files to a private
  Hugging Face dataset repo (write token from ``huggingface-cli login``) and prints the
  commit SHA.
- ``check-order [--n-shots 5]``: streams the first shots of each public-test config from
  the Hub, as the organizers' skeleton does, and compares each one's number of
  ``efit_times`` and content fingerprint with the local file at the same rank in sorted
  order (the Hub rows carry no shot id). Needs the network. Prints pass or fail per machine.
- ``pointer --run <date>_<name>``: reads the fine-grained read token from the environment
  variable ``HF_SUBMISSION_READ_TOKEN``, checks it with the organizers' token rules and
  against the pushed revision, and writes ``submissions/<date>_<name>/submission_pointer.zip``.
  The token is never printed, logged or written anywhere but the zip.

``submissions/`` is gitignored because the zip holds the token. Usage::

    python scripts/make_submission.py check-order
    python scripts/make_submission.py build --predictor baseline_v0 --name baseline_v0
    python scripts/make_submission.py validate --run 2026-10-11_baseline_v0
    python scripts/make_submission.py score-local --run 2026-10-11_baseline_v0
    python scripts/make_submission.py push --run 2026-10-11_baseline_v0 --repo <user>/fusion-eq-predictions
    HF_SUBMISSION_READ_TOKEN=hf_... python scripts/make_submission.py pointer --run 2026-10-11_baseline_v0
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np

from fusion_eq import baseline as bl, coil_flux as cf, dataset, frame_table as ft, predictors, split as sp, starter_kit, submission as sub
from fusion_eq.dataset import REPO_ROOT, default_data_root

DEFAULT_SUBMISSIONS_DIR = REPO_ROOT / "submissions"
DEFAULT_REPORTS_DIR = REPO_ROOT / "reports" / "submissions"
RECORD_NAME = "run.json"
TOKEN_ENV = "HF_SUBMISSION_READ_TOKEN"
MACHINES = ("DIII-D", "MAST")


def log(message: str) -> None:
    print(message, flush=True)


# ------------------------------------------------------------------------ run record
def run_dirs(args) -> tuple[Path, Path]:
    """``(submissions/<run>, reports/submissions/<run>)`` of a run label."""
    return args.submissions_dir / args.run, args.reports_dir / args.run


def load_record(report_dir: Path) -> dict:
    path = report_dir / RECORD_NAME
    if not path.is_file():
        raise FileNotFoundError(f"no run record at {path}; run build first")
    return json.loads(path.read_text())


def save_record(report_dir: Path, record: dict) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / RECORD_NAME
    text = json.dumps(record, indent=2, default=_json_default)
    if TOKEN_ENV in os.environ and os.environ[TOKEN_ENV] and os.environ[TOKEN_ENV] in text:
        raise RuntimeError("the read token must not enter the run record")
    path.write_text(text + "\n")
    return path


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serialisable: {type(value).__name__}")


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ------------------------------------------------------------------------- resources
def fallback_resources(args, machine: str, medians: dict[str, float], coil_model: str | None = None) -> sub.Fallbacks:
    """The pipeline's fallback resources of a machine: Green's table and the coil model of the
    predictor's maps (``coil_model``; the default when None) on both, the mean plasma shape of
    the split's training list on DIII-D only."""
    table = predictors.green_table(args.green_dir, machine)
    scales = cf.adopted_scales(machine, table.columns, coil_model)
    mean_shape = None
    if machine == "DIII-D":
        path = predictors.model_files(args.cache_dir, sorted(sp.read_split(args.split)["train"]))["mean_shape"]
        if path.is_file():
            with np.load(path, allow_pickle=False) as f:
                mean_shape = f["mean"]
        else:
            log(f"mean shape {path} not found; the coil-plus-mean-shape fallback is disabled")
    return sub.Fallbacks(machine=machine, scalar_medians=medians, table=table, scales=scales, mean_shape=mean_shape)


def fold_files(args, machine: str) -> list[Path]:
    files = sub.test_shot_files(args.data_root, machine)
    expected = dataset.EXPECTED_SHOTS[sub.TEST_CONFIGS[machine]]
    if len(files) != expected:
        log(f"warning: {len(files)} {machine} test files, {expected} expected")
    return files


def validate_fold(sub_dir: Path, args, machine: str) -> sub.ValidationResult:
    path = sub_dir / f"{sub.TEST_CONFIGS[machine]}.npz"
    result = sub.validate_npz(path, fold_files(args, machine), machine)
    status = "ok" if result.ok else f"{len(result.errors)} problem(s)"
    log(f"{path.name}: {result.n_shots} shots, {result.n_frames} frames, psirz {result.psi_dtype}: {status}")
    for error in result.errors[:20]:
        log(f"    - {error}")
    return result


# ----------------------------------------------------------------------- subcommands
def compare_order(hub_rows, local_files: list[Path]) -> list[dict]:
    """One comparison row per streamed shot: the Hub fingerprint against the local file at
    the same rank. ``hub_rows`` is any iterable of rows with the fingerprint columns."""
    rows = []
    for i, row in enumerate(hub_rows):
        hub = sub.shot_fingerprint(row)
        if i < len(local_files):
            local = sub.local_fingerprint(local_files[i])
            name = local_files[i].name
        else:
            local, name = None, None
        rows.append({"index": i, "local_file": name, "hub": hub, "local": local, "match": local == hub})
    return rows


def cmd_check_order(args) -> int:
    from datasets import load_dataset
    all_ok = True
    for machine in MACHINES:
        config = sub.TEST_CONFIGS[machine]
        files = fold_files(args, machine)
        log(f"{machine}: streaming the first {args.n_shots} shots of {sub.HUB_REPO_ID} {config} ...")
        ds = load_dataset(sub.HUB_REPO_ID, config, split=sub.HUB_TEST_SPLIT, streaming=True)
        rows = compare_order((row for i, row in zip(range(args.n_shots), ds)), files)
        for r in rows:
            local = r["local"] or {}
            log(f"  shot_{r['index']:04d}  hub T={r['hub']['n_frames']} {r['hub']['sha256']}  "
                f"local {r['local_file']} T={local.get('n_frames')} {local.get('sha256')}  "
                f"{'match' if r['match'] else 'MISMATCH'}")
        ok = bool(rows) and all(r["match"] for r in rows)
        all_ok &= ok
        log(f"{machine}: {'pass' if ok else 'FAIL'} ({sum(r['match'] for r in rows)}/{len(rows)} shots match the sorted local order)")
    return 0 if all_ok else 1


def cmd_build(args) -> int:
    started = time.time()
    args.run = f"{args.date}_{args.name}"
    sub_dir, report_dir = run_dirs(args)
    if (sub_dir / f"{sub.TEST_CONFIGS['DIII-D']}.npz").exists() and not args.overwrite:
        log(f"{sub_dir} already holds a build; pass --overwrite to rebuild")
        return 1
    extra = {}
    if args.mast_coil_model is not None:
        if args.predictor != "mean_current_v0":
            log("--mast-coil-model applies to mean_current_v0 only (baseline_v0 and coil_only_v0 keep MAST unit)")
            return 2
        extra["coil_models"] = {"MAST": args.mast_coil_model}
    predictor = sub.make_predictor(args.predictor, split_path=args.split, cache_dir=args.cache_dir, green_dir=args.green_dir,
                                   table_root=args.table_root, data_root=args.data_root, **extra)
    identifiers = predictor.setup(log)
    coil_models = identifiers.get("coil_models", {})
    medians = predictors.training_scalar_medians(args.split, args.cache_dir, args.table_root)
    record = {"run": args.run, "name": args.name, "built_at": now(), "repo_commit": starter_kit.git_commit(REPO_ROOT),
              "starter_commit": starter_kit.git_commit(starter_kit.starter_dir()), "args": " ".join(sys.argv[1:]),
              "split_file": str(args.split), "predictor": args.predictor, "predictor_identifiers": identifiers,
              "fallback_scalar_medians": medians, "data_root": str(args.data_root), "coil_models": coil_models,
              "folds": {}, "files": {}}
    table_mast = predictors.green_table(args.green_dir, "MAST")
    for machine in MACHINES:
        files = fold_files(args, machine)
        if args.max_shots:
            files = files[:args.max_shots]
        fallbacks = fallback_resources(args, machine, medians, coil_models.get(machine))
        check = (lambda i, shot: sub.mast_shot_checks(shot, table_mast, fallbacks.scales)) if machine == "MAST" else None
        path = sub_dir / f"{sub.TEST_CONFIGS[machine]}.npz"
        log(f"{machine}: predicting {len(files)} shots with {args.predictor}")
        t0 = time.time()
        with sub.NpzWriter(path) as writer:
            fold = sub.predict_fold(predictor, files, machine, fallbacks, writer.add, shot_check=check, log=log)
        entry = fold.to_dict()
        entry["seconds"] = time.time() - t0
        if machine == "MAST":
            entry["mast_checks_summary"] = sub.summarize_mast_checks(fold.shot_checks)
        record["folds"][machine] = entry
        record["files"][path.name] = {"path": str(path), "sha256": sub.sha256_of_file(path), "bytes": path.stat().st_size}
        log(f"{machine}: {fold.counts.n_shots} shots, {fold.counts.n_frames} frames in {entry['seconds']:.0f} s; "
            f"fallbacks {fold.counts.counts}; predictor errors {len(fold.counts.predictor_errors)}")
    if args.max_shots:
        record["max_shots"] = args.max_shots
        log("partial build (--max-shots): validation against the full folds is skipped")
    else:
        record["validation"] = {m: validate_fold(sub_dir, args, m).to_dict() for m in MACHINES}
    record["build_seconds"] = time.time() - started
    path = save_record(report_dir, record)
    log(f"run record: {path}")
    ok = args.max_shots or all(v["ok"] for v in record["validation"].values())
    return 0 if ok else 1


def cmd_validate(args) -> int:
    sub_dir, report_dir = run_dirs(args)
    record = load_record(report_dir)
    record["validation"] = {m: validate_fold(sub_dir, args, m).to_dict() for m in MACHINES}
    record["validated_at"] = now()
    save_record(report_dir, record)
    return 0 if all(v["ok"] for v in record["validation"].values()) else 1


def cmd_score_local(args) -> int:
    started = time.time()
    sub_dir, report_dir = run_dirs(args)
    record = load_record(report_dir)
    predictor = sub.make_predictor(record["predictor"], split_path=args.split, cache_dir=args.cache_dir, green_dir=args.green_dir,
                                   table_root=args.table_root, data_root=args.data_root)
    predictor.setup(log)
    medians = predictors.training_scalar_medians(args.split, args.cache_dir, args.table_root)
    split = sp.read_split(args.split)
    val_ids = sorted(split["working_validation" if args.working else "validation"])
    files = [dataset.shot_path(args.data_root, sp.TRAIN_SPLIT, s) for s in val_ids]
    fallbacks = fallback_resources(args, "DIII-D", medians)
    path = sub_dir / ("local_validation_working.npz" if args.working else "local_validation.npz")
    log(f"predicting {len(val_ids)} validation shots and writing {path.name}")
    with sub.NpzWriter(path) as writer:
        fold = sub.predict_fold(predictor, files, "DIII-D", fallbacks, writer.add, log=log)
    arrays = sub.read_npz(path)
    workers = bl.default_workers() if args.workers is None else args.workers
    references = bl.build_references(val_ids, args.data_root, frame_stride=args.frame_stride, workers=workers,
                                     cache_dir=args.cache_dir / "references")
    log(f"scorer references {'read from the cache' if references.from_cache else 'built'}; scoring with {workers} workers")
    index = {s: i for i, s in enumerate(val_ids)}
    result = bl.score_local(lambda s: sub.read_prediction(arrays, index[s], args.frame_stride), val_ids, args.data_root,
                            frame_stride=args.frame_stride, references=references, workers=workers)
    terms = bl.score_terms(result.finalize())
    log("S = {S:.4f}  R2 psi {r2_psi:.4f}  R2 q95,betaN {r2_qb:.4f}  1 - D_LCFS {one_minus_dlcfs:.4f}  "
        "consistency {consistency:.4f}  psi sign {psi_sign:+.0f}".format(**terms))
    log("  consistency terms: " + ", ".join(f"{k[3:]} {v:.4f}" for k, v in terms.items()
                                           if k.startswith("r2_") and k not in ("r2_psi", "r2_qb", "r2_q95", "r2_betaN")))
    log("  scalars: q95 {r2_q95:.4f}, betaN {r2_betaN:.4f}".format(**terms))
    record["score_local"] = {"scored_at": now(), "part": "working_validation" if args.working else "validation",
                             "n_shots": len(val_ids), "frame_stride": args.frame_stride, "file": str(path),
                             "file_sha256": sub.sha256_of_file(path), "fold": fold.to_dict(), "terms": terms,
                             "expected_S_baseline_v0_validation_stride4": 0.9219, "seconds": time.time() - started}
    save_record(report_dir, record)
    return 0


def cmd_push(args) -> int:
    from huggingface_hub import HfApi
    sub_dir, report_dir = run_dirs(args)
    record = load_record(report_dir)
    names = [f"{sub.TEST_CONFIGS[m]}.npz" for m in MACHINES]
    missing = [n for n in names if not (sub_dir / n).is_file()]
    if missing:
        log(f"missing in {sub_dir}: {missing}")
        return 1
    if not record.get("validation") or not all(v["ok"] for v in record["validation"].values()):
        log("the run record holds no passing validation; run validate first")
        return 1
    api = HfApi()
    who = api.whoami()
    log(f"logged in as {who['name']}; creating or reusing the private dataset repo {args.repo}")
    api.create_repo(args.repo, repo_type="dataset", private=True, exist_ok=True)
    info = api.repo_info(args.repo, repo_type="dataset")
    if not getattr(info, "private", True):
        log(f"warning: {args.repo} is PUBLIC; the predictions are visible to everyone")
    total_mb = sum((sub_dir / n).stat().st_size for n in names) / 1e6
    log(f"uploading {', '.join(names)} ({total_mb:.0f} MB)")
    commit = api.upload_folder(folder_path=str(sub_dir), repo_id=args.repo, repo_type="dataset", allow_patterns=names,
                               commit_message=f"{args.run}: {record['predictor']}")
    sha = commit.oid
    if not (len(sha) == 40 and all(c in "0123456789abcdef" for c in sha.lower())):
        log(f"expected a 40-character commit SHA, got {sha!r}")
        return 1
    at_rev = api.repo_info(args.repo, revision=sha, repo_type="dataset")
    root = {s.rfilename for s in at_rev.siblings if "/" not in s.rfilename}
    if not set(names) <= root:
        log(f"not every file sits at the repo root at {sha}: {sorted(root)}")
        return 1
    record["push"] = {"pushed_at": now(), "hf_repo": args.repo, "hf_revision": sha, "files": names,
                      "file_sha256": {n: sub.sha256_of_file(sub_dir / n) for n in names}}
    save_record(report_dir, record)
    log(f"pushed at commit {sha}")
    return 0


def starter_push_module():
    """Imports the organizers' ``push_predictions.py`` for its token rules, unmodified."""
    path = starter_kit.require() / "push_predictions.py"
    spec = importlib.util.spec_from_file_location("starter_push_predictions", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cmd_pointer(args) -> int:
    sub_dir, report_dir = run_dirs(args)
    record = load_record(report_dir)
    push = record.get("push") or {}
    repo = args.repo or push.get("hf_repo")
    revision = args.revision or push.get("hf_revision")
    if not repo or not revision:
        log("no pushed revision in the run record; pass --repo and --revision or run push first")
        return 1
    token = os.environ.get(TOKEN_ENV, "")
    if not token:
        log(f"set {TOKEN_ENV} to the fine-grained read token scoped to {repo}")
        return 1
    manifest = sub.manifest_dict(repo, revision, token)
    if not args.skip_token_check:
        from huggingface_hub import HfApi
        api = HfApi()
        if starter_push_module().validate_read_token(api, token, repo):
            return 1
        at_rev = api.repo_info(repo, revision=revision, repo_type="dataset", token=token)
        root = {s.rfilename for s in at_rev.siblings if "/" not in s.rfilename}
        names = [f"{sub.TEST_CONFIGS[m]}.npz" for m in MACHINES]
        if not set(names) <= root:
            log(f"the read token sees {sorted(root)} at {revision}, not {names}")
            return 1
        log("read token accepted: fine-grained, read-only, sees both files at the pinned revision")
    zip_path = sub.write_pointer_zip(sub_dir / sub.POINTER_NAME, manifest)
    back = sub.read_pointer_zip(zip_path)
    if tuple(back) != sub.MANIFEST_FIELDS or back["revision"] != revision or back["repo_id"] != repo:
        log("the written pointer does not read back as expected")
        return 1
    record["pointer"] = {"written_at": now(), "zip": str(zip_path), "zip_sha256": sub.sha256_of_file(zip_path),
                         "hf_repo": repo, "hf_revision": revision, "token_checked": not args.skip_token_check}
    save_record(report_dir, record)
    log(f"upload this to Codabench: {zip_path} ({zip_path.stat().st_size} bytes); repo {repo}, revision {revision}")
    return 0


# ------------------------------------------------------------------------------- main
def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--submissions-dir", type=Path, default=DEFAULT_SUBMISSIONS_DIR)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--data-root", type=Path, default=default_data_root())
    parser.add_argument("--split", type=Path, default=predictors.DEFAULT_SPLIT)
    parser.add_argument("--cache-dir", type=Path, default=bl.DEFAULT_CACHE_DIR)
    parser.add_argument("--green-dir", type=Path, default=predictors.DEFAULT_GREEN_DIR)
    parser.add_argument("--table-root", type=Path, default=ft.DEFAULT_TABLE_ROOT)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)

    p = commands.add_parser("check-order", help="compare the Hub stream order with the sorted local files (network)")
    p.add_argument("--n-shots", type=int, default=5, help="shots streamed per config")
    add_common(p)
    p.set_defaults(func=cmd_check_order)

    p = commands.add_parser("build", help="predict both test folds, write and validate the npz files")
    p.add_argument("--predictor", required=True, choices=sorted(sub.REGISTRY) or None)
    p.add_argument("--name", required=True, help="label of the run; the folder is <date>_<name>")
    p.add_argument("--date", default=date.today().isoformat())
    p.add_argument("--max-shots", type=int, default=0, help="cap shots per fold for a smoke run (0 = all)")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--mast-coil-model", default=None, choices=cf.coil_model_names("MAST"),
                   help="MAST coil model of mean_current_v0 (its default otherwise; MAST unit reproduces 2026-10-14)")
    add_common(p)
    p.set_defaults(func=cmd_build)

    p = commands.add_parser("validate", help="validate the written npz files against the raw folds")
    p.add_argument("--run", required=True)
    add_common(p)
    p.set_defaults(func=cmd_validate)

    p = commands.add_parser("score-local", help="write and score the validation shots with the local scorer")
    p.add_argument("--run", required=True)
    p.add_argument("--frame-stride", type=int, default=4)
    p.add_argument("--working", action="store_true", help="use the working validation list")
    p.add_argument("--workers", type=int, default=None)
    add_common(p)
    p.set_defaults(func=cmd_score_local)

    p = commands.add_parser("push", help="upload the two npz files to a private Hugging Face dataset repo")
    p.add_argument("--run", required=True)
    p.add_argument("--repo", required=True, help="<user>/<repo>")
    add_common(p)
    p.set_defaults(func=cmd_push)

    p = commands.add_parser("pointer", help=f"write submission_pointer.zip from {TOKEN_ENV}")
    p.add_argument("--run", required=True)
    p.add_argument("--repo", default=None, help="defaults to the pushed repo in the run record")
    p.add_argument("--revision", default=None, help="defaults to the pushed commit in the run record")
    p.add_argument("--skip-token-check", action="store_true", help="do not contact Hugging Face to check the token")
    add_common(p)
    p.set_defaults(func=cmd_pointer)
    return parser


def main(argv: list[str] | None = None) -> int:
    import fusion_eq.predictors  # noqa: F401 - fills the registry before the parser lists it
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
