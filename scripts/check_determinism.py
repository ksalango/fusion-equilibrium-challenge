"""Determinism check (D7): rebuilds the rows of a fixed list of shots and compares them with the
stored tables.

For 20 DIII-D training shots and 5 MAST test shots (ids below), in a temporary directory outside
the processed-data root:

1. Frame table: each shot's rows with ``frame_table.build_one`` and ``write_split`` of
   ``scripts/build_frame_table.py``, under the parameters recorded in the stored table's
   ``build.json``.
2. Features, version 2 (``configs/features_v2.yaml``): ``build_split`` of
   ``scripts/build_features.py`` on the rebuilt frame rows, with the machine resources of the
   full build (point tables from the cache in the green directory).
3. Target tables, when a target build exists: for every list table of the build that holds some
   of the DIII-D shots, ``build_table`` and ``write_table`` of ``scripts/build_targets.py`` on the
   rebuilt frame rows, with the stored fitted objects, the build's stride and split, and the
   shape numbers of the full build (the list's cached scorer references when the build used
   them). The fitted objects are checked against their checksums in ``build.json``; refitting
   them is outside this check.

Each rebuilt table is compared with the stored rows of the same shots
(``fusion_eq.determinism``): SHA-256 of the row bytes and exact equality of values. Per table
the report gives ``identical`` or the first differing column and row. ``--inputs stored`` builds
the features and targets from the stored frame table instead, to separate the layers when the
frame table differs.

Stored tables and caches are only read: point tables and plasma responses are linked into the
temporary directory, so a missing one is built there. Nothing is written under the
processed-data root or the raw data root. The report goes to
``<report-root>/determinism_check.md`` and ``.json``. Exit status 1 when a table differs.
Usage::

    python scripts/check_determinism.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq

from fusion_eq import (baseline as bl, current_target as ct, dataset, determinism as det, features as fe,
                       frame_table as ft, split as sp, starter_kit, target_build as tb)
from fusion_eq.coil_flux import GreenTable
from fusion_eq.dataset import default_data_root, filter_shots, list_shots, run_info

# Drawn with seed 20261013 from splits/split_v1.json and the shots of reports/qa without a QA
# error; 9 of the 20 DIII-D shots need the Ip time-axis fix, 11 do not.
D3D_SHOTS = (
    # working_train (also train)
    "14c85cdf41", "1962ebdbcf", "37b75485bc", "564a49a026", "652ed80a9f", "7cc68b9d2d", "a97f03b645", "e417a47b84",
    # working_validation (also validation)
    "1360cf2498", "187f9af914", "1cf2a6d87d", "cd9a52ab40",
    # train only
    "13e438a773", "2bbbc7108e", "d0881f4abd", "e43f2572c5",
    # validation only
    "5a5e3fec75", "9fc5d2b15a",
    # local_test
    "45027c603f",
    # train, flagged by the lighter label filter
    "5df1630aa7",
)
MAST_SHOTS = ("22ebf96cd9", "4f514e8960", "632728f599", "658a55d5d8", "b687d746cf")
SPLITS = {"diii_d_train": "DIII-D", "mast_public_test": "MAST"}
DEFAULT_FEATURE_CONFIG = dataset.REPO_ROOT / "configs" / "features_v2.yaml"
DEFAULT_REPORT_ROOT = dataset.REPO_ROOT / "reports" / "determinism"


def log(message: str) -> None:
    print(message, flush=True)


def load_script(name: str):
    """A build script as a module, so the check calls the full build's own functions."""
    spec = importlib.util.spec_from_file_location(name, dataset.REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inside(path: Path, root: Path) -> bool:
    path, root = Path(path).resolve(), Path(root).resolve()
    return path == root or path.is_relative_to(root)


def guard_outputs(paths: dict[str, Path], protected: dict[str, Path]) -> None:
    """Raises SystemExit when an output location lies inside a protected one."""
    for name, path in paths.items():
        for pname, root in protected.items():
            if inside(path, root):
                raise SystemExit(f"{name} {path} lies inside {pname} {root}; nothing is written there")


def link_files(source: Path, pattern: str, dest: Path) -> list[str]:
    """Links the files of ``source`` matching ``pattern`` into ``dest`` (read through the link;
    a file built later lands in ``dest``)."""
    dest.mkdir(parents=True, exist_ok=True)
    names = []
    for path in sorted(Path(source).glob(pattern)) if Path(source).is_dir() else []:
        (dest / path.name).symlink_to(path.resolve())
        names.append(path.name)
    return names


def compare(table: str, part: str, stored_path: Path, rebuilt_path: Path, shots, note: str = "") -> dict:
    stored = det.read_rows(stored_path, shots)
    rebuilt = det.read_rows(rebuilt_path, shots) if rebuilt_path.is_file() else stored.schema.empty_table()
    result = det.compare_tables(stored, rebuilt)
    found = sorted(set(stored.column("shot").to_pylist())) if stored.num_rows else []
    return {"table": table, "part": part, "n_shots": len(found), "note": note, **result}


def skipped(table: str, part: str, reason: str) -> dict:
    return {"table": table, "part": part, "skipped": reason}


# --------------------------------------------------------------------------- frame table
def check_frame_table(args, work: Path, scripts: dict, workers: int) -> tuple[list[dict], dict]:
    record_path = args.table_dir / "build.json"
    if not record_path.is_file():
        raise SystemExit(f"{record_path} not found: no stored frame table to compare with")
    record = json.loads(record_path.read_text())
    params = ft.BuildParams(**record["params"])
    out_dir = work / "frame_table"
    out_dir.mkdir(parents=True, exist_ok=True)
    results, info = [], {"params": params.to_dict(), "errors": {}, "not_found": {}, "built_at_commit": record.get("repo_commit")}
    for split, ids in (("diii_d_train", args.d3d_shots), ("mast_public_test", args.mast_shots)):
        files = filter_shots(list_shots(args.data_root, split), ids)
        found = {dataset.parse_shot_filename(p)[1] for p in files}
        info["not_found"][split] = sorted(set(ids) - found)
        built = list(bl._map_ordered(ft.build_one, [(p, split, params) for p in files], workers))
        info["errors"][split] = {row["shot"]: row["error"] for _, row in built if row["error"]}
        scripts["build_frame_table"].write_split(split, (frames for frames, _ in built), out_dir,
                                                 scripts["build_frame_table"].DEFAULT_SHOTS_PER_GROUP, params.rate_window_ms)
        note = f"{len(info['not_found'][split])} shots not in the raw data" if info["not_found"][split] else ""
        results.append(compare("frame table", split, ft.table_path(split, args.table_dir), ft.table_path(split, out_dir), ids, note))
        log(f"frame table, {split}: {det.summary_line(results[-1])}")
    return results, info


# ------------------------------------------------------------------------------ features
def check_features(args, work: Path, inputs_dir: Path, scripts: dict) -> tuple[list[dict], dict]:
    config = fe.FeatureConfig.load(args.feature_config)
    stored_dir = fe.features_dir(config.version, args.feature_root)
    name = f"features v{config.version}"
    if not (stored_dir / "build.json").is_file():
        return [skipped(name, s, f"no feature build at {stored_dir}") for s in SPLITS], {}
    record = json.loads((stored_dir / "build.json").read_text())
    points = work / "points"
    linked = link_files(args.green_dir, "points_*.npz", points)
    resources = {m: fe.load_resources(m, config, args.data_root, args.green_dir, cache_dir=points) for m in SPLITS.values()}
    info = {"version": config.version, "config_hash": config.hash(), "config_hash_at_build": record.get("config_hash"),
            "built_at_commit": record.get("repo_commit"), "point_tables": {m: "built in the work directory" if r.record["point_table"]["built"] else "from the cache"
                             for m, r in resources.items()}, "point_tables_linked": linked}
    ns = argparse.Namespace(table_dir=inputs_dir, out=work / "features", shots_per_chunk=500)
    fe.features_dir(config.version, ns.out).mkdir(parents=True, exist_ok=True)
    results = []
    for split, ids in (("diii_d_train", args.d3d_shots), ("mast_public_test", args.mast_shots)):
        scripts["build_features"].build_split(split, config, resources[SPLITS[split]], ns, list(ids))
        note = "" if info["config_hash"] == info["config_hash_at_build"] else "configuration changed since the build"
        results.append(compare(name, split, fe.features_path(split, config.version, args.feature_root),
                               fe.features_path(split, config.version, ns.out), ids, note))
        log(f"{name}, {split}: {det.summary_line(results[-1])}")
    return results, info


# ------------------------------------------------------------------------------- targets
def check_targets(args, work: Path, inputs_dir: Path, scripts: dict, workers: int) -> tuple[list[dict], dict]:
    config = tb.TargetConfig.load(args.target_config)
    folder = tb.target_dir(config.version, args.targets)
    name = f"targets v{config.version}"
    if not (folder / "build.json").is_file():
        return [skipped(name, "all lists", f"no target build at {folder}")], {}
    objects = tb.TargetObjects.load(folder)
    record = objects.record
    arguments = record.get("arguments", {})
    split_name = arguments.get("split_name", tb.LABELLED_SPLIT)
    split_path = args.split or Path(arguments.get("split", dataset.REPO_ROOT / "splits" / "split_v1.json"))
    split = sp.read_split(split_path)
    stride = int(record["stride"])
    response_dir = work / "response"
    info = {"version": config.version, "config_hash": config.hash(), "config_hash_at_build": record.get("config_hash"),
            "built_at_commit": record.get("repo_commit"), "stride": stride, "split_name": split_name, "split_file": str(split_path),
            "object_checksums": {f: tb.sha256_of(folder / f) == h for f, h in record.get("checksums", {}).items()},
            "responses_linked": link_files(args.response_cache, "response_*.npz", response_dir), "shape_numbers": {}}
    ns = argparse.Namespace(split_name=split_name, stride=stride, data_root=args.data_root, green_dir=args.green_dir,
                            response_cache=response_dir, references_dir=args.references_dir)
    bt = scripts["build_targets"]
    columns, _, _ = bt.frame_columns(config, split_name, inputs_dir, GreenTable.load(args.green_dir / "green_d3d.npz"))
    results = []
    for list_name, file in record["tables"].items():
        stored_shots = set(pq.read_table(folder / file, columns=["shot"]).column("shot").to_pylist())
        ids = sorted(set(args.d3d_shots) & stored_shots)
        if not ids:
            results.append(skipped(name, list_name, "none of the check shots is in this list"))
            continue
        rows = bt.frame_rows(ids, columns, inputs_dir, stride)
        cons = None
        if config.shape_numbers and record["lists"][list_name].get("shape_numbers") == "cached scorer references":
            cached = tb.cached_shape_numbers(sorted(split[list_name]), split_name, stride, args.references_dir)
            info["shape_numbers"][list_name] = "cached scorer references of the list" if cached else \
                "cached references not found; scorer reference builder"
            cons = None if cached is None else {s: cached[s] for s in ids}
        else:
            info["shape_numbers"][list_name] = record["lists"][list_name].get("shape_numbers")
        table, _ = bt.build_table(list_name, rows, objects.current_basis, objects.remainder_model, config, ns, workers, cons=cons)
        path = tb.table_path(list_name, config.version, work / "targets")
        bt.write_table(table, path)
        note = "" if info["config_hash"] == info["config_hash_at_build"] else "configuration changed since the build"
        results.append(compare(name, list_name, folder / file, path, ids, note))
        log(f"{name}, {list_name}: {det.summary_line(results[-1])}")
    return results, info


# -------------------------------------------------------------------------------- report
def render(results: list[dict], info: dict) -> str:
    lines = ["# Determinism check (D7)", "",
             f"{len(info['d3d_shots'])} DIII-D training shots and {len(info['mast_shots'])} MAST test shots rebuilt in a "
             f"temporary directory and compared with the stored tables. Features and targets built from the "
             f"{info['inputs']} frame table. Row digest: SHA-256 of the row bytes, column by column; values: exact "
             "equality, NaN equal to NaN.", "",
             "| Table | Split or list | Shots | Rows stored | Rows rebuilt | SHA-256 stored | SHA-256 rebuilt | Result |",
             "| --- " * 8 + "|"]
    for r in results:
        if r.get("skipped"):
            lines.append(f"| {r['table']} | {r['part']} | | | | | | {det.summary_line(r)} |")
            continue
        result = det.summary_line(r).replace("|", "/") + (f" ({r['note']})" if r.get("note") else "")
        lines.append(f"| {r['table']} | {r['part']} | {r['n_shots']} | {r['rows_stored']} | {r['rows_rebuilt']} | "
                     f"`{r['digest_stored'][:16]}` | `{r['digest_rebuilt'][:16]}` | {result} |")
    lines.append("")
    commits = {"frame table": info.get("frame_table", {}), "features": info.get("features", {}), "targets": info.get("targets", {})}
    lines.append(f"- Commit now `{str(info.get('repo_commit'))[:12]}`; stored builds made at: " + ", ".join(
        f"{k} `{str(v.get('built_at_commit'))[:12]}`" for k, v in commits.items() if v)
        + ". A difference with a different commit can come from a code change rather than non-determinism.")
    frame = info.get("frame_table", {})
    for split, errors in frame.get("errors", {}).items():
        for shot, error in errors.items():
            lines.append(f"- Frame table, {split}, shot {shot}: no rows ({error})")
    targets = info.get("targets", {})
    if targets:
        bad = [f for f, ok in targets["object_checksums"].items() if not ok]
        lines.append(f"- Target objects against their checksums in build.json: {'all match' if not bad else 'differ: ' + ', '.join(bad)}")
        lines.append(f"- Target shape numbers: " + "; ".join(f"{k} {v}" for k, v in targets["shape_numbers"].items()))
    features = info.get("features", {})
    if features:
        lines.append("- Point tables: " + ", ".join(f"{m} {v}" for m, v in features["point_tables"].items()))
    lines += ["", f"Runtime {info['runtime_s']:.0f} s.", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--d3d-shots", nargs="+", default=list(D3D_SHOTS), help="DIII-D training shot ids")
    parser.add_argument("--mast-shots", nargs="+", default=list(MAST_SHOTS), help="MAST test shot ids")
    parser.add_argument("--inputs", choices=("rebuilt", "stored"), default="rebuilt",
                        help="frame table the features and targets are built from")
    parser.add_argument("--data-root", type=Path, default=default_data_root())
    parser.add_argument("--table-dir", type=Path, default=ft.DEFAULT_TABLE_ROOT)
    parser.add_argument("--feature-root", type=Path, default=fe.DEFAULT_FEATURE_ROOT)
    parser.add_argument("--feature-config", type=Path, default=DEFAULT_FEATURE_CONFIG)
    parser.add_argument("--green-dir", type=Path, default=fe.DEFAULT_GREEN_DIR)
    parser.add_argument("--targets", type=Path, default=tb.DEFAULT_TARGET_ROOT)
    parser.add_argument("--target-config", type=Path, default=tb.DEFAULT_CONFIG)
    parser.add_argument("--split", type=Path, default=None, help="split file of the target build (default: from its build.json)")
    parser.add_argument("--response-cache", type=Path, default=ct.DEFAULT_CACHE_DIR)
    parser.add_argument("--references-dir", type=Path, default=bl.DEFAULT_REFERENCE_DIR)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--work-dir", type=Path, default=None, help="rebuild directory (default: a new temporary directory)")
    parser.add_argument("--keep-work-dir", action="store_true")
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args(argv)

    started = time.time()
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    starter_kit.require()
    work = Path(tempfile.mkdtemp(prefix="check_determinism_")) if args.work_dir is None else args.work_dir
    protected = {"the processed-data root": dataset.DEFAULT_PROCESSED_ROOT, "the raw data root": args.data_root,
                 "the frame table": args.table_dir, "the feature root": args.feature_root, "the target root": args.targets,
                 "the green directory": args.green_dir, "the response cache": args.response_cache,
                 "the reference cache": args.references_dir}
    guard_outputs({"--work-dir": work, "--report-root": args.report_root}, protected)
    if args.work_dir is not None:
        if work.exists() and any(work.iterdir()):
            raise SystemExit(f"--work-dir {work} is not empty")
        work.mkdir(parents=True, exist_ok=True)
    workers = bl.default_workers() if args.workers is None else args.workers
    scripts = {name: load_script(name) for name in ("build_frame_table", "build_features", "build_targets")}
    log(f"rebuilding {len(args.d3d_shots)} DIII-D and {len(args.mast_shots)} MAST shots in {work}")

    info = {"d3d_shots": list(args.d3d_shots), "mast_shots": list(args.mast_shots), "inputs": args.inputs}
    try:
        results, info["frame_table"] = check_frame_table(args, work, scripts, workers)
        inputs_dir = work / "frame_table" if args.inputs == "rebuilt" else args.table_dir
        feature_results, info["features"] = check_features(args, work, inputs_dir, scripts)
        target_results, info["targets"] = check_targets(args, work, inputs_dir, scripts, workers)
        results += feature_results + target_results
    finally:
        if not args.keep_work_dir:
            shutil.rmtree(work, ignore_errors=True)

    info.update(run_info(argv, vars(args), timestamp, time.time() - started))
    info["work_dir"] = str(work) if args.keep_work_dir else "removed"
    info["results"] = results
    args.report_root.mkdir(parents=True, exist_ok=True)
    (args.report_root / "determinism_check.json").write_text(json.dumps(info, indent=2, default=str) + "\n")
    report = render(results, info)
    (args.report_root / "determinism_check.md").write_text(report)
    print(report)
    differs = [r for r in results if not r.get("skipped") and not r["identical"]]
    log(f"{len(results) - len(differs)} of {len(results)} tables identical or skipped; wrote {args.report_root}")
    return 1 if differs else 0


if __name__ == "__main__":
    raise SystemExit(main())
