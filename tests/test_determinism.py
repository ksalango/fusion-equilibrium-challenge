"""Tests of the table comparison of the determinism check (``fusion_eq.determinism``) and of
``scripts/check_determinism.py`` on a synthetic workspace."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from fusion_eq import determinism as det, starter_kit
from tests.test_build_features import REPO_ROOT, make_workspace
from tests.test_target_build import SMALL_CONFIG


def table(**overrides) -> pa.Table:
    columns = {"shot": pa.array(["a", "a", "b", "b"]), "frame": pa.array([0, 1, 0, 1], pa.int32()),
               "x": pa.array([1.0, np.nan, 3.0, 4.0], pa.float32()), "flag": pa.array([True, False, None, True]),
               "phase": pa.array(["rising", "late", None, "flat_top"])}
    columns.update(overrides)
    return pa.table(columns)


def test_digest_ignores_layout_and_identical_tables_compare_equal():
    t = table()
    chunked = pa.concat_tables([t.slice(0, 1), t.slice(1, 3)])
    assert chunked.column("x").num_chunks == 2 and det.table_digest(chunked) == det.table_digest(t)
    r = det.compare_tables(t, chunked)
    assert r["identical"] and r["values_equal"] and "first" not in r and det.summary_line(r) == "identical"
    assert det.table_digest(t) != det.table_digest(t.rename_columns(["shot", "frame", "y", "flag", "phase"]))


def test_first_differing_column_and_row_are_reported():
    t = table()
    changed = table(x=pa.array([1.0, np.nan, 3.0, 4.5], pa.float32()), phase=pa.array(["rising", "late", None, "late"]))
    r = det.compare_tables(t, changed)
    assert not r["identical"] and not r["values_equal"] and r["n_differing_columns"] == 2
    assert r["first"] == {"column": "x", "row": 3, "key": {"shot": "b", "frame": 1}, "key_rebuilt": {"shot": "b", "frame": 1},
                          "stored": 4.0, "rebuilt": 4.5, "n_rows": 1}
    assert r["differing_columns"] == ["x", "phase"]
    assert det.summary_line(r).startswith("differs: first column x, row 3 (shot b, frame 1): stored 4.0, rebuilt 4.5")
    # A null against a value is a difference; a null against a null is not.
    r = det.compare_tables(t, table(flag=pa.array([True, False, False, True])))
    assert r["first"]["column"] == "flag" and r["first"]["row"] == 2 and r["first"]["stored"] is None


def test_equal_values_with_different_bytes_and_structural_differences():
    zero = det.compare_tables(table(x=pa.array([0.0, 1.0, 2.0, 3.0], pa.float32())), table(x=pa.array([-0.0, 1.0, 2.0, 3.0], pa.float32())))
    assert zero["values_equal"] and not zero["identical"] and "bytes differ" in zero["reason"]
    t = table()
    assert "row counts differ: stored 4, rebuilt 3; keys agree up to row 3" in det.compare_tables(t, t.slice(0, 3))["reason"]
    assert "missing ['phase']" in det.compare_tables(t, t.drop_columns(["phase"]))["reason"]
    typed = det.compare_tables(t, table(x=pa.array([1.0, np.nan, 3.0, 4.0], pa.float64())))
    assert typed["reason"] == "type of x differs: stored float, rebuilt double" and not typed["identical"]


def test_read_rows_keeps_file_order(tmp_path):
    path = tmp_path / "t.parquet"
    pq.write_table(table(), path, row_group_size=1)
    rows = det.read_rows(path, ["b"])
    assert rows.column("shot").to_pylist() == ["b", "b"] and rows.column("frame").to_pylist() == [0, 1]
    assert det.read_rows(path, ["z"]).num_rows == 0


# ------------------------------------------------------------------------------ the script
def snapshot(*roots: Path) -> dict:
    return {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for root in roots for p in sorted(Path(root).rglob("*")) if p.is_file()}


def run_check(ws: dict, report: Path, *extra: str) -> subprocess.CompletedProcess:
    cmd = [sys.executable, str(REPO_ROOT / "scripts" / "check_determinism.py"), "--d3d-shots", "a1", "a2", "a3",
           "--mast-shots", "m1", "--data-root", str(ws["data"]), "--table-dir", str(ws["table_dir"]),
           "--green-dir", str(ws["green"]), "--split", str(ws["split"]), "--response-cache", str(ws["tmp"] / "response"),
           "--references-dir", str(ws["tmp"] / "refs"), "--report-root", str(report), "--workers", "1",
           "--feature-root", str(ws["features"]), "--targets", str(ws["targets"]), "--target-config", str(ws["config"]), *extra]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=600)


@pytest.fixture(scope="module")
def ws(tmp_path_factory) -> dict:
    if not starter_kit.is_available():
        pytest.skip("starter kit not available")
    root = tmp_path_factory.mktemp("determinism")
    ws = make_workspace(root)
    ws["features"], ws["targets"], ws["config"] = root / "features", root / "targets", root / "target.yaml"
    ws["config"].write_text(SMALL_CONFIG)
    common = ["--data-root", str(ws["data"]), "--table-dir", str(ws["table_dir"]), "--green-dir", str(ws["green"]),
              "--split", str(ws["split"])]
    result = subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "build_features.py"), "--config",
                             str(REPO_ROOT / "configs" / "features_v2.yaml"), "--out", str(ws["features"]),
                             "--report-root", str(root / "feature_reports"), *common], capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stderr
    result = subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "build_targets.py"), "--config", str(ws["config"]),
                             "--out", str(ws["targets"]), "--response-cache", str(ws["tmp"] / "response"),
                             "--references-dir", str(root / "refs"), "--workers", "1", *common],
                            capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stderr
    return ws


def test_the_rebuild_matches_the_stored_tables_and_writes_nothing_there(ws, tmp_path):
    stored = (ws["data"], ws["table_dir"], ws["features"], ws["targets"], ws["green"], ws["tmp"] / "response")
    before = snapshot(*stored)
    result = run_check(ws, tmp_path / "report")
    assert result.returncode == 0, result.stdout + result.stderr
    assert snapshot(*stored) == before
    info = json.loads((tmp_path / "report" / "determinism_check.json").read_text())
    parts = [(r["table"], r["part"]) for r in info["results"]]
    assert parts == [("frame table", "diii_d_train"), ("frame table", "mast_public_test"), ("features v2", "diii_d_train"),
                     ("features v2", "mast_public_test"), ("targets v1", "train"), ("targets v1", "validation")]
    for r in info["results"]:
        assert r["identical"] and r["values_equal"] and r["rows_stored"] == r["rows_rebuilt"] > 0, r
    by_part = {(r["table"], r["part"]): r for r in info["results"]}
    assert by_part[("frame table", "diii_d_train")]["n_shots"] == 3 and by_part[("targets v1", "validation")]["n_shots"] == 1
    assert all(info["targets"]["object_checksums"].values()) and info["work_dir"] == "removed"
    assert info["features"]["point_tables"] == {"DIII-D": "from the cache", "MAST": "from the cache"}
    report = (tmp_path / "report" / "determinism_check.md").read_text()
    assert "| frame table | diii_d_train | 3 |" in report and "/Users/" not in report


def test_a_changed_value_is_located(ws, tmp_path):
    features = tmp_path / "features"
    shutil.copytree(ws["features"], features)
    path = features / "v2" / "diii_d_train.parquet"
    t = pq.read_table(path)
    column = t.column_names[2]
    values = t.column(column).to_numpy(zero_copy_only=False).copy()
    values[5] = values[5] + 1.0 if np.isfinite(values[5]) else 1.0
    pq.write_table(t.set_column(2, column, pa.array(values, t.schema.field(column).type)), path)
    result = run_check({**ws, "features": features}, tmp_path / "report", "--inputs", "stored", "--targets", str(tmp_path / "none"))
    assert result.returncode == 1, result.stderr
    info = json.loads((tmp_path / "report" / "determinism_check.json").read_text())
    by_part = {(r["table"], r["part"]): r for r in info["results"]}
    bad = by_part[("features v2", "diii_d_train")]
    assert not bad["identical"] and bad["first"]["column"] == column and bad["first"]["row"] == 5 and bad["first"]["n_rows"] == 1
    assert bad["first"]["key"] == {"shot": t.column("shot")[5].as_py(), "frame": t.column("frame")[5].as_py()}
    assert by_part[("features v2", "mast_public_test")]["identical"] and by_part[("frame table", "diii_d_train")]["identical"]
    assert by_part[("targets v1", "all lists")]["skipped"].startswith("no target build")
    assert f"differs: first column {column}, row 5" in (tmp_path / "report" / "determinism_check.md").read_text()


def test_outputs_inside_the_stored_locations_are_refused(ws, tmp_path):
    work = ws["table_dir"] / "check"
    result = run_check(ws, tmp_path / "report", "--work-dir", str(work))
    assert result.returncode != 0 and "nothing is written there" in result.stderr and not work.exists()
    result = run_check(ws, ws["features"] / "report")
    assert result.returncode != 0 and "--report-root" in result.stderr
