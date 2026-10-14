"""End-to-end tests of ``scripts/build_targets.py`` and ``scripts/profile_targets.py`` on a
synthetic workspace (three DIII-D training shots, one DIII-D and one MAST test shot)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fusion_eq import current_basis as cb, starter_kit, target_build as tb, targets
from fusion_eq.coil_flux import GreenTable, adopted_scales
from tests.test_build_features import REPO_ROOT, make_workspace

SMALL_CONFIG = "version: 1\ncurrent:\n  n_components: 6\n  fit_frames: 15\nremainder:\n  n_components: 3\n"


def build(ws: dict, out: Path, *extra: str) -> subprocess.CompletedProcess:
    cmd = [sys.executable, str(REPO_ROOT / "scripts" / "build_targets.py"), "--config", str(ws["config"]), "--split", str(ws["split"]),
           "--data-root", str(ws["data"]), "--table-dir", str(ws["table_dir"]), "--green-dir", str(ws["green"]),
           "--response-cache", str(ws["tmp"] / "response"), "--references-dir", str(ws["tmp"] / "refs"), "--out", str(out),
           "--workers", "1", *extra]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=600)


@pytest.fixture(scope="module")
def ws(tmp_path_factory) -> dict:
    if not starter_kit.is_available():
        pytest.skip("starter kit not available")
    root = tmp_path_factory.mktemp("targets")
    ws = make_workspace(root)
    ws["config"] = root / "target.yaml"
    ws["config"].write_text(SMALL_CONFIG)
    result = build(ws, root / "out")
    assert result.returncode == 0, result.stderr
    ws["out"] = root / "out"
    return ws


def test_build_writes_the_tables_objects_and_record(ws):
    folder = tb.target_dir(1, ws["out"])
    record = json.loads((folder / "build.json").read_text())
    assert record["fit_list"] == "train" and record["validation_list"] == "validation" and record["stride"] == 1
    assert record["lists"]["local_test"]["skipped"] == "empty list" and set(record["tables"]) == {"train", "validation"}
    assert record["sample"]["n_frames"] == 15 and record["sample"]["requested"] == 15
    assert record["current_basis"]["n_components"] == 6 and record["current_basis"]["even"] + record["current_basis"]["odd"] == 6
    assert set(record["remainder"]["windows"]) == {"10ms", "50ms"} and record["remainder"]["default_window"] in {"10ms", "50ms"}
    assert all(len(w["columns"]) == 42 for w in record["remainder"]["windows"].values())
    assert record["machine_constants"]["MAST"]["major_radius"]["verified"] is True
    assert set(record["checksums"]) == set(tb.OBJECT_FILES.values())
    assert all(record["checksums"][f] == tb.sha256_of(folder / f) for f in record["checksums"])
    train = pd.read_parquet(folder / "train.parquet")
    assert len(train) == record["lists"]["train"]["n_rows"] == 24
    expected = {"shot", "frame", "ip_ka", "phase", "boundary_found", "share_total", "share_outside", "recon_error",
                "recon_error_mirror", "flux_range", "remainder_rms_over_range", "passes_filter", *tb.SHAPE_COLUMNS,
                *[f"c_{j:02d}" for j in range(6)], *[f"r_{j:02d}" for j in range(3)]}
    assert expected <= set(train.columns) and train["frame"].dtype == np.int32 and train["c_00"].dtype == np.float32
    assert train["boundary_found"].all() and np.isfinite(train["R_axis"]).all()
    objects = tb.TargetObjects.load(folder)
    assert objects.current_basis.n_components == 6 and objects.remainder_model.n_components == 3
    assert objects.placements["DIII-D"].identity and not objects.placements["MAST"].identity
    assert objects.record["config_hash"] == tb.TargetConfig.load(ws["config"]).hash()


def test_rebuild_is_deterministic(ws, tmp_path):
    result = build(ws, tmp_path / "again")
    assert result.returncode == 0, result.stderr
    first, second = tb.target_dir(1, ws["out"]), tb.target_dir(1, tmp_path / "again")
    a, b = json.loads((first / "build.json").read_text()), json.loads((second / "build.json").read_text())
    assert a["checksums"] == b["checksums"]
    for name in ("train", "validation"):
        assert tb.sha256_of(first / f"{name}.parquet") == tb.sha256_of(second / f"{name}.parquet")


def test_the_build_refuses_unlabelled_and_mast_splits_and_unknown_lists(ws, tmp_path):
    for split, message in (("mast_public_test", "MAST equilibrium labels are not read"), ("diii_d_public_test", "no equilibrium labels")):
        result = build(ws, tmp_path / "x", "--split-name", split)
        assert result.returncode != 0 and message in result.stderr
    result = build(ws, tmp_path / "x", "--lists", "validation")
    assert result.returncode != 0 and "training list" in result.stderr
    result = build(ws, tmp_path / "x", "--lists", "train", "mast_public_test")
    assert result.returncode != 0 and "not in the split file" in result.stderr
    assert not (tmp_path / "x").exists()


def test_mean_current_maps_compose_the_coil_flux_and_the_mean_share(ws):
    objects = tb.TargetObjects.load(tb.target_dir(1, ws["out"]))
    from fusion_eq import frame_table as ft
    for machine, split, green in (("DIII-D", "diii_d_public_test", "green_d3d.npz"), ("MAST", "mast_public_test", "green_mast.npz")):
        table = GreenTable.load(ws["green"] / green)
        scales = adopted_scales(machine, table.columns)
        rows = ft.read_frame_table(split, root=ws["table_dir"])
        maps = tb.mean_current_maps(machine, rows, table, scales, objects, with_remainder=False)
        ip = rows["ip__mean"].to_numpy(dtype=np.float64)
        coil = targets.coil_flux_of_frames(table, rows, scales, machine)
        expected = targets.to_stored(coil + ip[:, None, None] * 1e3 * objects.flux_bases[machine].phi_mean.reshape(65, 65), ip, machine)
        assert np.allclose(maps, expected, atol=1e-12 * np.abs(expected).max())
    # On DIII-D the remainder changes the map; on MAST it is never added.
    rows = ft.read_frame_table("diii_d_public_test", root=ws["table_dir"])
    table = GreenTable.load(ws["green"] / "green_d3d.npz")
    scales = adopted_scales("DIII-D", table.columns)
    assert not np.allclose(tb.mean_current_maps("DIII-D", rows, table, scales, objects),
                           tb.mean_current_maps("DIII-D", rows, table, scales, objects, with_remainder=False))


def test_profile_writes_every_check(ws, tmp_path):
    cmd = [sys.executable, str(REPO_ROOT / "scripts" / "profile_targets.py"), "--config", str(ws["config"]), "--split", str(ws["split"]),
           "--targets", str(ws["out"]), "--report-root", str(tmp_path / "reports"), "--ceiling", str(tmp_path / "none.json"),
           "--data-root", str(ws["data"]), "--table-dir", str(ws["table_dir"]), "--green-dir", str(ws["green"]),
           "--response-cache", str(ws["tmp"] / "response"), "--cache-dir", str(tmp_path / "cache"), "--check-shots", "2",
           "--check-frames", "10", "--envelope-frames", "20", "--y16-frames", "6", "--y16-stride", "1", "--workers", "1"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stderr
    info = json.loads((tmp_path / "reports" / "v1" / "target_summary.json").read_text())
    checks = {c["id"]: c for c in info["checks"]}
    assert list(checks) == [f"Y{i}" for i in range(1, 17)]
    for cid in ("Y1", "Y2", "Y4", "Y5", "Y7", "Y8"):
        assert checks[cid]["status"] == "pass", checks[cid]
    for cid in ("Y6", "Y10", "Y11", "Y12", "Y13", "Y14", "Y15", "Y16"):
        assert checks[cid]["status"] == "review"
    assert "not found" in checks["Y15"]["summary"]
    y9 = info["Y9"]
    assert set(y9) == {"DIII-D", "MAST"} and all(v["n_sources"] == 121 and v["flip"] < 1e-12 for v in y9.values())
    assert y9["MAST"]["thresholds"]["beyond_5"] == 0.1 and y9["DIII-D"]["thresholds"]["symmetry"] == 1e-3
    assert info["Y2_Y5"]["chain_max"] < 1e-10 and info["Y2_Y5"]["n_frames"] == 10
    cases = info["Y10_Y11"]
    assert set(cases) == {"nominal", "R0 +5%", "R0 -5%", "a0 +5%", "a0 -5%", "Morris 1998 (0.70, 0.50)"}
    assert cases["Morris 1998 (0.70, 0.50)"]["r0_m"] == 0.70 and cases["nominal"]["envelope"]["n_cells"] > 0
    assert cases["R0 +5%"]["envelope"]["centroid_r_m"] > cases["nominal"]["envelope"]["centroid_r_m"]
    y16 = info["Y16"]
    assert {"validation_with_remainder", "validation_without_remainder", "test_DIII-D", "test_MAST"} <= set(y16)
    assert 0.0 <= y16["test_MAST"]["boundary_share"] <= 1.0 and y16["test_MAST"]["n_frames"] > 0
    report = (tmp_path / "reports" / "v1" / "target_report.md").read_text()
    assert "## Y9. Response" in report and "## Y10 and Y11" in report and "/Users/" not in report
