"""End-to-end tests of ``scripts/build_targets.py``, ``scripts/profile_targets.py`` and the
``mean_current_v0`` predictor on a synthetic workspace (three DIII-D training shots, one DIII-D
and one MAST test shot)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fusion_eq import current_basis as cb, frame_table as ft, io, starter_kit, submission as sub, target_build as tb, targets
from fusion_eq.coil_flux import GreenTable, adopted_scales
from fusion_eq.machines import machine_config
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
    rings = record["sample"]["border_rings"]
    assert set(rings) == {"DIII-D", "MAST"} and all(g["n_frames"] == 15 and g["rings"] == 3 for g in rings.values())
    assert all(0.0 <= g["pooled"] <= g["max"] <= 1.0 and g["pass"] == (g["max"] <= 1e-4) for g in rings.values())
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


@pytest.fixture(scope="module")
def profiled(ws) -> Path:
    """Runs the profile on the synthetic build (no ceiling report); returns the report root."""
    reports = ws["tmp"] / "reports"
    cmd = [sys.executable, str(REPO_ROOT / "scripts" / "profile_targets.py"), "--config", str(ws["config"]), "--split", str(ws["split"]),
           "--targets", str(ws["out"]), "--report-root", str(reports), "--ceiling", str(ws["tmp"] / "none.json"),
           "--data-root", str(ws["data"]), "--table-dir", str(ws["table_dir"]), "--green-dir", str(ws["green"]),
           "--response-cache", str(ws["tmp"] / "response"), "--cache-dir", str(ws["tmp"] / "profile_cache"), "--check-shots", "2",
           "--check-frames", "10", "--envelope-frames", "20", "--y16-frames", "6", "--y16-stride", "1", "--workers", "1"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stderr
    return reports


def test_profile_writes_every_check(ws, profiled):
    tmp_path = profiled.parent
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
    assert all(v["n_symmetry_sources"] == 81 and v["symmetry"] <= v["symmetry_all_sources"] for v in y9.values())
    assert all(v["ring_guard"]["rule"] == 1e-4 and v["thresholds"]["ring_share"] == 1e-4 for v in y9.values())
    expected_y9 = "pass" if all(v["pass"] for v in y9.values()) else "fail"
    assert checks["Y9"]["status"] == expected_y9 and "all sources" in checks["Y9"]["summary"]
    assert info["Y2_Y5"]["chain_max"] < 1e-10 and info["Y2_Y5"]["n_frames"] == 10
    cases = info["Y10_Y11"]
    assert set(cases) == {"nominal", "R0 +5%", "R0 -5%", "a0 +5%", "a0 -5%", "Morris 1998 (0.70, 0.50)"}
    assert cases["Morris 1998 (0.70, 0.50)"]["r0_m"] == 0.70 and cases["nominal"]["envelope"]["n_cells"] > 0
    assert cases["R0 +5%"]["envelope"]["centroid_r_m"] > cases["nominal"]["envelope"]["centroid_r_m"]
    y16 = info["Y16"]
    assert {"validation_with_remainder", "validation_without_remainder", "test_DIII-D", "test_MAST"} <= set(y16)
    assert 0.0 <= y16["test_MAST"]["boundary_share"] <= 1.0 and y16["test_MAST"]["n_frames"] > 0
    for m in ("DIII-D", "MAST"):
        v = y16[f"test_{m}"]
        spread = v["spread"]
        assert set(spread) == {"R_axis", "Z_axis", "kappa", "inner_R", "outer_R", "width", "height"}
        # Every quantity is counted over the frames with a boundary.
        assert all(s["n"] == round(v["boundary_share"] * v["n_frames"]) for s in spread.values())
        if spread["width"]["n"]:
            assert spread["inner_R"]["q50"] == v["median_inner_R"] and spread["kappa"]["q50"] == v["median_kappa"]
            assert spread["inner_R"]["q5"] <= spread["inner_R"]["q50"] <= spread["inner_R"]["q95"]
            assert spread["inner_R"]["q50"] < v["median_R_axis"] < spread["outer_R"]["q50"]
            assert spread["width"]["q5"] > 0.0 and spread["height"]["q5"] > 0.0
            assert spread["width"]["q50"] <= spread["outer_R"]["q95"] - spread["inner_R"]["q5"]
    assert "inner R" in checks["Y16"]["summary"] and "height" in checks["Y16"]["summary"]
    report = (tmp_path / "reports" / "v1" / "target_report.md").read_text()
    assert "## Y9. Response" in report and "Ring share, largest" in report and "## Y10 and Y11" in report and "/Users/" not in report
    assert "## Y16. Mean current on the test frames" in report and "| Inner R (m) | Outer R (m) | Width (m) | Height (m) |" in report
    assert "| MAST | " in report.split("## Y16. Mean current on the test frames")[1]


def test_mean_current_v0_composes_the_target_build_on_both_machines(ws, tmp_path):
    from fusion_eq.predictors import BaselineV0, MeanCurrentV0
    args = dict(split_path=ws["split"], cache_dir=tmp_path / "cache", green_dir=ws["green"], table_root=ws["table_dir"],
                data_root=ws["data"])
    predictor = sub.make_predictor("mean_current_v0", **args, targets_root=ws["out"], target_config=ws["config"])
    ids = predictor.setup(log=lambda m: None)
    objects = tb.TargetObjects.load(tb.target_dir(1, ws["out"]))
    assert isinstance(predictor, MeanCurrentV0) and predictor.name == "mean_current_v0"
    assert ids["target_build"]["config_hash"] == tb.TargetConfig.load(ws["config"]).hash()
    assert ids["target_build"]["checksums"] == objects.record["checksums"] and ids["sign_rule_c"] == {"DIII-D": -1.0, "MAST": 1.0}
    assert ids["machine_constants"]["MAST"]["major_radius"]["verified"] is True
    baseline = BaselineV0(**args)
    baseline.setup(log=lambda m: None)
    without = MeanCurrentV0(**args, targets_root=ws["out"], target_config=ws["config"], with_remainder=False)
    without.setup(log=lambda m: None)
    for machine, split, name in (("DIII-D", "diii_d_public_test", "d3d_shot_t1"), ("MAST", "mast_public_test", "mast_shot_m1")):
        shot = sub.strip_labels(io.load_shot(ws["data"] / split / f"{name}.parquet"))
        pred = predictor.predict_shot(shot, machine)
        frames = ft.build_shot_frames(shot, split, ft.BuildParams(), False)
        table = GreenTable.load(ws["green"] / ("green_d3d.npz" if machine == "DIII-D" else "green_mast.npz"))
        expected = tb.mean_current_maps(machine, frames, table, adopted_scales(machine, table.columns), objects)
        assert pred.psirz.dtype == np.float32 and pred.psirz.shape == (shot.n_frames, 65, 65)
        assert np.array_equal(pred.psirz, expected.astype(np.float32))
        base = baseline.predict_shot(shot, machine)
        assert np.array_equal(pred.q95, base.q95) and np.array_equal(pred.betaN, base.betaN)
        # The remainder is added on DIII-D only.
        same = np.array_equal(without.predict_shot(shot, machine).psirz, pred.psirz)
        assert same == (machine == "MAST")
        # Reversing every current together with Ip leaves the stored map unchanged.
        flipped = sub.strip_labels(io.load_shot(ws["data"] / split / f"{name}.parquet"))
        for col in machine_config(machine).current_columns:
            flipped.data[col] = -flipped[col]
        assert np.allclose(predictor.predict_shot(flipped, machine).psirz, pred.psirz, atol=1e-6 * np.abs(pred.psirz).max())
    # A build made with another configuration is refused.
    other = tmp_path / "other.yaml"
    other.write_text(SMALL_CONFIG.replace("n_components: 6", "n_components: 5"))
    with pytest.raises(ValueError, match="another configuration"):
        MeanCurrentV0(**args, targets_root=ws["out"], target_config=other).setup(log=lambda m: None)


def test_notebook_runs_on_the_synthetic_outputs(ws, profiled, tmp_path):
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    ceiling = tmp_path / "ceiling.json"
    keys = ("S", "r2_psi", "consistency")
    ceiling.write_text(json.dumps({"scores": {"f_mirror": dict.fromkeys(keys, 0.99), "h_symmetric_50": dict.fromkeys(keys, 0.98),
                                              "b_other": dict.fromkeys(keys, 0.5)}}))
    nb = nbformat.read(REPO_ROOT / "notebooks" / "07_targets.ipynb", as_version=4)
    parameters = [c for c in nb.cells if "parameters" in c.metadata.get("tags", [])]
    assert len(parameters) == 1
    parameters[0].source = "\n".join([
        f'DATA_ROOT = Path("{ws["data"]}")', f'TABLE_DIR = Path("{ws["table_dir"]}")', f'TARGET_ROOT = Path("{ws["out"]}")',
        f'GREEN_DIR = Path("{ws["green"]}")', f'RESPONSE_CACHE = Path("{ws["tmp"] / "response"}")',
        f'REPORT_ROOT = Path("{profiled}")', f'CEILING_PATH = Path("{ceiling}")', f'SPLIT_PATH = Path("{ws["split"]}")',
        f'CONFIG_PATH = Path("{ws["config"]}")', 'SPLIT_NAME = "diii_d_train"', "ENVELOPE_FRAMES = 20", "PLOT_FRAMES = 50", "SEED = 0"])
    nbclient.NotebookClient(nb, timeout=300, kernel_name="python3", resources={"metadata": {"path": str(REPO_ROOT)}}).execute()
    assert not any(o.get("output_type") == "error" for c in nb.cells if c.cell_type == "code" for o in c.outputs)
    headings = [c.source.splitlines()[0] for c in nb.cells if c.cell_type == "markdown" and c.source.startswith("## ")]
    assert [h.split(".")[0] for h in headings] == [f"## {i}" for i in range(1, 9)]
    # Section 8 holds the six decisions of docs/TARGET.md, each with its answer and the finding it rests on.
    findings = [c for c in nb.cells if c.cell_type == "markdown" and c.source.startswith("## 8.")][0]
    decisions = findings.source.split("\n### ")[1:]
    assert [d.splitlines()[0] for d in decisions] == [
        "Number of current components", "Whether the model phase rescales the predicted share", "Placement resolution",
        "Rate window of the remainder ridge", "MAST reference radii", "Target B or the fallback"]
    for d in decisions:
        body = " ".join(line for line in d.splitlines()[1:] if line.strip())
        assert body and "(F9" in body, d
    text = "\n".join(str(o.get("data", {}).get("text/plain", "")) + o.get("text", "")
                     for c in nb.cells if c.cell_type == "code" for o in c.outputs)
    # The notebook found the profile summary and the ceiling report (the profile itself ran without a ceiling).
    assert "target_summary.json not found" not in text and f"{ceiling} not found" not in text
    assert "| Y16 | Mean current |" in text and "Ring guard" in text
    assert "| inner_R |" in text and "| height |" in text and "no boundary size" not in text
    assert "recomposed from the parts against the stored map" in text and "| h_symmetric_50 |" in text and "b_other" not in text
    recomposed = float(text.split("recomposed from the parts against the stored map: ")[1].split(" ")[0])
    assert recomposed < 1e-10
    assert "the 50 ms ridge keeps the toroidal field's 10 ms rate" in text
