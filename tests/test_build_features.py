"""End-to-end test of scripts/build_features.py on synthetic shots: frame table with Part 0,
feature tables, layout check, scaler and a deterministic rebuild."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fusion_eq import coil_flux as cf, dataset, features as fe, io
from fusion_eq.feature_scaler import RobustScaler
from tests.conftest import D3D_F_COILS, drop_labels, make_d3d_shot, make_mast_shot, write_shot

REPO_ROOT = Path(__file__).resolve().parents[1]


def _far_coils_d3d(shot: dict) -> dict:
    """Coils kept away from the sample points (R up to 2.17 m, Z within 1.0 m)."""
    n = len(shot["coil_name"])
    shot["coil_R"] = np.full(n, 2.45)
    shot["coil_Z"] = np.linspace(-1.55, 1.55, n)
    return shot


def _varied_d3d(shot: dict, phase: float) -> dict:
    """Ip, the shaping coils and the Thomson profile shape varying in time, so that no feature
    is constant over the training frames (a constant feature cannot be scaled)."""
    ip = np.asarray(shot["magnetics_plasma_current"], dtype=np.float64)
    k = np.arange(ip.size)
    shot["magnetics_plasma_current"] = (ip * (1.0 + 0.3 * np.sin(k / 400.0 + phase))).astype(np.float32)
    t = np.asarray(shot["magnetics_time"], dtype=np.float64)
    for i, name in enumerate(D3D_F_COILS):
        shot[f"magnetics_{name}"] = shot[f"magnetics_{name}"] * (1.0 + 0.4 * np.sin(t / (60.0 + 7.0 * i) + phase + i))
    times = np.asarray(shot["thomson_core_times"], dtype=np.float64)
    te = np.asarray(shot["thomson_core_Te"], dtype=np.float64).copy()
    base = np.linspace(1.5, 0.5, 5)
    shape = base[None, :] ** (1.0 + 0.5 * np.sin(times / 50.0 + phase))[:, None]
    te[:, :5] = np.where(te[:, :5] > 0, 1000.0 * shape, 0.0)
    shot["thomson_core_Te"] = te
    return shot


def _far_coils_mast(shot: dict) -> dict:
    names = list(shot["coil_name"])
    shot["coil_R"] = np.array([0.12 if name.startswith("sol") else 1.75 for name in names])
    return shot


@pytest.fixture
def workspace(tmp_path, starter) -> dict:
    return make_workspace(tmp_path)


def make_workspace(tmp_path: Path) -> dict:
    """Synthetic raw data (three DIII-D training shots, one DIII-D and one MAST test shot),
    their frame table, grid Green's tables and a split file under ``tmp_path``."""
    data = tmp_path / "data"
    for split in dataset.DATA_SPLITS:
        (data / split).mkdir(parents=True)
    for phase, (shot_id, kwargs) in enumerate((("a1", {}), ("a2", {"n_frames": 5}), ("a3", {"ip_fix_needed": False}))):
        shot = _varied_d3d(_far_coils_d3d(make_d3d_shot(**kwargs)), float(phase))
        write_shot(data / "diii_d_train" / f"d3d_shot_{shot_id}.parquet", shot)
    write_shot(data / "diii_d_public_test" / "d3d_shot_t1.parquet", drop_labels(_far_coils_d3d(make_d3d_shot(n_frames=4))))
    write_shot(data / "mast_public_test" / "mast_shot_m1.parquet", drop_labels(_far_coils_mast(make_mast_shot())))
    table_dir = tmp_path / "frame_table"
    result = subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "build_frame_table.py"), "--data-root", str(data),
                             "--out", str(table_dir), "--workers", "1"], capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr
    green = tmp_path / "green"
    green.mkdir()
    for split, name in (("diii_d_public_test", "green_d3d.npz"), ("mast_public_test", "green_mast.npz")):
        path = dataset.list_shots(data, split)[0]
        shot = io.load_shot(path, [*cf.GEOMETRY_COLUMNS, *cf.GRID_COLUMNS], apply=False)
        cf.build_green_table(shot.machine, cf.CoilGeometry.from_shot(shot), shot["efit_grid_R"], shot["efit_grid_Z"],
                             n_filaments=2).save(green / name)
    split_file = tmp_path / "split.json"
    split_file.write_text(json.dumps({"train": ["a1", "a3"], "validation": ["a2"], "local_test": [],
                                      "working_train": ["a1"], "working_validation": ["a2"]}))
    return {"data": data, "table_dir": table_dir, "green": green, "split": split_file, "tmp": tmp_path}


def _run(ws: dict, out: Path, reports: Path, *extra: str) -> subprocess.CompletedProcess:
    cmd = [sys.executable, str(REPO_ROOT / "scripts" / "build_features.py"), "--data-root", str(ws["data"]),
           "--table-dir", str(ws["table_dir"]), "--green-dir", str(ws["green"]), "--split", str(ws["split"]),
           "--out", str(out), "--report-root", str(reports), "--shots-per-chunk", "2", *extra]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=300)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_build_features_script_end_to_end_and_deterministic(workspace):
    ws = workspace
    out1, out2, reports1, reports2 = (ws["tmp"] / n for n in ("f1", "f2", "r1", "r2"))
    for out, reports in ((out1, reports1), (out2, reports2)):
        result = _run(ws, out, reports)
        assert result.returncode == 0, result.stderr
    config = fe.FeatureConfig.load()
    for split in dataset.DATA_SPLITS:
        assert _sha256(fe.features_path(split, 1, out1)) == _sha256(fe.features_path(split, 1, out2))
    assert _sha256(reports1 / "v1" / "scaler.json") == _sha256(reports2 / "v1" / "scaler.json")

    info = json.loads((out1 / "v1" / "build.json").read_text())
    assert info["x1_pass"] and info["layout_check"]["status"] == "pass"
    assert info["counts"]["diii_d_train"] == {"n_shots": 3, "n_rows": 29, "n_frame_table_rows": 29, "x1_rows_match": True}
    assert info["resources"]["MAST"]["constants"]["tf_turns"]["verified"] is True
    assert info["resources"]["DIII-D"]["point_table"]["built"] and info["config_hash"] == config.hash()
    assert info["scaler"]["n_train_shots_found"] == 2 and info["scaler"]["n_rows_with_mirrors"] == 2 * (12 + 12)
    assert all(fe.check_reads(cols, m) for m, reads in info["x2_reads"].items() for cols in reads.values())

    train = fe.read_features("diii_d_train", 1, root=out1)
    assert list(train.columns) == ["shot", "frame", *config.all_columns()]
    assert train["shot"].tolist() == ["a1"] * 12 + ["a2"] * 5 + ["a3"] * 12 and train["frame"].dtype == np.int32
    assert np.isfinite(train[config.columns_of("coil_pt")].to_numpy()).all()
    mast = fe.read_features("mast_public_test", 1, root=out1)
    assert len(mast) == 8 and np.isfinite(mast[config.columns_of("coil_pt")].to_numpy()).all()
    clearance = pd.read_csv(reports1 / "v1" / "layout_clearance.csv")
    assert len(clearance) == 2 * config.layout.n_points and clearance["ok"].all()

    scaler = RobustScaler.load(reports1 / "v1" / "scaler.json")
    x, columns = fe.feature_matrix("mast_public_test", None, "set_b", config=config, scaler=scaler, root=out1)
    assert x.shape == (8, 16) and np.isfinite(x).all()


def test_build_features_script_stops_without_part_0_columns(workspace):
    ws = workspace
    path = ws["table_dir"] / "diii_d_train.parquet"
    frames = pd.read_parquet(path).drop(columns=["ip__rate50"])
    frames.to_parquet(path, index=False)
    result = _run(ws, ws["tmp"] / "f", ws["tmp"] / "r")
    assert result.returncode != 0 and "rebuild the frame table with Part 0" in result.stderr


def test_variant_build_under_another_mast_coil_model(workspace):
    ws = workspace
    out, reports = ws["tmp"] / "f", ws["tmp"] / "r"
    assert _run(ws, out, reports).returncode == 0
    base = json.loads((out / "v1" / "build.json").read_text())
    assert base["coil_models"] == {"DIII-D": "solenoid_flat", "MAST": "sol_half"} and base["variant"] is None
    assert base["resources"]["MAST"]["coil_model"]["name"] == base["resources"]["MAST"]["coil_model_name"] == "sol_half"
    assert fe.build_coil_model(out / "v1", "MAST") == "sol_half" and fe.build_coil_model(out / "v1", "DIII-D") == "solenoid_flat"
    stored = {s: _sha256(fe.features_path(s, 1, out)) for s in dataset.DATA_SPLITS}
    scaler = _sha256(reports / "v1" / "scaler.json")

    # The MAST table under the unit model, twice into separate roots: byte-identical (D7), in its
    # own folder, with no scaler; the version's tables and scaler are untouched.
    variant = ["--splits", "mast_public_test", "--coil-model", "MAST=unit", "--variant", "unit"]
    for root in ("f", "g"):
        result = _run(ws, ws["tmp"] / root, ws["tmp"] / f"r_{root}" if root == "g" else reports, *variant)
        assert result.returncode == 0, result.stderr
    path = fe.features_path("mast_public_test", 1, out, "unit")
    assert path.parent.name == "v1_unit" and _sha256(path) == _sha256(fe.features_path("mast_public_test", 1, ws["tmp"] / "g", "unit"))
    assert sorted(p.name for p in path.parent.glob("*.parquet")) == ["mast_public_test.parquet"]
    assert {s: _sha256(fe.features_path(s, 1, out)) for s in dataset.DATA_SPLITS} == stored
    assert _sha256(reports / "v1" / "scaler.json") == scaler and not (reports / "v1_unit" / "scaler.json").exists()
    assert (reports / "v1_unit" / "layout_clearance.csv").is_file()
    record = json.loads((path.parent / "build.json").read_text())
    assert record["variant"] == "unit" and record["coil_models"]["MAST"] == "unit" and "variant build" in record["scaler"]["note"]
    assert fe.build_coil_model(path.parent, "MAST") == "unit"

    # Only the features that read the coil currents change: every other column is identical, the
    # coil points move, and the solenoid flux spent (linear in the solenoid scale) doubles exactly.
    config = fe.FeatureConfig.load()
    a, b = fe.read_features("mast_public_test", 1, root=out), fe.read_features("mast_public_test", 1, root=out, variant="unit")
    reads = fe.feature_reads(config, "MAST")
    for c, spec in config.column_specs().items():
        if "sol__mean" not in reads[spec.name]:
            assert np.array_equal(a[c].to_numpy(), b[c].to_numpy(), equal_nan=True), c
    assert not np.allclose(a[config.columns_of("coil_pt")].to_numpy(), b[config.columns_of("coil_pt")].to_numpy())
    np.testing.assert_array_equal(b["ohmic_flux_used"].to_numpy(), 2.0 * a["ohmic_flux_used"].to_numpy())

    # A rebuild of the version's folder under another MAST model stops and leaves it as it was.
    result = _run(ws, out, reports, "--splits", "mast_public_test", "--coil-model", "MAST=unit")
    assert result.returncode != 0 and "was built with the MAST coil model 'sol_half'" in result.stderr
    assert _sha256(fe.features_path("mast_public_test", 1, out)) == stored["mast_public_test"]
    result = _run(ws, out, reports, "--coil-model", "MAST=solenoid_flat")
    assert result.returncode != 0 and "no model 'solenoid_flat' on MAST" in result.stderr
