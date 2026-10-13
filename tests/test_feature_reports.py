"""End-to-end tests of the feature reports on synthetic shots: ``scripts/profile_features.py``
(X1 to X15), ``scripts/feature_linear_check.py`` (X16) and ``notebooks/06_features.ipynb``."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fusion_eq import features as fe, starter_kit
from tests.test_build_features import REPO_ROOT, _run, make_workspace

PROFILE_SAMPLES = ["--nn-train-frames", "100", "--nn-eval-frames", "10", "--corr-frames", "100", "--mirror-shots", "2"]
RUN_FIELDS = ("timestamp", "runtime_s", "args", "arguments", "repo_commit", "starter_commit")


@pytest.fixture(scope="module")
def ws(tmp_path_factory) -> dict:
    """Synthetic workspace with the frame table, the feature tables and the scaler built."""
    if not starter_kit.is_available():
        pytest.skip("starter kit not available")
    root = tmp_path_factory.mktemp("reports")
    ws = make_workspace(root)
    ws.update(features=root / "features", reports=root / "reports", cache=root / "cache")
    result = _run(ws, ws["features"], ws["reports"])
    assert result.returncode == 0, result.stderr
    return ws


def run_profile(ws: dict, feature_root: Path | None = None) -> subprocess.CompletedProcess:
    cmd = [sys.executable, str(REPO_ROOT / "scripts" / "profile_features.py"), "--split", str(ws["split"]),
           "--feature-root", str(feature_root or ws["features"]), "--report-root", str(ws["reports"]),
           "--table-dir", str(ws["table_dir"]), "--green-dir", str(ws["green"]), "--data-root", str(ws["data"]),
           *PROFILE_SAMPLES]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=300)


def run_linear_check(ws: dict) -> subprocess.CompletedProcess:
    cmd = [sys.executable, str(REPO_ROOT / "scripts" / "feature_linear_check.py"), "--full", "--split", str(ws["split"]),
           "--feature-root", str(ws["features"]), "--report-root", str(ws["reports"]), "--table-dir", str(ws["table_dir"]),
           "--green-dir", str(ws["green"]), "--data-root", str(ws["data"]), "--cache-dir", str(ws["cache"]),
           "--n-components", "2", "--frame-stride", "2", "--workers", "1"]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=300)


@pytest.fixture(scope="module")
def profiled(ws) -> dict:
    result = run_profile(ws)
    assert result.returncode == 0, result.stderr
    out = ws["reports"] / "v1"
    return {"dir": out, "info": json.loads((out / "profile.json").read_text())}


@pytest.fixture(scope="module")
def linear(ws) -> dict:
    result = run_linear_check(ws)
    assert result.returncode == 0, result.stderr
    return json.loads((ws["reports"] / "v1" / "linear_check.json").read_text())


def test_profile_writes_every_check_and_table(profiled):
    out, info = profiled["dir"], profiled["info"]
    config = fe.FeatureConfig.load()
    checks = {c["id"]: c for c in info["checks"]}
    assert list(checks) == [f"X{i}" for i in range(1, 16)]
    assert all(c["status"] in ("pass", "fail", "review") for c in checks.values())
    assert checks["X1"]["status"] == "pass" and checks["X2"]["status"] == "pass" and checks["X4"]["status"] == "pass"
    assert all(info["x5"]["identity"].values())
    # No synthetic frame has Ip below the threshold.
    assert all(d["n_affected"] == 0 for d in info["x3"]["low_ip"].values())
    assert info["n_rows"] == {"train": 24, "validation": 5, "diii_d_public_test": 4, "mast_public_test": 8}

    ranges = pd.read_parquet(out / "range_comparison.parquet")
    n_columns = len(config.all_columns())
    assert len(ranges) == n_columns * 3 * 4 and set(ranges["phase"]) == {"all", "rising", "flat_top", "late"}
    # The reference statistics do not depend on the compared split.
    band = ranges[ranges["phase"] == "all"].groupby("feature")[["ref_q01", "ref_q99", "ref_median"]].nunique(dropna=False)
    assert (band == 1).all().all()

    nn = pd.read_parquet(out / "nn_distance.parquet")
    assert len(nn) == len(config.sets) * (5 + 4 + 8)
    # Adding the mirrored training frames to the reference can only shorten a distance.
    assert (nn["distance_with_mirrors"] <= nn["distance"] + 1e-12).all()
    shares = nn[[c for c in nn.columns if c.startswith("share_")]]
    sums = shares.sum(axis=1, min_count=1)
    assert np.allclose(sums[sums.notna()], 1.0)

    corr = pd.read_parquet(out / "correlations.parquet")
    assert set(corr["kind"]) == {"coil_features", "raw_currents"}
    diagonal = corr[corr["a"] == corr["b"]]
    assert np.allclose(diagonal["r"].dropna(), 1.0)
    report = (out / "feature_report.md").read_text()
    assert all(f"| X{i} |" in report for i in range(1, 16)) and "/Users/" not in report


def test_profile_is_deterministic(ws, profiled):
    out = profiled["dir"]
    tables = {name: pd.read_parquet(out / name) for name in ("range_comparison.parquet", "nn_distance.parquet", "correlations.parquet")}
    result = run_profile(ws)
    assert result.returncode == 0, result.stderr
    for name, table in tables.items():
        pd.testing.assert_frame_equal(pd.read_parquet(out / name), table)
    again = json.loads((out / "profile.json").read_text())
    strip = lambda d: {k: v for k, v in d.items() if k not in RUN_FIELDS}  # noqa: E731
    assert json.dumps(strip(again), sort_keys=True) == json.dumps(strip(profiled["info"]), sort_keys=True)


def test_profile_stops_when_feature_rows_differ_from_the_frame_table(ws, tmp_path):
    root = tmp_path / "features"
    shutil.copytree(ws["features"], root)
    path = fe.features_path("diii_d_public_test", 1, root)
    pd.read_parquet(path).iloc[:-1].to_parquet(path, index=False)
    result = run_profile(ws, root)
    assert result.returncode != 0 and "differ (X1)" in result.stderr


def test_linear_check_scores_the_reference_and_the_feature_arms_on_the_same_frames(ws, linear):
    config = fe.FeatureConfig.load()
    scores = pd.DataFrame(linear["scores"])
    arms = ["hybrid_ridge", *[f"features_{s}" for s in config.sets]]
    assert sorted(scores["model"].unique()) == sorted(arms) and set(scores["subset"]) == {"all", "ip_positive"}
    assert linear["n_val_shots"] == 1 and linear["n_val_frames"] == 3 and linear["n_components"] == 2
    for _, g in scores.groupby("subset"):
        # Every arm takes q95 and betaN from the same scalar ridge, so the scalar term is shared.
        assert g["r2_qb"].nunique() == 1 and np.isfinite(g["S"]).all()
    overall = scores[scores["subset"] == "all"].set_index("model")["S"]
    for arm in arms[1:]:
        s = linear["status"][arm]
        assert s["delta_s"] == pytest.approx(overall[arm] - overall["hybrid_ridge"])
        assert s["status"] == ("pass" if s["delta_s"] >= 0 else "review")
        assert s["n_inputs"] == len(config.set_columns(arm.removeprefix("features_")))
    coef = linear["coefficient_r2"]
    assert {"features_set_a", "features_set_a without coil_pattern", "features_set_b without coil_terms",
            "raw inputs of hybrid_ridge"} <= set(coef)
    assert all(r["train"] <= 1.0 for r in coef.values())
    assert any(p.name.endswith(".npz") and "plasma_per_ip" in p.name for p in ws["cache"].iterdir())
    report = (ws["reports"] / "v1" / "linear_check.md").read_text()
    assert "features_set_a" in report and "hybrid_ridge" in report


def test_notebook_runs_on_the_synthetic_outputs(ws, profiled, linear):
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    nb = nbformat.read(REPO_ROOT / "notebooks" / "06_features.ipynb", as_version=4)
    parameters = [c for c in nb.cells if "parameters" in c.metadata.get("tags", [])]
    assert len(parameters) == 1
    parameters[0].source = "\n".join([
        f'DATA_ROOT = Path("{ws["data"]}")', f'TABLE_DIR = Path("{ws["table_dir"]}")',
        f'FEATURE_ROOT = Path("{ws["features"]}")', f'GREEN_DIR = Path("{ws["green"]}")',
        f'REPORT_ROOT = Path("{ws["reports"]}")', f'SPLIT_PATH = Path("{ws["split"]}")',
        f'CONFIG_PATH = Path("{REPO_ROOT / "configs" / "features_v1.yaml"}")',
        "PLOT_FRAMES = 1000", "SCATTER_FRAMES = 1000", "SEED = 0"])
    nbclient.NotebookClient(nb, timeout=300, kernel_name="python3", resources={"metadata": {"path": str(REPO_ROOT)}}).execute()
    headings = [c.source.splitlines()[0] for c in nb.cells if c.cell_type == "markdown" and c.source.startswith("## ")]
    assert [h.split(".")[0] for h in headings] == [f"## {i}" for i in range(1, 11)]
    text = "\n".join(str(o.get("data", {}).get("text/plain", "")) + o.get("text", "")
                     for c in nb.cells if c.cell_type == "code" for o in c.outputs)
    assert "X15" in text and "features_set_a" in text and "not found" not in text
    assert not any(o.get("output_type") == "error" for c in nb.cells if c.cell_type == "code" for o in c.outputs)
