"""End-to-end tests of the feature reports on synthetic shots: ``scripts/profile_features.py``
(X1 to X15), ``scripts/feature_linear_check.py`` (X16) and ``notebooks/06_features.ipynb``."""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fusion_eq import baseline as bl, current_target as ct, features as fe, frame_table as ft, starter_kit, targets
from fusion_eq.coil_flux import GreenTable, adopted_scales
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


def run_linear_check(ws: dict, *extra: str, cache: Path | None = None) -> subprocess.CompletedProcess:
    cmd = [sys.executable, str(REPO_ROOT / "scripts" / "feature_linear_check.py"), "--full", "--split", str(ws["split"]),
           "--feature-root", str(ws["features"]), "--report-root", str(ws["reports"]), "--table-dir", str(ws["table_dir"]),
           "--green-dir", str(ws["green"]), "--data-root", str(ws["data"]), "--cache-dir", str(cache or ws["cache"]),
           "--n-components", "2", "--frame-stride", "2", "--workers", "1", *extra]
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
    assert [h.split(".")[0] for h in headings] == [*(f"## {i}" for i in range(1, 7)), "## 6b", *(f"## {i}" for i in range(7, 11))]
    text = "\n".join(str(o.get("data", {}).get("text/plain", "")) + o.get("text", "")
                     for c in nb.cells if c.cell_type == "code" for o in c.outputs)
    assert "X15" in text and "features_set_a" in text and "not found" not in text
    assert not any(o.get("output_type") == "error" for c in nb.cells if c.cell_type == "code" for o in c.outputs)
    # Section 6b: the tables are printed as Markdown, and the in-memory refit of the version-1
    # sets reproduces scaler.json and the X10 distances of profile.json exactly.
    assert "| quantity | phase | split |" in text and "| variant | split | median | median, v1 |" in text
    assert "shaping even under the mirror: True" in text
    recomputed = [line for line in text.splitlines() if line.startswith("| set_") and "version 1 recomputed" in line]
    assert len(recomputed) == 2 + 2 * 3
    assert all(cell.strip() == "0" for line in recomputed for cell in line.strip("|").split("|")[-1:])
    profile_rows = [line for line in recomputed if "DIII-D" in line or "MAST" in line]
    assert len(profile_rows) == 6 and all(c.strip() == "0" for line in profile_rows for c in line.strip("|").split("|")[2:])


FOLLOWUP = ("--by-phase", "--target", "masked_current_flux", "--ip-frac-unscaled")


def files_under(folder: Path) -> list[str]:
    return sorted(str(p.relative_to(folder)) for p in folder.rglob("*") if p.is_file())


def test_followup_reports_every_variant_beside_version_1_and_writes_nothing_else(ws, linear, tmp_path):
    out = ws["reports"] / "v1"
    before = {p.name: p.read_bytes() for p in out.glob("linear_check.*")}
    cache_before = files_under(ws["cache"])
    result = run_linear_check(ws, *FOLLOWUP, "--response-cache", str(tmp_path / "response"))
    assert result.returncode == 0, result.stderr
    # Version 1 outputs, the caches and the response folder are untouched.
    assert {p.name: p.read_bytes() for p in out.glob("linear_check.*")} == before
    assert files_under(ws["cache"]) == cache_before and not (tmp_path / "response").exists()

    info = json.loads((out / "linear_check_followup.json").read_text())
    assert info["variants"] == ["version 1", "masked_current_flux", "ip_frac_unscaled"]
    scores = pd.DataFrame(info["scores"])
    assert scores.equals(pd.read_parquet(out / "linear_check_followup.parquet"))
    arms = {v: sorted(g["model"].unique()) for v, g in scores.groupby("variant")}
    assert arms == {"version 1": ["features_set_a", "features_set_b", "hybrid_ridge"],
                    "masked_current_flux": ["features_set_a", "features_set_b", "hybrid_ridge"],
                    "ip_frac_unscaled": ["features_set_a", "features_set_b"]}
    # Version 1 in the follow-up is the default run, number for number.
    v1 = scores[(scores["variant"] == "version 1") & (scores["phase"] == "all")].set_index(["model", "subset"])["S"]
    default = pd.DataFrame(linear["scores"]).set_index(["model", "subset"])["S"]
    assert v1.sort_index().equals(default.sort_index())
    # Every phase with validation frames scores every arm, on subsets of the same shots.
    phases = {p: c for p, c in info["phase_frames"].items() if c["frames"]}
    assert phases and sum(c["frames"] for c in info["phase_frames"].values()) == info["n_val_frames"]
    for phase in phases:
        assert len(scores[scores["phase"] == phase]) == 8 and np.isfinite(scores.loc[scores["phase"] == phase, "S"]).all()
    assert np.isfinite(scores["S"]).all()
    # Masked target: the true remainder is added back, so the flux term stays high on every arm.
    masked = scores[(scores["variant"] == "masked_current_flux") & (scores["phase"] == "all") & (scores["subset"] == "all")]
    assert (masked["r2_psi"] > 0.9).all()
    assert set(info["spaces"]["masked_current_flux"]["share_without_boundary"]) == {"train", "validation"}

    coef = info["coefficient_r2"]
    for variant in info["variants"]:
        assert {"features_set_a", "features_set_b without coil_terms", "raw inputs of hybrid_ridge"} <= set(coef[variant])
        assert all(set(r["validation_by_phase"]) == {"rising", "flat_top", "late"} for r in coef[variant].values())
        shares = info["variance_shares"][variant]["train"]
        assert sum(v for v in shares.values() if np.isfinite(v)) == pytest.approx(1.0)
    # ip_frac is only in shot_position: without that group, and for the raw inputs, the inputs are
    # the version-1 inputs and the coefficient R2 is the same.
    for name in ("features_set_a without shot_position", "features_set_b without shot_position", "raw inputs of hybrid_ridge"):
        assert coef["ip_frac_unscaled"][name]["validation"] == coef["version 1"][name]["validation"]
    assert coef["version 1"]["features_set_a"]["validation"] == pytest.approx(linear["coefficient_r2"]["features_set_a"]["validation"])
    report = (out / "linear_check_followup.md").read_text()
    assert "## Scores, phase" in report and "## Coefficient R2, masked_current_flux" in report and "/Users/" not in report


def test_followup_stops_when_a_cache_entry_is_missing(ws, tmp_path):
    empty = tmp_path / "empty"
    result = run_linear_check(ws, "--ip-frac-unscaled", "--response-cache", str(tmp_path / "response"), cache=empty)
    assert result.returncode != 0 and "default run" in result.stderr
    assert not empty.exists()


def test_masked_split_rebuilds_the_true_maps(ws, monkeypatch):
    spec = importlib.util.spec_from_file_location("feature_linear_check", REPO_ROOT / "scripts" / "feature_linear_check.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)          # dataclasses resolve their module by name
    spec.loader.exec_module(module)
    table = GreenTable.load(ws["green"] / "green_d3d.npz")
    scales = adopted_scales("DIII-D", table.columns)
    shot = sorted(json.loads(Path(ws["split"]).read_text())["train"])[0]
    rows = bl.frames_of_shots([shot], bl.input_columns(), ws["table_dir"])
    maps = ft.read_flux_maps([shot], "diii_d_train", ws["data"])
    packed = ct.shot_masks(shot, str(ws["data"]), "diii_d_train", 1, table.grid_r, table.grid_z)
    response = ct.PlasmaResponse.build("DIII-D", table.grid_r, table.grid_z)
    inside, remainder = module.split_masked(table, rows, maps, packed, response, scales)
    ip = rows["ip__mean"].to_numpy(dtype=np.float64)
    coil = targets.coil_flux_of_frames(table, rows, scales, "DIII-D")
    rebuilt = targets.stored_from_target(inside + remainder, coil, ip, "DIII-D")
    assert np.abs(rebuilt - maps).max() < 1e-6 * np.ptp(maps)
    masks = ct.unpack_masks(packed, maps.shape[1:])
    assert masks.any() and np.abs(inside[masks.any(axis=(1, 2))]).max() > 0
    with pytest.raises(ValueError, match="rows"):
        module.split_masked(table, rows.iloc[:-1], maps, packed, response, scales)


def run_backup_radii(ws: dict, *extra: str) -> subprocess.CompletedProcess:
    cmd = [sys.executable, str(REPO_ROOT / "scripts" / "backup_radii_check.py"), "--split", str(ws["split"]),
           "--feature-root", str(ws["features"]), "--report-root", str(ws["reports"]), "--table-dir", str(ws["table_dir"]),
           "--green-dir", str(ws["green"]), "--data-root", str(ws["data"]), "--nn-train-frames", "100",
           "--nn-eval-frames", "10", *extra]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=300)


def test_backup_radii_check_builds_mast_apart_and_leaves_version_1_unchanged(ws, profiled):
    v1_dir = ws["features"] / "v1"
    before = {p.name: p.read_bytes() for p in v1_dir.iterdir()}
    scaler = (ws["reports"] / "v1" / "scaler.json").read_bytes()
    points_before = sorted(p.name for p in ws["green"].glob("points_mast_*.npz"))
    result = run_backup_radii(ws, "--r0", "0.70", "--a0", "0.50")
    assert result.returncode == 0, result.stderr
    assert {p.name: p.read_bytes() for p in v1_dir.iterdir()} == before
    assert (ws["reports"] / "v1" / "scaler.json").read_bytes() == scaler
    points_after = sorted(p.name for p in ws["green"].glob("points_mast_*.npz"))
    assert len(points_after) == len(points_before) + 1 and set(points_before) <= set(points_after)

    scratch = ws["features"] / "scratch" / "mast_r0.70_a0.50"
    record = json.loads((scratch / "build.json").read_text())
    assert record["radii"] == {"r0_m": 0.7, "a0_m": 0.5, "source": "Morris 1998"} and record["build"]["x1_rows_match"]
    backup = pd.read_parquet(scratch / "mast_public_test.parquet")
    v1 = fe.read_features("mast_public_test", 1, None, None, ws["features"])
    assert backup[["shot", "frame"]].equals(v1[["shot", "frame"]]) and list(backup.columns) == list(v1.columns)
    # Features without the radii are unchanged; the others scale exactly with them.
    for column in ("ip_frac", "ts_pe_peaking", "ts_valid"):
        assert np.array_equal(backup[column].to_numpy(), v1[column].to_numpy(), equal_nan=True)
    r0, a0, r1, a1 = 0.85, 0.65, 0.70, 0.50
    for column, factor in (("q_cyl", (a1 / r1) ** 2 / (a0 / r0) ** 2), ("ts_beta_p", (a1 / a0) ** 2),
                           ("greenwald_frac", (a1 / a0) ** 2)):
        ok = np.isfinite(v1[column].to_numpy()) & (v1[column].to_numpy() != 0)
        assert ok.any() and np.allclose(backup[column].to_numpy()[ok], factor * v1[column].to_numpy()[ok], rtol=1e-5)
    assert not np.allclose(backup["coil_c20"].to_numpy(), v1["coil_c20"].to_numpy())

    report = (ws["reports"] / "v1" / "backup_radii_check.md").read_text()
    for heading in ("## 1. Layout clearance", "## 3. Ranges", "### By phase", "## 4. Stretch terms", "## 5. Thomson",
                    "## 6. Combined distance"):
        assert heading in report
    assert "Status **pass**" in report and "MAST, backup radii" in report and "/Users/" not in report
    assert report.count("| set_b, shaping for coil_c20 and coil_c02, ip_frac unscaled |") == 2 * 4


def test_backup_radii_check_stops_on_a_failing_clearance_and_refuses_the_version_folder(ws, profiled, tmp_path):
    result = run_backup_radii(ws, "--r0", "0.70", "--a0", "3.0", "--scratch-dir", str(tmp_path / "scratch"))
    assert result.returncode != 0 and "clearance fails" in result.stderr
    assert not (tmp_path / "scratch").exists()
    assert "Stopped" in (ws["reports"] / "v1" / "backup_radii_check.md").read_text()
    inside = run_backup_radii(ws, "--scratch-dir", str(ws["features"] / "v1" / "backup"))
    assert inside.returncode != 0 and "inside" in inside.stderr and not (ws["features"] / "v1" / "backup").exists()
