"""Tests of ``scripts/worst_mast_shots.py`` on synthetic inputs: the scorer-report parser, the
key-to-shot mapping, the per-shot and group statistics, and the script end to end."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fusion_eq import starter_kit
from tests.conftest import drop_labels, make_mast_shot, write_shot
from tests.test_build_features import REPO_ROOT, _far_coils_mast, make_workspace
from tests.test_target_build import SMALL_CONFIG

SCRIPT = REPO_ROOT / "scripts" / "worst_mast_shots.py"


def load():
    spec = importlib.util.spec_from_file_location("worst_mast_shots", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


w = load()


def page(mast_rows, d3d_rows=(("shot_0001", 0.9, 0.1),)) -> str:
    """A detailed-results page shaped like the scorer's: h2 per machine, h4 above each table."""
    def table(rows):
        body = "".join(f"<tr><td>{k}</td><td>{r}</td><td>{d}</td><td></td></tr>" for k, r, d in rows)
        return ("<table border=1><tr><th>submission key</th><th>R2_psi</th><th>D_LCFS</th><th>missing</th></tr>"
                f"{body}</table>")
    return ("<html><h1>Fusion Equilibrium Challenge &mdash; detailed results</h1><h2>Leaderboard summary</h2>"
            "<table><tr><th>metric</th><th>value</th></tr><tr><td>MAST S_model</td><td>0.48</td></tr></table>"
            "<h2>DIII-D</h2><h4>Submitted scalars (value R&sup2; vs stored EFIT labels)</h4><table><tr><th>scalar</th></tr></table>"
            f"<h4>Worst 20 shots (lowest R&sup2;_psi)</h4>{table(d3d_rows)}"
            "<h2>MAST</h2><h4>Submitted scalars (value R&sup2; vs stored EFIT labels)</h4><table><tr><th>scalar</th></tr></table>"
            f"<h4>Worst 20 shots (lowest R&sup2;_psi)</h4>{table(mast_rows)}</html>")


def record(shots) -> dict:
    return {"folds": {"MAST": {"n_shots": len(shots), "shot_checks": [{"index": i, "shot": s} for i, s in enumerate(shots)]}},
            "predictor_identifiers": {"target_build": {"config_hash": "f" * 64}}}


def test_the_worst_table_of_the_mast_section_is_read():
    t = w.worst_table(page([("shot_1170", -0.0447, 0.2124), ("shot_0060", 0.2086, 0.6731)]))
    assert t["key"].tolist() == ["shot_1170", "shot_0060"] and t["index"].tolist() == [1170, 60]
    assert t["r2_psi"].tolist() == [-0.0447, 0.2086] and t["d_lcfs"].tolist() == [0.2124, 0.6731]
    d3d = w.worst_table(page([("shot_0002", 0.1, 0.2)]), "DIII-D")
    assert d3d["key"].tolist() == ["shot_0001"]
    with pytest.raises(ValueError, match="no 'Worst' table"):
        w.worst_table("<h2>DIII-D</h2><h4>Worst 20</h4><table><tr><th>submission key</th></tr></table>")
    with pytest.raises(ValueError, match="does not read"):
        w.worst_table(page([("mast_7", 0.1, 0.2)]))


def test_keys_map_to_shots_through_the_recorded_order():
    order = w.shot_order(record(["s0", "s1", "s2", "s3"]))
    assert order == ["s0", "s1", "s2", "s3"]
    worst = pd.DataFrame({"key": ["shot_0003", "shot_0000"], "index": [3, 0], "r2_psi": [0.1, 0.2], "d_lcfs": [0.21, 0.68]})
    mapped = w.assign_groups(w.map_keys(worst, order), 0.45)
    assert mapped["shot"].tolist() == ["s3", "s0"] and mapped["group"].tolist() == ["low", "high"]
    with pytest.raises(ValueError, match="outside"):
        w.map_keys(worst.assign(index=[4, 0]), order)
    gap = record(["s0", "s1"])
    gap["folds"]["MAST"]["shot_checks"][1]["index"] = 2
    with pytest.raises(ValueError, match="not 0 to 1"):
        w.shot_order(gap)
    short = record(["s0", "s1"])
    short["folds"]["MAST"]["n_shots"] = 3
    with pytest.raises(ValueError, match="2 shot_checks for 3 shots"):
        w.shot_order(short)


def synthetic_rows() -> pd.DataFrame:
    rows = []
    for shot, n, ip_peak, rise in (("a", 4, 800.0, 2.0), ("b", 6, 500.0, 3.5)):
        for f in range(n):
            frac = [0.3, 1.0, 0.95, 0.2, 1.0, 0.5][f]
            rows.append({"shot": shot, "frame": f, "ip__mean": frac * ip_peak, "ip_frac": frac, "ip_rise_ms": rise,
                         **{c: float(f) + (10.0 if shot == "b" else 0.0) for c in w.FEATURE_COLUMNS}})
    t = pd.DataFrame(rows)
    t["n_frames"] = t.groupby("shot")["frame"].transform("size")
    t["phase"] = np.where(t["ip_frac"] >= 0.9, "flat_top", np.where(t["frame"] < t["n_frames"] / 2, "rising", "late"))
    return t


def test_shot_statistics_and_the_group_table():
    rows = synthetic_rows()
    stats = w.shot_stats(rows).set_index("shot")
    assert stats.loc["a", "frames"] == 4 and stats.loc["b", "frames"] == 6
    assert stats.loc["a", "ip_peak_ka"] == pytest.approx(800.0) and stats.loc["a", "ip_median_ka"] == pytest.approx(0.625 * 800.0)
    # a: frames 0 rising, 1 and 2 flat-top, 3 late.
    assert (stats.loc["a", "share_rising"], stats.loc["a", "share_flat_top"], stats.loc["a", "share_late"]) == (0.25, 0.5, 0.25)
    assert stats.loc["b", "ip_rise_ms"] == 3.5 and stats.loc["b", "coil_c10"] == pytest.approx(12.5)
    boundary = {"low": {"n_frames": 2, "boundary_share": 0.5,
                        "spread": {k: {"n": 1, "q5": 1.0, "q50": 1.0, "q95": 1.0} for k in w.BOUNDARY_KEYS}}}
    table = {r["quantity"]: r for r in w.group_rows(rows, stats.reset_index(), {"low": ["a"], "rest": ["b"]}, boundary)}
    assert table["Shots"] == {"quantity": "Shots", "low": "1", "rest": "1"}
    assert table["Share of frames, flat_top"]["low"] == "0.500" and table["Share of frames, late"]["rest"] == "0.333"
    assert table["coil_c10"]["rest"].startswith("12.500 (")
    assert table["Boundary found (sampled frames)"] == {"quantity": "Boundary found (sampled frames)", "low": "0.500 of 2", "rest": "n/a"}
    assert table["Mean-current boundary: Width (m)"]["low"] == "1.000 (1.000 to 1.000)"


def test_spaced_frames():
    assert w.spaced_frames(10, 5).tolist() == [0, 2, 4, 7, 9]
    assert w.spaced_frames(3, 5).tolist() == [0, 1, 2] and w.spaced_frames(0, 5).size == 0
    assert w.spaced_frames(7, 1).tolist() == [0]


# ------------------------------------------------------------------------- end to end
@pytest.fixture(scope="module")
def ws(tmp_path_factory) -> dict:
    if not starter_kit.is_available():
        pytest.skip("starter kit not available")
    root = tmp_path_factory.mktemp("worst")
    ws = make_workspace(root)
    for name, n in (("m2", 6), ("m3", 10)):
        write_shot(ws["data"] / "mast_public_test" / f"mast_shot_{name}.parquet", drop_labels(_far_coils_mast(make_mast_shot(n_frames=n))))
    ws["table_dir"] = root / "frame_table_3mast"
    common = ["--data-root", str(ws["data"]), "--table-dir", str(ws["table_dir"]), "--green-dir", str(ws["green"]),
              "--split", str(ws["split"])]
    steps = [
        [str(REPO_ROOT / "scripts" / "build_frame_table.py"), "--data-root", str(ws["data"]), "--out", str(ws["table_dir"]), "--workers", "1"],
        [str(REPO_ROOT / "scripts" / "build_features.py"), "--config", str(REPO_ROOT / "configs" / "features_v2.yaml"),
         "--out", str(root / "features"), "--report-root", str(root / "feature_reports"), *common],
    ]
    config = root / "target.yaml"
    config.write_text(SMALL_CONFIG)
    steps.append([str(REPO_ROOT / "scripts" / "build_targets.py"), "--config", str(config), "--out", str(root / "targets"),
                  "--response-cache", str(root / "response"), "--references-dir", str(root / "refs"), "--workers", "1", *common])
    for step in steps:
        result = subprocess.run([sys.executable, *step], capture_output=True, text=True, timeout=600)
        assert result.returncode == 0, result.stderr
    return {**ws, "features": root / "features", "targets": root / "targets", "config": config, "root": root}


def test_the_script_maps_groups_and_writes_the_report(ws, tmp_path):
    run = "2026-10-14_synthetic"
    reports, subs = tmp_path / "reports", tmp_path / "submissions"
    (reports / run).mkdir(parents=True)
    (reports / run / "run.json").write_text(json.dumps(record(["m1", "m2", "m3"])))
    detailed = subs / run / "codabench_scoring_result" / "detailed_results.html"
    detailed.parent.mkdir(parents=True)
    detailed.write_text(page([("shot_0002", -0.04, 0.21), ("shot_0000", 0.2, 0.67)], d3d_rows=(("shot_0001", 0.9, 0.1),)))
    cmd = [sys.executable, str(SCRIPT), "--run", run, "--reports-dir", str(reports), "--submissions-dir", str(subs),
           "--table-dir", str(ws["table_dir"]), "--feature-root", str(ws["features"]), "--targets", str(ws["targets"]),
           "--target-config", str(ws["config"]), "--green-dir", str(ws["green"]), "--frames-per-shot", "3", "--rest-shots", "5"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stdout + result.stderr
    text = (reports / run / "worst_mast_shots.md").read_text()
    assert "No equilibrium label is read" in text and "/Users/" not in text
    shots = {line.split(" | ")[0].lstrip("| "): line.split(" | ") for line in text.split("## Worst 20 shots")[1].split("Feature medians")[0].splitlines()
             if line.startswith("| shot_")}
    assert shots["shot_0002"][1] == "`m3`" and shots["shot_0002"][2].startswith("D_LCFS below 0.45 m (median 0.21 m)")
    assert shots["shot_0000"][1] == "`m1`" and shots["shot_0000"][2].startswith("D_LCFS at or above 0.45 m")
    assert shots["shot_0002"][5] == "10" and shots["shot_0000"][5] == "8"      # frames of m3 and m1
    groups = text.split("## Groups against the rest")[1]
    assert "| Shots | 1 | 1 | 1 |" in groups and "Rest of the split" in groups
    # Three sampled frames per shot, one shot in each group.
    assert "| Boundary found (sampled frames) |" in groups and groups.count(" of 3") == 3
    # The DIII-D table's key (shot_0001, m2) is not among the worst MAST shots.
    assert "| shot_0001 |" not in text and "`m2`" not in text
