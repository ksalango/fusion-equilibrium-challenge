"""End-to-end test of ``scripts/current_ceiling.py`` with the target-phase variants h to k, and of
the per-machine response tables, on a synthetic workspace (three DIII-D training shots)."""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest

from fusion_eq import current_basis as cb, starter_kit
from tests.test_build_features import REPO_ROOT, make_workspace

SMALL = ["--components", "2", "5", "--f-current-components", "5", "--remainder-components", "2", "3",
         "--ridge-components", "3", "--symmetric-components", "4", "6", "--symmetric-base", "4",
         "--h-remainder-components", "3"]


@pytest.fixture(scope="module")
def ceiling(tmp_path_factory) -> dict:
    if not starter_kit.is_available():
        pytest.skip("starter kit not available")
    root = tmp_path_factory.mktemp("ceiling")
    ws = make_workspace(root)
    out = root / "out"
    cmd = [sys.executable, str(REPO_ROOT / "scripts" / "current_ceiling.py"), "--full", "--split", str(ws["split"]),
           "--frame-stride", "1", "--workers", "1", "--data-root", str(ws["data"]), "--table-dir", str(ws["table_dir"]),
           "--green-dir", str(ws["green"]), "--cache-dir", str(root / "cache"), "--response-cache", str(root / "response"),
           "--out", str(out), *SMALL]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stderr
    return {"ws": ws, "root": root, "info": json.loads((out / "ceiling.json").read_text()),
            "report": (out / "ceiling.md").read_text()}


def test_variants_h_to_k_are_scored_beside_a_to_g(ceiling):
    scores = ceiling["info"]["scores"]
    expected = {"h_symmetric_4", "h_symmetric_6", "i_placed_mast_4", "j_rescaled_4", "k_ridge_10ms_4", "k_ridge_50ms_4",
                "f_remainder_pca_3", "g_remainder_ridge_3", "e_remainder_exact"}
    assert expected <= set(scores) and all(np.isfinite(scores[v]["S"]) for v in expected)
    # The bookkeeping of variant e is untouched.
    assert ceiling["info"]["remainder"]["e_max_error_over_range"] < 1e-12
    t = ceiling["info"]["target_phase"]
    sym = t["symmetric_basis"]
    assert sym["n_components"] == 6 and sym["n_frames"] > 0
    assert all(v["even"] + v["odd"] == int(k) for k, v in sym["parity_counts"].items())
    # A share inside the reference lattice keeps its total through placement on MAST and back.
    assert t["placement"]["r0_m"] == 0.85 and t["placement"]["retained"]["median"] == pytest.approx(1.0, abs=1e-9)
    # The frame table of the workspace carries the 50 ms rates: both ridges, 42 inputs each.
    assert set(t["ridges"]) == {"10ms", "50ms"}
    for window, r in t["ridges"].items():
        assert len(r["columns"]) == 42 and np.isfinite(r["val_coefficient_r2"])
    assert "tf__rate" in t["ridges"]["50ms"]["columns"] and "ip__rate50" in t["ridges"]["50ms"]["columns"]
    # The 10 ms ridge of variant k is the ridge of variant g.
    assert t["ridges"]["10ms"]["train_coefficient_r2"] == ceiling["info"]["remainder"]["ridge"]["train_coefficient_r2"]
    references = {r["variant"]: r["reference"] for r in t["comparisons"]}
    assert references == {"h_symmetric_4": "f_remainder_pca_3", "h_symmetric_6": "f_remainder_pca_3",
                          "i_placed_mast_4": "h_symmetric_4", "j_rescaled_4": "h_symmetric_4",
                          "k_ridge_10ms_4": "g_remainder_ridge_3", "k_ridge_50ms_4": "g_remainder_ridge_3"}
    assert "## Target-phase variants h to k" in ceiling["report"]


def test_response_tables_for_both_machines_are_cached(ceiling, tmp_path):
    for machine in ("DIII-D", "MAST"):
        memory = cb.machine_response(machine, ceiling["ws"]["green"], tmp_path / "none", write=False)
        assert not (tmp_path / "none").exists()
        built = cb.machine_response(machine, ceiling["ws"]["green"], tmp_path)
        cached = cb.machine_response(machine, ceiling["ws"]["green"], tmp_path)
        assert built.machine == machine and np.array_equal(built.green, cached.green) and np.array_equal(memory.green, built.green)
        assert cb.response_record(built)["grid_hash"] == built.grid_hash
    assert len(list(tmp_path.glob("response_*.npz"))) == 2
