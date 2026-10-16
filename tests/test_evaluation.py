"""Tests of the MAST feature folder in the evaluation (``evaluate.py --mast-feature-dir``) and of
the feature-build check X1 under a configuration that differs from the build in its sets only.
Synthetic files only."""

from __future__ import annotations

import importlib.util
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from fusion_eq import dataset, evaluation as ev, features as fe
from fusion_eq.dataset import REPO_ROOT


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_table(path, values) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({"shot": ["1", "1"], "frame": [0, 1], "shaping": values})
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), path)


def test_clip_rates_read_mast_from_the_named_folder(tmp_path):
    scaler = SimpleNamespace(transform=lambda table, machine, cols: table[list(cols)].to_numpy(dtype=float))
    write_table(fe.features_path("mast_public_test", 2, tmp_path), [0.0, 0.0])         # the unit-model build
    write_table(tmp_path / "v2_solhalf" / "mast_public_test.parquet", [20.0, 0.0])     # the sol_half build
    plain = ev.clip_rate_table(["shaping"], scaler, 10.0, tmp_path, 2, [])
    named = ev.clip_rate_table(["shaping"], scaler, 10.0, tmp_path, 2, [], {"MAST": tmp_path / "v2_solhalf"})
    assert plain["MAST test"]["rates"]["shaping"] == 0.0
    assert named["MAST test"]["rates"]["shaping"] == 0.5
    assert named["MAST test"]["feature_file"].endswith("v2_solhalf/mast_public_test.parquet")


def test_evaluate_passes_the_mast_feature_folder_to_the_chain(tmp_path, monkeypatch):
    script = load_script("evaluate")
    seen = {}

    def fake_evaluate(run_id, paths, *args, **kwargs):
        seen["paths"] = paths
        raise SystemExit(0)

    monkeypatch.setattr(script.ev, "evaluate", fake_evaluate)
    folder = tmp_path / "v2_solhalf"
    try:
        script.main(["--run", "r", "--mast-feature-dir", str(folder), "--run-root", str(tmp_path)])
    except SystemExit:
        pass
    assert seen["paths"].chain.feature_dirs == {"MAST": folder}
    assert seen["paths"].chain.feature_dir("MAST", 2) == folder
    assert seen["paths"].chain.feature_dir("DIII-D", 2) == fe.features_dir(2, seen["paths"].chain.feature_root)
    try:
        script.main(["--run", "r", "--run-root", str(tmp_path)])
    except SystemExit:
        pass
    assert seen["paths"].chain.feature_dirs is None


def test_x1_passes_when_the_configuration_differs_in_its_sets_only():
    profile = load_script("profile_features")
    config = fe.FeatureConfig.load(fe.DEFAULT_CONFIG.with_name("features_v2.yaml"))
    recorded = config.to_dict()
    recorded["sets"] = {k: v for k, v in recorded["sets"].items() if k in ("set_a", "set_b")}
    data = {s: pd.DataFrame({"x": np.zeros(3)}) for s in dataset.DATA_SPLITS}
    counts = {s: {"x1_rows_match": True} for s in dataset.DATA_SPLITS}
    build = {"counts": counts, "config_hash": "an older hash", "config": recorded}
    check, info = profile.check_x1(build, config, data)
    assert check.status == "pass" and "in its sets only" in check.summary
    assert info["config_hash_matches"] is False and info["build_inputs_match"] is True
    changed = json.loads(json.dumps(recorded))
    changed["min_abs_ip_ka"] = 30.0
    check, _ = profile.check_x1({**build, "config": changed}, config, data)
    assert check.status == "fail" and check.summary.endswith("changed since the build")
