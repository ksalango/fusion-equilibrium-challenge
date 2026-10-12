"""Tests for the frame table schema, build, guards, readers and the build script."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from fusion_eq import dataset, frame_table as ft, io, ip_alignment, profile
from fusion_eq.machines import DIII_D, MAST
from tests.conftest import D3D_MAGNETICS_ORIGIN_MS, drop_labels, make_d3d_shot, make_mast_shot, write_shot

REPO_ROOT = Path(__file__).resolve().parents[1]
PARAMS = ft.BuildParams()


def _load(path: Path, with_labels: bool, align_ip: bool = True) -> io.Shot:
    cfg, _ = io.parse_shot_filename(path)
    return io.load_shot(path, ft.columns_to_load(cfg, with_labels, True), align_ip=align_ip)


def _retimed_d3d(step_ms: float, **kwargs) -> dict:
    """Synthetic DIII-D shot with magnetics re-sampled onto a finer time step."""
    shot = make_d3d_shot(**kwargs)
    old = shot["magnetics_time"].astype(np.float64)
    new = D3D_MAGNETICS_ORIGIN_MS + step_ms * np.arange(int(round((old[-1] - old[0]) / step_ms)) + 1)
    for col in DIII_D.current_columns:
        if col != DIII_D.ip_column and col in shot:
            shot[col] = np.interp(new, old, np.asarray(shot[col], dtype=np.float64))
    shot["magnetics_time"] = new.astype(np.float32)
    return shot


# ------------------------------------------------------------------------------ schema
def test_schema_roles_and_input_columns_exclude_labels():
    for machine in (DIII_D, MAST):
        train = ft.table_schema(machine, with_labels=True)
        test = ft.table_schema(machine, with_labels=False)
        assert all(c.role in ft.ROLES for c in train)
        assert [c.name for c in test] == [c.name for c in train if c.role != "label"]
        inputs = ft.input_columns(machine)
        assert inputs and not any(ft.is_label_column(c) for c in inputs)
        assert "ip__mean" in inputs and "ts_core_pe_mean__interp" in inputs
        assert "mag_rate_hz" not in inputs and "mag_rate_hz" in ft.input_columns(machine, include_flags=True)
    d3d_names = {c.name for c in ft.table_schema(DIII_D, False)}
    mast_names = {c.name for c in ft.table_schema(MAST, False)}
    assert "magnetics_F1A__mean" in d3d_names and "magnetics_p2l_current__mean" in mast_names
    shared = {n for n in d3d_names if not n.startswith("magnetics_")}
    assert shared == {n for n in mast_names if not n.startswith("magnetics_")}


def test_label_guard_raises():
    assert ft.is_label_column("efit_q95") and ft.is_label_column("magnetics_dsep") and ft.is_label_column("psi_min")
    assert not ft.is_label_column("efit_times") and not ft.is_label_column("ip__mean")
    assert ft.assert_input_columns(["ip__mean", "sol__rate"], "DIII-D") == ["ip__mean", "sol__rate"]
    with pytest.raises(ValueError, match="label"):
        ft.assert_input_columns(["ip__mean", "efit_q95"], "DIII-D")
    with pytest.raises(ValueError, match="label"):
        ft.assert_input_columns(["psi_mean"], "MAST")
    with pytest.raises(ValueError, match="not input"):
        ft.assert_input_columns(["magnetics_F1A__mean"], "MAST")


# --------------------------------------------------------------------- shot-level rules
def test_current_direction():
    assert ft.current_direction([0.0, 1.0, np.nan, 3.0]) == 1
    assert ft.current_direction([-2.0, 0.0]) == -1
    assert ft.current_direction([-1.0, 1.0]) == 0
    assert ft.current_direction([np.nan]) == 0


def test_recording_rate_from_endpoints_survives_float32_quantisation():
    t64 = -4010.75 + 0.05 * np.arange(480_001)                # 20 kHz over 24 s
    t32 = t64.astype(np.float32)
    assert ft.recording_rate_hz(t32) == 20000
    assert np.median(np.diff(t32.astype(np.float64))) != pytest.approx(0.05, rel=1e-6)
    assert ft.recording_rate_hz(np.arange(0.0, 1000.0, 0.5)) == 2000
    assert ft.recording_rate_hz(np.array([np.nan, 1.0, 1.2, np.nan, 1.4])) == 5000
    assert ft.recording_rate_hz([1.0]) == 0 and ft.recording_rate_hz([1.0, 1.0]) == 0


def test_shot_flags_and_ip_fix_kind(tmp_path, starter):
    path = write_shot(tmp_path / "d3d_shot_1.parquet", make_d3d_shot())
    flags = ft.shot_flags(_load(path, True))
    assert flags["mag_rate_hz"] == 500 and flags["ip_direction"] == 1 and flags["tf_direction"] == -1
    assert flags["ip_fix"] == "organizers" and flags["ip_shift_ms"] == pytest.approx(-3152.56, abs=0.01)
    assert flags["mag_has_nan"] is False
    untouched = write_shot(tmp_path / "d3d_shot_2.parquet", make_d3d_shot(ip_fix_needed=False, breakdown_ms=200.0))
    shot = _load(untouched, True)
    flags = ft.shot_flags(shot)
    assert flags["ip_fix"] == "alignment"
    assert flags["ip_shift_ms"] == pytest.approx(shot.fixes["ip_alignment_shift_ms"])
    assert ft.shot_flags(_load(untouched, True, align_ip=False))["ip_fix"] == "none"
    mast = make_mast_shot()
    mast["magnetics_sol_current"][::6] = np.nan
    mast["magnetics_plasma_current"] = -mast["magnetics_plasma_current"]
    flags = ft.shot_flags(_load(write_shot(tmp_path / "mast_shot_3.parquet", mast), True))
    assert flags == {"mag_rate_hz": 500, "mag_has_nan": True, "ip_direction": -1, "tf_direction": -1,
                     "ip_fix": "none", "ip_shift_ms": 0.0}


# ------------------------------------------------------------------------------- build
def test_build_copies_labels_exactly_and_orders_columns(tmp_path, starter):
    raw = make_d3d_shot(n_frames=7)
    raw["efit_q95"] = raw["efit_q95"].astype(np.float32) + np.float32(0.1234567)
    path = write_shot(tmp_path / "d3d_shot_1.parquet", raw)
    shot = _load(path, True)
    frames = ft.build_shot_frames(shot, "diii_d_train", PARAMS, with_labels=True)
    schema = ft.table_schema(DIII_D, True)
    assert list(frames.columns) == [c.name for c in schema]
    assert len(frames) == 7 and frames["frame"].tolist() == list(range(7))
    assert frames["time_ms"].to_numpy().tolist() == raw["efit_times"].tolist()
    assert np.isnan(frames["dt_prev_ms"].iloc[0]) and frames["dt_prev_ms"].iloc[1] == 20.0
    for name in ft.COPIED_LABELS:
        assert np.array_equal(frames[name].to_numpy(), np.asarray(raw[name], dtype=np.float64))
    psi = raw["efit_psirz"]
    assert np.allclose(frames["psi_min"], psi.min(axis=(1, 2)), rtol=1e-6)
    assert np.allclose(frames["psi_mean"], psi.mean(axis=(1, 2)), rtol=1e-6)
    assert frames["ip__mean"].dtype == np.float32 and frames["ip_direction"].dtype == np.int8
    assert frames["inputs_complete"].all() and frames["ts_core_valid"].all()
    assert (frames["ip_over_peak"] > 0.9).all()


def test_build_without_psi_summary_and_without_labels(tmp_path, starter):
    path = write_shot(tmp_path / "d3d_shot_1.parquet", make_d3d_shot())
    frames = ft.build_shot_frames(_load(path, True), "diii_d_train", ft.BuildParams(psi_summary=False), True)
    assert frames["psi_min"].isna().all() and frames["efit_q95"].notna().all()
    test_path = write_shot(tmp_path / "d3d_shot_2.parquet", drop_labels(make_d3d_shot()))
    frames = ft.build_shot_frames(_load(test_path, False), "diii_d_public_test", PARAMS, with_labels=False)
    assert not any(ft.is_label_column(c) for c in frames.columns)
    with pytest.raises(KeyError):
        ft.build_shot_frames(_load(test_path, False), "diii_d_train", PARAMS, with_labels=True)


def test_build_same_signal_at_two_rates_gives_same_means(tmp_path, starter):
    fast = write_shot(tmp_path / "d3d_shot_fast.parquet", _retimed_d3d(0.05))
    slow = write_shot(tmp_path / "d3d_shot_slow.parquet", _retimed_d3d(0.5))
    f = ft.build_shot_frames(_load(fast, True), "diii_d_train", PARAMS, True)
    s = ft.build_shot_frames(_load(slow, True), "diii_d_train", PARAMS, True)
    assert f["mag_rate_hz"].iloc[0] == 20000 and s["mag_rate_hz"].iloc[0] == 2000
    for col in ("sol__mean", "magnetics_F3B__mean", "sol__rate"):
        assert np.allclose(f[col], s[col], atol=1e-3)


def test_build_handles_missing_thomson_and_nan_holes(tmp_path, starter):
    raw = make_mast_shot()
    raw["thomson_core_Te"][:] = np.nan                                # no valid core slice
    del raw["thomson_edge_times"], raw["thomson_edge_Te"], raw["thomson_edge_ne"]
    raw["magnetics_tf_current"][1::6] = np.nan
    raw["magnetics_plasma_current"][::6] = np.nan
    path = write_shot(tmp_path / "mast_shot_1.parquet", raw)
    frames = ft.build_shot_frames(_load(path, False), "mast_public_test", PARAMS, False)
    assert frames["ts_core_valid"].eq(False).all() and frames["ts_core_dt_ms"].isna().all()
    assert frames["ts_edge_valid"].eq(False).all() and frames["ts_edge_te_mean__interp"].isna().all()
    assert frames["mag_has_nan"].all()
    assert frames["ip__mean"].notna().all() and frames["tf__mean"].notna().all()
    assert np.allclose(frames["ip__mean"], 800.0) and np.allclose(frames["tf__nearest"], -85.0)


def test_build_one_records_errors_and_continues(tmp_path, starter):
    bad = make_d3d_shot()
    bad["efit_times"] = bad["efit_times"][::-1]
    bad_path = write_shot(tmp_path / "d3d_shot_bad.parquet", bad)
    frames, row = ft.build_one(bad_path, "diii_d_train", PARAMS)
    assert frames is None and "ValueError" in row["error"] and row["shot"] == "bad"
    good_path = write_shot(tmp_path / "d3d_shot_good.parquet", make_d3d_shot())
    frames, row = ft.build_one(good_path, "diii_d_train", PARAMS)
    assert row["error"] is None and row["n_frames"] == 12 and row["ip_fix"] == "organizers"
    index = ft.shot_index_frame([row, ft.build_one(bad_path, "diii_d_train", PARAMS)[1]])
    assert index["shot"].tolist() == ["bad", "good"] and index["error"].notna().tolist() == [True, False]
    assert str(index["n_frames"].dtype) == "Int64"


def test_build_one_passes_align_ip_to_loader(tmp_path, starter):
    path = write_shot(tmp_path / "d3d_shot_1.parquet", make_d3d_shot(ip_fix_needed=False, breakdown_ms=200.0))
    _, aligned = ft.build_one(path, "diii_d_train", PARAMS)
    _, raw = ft.build_one(path, "diii_d_train", ft.BuildParams(align_ip=False))
    assert aligned["ip_fix"] == "alignment" and raw["ip_fix"] == "none" and raw["ip_shift_ms"] == 0.0


# ----------------------------------------------------------------- script, readers, checks
@pytest.fixture
def data_root(tmp_path) -> Path:
    root = tmp_path / "data"
    (root / "diii_d_train").mkdir(parents=True)
    (root / "diii_d_public_test").mkdir()
    (root / "mast_public_test").mkdir()
    for shot_id, kwargs in (("a1", {}), ("a2", {"n_frames": 5}), ("a3", {"ip_fix_needed": False})):
        write_shot(root / "diii_d_train" / f"d3d_shot_{shot_id}.parquet", make_d3d_shot(**kwargs))
    write_shot(root / "diii_d_public_test" / "d3d_shot_t1.parquet", drop_labels(make_d3d_shot(n_frames=4)))
    mast = make_mast_shot()
    write_shot(root / "mast_public_test" / "mast_shot_m1.parquet", drop_labels(mast))
    mast["magnetics_sol_current"][::6] = np.nan
    write_shot(root / "mast_public_test" / "mast_shot_m2.parquet", drop_labels(mast))
    return root


def _run_build(data_root: Path, out: Path, *extra: str) -> subprocess.CompletedProcess:
    cmd = [sys.executable, str(REPO_ROOT / "scripts" / "build_frame_table.py"), "--data-root", str(data_root),
           "--out", str(out), "--workers", "2", "--shots-per-group", "2", *extra]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=300)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_build_script_end_to_end_is_deterministic(data_root, tmp_path, starter):
    out1, out2 = tmp_path / "out1", tmp_path / "out2"
    for out in (out1, out2):
        result = _run_build(data_root, out)
        assert result.returncode == 0, result.stderr
    for split in dataset.DATA_SPLITS:
        assert _sha256(out1 / f"{split}.parquet") == _sha256(out2 / f"{split}.parquet")
    assert _sha256(out1 / "shot_index.parquet") == _sha256(out2 / "shot_index.parquet")

    index = pd.read_parquet(out1 / "shot_index.parquet")
    assert index["shot"].tolist() == ["t1", "a1", "a2", "a3", "m1", "m2"]
    assert index["error"].isna().all()
    assert index.set_index("shot")["mag_has_nan"].to_dict() == {"t1": False, "a1": False, "a2": False, "a3": False,
                                                                 "m1": False, "m2": True}
    train = pq.ParquetFile(out1 / "diii_d_train.parquet")
    assert train.metadata.num_rows == 12 + 5 + 12 and train.metadata.num_row_groups == 2
    assert "efit_q95" in train.schema_arrow.names
    assert "efit_q95" not in pq.ParquetFile(out1 / "diii_d_public_test.parquet").schema_arrow.names
    info = json.loads((out1 / "build.json").read_text())
    assert info["params"]["align_ip"] is True and info["rows_per_split"]["mast_public_test"] == 16
    assert info["n_errors"] == 0

    # Readers.
    rows = ft.read_frame_table("diii_d_train", columns=["frame", "efit_q95"], shots=["a2"], root=out1)
    assert len(rows) == 5 and list(rows.columns) == ["frame", "efit_q95"]
    everything = ft.read_frame_table("diii_d_train", root=out1)
    assert everything["shot"].tolist() == ["a1"] * 12 + ["a2"] * 5 + ["a3"] * 12
    maps = ft.read_flux_maps(["a3", "a2"], "diii_d_train", data_root)
    assert maps.shape == (17, 65, 65) and maps.dtype == np.float32
    assert np.allclose(maps.min(axis=(1, 2)),
                       ft.read_frame_table("diii_d_train", ["psi_min"], ["a2", "a3"], out1)["psi_min"], rtol=1e-6)

    # Acceptance checks and report on the smoke build.
    results = profile.run_checks(out1, data_root, qa_path=None)
    status = {r.check_id: r.status for r in results}
    assert status["T1"] == "pass" and status["T2"] == "pass" and status["T3"] == "pass" and status["T6"] == "pass"
    assert status["T5"] == "review"                                     # partial run, no QA table
    prof = pd.concat([profile.profile_table(out1 / f"{s}.parquet", s, profile.schema_roles(dataset.SPLIT_MACHINE[s],
                                                                                           s == "diii_d_train"))
                      for s in dataset.DATA_SPLITS], ignore_index=True)
    assert set(prof["role"]) == {"key", "input", "flag", "label"}
    report = profile.render_report(results, prof, info)
    assert "| T1 | Row and shot counts | pass |" in report and "## Missing values per column" in report


def test_build_script_requires_out_for_no_align_ip(data_root, tmp_path):
    cmd = [sys.executable, str(REPO_ROOT / "scripts" / "build_frame_table.py"), "--data-root", str(data_root),
           "--no-align-ip"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    assert result.returncode != 0 and "--no-align-ip requires an explicit --out" in result.stderr


def test_build_script_records_a_failing_shot_and_align_flag(data_root, tmp_path, starter):
    bad = make_d3d_shot()
    bad["efit_times"] = np.array([5.0, 5.0, 1.0])
    bad["efit_psirz"] = bad["efit_psirz"][:3]
    for name in ft.COPIED_LABELS:
        bad[name] = bad[name][:3]
    write_shot(data_root / "diii_d_train" / "d3d_shot_a0.parquet", bad)
    out = tmp_path / "out"
    result = _run_build(data_root, out, "--splits", "diii_d_train", "--no-align-ip", "--no-psi-summary")
    assert result.returncode == 0, result.stderr
    index = pd.read_parquet(out / "shot_index.parquet")
    assert index.loc[index["shot"] == "a0", "error"].notna().all()
    assert index.loc[index["shot"] != "a0", "error"].isna().all()
    info = json.loads((out / "build.json").read_text())
    assert info["params"]["align_ip"] is False and info["params"]["psi_summary"] is False and info["n_errors"] == 1
    train = ft.read_frame_table("diii_d_train", ["shot", "psi_min", "ip_fix"], root=out)
    assert train["psi_min"].isna().all() and set(train["shot"]) == {"a1", "a2", "a3"}
    assert set(train["ip_fix"]) == {"organizers", "none"}


# --------------------------------------------------------------- Part 0: rates and Ip rise
def test_rate_windows_name_the_slope_columns():
    assert ft.rate_suffixes((10.0, 50.0)) == [("__rate", 10.0), ("__rate50", 50.0)]
    assert ft.BuildParams().rate_window_ms == (10.0, 50.0)
    assert ft.BuildParams(rate_window_ms=10.0).rate_window_ms == (10.0,)          # one number, one window
    assert ft.BuildParams(rate_window_ms=[10, 50]).to_dict()["rate_window_ms"] == [10.0, 50.0]
    for bad in ((10.0, 12.5), (10.0, 50.0, 50.0), (0.0,), ()):
        with pytest.raises(ValueError):
            ft.BuildParams(rate_window_ms=bad)


@pytest.mark.parametrize("cfg", [DIII_D, MAST])
def test_schema_has_long_rates_for_ip_solenoid_and_shaping_coils_only(cfg):
    names = [c.name for c in ft.table_schema(cfg, with_labels=False)]
    expected = {"ip__rate50", "sol__rate50", *(f"{c}__rate50" for c in cfg.shaping_coil_columns)}
    assert {n for n in names if n.endswith("__rate50")} == expected
    assert "tf__rate50" not in names and "tf__rate" in names
    inputs = ft.input_columns(cfg)
    assert {"ip_rise_ms", "sol_at_rise", "ip__rate50"} <= set(inputs)
    assert ft.assert_input_columns(["ip_rise_ms", "sol_at_rise"], cfg) == ["ip_rise_ms", "sol_at_rise"]
    # A build with one window has no long-rate columns.
    assert not any(n.endswith("__rate50") for n in (c.name for c in ft.table_schema(cfg, False, (10.0,))))


def test_build_long_rate_and_ip_rise_columns(tmp_path, starter):
    shot = make_d3d_shot()
    t = shot["magnetics_time"].astype(np.float64)
    shot["magnetics_ECOILA"] = 0.2 * t                                        # 200 kA/s everywhere
    noise = np.random.default_rng(0).normal(size=t.size)
    shot["magnetics_F1A"] = np.where((t > 0) & (t < 5000), 100.0, 0.0) + noise  # flat with 1 kA-turn noise
    path = write_shot(tmp_path / "d3d_shot_rate.parquet", shot)
    loaded = _load(path, True)
    frames = ft.build_shot_frames(loaded, "diii_d_train", PARAMS, True)
    assert np.allclose(frames["sol__rate"], 200.0, rtol=1e-5) and np.allclose(frames["sol__rate50"], 200.0, rtol=1e-5)
    # The longer window averages the sample noise down: slope noise falls as window^-1.5.
    short, long = frames["magnetics_F1A__rate"].to_numpy(), frames["magnetics_F1A__rate50"].to_numpy()
    assert np.sqrt(np.mean(long ** 2)) < 0.3 * np.sqrt(np.mean(short ** 2)) and np.sqrt(np.mean(short ** 2)) > 10.0
    # One rise per shot, on the loaded Ip axis: the 10 percent crossing, the first 0.5 ms sample after 10 ms.
    rise = frames["ip_rise_ms"].to_numpy()
    expected = ip_alignment.breakdown_time_ms(loaded["magnetics_plasma_current_times"], loaded["magnetics_plasma_current"])
    assert np.all(rise == expected) and expected == pytest.approx(10.25, abs=1e-3)
    assert np.allclose(frames["sol_at_rise"], 0.2 * expected, atol=0.2 * 1.0)   # ramp value within the 2 ms window
    _, row = ft.build_one(path, "diii_d_train", PARAMS)
    assert row["ip_rise_ms"] == expected and row["sol_at_rise"] == pytest.approx(float(frames["sol_at_rise"].iloc[0]))
    index = ft.shot_index_frame([row])
    assert index["ip_rise_ms"].dtype == np.float64 and index["sol_at_rise"].dtype == np.float64


def test_ip_rise_on_mast_skips_the_merged_clock_holes(tmp_path, starter):
    clean = make_mast_shot()
    holed = make_mast_shot()
    for column in ("magnetics_plasma_current", "magnetics_sol_current"):
        holed[column] = np.asarray(holed[column], dtype=np.float64)
        holed[column][::6] = np.nan
    rises = []
    for name, shot in (("clean", clean), ("holed", holed)):
        path = write_shot(tmp_path / f"mast_shot_{name}.parquet", drop_labels(shot))
        frames = ft.build_shot_frames(_load(path, False), "mast_public_test", PARAMS, False)
        rises.append((float(frames["ip_rise_ms"].iloc[0]), float(frames["sol_at_rise"].iloc[0])))
    assert np.isfinite(rises[1][0]) and rises[1][0] - rises[0][0] in (0.0, 2.0)  # at most the one skipped sample
    assert rises[0][1] == -40.0 and rises[1][1] == -40.0
