"""Tests for the shared shot loader."""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

from fusion_eq import io, starter_kit
from fusion_eq.machines import machine_from_filename
from tests.conftest import D3D_IP_SHIPPED_ORIGIN_MS, D3D_MAGNETICS_ORIGIN_MS, make_d3d_shot, make_mast_shot, write_shot


def test_parse_shot_filename():
    cfg, shot_id = io.parse_shot_filename("some/dir/d3d_shot_00000a10ac.parquet")
    assert cfg.name == "DIII-D" and shot_id == "00000a10ac"
    cfg, shot_id = io.parse_shot_filename("mast_shot_28348.parquet")
    assert cfg.name == "MAST" and shot_id == "28348"
    with pytest.raises(ValueError):
        machine_from_filename("jet_shot_1")


def test_read_table_one_row_and_missing_columns_ignored(d3d_file):
    table = io.read_table(d3d_file, ["efit_times", "not_a_column"])
    assert table.num_rows == 1
    assert table.column_names == ["efit_times"]


def test_read_table_rejects_two_rows(tmp_path):
    import pyarrow.parquet as pq
    path = tmp_path / "d3d_shot_1.parquet"
    pq.write_table(pa.table({"efit_times": pa.array([[1.0], [2.0]])}), path)
    with pytest.raises(ValueError):
        io.read_table(path)


def test_to_ndarray_shapes(d3d_file):
    table = io.read_table(d3d_file)
    data = io.table_to_arrays(table)
    assert data["efit_psirz"].shape == (12, 65, 65)
    assert data["thomson_core_Te"].shape == (3600, 6)
    assert data["efit_times"].shape == (12,)
    assert data["source"] == "DIII-D"
    assert data["coil_name"].dtype == object and data["coil_name"][0] == "F1A"
    assert data["magnetics_time"].dtype == np.float32


def test_to_ndarray_ragged_becomes_list():
    cell = pa.array([[[1.0, 2.0], [3.0]]])[0]
    out = io.to_ndarray(cell)
    assert isinstance(out, list) and out[0].tolist() == [1.0, 2.0] and out[1].tolist() == [3.0]


def test_load_shot_applies_ip_fix_and_drops_dependencies(d3d_file, starter):
    shot = io.load_shot(d3d_file, ["magnetics_plasma_current", "efit_times"])
    assert set(shot.columns) == {"magnetics_plasma_current", "efit_times"}
    assert shot.fixes["ip_fix_applied"] is True
    expected_shift = D3D_MAGNETICS_ORIGIN_MS - D3D_IP_SHIPPED_ORIGIN_MS
    assert shot.fixes["ip_time_shift_ms"] == pytest.approx(expected_shift, abs=0.01)
    assert shot.machine.name == "DIII-D" and shot.shot_id == "100001"
    assert "efit_psirz" in shot.file_columns


def test_load_shot_keeps_correct_ip_axis(tmp_path, starter):
    path = write_shot(tmp_path / "d3d_shot_2.parquet", make_d3d_shot(ip_fix_needed=False))
    shot = io.load_shot(path, ["magnetics_plasma_current_times"])
    assert shot.fixes["ip_fix_applied"] is False
    assert shot.fixes["ip_time_shift_ms"] == 0.0


def test_load_shot_without_fixes_returns_shipped_axis(d3d_file, starter):
    shot = io.load_shot(d3d_file, ["magnetics_plasma_current_times"], apply=False)
    assert shot.fixes == {}
    assert shot["magnetics_plasma_current_times"][0] == pytest.approx(D3D_IP_SHIPPED_ORIGIN_MS, abs=0.01)


def test_load_shot_aligns_mast_thomson(mast_file, starter):
    shot = io.load_shot(mast_file, ["thomson_core_Te"])
    assert shot["thomson_core_Te"].shape[1] == 130
    assert shot.fixes["thomson_core_fix_applied"] is True
    assert shot.fixes["thomson_core_channels_before"] == 131
    assert shot.fixes["thomson_core_dropped_nan_frac_te"] == 1.0
    assert set(shot.columns) == {"thomson_core_Te"}


def test_thomson_fix_error_can_be_recorded(tmp_path, starter):
    shot_dict = make_mast_shot()
    shot_dict["thomson_core_Te"] = np.hstack([shot_dict["thomson_core_Te"]] * 2)
    shot_dict["thomson_core_ne"] = np.hstack([shot_dict["thomson_core_ne"]] * 2)
    path = write_shot(tmp_path / "mast_shot_3.parquet", shot_dict)
    with pytest.raises(ValueError):
        io.load_shot(path, ["thomson_core_Te"])
    shot = io.load_shot(path, ["thomson_core_Te"], on_fix_error="record")
    assert "thomson_core_fix_error" in shot.fixes
    assert shot["thomson_core_Te"].shape[1] == 262


def test_targets_untouched_by_loader(d3d_file, starter):
    raw = io.load_shot(d3d_file, ["efit_psirz", "efit_q95"], apply=False)
    fixed = io.load_shot(d3d_file, ["efit_psirz", "efit_q95"], apply=True)
    assert np.array_equal(raw["efit_psirz"], fixed["efit_psirz"])
    assert np.array_equal(raw["efit_q95"], fixed["efit_q95"])


def test_sample_shot_fix_matches_dictionary(starter):
    paths = starter_kit.sample_shot_paths("DIII-D")
    if not paths:
        pytest.skip("sample shots not available")
    shot = io.load_shot(paths[0], ["magnetics_plasma_current", "efit_times"])
    assert shot.fixes["ip_fix_applied"] is True
    assert shot.fixes["ip_time_shift_ms"] == pytest.approx(-3152.57, abs=0.01)
