"""Tests for the quality checks on small synthetic shots.

Every check has one passing case and one case built to fail it. The scorer
comparison needs the starter kit and skips without it.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from fusion_eq import io
from fusion_eq.qa import registry
from fusion_eq.qa.currents import check_current_units, check_ip_time_axis
from fusion_eq.qa.efit_window import nearest_distance_ms, window_mask
from fusion_eq.qa.geometry import check_coil_geometry, check_grid
from fusion_eq.qa.inventory import check_column_set, check_file_inventory, null_leaf_count
from fusion_eq.qa.labels import check_boundary, check_dsep, check_flux_map, check_label_sanity
from fusion_eq.qa.report import RunContext, render_report, summarize
from fusion_eq.qa.run import rows_to_frame, run_shot
from fusion_eq.qa.scorer_check import check_against_scorer
from fusion_eq.qa.thomson import check_thomson_alignment, check_thomson_missing
from tests.conftest import D3D_MAGNETICS_ORIGIN_MS, drop_labels, make_d3d_shot, make_mast_shot, write_shot


def load(tmp_path, shot_dict: dict, name: str = "d3d_shot_1.parquet", columns=None) -> io.Shot:
    path = write_shot(tmp_path / name, shot_dict)
    return io.load_shot(path, columns, on_fix_error="record")


# ----------------------------------------------------------------------------- window helpers
def test_window_mask_and_nearest_distance():
    efit = np.array([100.0, 120.0, 140.0])
    mask = window_mask(np.array([90.0, 100.0, 130.0, 140.0, 141.0, np.nan]), efit)
    assert mask.tolist() == [False, True, True, True, False, False]
    d = nearest_distance_ms(efit, np.array([103.0, np.nan, 150.0]))
    assert d.tolist() == [3.0, 17.0, 10.0]
    assert np.isinf(nearest_distance_ms(efit, np.array([np.nan]))).all()


# ----------------------------------------------------------------------------- 0.1 / 0.14
def test_inventory_and_column_set(d3d_file):
    inv = check_file_inventory(d3d_file)
    assert inv["inv_opens"] is True and inv["inv_n_rows"] == 1 and inv["inv_file_size_bytes"] > 0
    cols = check_column_set(d3d_file)
    assert cols["col_n"] == inv["inv_n_columns"]
    assert "efit_psirz" in cols["col_names"].split("|")
    assert cols["col_dtype_magnetics_time"] == "float" and cols["col_dtype_efit_times"] == "double"
    nulls = null_leaf_count(io.read_table(d3d_file, ["efit_times", "efit_psirz"]))
    assert nulls["inv_null_leaves"] == 0


def test_inventory_fails_on_corrupt_file(tmp_path):
    path = tmp_path / "d3d_shot_bad.parquet"
    path.write_bytes(b"not a parquet file")
    inv = check_file_inventory(path)
    assert inv["inv_opens"] is False and "inv_open_error" in inv


# ----------------------------------------------------------------------------- 0.2
IP_CHECK_COLUMNS = ["magnetics_plasma_current", "magnetics_plasma_current_times", "magnetics_time",
                    "magnetics_ECOILA", "efit_times"]


def test_ip_fix_affected_shot_passes(tmp_path, starter):
    shot = load(tmp_path, make_d3d_shot(ip_fix_needed=True), columns=IP_CHECK_COLUMNS)
    out = check_ip_time_axis(shot)
    assert out["ip_fix_applied"] is True and out["ip_alignment_applied"] is False
    assert out["ip_shift_ms"] == pytest.approx(-3152.56, abs=0.01) and out["ip_alignment_shift_ms"] == 0.0
    assert out["ip_breakdown_shipped_ms"] > 1000 and out["ip_breakdown_fixed_ms"] == pytest.approx(10.25, abs=1e-3)
    assert out["ip_breakdown_aligned_ms"] == out["ip_breakdown_fixed_ms"]
    assert out["ip_covers_efit"] is True and out["ip_near_zero"] is False
    assert out["ip_median_abs_efit_ka"] == pytest.approx(1001, abs=2)
    assert out["ip_median_abs_efit_shipped_ka"] < 5
    assert 0.9 < out["ip_median_over_p995"] <= 1.0
    assert out["ip_first_over_p995"] == pytest.approx(1.0, abs=0.01)
    assert out["ip_breakdown_to_first_frame_ms"] == pytest.approx(300 - 10.25, abs=1e-3)
    assert out["ip_first_frame_before_breakdown"] is False
    assert 0.0 <= out["ip_sol_anchor_ms"] <= 2.0 and out["ip_lag_a_ms"] == pytest.approx(10.25 - out["ip_sol_anchor_ms"], abs=1e-3)
    assert abs(out["ip_a_minus_b_ms"]) < 2.0


def test_ip_untouched_shot_is_aligned_by_rule_b(tmp_path, starter):
    shot = load(tmp_path, make_d3d_shot(ip_fix_needed=False, breakdown_ms=250.0), columns=IP_CHECK_COLUMNS)
    out = check_ip_time_axis(shot)
    assert out["ip_fix_applied"] is False and out["ip_shift_ms"] == 0.0
    assert out["ip_alignment_applied"] is True
    assert out["ip_breakdown_fixed_ms"] == pytest.approx(250.25, abs=1e-3)
    assert out["ip_alignment_shift_ms"] == pytest.approx(14.9 - 250.25, abs=1e-3)
    assert out["ip_breakdown_aligned_ms"] == pytest.approx(14.9)
    assert out["ip_shift_b_ms"] == pytest.approx(out["ip_alignment_shift_ms"])
    assert out["ip_near_zero"] is False and out["ip_first_frame_before_breakdown"] is False


def test_ip_first_frame_before_breakdown_fails(tmp_path, starter):
    d = make_d3d_shot(ip_fix_needed=True)
    d["efit_times"] = d["efit_times"] - 295.0              # first frame at 5 ms, breakdown at 10.25 ms
    d["magnetics_dsep_times"] = d["efit_times"].astype(np.float32)
    shot = load(tmp_path, d, columns=IP_CHECK_COLUMNS)
    out = check_ip_time_axis(shot)
    assert out["ip_breakdown_to_first_frame_ms"] < 0 and out["ip_first_frame_before_breakdown"] is True
    assert out["ip_first_over_p995"] < 0.1


def test_ip_near_zero_fails_when_plasma_misses_efit_window(tmp_path, starter):
    # Organizer-fixed shot (so rule B leaves it alone) whose plasma sits after the EFIT window.
    d = make_d3d_shot(ip_fix_needed=True)
    axis = D3D_MAGNETICS_ORIGIN_MS + 0.5 * np.arange(d["magnetics_plasma_current"].size)
    d["magnetics_plasma_current"] = np.where((axis > 6000) & (axis < 9000), 1000.0, 0.0).astype(np.float32)
    shot = load(tmp_path, d, columns=IP_CHECK_COLUMNS)
    out = check_ip_time_axis(shot)
    assert out["ip_fix_applied"] is True and out["ip_alignment_applied"] is False
    assert out["ip_near_zero"] is True and out["ip_median_over_p995"] < 0.1


def test_ip_rule_a_disagreement_is_measured(tmp_path, starter):
    d = make_d3d_shot(ip_fix_needed=True)
    shifted = np.roll(d["magnetics_ECOILA"], 10)            # solenoid swing 20 ms later (2 ms magnetics step)
    d["magnetics_ECOILA"] = shifted
    shot = load(tmp_path, d, columns=IP_CHECK_COLUMNS)
    out = check_ip_time_axis(shot)
    assert out["ip_sol_anchor_ms"] == pytest.approx(20.0, abs=2.5)
    assert out["ip_a_minus_b_ms"] > 10.0


def test_ip_check_on_mast_uses_magnetics_time(tmp_path, starter):
    shot = load(tmp_path, make_mast_shot(), "mast_shot_1.parquet",
                columns=["magnetics_plasma_current", "magnetics_time", "efit_times"])
    out = check_ip_time_axis(shot)
    assert out["ip_fix_applied"] is False and out["ip_covers_efit"] is True
    assert out["ip_median_abs_efit_ka"] == pytest.approx(800.0)


# ----------------------------------------------------------------------------- 0.4
def test_current_units_inside_window(tmp_path, starter):
    shot = load(tmp_path, make_d3d_shot(), columns=["magnetics_*", "efit_times"]
                if False else None)
    out = check_current_units(shot)
    assert out["cur_n_columns"] == 21
    assert out["cur_F1A_min"] == 100.0 and out["cur_F1A_max"] == 100.0 and out["cur_F1A_p99abs"] == 100.0
    assert out["cur_bcoil_min"] == -117.0
    assert out["cur_plasma_current_min"] == pytest.approx(1000.0) and out["cur_plasma_current_nan_frac"] == 0.0


def test_current_units_reports_nan_holes(tmp_path, starter):
    m = make_mast_shot()
    m["magnetics_tf_current"][::2] = np.nan           # union-grid style holes
    shot = load(tmp_path, m, "mast_shot_1.parquet")
    out = check_current_units(shot)
    assert out["cur_tf_current_nan_frac"] == pytest.approx(0.5, abs=0.01)
    assert out["cur_tf_current_min"] == -85.0
    assert out["cur_n_columns"] == 14


# ----------------------------------------------------------------------------- 0.3
def test_thomson_alignment_mast_and_d3d(tmp_path, starter):
    mast = load(tmp_path, make_mast_shot(), "mast_shot_1.parquet")
    out = check_thomson_alignment(mast)
    assert out["ts_fix_applied"] is True and out["ts_channels_before"] == 131
    assert out["ts_channels_after"] == 130 == out["ts_n_positions"] and out["ts_aligned"] is True
    assert out["ts_dropped_nan_frac_te"] == 1.0
    d3d = load(tmp_path, make_d3d_shot())
    out = check_thomson_alignment(d3d)
    assert out["ts_fix_applied"] is False and out["ts_aligned"] is True and out["ts_channels_after"] == 6


def test_thomson_alignment_flags_unexpected_layout(tmp_path, starter):
    m = make_mast_shot()
    m["thomson_core_Te"] = np.hstack([m["thomson_core_Te"], m["thomson_core_Te"][:, :1]])
    m["thomson_core_ne"] = np.hstack([m["thomson_core_ne"], m["thomson_core_ne"][:, :1]])
    shot = load(tmp_path, m, "mast_shot_1.parquet")
    out = check_thomson_alignment(shot)
    assert out["ts_aligned"] is False and "ts_fix_error" in out


# ----------------------------------------------------------------------------- 0.7
def test_thomson_missing_d3d_zero_convention(tmp_path, starter):
    shot = load(tmp_path, make_d3d_shot())
    out = check_thomson_missing(shot)
    per_channel = json.loads(out["tsm_core_missing_frac_per_channel"])
    assert len(per_channel) == 6 and per_channel[5] == 1.0           # dead channel
    assert per_channel[0] == pytest.approx(0.25, abs=0.02)            # every fourth slice is zero
    assert out["tsm_core_slice_all_missing_frac"] == pytest.approx(0.25, abs=0.02)
    assert out["tsm_core_shape_matches_times"] is True
    assert out["tsm_core_frames_valid_within_10ms"] == 1.0 and out["tsm_core_frames_any_within_20ms"] == 1.0
    assert out["tsm_edge_missing_frac_mean"] == 0.0
    assert out["tsm_core_n_channels"] == 6


def test_thomson_missing_requires_both_te_and_ne(tmp_path, starter):
    d = make_d3d_shot()
    d["thomson_core_ne"][:, 0] = 0.0          # Te present, ne missing on channel 0
    shot = load(tmp_path, d)
    out = check_thomson_missing(shot)
    assert json.loads(out["tsm_core_missing_frac_per_channel"])[0] == 1.0


def test_thomson_missing_mast_nan_convention_and_sparse_frames(tmp_path, starter):
    m = make_mast_shot()
    m["efit_times"] = np.array([150.0, 400.0, 420.0], dtype=np.float32)   # two frames far from any pulse
    for key in ("efit_q95", "efit_beta_n", "efit_li", "efit_r_axis", "efit_z_axis", "efit_lcfs_n",
                "efit_elongation", "magnetics_dsep", "magnetics_dsep_times"):
        m[key] = m[key][:3]
    m["efit_psirz"] = m["efit_psirz"][:3]
    m["efit_lcfs_r"] = m["efit_lcfs_r"][:3]
    m["efit_lcfs_z"] = m["efit_lcfs_z"][:3]
    shot = load(tmp_path, m, "mast_shot_1.parquet")
    out = check_thomson_missing(shot)
    assert out["tsm_core_frames_valid_within_10ms"] == pytest.approx(1 / 3)
    assert out["tsm_core_frames_valid_within_20ms"] == pytest.approx(1 / 3)
    assert out["tsm_core_missing_frac_min"] == 0.0


def test_thomson_missing_detects_shape_mismatch(tmp_path, starter):
    d = make_d3d_shot()
    d["thomson_edge_Te"] = d["thomson_edge_Te"][:-1]
    d["thomson_edge_ne"] = d["thomson_edge_ne"][:-1]
    shot = load(tmp_path, d)
    out = check_thomson_missing(shot)
    assert out["tsm_edge_shape_matches_times"] is False


# ----------------------------------------------------------------------------- 0.9
def test_coil_geometry_hash_and_mapping(tmp_path, starter):
    a = load(tmp_path, make_d3d_shot(), "d3d_shot_1.parquet")
    b = load(tmp_path, make_d3d_shot(), "d3d_shot_2.parquet")
    out_a, out_b = check_coil_geometry(a), check_coil_geometry(b)
    assert out_a["coil_hash"] == out_b["coil_hash"]
    assert out_a["coil_n_elements"] == 19 and out_a["coil_mapping_ok"] is True
    assert out_a["coil_solenoid_present"] is True and out_a["coil_solenoid_n_elements"] == 1
    mast = load(tmp_path, make_mast_shot(), "mast_shot_1.parquet")
    out_m = check_coil_geometry(mast)
    assert out_m["coil_solenoid_n_elements"] == 20 and json.loads(out_m["coil_elements_per_column"])["magnetics_p2l_current"] == 4


def test_coil_geometry_fails_on_unmapped_column_and_changed_geometry(tmp_path, starter):
    d = make_d3d_shot()
    d["coil_input_column"][0] = "magnetics_F0A"
    d["coil_R"][3] += 0.01
    shot = load(tmp_path, d)
    out = check_coil_geometry(shot)
    assert out["coil_mapping_ok"] is False and out["coil_unmapped_columns"] == "magnetics_F0A"
    reference = check_coil_geometry(load(tmp_path, make_d3d_shot(), "d3d_shot_2.parquet"))
    assert out["coil_hash"] != reference["coil_hash"]


# ----------------------------------------------------------------------------- 0.10
def test_grid_passes_and_detects_reversed_axis(tmp_path, starter):
    shot = load(tmp_path, make_d3d_shot())
    out = check_grid(shot)
    assert out["grid_ok"] is True and out["grid_R_n"] == 65 and out["grid_R_uniform"] is True
    assert out["grid_R_min"] == pytest.approx(0.84) and out["grid_Z_max"] == pytest.approx(1.6)
    d = make_d3d_shot()
    d["efit_grid_Z"] = d["efit_grid_Z"][::-1].copy()
    bad = check_grid(load(tmp_path, d, "d3d_shot_2.parquet"))
    assert bad["grid_ok"] is False and bad["grid_Z_increasing"] is False
    assert bad["grid_hash"] != out["grid_hash"]


# ----------------------------------------------------------------------------- 0.5
def test_flux_map_indexing_passes(tmp_path, starter):
    for shot_dict, name in ((make_d3d_shot(), "d3d_shot_1.parquet"), (make_mast_shot(), "mast_shot_1.parquet")):
        out = check_flux_map(load(tmp_path, shot_dict, name))
        assert out["psi_shape_ok"] is True
        assert out["psi_opoint_within_1cell_frac"] == 1.0 and out["psi_opoint_cell_dist_max"] <= 1.0
        assert out["psi_opoint_used_local_frac"] == 1.0
        assert out["psi_axis_within_1cell_frac"] == 1.0 and out["psi_axis_cell_dist_max"] <= 1.0
        assert out["psi_axis_on_border_frac"] == 0.0
        assert out["psi_axis_opposite_cell_dist_median"] > 5


def test_flux_map_detects_transposition_and_bad_shape(tmp_path, starter):
    d = make_d3d_shot()
    d["efit_psirz"] = np.transpose(d["efit_psirz"], (0, 2, 1))
    out = check_flux_map(load(tmp_path, d))
    assert out["psi_opoint_within_1cell_frac"] < 1.0 and out["psi_opoint_cell_dist_max"] > 1.0
    assert out["psi_axis_within_1cell_frac"] < 1.0
    assert out["psi_axis_transposed_cell_dist_median"] < out["psi_axis_cell_dist_median"]
    d = make_d3d_shot()
    d["efit_psirz"] = d["efit_psirz"][:, :64, :64]
    out = check_flux_map(load(tmp_path, d, "d3d_shot_2.parquet"))
    assert out["psi_shape_ok"] is False


def test_flux_map_border_extremum_is_review_not_fail(tmp_path, starter):
    # A strong coil-like flux at a grid corner: the global minimum of psi (DIII-D convention)
    # walks to the border while the scorer's interior O-point stays at the plasma centre.
    d = make_d3d_shot()
    d["efit_psirz"][:, 0, 0] = -5.0
    out = check_flux_map(load(tmp_path, d))
    assert out["psi_axis_on_border_frac"] == 1.0 and out["psi_axis_within_1cell_frac"] == 0.0
    assert out["psi_opoint_within_1cell_frac"] == 1.0 and out["psi_opoint_cell_dist_max"] <= 1.0


# ----------------------------------------------------------------------------- 0.6
def test_boundary_padding_conventions(tmp_path, starter):
    out = check_boundary(load(tmp_path, make_d3d_shot()))
    assert out["lcfs_pad_kind"] == "zero" and out["lcfs_width"] == 44 and out["lcfs_n_min"] == 41
    assert out["lcfs_valid_nan_frac"] == 0.0 and out["lcfs_closed_frac"] == 1.0 and out["lcfs_n_integer"] is True
    out = check_boundary(load(tmp_path, make_mast_shot(), "mast_shot_1.parquet"))
    assert out["lcfs_pad_kind"] == "nan" and out["lcfs_width"] == 64


def test_boundary_flags_nan_inside_and_odd_padding(tmp_path, starter):
    d = make_d3d_shot()
    d["efit_lcfs_r"][0, 3] = np.nan            # inside the valid points
    d["efit_lcfs_r"][1, 43] = 7.0              # padding that is neither 0 nor NaN
    out = check_boundary(load(tmp_path, d))
    assert out["lcfs_valid_nan_frac"] > 0 and out["lcfs_pad_kind"] == "mixed" and out["lcfs_pad_other_count"] == 1


# ----------------------------------------------------------------------------- 0.8
def test_dsep_sentinel_counting(tmp_path, starter):
    out = check_dsep(load(tmp_path, make_mast_shot(), "mast_shot_1.parquet"))
    assert out["dsep_n_sentinel"] == 2 and out["dsep_n_nan"] == 0
    assert out["dsep_min"] == pytest.approx(0.002) and out["dsep_times_equal_efit"] is True
    out = check_dsep(load(tmp_path, make_d3d_shot()))
    assert out["dsep_n_sentinel"] == 0 and out["dsep_frac_negative"] == 0.0


# ----------------------------------------------------------------------------- 0.11
def test_label_sanity_passes(tmp_path, starter):
    out = check_label_sanity(load(tmp_path, make_d3d_shot()))
    assert out["time_n_frames"] == 12 and out["time_efit_increasing"] is True
    assert out["time_efit_dt_median_ms"] == 20.0 and out["time_ip_dt_median_ms"] == 0.5
    assert out["lab_all_finite"] is True and out["lab_q95_min"] == 4.0 and out["lab_psi_len_ok"] is True


def test_label_sanity_fails_on_nan_label_and_time_order(tmp_path, starter):
    d = make_d3d_shot()
    d["efit_q95"][2] = np.nan
    d["efit_times"][5] = d["efit_times"][4]
    out = check_label_sanity(load(tmp_path, d))
    assert out["lab_all_finite"] is False and out["lab_q95_nan"] == 1
    assert out["time_efit_increasing"] is False
    inputs_only = check_label_sanity(load(tmp_path, drop_labels(make_d3d_shot()), "d3d_shot_2.parquet"))
    assert "lab_all_finite" not in inputs_only and inputs_only["time_n_frames"] == 12


# ----------------------------------------------------------------------------- 0.15
def test_scorer_check_recovers_synthetic_axis(tmp_path, starter):
    shot = load(tmp_path, make_d3d_shot())
    out = check_against_scorer(shot, n_sample=3)
    assert out["sc_n_frames"] == 3
    assert out["sc_raxis_absdiff_max"] < 0.03 and out["sc_zaxis_absdiff_max"] < 0.03
    assert out["sc_mask_grid_max_abs_diff_m"] < 1e-5
    assert out["sc_lcfs_fail_count"] == 0 and np.isfinite(out["sc_hausdorff_median"])


def test_scorer_check_counts_failed_extractions(tmp_path, starter):
    d = make_d3d_shot()
    d["efit_psirz"][:] = 0.0
    out = check_against_scorer(load(tmp_path, d), n_sample=2)
    assert out["sc_lcfs_fail_count"] == 2 and np.isnan(out["sc_hausdorff_median"])


# ----------------------------------------------------------------------------- registry and runner
def test_registry_skips_label_checks_on_inputs_only_files():
    train_cols = list(make_d3d_shot())
    test_cols = list(drop_labels(make_d3d_shot()))
    ids_train = {c.check_id for c in registry.applicable_checks(train_cols, with_sampled=True)}
    ids_test = {c.check_id for c in registry.applicable_checks(test_cols, with_sampled=True)}
    assert {"0.5", "0.6", "0.8", "0.15"} <= ids_train
    assert ids_test.isdisjoint({"0.5", "0.6", "0.8", "0.15"})
    assert {"0.2", "0.4", "0.7", "0.9", "0.10", "0.11"} <= ids_test
    cols = registry.columns_for(registry.applicable_checks(test_cols, False), test_cols)
    assert "efit_psirz" not in cols and "magnetics_F1A" in cols and "thomson_edge_Te" in cols


def test_run_shot_produces_row_and_survives_bad_file(tmp_path, starter):
    path = write_shot(tmp_path / "d3d_shot_7.parquet", make_d3d_shot())
    row = run_shot(path, "diii_d_train", with_sampled=True)
    assert row["error"] is None and row["machine"] == "DIII-D" and row["shot"] == "7"
    assert row["ip_fix_applied"] is True and row["psi_shape_ok"] is True and "sc_n_frames" in row
    assert row["inv_null_leaves"] == 0
    bad = tmp_path / "d3d_shot_8.parquet"
    bad.write_bytes(b"garbage")
    row = run_shot(bad, "diii_d_train")
    assert row["inv_opens"] is False and "0.1" in row["error"]


def test_run_shot_records_check_error_and_continues(tmp_path, starter):
    d = make_d3d_shot()
    d["efit_lcfs_n"] = np.array(["a"] * 12, dtype=object)       # breaks check 0.6 only
    row = run_shot(write_shot(tmp_path / "d3d_shot_9.parquet", d), "diii_d_train")
    assert row["error"] is not None and "0.6" in row["error"]
    assert row["psi_shape_ok"] is True and row["grid_ok"] is True


def test_rows_to_frame_types_and_parquet_round_trip(tmp_path):
    rows = [
        {"split": "a", "machine": "DIII-D", "shot": "2", "path": "p", "error": None, "flag": True, "x": 1.0, "s": "k"},
        {"split": "a", "machine": "MAST", "shot": "1", "path": "p", "error": "e", "flag": None, "x": None, "s": None},
    ]
    df = rows_to_frame(rows)
    assert list(df["shot"]) == ["1", "2"]
    assert str(df["flag"].dtype) == "boolean" and str(df["x"].dtype) == "float64" and str(df["s"].dtype) == "string"
    df.to_parquet(tmp_path / "t.parquet", index=False)
    back = pd.read_parquet(tmp_path / "t.parquet")
    assert back.shape == df.shape


def test_summarize_and_render_report(tmp_path, starter):
    rows = [run_shot(write_shot(tmp_path / "d3d_shot_1.parquet", make_d3d_shot()), "diii_d_train", True),
            run_shot(write_shot(tmp_path / "d3d_shot_2.parquet", drop_labels(make_d3d_shot())), "diii_d_public_test"),
            run_shot(write_shot(tmp_path / "mast_shot_1.parquet", drop_labels(make_mast_shot())), "mast_public_test"),
            run_shot(write_shot(tmp_path / "mast_shot_9.parquet", make_mast_shot()), "mast_samples")]
    df = rows_to_frame(rows)
    results = summarize(df, RunContext(full_run=False, splits=("diii_d_train",)))
    by_id = {r.check_id: r for r in results}
    assert by_id["0.12"].status == "pass" and by_id["0.2"].status == "pass"
    assert by_id["0.5"].status == "pass" and by_id["0.10"].status == "pass" and by_id["0.11"].status == "pass"
    assert by_id["0.1"].status == "review" and by_id["0.11b"].status == "review"
    assert "matches the expected" in by_id["0.14"].headline
    report = render_report(results, {"timestamp": "t", "repo_commit": "c", "starter_commit": "s",
                                     "runtime_s": 1.0, "args": ""}, df)
    assert "## Summary" in report and "### 0.15" in report and "D1" in report


def test_summarize_flags_failures(tmp_path, starter):
    bad = make_d3d_shot()
    bad["efit_beta_n"][0] = np.inf
    rows = [run_shot(write_shot(tmp_path / "d3d_shot_1.parquet", bad), "diii_d_train"),
            run_shot(write_shot(tmp_path / "d3d_shot_1.parquet", drop_labels(make_d3d_shot())), "diii_d_public_test")]
    results = {r.check_id: r for r in summarize(rows_to_frame(rows), RunContext(True, ("diii_d_train",)))}
    assert results["0.11"].status == "fail" and results["0.11"].flagged
    assert results["0.12"].status == "fail"            # same shot id in two splits
    assert results["0.1"].status == "fail"             # full run with wrong counts


def test_summarize_flux_map_fail_and_review_rules(tmp_path, starter):
    border = make_d3d_shot()
    border["efit_psirz"][:, 0, 0] = -5.0
    transposed = make_d3d_shot()
    transposed["efit_psirz"] = np.transpose(transposed["efit_psirz"], (0, 2, 1))
    rows = [run_shot(write_shot(tmp_path / "d3d_shot_border.parquet", border), "diii_d_train"),
            run_shot(write_shot(tmp_path / "d3d_shot_good.parquet", make_d3d_shot()), "diii_d_train")]
    r = {r.check_id: r for r in summarize(rows_to_frame(rows), RunContext(False, ("diii_d_train",)))}["0.5"]
    assert r.status == "pass" and not r.flagged
    assert len(r.reviewed) == 1 and "/border:" in r.reviewed[0]
    rows.append(run_shot(write_shot(tmp_path / "d3d_shot_transposed.parquet", transposed), "diii_d_train"))
    r = {r.check_id: r for r in summarize(rows_to_frame(rows), RunContext(False, ("diii_d_train",)))}["0.5"]
    assert r.status == "fail" and len(r.flagged) == 1 and "/transposed:" in r.flagged[0]


def test_summarize_ip_fail_and_review_rules(tmp_path, starter):
    early_frames = make_d3d_shot(ip_fix_needed=True)
    early_frames["efit_times"] = early_frames["efit_times"] - 295.0
    early_frames["magnetics_dsep_times"] = early_frames["efit_times"].astype(np.float32)
    late_anchor = make_d3d_shot(ip_fix_needed=True)
    late_anchor["magnetics_ECOILA"] = np.roll(late_anchor["magnetics_ECOILA"], 10)
    late_fixed_breakdown = make_d3d_shot(ip_fix_needed=True, breakdown_ms=120.0)
    no_anchor = make_d3d_shot(ip_fix_needed=True)
    no_anchor["magnetics_ECOILA"][:] = 0.0
    rows = [run_shot(write_shot(tmp_path / f"d3d_shot_{i}.parquet", d), "diii_d_train")
            for i, d in enumerate([early_frames, late_anchor, late_fixed_breakdown, no_anchor, make_d3d_shot(ip_fix_needed=False)])]
    r = {r.check_id: r for r in summarize(rows_to_frame(rows), RunContext(False, ("diii_d_train",)))}["0.2"]
    assert r.status == "fail"
    assert len(r.flagged) == 1 and "/0:" in r.flagged[0]                       # first frame before breakdown
    reasons = {f.split(": ", 1)[1] for f in r.reviewed}
    assert any("differ by more than" in x for x in reasons)
    assert any("no solenoid" in x for x in reasons)
    assert any("outside 0 to 60 ms" in x for x in reasons)
    assert sum("/1:" in f for f in r.reviewed) == 1 and sum("/2:" in f for f in r.reviewed) == 1
    assert sum("/3:" in f for f in r.reviewed) == 1 and not any("/4:" in f for f in r.reviewed)
