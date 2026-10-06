"""Tests for the shared Thomson validity rules and slice summaries."""

from __future__ import annotations

import numpy as np
import pytest

from fusion_eq import thomson


def _profiles(n_times=5, n_channels=4, te=1000.0, ne=3e19):
    t = 10.0 * np.arange(n_times)
    te_arr = np.full((n_times, n_channels), te)
    ne_arr = np.full((n_times, n_channels), ne)
    return t, te_arr, ne_arr


def test_missing_mask_conventions():
    values = np.array([0.0, 1.0, np.nan, np.inf])
    assert thomson.missing_mask(values, "zero").tolist() == [True, False, True, True]
    assert thomson.missing_mask(values, "nan").tolist() == [False, False, True, True]
    with pytest.raises(ValueError):
        thomson.missing_mask(values, "sentinel")


def test_zero_and_nan_both_mark_missing_and_both_te_and_ne_are_needed():
    t, te, ne = _profiles(n_times=3, n_channels=4)
    te[0, 0] = 0.0                           # DIII-D style missing
    ne[1, 1] = np.nan                        # MAST style missing
    valid_zero = thomson.valid_channel_mask(te, ne, "zero")
    valid_nan = thomson.valid_channel_mask(te, ne, "nan")
    assert not valid_zero[0, 0] and not valid_zero[1, 1]
    assert valid_nan[0, 0] and not valid_nan[1, 1]


def test_valid_slice_rule_is_ceil_half_of_channels():
    valid = np.zeros((4, 5), dtype=bool)
    valid[0, :3] = True                      # 3 of 5: valid (ceil(2.5) = 3)
    valid[1, :2] = True                      # 2 of 5: not valid
    valid[2, :] = True
    assert thomson.valid_slice_mask(valid).tolist() == [True, False, True, False]
    assert thomson.valid_slice_mask(np.zeros((3, 0), dtype=bool)).tolist() == [False] * 3
    with pytest.raises(ValueError):
        thomson.valid_slice_mask(np.zeros(3, dtype=bool))


def test_pressure_units():
    assert thomson.pressure_kpa(1000.0, 1e20) == pytest.approx(1e20 * 1000.0 * 1.602e-19 / 1000.0)


def test_core_summaries_use_valid_channels_only():
    t, te, ne = _profiles(n_times=4, n_channels=4)
    te[:, 3] = 0.0                           # dead channel on every slice
    te[1, :] = 0.0                           # a slice with no laser data
    te[2, 0] = 5000.0                        # outlier survives the 90th percentile partly
    s = thomson.slice_summaries(t, te, ne, "zero", "core")
    assert s.times_ms.tolist() == [0.0, 20.0, 30.0]
    assert s.n_valid.tolist() == [3, 3, 3]
    assert s.values["te_peak"][0] == pytest.approx(1000.0)
    assert s.values["te_peak"][1] > 1000.0
    assert s.values["pe_mean"][0] == pytest.approx(thomson.pressure_kpa(1000.0, 3e19))
    assert set(s.values) == set(thomson.SUMMARY_NAMES["core"])


def test_edge_summaries_and_no_valid_slice():
    t, te, ne = _profiles(n_times=3, n_channels=2, te=200.0, ne=1e19)
    s = thomson.slice_summaries(t, te, ne, "nan", "edge")
    assert s.values["te_mean"].tolist() == [200.0] * 3
    empty = thomson.slice_summaries(t, np.full_like(te, np.nan), ne, "nan", "edge")
    assert empty.n_slices == 0 and all(v.size == 0 for v in empty.values.values())


def test_slice_summaries_rejects_bad_shapes():
    t, te, ne = _profiles()
    with pytest.raises(ValueError):
        thomson.slice_summaries(t[:-1], te, ne, "nan", "core")
    with pytest.raises(ValueError):
        thomson.slice_summaries(t, te, ne, "nan", "divertor")


def test_frame_columns_interpolation_nearest_and_validity():
    times = np.array([0.0, 10.0, 30.0])
    s = thomson.SliceSummaries("core", times, np.array([5, 6, 7]),
                               {"te_peak": np.array([100.0, 300.0, 700.0])})
    frames = np.array([-5.0, 5.0, 20.0, 29.0, 50.0])
    cols = thomson.frame_columns(s, frames, tol_ms=6.0)
    assert cols["ts_core_te_peak__interp"].tolist() == [100.0, 200.0, 500.0, 680.0, 700.0]
    assert cols["ts_core_te_peak__nearest"].tolist() == [100.0, 100.0, 300.0, 700.0, 700.0]
    assert cols["ts_core_n_valid"].tolist() == [5.0, 5.0, 6.0, 7.0, 7.0]
    assert cols["ts_core_dt_ms"].tolist() == [5.0, 5.0, 10.0, 1.0, 20.0]
    gap = cols["ts_core_gap_ms"]
    assert np.isnan(gap[0]) and gap[1] == 10.0 and gap[2] == 20.0 and gap[3] == 20.0 and np.isnan(gap[4])
    assert cols["ts_core_valid"].tolist() == [True, True, False, True, False]


def test_frame_columns_without_any_valid_slice():
    s = thomson.SliceSummaries("edge", np.zeros(0), np.zeros(0, dtype=int),
                               {name: np.zeros(0) for name in thomson.SUMMARY_NAMES["edge"]})
    cols = thomson.frame_columns(s, np.array([1.0, 2.0]), tol_ms=20.0)
    assert np.isnan(cols["ts_edge_te_mean__interp"]).all()
    assert np.isnan(cols["ts_edge_dt_ms"]).all()
    assert cols["ts_edge_valid"].tolist() == [False, False]
