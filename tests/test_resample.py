"""Tests for the signal resampling functions."""

from __future__ import annotations

import numpy as np
import pytest

from fusion_eq import resample

# Realistic axis: DIII-D magnetics origin, 20 s long, targets several seconds in.
ORIGIN_MS = -4010.75
TARGETS_MS = np.array([300.0, 1234.5, 5000.0, 8765.4])


def ramp(rate_khz: float, slope_ka_per_s: float, offset_ka: float = 500.0):
    """Linear ramp sampled at a given rate; slope given in kA/s."""
    step_ms = 1.0 / rate_khz
    t = ORIGIN_MS + step_ms * np.arange(int(24_000 / step_ms))
    v = offset_ka + slope_ka_per_s * (t / 1000.0)
    return t, v


def test_linear_ramp_reproduced_by_all_forms():
    t, v = ramp(20.0, 250.0)
    expected = 500.0 + 250.0 * TARGETS_MS / 1000.0
    near = resample.nearest_valid(t, v, TARGETS_MS, max_gap_ms=5.0)
    mean = resample.window_mean(t, v, TARGETS_MS, window_ms=2.0, max_gap_ms=5.0)
    slope = resample.window_slope(t, v, TARGETS_MS, window_ms=10.0)
    assert np.allclose(near, expected, atol=250.0 * 0.05 / 1000.0 + 1e-9)   # within one step
    assert np.allclose(mean, expected, atol=1e-6)                             # symmetric window
    assert np.allclose(slope, 250.0, rtol=1e-6)


@pytest.mark.parametrize("rate_khz", [20.0, 2.0, 5.0])
def test_same_signal_at_different_rates_gives_same_mean_and_slope(rate_khz):
    t, v = ramp(rate_khz, -800.0, offset_ka=1200.0)
    t_ref, v_ref = ramp(20.0, -800.0, offset_ka=1200.0)
    # Targets on the common 1 ms grid of the three rates, so every window is symmetric.
    aligned = ORIGIN_MS + np.array([4311.0, 5245.0, 9011.0, 12776.0])
    mean = resample.window_mean(t, v, aligned, window_ms=2.0, max_gap_ms=5.0)
    mean_ref = resample.window_mean(t_ref, v_ref, aligned, window_ms=2.0, max_gap_ms=5.0)
    slope = resample.window_slope(t, v, aligned, window_ms=10.0)
    slope_ref = resample.window_slope(t_ref, v_ref, aligned, window_ms=10.0)
    assert np.allclose(mean, mean_ref, atol=1e-6)
    assert np.allclose(slope, slope_ref, rtol=1e-6)
    # Off the grid, the windowed mean of a ramp is biased by at most half a step times the slope.
    off_grid = resample.window_mean(t, v, TARGETS_MS, window_ms=2.0, max_gap_ms=5.0)
    expected = 1200.0 - 800.0 * TARGETS_MS / 1000.0
    assert np.all(np.abs(off_grid - expected) <= 800.0 * (0.5 / rate_khz) / 1000.0 + 1e-9)


def test_slope_matches_polyfit_on_noisy_signal():
    rng = np.random.default_rng(0)
    t, v = ramp(20.0, 300.0)
    v = v + rng.normal(0.0, 5.0, v.size)
    slope = resample.window_slope(t, v, TARGETS_MS, window_ms=10.0)
    for target, got in zip(TARGETS_MS, slope):
        inside = (t >= target - 5.0) & (t <= target + 5.0)
        expected = np.polyfit(t[inside] / 1000.0, v[inside], 1)[0]
        assert got == pytest.approx(expected, rel=1e-6)


def test_nan_samples_are_skipped():
    t, v = ramp(5.0, 100.0)
    v = v.copy()
    v[::6] = np.nan                                        # the MAST one-in-six hole
    expected = 500.0 + 100.0 * TARGETS_MS / 1000.0
    mean = resample.window_mean(t, v, TARGETS_MS, window_ms=2.0, max_gap_ms=5.0)
    slope = resample.window_slope(t, v, TARGETS_MS, window_ms=10.0)
    near = resample.nearest_valid(t, v, TARGETS_MS, max_gap_ms=5.0)
    assert np.all(np.isfinite(mean)) and np.allclose(mean, expected, atol=0.05)
    assert np.allclose(slope, 100.0, rtol=1e-6)
    assert np.all(np.isfinite(near))


def test_no_valid_sample_near_target_gives_nan():
    t = np.array([0.0, 1.0, 2.0, 100.0, 101.0])
    v = np.array([1.0, 1.0, 1.0, 5.0, 5.0])
    targets = np.array([1.0, 50.0, 100.5, 300.0])
    near = resample.nearest_valid(t, v, targets, max_gap_ms=5.0)
    assert near[0] == 1.0 and np.isnan(near[1]) and near[2] == 5.0 and np.isnan(near[3])
    mean = resample.window_mean(t, v, targets, window_ms=2.0, max_gap_ms=5.0)
    assert mean[0] == 1.0 and np.isnan(mean[1]) and mean[2] == 5.0 and np.isnan(mean[3])
    slope = resample.window_slope(t, v, targets, window_ms=10.0)
    assert slope[0] == 0.0 and np.isnan(slope[1]) and np.isnan(slope[2])   # two samples only
    assert np.all(np.isnan(resample.window_mean(t, np.full(5, np.nan), targets, 2.0, 5.0)))


def test_window_mean_interpolates_small_gaps_only():
    t = np.array([0.0, 4.0, 20.0])
    v = np.array([0.0, 8.0, 100.0])
    mean = resample.window_mean(t, v, np.array([2.0, 12.0]), window_ms=1.0, max_gap_ms=5.0)
    assert mean[0] == pytest.approx(4.0)            # 4 ms gap, interpolated
    assert np.isnan(mean[1])                        # 16 ms gap, too wide


def test_nearest_index_ties_go_to_earlier_sample_and_distance():
    t = np.array([0.0, 2.0, 4.0])
    idx = resample.nearest_index(t, np.array([1.0, 2.9, 3.0, 10.0]))
    assert idx.tolist() == [0, 1, 1, 2]
    d = resample.nearest_distance_ms(np.array([1.0, 7.0]), np.array([4.0, np.nan, 0.0]))
    assert d.tolist() == [1.0, 3.0]
    assert np.isinf(resample.nearest_distance_ms(np.array([1.0]), np.array([np.nan]))).all()


def test_linear_interpolate_holds_or_drops_ends():
    t = np.array([10.0, 20.0, np.nan])
    v = np.array([1.0, 3.0, 9.0])
    targets = np.array([0.0, 15.0, 30.0])
    held = resample.linear_interpolate(t, v, targets, hold_ends=True)
    assert held.tolist() == [1.0, 2.0, 3.0]
    dropped = resample.linear_interpolate(t, v, targets, hold_ends=False)
    assert np.isnan(dropped[0]) and dropped[1] == 2.0 and np.isnan(dropped[2])


def test_unsorted_input_and_length_mismatch():
    t = np.array([2.0, 0.0, 1.0])
    v = np.array([20.0, 0.0, 10.0])
    assert resample.window_mean(t, v, np.array([1.0]), 0.5, 5.0)[0] == 10.0
    with pytest.raises(ValueError):
        resample.nearest_valid(t, v[:2], np.array([1.0]), 5.0)
