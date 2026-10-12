"""Tests for the Ip time-axis alignment (rule B) and the rule A diagnostic."""

from __future__ import annotations

import numpy as np
import pytest

from fusion_eq import io, ip_alignment
from tests.conftest import D3D_MAGNETICS_ORIGIN_MS, make_d3d_shot, write_shot

TARGET = ip_alignment.TARGET_BREAKDOWN_MS


def test_breakdown_matches_organizers_definition(starter):
    t = -100.0 + 0.5 * np.arange(1000)
    ip = np.where(t > 37.0, 800.0, 1.5)              # noise floor then plasma
    assert ip_alignment.breakdown_time_ms(t, ip) == 37.5
    row = {"magnetics_plasma_current_times": t, "magnetics_plasma_current": ip, "magnetics_time": t}
    fixed = starter.data_fixes().fix_d3d_ip_times(row)
    assert ip_alignment.breakdown_time_ms(fixed, ip) == 37.5   # same decision point as the organizers' helper
    assert np.isnan(ip_alignment.breakdown_time_ms(t, np.zeros_like(t)))


def test_align_ip_times_moves_breakdown_to_target():
    t = -100.0 + 0.5 * np.arange(1000)
    ip = np.where(t > 200.0, 800.0, 0.0)
    aligned, shift, before = ip_alignment.align_ip_times(t, ip)
    assert before == 200.5 and shift == pytest.approx(TARGET - 200.5)
    assert ip_alignment.breakdown_time_ms(aligned, ip) == pytest.approx(TARGET)
    unchanged, shift0, nan_before = ip_alignment.align_ip_times(t, np.zeros_like(t))
    assert shift0 == 0.0 and np.isnan(nan_before) and np.array_equal(unchanged, t)


def test_solenoid_swing_down_anchor():
    t = -2000.0 + 0.05 * np.arange(200_000)          # 20 kHz, to 8 s
    c = np.zeros_like(t)
    c[(t >= -800) & (t < 0)] = 40.0
    fall = (t >= 0) & (t <= 50)
    c[fall] = 40.0 - 1.5 * t[fall]
    c[(t > 50) & (t < 4000)] = -35.0
    anchor = ip_alignment.solenoid_swing_down_ms(t, c)
    assert 0.0 <= anchor <= 2.0                       # leaves the 5 percent band about 1.3 ms into the fall
    # Opposite polarity gives the same anchor; a flat trace gives none.
    assert ip_alignment.solenoid_swing_down_ms(t, -c) == pytest.approx(anchor)
    assert np.isnan(ip_alignment.solenoid_swing_down_ms(t, np.zeros_like(t)))


def test_rule_a_diagnostics_arithmetic():
    d = ip_alignment.rule_a_diagnostics(anchor_ms=1.0, breakdown_fixed_ms=200.0,
                                        target_breakdown_ms=15.0, target_anchor_ms=1.0)
    assert d["lag_a_ms"] == 199.0 and d["shift_a_ms"] == -185.0 and d["shift_b_ms"] == -185.0
    assert d["a_minus_b_ms"] == 0.0
    d = ip_alignment.rule_a_diagnostics(anchor_ms=6.0, breakdown_fixed_ms=200.0,
                                        target_breakdown_ms=15.0, target_anchor_ms=1.0)
    assert d["a_minus_b_ms"] == pytest.approx(5.0)    # equals anchor minus reference anchor
    assert np.isnan(ip_alignment.rule_a_diagnostics(np.nan, 200.0)["a_minus_b_ms"])


IP_LOAD = ["magnetics_plasma_current", "magnetics_plasma_current_times", "efit_times"]


@pytest.mark.parametrize("breakdown_ms, label", [(300.0, "late"), (-40.0, "early")])
def test_loader_aligns_untouched_shots(tmp_path, starter, breakdown_ms, label):
    path = write_shot(tmp_path / f"d3d_shot_{label}.parquet", make_d3d_shot(ip_fix_needed=False, breakdown_ms=breakdown_ms))
    shot = io.load_shot(path, IP_LOAD)
    f = shot.fixes
    assert f["ip_fix_applied"] is False and f["ip_alignment_applied"] is True
    assert f["ip_breakdown_before_ms"] == pytest.approx(breakdown_ms, abs=0.5)
    assert f["ip_alignment_shift_ms"] == pytest.approx(TARGET - f["ip_breakdown_before_ms"])
    assert f["ip_breakdown_after_ms"] == pytest.approx(TARGET)
    assert (f["ip_alignment_shift_ms"] < 0) == (label == "late")
    axis = shot["magnetics_plasma_current_times"]
    assert ip_alignment.breakdown_time_ms(axis, shot["magnetics_plasma_current"]) == pytest.approx(TARGET)
    assert axis[0] == pytest.approx(D3D_MAGNETICS_ORIGIN_MS + f["ip_alignment_shift_ms"])


def test_loader_leaves_already_aligned_shot_nearly_still(tmp_path, starter):
    path = write_shot(tmp_path / "d3d_shot_aligned.parquet", make_d3d_shot(ip_fix_needed=False, breakdown_ms=TARGET))
    shot = io.load_shot(path, IP_LOAD)
    assert shot.fixes["ip_alignment_applied"] is True
    assert abs(shot.fixes["ip_alignment_shift_ms"]) <= 0.5          # within one 0.5 ms sample
    assert shot.fixes["ip_breakdown_after_ms"] == pytest.approx(TARGET)


def test_loader_keeps_organizer_fixed_shot_untouched(tmp_path, starter):
    path = write_shot(tmp_path / "d3d_shot_fixed.parquet", make_d3d_shot(ip_fix_needed=True, breakdown_ms=300.0))
    shot = io.load_shot(path, IP_LOAD)
    f = shot.fixes
    assert f["ip_fix_applied"] is True and f["ip_alignment_applied"] is False
    assert f["ip_alignment_shift_ms"] == 0.0
    assert f["ip_breakdown_before_ms"] == f["ip_breakdown_after_ms"] == pytest.approx(300.0, abs=0.5)
    assert shot["magnetics_plasma_current_times"][0] == pytest.approx(D3D_MAGNETICS_ORIGIN_MS)


def test_align_ip_switch_and_targets_untouched(tmp_path, starter):
    path = write_shot(tmp_path / "d3d_shot_switch.parquet", make_d3d_shot(ip_fix_needed=False, breakdown_ms=300.0))
    on = io.load_shot(path, IP_LOAD + ["efit_psirz", "efit_q95"])
    off = io.load_shot(path, IP_LOAD + ["efit_psirz", "efit_q95"], align_ip=False)
    assert off.fixes["ip_alignment_applied"] is False and off.fixes["ip_alignment_shift_ms"] == 0.0
    assert off["magnetics_plasma_current_times"][0] == pytest.approx(D3D_MAGNETICS_ORIGIN_MS)
    assert on["magnetics_plasma_current_times"][0] != off["magnetics_plasma_current_times"][0]
    assert np.array_equal(on["efit_psirz"], off["efit_psirz"]) and np.array_equal(on["efit_q95"], off["efit_q95"])
    assert np.array_equal(on["magnetics_plasma_current"], off["magnetics_plasma_current"])


def test_alignment_skipped_without_breakdown(tmp_path, starter):
    d = make_d3d_shot(ip_fix_needed=False)
    d["magnetics_plasma_current"][:] = 0.0
    path = write_shot(tmp_path / "d3d_shot_flat.parquet", d)
    shot = io.load_shot(path, IP_LOAD)
    assert shot.fixes["ip_alignment_applied"] is False and "ip_alignment_error" in shot.fixes


def test_breakdown_skips_nan_and_orders_merged_clock_axes():
    # MAST shots with merged clocks (F20): one sample in six is NaN on Ip, and an axis
    # assembled from two clocks need not be in time order.
    t = -100.0 + 0.5 * np.arange(1200)
    ip = np.where(t > 37.0, 800.0, 1.5)
    clean = ip_alignment.breakdown_time_ms(t, ip)
    holed = ip.copy()
    holed[::6] = np.nan
    assert ip_alignment.breakdown_time_ms(t, holed) == clean == 37.5
    # A NaN time stamp is skipped together with its sample, even when its current is high.
    t_nan = t.copy()
    t_nan[0] = np.nan
    spiked = holed.copy()
    spiked[0] = 900.0
    assert ip_alignment.breakdown_time_ms(t_nan, spiked) == clean
    # Two interleaved clocks stored one after the other give the same crossing.
    order = np.concatenate([np.arange(0, t.size, 2), np.arange(1, t.size, 2)])
    assert ip_alignment.breakdown_time_ms(t[order], holed[order]) == clean
    assert np.isnan(ip_alignment.breakdown_time_ms(t, np.full_like(t, np.nan)))
    with pytest.raises(ValueError):
        ip_alignment.breakdown_time_ms(t, ip[:-1])
