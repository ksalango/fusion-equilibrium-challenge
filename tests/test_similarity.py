"""Tests for fingerprints, pairwise distances, neighbours, groups and the B8a comparison."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fusion_eq import similarity as sim

SIGNALS = ("ip__mean", "sol__mean", "tf__mean")
GRID = np.arange(100.0, 1000.0 + 0.5, 100.0)      # 10 grid times


def frames_of(shot_id, times, values):
    """Frame-table rows of one shot from (frames,) times and (frames, 3) values."""
    out = pd.DataFrame(np.asarray(values, dtype=np.float64), columns=list(SIGNALS))
    out.insert(0, "time_ms", np.asarray(times, dtype=np.float64))
    out.insert(0, "shot", shot_id)
    return out


def smooth_shot(shot_id, t0=50.0, t1=1050.0, step=20.0, offset=0.0):
    t = np.arange(t0, t1 + 0.5, step)
    v = np.column_stack([800.0 + 100.0 * np.sin(t / 300.0) + offset, 20.0 - t / 100.0, np.full(t.size, -117.0)])
    return frames_of(shot_id, t, v)


# -------------------------------------------------------------------- interpolation
def test_interpolate_to_grid_exact_hits_gaps_and_outside():
    t = np.array([0.0, 100.0, 200.0, 500.0, 600.0])
    v = np.column_stack([t, 2 * t])
    out = sim.interpolate_to_grid(t, v, [50.0, 100.0, 350.0, 550.0, 700.0, -10.0], max_gap_ms=200.0)
    assert np.allclose(out[0], [50.0, 100.0])          # between frames
    assert np.allclose(out[1], [100.0, 200.0])         # exactly on a frame
    assert np.isnan(out[2]).all()                      # inside a 300 ms gap
    assert np.allclose(out[3], [550.0, 1100.0])
    assert np.isnan(out[4]).all() and np.isnan(out[5]).all()   # outside the window
    nan_v = v.copy()
    nan_v[1, 0] = np.nan
    out = sim.interpolate_to_grid(t, nan_v, [50.0])
    assert np.isnan(out[0, 0]) and out[0, 1] == 100.0
    with pytest.raises(ValueError):
        sim.interpolate_to_grid([0.0, 0.0], v[:2], GRID)


def test_shot_fingerprints_shape_order_and_presence():
    frames = pd.concat([smooth_shot("b"), smooth_shot("a", t0=50.0, t1=450.0)], ignore_index=True)
    fp = sim.shot_fingerprints(frames.sample(frac=1.0, random_state=0), SIGNALS, GRID)
    assert fp.shots.tolist() == ["a", "b"] and fp.values.shape == (2, GRID.size, 3)
    assert fp.n_times_present.tolist() == [4, 10]
    assert fp.index_of(["b"]).tolist() == [1]


# ------------------------------------------------------------------------ distances
def test_identical_shots_have_distance_zero_and_a_shifted_signal_the_expected_distance():
    frames = pd.concat([smooth_shot("a"), smooth_shot("b"), smooth_shot("c", offset=30.0)], ignore_index=True)
    fp = sim.shot_fingerprints(frames, SIGNALS, GRID)
    scales = np.array([10.0, 2.0, 5.0])
    d = sim.pairwise_distance(fp, scales)
    assert d.distance[0, 1] == pytest.approx(0.0, abs=1e-6) and d.distance[0, 0] == 0.0
    # One of three signals differs by 30 everywhere: scaled difference 3 on a third of the entries.
    assert d.distance[0, 2] == pytest.approx(3.0 / np.sqrt(3.0))
    assert np.allclose(d.distance, d.distance.T, equal_nan=True)
    assert d.n_common_times[0, 2] == GRID.size


def test_truncated_copy_has_zero_distance_over_the_overlap_and_none_when_too_short():
    full = smooth_shot("full")
    frames = pd.concat([full, smooth_shot("half", t0=50.0, t1=650.0), smooth_shot("tiny", t0=50.0, t1=250.0)],
                       ignore_index=True)
    fp = sim.shot_fingerprints(frames, SIGNALS, GRID)
    d = sim.pairwise_distance(fp, np.ones(3), min_common=5, min_common_fraction=0.5)
    i_full, i_half, i_tiny = fp.index_of(["full", "half", "tiny"])
    assert d.distance[i_full, i_half] == pytest.approx(0.0, abs=1e-6) and d.n_common_times[i_full, i_half] == 6
    assert np.isnan(d.distance[i_full, i_tiny]) and np.isnan(d.distance[i_half, i_tiny])   # 2 shared times
    assert np.isnan(d.distance[i_tiny, i_tiny])                                             # not fingerprinted


def test_pairwise_distance_matches_direct_computation_with_nan():
    rng = np.random.default_rng(1)
    n, t, s = 7, 12, 3
    values = rng.normal(size=(n, t, s))
    values[rng.random(values.shape) < 0.2] = np.nan
    values[2, :3] = np.nan                     # a late start
    values[5, 7:] = np.nan                     # an early end
    values[6, :] = np.nan                      # no fingerprint at all
    fp = sim.Fingerprints(np.array([f"s{i}" for i in range(n)], dtype=object), np.arange(t, dtype=float), SIGNALS, values)
    scales = np.array([0.5, 2.0, 1.5])
    d = sim.pairwise_distance(fp, scales, min_common=3, min_common_fraction=0.5, chunk=2)
    for i in range(n):
        for j in range(n):
            ref = sim.direct_distance(values[i], values[j], scales, min_common=3, min_common_fraction=0.5)
            if np.isnan(ref):
                assert np.isnan(d.distance[i, j]), (i, j)
            else:
                assert d.distance[i, j] == pytest.approx(ref, abs=1e-10), (i, j)
    assert np.isnan(d.distance[6]).all()
    with pytest.raises(ValueError):
        sim.pairwise_distance(fp, np.ones(2))


def test_signal_scales_iqr_and_absolute_for_tf():
    frames = pd.DataFrame({"ip__mean": np.arange(101.0), "sol__mean": np.arange(101.0) * 2,
                           "tf__mean": np.where(np.arange(101) % 2 == 0, 117.0, -117.0)})
    with pytest.raises(ValueError, match="tf__mean"):     # abs(tf) is constant: no range
        sim.signal_scales(frames, SIGNALS)
    frames.loc[:30, "tf__mean"] = 50.0
    scales = sim.signal_scales(frames, SIGNALS)
    assert scales[0] == pytest.approx(50.0) and scales[1] == pytest.approx(100.0)
    assert scales[2] == pytest.approx(67.0)                # IQR of |tf|: 117 - 50, not of the signed values


# ----------------------------------------------------------- neighbours and groups
def chain_distance():
    # A near B, B near C, A far from C; D isolated; E has no distances.
    d = np.array([[0.0, 0.05, 0.15, 0.9, np.nan],
                  [0.05, 0.0, 0.05, 0.9, np.nan],
                  [0.15, 0.05, 0.0, 0.9, np.nan],
                  [0.9, 0.9, 0.9, 0.0, np.nan],
                  [np.nan] * 5])
    return d


def test_nearest_neighbours_excludes_self_and_respects_candidates():
    d = chain_distance()
    idx, dist = sim.nearest_neighbours(d, k=2)
    assert idx[0].tolist() == [1, 2] and np.allclose(dist[0], [0.05, 0.15])
    assert idx[4].tolist() == [-1, -1] and np.isnan(dist[4]).all()
    idx, dist = sim.nearest_neighbours(d, k=3, rows=[0], columns=[0, 2, 3])
    assert idx[0].tolist() == [2, 3, -1] and np.allclose(dist[0, :2], [0.15, 0.9]) and np.isnan(dist[0, 2])


def test_group_shots_complete_linkage_does_not_chain():
    labels = sim.group_shots(chain_distance(), threshold=0.10)
    assert labels[0] == labels[1] and labels[1] != labels[2]      # A-B together, C apart (A-C = 0.15)
    assert len(set(labels[[2, 3, 4]])) == 3 and labels.max() == 3
    loose = sim.group_shots(chain_distance(), threshold=0.20)
    assert loose[0] == loose[1] == loose[2] and loose[3] != loose[0] and loose[4] != loose[0]
    # Strictly below the threshold: a pair at exactly the threshold is not a near-repeat.
    assert sim.group_shots(chain_distance(), threshold=0.05)[0] != sim.group_shots(chain_distance(), threshold=0.05)[1]
    assert sim.group_shots(np.zeros((1, 1)), 0.1).tolist() == [0] and sim.group_shots(np.zeros((0, 0)), 0.1).size == 0


def test_group_column_names():
    assert [sim.group_column(t) for t in sim.THRESHOLDS] == ["group_005", "group_010", "group_020"]


def test_has_near_repeat_in_excludes_self():
    d = sim.PairwiseDistance(np.array(["a", "b", "c", "d", "e"], dtype=object), chain_distance(), np.zeros((5, 5), int))
    flags = sim.has_near_repeat_in(d, ["a", "c", "d", "e"], ["a", "b"], threshold=0.10)
    assert flags.tolist() == [True, True, False, False]
    assert sim.has_near_repeat_in(d, ["a"], ["a"], 0.10).tolist() == [False]


# ------------------------------------------------------------------------- B8a
def test_flat_top_medians_and_compare_distributions():
    frames = pd.DataFrame({"shot": ["a"] * 4 + ["b"] * 3, "ip_over_peak": [0.5, 0.95, 1.0, 0.95, 0.5, 0.6, 0.7],
                           "x": [1.0, 2.0, 4.0, 6.0, 9.0, 9.0, 9.0]})
    med = sim.flat_top_medians(frames, ["x"])
    assert med.loc["a", "x"] == 4.0 and np.isnan(med.loc["b", "x"])
    rng = np.random.default_rng(0)
    train = pd.DataFrame({"same": rng.normal(size=400), "shifted": rng.normal(size=400)})
    test = pd.DataFrame({"same": rng.normal(size=200), "shifted": rng.normal(size=200) + 2.0})
    out = sim.compare_distributions(train, test)
    assert out["quantity"].tolist() == ["shifted", "same"] and out["ks"].iloc[0] > 0.5 > out["ks"].iloc[1]
    assert out["median_diff_iqr"].iloc[0] == pytest.approx(2.0 / 1.35, rel=0.2)
    assert out["n_train"].iloc[0] == 400 and out["n_test"].iloc[0] == 200
