"""Tests for the physical frame: sign handling, signed inputs, coil flux of frames and the plasma target."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fusion_eq import coil_flux as cf, targets
from fusion_eq.machines import DIII_D

GRID_R = np.linspace(0.84, 2.54, 65)
GRID_Z = np.linspace(-1.6, 1.6, 65)


def small_table():
    geometry = cf.CoilGeometry(np.array(["F1A", "F1B", "E"]), np.array(["magnetics_F1A", "magnetics_F1B", "magnetics_ECOILA"]),
                               np.array([1.0, 1.0, 0.72]), np.array([1.2, -1.2, 0.0]), np.array([0.14, 0.14, 0.13]),
                               np.array([0.12, 0.12, 3.5]), np.zeros(3), np.zeros(3))
    return cf.build_green_table(DIII_D, geometry, GRID_R, GRID_Z, n_filaments=2)


def test_sign_rule_and_frame_conversions_roundtrip():
    rng = np.random.default_rng(0)
    psi = rng.normal(size=(4, 65, 65))
    coil = rng.normal(size=(4, 65, 65)) * 0.1
    ip = np.array([900.0, -800.0, 0.0, np.nan])
    assert targets.sign_rule_c("DIII-D") == -1.0 and targets.sign_of_ip(ip).tolist() == [1.0, -1.0, 1.0, 1.0]
    phys = targets.to_physical(psi, ip)
    assert np.allclose(phys[0], -psi[0]) and np.allclose(phys[1], psi[1])
    assert np.allclose(targets.to_stored(phys, ip), psi)
    t = targets.plasma_target(psi, coil, ip)
    assert np.allclose(t[0], -psi[0] - coil[0]) and np.allclose(t[1], -(psi[1] - coil[1]))
    assert np.allclose(targets.stored_from_target(t, coil, ip), psi)
    # One map at a time works too.
    assert np.allclose(targets.plasma_target(psi[1], coil[1], ip[1]), t[1])
    with pytest.raises(KeyError):
        targets.sign_rule_c("JET")


def test_mast_sign_rule_is_opposite_to_diii_d_on_positive_ip():
    # F71: on positive-Ip MAST shots the stored map is the physical flux itself.
    rng = np.random.default_rng(1)
    psi = rng.normal(size=(2, 65, 65))
    coil = rng.normal(size=(2, 65, 65)) * 0.1
    ip = np.array([600.0, 750.0])
    assert targets.sign_rule_c("MAST") == 1.0 and targets.sign_rule_c(DIII_D) == -1.0
    assert np.allclose(targets.to_physical(psi, ip, "MAST"), psi)
    assert np.allclose(targets.to_physical(psi, ip, "MAST"), -targets.to_physical(psi, ip, "DIII-D"))
    t = targets.plasma_target(psi, coil, ip, "MAST")
    assert np.allclose(t, psi - coil) and np.allclose(targets.stored_from_target(t, coil, ip, "MAST"), psi)


def test_signed_frames_flip_currents_only_on_negative_ip():
    frames = pd.DataFrame({"ip__mean": [800.0, -700.0], "sol__mean": [30.0, 20.0], "tf__mean": [-117.0, 117.0],
                           "magnetics_F1A__mean": [-50.0, -40.0], "ts_core_te_peak__interp": [1.0, 2.0],
                           "ip_over_peak": [0.9, 0.9]})
    out = targets.signed_frames(frames)
    assert out["ip__mean"].tolist() == [800.0, 700.0] and out["sol__mean"].tolist() == [30.0, -20.0]
    assert out["magnetics_F1A__mean"].tolist() == [-50.0, 40.0]
    assert out["tf__mean"].tolist() == [-117.0, 117.0] and out["ts_core_te_peak__interp"].tolist() == [1.0, 2.0]
    assert out["ip_over_peak"].tolist() == [0.9, 0.9] and frames["sol__mean"].tolist() == [30.0, 20.0]
    assert set(targets.signed_current_columns()) == {"ip__mean", "sol__mean", *[f"magnetics_F{i}{s}__mean" for i in range(1, 10) for s in "AB"]}


def test_coil_flux_of_frames_uses_the_table_order_and_scales():
    table = small_table()
    assert targets.current_columns_of(table) == ["magnetics_F1A__mean", "magnetics_F1B__mean", "sol__mean"]
    frames = pd.DataFrame({"ip__mean": [800.0, -800.0], "sol__mean": [30.0, -30.0],
                           "magnetics_F1A__mean": [100.0, -100.0], "magnetics_F1B__mean": [-60.0, 60.0]})
    record = cf.CoilScales("DIII-D", {"magnetics_F1A": 1.0, "magnetics_F1B": 1.0, "magnetics_ECOILA": 0.0}, flat_wb_per_ka=0.01)
    coil = targets.coil_flux_of_frames(table, frames, record)
    expected = table.flux(np.array([[100.0, -60.0, 30.0], [-100.0, 60.0, -30.0]]), record)
    assert coil.shape == (2, 65, 65) and np.allclose(coil, expected)
    assert np.allclose(coil[1], -coil[0])
    # Stored maps whose plasma target is +0.2 on both frames: psi = -coil - 0.2 at positive Ip
    # (psi_physical = -psi), psi = coil - 0.2 at negative Ip (psi_physical = psi).
    psi = np.stack([-coil[0] - 0.2, coil[1] - 0.2])
    t = targets.plasma_target_of_frames(psi, table, frames, record)
    assert np.allclose(t, 0.2)


def test_mean_shape_accumulator_divides_by_abs_ip_and_skips_low_current():
    acc = targets.MeanShapeAccumulator(shape=(2, 2), min_ip_ka=1.0)
    acc.add(np.array([[[2.0, 4.0], [6.0, 8.0]], [[1.0, 1.0], [1.0, 1.0]], [[9.0, 9.0], [9.0, 9.0]]]), [2.0, -1.0, 0.5])
    assert acc.n == 2 and np.allclose(acc.mean, [[1.0, 1.5], [2.0, 2.5]])      # |Ip| divides; 0.5 kA skipped
    with pytest.raises(ValueError):
        targets.MeanShapeAccumulator().mean
