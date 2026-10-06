"""Synthetic one-row shot files for the tests.

The builders return a plain dict of column name to value that mirrors the layout of
the challenge files (see ``docs/data_dictionary.md``) at a small size. Tests modify
the dict to create failing cases and then write it with ``write_shot``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from fusion_eq import starter_kit

N_GRID = 65

# Shipped (wrong) DIII-D Ip axis origin and the magnetics origin, in ms, as in the data.
D3D_IP_SHIPPED_ORIGIN_MS = -858.19
D3D_MAGNETICS_ORIGIN_MS = -4010.75
D3D_F_COILS = [f"F{i}{s}" for i in range(1, 10) for s in "AB"]
MAST_P_COILS = [f"p{i}{s}" for i in range(2, 7) for s in "lu"]


def gaussian_psi(grid_r, grid_z, r0, z0, width_r, width_z, amplitude, offset):
    """Flux map [iZ, iR] with one extremum of the given sign at (r0, z0)."""
    rr, zz = np.meshgrid(grid_r, grid_z)
    return amplitude * np.exp(-((rr - r0) / width_r) ** 2 - ((zz - z0) / width_z) ** 2) + offset


def closed_contour(r0, z0, a, b, n):
    """Ellipse of n distinct points plus the repeated first point (closed)."""
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = r0 + a * np.cos(theta)
    z = z0 + b * np.sin(theta)
    return np.append(r, r[0]), np.append(z, z[0])


def padded(rows: list[np.ndarray], width: int, pad_value: float) -> np.ndarray:
    out = np.full((len(rows), width), pad_value, dtype=np.float64)
    for i, row in enumerate(rows):
        out[i, : row.size] = row
    return out


def make_d3d_shot(n_frames: int = 12, ip_fix_needed: bool = True, breakdown_ms: float = 10.0) -> dict:
    """Synthetic DIII-D shot with labels.

    Args:
        n_frames: Number of EFIT frames (20 ms cadence from 300 ms).
        ip_fix_needed: When True the shipped Ip axis is the wrong one (as on 69
            percent of shots); when False it already starts at the magnetics origin.
        breakdown_ms: Time on the correct axis at which Ip rises from the noise floor.
            The solenoid swing-down is always at 0 ms, so this also sets the lag.
    """
    efit_times = 300.0 + 20.0 * np.arange(n_frames)
    grid_r = np.linspace(0.84, 2.54, N_GRID)
    grid_z = np.linspace(-1.6, 1.6, N_GRID)
    # Off-centre axis, so that a transposed map lands far from the stored axis.
    r0, z0 = 1.85, -0.30
    psi = np.stack([gaussian_psi(grid_r, grid_z, r0, z0, 0.35, 0.6, -0.5 - 0.01 * k, 0.1)
                    for k in range(n_frames)])
    contour_r, contour_z = closed_contour(r0, z0, 0.45, 0.8, 40)
    width = 44
    magnetics_time = D3D_MAGNETICS_ORIGIN_MS + 2.0 * np.arange(12_001)      # to 20 s, 2 ms step
    in_shot = (magnetics_time > 0) & (magnetics_time < 5000)
    ip_fixed_axis = D3D_MAGNETICS_ORIGIN_MS + 0.5 * np.arange(30_719)
    ip = np.where((ip_fixed_axis > breakdown_ms) & (ip_fixed_axis < 4500), 1000.0, 0.0).astype(np.float32)
    ip += np.float32(2.0) * (np.arange(ip.size) % 2).astype(np.float32)      # 2 kA noise floor
    ip_times = (D3D_IP_SHIPPED_ORIGIN_MS + 0.5 * np.arange(30_719)) if ip_fix_needed else ip_fixed_axis
    # Solenoid: pre-charge ramp from -800 ms to a 40 kA plateau, swing-down starting at 0 ms
    # (falling 1.5 kA/ms to -35 kA by 50 ms), hold, then zero after the shot.
    ecoila = np.zeros_like(magnetics_time)
    ramp = (magnetics_time >= -800) & (magnetics_time < -100)
    ecoila[ramp] = 40.0 * (magnetics_time[ramp] + 800) / 700
    ecoila[(magnetics_time >= -100) & (magnetics_time <= 0)] = 40.0
    fall = (magnetics_time > 0) & (magnetics_time <= 50)
    ecoila[fall] = 40.0 - 1.5 * magnetics_time[fall]
    ecoila[(magnetics_time > 50) & (magnetics_time < 5000)] = -35.0

    ts_core_times = -1500.0 + 4.0 * np.arange(3600)
    n_core = 6
    te_core = np.full((ts_core_times.size, n_core), 1000.0)
    te_core[::4, :] = 0.0                       # every fourth slice has no laser data
    te_core[:, 5] = 0.0                         # one dead channel
    ne_core = np.where(te_core > 0, 3e19, 0.0)
    ts_edge_times = -1500.0 + 10.0 * np.arange(1440)
    n_edge = 3
    te_edge = np.full((ts_edge_times.size, n_edge), 200.0)
    ne_edge = np.full((ts_edge_times.size, n_edge), 1e19)

    coil_names = D3D_F_COILS + ["ECOILA"]
    shot = {
        "source": "DIII-D",
        "efit_times": efit_times,
        "efit_grid_R": grid_r,
        "efit_grid_Z": grid_z,
        "efit_psirz": psi,
        "efit_q95": 4.0 + 0.1 * np.arange(n_frames),
        "efit_beta_n": 0.3 + 0.01 * np.arange(n_frames),
        "efit_li": np.full(n_frames, 1.0),
        "efit_r_axis": np.full(n_frames, r0),
        "efit_z_axis": np.full(n_frames, z0),
        "efit_lcfs_n": np.full(n_frames, float(contour_r.size)),
        "efit_lcfs_r": padded([contour_r] * n_frames, width, 0.0),
        "efit_lcfs_z": padded([contour_z] * n_frames, width, 0.0),
        "magnetics_dsep": np.full(n_frames, 0.01, dtype=np.float32),
        "magnetics_dsep_times": efit_times.astype(np.float32),
        "magnetics_time": magnetics_time.astype(np.float32),
        "magnetics_plasma_current": ip,
        "magnetics_plasma_current_times": ip_times.astype(np.float32),
        "magnetics_ECOILA": ecoila,
        "magnetics_bcoil": np.where(in_shot, -117.0, 0.0),
        "thomson_core_times": ts_core_times,
        "thomson_core_Te": te_core,
        "thomson_core_ne": ne_core,
        "thomson_core_R": np.full(n_core, 1.94),
        "thomson_edge_times": ts_edge_times,
        "thomson_edge_Te": te_edge,
        "thomson_edge_ne": ne_edge,
        "thomson_edge_spatial": np.linspace(-0.09, -0.04, n_edge),
        "thomson_chord_name": np.array([f"TS_core_{i}" for i in range(n_core)]
                                       + [f"TS_tangential_{i}" for i in range(n_edge)], dtype=object),
        "thomson_chord_R": np.concatenate([np.full(n_core, 1.94), np.linspace(1.7, 2.2, n_edge)]),
        "thomson_chord_Z": np.concatenate([np.linspace(0.03, 0.8, n_core), np.linspace(-0.09, -0.04, n_edge)]),
        "coil_name": np.array(coil_names, dtype=object),
        "coil_input_column": np.array([f"magnetics_{n}" for n in coil_names], dtype=object),
        "coil_R": np.linspace(0.72, 2.61, len(coil_names)),
        "coil_Z": np.linspace(-1.6, 1.6, len(coil_names)),
        "coil_width": np.full(len(coil_names), 0.1),
        "coil_height": np.full(len(coil_names), 0.2),
        "coil_angle1": np.zeros(len(coil_names)),
        "coil_angle2": np.zeros(len(coil_names)),
    }
    for i, name in enumerate(D3D_F_COILS):
        shot[f"magnetics_{name}"] = np.where(in_shot, 100.0 + 10.0 * i, 0.0)
    return shot


def make_mast_shot(n_frames: int = 8) -> dict:
    """Synthetic MAST shot with labels, including the 131-channel Thomson layout."""
    efit_times = (65.0 + 5.0 * np.arange(n_frames)).astype(np.float32)
    grid_r = np.linspace(0.06, 2.0, N_GRID)
    grid_z = np.linspace(-2.0, 2.0, N_GRID)
    r0, z0 = 0.85, 0.01
    psi = np.stack([gaussian_psi(grid_r, grid_z, r0, z0, 0.3, 0.7, 0.06, -0.02) for _ in range(n_frames)])
    contour_r, contour_z = closed_contour(r0, z0, 0.5, 1.0, 60)
    width = 64
    magnetics_time = (-2000.0 + 2.0 * np.arange(3000)).astype(np.float32)
    in_shot = (magnetics_time > 0) & (magnetics_time < 400)

    n_core_data = 131                                   # channel 0 is the ghost channel
    ts_core_times = np.concatenate([np.array([0.0, 2.0, 4.0]) + 21.33 * k for k in range(13)])
    te_core = np.full((ts_core_times.size, n_core_data), 500.0)
    te_core[:, 0] = np.nan
    te_core[::3, 10] = np.nan
    ne_core = np.where(np.isfinite(te_core), 1e20, np.nan)
    n_edge = 16
    ts_edge_times = ts_core_times[:-1]
    te_edge = np.full((ts_edge_times.size, n_edge), 50.0)
    ne_edge = np.full((ts_edge_times.size, n_edge), 1e19)

    coil_names = [f"sol_{i}" for i in range(20)] + [f"{p}_{i}" for p in MAST_P_COILS for i in range(4)]
    coil_inputs = ["magnetics_sol_current"] * 20 + [f"magnetics_{p}_current" for p in MAST_P_COILS for _ in range(4)]
    dsep = np.full(n_frames, 0.002, dtype=np.float32)
    dsep[1] = -1.0
    dsep[2] = -1.0
    shot = {
        "source": "MAST",
        "efit_times": efit_times,
        "efit_grid_R": grid_r,
        "efit_grid_Z": grid_z,
        "efit_psirz": psi,
        "efit_q95": np.full(n_frames, 6.0),
        "efit_beta_n": np.full(n_frames, 0.8),
        "efit_li": np.full(n_frames, 0.9),
        "efit_r_axis": np.full(n_frames, r0),
        "efit_z_axis": np.full(n_frames, z0),
        "efit_elongation": np.full(n_frames, 2.0),
        "efit_lcfs_n": np.full(n_frames, float(contour_r.size)),
        "efit_lcfs_r": padded([contour_r] * n_frames, width, np.nan),
        "efit_lcfs_z": padded([contour_z] * n_frames, width, np.nan),
        "magnetics_dsep": dsep,
        "magnetics_dsep_times": efit_times,
        "magnetics_time": magnetics_time,
        "magnetics_plasma_current": np.where(in_shot, 800.0, 0.0),
        "magnetics_sol_current": np.where(in_shot, -40.0, 0.0),
        "magnetics_tf_current": np.where(in_shot, -85.0, 0.0),
        "magnetics_efps_current": np.where(in_shot, 10.0, 0.0),
        "thomson_core_times": ts_core_times,
        "thomson_core_Te": te_core,
        "thomson_core_ne": ne_core,
        "thomson_core_R": np.linspace(0.23, 1.52, n_core_data - 1),
        "thomson_edge_times": ts_edge_times,
        "thomson_edge_Te": te_edge,
        "thomson_edge_ne": ne_edge,
        "thomson_edge_spatial": np.linspace(1.33, 1.49, n_edge),
        "thomson_chord_name": np.array([f"TS_core_{i}" for i in range(n_core_data - 1)]
                                       + [f"TS_edge_{i}" for i in range(n_edge)], dtype=object),
        "thomson_chord_R": np.concatenate([np.linspace(0.23, 1.52, n_core_data - 1), np.linspace(1.33, 1.49, n_edge)]),
        "thomson_chord_Z": np.zeros(n_core_data - 1 + n_edge),
        "coil_name": np.array(coil_names, dtype=object),
        "coil_input_column": np.array(coil_inputs, dtype=object),
        "coil_R": np.linspace(0.12, 1.7, len(coil_names)),
        "coil_Z": np.linspace(-1.8, 1.8, len(coil_names)),
        "coil_width": np.full(len(coil_names), 0.02),
        "coil_height": np.full(len(coil_names), 0.03),
        "coil_angle1": np.zeros(len(coil_names)),
        "coil_angle2": np.zeros(len(coil_names)),
    }
    for i, p in enumerate(MAST_P_COILS):
        shot[f"magnetics_{p}_current"] = np.where(in_shot, 50.0 + 20.0 * i, 0.0)
    return shot


def drop_labels(shot: dict) -> dict:
    """Inputs-only copy of a shot, as the test splits ship it."""
    labels = {"efit_psirz", "efit_q95", "efit_beta_n", "efit_li", "efit_r_axis", "efit_z_axis",
              "efit_lcfs_n", "efit_lcfs_r", "efit_lcfs_z", "magnetics_dsep", "magnetics_dsep_times",
              "efit_elongation"}
    return {k: v for k, v in shot.items() if k not in labels}


def _to_arrow(value) -> pa.Array:
    if isinstance(value, str):
        return pa.array([value])
    arr = np.asarray(value)
    if arr.dtype == object:
        return pa.array([arr.tolist()], type=pa.list_(pa.string()))
    return pa.array([arr.tolist()], type=_list_type(arr))


def _list_type(arr: np.ndarray) -> pa.DataType:
    inner = pa.float32() if arr.dtype == np.float32 else pa.float64()
    t = inner
    for _ in range(arr.ndim):
        t = pa.list_(t)
    return t


def write_shot(path: Path, shot: dict) -> Path:
    """Writes a one-row shot file with nested array cells."""
    table = pa.table({name: _to_arrow(value) for name, value in shot.items()})
    pq.write_table(table, path)
    return path


@pytest.fixture
def d3d_file(tmp_path) -> Path:
    return write_shot(tmp_path / "d3d_shot_100001.parquet", make_d3d_shot())


@pytest.fixture
def mast_file(tmp_path) -> Path:
    return write_shot(tmp_path / "mast_shot_200001.parquet", make_mast_shot())


@pytest.fixture
def starter():
    """Skips the test when the starter kit is not checked out next to the repo."""
    if not starter_kit.is_available():
        pytest.skip("starter kit not available")
    return starter_kit
