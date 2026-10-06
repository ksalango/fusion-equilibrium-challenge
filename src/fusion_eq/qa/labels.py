"""Checks on the EFIT labels and time bases.

0.5 flux-map shape and indexing, 0.6 boundary contour, 0.8 ``magnetics_dsep``,
0.11 label and time-base sanity. The time-base part of 0.11 runs on every split;
the label parts run where labels exist (train, MAST sample shots).
"""

from __future__ import annotations

import numpy as np

from fusion_eq import starter_kit
from fusion_eq.io import Shot

EXPECTED_MAP_SHAPE = (65, 65)

# Time bases reported by 0.11, with the short name used in the metric columns.
TIME_BASES = {
    "efit_times": "efit",
    "magnetics_time": "magnetics",
    "magnetics_plasma_current_times": "ip",
    "magnetics_dsep_times": "dsep",
    "thomson_core_times": "ts_core",
    "thomson_edge_times": "ts_edge",
}

# Scalar labels reported by 0.11, with the short name used in the metric columns.
SCALAR_LABELS = {
    "efit_q95": "q95",
    "efit_beta_n": "betan",
    "efit_li": "li",
    "efit_r_axis": "raxis",
    "efit_z_axis": "zaxis",
}

DSEP_SENTINEL = -1.0


def sample_frames(n_frames: int, n_sample: int) -> np.ndarray:
    """Indices of up to ``n_sample`` frames spread evenly over ``n_frames``."""
    if n_frames <= 0:
        return np.zeros(0, dtype=int)
    return np.unique(np.round(np.linspace(0, n_frames - 1, min(n_sample, n_frames))).astype(int))


def _cell_distance(r_idx: int, z_idx: int, grid_r, grid_z, r_axis: float, z_axis: float) -> float:
    """Distance in grid cells between a grid node and a physical (R, Z) point."""
    step_r = float(np.median(np.diff(grid_r)))
    step_z = float(np.median(np.diff(grid_z)))
    return float(max(abs(grid_r[r_idx] - r_axis) / step_r, abs(grid_z[z_idx] - z_axis) / step_z))


def check_flux_map(shot: Shot, n_sample: int = 10) -> dict:
    """Check 0.5: shape of ``efit_psirz`` and its indexing against the stored axis.

    The map is read as ``[t, iZ, iR]``. Two axis estimates are compared with
    ``efit_r_axis`` / ``efit_z_axis`` on the sampled frames, both under
    ``axis_sign * psi`` (the axis is a minimum of psi on DIII-D, a maximum on MAST):

    - the scorer's own method (``fusion_scoring.o_point.find_o_point`` with the
      machine's envelope mask), the strongest interior local maximum; this is the
      estimate the fail rule uses;
    - the global extremum of the whole map, which walks to the grid border on shots
      where the coil flux at the edge exceeds the plasma flux; a border extremum is
      reported for review. The global distance is also computed under the transposed
      reading and for the opposite extremum, which tells those failure modes apart.

    Returns:
        ``psi_shape`` (string), ``psi_shape_ok`` (3-D, 65 x 65 maps, one per EFIT
        frame), ``psi_n_frames_checked``,
        ``psi_opoint_cell_dist_max``, ``psi_opoint_cell_dist_median``,
        ``psi_opoint_within_1cell_frac``, ``psi_opoint_used_local_frac`` (frames where
        an interior local maximum existed),
        ``psi_axis_cell_dist_max``, ``psi_axis_cell_dist_median``,
        ``psi_axis_within_1cell_frac`` (global extremum),
        ``psi_axis_on_border_frac`` (global extremum on the outermost ring),
        ``psi_axis_transposed_cell_dist_median``, ``psi_axis_opposite_cell_dist_median``.
    """
    psi = shot.get("efit_psirz")
    out: dict = {}
    if not isinstance(psi, np.ndarray):
        out["psi_shape"] = "ragged" if isinstance(psi, list) else "missing"
        out["psi_shape_ok"] = False
        return out
    out["psi_shape"] = str(psi.shape)
    n_frames = shot.n_frames if "efit_times" in shot else psi.shape[0]
    out["psi_shape_ok"] = bool(psi.ndim == 3 and psi.shape[1:] == EXPECTED_MAP_SHAPE
                               and psi.shape[0] == n_frames)
    if psi.ndim != 3 or psi.shape[1] != psi.shape[2]:
        return out

    grid_r = np.asarray(shot["efit_grid_R"], dtype=np.float64)
    grid_z = np.asarray(shot["efit_grid_Z"], dtype=np.float64)
    r_axis = np.asarray(shot["efit_r_axis"], dtype=np.float64)
    z_axis = np.asarray(shot["efit_z_axis"], dtype=np.float64)
    sign = shot.machine.axis_sign
    o_point = starter_kit.scoring().o_point
    mask = starter_kit.envelope_mask(shot.machine).mask_coarse if psi.shape[1:] == EXPECTED_MAP_SHAPE else None
    frames = sample_frames(min(psi.shape[0], r_axis.size), n_sample)
    dist, dist_t, dist_opp, border, dist_o, used_local = [], [], [], [], [], []
    for k in frames:
        phi = sign * psi[k]
        if not np.isfinite(phi).any():
            continue
        iz, ir = np.unravel_index(np.nanargmax(phi), phi.shape)
        iz_o, ir_o = np.unravel_index(np.nanargmin(phi), phi.shape)
        dist.append(_cell_distance(ir, iz, grid_r, grid_z, r_axis[k], z_axis[k]))
        dist_t.append(_cell_distance(iz, ir, grid_r, grid_z, r_axis[k], z_axis[k]))
        dist_opp.append(_cell_distance(ir_o, iz_o, grid_r, grid_z, r_axis[k], z_axis[k]))
        n = phi.shape[0]
        border.append(iz in (0, n - 1) or ir in (0, n - 1))
        _, _, iz_s, ir_s, local = o_point.find_o_point(phi, grid_r, grid_z, mask)
        dist_o.append(_cell_distance(ir_s, iz_s, grid_r, grid_z, r_axis[k], z_axis[k]))
        used_local.append(bool(local))
    out["psi_n_frames_checked"] = len(dist)
    if dist:
        d = np.asarray(dist)
        d_o = np.asarray(dist_o)
        out["psi_opoint_cell_dist_max"] = float(d_o.max())
        out["psi_opoint_cell_dist_median"] = float(np.median(d_o))
        out["psi_opoint_within_1cell_frac"] = float((d_o <= 1.0).mean())
        out["psi_opoint_used_local_frac"] = float(np.mean(used_local))
        out["psi_axis_cell_dist_max"] = float(d.max())
        out["psi_axis_cell_dist_median"] = float(np.median(d))
        out["psi_axis_within_1cell_frac"] = float((d <= 1.0).mean())
        out["psi_axis_on_border_frac"] = float(np.mean(border))
        out["psi_axis_transposed_cell_dist_median"] = float(np.median(dist_t))
        out["psi_axis_opposite_cell_dist_median"] = float(np.median(dist_opp))
    return out


def _contour_rows(values) -> list[np.ndarray]:
    """Per-frame rows of a contour column, whether rectangular or ragged."""
    if isinstance(values, np.ndarray):
        return [np.asarray(row, dtype=np.float64) for row in values]
    return [np.asarray(row, dtype=np.float64) for row in values]


def check_boundary(shot: Shot) -> dict:
    """Check 0.6: the stored LCFS contour and its padding.

    Returns:
        ``lcfs_n_min``, ``lcfs_n_max``, ``lcfs_n_integer`` (every count is a whole
        number), ``lcfs_width`` (padded width; -1 when it varies per frame),
        ``lcfs_n_frac_full`` (frames whose count equals the width),
        ``lcfs_pad_kind`` (``zero``, ``nan``, ``none``, ``mixed`` or ``other``),
        ``lcfs_pad_other_count`` (padding values that are neither 0 nor NaN),
        ``lcfs_valid_nan_frac`` (NaN inside the valid points),
        ``lcfs_closed_frac`` (first valid point equals last), ``lcfs_r_min/max``,
        ``lcfs_z_min/max`` over valid points, ``lcfs_rows_match_frames``.
    """
    n_valid = np.asarray(shot["efit_lcfs_n"], dtype=np.float64).ravel()
    rows_r = _contour_rows(shot["efit_lcfs_r"])
    rows_z = _contour_rows(shot["efit_lcfs_z"])
    out: dict = {
        "lcfs_n_min": float(np.nanmin(n_valid)) if n_valid.size else np.nan,
        "lcfs_n_max": float(np.nanmax(n_valid)) if n_valid.size else np.nan,
        "lcfs_n_integer": bool(np.all(np.isfinite(n_valid)) and np.all(n_valid == np.round(n_valid))),
        "lcfs_rows_match_frames": bool(len(rows_r) == n_valid.size == len(rows_z)),
    }
    widths = {row.size for row in rows_r}
    out["lcfs_width"] = int(next(iter(widths))) if len(widths) == 1 else -1

    pad_zero = pad_nan = pad_other = 0
    valid_total = valid_nan = 0
    closed = 0
    r_all, z_all = [], []
    n_frames = min(len(rows_r), len(rows_z), n_valid.size)
    for k in range(n_frames):
        n = int(n_valid[k]) if np.isfinite(n_valid[k]) else 0
        r, z = rows_r[k], rows_z[k]
        n = max(0, min(n, r.size, z.size))
        for pad in (r[n:], z[n:]):
            pad_nan += int(np.isnan(pad).sum())
            pad_zero += int((pad == 0).sum())
            pad_other += int((~np.isnan(pad) & (pad != 0)).sum())
        vr, vz = r[:n], z[:n]
        valid_total += 2 * n
        valid_nan += int(np.isnan(vr).sum() + np.isnan(vz).sum())
        if n >= 2 and np.isfinite([vr[0], vr[-1], vz[0], vz[-1]]).all():
            closed += int(np.isclose(vr[0], vr[-1]) and np.isclose(vz[0], vz[-1]))
        r_all.append(vr[np.isfinite(vr)])
        z_all.append(vz[np.isfinite(vz)])
    out["lcfs_n_frac_full"] = float(np.mean(n_valid == out["lcfs_width"])) if n_valid.size else np.nan
    total_pad = pad_zero + pad_nan + pad_other
    if total_pad == 0:
        kind = "none"
    elif pad_other:
        kind = "other" if (pad_zero + pad_nan) == 0 else "mixed"
    elif pad_zero and pad_nan:
        kind = "mixed"
    else:
        kind = "zero" if pad_zero else "nan"
    out["lcfs_pad_kind"] = kind
    out["lcfs_pad_other_count"] = pad_other
    out["lcfs_valid_nan_frac"] = valid_nan / valid_total if valid_total else np.nan
    out["lcfs_closed_frac"] = closed / n_frames if n_frames else np.nan
    r_cat = np.concatenate(r_all) if r_all else np.zeros(0)
    z_cat = np.concatenate(z_all) if z_all else np.zeros(0)
    out["lcfs_r_min"] = float(r_cat.min()) if r_cat.size else np.nan
    out["lcfs_r_max"] = float(r_cat.max()) if r_cat.size else np.nan
    out["lcfs_z_min"] = float(z_cat.min()) if z_cat.size else np.nan
    out["lcfs_z_max"] = float(z_cat.max()) if z_cat.size else np.nan
    return out


def check_dsep(shot: Shot) -> dict:
    """Check 0.8: distribution of ``magnetics_dsep`` and its sentinel.

    Returns:
        ``dsep_n``, ``dsep_n_sentinel`` (values exactly -1), ``dsep_n_nan``,
        ``dsep_min``, ``dsep_max``, ``dsep_median`` (excluding sentinel and NaN),
        ``dsep_frac_negative`` (same exclusion), ``dsep_times_equal_efit``
        (``magnetics_dsep_times`` matches ``efit_times`` to float32 precision).
    """
    d = np.asarray(shot["magnetics_dsep"], dtype=np.float64).ravel()
    # The column is float32; a stored -1 can arrive as -0.9999985 after the cast.
    sentinel = np.isclose(d, DSEP_SENTINEL, rtol=0, atol=1e-5)
    nan = np.isnan(d)
    rest = d[~sentinel & ~nan]
    out: dict = {
        "dsep_n": int(d.size),
        "dsep_n_sentinel": int(sentinel.sum()),
        "dsep_n_nan": int(nan.sum()),
        "dsep_min": float(rest.min()) if rest.size else np.nan,
        "dsep_max": float(rest.max()) if rest.size else np.nan,
        "dsep_median": float(np.median(rest)) if rest.size else np.nan,
        "dsep_frac_negative": float((rest < 0).mean()) if rest.size else np.nan,
    }
    times = shot.get("magnetics_dsep_times")
    efit = shot.get("efit_times")
    if times is not None and efit is not None:
        t = np.asarray(times, dtype=np.float64).ravel()
        e = np.asarray(efit, dtype=np.float64).ravel()
        out["dsep_times_equal_efit"] = bool(t.size == e.size and np.allclose(t, e, rtol=1e-6, atol=1e-3))
    return out


def check_label_sanity(shot: Shot) -> dict:
    """Check 0.11: time bases on every split, labels where present.

    Returns:
        Time bases: ``time_n_frames``, ``time_efit_increasing`` (strictly),
        ``time_efit_dt_max_ms``, and for each present base ``<b>`` in
        ``TIME_BASES``: ``time_<b>_n``, ``time_<b>_dt_median_ms``, ``time_<b>_min_ms``,
        ``time_<b>_max_ms``, ``time_<b>_n_nan``.
        Labels, when present: ``lab_psi_nan``, ``lab_psi_inf``, ``lab_psi_min``,
        ``lab_psi_max``, ``lab_psi_len_ok``; for each scalar ``<s>`` in
        ``SCALAR_LABELS``: ``lab_<s>_nan``, ``lab_<s>_inf``, ``lab_<s>_min``,
        ``lab_<s>_max``, ``lab_<s>_len_ok``; and ``lab_all_finite`` over the three
        scored labels psi, q95 and betaN.
    """
    out: dict = {}
    efit = np.asarray(shot["efit_times"], dtype=np.float64).ravel()
    out["time_n_frames"] = int(efit.size)
    d = np.diff(efit)
    out["time_efit_increasing"] = bool(efit.size > 0 and np.all(np.isfinite(efit)) and np.all(d > 0))
    out["time_efit_dt_max_ms"] = float(d.max()) if d.size else np.nan
    for column, short in TIME_BASES.items():
        v = shot.get(column)
        if v is None:
            continue
        v = np.asarray(v, dtype=np.float64).ravel()
        finite = v[np.isfinite(v)]
        out[f"time_{short}_n"] = int(v.size)
        out[f"time_{short}_n_nan"] = int(v.size - finite.size)
        out[f"time_{short}_min_ms"] = float(finite.min()) if finite.size else np.nan
        out[f"time_{short}_max_ms"] = float(finite.max()) if finite.size else np.nan
        out[f"time_{short}_dt_median_ms"] = float(np.median(np.diff(finite))) if finite.size > 1 else np.nan

    n_frames = efit.size
    scored_finite = []
    psi = shot.get("efit_psirz")
    if psi is not None:
        if isinstance(psi, np.ndarray):
            out["lab_psi_nan"] = int(np.isnan(psi).sum())
            out["lab_psi_inf"] = int(np.isinf(psi).sum())
            finite = psi[np.isfinite(psi)]
            out["lab_psi_min"] = float(finite.min()) if finite.size else np.nan
            out["lab_psi_max"] = float(finite.max()) if finite.size else np.nan
            out["lab_psi_len_ok"] = bool(psi.shape[0] == n_frames)
            scored_finite.append(out["lab_psi_nan"] == 0 and out["lab_psi_inf"] == 0)
        else:
            out["lab_psi_len_ok"] = False
            scored_finite.append(False)
    for column, short in SCALAR_LABELS.items():
        v = shot.get(column)
        if v is None:
            continue
        v = np.asarray(v, dtype=np.float64).ravel()
        out[f"lab_{short}_nan"] = int(np.isnan(v).sum())
        out[f"lab_{short}_inf"] = int(np.isinf(v).sum())
        finite = v[np.isfinite(v)]
        out[f"lab_{short}_min"] = float(finite.min()) if finite.size else np.nan
        out[f"lab_{short}_max"] = float(finite.max()) if finite.size else np.nan
        out[f"lab_{short}_len_ok"] = bool(v.size == n_frames)
        if column in ("efit_q95", "efit_beta_n"):
            scored_finite.append(out[f"lab_{short}_nan"] == 0 and out[f"lab_{short}_inf"] == 0)
    if scored_finite:
        out["lab_all_finite"] = bool(all(scored_finite))
    return out
