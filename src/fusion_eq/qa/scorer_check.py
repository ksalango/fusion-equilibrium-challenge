"""Check 0.15: stored labels against the scorer's own functionals.

The scorer never reads the stored contour or the stored axis; it derives both from
the flux map with ``fusion_scoring``. This check runs those same functions on the
true flux map of a sample of frames and compares with the stored labels. The
differences bound how closely the stored labels can be matched by any model that
is scored through the flux map.
"""

from __future__ import annotations

import numpy as np

from fusion_eq import starter_kit
from fusion_eq.io import Shot
from fusion_eq.qa.labels import sample_frames


def _stored_contour(shot: Shot, k: int) -> np.ndarray | None:
    n = int(np.asarray(shot["efit_lcfs_n"])[k])
    r = np.asarray(shot["efit_lcfs_r"][k], dtype=np.float64)[:n]
    z = np.asarray(shot["efit_lcfs_z"][k], dtype=np.float64)[:n]
    keep = np.isfinite(r) & np.isfinite(z)
    if keep.sum() < 3:
        return None
    return np.column_stack([r[keep], z[keep]])


def _mean_nearest_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Mean over points of ``a`` of the distance to the nearest point of ``b``."""
    d = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=2))
    return float(d.min(axis=1).mean())


def _stat(values: list[float], fn) -> float:
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=np.float64)
    return float(fn(v)) if v.size else np.nan


def check_against_scorer(shot: Shot, n_sample: int = 5) -> dict:
    """Check 0.15: scorer-derived boundary and scalars versus the stored labels.

    For each sampled frame the scorer's ``extract_lcfs`` and ``derive_frame`` run
    on the true ``efit_psirz`` with the machine's envelope mask, as the reference
    builder does. Distances are in metres.

    Returns:
        ``sc_n_frames``, ``sc_lcfs_fail_count`` (frames where the scorer found no
        closed surface), ``sc_hausdorff_median/max`` (symmetric Hausdorff distance
        between the extracted 512-point contour and the stored contour),
        ``sc_mean_dist_median/max`` (mean distance from stored points to the
        extracted contour), ``sc_raxis_absdiff_median/max``,
        ``sc_zaxis_absdiff_median/max``, ``sc_li_diff_median`` (scorer minus stored,
        signed), ``sc_li_absdiff_max``, ``sc_li_ratio_median`` (scorer over stored),
        ``sc_kappa_absdiff_median`` when ``efit_elongation`` is present,
        ``sc_mask_grid_max_abs_diff_m`` (scorer mask grid against the shot grid).
    """
    sc = starter_kit.scoring()
    env = starter_kit.envelope_mask(shot.machine)
    machine = shot.machine.name
    psi = np.asarray(shot["efit_psirz"], dtype=np.float64)
    grid_r = np.asarray(shot["efit_grid_R"], dtype=np.float64)
    grid_z = np.asarray(shot["efit_grid_Z"], dtype=np.float64)
    r_axis = np.asarray(shot["efit_r_axis"], dtype=np.float64)
    z_axis = np.asarray(shot["efit_z_axis"], dtype=np.float64)
    li = np.asarray(shot["efit_li"], dtype=np.float64) if "efit_li" in shot else None
    kappa = np.asarray(shot["efit_elongation"], dtype=np.float64) if "efit_elongation" in shot else None

    out: dict = {
        "sc_mask_grid_max_abs_diff_m": float(max(
            np.abs(env.grid_R - grid_r).max(), np.abs(env.grid_Z - grid_z).max())),
    }
    frames = sample_frames(min(psi.shape[0], r_axis.size), n_sample)
    haus, mean_d, dr, dz, dli, rli, dkappa = [], [], [], [], [], [], []
    fails = 0
    for k in frames:
        contour = sc.lcfs.extract_lcfs(psi[k], grid_r, grid_z, machine, env.mask_coarse, env.mask_f)
        derived = sc.derive.derive_frame(psi[k], grid_r, grid_z, machine, env.mask_coarse,
                                         env.mask_f, contour=contour)
        dr.append(abs(derived["R_axis"] - r_axis[k]))
        dz.append(abs(derived["Z_axis"] - z_axis[k]))
        stored = _stored_contour(shot, k)
        if contour is None or stored is None:
            fails += 1
        else:
            haus.append(sc.contour.symmetric_hausdorff(contour, stored))
            mean_d.append(_mean_nearest_distance(stored, contour))
            if kappa is not None:
                dkappa.append(abs(derived["kappa"] - kappa[k]))
        if li is not None and np.isfinite(derived["li"]) and np.isfinite(li[k]) and li[k] != 0:
            dli.append(derived["li"] - li[k])
            rli.append(derived["li"] / li[k])
    out["sc_n_frames"] = int(frames.size)
    out["sc_lcfs_fail_count"] = fails
    out["sc_hausdorff_median"] = _stat(haus, np.median)
    out["sc_hausdorff_max"] = _stat(haus, np.max)
    out["sc_mean_dist_median"] = _stat(mean_d, np.median)
    out["sc_mean_dist_max"] = _stat(mean_d, np.max)
    out["sc_raxis_absdiff_median"] = _stat(dr, np.median)
    out["sc_raxis_absdiff_max"] = _stat(dr, np.max)
    out["sc_zaxis_absdiff_median"] = _stat(dz, np.median)
    out["sc_zaxis_absdiff_max"] = _stat(dz, np.max)
    out["sc_li_diff_median"] = _stat(dli, np.median)
    out["sc_li_absdiff_max"] = _stat([abs(x) for x in dli], np.max)
    out["sc_li_ratio_median"] = _stat(rli, np.median)
    if kappa is not None:
        out["sc_kappa_absdiff_median"] = _stat(dkappa, np.median)
    return out
