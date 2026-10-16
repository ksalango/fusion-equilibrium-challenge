"""Boundary diagnostic of predicted MAST flux maps, without any MAST label.

For every frame of a map set (a submission's MAST maps, or the physics-only control), the
organizers' boundary and axis functions are called unmodified with the machine set to MAST
(``synthetic_mast.mast_local_score_module``), and the boundary is characterised:

- Geometry of the boundary contour: extent, minor radius ``a``, geometric centre ``R0``, area,
  the scorer's kappa and triangularities (``derive_frame``), poloidal length.
- Axis: the scorer's O-point, its flux, every local maximum the scorer considered (the scorer
  takes the strongest), and the boundary flux (the mean of the map along the contour).
- Topology: every critical point of the map inside the grid (``critical_points``: minima of
  ``|grad psi|^2`` refined by Newton steps on a bicubic spline, classified by the Hessian), the
  saddle that defines the boundary, the separation of the two saddles inside the envelope
  closest in flux to the boundary, ``dpsi_sep = (psi_X2 - psi_X1) / (psi_axis - psi_boundary)``,
  and whether the upper or the lower saddle dominates. The scorer has no saddle finder; this
  is the one part not taken from it.
- Limiting: the scorer requires the boundary to stay inside its envelope mask (bilinear value
  above 0.5), so the 0.5 isoline of that mask is the only wall the scorer knows. Its points at
  small major radius form the centre column. ``machines.py`` holds no wall or centre column.
- Whether the boundary was found, why not (``fail_reason``), and whether it touches the grid
  edge.

All flux values are in the stored frame, Wb/rad. With ``phi = axis_sign psi`` the axis is a
maximum of ``phi``; normalised flux differences are taken on ``phi``.

Flags (``Thresholds``, loaded from a configuration file and frozen):

- F1 size: ``a`` or the area beyond the robust z-score bound of the map set, or outside the
  physical band.
- F2 topology: the defining saddle changes side (upper or lower) between consecutive frames of
  a shot, or, when the configuration sets ``dpsi_sep_max`` (version 1), ``|dpsi_sep|`` below it.
  Version 2 drops the ``dpsi_sep`` bound: on the quick run of 15 October 2026 it fired on about
  0.9 of the frames of every set, the control included, so it marks MAST's usual near-double-null
  state rather than an anomaly; ``dpsi_sep`` stays a per-frame column.
- F3 limiting: the boundary is limited on the centre column, touches the grid edge, or was not
  found (no closed surface).
- F4 arm disagreement: two submissions' boundaries on the same frame differ in ``a`` or
  ``R0`` beyond their bounds, or only one was found.
- F5 control agreement: F1, F2 or F3 in both submissions and in the control.

``overlap`` compares flagged shots with a list of known shots after the flags are computed; it
takes no threshold.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml
from scipy.interpolate import RectBivariateSpline
from scipy.spatial import cKDTree

from fusion_eq import dataset, frame_table as ft, starter_kit
from fusion_eq.coil_scales import phase_of

MACHINE = "MAST"
FLAGS = ("F1", "F2", "F3")
SET_NAMES = ("arm_g", "set_b", "control")
SUBMISSION_FIELDS = ("psirz", "q95", "betaN")
PHASE_LABELS = {"rising": "rising", "flat_top": "flat-top", "late": "falling"}
# Newton refinement of a critical point.
NEWTON_STEPS = 30
NEWTON_TOL_M = 1e-7
# A refined point farther than this many cells from its starting cell is discarded.
NEWTON_MAX_CELLS = 1.5
# Critical points closer than this many cells are one point.
DUPLICATE_CELLS = 0.5
# Local maxima within this many cells (Chebyshev) of the axis belong to the axis peak: float16
# storage makes neighbouring cells at the peak tie, and the scorer's test (at least every
# neighbour) marks each of them.
SAME_PEAK_CELLS = 2
# Wall points per metre of wall length for the distance search.
WALL_POINTS_PER_M = 500


# ------------------------------------------------------------------------------- thresholds
@dataclass(frozen=True)
class Thresholds:
    """The frozen thresholds of the flags (``configs/diag_mast_boundary_v<version>.yaml``).
    ``dpsi_sep_max`` is None when the configuration has no near-double-null bound."""

    axis_sign: float
    robust_z_max: float
    a_band_m: tuple[float, float]
    area_band_m2: tuple[float, float]
    dpsi_sep_max: float | None
    xpoint_flux_tol: float
    xpoint_contour_max_m: float
    wall_tol_m: float
    centre_column_max_r_m: float
    grid_edge_m: float
    da_max_m: float
    dr0_max_m: float
    flag_fraction: float
    coil_active_ka: float
    version: int = 1
    note: str = ""
    source: str = ""
    sha256: str = ""

    def record(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def analysis_record(self) -> dict:
        """The entries the per-frame analysis (``analyze_frame``) reads; two configurations that
        agree on them share one analysis and differ only in their flags."""
        return {k: getattr(self, k) for k in ANALYSIS_FIELDS}


# Entries read by ``analyze_frame``; the others act on its records only.
ANALYSIS_FIELDS = ("axis_sign", "xpoint_flux_tol", "xpoint_contour_max_m", "wall_tol_m", "centre_column_max_r_m",
                   "grid_edge_m")
# Entries a configuration may leave out.
OPTIONAL_FIELDS = ("dpsi_sep_max",)


_SECTIONS = {"size": ("robust_z_max", "a_band_m", "area_band_m2"),
             "topology": ("dpsi_sep_max", "xpoint_flux_tol", "xpoint_contour_max_m"),
             "limiting": ("wall_tol_m", "centre_column_max_r_m", "grid_edge_m"),
             "arms": ("da_max_m", "dr0_max_m"), "shots": ("flag_fraction", "coil_active_ka")}


def load_thresholds(path: Path | str) -> Thresholds:
    """Thresholds from a configuration file, with its path, SHA-256, version and note.

    Raises:
        KeyError: for a missing entry (other than ``OPTIONAL_FIELDS``) or an unknown one.
    """
    path = Path(path)
    text = path.read_bytes()
    raw = yaml.safe_load(text)
    values = {"axis_sign": float(raw["axis_sign"])}
    for section, keys in _SECTIONS.items():
        entries = dict(raw[section])
        unknown = set(entries) - set(keys)
        if unknown:
            raise KeyError(f"unknown entries in {section}: {sorted(unknown)}")
        for key in keys:
            if key not in entries and key in OPTIONAL_FIELDS:
                values[key] = None
                continue
            v = entries[key]
            values[key] = tuple(float(x) for x in v) if isinstance(v, list) else float(v)
    unknown = set(raw) - {"version", "note", "axis_sign", *_SECTIONS}
    if unknown:
        raise KeyError(f"unknown sections: {sorted(unknown)}")
    return Thresholds(**values, version=int(raw.get("version", 1)), note=str(raw.get("note", "")).strip(),
                      source=str(path), sha256=hashlib.sha256(text).hexdigest())


# ------------------------------------------------------------------------------ path guard
def refuse_label_paths(paths: Iterable[Path | str]) -> None:
    """Refuses any input that could hold a MAST equilibrium label: the labelled MAST sample
    shots (starter-kit ``parquet_data`` and the ``mast_samples`` split), any raw shot file
    (``*_shot_*.parquet``; this diagnostic reads none), and any Parquet file with a label
    column (``frame_table.is_label_column``).

    Raises:
        ValueError: naming the refused path.
    """
    samples = (starter_kit.starter_dir() / "parquet_data").resolve()
    sample_split = (Path(dataset.DEFAULT_DATA_ROOT) / dataset.SAMPLE_SPLIT).resolve()
    for p in paths:
        path = Path(p).resolve()
        for forbidden in (samples, sample_split):
            if path == forbidden or forbidden in path.parents:
                raise ValueError(f"{p}: labelled MAST sample shots are never read")
        if "_shot_" in path.name and path.suffix == ".parquet":
            raise ValueError(f"{p}: raw shot files are not read by this diagnostic")
        if path.suffix == ".parquet" and path.is_file():
            import pyarrow.parquet as pq

            labels = [c for c in pq.ParquetFile(path).schema_arrow.names if ft.is_label_column(c)]
            if labels:
                raise ValueError(f"{p}: holds label columns {labels[:5]}")
        if path.suffix == ".npz" and path.is_file():
            with np.load(path, allow_pickle=False) as f:
                bad = [k for k in f.files if not (k.startswith("shot_") and k.rsplit("_", 1)[-1] in SUBMISSION_FIELDS)]
            if bad:
                raise ValueError(f"{p}: not a submission file (keys {bad[:5]})")


# ------------------------------------------------------------------------------------ wall
@dataclass
class Wall:
    """The scorer's wall: the 0.5 isoline of the envelope mask, densely resampled.

    Attributes:
        points: ``(n, 2)`` (R, Z) in metres.
        centre_column: ``(n,)`` bool, points at or below the centre-column radius.
    """

    points: np.ndarray
    centre_column: np.ndarray

    def __post_init__(self) -> None:
        self._tree = cKDTree(self.points)
        cc = self.points[self.centre_column]
        self._cc_tree = cKDTree(cc) if len(cc) else None

    def distance(self, pts) -> tuple[np.ndarray, np.ndarray]:
        """Distance of points to the wall in metres and the index of the nearest wall point."""
        return self._tree.query(np.asarray(pts, dtype=np.float64))

    def centre_column_distance(self, pts) -> np.ndarray:
        if self._cc_tree is None:
            return np.full(len(pts), np.inf)
        return self._cc_tree.query(np.asarray(pts, dtype=np.float64))[0]


def scorer_wall(envelope, centre_column_max_r_m: float) -> Wall:
    """The wall of an envelope (``starter_kit.envelope_mask``): its longest closed 0.5 isoline."""
    sk = starter_kit.scoring()
    contours = [c for c in sk.contour.find_contours(envelope.mask_f, 0.5) if sk.contour.is_closed(c)]
    if not contours:
        raise ValueError("the envelope mask has no closed outline")
    outline = sk.contour.grid_to_physical(max(contours, key=len), envelope.grid_R, envelope.grid_Z)
    length = float(np.sum(np.hypot(*np.diff(outline, axis=0).T)))
    points = sk.contour.resample_closed(outline, max(64, int(length * WALL_POINTS_PER_M)))
    return Wall(points=points, centre_column=points[:, 0] <= centre_column_max_r_m)


# -------------------------------------------------------------------------- critical points
@dataclass(frozen=True)
class CriticalPoint:
    """A zero of the flux gradient: position in metres, ``phi`` there, and its kind
    (``saddle``, ``maximum`` or ``minimum``)."""

    r: float
    z: float
    phi: float
    kind: str


def critical_points(phi, grid_r, grid_z) -> list[CriticalPoint]:
    """Critical points of a map inside the grid.

    Starting cells are the interior local minima (eight neighbours) of ``|grad phi|^2``; each
    is refined by Newton steps on a bicubic spline of the map and kept when it converges within
    ``NEWTON_MAX_CELLS`` of its cell. Points within ``DUPLICATE_CELLS`` of each other are merged.
    """
    p = np.asarray(phi, dtype=np.float64)
    r = np.asarray(grid_r, dtype=np.float64)
    z = np.asarray(grid_z, dtype=np.float64)
    d_r, d_z = float(r[1] - r[0]), float(z[1] - z[0])
    spline = RectBivariateSpline(z, r, p, kx=3, ky=3)
    gz, gr = np.gradient(p, z, r)
    b2 = gz ** 2 + gr ** 2
    c = b2[1:-1, 1:-1]
    is_min = np.ones_like(c, dtype=bool)
    for dz in (-1, 0, 1):
        for dr in (-1, 0, 1):
            if dz or dr:
                is_min &= c <= b2[1 + dz:b2.shape[0] - 1 + dz, 1 + dr:b2.shape[1] - 1 + dr]
    found: list[CriticalPoint] = []
    for iz, ir in zip(*np.nonzero(is_min)):
        z0, r0 = float(z[iz + 1]), float(r[ir + 1])
        zz, rr = z0, r0
        ok = False
        for _ in range(NEWTON_STEPS):
            g = np.array([spline.ev(zz, rr, dx=1), spline.ev(zz, rr, dy=1)])
            h = np.array([[spline.ev(zz, rr, dx=2), spline.ev(zz, rr, dx=1, dy=1)],
                          [spline.ev(zz, rr, dx=1, dy=1), spline.ev(zz, rr, dy=2)]])
            try:
                step = np.linalg.solve(h, g)
            except np.linalg.LinAlgError:
                break
            zz, rr = zz - step[0], rr - step[1]
            if abs((zz - z0) / d_z) > NEWTON_MAX_CELLS or abs((rr - r0) / d_r) > NEWTON_MAX_CELLS:
                break
            if np.hypot(*step) < NEWTON_TOL_M:
                ok = True
                break
        if not ok or not (z[0] < zz < z[-1] and r[0] < rr < r[-1]):
            continue
        hzz, hzr, hrr = spline.ev(zz, rr, dx=2), spline.ev(zz, rr, dx=1, dy=1), spline.ev(zz, rr, dy=2)
        det = hzz * hrr - hzr ** 2
        kind = "saddle" if det < 0 else ("maximum" if hzz < 0 else "minimum")
        if any(abs((q.z - zz) / d_z) < DUPLICATE_CELLS and abs((q.r - rr) / d_r) < DUPLICATE_CELLS for q in found):
            continue
        found.append(CriticalPoint(r=float(rr), z=float(zz), phi=float(spline.ev(zz, rr)), kind=kind))
    return sorted(found, key=lambda q: (q.z, q.r))


def inside_mask(mask, grid_r, grid_z, r: float, z: float) -> bool:
    """Whether the mask cell nearest to (r, z) is set."""
    ir = int(np.argmin(np.abs(np.asarray(grid_r) - r)))
    iz = int(np.argmin(np.abs(np.asarray(grid_z) - z)))
    return bool(np.asarray(mask)[iz, ir])


# ---------------------------------------------------------------------------------- geometry
def contour_geometry(contour) -> dict:
    """Extent, minor radius, geometric centre, area (shoelace) and poloidal length of a closed
    (R, Z) contour, in metres and square metres."""
    c = np.asarray(contour, dtype=np.float64)
    rr, zz = c[:, 0], c[:, 1]
    closed = np.vstack([c, c[:1]])
    area = 0.5 * abs(float(np.sum(closed[:-1, 0] * closed[1:, 1] - closed[1:, 0] * closed[:-1, 1])))
    return {"r_min": float(rr.min()), "r_max": float(rr.max()), "z_min": float(zz.min()), "z_max": float(zz.max()),
            "a": 0.5 * float(rr.max() - rr.min()), "r0": 0.5 * float(rr.max() + rr.min()), "area": area,
            "length": float(np.sum(np.hypot(*np.diff(closed, axis=0).T)))}


def _bilinear(field, grid_r, grid_z, rq, zq) -> np.ndarray:
    from fusion_eq.scalars import _bilinear as bil

    return bil(field, grid_r, grid_z, rq, zq)


# ------------------------------------------------------------------------------- one frame
NAN_GEOMETRY = {k: np.nan for k in ("r_min", "r_max", "z_min", "z_max", "a", "r0", "area", "length")}


def analyze_frame(psi, grid_r, grid_z, envelope, wall: Wall, thr: Thresholds) -> tuple[dict, np.ndarray | None]:
    """The boundary record of one map, and its boundary contour ``(n, 2)`` (None when not found).

    Keys: ``found``, ``fail_reason`` (``""``, ``axis_on_rim`` when the scorer fell back to the
    global maximum, ``open_at_axis`` when not even a tight loop around the axis is closed inside
    the envelope, ``other``), the geometry of ``contour_geometry``, the scorer's ``R_axis``,
    ``Z_axis``, ``kappa``, ``tri_top``, ``tri_bot``, ``psi_axis``, ``psi_boundary``, ``n_maxima``
    (separate peaks among the local maxima the scorer considered: maxima within
    ``SAME_PEAK_CELLS`` of the axis count as the axis), ``second_max_gap`` (flux gap of the
    strongest other peak to the axis over ``phi_axis - phi_boundary``), ``second_max_r``/``z``, the saddles (``xpt_r``,
    ``xpt_z``, ``xpt_psi``, ``xpt_in_envelope``, lists), ``n_saddles_envelope``, ``boundary_type``
    (``diverted``, ``limited``, ``unresolved``, ``none``), ``def_xpt_r``/``z``/``side``,
    ``dpsi_sep``, ``dominant`` (``upper``/``lower``), ``wall_distance``, ``cc_distance``,
    ``limited_on`` (``centre_column``, ``wall`` or ``""``), ``edge_distance``, ``touches_edge``.
    """
    sk = starter_kit.scoring()
    from fusion_eq.synthetic_mast import mast_local_score_module

    ls = mast_local_score_module()
    r = np.asarray(grid_r, dtype=np.float64)
    z = np.asarray(grid_z, dtype=np.float64)
    p = np.asarray(psi, dtype=np.float64)
    s = float(thr.axis_sign)
    phi = s * p
    mask = envelope.mask_coarse if envelope.mask_coarse.any() else None
    contour, _ = ls.extract_lcfs_with_sign(p, r, z, MACHINE, envelope.mask_coarse, envelope.mask_f, n_iter=ls.N_ITER,
                                           n_points=ls.N_POINTS, axis_sign=s)
    derived = ls.derive_frame(p, r, z, MACHINE, envelope.mask_coarse, envelope.mask_f, contour=contour, with_li=False,
                              axis_sign=s)
    _, _, iz, ir, used_local = sk.o_point.find_o_point(phi, r, z, mask)
    phi_axis = float(phi[iz, ir])
    maxima = sk.o_point.local_maxima(phi, mask, 2)
    mz, mr = np.nonzero(maxima)
    other = np.maximum(np.abs(mz - iz), np.abs(mr - ir)) > SAME_PEAK_CELLS
    mz, mr = mz[other], mr[other]
    rec: dict = {"found": contour is not None, "fail_reason": "", "R_axis": derived["R_axis"], "Z_axis": derived["Z_axis"],
                 "kappa": derived["kappa"], "tri_top": derived["tri_top"], "tri_bot": derived["tri_bot"],
                 "psi_axis": s * phi_axis, "psi_boundary": np.nan, "n_maxima": int(bool(maxima.any())) + int(mz.size),
                 "second_max_gap": np.nan, "second_max_r": np.nan, "second_max_z": np.nan}
    second = None
    if mz.size:
        k2 = int(np.argmax(phi[mz, mr]))
        second = float(phi[mz[k2], mr[k2]])
        rec["second_max_r"], rec["second_max_z"] = float(r[mr[k2]]), float(z[mz[k2]])
    crit = critical_points(phi, r, z)
    saddles = [q for q in crit if q.kind == "saddle"]
    in_env = [inside_mask(envelope.mask_coarse, r, z, q.r, q.z) for q in saddles]
    rec.update({"xpt_r": [q.r for q in saddles], "xpt_z": [q.z for q in saddles], "xpt_psi": [s * q.phi for q in saddles],
                "xpt_in_envelope": in_env, "n_saddles_envelope": int(sum(in_env)), "boundary_type": "none",
                "def_xpt_r": np.nan, "def_xpt_z": np.nan, "def_xpt_side": "", "dpsi_sep": np.nan, "dominant": "",
                "wall_distance": np.nan, "cc_distance": np.nan, "limited_on": "", "edge_distance": np.nan,
                "touches_edge": False})
    if contour is None:
        rec.update(NAN_GEOMETRY)
        if not used_local:
            rec["fail_reason"] = "axis_on_rim"
        elif ls.lcfs._best_contour_at(phi, phi_axis - 1e-9, iz, ir, envelope.mask_f) is None:
            rec["fail_reason"] = "open_at_axis"
        else:
            rec["fail_reason"] = "other"
        return rec, None

    contour = np.asarray(contour, dtype=np.float64)
    rec.update(contour_geometry(contour))
    level = float(np.mean(_bilinear(phi, r, z, contour[:, 0], contour[:, 1])))
    rec["psi_boundary"] = s * level
    depth = phi_axis - level
    if second is not None and depth > 0:
        rec["second_max_gap"] = float((phi_axis - second) / depth)
    tree = cKDTree(contour)
    candidates = []
    for q, inside in zip(saddles, in_env):
        if not inside or depth <= 0:
            continue
        norm = (q.phi - level) / depth
        dist = float(tree.query([q.r, q.z])[0])
        candidates.append((abs(norm), norm, dist, q))
    candidates.sort(key=lambda t: t[0])
    if len(candidates) >= 2:
        rec["dpsi_sep"] = float(candidates[1][1] - candidates[0][1])
    if candidates:
        rec["dominant"] = "upper" if candidates[0][3].z > rec["Z_axis"] else "lower"
    defining = [t for t in candidates if t[0] <= thr.xpoint_flux_tol and t[2] <= thr.xpoint_contour_max_m]
    wall_d, wall_i = wall.distance(contour)
    k = int(np.argmin(wall_d))
    rec["wall_distance"] = float(wall_d[k])
    rec["cc_distance"] = float(np.min(wall.centre_column_distance(contour)))
    if defining:
        q = defining[0][3]
        rec.update({"boundary_type": "diverted", "def_xpt_r": q.r, "def_xpt_z": q.z,
                    "def_xpt_side": "upper" if q.z > rec["Z_axis"] else "lower"})
    elif rec["wall_distance"] <= thr.wall_tol_m:
        rec["boundary_type"] = "limited"
        rec["limited_on"] = "centre_column" if wall.centre_column[wall_i[k]] else "wall"
    else:
        rec["boundary_type"] = "unresolved"
    edge = np.min(np.stack([contour[:, 0] - r[0], r[-1] - contour[:, 0], contour[:, 1] - z[0], z[-1] - contour[:, 1]]))
    rec["edge_distance"] = float(edge)
    rec["touches_edge"] = bool(edge <= thr.grid_edge_m)
    return rec, contour


# ------------------------------------------------------------------------------------ flags
def robust_z(values) -> np.ndarray:
    """``(x - median) / (1.4826 MAD)`` over the finite values; NaN where not finite, 0 when the
    MAD is zero."""
    v = np.asarray(values, dtype=np.float64)
    ok = np.isfinite(v)
    out = np.full(v.shape, np.nan)
    if not ok.any():
        return out
    med = np.median(v[ok])
    mad = 1.4826 * np.median(np.abs(v[ok] - med))
    out[ok] = (v[ok] - med) / mad if mad > 0 else 0.0
    return out


def set_flags(frames: pd.DataFrame, thr: Thresholds) -> pd.DataFrame:
    """F1 to F3 of one map set's frames (columns of ``analyze_frame`` plus ``shot`` and
    ``frame``), with ``z_a``, ``z_area`` and ``xpt_switch`` beside them. Frames keep their order."""
    out = frames.copy()
    out["z_a"] = robust_z(out["a"])
    out["z_area"] = robust_z(out["area"])
    lo_a, hi_a = thr.a_band_m
    lo_s, hi_s = thr.area_band_m2
    a, area = out["a"].to_numpy(dtype=np.float64), out["area"].to_numpy(dtype=np.float64)
    found = out["found"].to_numpy(dtype=bool)
    with np.errstate(invalid="ignore"):
        out["F1"] = found & ((np.abs(out["z_a"]) > thr.robust_z_max) | (np.abs(out["z_area"]) > thr.robust_z_max)
                             | (a < lo_a) | (a > hi_a) | (area < lo_s) | (area > hi_s)).to_numpy(dtype=bool)
    side = out["def_xpt_side"].where(out["boundary_type"] == "diverted", "")
    order = np.lexsort((out["frame"].to_numpy(), out["shot"].astype(str).to_numpy()))
    switch = np.zeros(len(out), dtype=bool)
    sh, sd = out["shot"].astype(str).to_numpy()[order], side.to_numpy()[order]
    for k in range(1, len(order)):
        if sh[k] == sh[k - 1] and sd[k] and sd[k - 1] and sd[k] != sd[k - 1]:
            switch[order[k]] = True
    out["xpt_switch"] = switch
    near_dn = np.zeros(len(out), dtype=bool)
    if thr.dpsi_sep_max is not None:
        with np.errstate(invalid="ignore"):
            near_dn = np.abs(out["dpsi_sep"].to_numpy(dtype=np.float64)) < thr.dpsi_sep_max
    out["F2"] = found & (near_dn | switch)
    out["F3"] = (~found) | (out["limited_on"] == "centre_column").to_numpy() | out["touches_edge"].to_numpy(dtype=bool)
    out["any_flag"] = out["F1"] | out["F2"] | out["F3"]
    return out


def arm_flags(a: pd.DataFrame, b: pd.DataFrame, control: pd.DataFrame, thr: Thresholds) -> pd.DataFrame:
    """F4 and F5 per frame from the flagged frames of two submissions and the control, joined on
    ``shot`` and ``frame``."""
    keys = ["shot", "frame"]
    m = a[keys + ["found", "a", "r0", "any_flag"]].merge(b[keys + ["found", "a", "r0", "any_flag"]], on=keys,
                                                         suffixes=("_g", "_b"), validate="one_to_one")
    m = m.merge(control[keys + ["any_flag"]].rename(columns={"any_flag": "any_flag_c"}), on=keys, validate="one_to_one")
    with np.errstate(invalid="ignore"):
        differ = ((np.abs(m["a_g"] - m["a_b"]) > thr.da_max_m) | (np.abs(m["r0_g"] - m["r0_b"]) > thr.dr0_max_m)).to_numpy()
    m["F4"] = differ | (m["found_g"].to_numpy(dtype=bool) != m["found_b"].to_numpy(dtype=bool))
    m["F5"] = m["any_flag_g"] & m["any_flag_b"] & m["any_flag_c"]
    return m[keys + ["F4", "F5"]]


# --------------------------------------------------------------------------------- per shot
def phases(frames: pd.DataFrame) -> np.ndarray:
    """Phase label of every frame (``coil_scales.phase_of``: rising, flat-top, falling)."""
    return np.array([PHASE_LABELS[phase_of(float(x), int(f), int(n))]
                     for x, f, n in zip(frames["ip_over_peak"], frames["frame"], frames["n_frames"])], dtype=object)


def shot_flag_table(flags: pd.DataFrame, columns: Sequence[str], thr: Thresholds) -> pd.DataFrame:
    """Per shot, the share of frames carrying each flag column (``frac_<column>``) and whether
    the shot counts as flagged for it (``shot_<column>``, at least ``flag_fraction``)."""
    g = flags.groupby("shot", sort=True)[list(columns)].mean()
    out = g.add_prefix("frac_")
    for c in columns:
        out[f"shot_{c}"] = out[f"frac_{c}"] >= thr.flag_fraction
    return out.reset_index()


def shot_inputs(frames: pd.DataFrame, coil_columns: Sequence[str], thr: Thresholds) -> pd.DataFrame:
    """Shot-level input measures: mean and max Ip, max ip_rate_rel, solenoid swing, active
    P-coils and the range of each P-coil, from frame-table and feature rows."""
    rows = []
    for shot, g in frames.groupby("shot", sort=True):
        ip = g["ip__mean"].to_numpy(dtype=np.float64)
        sol = g["sol__mean"].to_numpy(dtype=np.float64)
        rec = {"shot": shot, "n_frames": len(g), "ip_mean_ka": float(np.nanmean(ip)), "ip_max_ka": float(np.nanmax(ip)),
               "ip_rate_rel_max": float(np.nanmax(g["ip_rate_rel"])) if g["ip_rate_rel"].notna().any() else np.nan,
               "sol_swing_ka": float(np.nanmax(sol) - np.nanmin(sol))}
        active = []
        for c in coil_columns:
            v = g[c].to_numpy(dtype=np.float64)
            name = c.replace("magnetics_", "").replace("_current__mean", "")
            rec[f"{name}_min"], rec[f"{name}_max"] = float(np.nanmin(v)), float(np.nanmax(v))
            if np.nanmax(np.abs(v)) > thr.coil_active_ka:
                active.append(name)
        rec["active_coils"] = ",".join(active)
        rows.append(rec)
    return pd.DataFrame(rows)


def overlap(shot_flags: pd.DataFrame, known: Iterable[str], flag_columns: Sequence[str]) -> pd.DataFrame:
    """For each shot-level flag column, the share of known shots it catches and how many other
    shots it raises. Reads flags only; takes no threshold."""
    known = {str(s) for s in known}
    in_known = shot_flags["shot"].astype(str).isin(known).to_numpy()
    missing = known - set(shot_flags["shot"].astype(str))
    rows = []
    for c in flag_columns:
        f = shot_flags[c].to_numpy(dtype=bool)
        rows.append({"flag": c, "known_in_table": int(in_known.sum()), "known_caught": int((f & in_known).sum()),
                     "known_caught_share": float((f & in_known).sum() / in_known.sum()) if in_known.any() else np.nan,
                     "other_raised": int((f & ~in_known).sum()), "known_not_in_table": len(missing)})
    return pd.DataFrame(rows)


KEY_PATTERN = r"^shot_(\d+)$"
# Board D_LCFS groups of the known shots (ordering and labels of the figures, not thresholds).
D_LCFS_GROUPS = ((0.30, 0.45, "cluster 0.3-0.45 m"), (0.60, 0.86, "cluster 0.6-0.86 m"))


def map_known(entries: pd.DataFrame, order: Sequence[str]) -> pd.DataFrame:
    """Known shots to MAST shot numbers.

    An entry is either a submission key ``shot_<index>`` (the writer's ``submission.shot_key``
    without the field: position ``index`` in the fold's writing order, ``order``) or a MAST
    shot number (zero-padded to ten digits).

    Args:
        entries: Rows with ``key`` and optionally ``d_lcfs``.
        order: Shot numbers of the submission file in writing order (the run record's
            ``shot_checks``).

    Returns:
        ``key``, ``shot`` (empty when unmatched), ``d_lcfs``, ``matched``, ``group``.

    Raises:
        ValueError: when no entry matches a shot of the submission.
    """
    order = [str(s) for s in order]
    known_ids = set(order)
    out = entries.copy()
    out["key"] = out["key"].astype(str).str.strip()
    if "d_lcfs" not in out:
        out["d_lcfs"] = np.nan
    index = pd.to_numeric(out["key"].str.extract(KEY_PATTERN, expand=False), errors="coerce")
    shots = []
    for key, i in zip(out["key"], index):
        if np.isfinite(i):
            shots.append(order[int(i)] if 0 <= int(i) < len(order) else "")
        elif key.isdigit():
            shots.append(key.zfill(10) if key.zfill(10) in known_ids else "")
        else:
            shots.append("")
    out["shot"] = shots
    out["matched"] = out["shot"] != ""
    if not out["matched"].any():
        raise ValueError(f"none of the {len(out)} known shots matches a shot of the submission: give submission keys "
                         f"(shot_0678) or MAST shot numbers (0035568738); first entries {out['key'].head(3).tolist()}")
    out["group"] = [d_lcfs_group(d) for d in pd.to_numeric(out["d_lcfs"], errors="coerce")]
    return out[["key", "shot", "d_lcfs", "matched", "group"]]


def d_lcfs_group(d: float) -> str:
    """Label of a board D_LCFS value among ``D_LCFS_GROUPS`` (``no d_lcfs``, ``other`` outside)."""
    if not np.isfinite(d):
        return "no d_lcfs"
    for lo, hi, label in D_LCFS_GROUPS:
        if lo <= d <= hi:
            return label
    return "other"


def dpsi_sep_distribution(frames: pd.DataFrame, by: Sequence[str] = ("set", "phase")) -> pd.DataFrame:
    """Distribution of ``|dpsi_sep|`` per group: frames, share with a value (two saddles in the
    envelope), and the 5th, 25th, 50th, 75th and 95th percentiles."""
    rows = []
    for key, g in frames.groupby(list(by), sort=True):
        v = np.abs(g["dpsi_sep"].to_numpy(dtype=np.float64))
        v = v[np.isfinite(v)]
        row = dict(zip(by, key if isinstance(key, tuple) else (key,)))
        row.update({"frames": len(g), "with_value": v.size / len(g) if len(g) else np.nan})
        for q in (5, 25, 50, 75, 95):
            row[f"p{q}"] = float(np.percentile(v, q)) if v.size else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------- map inputs
def read_submission_maps(npz, index: int) -> np.ndarray:
    """Stored-frame MAST maps of shot ``index`` of an open submission npz, as float64 (the
    scorer reads the float16 values the same way)."""
    return np.asarray(npz[f"shot_{int(index):04d}_psirz"], dtype=np.float64)


def control_maps(rows: pd.DataFrame, table, scales, objects) -> np.ndarray:
    """The physics-only control: coil flux of the coil model plus ``Ip phi_mean`` (``c = 0``),
    no remainder, through ``compose_map`` (``target_build.mean_current_maps``)."""
    from fusion_eq import target_build as tb

    return tb.mean_current_maps(MACHINE, rows, table, scales, objects, with_remainder=False)


def clip_counts(features: pd.DataFrame, scaler, columns: Sequence[str], bound: float) -> np.ndarray:
    """Per frame, how many of a model's scaled features lie beyond the clip bound."""
    x = scaler.transform(features, MACHINE, list(columns))
    return (np.abs(x) > float(bound)).sum(axis=1)
