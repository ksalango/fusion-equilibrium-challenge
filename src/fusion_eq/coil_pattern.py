"""Coil pattern: the calculated coil flux sampled at fixed points in normalized coordinates.

The sample points are given in coordinates normalized by each machine's reference radii,
``x = (R - R0) / a0`` and ``y = Z / a0``, so the same layout serves DIII-D and MAST. The
coil flux at the points comes from a point table (``coil_flux.PointTable``) built with the
grid tables' code and filament subdivision, so a frame's values are one matrix product.
Each point's value is taken relative to the reference point (0, 0) and divided by
``mu0 R0 Ip``, Ip signed and in amperes:

    coil_pt_i = (psi_coil(P_i) - psi_coil(P_ref)) / (mu0 R0 Ip)

which carries no units, keeps its sign for both current directions, and drops any flux
that is the same at every point (the DIII-D flat solenoid term).

The named terms fit a polynomial of total order 1 to 3 in x and y to the point values by
least squares, with one fixed matrix that depends on the layout only:
``coil_pt(x, y) ~ sum c_mn x^m y^n``, ``1 <= m + n <= 3``.

Layout rule (``docs/FEATURES.md``, check X4): every point lies inside the flux grid and at
least ``min_clearance_m`` from the outline of every coil element on both machines, and the
layout is symmetric about y = 0, so that the mirror of a frame is a permutation of its
points.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from fusion_eq import dataset, io
from fusion_eq.coil_flux import (A_PER_KA, GEOMETRY_COLUMNS, MU0, CoilGeometry, GreenTable, PointTable,
                                 build_point_table, element_outline)
from fusion_eq.machines import MachineConfig, machine_config

REFERENCE_POINT = (0.0, 0.0)
DEFAULT_MIN_CLEARANCE_M = 0.15
# Polynomial terms (m, n) of x^m y^n, in the order of the named features.
TERMS = ((1, 0), (0, 1), (2, 0), (1, 1), (0, 2), (3, 0), (2, 1), (1, 2), (0, 3))
TERM_NAMES = tuple(f"c{m}{n}" for m, n in TERMS)
# Split whose first shot supplies the coil geometry of each machine for the point tables.
GEOMETRY_SPLITS = {"DIII-D": "diii_d_public_test", "MAST": "mast_public_test"}
DEFAULT_POINT_CACHE = dataset.DEFAULT_PROCESSED_ROOT / "green"


# ------------------------------------------------------------------------------ layout
@dataclass(frozen=True)
class Layout:
    """Sample points in normalized coordinates, the reference point excluded.

    Attributes:
        points: ``(n, 2)`` array of (x, y), in the order of the ``coil_pt_<i>`` columns.
        min_clearance_m: Smallest allowed distance in metres from a point to a coil element.
    """

    points: np.ndarray
    min_clearance_m: float = DEFAULT_MIN_CLEARANCE_M

    def __post_init__(self) -> None:
        pts = np.asarray(self.points, dtype=np.float64).reshape(-1, 2)
        if len({tuple(p) for p in pts.tolist()}) != len(pts):
            raise ValueError("the layout repeats a point")
        if any(tuple(p) == REFERENCE_POINT for p in pts.tolist()):
            raise ValueError("the reference point (0, 0) is not a sample point")
        object.__setattr__(self, "points", pts)

    @classmethod
    def from_config(cls, d: dict) -> "Layout":
        """From the ``layout`` entry of a feature configuration: ``points`` as [x, y] pairs
        and an optional ``min_clearance_m``."""
        return cls(np.asarray(d["points"], dtype=np.float64), float(d.get("min_clearance_m", DEFAULT_MIN_CLEARANCE_M)))

    @property
    def n_points(self) -> int:
        return int(self.points.shape[0])

    def column_names(self, prefix: str = "coil_pt") -> list[str]:
        """Feature column of each point: ``<prefix>_00``, ``<prefix>_01``, ..."""
        return [f"{prefix}_{i:02d}" for i in range(self.n_points)]

    def mirror_index(self) -> np.ndarray:
        """Index of each point's mirror partner (x, -y); a point on y = 0 is its own.

        Raises:
            ValueError: when a point has no partner, that is when the layout is not
                symmetric about y = 0.
        """
        lookup = {tuple(p): i for i, p in enumerate(self.points.tolist())}
        out = []
        for x, y in self.points.tolist():
            partner = lookup.get((x, -y if y != 0 else 0.0))
            if partner is None:
                raise ValueError(f"point ({x}, {y}) has no mirror partner")
            out.append(partner)
        return np.array(out, dtype=int)

    def is_symmetric(self) -> bool:
        try:
            self.mirror_index()
        except ValueError:
            return False
        return True

    def physical(self, r0_m: float, a0_m: float) -> tuple[np.ndarray, np.ndarray]:
        """(R, Z) in metres of the sample points followed by the reference point last."""
        pts = np.vstack([self.points, np.asarray(REFERENCE_POINT)[None, :]])
        return r0_m + a0_m * pts[:, 0], a0_m * pts[:, 1]

    def hash(self) -> str:
        return hashlib.sha256(json.dumps({"points": self.points.tolist(), "min_clearance_m": self.min_clearance_m},
                                         sort_keys=True).encode()).hexdigest()


# --------------------------------------------------------------------------- clearance
def _distance_to_outline(r: float, z: float, outline: np.ndarray) -> float:
    """Distance in metres from (r, z) to a convex quadrilateral, 0 inside it."""
    p = np.array([r, z])
    inside = True
    best = np.inf
    for k in range(4):
        a, b = outline[k], outline[(k + 1) % 4]
        ab, ap = b - a, p - a
        if ab[0] * ap[1] - ab[1] * ap[0] < 0:          # counter-clockwise corners: right of an edge is outside
            inside = False
        t = np.clip(np.dot(ap, ab) / np.dot(ab, ab), 0.0, 1.0)
        best = min(best, float(np.hypot(*(ap - t * ab))))
    return 0.0 if inside else best


def clearances(points_r, points_z, geometry: CoilGeometry) -> tuple[np.ndarray, np.ndarray]:
    """Distance from each point to the nearest coil element outline (0 inside an element)
    and the index of that element. Outlines follow ``coil_flux.element_outline`` (skewed
    parallelograms included)."""
    outlines = [element_outline(geometry.r[i], geometry.z[i], geometry.width[i], geometry.height[i],
                                geometry.angle1[i], geometry.angle2[i]) for i in range(geometry.n_elements)]
    pr = np.asarray(points_r, dtype=np.float64).ravel()
    pz = np.asarray(points_z, dtype=np.float64).ravel()
    dist = np.full(pr.size, np.inf)
    nearest = np.full(pr.size, -1, dtype=int)
    for j in range(pr.size):
        for i, outline in enumerate(outlines):
            d = _distance_to_outline(pr[j], pz[j], outline)
            if d < dist[j]:
                dist[j], nearest[j] = d, i
    return dist, nearest


def clearance_table(layout: Layout, cfg: MachineConfig, geometry: CoilGeometry, grid_r, grid_z,
                    r0_m: float | None = None, a0_m: float | None = None) -> pd.DataFrame:
    """One row per sample point of one machine: x, y, R, Z, clearance in metres, nearest
    element and its current column, whether the point is inside the grid, and ``ok``
    (inside the grid and at least ``layout.min_clearance_m`` from every element)."""
    r0 = cfg.major_radius.value if r0_m is None else r0_m
    a0 = cfg.minor_radius.value if a0_m is None else a0_m
    pr, pz = layout.physical(r0, a0)
    pr, pz = pr[:-1], pz[:-1]
    dist, nearest = clearances(pr, pz, geometry)
    gr, gz = np.asarray(grid_r, dtype=np.float64), np.asarray(grid_z, dtype=np.float64)
    in_grid = (pr >= gr.min()) & (pr <= gr.max()) & (pz >= gz.min()) & (pz <= gz.max())
    return pd.DataFrame({
        "machine": cfg.name, "column": layout.column_names(), "x": layout.points[:, 0], "y": layout.points[:, 1],
        "R_m": pr, "Z_m": pz, "clearance_m": dist, "nearest_element": geometry.names[nearest],
        "nearest_column": geometry.input_columns[nearest], "in_grid": in_grid,
        "ok": in_grid & (dist >= layout.min_clearance_m),
    })


def check_layout(layout: Layout, tables: dict[str, pd.DataFrame]) -> dict:
    """Check X4: every point meets the clearance rule on every machine and the layout is
    symmetric about y = 0.

    Args:
        layout: The layout.
        tables: Machine name to its ``clearance_table``.

    Returns:
        ``status`` (``pass`` or ``fail``), ``symmetric``, ``failing`` (list of
        ``{"machine", "column", "x", "y", "clearance_m"}``) and the smallest clearance per machine.
    """
    failing = []
    for machine, table in tables.items():
        for row in table.loc[~table["ok"]].itertuples(index=False):
            failing.append({"machine": machine, "column": row.column, "x": float(row.x), "y": float(row.y),
                            "clearance_m": float(row.clearance_m), "in_grid": bool(row.in_grid)})
    symmetric = layout.is_symmetric()
    return {"status": "pass" if symmetric and not failing else "fail", "symmetric": symmetric, "failing": failing,
            "min_clearance_m": {m: float(t["clearance_m"].min()) for m, t in tables.items()}}


# ------------------------------------------------------------------------- point tables
def geometry_from_split(data_root: Path | str, machine: str) -> tuple[CoilGeometry, str]:
    """Coil geometry of a machine from the first shot (sorted file order) of its public test
    folder, reading the geometry columns only. Geometry is the same in every shot (F06).

    Returns:
        ``(geometry, path of the shot read)``.
    """
    files = dataset.list_shots(data_root, GEOMETRY_SPLITS[machine])
    if not files:
        raise FileNotFoundError(f"no shot files for {machine} under {data_root}")
    shot = io.load_shot(files[0], list(GEOMETRY_COLUMNS), apply=False)
    return CoilGeometry.from_shot(shot), str(files[0])


def point_table_key(layout: Layout, cfg: MachineConfig, r0_m: float, a0_m: float, geometry_hash: str,
                    n_filaments: int) -> str:
    """Cache key of a point table: the physical points, the geometry and the subdivision."""
    pr, pz = layout.physical(r0_m, a0_m)
    payload = {"machine": cfg.name, "r": pr.tolist(), "z": pz.tolist(), "geometry": geometry_hash, "n": n_filaments}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def load_or_build_point_table(layout: Layout, cfg: MachineConfig, grid_table: GreenTable, data_root: Path | str,
                              cache_dir: Path | str = DEFAULT_POINT_CACHE, r0_m: float | None = None,
                              a0_m: float | None = None) -> tuple[PointTable, dict]:
    """The point table of a machine's layout points (reference point last), from the cache
    ``points_<machine>_<key>.npz`` or built and written there.

    The geometry must hash to the grid table's ``geometry_hash``, and the subdivision is the
    grid table's, so that the points see the same coil model as the grid.

    Returns:
        ``(table, record)`` with the cache path, whether it was built, and the geometry source.

    Raises:
        ValueError: when the geometry read differs from the grid table's.
    """
    r0 = cfg.major_radius.value if r0_m is None else r0_m
    a0 = cfg.minor_radius.value if a0_m is None else a0_m
    key = point_table_key(layout, cfg, r0, a0, grid_table.geometry_hash, grid_table.n_filaments)
    path = Path(cache_dir) / f"points_{cfg.file_prefix.split('_')[0]}_{key}.npz"
    record = {"path": str(path), "key": key, "built": False, "geometry_source": None}
    if path.is_file():
        table = PointTable.load(path)
        if table.geometry_hash != grid_table.geometry_hash:
            raise ValueError(f"{path}: geometry hash differs from the grid table's")
        return table, record
    geometry, source = geometry_from_split(data_root, cfg.name)
    if geometry.hash() != grid_table.geometry_hash:
        raise ValueError(f"geometry of {source} differs from the grid table's (hash {geometry.hash()[:12]} "
                         f"against {grid_table.geometry_hash[:12]})")
    pr, pz = layout.physical(r0, a0)
    table = build_point_table(cfg, geometry, pr, pz, grid_table.n_filaments)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    table.save(tmp)
    tmp.replace(path)
    record.update(built=True, geometry_source=source)
    return table, record


# ------------------------------------------------------------------------ point values
def coil_point_values(table: PointTable, currents_ka, ip_ka, r0_m: float, scales=None,
                      min_abs_ip_ka: float = 0.0) -> np.ndarray:
    """``coil_pt`` of every frame: ``(frames, n_points)`` without units.

    Args:
        table: Point table with the reference point as its last point.
        currents_ka: ``(frames, n_columns)`` currents in kA in ``table.columns`` order, as
            stored (the coil flux is then in the physical frame).
        ip_ka: Signed plasma current per frame in kA.
        r0_m: Reference major radius in metres.
        scales: Coil model (``CoilScales``) or None.
        min_abs_ip_ka: Frames with ``abs(Ip)`` below this, or without Ip, give NaN.
    """
    psi = np.atleast_2d(table.flux(currents_ka, scales))
    ip_a = np.asarray(ip_ka, dtype=np.float64).ravel() * A_PER_KA
    ok = np.isfinite(ip_a) & (np.abs(ip_a) >= min_abs_ip_ka * A_PER_KA) & (ip_a != 0)
    denominator = np.where(ok, MU0 * r0_m * ip_a, np.nan)
    return (psi[:, :-1] - psi[:, -1:]) / denominator[:, None]


def term_matrix(points) -> np.ndarray:
    """Design matrix ``(n_points, 9)`` of the polynomial terms at normalized points."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    return np.stack([pts[:, 0] ** m * pts[:, 1] ** n for m, n in TERMS], axis=1)


def term_operator(layout: Layout) -> np.ndarray:
    """Least-squares operator ``(9, n_points)``: named terms = ``operator @ coil_pt``.

    Raises:
        ValueError: when the layout cannot determine the nine terms.
    """
    a = term_matrix(layout.points)
    if np.linalg.matrix_rank(a) < len(TERMS):
        raise ValueError("the layout does not determine the nine polynomial terms")
    return np.linalg.pinv(a)
