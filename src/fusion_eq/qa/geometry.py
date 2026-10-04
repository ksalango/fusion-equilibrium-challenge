"""Checks 0.9 (coil geometry) and 0.10 (flux grid)."""

from __future__ import annotations

import hashlib
import json
from collections import Counter

import numpy as np

from fusion_eq.io import Shot

COIL_COLUMNS = ("coil_name", "coil_input_column", "coil_R", "coil_Z",
                "coil_width", "coil_height", "coil_angle1", "coil_angle2")
EXPECTED_GRID_LENGTH = 65


def sha256_of_arrays(*arrays) -> str:
    """Hex SHA-256 over the exact bytes of several arrays, in order.

    Numeric arrays are hashed as float64; string arrays as UTF-8 joined by NUL.
    A missing array (None) contributes a fixed marker so the hash stays defined.
    """
    h = hashlib.sha256()
    for a in arrays:
        if a is None:
            h.update(b"<none>")
            continue
        a = np.asarray(a)
        if a.dtype.kind in ("U", "S", "O"):
            h.update("\x00".join(str(x) for x in a.ravel()).encode())
        else:
            h.update(np.ascontiguousarray(a, dtype=np.float64).tobytes())
        h.update(b"|")
    return h.hexdigest()


def check_coil_geometry(shot: Shot) -> dict:
    """Check 0.9: coil geometry configuration of the shot.

    Returns:
        ``coil_hash`` (SHA-256 over all coil columns), ``coil_n_elements``,
        ``coil_n_input_columns``, ``coil_input_columns`` (sorted, joined by ``|``),
        ``coil_elements_per_column`` (JSON), ``coil_unmapped_columns`` (values of
        ``coil_input_column`` with no matching column in the file),
        ``coil_mapping_ok``, ``coil_solenoid_present``, ``coil_solenoid_n_elements``,
        ``coil_R_min/max``, ``coil_Z_min/max``, ``coil_angles_all_zero``.
    """
    arrays = [shot.get(c) for c in COIL_COLUMNS]
    out: dict = {"coil_hash": sha256_of_arrays(*arrays)}
    names = shot.get("coil_name")
    inputs = shot.get("coil_input_column")
    if names is None or inputs is None:
        out["coil_error"] = "coil_name or coil_input_column missing"
        return out
    inputs = [str(x) for x in np.asarray(inputs).ravel()]
    counts = Counter(inputs)
    out["coil_n_elements"] = int(len(inputs))
    out["coil_n_names_distinct"] = int(len(set(str(x) for x in np.asarray(names).ravel())))
    out["coil_n_input_columns"] = len(counts)
    out["coil_input_columns"] = "|".join(sorted(counts))
    out["coil_elements_per_column"] = json.dumps(dict(sorted(counts.items())))
    available = set(shot.file_columns) or set(shot.columns)
    unmapped = sorted(c for c in counts if c not in available)
    out["coil_unmapped_columns"] = "|".join(unmapped)
    out["coil_mapping_ok"] = not unmapped
    sol = shot.machine.solenoid_column
    out["coil_solenoid_present"] = sol in counts
    out["coil_solenoid_n_elements"] = int(counts.get(sol, 0))
    for key in ("R", "Z"):
        v = shot.get(f"coil_{key}")
        if v is not None:
            v = np.asarray(v, dtype=np.float64)
            out[f"coil_{key}_min"] = float(np.nanmin(v))
            out[f"coil_{key}_max"] = float(np.nanmax(v))
    a1, a2 = shot.get("coil_angle1"), shot.get("coil_angle2")
    if a1 is not None and a2 is not None:
        out["coil_angles_all_zero"] = bool(
            np.all(np.asarray(a1, dtype=np.float64) == 0) and np.all(np.asarray(a2, dtype=np.float64) == 0))
    return out


def check_grid(shot: Shot) -> dict:
    """Check 0.10: the flux grid axes.

    Returns:
        For ``R`` and ``Z``: ``grid_<a>_n``, ``grid_<a>_increasing`` (strictly),
        ``grid_<a>_min``, ``grid_<a>_max`` (metres), ``grid_<a>_step`` (median
        spacing), ``grid_<a>_uniform`` (spacing constant to 10 micrometres). Plus
        ``grid_hash`` over both arrays and ``grid_ok`` (both strictly increasing with
        length 65).
    """
    out: dict = {}
    ok = True
    for axis in ("R", "Z"):
        v = shot.get(f"efit_grid_{axis}")
        if v is None:
            out[f"grid_{axis}_n"] = 0
            ok = False
            continue
        v = np.asarray(v, dtype=np.float64).ravel()
        d = np.diff(v)
        increasing = bool(v.size > 1 and np.all(d > 0))
        out[f"grid_{axis}_n"] = int(v.size)
        out[f"grid_{axis}_increasing"] = increasing
        out[f"grid_{axis}_min"] = float(v.min()) if v.size else np.nan
        out[f"grid_{axis}_max"] = float(v.max()) if v.size else np.nan
        out[f"grid_{axis}_step"] = float(np.median(d)) if d.size else np.nan
        # The grids are stored as float32, so spacings differ by rounding at the
        # 1e-7 m level. Ten micrometres separates rounding from a real step change.
        out[f"grid_{axis}_uniform"] = bool(d.size and np.allclose(d, d[0], rtol=0, atol=1e-5))
        ok &= increasing and v.size == EXPECTED_GRID_LENGTH
    out["grid_hash"] = sha256_of_arrays(shot.get("efit_grid_R"), shot.get("efit_grid_Z"))
    out["grid_ok"] = bool(ok)
    return out
