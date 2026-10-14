"""Target build: configuration, the per-shot targets, the fitted objects and the mean-current
map (``docs/TARGET.md``). ``scripts/build_targets.py`` runs it; ``scripts/profile_targets.py``,
the review notebook and the predictor registry read what it writes.

Only DIII-D training shots carry equilibrium labels. The build refuses any other split, and
MAST equilibrium labels are never read: every MAST quantity of the target phase comes from
geometry, machine constants and test inputs.

Per frame (``table_shot``): the share and its diagnostics (``current_basis.current_share``),
the remainder in the target frame (``current_target.vacuum_remainder`` with the same mask),
the current coefficients and the relative reconstruction error of the share and of its
mirror, the remainder coefficients and the remainder's RMS over the frame's flux range, and
the seven shape numbers the scorer derives from the true map (taken from the cached scorer
references when they exist for the list and stride, otherwise computed with the scorer's own
reference builder on the stored map).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from fusion_eq import baseline as bl, current_basis as cb, current_target as ct, dataset, starter_kit, targets
from fusion_eq.coil_flux import GreenTable, adopted_scales
from fusion_eq.coil_scales import phase_of
from fusion_eq.lattice import Placement
from fusion_eq.machines import machine_config
from fusion_eq.remainder_model import RemainderModel

MACHINE = "DIII-D"
LABELLED_SPLIT = "diii_d_train"
DEFAULT_CONFIG = dataset.REPO_ROOT / "configs" / "target_v1.yaml"
DEFAULT_TARGET_ROOT = dataset.DEFAULT_PROCESSED_ROOT / "targets"
SHAPE_COLUMNS = ("R_axis", "Z_axis", "kappa", "tri_top", "tri_bot", "volume", "li")
OBJECT_FILES = {"current_basis": "current_basis.npz", "remainder_model": "remainder_model.npz",
                "flux_basis_DIII-D": "flux_basis_d3d.npz", "flux_basis_MAST": "flux_basis_mast.npz",
                "placement_DIII-D": "placement_d3d.npz", "placement_MAST": "placement_mast.npz"}
MACHINES = ("DIII-D", "MAST")


# ---------------------------------------------------------------------------- configuration
@dataclass(frozen=True)
class TargetConfig:
    """``configs/target_v1.yaml``; field meanings in that file."""

    version: int
    n_current: int = 50
    fit_frames: int = 100_000
    seed: int = 0
    symmetric: bool = True
    n_remainder: int = 10
    ridge_alpha: float = 1.0
    remainder_inputs: str = "currents_and_rates"
    rate_windows_ms: tuple[int, ...] = (10, 50)
    mask: str = "scorer_boundary"
    placement: str = "conservative"
    shape_numbers: bool = True

    def __post_init__(self) -> None:
        if not self.symmetric:
            raise ValueError("only the symmetric current basis is built (current.symmetric: true)")
        if self.mask != "scorer_boundary" or self.placement != "conservative":
            raise ValueError("mask must be scorer_boundary and placement conservative")
        if self.remainder_inputs != "currents_and_rates":
            raise ValueError("remainder.inputs must be currents_and_rates (the inputs of ceiling variant g)")
        if 10 not in self.rate_windows_ms:
            raise ValueError("remainder.rate_windows_ms must include 10")
        if self.n_current < 1 or self.n_remainder < 1 or self.fit_frames < 2:
            raise ValueError("component counts must be positive and fit_frames at least 2")

    @classmethod
    def from_dict(cls, d: dict) -> "TargetConfig":
        known = {"version", "current", "remainder", "mask", "placement", "shape_numbers"}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown configuration fields {sorted(unknown)}")
        cur, rem = dict(d.get("current", {})), dict(d.get("remainder", {}))
        return cls(version=int(d["version"]), n_current=int(cur.get("n_components", 50)),
                   fit_frames=int(cur.get("fit_frames", 100_000)), seed=int(cur.get("seed", 0)),
                   symmetric=bool(cur.get("symmetric", True)), n_remainder=int(rem.get("n_components", 10)),
                   ridge_alpha=float(rem.get("ridge_alpha", 1.0)), remainder_inputs=str(rem.get("inputs", "currents_and_rates")),
                   rate_windows_ms=tuple(int(v) for v in rem.get("rate_windows_ms", (10, 50))),
                   mask=str(d.get("mask", "scorer_boundary")), placement=str(d.get("placement", "conservative")),
                   shape_numbers=bool(d.get("shape_numbers", True)))

    @classmethod
    def load(cls, path: Path | str = DEFAULT_CONFIG) -> "TargetConfig":
        return cls.from_dict(yaml.safe_load(Path(path).read_text()))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["rate_windows_ms"] = list(self.rate_windows_ms)
        return d

    def hash(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()


def target_dir(version: int, root: Path | str = DEFAULT_TARGET_ROOT) -> Path:
    return Path(root) / f"v{version}"


def table_path(list_name: str, version: int, root: Path | str = DEFAULT_TARGET_ROOT) -> Path:
    """Target table of one list of the split file: ``<root>/v<version>/<list>.parquet``."""
    return target_dir(version, root) / f"{list_name}.parquet"


# ------------------------------------------------------------------------------- guards
def require_labelled(split: str) -> None:
    """Raises unless ``split`` is the labelled DIII-D training split.

    Raises:
        ValueError: for a MAST split (MAST equilibrium labels are not read) or a split
            without labels.
    """
    if split == LABELLED_SPLIT:
        return
    machine = dataset.SPLIT_MACHINE.get(split)
    if machine == "MAST" or "mast" in split.lower():
        raise ValueError(f"{split}: MAST equilibrium labels are not read; targets are built on DIII-D training shots only")
    raise ValueError(f"{split}: no equilibrium labels; targets are built on {LABELLED_SPLIT} only")


def read_truth(shot_id: str, data_root: Path | str, split: str, stride: int) -> dict:
    """The stored maps (float32) and the two scalar labels of a labelled shot at a stride.

    Raises:
        ValueError: when the split is not the labelled one or the file has no maps.
    """
    require_labelled(split)
    try:
        return bl.load_truth(shot_id, data_root, split, stride)
    except KeyError as exc:
        raise ValueError(f"{shot_id}: the shot file has no equilibrium labels ({exc})") from exc


# --------------------------------------------------------------------------- resources
@lru_cache(maxsize=2)
def _resources(green_dir: str, response_cache: str) -> tuple:
    """DIII-D Green's table, adopted coil model, plasma response, scorer envelope and modules;
    one load per process."""
    cfg = machine_config(MACHINE)
    table = GreenTable.load(Path(green_dir) / "green_d3d.npz")
    scales = adopted_scales(MACHINE, table.columns)
    response = ct.PlasmaResponse.load_or_build(MACHINE, table.grid_r, table.grid_z, response_cache)
    return table, scales, response, starter_kit.envelope_mask(cfg), starter_kit.scoring()


def split_frames(psi, rows: pd.DataFrame, table, scales, response, envelope, scorer) -> tuple[cb.ShareResult, np.ndarray]:
    """Share and remainder (target frame, ``sign(Ip) r``) of stored maps and their frame rows."""
    cfg = machine_config(MACHINE)
    ip = rows["ip__mean"].to_numpy(dtype=np.float64)
    coil = targets.coil_flux_of_frames(table, rows, scales, cfg)
    share = cb.current_share(psi, coil, ip, table.grid_r, table.grid_z, cfg, envelope=envelope, scorer=scorer)
    _, remainder = ct.vacuum_remainder(np.asarray(psi, dtype=np.float64), coil, ip, share.mask, response, cfg)
    return share, remainder


def fit_shot(shot_id: str, rows: pd.DataFrame, chosen: np.ndarray, data_root: str, split: str, stride: int,
             green_dir: str, response_cache: str) -> dict:
    """Fit pass of one shot: shares and remainders (target frame) of the chosen strided frames,
    with their boundary flags. Runs in a worker process."""
    table, scales, response, envelope, scorer = _resources(green_dir, response_cache)
    psi = read_truth(shot_id, data_root, split, stride)["psi"]
    if psi.shape[0] != len(rows):
        raise ValueError(f"{shot_id}: {len(rows)} rows for {psi.shape[0]} maps at stride {stride}")
    idx = np.flatnonzero(np.asarray(chosen, dtype=bool))
    share, remainder = split_frames(psi[idx], rows.iloc[idx], table, scales, response, envelope, scorer)
    return {"shot": shot_id, "frames": idx, "share": share.share, "remainder": remainder, "boundary": share.boundary_found}


def shape_numbers(psi_float32: np.ndarray) -> np.ndarray:
    """``(frames, 7)`` shape numbers the scorer derives from stored maps (``SHAPE_COLUMNS``),
    NaN where the scorer finds them implausible; the scorer's own reference builder."""
    ls = bl.local_score_module()
    env = starter_kit.envelope_mask(machine_config(MACHINE))
    _, cons, _, _ = ls.build_reference(psi_float32, env.grid_R, env.grid_Z, env.mask_coarse, env.mask_f)
    return np.asarray(cons, dtype=np.float64)


def table_shot(shot_id: str, rows: pd.DataFrame, data_root: str, split: str, stride: int, green_dir: str,
               response_cache: str, basis: cb.CurrentBasis, remainder: RemainderModel, cons: np.ndarray | None,
               with_shapes: bool) -> dict:
    """Table rows of one shot's strided frames (column arrays). Runs in a worker process.

    Args:
        cons: Shape numbers from the cached scorer references, or None to compute them when
            ``with_shapes``.
    """
    table, scales, response, envelope, scorer = _resources(green_dir, response_cache)
    psi32 = read_truth(shot_id, data_root, split, stride)["psi"]
    if psi32.shape[0] != len(rows):
        raise ValueError(f"{shot_id}: {len(rows)} rows for {psi32.shape[0]} maps at stride {stride}")
    psi = psi32.astype(np.float64)
    share, rem = split_frames(psi, rows, table, scales, response, envelope, scorer)
    s = share.share
    c = basis.transform(s)
    c_mirror = basis.transform(cb.mirror_share(s))
    norm = np.sqrt((s.reshape(len(s), -1) ** 2).sum(axis=1))

    def relative_error(original, coefficients):
        diff = original.reshape(len(original), -1) - basis.inverse(coefficients).reshape(len(original), -1)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(norm > 0, np.sqrt((diff ** 2).sum(axis=1)) / norm, np.nan)

    flux_range = psi.max(axis=(1, 2)) - psi.min(axis=(1, 2))
    r_rms = np.sqrt(np.mean(rem * rem, axis=(1, 2)))
    out = {"shot": np.full(len(rows), shot_id), "frame": rows["frame"].to_numpy(dtype=np.int32),
           "ip_ka": rows["ip__mean"].to_numpy(dtype=np.float64),
           "phase": np.array([phase_of(float(o), int(f), int(n)) for o, f, n in
                              zip(rows["ip_over_peak"], rows["frame"], rows["n_frames"])], dtype=object),
           "boundary_found": share.boundary_found, "share_total": share.total, "share_outside": share.outside_abs_share,
           "recon_error": relative_error(s, c), "recon_error_mirror": relative_error(cb.mirror_share(s), c_mirror),
           "flux_range": flux_range,
           "remainder_rms_over_range": np.where(flux_range > 0, r_rms / np.where(flux_range > 0, flux_range, 1.0), np.nan)}
    for j in range(c.shape[1]):
        out[f"c_{j:02d}"] = c[:, j]
    rc = remainder.encode(rem)
    for j in range(rc.shape[1]):
        out[f"r_{j:02d}"] = rc[:, j]
    if with_shapes:
        values = shape_numbers(psi32) if cons is None else np.asarray(cons, dtype=np.float64)
        if values.shape != (len(rows), len(SHAPE_COLUMNS)):
            raise ValueError(f"{shot_id}: shape numbers {values.shape} for {len(rows)} frames")
        for j, name in enumerate(SHAPE_COLUMNS):
            out[name] = values[:, j]
    return out


def cached_shape_numbers(shot_ids, split: str, stride: int, cache_dir: Path | str | None) -> dict[str, np.ndarray] | None:
    """Shape numbers per shot from the cached scorer references of exactly these shots and
    stride (``baseline.build_references``), or None when no such cache exists."""
    if cache_dir is None:
        return None
    path = Path(cache_dir) / f"refs_{bl.reference_key(list(shot_ids), split, stride)}.npz"
    if not path.is_file():
        return None
    meta, records = bl.load_references(path)
    if meta["shots"] != list(shot_ids) or meta["stride"] != int(stride):
        return None
    return {s: r["cons"] for s, r in zip(meta["shots"], records)}


# ------------------------------------------------------------------------------ objects
@dataclass
class TargetObjects:
    """The fitted objects of a target build.

    Attributes:
        current_basis: Share basis.
        remainder_model: Remainder basis and ridges.
        flux_bases: Machine to its flux basis.
        placements: Machine to its placement.
        record: The build record (``build.json``).
    """

    current_basis: cb.CurrentBasis
    remainder_model: RemainderModel
    flux_bases: dict[str, cb.MachineFluxBasis]
    placements: dict[str, Placement]
    record: dict = field(default_factory=dict)

    def save(self, folder: Path | str) -> dict[str, str]:
        """Writes every object; returns file name to its SHA-256."""
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        self.current_basis.save(folder / OBJECT_FILES["current_basis"])
        self.remainder_model.save(folder / OBJECT_FILES["remainder_model"])
        for m in MACHINES:
            self.flux_bases[m].save(folder / OBJECT_FILES[f"flux_basis_{m}"])
            self.placements[m].save(folder / OBJECT_FILES[f"placement_{m}"])
        return {name: sha256_of(folder / name) for name in OBJECT_FILES.values()}

    @classmethod
    def load(cls, folder: Path | str) -> "TargetObjects":
        folder = Path(folder)
        record_path = folder / "build.json"
        return cls(current_basis=cb.CurrentBasis.load(folder / OBJECT_FILES["current_basis"]),
                   remainder_model=RemainderModel.load(folder / OBJECT_FILES["remainder_model"]),
                   flux_bases={m: cb.MachineFluxBasis.load(folder / OBJECT_FILES[f"flux_basis_{m}"]) for m in MACHINES},
                   placements={m: Placement.load(folder / OBJECT_FILES[f"placement_{m}"]) for m in MACHINES},
                   record=json.loads(record_path.read_text()) if record_path.is_file() else {})


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# --------------------------------------------------------------------------- mean current
def mean_current_maps(machine: str, rows: pd.DataFrame, table: GreenTable, scales, objects: TargetObjects,
                      with_remainder: bool = True, window: str | None = None) -> np.ndarray:
    """Stored-frame maps of the mean-current model: the coil flux plus ``Ip phi_mean`` of the
    machine, plus the ridge remainder on DIII-D when ``with_remainder`` (zero on MAST).

    Args:
        machine: ``DIII-D`` or ``MAST``.
        rows: Frame-table rows with ``ip__mean``, the coil currents and, for the remainder, the
            ridge inputs.
        table, scales: The machine's coil Green's table and coil model.
        objects: The fitted target objects.
        with_remainder: Add the ridge remainder on DIII-D.
        window: Rate window of the ridge; the model's default when None.
    """
    cfg = machine_config(machine)
    ip = rows["ip__mean"].to_numpy(dtype=np.float64)
    coil = targets.coil_flux_of_frames(table, rows, scales, cfg)
    basis = objects.flux_bases[cfg.name]
    c = np.zeros((len(rows), basis.n_components))
    remainder = objects.remainder_model.predict(rows, window) if (with_remainder and cfg.name == MACHINE) else None
    return cb.compose_map(cfg, coil, ip, c, basis, remainder)
