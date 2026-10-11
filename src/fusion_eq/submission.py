"""Submission pipeline: predictor interface, fold prediction with fallbacks, npz files,
validation and the Codabench pointer.

Independent of the model. A predictor is any object with ``name``, ``setup(log)`` and
``predict_shot(shot, machine)``; the pipeline hands it one raw shot at a time (inputs
only, label columns removed) and expects one map and two scalars per ``efit_times``
entry. Everything else here, the shot order, the fallbacks, the writing, the
validation and the pointer, never looks inside the predictor.

Shot order follows the Hugging Face stream the organizers' skeleton and validator use:
the data files of a config are resolved with a sorted glob and streamed one row per
file, so the key index of a shot is the rank of its file name in sorted order of the
local split folder (``dataset.list_shots``). Keys are ``shot_XXXX_psirz`` (T, 65, 65)
float16, ``shot_XXXX_q95`` and ``shot_XXXX_betaN`` (T,) float32, exactly what the
skeleton writes.

Fallbacks, applied to any frame whose predicted map holds a non-finite value, in this
order: the frame's coil flux plus the mean plasma shape scaled by Ip (DIII-D only),
the nearest frame of the same shot with a finite map, a constant map. Scalars fall
back to the nearest finite frame of the shot, then to the training median. Every
fallback use is counted per machine.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Protocol, Sequence, runtime_checkable

import numpy as np

from fusion_eq import coil_flux as cf, dataset, frame_table as ft, io, targets
from fusion_eq.baseline import Prediction
from fusion_eq.machines import MachineConfig, machine_config

GRID = 65
SCALARS = ("q95", "betaN")
FIELDS = ("psirz", *SCALARS)
PSI_DTYPE = np.float16
SCALAR_DTYPE = np.float32
# Machine name to the public-test config, which is also the split folder and the npz file stem.
TEST_CONFIGS = {"DIII-D": "diii_d_public_test", "MAST": "mast_public_test"}
HUB_REPO_ID = "Sophelio/fusion-equilibrium-challenge"
HUB_TEST_SPLIT = "public_test"
FINGERPRINT_COLUMNS = ("efit_times", "magnetics_plasma_current")
MANIFEST_FIELDS = ("repo_id", "revision", "token")
MANIFEST_NAME = "manifest.json"
POINTER_NAME = "submission_pointer.zip"
# Fixed member timestamp, so that identical predictions give byte-identical npz files.
_ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)
FALLBACK_NAMES = ("coil_mean_shape", "nearest_frame", "constant_map", "scalar_nearest_frame", "scalar_median")


# ------------------------------------------------------------------ predictor interface
@runtime_checkable
class Predictor(Protocol):
    """What the pipeline requires of a model.

    ``setup`` loads a fitted model saved by a training script, or fits and saves it
    once so that later subcommands do not refit; it returns a JSON-serialisable dict of
    identifiers (files, hashes, parameters) for the run record. ``predict_shot``
    returns a ``Prediction`` with ``psirz`` (T, 65, 65), ``q95`` (T,) and ``betaN``
    (T,) for every ``efit_times`` entry of the shot, in the stored flux frame.
    """

    name: str

    def setup(self, log: Callable[[str], None] = print) -> dict: ...

    def predict_shot(self, shot: io.Shot, machine: str) -> Prediction: ...


REGISTRY: dict[str, Callable[..., Predictor]] = {}


def register(name: str) -> Callable:
    """Class decorator adding a predictor factory to the registry under ``name``."""
    def wrap(factory):
        if name in REGISTRY:
            raise ValueError(f"predictor {name!r} is already registered")
        REGISTRY[name] = factory
        return factory
    return wrap


def make_predictor(name: str, **kwargs) -> Predictor:
    """Instantiates a registered predictor.

    Raises:
        KeyError: for an unknown name.
    """
    import fusion_eq.predictors  # noqa: F401 - registers the shipped predictors
    try:
        factory = REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown predictor {name!r}; registered: {sorted(REGISTRY)}") from None
    return factory(**kwargs)


# ------------------------------------------------------------------------ shot order
def shot_key(index: int, name: str) -> str:
    """``shot_XXXX_<name>`` with the 0-indexed stream position."""
    if name not in FIELDS:
        raise ValueError(f"unknown field {name!r}; fields: {FIELDS}")
    return f"shot_{index:04d}_{name}"


def expected_keys(n_shots: int) -> list[str]:
    """Every key of a fold of ``n_shots`` shots, in writing order."""
    return [shot_key(i, name) for i in range(n_shots) for name in FIELDS]


def test_shot_files(data_root: Path | str, machine: str) -> list[Path]:
    """Shot files of a machine's public-test fold in stream order (sorted file names)."""
    files = dataset.list_shots(data_root, TEST_CONFIGS[machine])
    return sorted(files, key=lambda p: p.name)


def shot_fingerprint(row) -> dict:
    """Content fingerprint of one shot, from a Hub row or a loaded shot, as shipped.

    The Hub rows carry no shot id, so a shot is identified by the length, first and
    last value of ``efit_times`` and a SHA-256 over ``efit_times`` and the raw plasma
    current, both taken as float64. The local side must be loaded without the data
    fixes, since the Hub stream applies none.
    """
    times = np.asarray(row["efit_times"], dtype=np.float64).ravel()
    ip = np.asarray(row["magnetics_plasma_current"], dtype=np.float64).ravel()
    return {"n_frames": int(times.size), "first_ms": float(times[0]) if times.size else None,
            "last_ms": float(times[-1]) if times.size else None, "sha256": io.sha256_of_arrays(times, ip)[:16]}


def local_fingerprint(path: Path | str) -> dict:
    """``shot_fingerprint`` of a local shot file, read as shipped."""
    return shot_fingerprint(io.load_shot(path, FINGERPRINT_COLUMNS, apply=False))


def strip_labels(shot: io.Shot) -> io.Shot:
    """Removes every label column from a loaded shot, in place, and returns it.

    Guards data rule 4 on the labelled shots used for local scoring: a predictor only
    ever sees columns that exist in the test files.
    """
    shot.data = {k: v for k, v in shot.data.items() if not ft.is_label_column(k)}
    return shot


# -------------------------------------------------------------------------- fallbacks
@dataclass
class FallbackCounts:
    """Frames that used each fallback, plus the frames and shots seen."""

    counts: dict[str, int] = field(default_factory=lambda: {name: 0 for name in FALLBACK_NAMES})
    n_frames: int = 0
    n_shots: int = 0
    predictor_errors: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"n_shots": self.n_shots, "n_frames": self.n_frames, "fallback_frames": dict(self.counts),
                "n_predictor_errors": len(self.predictor_errors), "predictor_errors": list(self.predictor_errors)}


@dataclass
class Fallbacks:
    """Resources of the pipeline's fallbacks for one machine.

    Attributes:
        machine: Machine name.
        scalar_medians: ``q95`` and ``betaN`` medians of the DIII-D training frames.
        table: Green's table of the machine, for the coil flux of a frame; None skips
            the first fallback.
        scales: Coil model applied to the table.
        mean_shape: Mean plasma shape per kA, ``(65, 65)``; None skips the first
            fallback (MAST, where no plasma model exists).
        constant: Value of the constant map, the last resort.
    """

    machine: str
    scalar_medians: dict[str, float]
    table: cf.GreenTable | None = None
    scales: cf.CoilScales | None = None
    mean_shape: np.ndarray | None = None
    constant: float = 0.0

    def coil_mean_shape_maps(self, shot: io.Shot) -> np.ndarray | None:
        """Stored-frame maps of the coil flux plus ``|Ip|`` times the mean shape, per frame,
        or None when the machine has no mean shape or no sign rule."""
        if self.table is None or self.mean_shape is None or self.machine not in targets.PSI_SIGN_RULE_C:
            return None
        cfg = machine_config(self.machine)
        currents = cf.currents_at_frames(shot, self.table.columns)
        ip = cf.currents_at_frames(shot, [cfg.ip_column])[:, 0]
        coil = self.table.flux(currents, self.scales)
        target = np.abs(np.nan_to_num(ip))[:, None, None] * self.mean_shape[None]
        return targets.stored_from_target(target, coil, ip, cfg)


def _nearest_finite(good: np.ndarray) -> np.ndarray:
    """Index of the nearest frame with ``good`` True for every frame (ties go to the earlier
    frame); frames are their own nearest when good. ``good`` must hold a True."""
    idx = np.flatnonzero(good)
    pos = np.arange(good.size)
    right = np.searchsorted(idx, pos)
    left = np.clip(right - 1, 0, idx.size - 1)
    right = np.clip(right, 0, idx.size - 1)
    choose_left = (pos - idx[left]) <= (idx[right] - pos)
    return np.where(good, pos, np.where(choose_left, idx[left], idx[right]))


def apply_fallbacks(pred: Prediction, shot: io.Shot, fallbacks: Fallbacks, counts: FallbackCounts) -> Prediction:
    """Replaces non-finite frames and scalars of a prediction in the documented order and
    counts each replacement. Returns a new ``Prediction`` with float64 arrays, finite."""
    psi = np.array(pred.psirz, dtype=np.float64)
    n = psi.shape[0]
    bad = ~np.isfinite(psi).all(axis=(1, 2))
    if bad.any():
        physics = fallbacks.coil_mean_shape_maps(shot)
        if physics is not None:
            ok = bad & np.isfinite(physics).all(axis=(1, 2))
            psi[ok] = physics[ok]
            counts.counts["coil_mean_shape"] += int(ok.sum())
            bad &= ~ok
    if bad.any():
        good = ~bad
        if good.any():
            psi[bad] = psi[_nearest_finite(good)[bad]]
            counts.counts["nearest_frame"] += int(bad.sum())
        else:
            psi[bad] = fallbacks.constant
            counts.counts["constant_map"] += int(bad.sum())
    scalars = {}
    for name in SCALARS:
        v = np.array(getattr(pred, name), dtype=np.float64).reshape(-1)
        if v.shape != (n,):
            raise ValueError(f"{name}: shape {v.shape}, expected ({n},)")
        bad = ~np.isfinite(v)
        if bad.any():
            good = ~bad
            if good.any():
                v[bad] = v[_nearest_finite(good)[bad]]
                counts.counts["scalar_nearest_frame"] += int(bad.sum())
            else:
                v[bad] = fallbacks.scalar_medians[name]
                counts.counts["scalar_median"] += int(bad.sum())
        scalars[name] = v
    return Prediction(psirz=psi, q95=scalars["q95"], betaN=scalars["betaN"])


# -------------------------------------------------------------------- fold prediction
@dataclass
class FoldRecord:
    """What ``predict_fold`` records about one machine's fold."""

    machine: str
    shot_ids: list[str]
    n_frames_per_shot: list[int]
    counts: FallbackCounts
    shot_checks: list[dict]

    def to_dict(self) -> dict:
        out = {"machine": self.machine, **self.counts.to_dict(), "first_shot": self.shot_ids[0] if self.shot_ids else None,
               "last_shot": self.shot_ids[-1] if self.shot_ids else None}
        if self.shot_checks:
            out["shot_checks"] = self.shot_checks
        return out


def predict_fold(predictor: Predictor, files: Sequence[Path], machine: str, fallbacks: Fallbacks,
                 sink: Callable[[str, np.ndarray], None], shot_check: Callable[[int, io.Shot], dict] | None = None,
                 log: Callable[[str], None] = print, every: int = 50) -> FoldRecord:
    """Predicts every shot of a fold in order and hands the keyed arrays to ``sink``.

    Args:
        predictor: The model.
        files: Shot files in stream order (``test_shot_files``, or any labelled list).
        machine: Machine name.
        fallbacks: Fallback resources of the machine.
        sink: Receives ``(key, array)`` for every array, in key order; ``writer.add`` or
            ``dict.__setitem__``.
        shot_check: Optional input-only check per shot, ``(index, shot) -> dict``, stored
            in the record.
        log: Progress printer.
        every: Shots between progress lines.

    A predictor exception on a shot is recorded and the shot is predicted by the
    fallbacks alone. A prediction of the wrong shape is a programming error and raises.
    """
    record = FoldRecord(machine=machine, shot_ids=[], n_frames_per_shot=[], counts=FallbackCounts(), shot_checks=[])
    for i, path in enumerate(files):
        shot = strip_labels(io.load_shot(path))
        n = shot.n_frames
        try:
            pred = predictor.predict_shot(shot, machine)
            psi = np.asarray(pred.psirz)
            if psi.shape != (n, GRID, GRID):
                raise ValueError(f"psirz shape {psi.shape}, expected {(n, GRID, GRID)}")
        except Exception as exc:  # noqa: BLE001 - recorded, the shot falls back
            record.counts.predictor_errors.append({"index": i, "shot": shot.shot_id, "error": f"{type(exc).__name__}: {exc}"})
            log(f"  {machine} shot {i} ({shot.shot_id}): predictor failed, {type(exc).__name__}: {exc}")
            pred = Prediction(np.full((n, GRID, GRID), np.nan), np.full(n, np.nan), np.full(n, np.nan))
        pred = apply_fallbacks(pred, shot, fallbacks, record.counts)
        arrays = {"psirz": pred.psirz.astype(PSI_DTYPE), "q95": pred.q95.astype(SCALAR_DTYPE),
                  "betaN": pred.betaN.astype(SCALAR_DTYPE)}
        for name, arr in arrays.items():
            if not np.isfinite(arr).all():
                raise ValueError(f"{machine} shot {i} ({shot.shot_id}): {name} not finite after the cast to {arr.dtype}")
            sink(shot_key(i, name), arr)
        if shot_check is not None:
            record.shot_checks.append({"index": i, "shot": shot.shot_id, **shot_check(i, shot)})
        record.shot_ids.append(shot.shot_id)
        record.n_frames_per_shot.append(n)
        record.counts.n_shots += 1
        record.counts.n_frames += n
        if (i + 1) % every == 0:
            log(f"  {machine}: {i + 1}/{len(files)} shots, {record.counts.n_frames} frames")
    return record


# --------------------------------------------------------------------------- npz files
class NpzWriter:
    """Writes an npz incrementally, one array at a time, in the format ``np.savez_compressed``
    produces (a deflated zip of ``.npy`` members), with fixed member timestamps."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.keys: list[str] = []
        self._zip: zipfile.ZipFile | None = None

    def __enter__(self) -> "NpzWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._zip = zipfile.ZipFile(self.path, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True)
        return self

    def add(self, key: str, array: np.ndarray) -> None:
        if self._zip is None:
            raise RuntimeError("writer is not open")
        if key in self.keys:
            raise ValueError(f"duplicate key {key}")
        info = zipfile.ZipInfo(f"{key}.npy", date_time=_ZIP_EPOCH)
        info.compress_type = zipfile.ZIP_DEFLATED
        with self._zip.open(info, "w", force_zip64=True) as member:
            np.lib.format.write_array(member, np.ascontiguousarray(array), allow_pickle=False)
        self.keys.append(key)

    def __exit__(self, *exc) -> None:
        if self._zip is not None:
            self._zip.close()
            self._zip = None


def write_npz(path: Path | str, arrays: Iterable[tuple[str, np.ndarray]] | dict) -> Path:
    """Writes ``(key, array)`` pairs, or a dict, as one npz file."""
    items = arrays.items() if isinstance(arrays, dict) else arrays
    with NpzWriter(path) as writer:
        for key, arr in items:
            writer.add(key, arr)
    return Path(path)


def read_npz(path: Path | str) -> dict[str, np.ndarray]:
    """Reads an npz into a dict, keys in archive order, no pickling."""
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


def read_prediction(arrays: dict[str, np.ndarray], index: int, stride: int = 1) -> Prediction:
    """The ``Prediction`` of shot ``index`` from read-back arrays, every ``stride``-th frame."""
    s = slice(None, None, stride) if stride > 1 else slice(None)
    return Prediction(psirz=arrays[shot_key(index, "psirz")][s], q95=arrays[shot_key(index, "q95")][s],
                      betaN=arrays[shot_key(index, "betaN")][s])


def sha256_of_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# -------------------------------------------------------------------------- validation
@dataclass
class ValidationResult:
    """Outcome of ``validate_npz``."""

    path: str
    machine: str
    n_shots: int
    n_frames: int
    errors: list[str]
    psi_dtype: str | None = None

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict:
        return {"path": self.path, "machine": self.machine, "ok": self.ok, "n_shots": self.n_shots,
                "n_frames": self.n_frames, "psi_dtype": self.psi_dtype, "n_errors": len(self.errors),
                "errors": self.errors[:50]}


def frame_counts(files: Sequence[Path]) -> list[int]:
    """``len(efit_times)`` of every shot file, read without the data fixes."""
    return [int(len(io.load_shot(p, ["efit_times"], apply=False)["efit_times"])) for p in files]


def validate_npz(path: Path | str, files: Sequence[Path], machine: str, max_errors: int = 1000) -> ValidationResult:
    """Checks a written npz against the raw shots of its fold.

    For every shot in stream order: the three keys exist, the map has shape
    ``(T, 65, 65)`` and the scalars ``(T,)`` with ``T`` the shot's ``efit_times`` length,
    every array is floating and finite; no key beyond the fold's; the archive holds the
    keys in writing order. Finiteness is required on both machines (the organizers'
    validator requires it on DIII-D only).
    """
    path = Path(path)
    counts = frame_counts(files)
    errors: list[str] = []
    psi_dtype = None
    with np.load(path, allow_pickle=False) as f:
        keys = list(f.files)
        expected = expected_keys(len(files))
        extra = sorted(set(keys) - set(expected))
        if extra:
            errors.append(f"unexpected keys: {extra[:5]}{' ...' if len(extra) > 5 else ''}")
        present = [k for k in expected if k in set(keys)]
        if [k for k in keys if k in set(expected)] != present:
            errors.append("keys are not in shot order in the archive")
        for i, (shot_file, t) in enumerate(zip(files, counts)):
            if len(errors) >= max_errors:
                errors.append("... further errors not listed")
                break
            for name in FIELDS:
                key = shot_key(i, name)
                shape = (t, GRID, GRID) if name == "psirz" else (t,)
                if key not in f.files:
                    errors.append(f"{key} ({shot_file.name}): missing, expected {shape}")
                    continue
                arr = f[key]
                if arr.shape != shape:
                    errors.append(f"{key}: shape {arr.shape}, expected {shape}")
                if not np.issubdtype(arr.dtype, np.floating):
                    errors.append(f"{key}: dtype {arr.dtype}, expected a float type")
                elif not np.isfinite(arr).all():
                    errors.append(f"{key}: {int((~np.isfinite(arr)).sum())} non-finite values")
                if name == "psirz":
                    psi_dtype = str(arr.dtype) if psi_dtype is None else psi_dtype
                    if arr.dtype != PSI_DTYPE:
                        errors.append(f"{key}: dtype {arr.dtype}, the pipeline writes {np.dtype(PSI_DTYPE)}")
    return ValidationResult(path=str(path), machine=machine, n_shots=len(files), n_frames=int(sum(counts)),
                            errors=errors, psi_dtype=psi_dtype)


# ----------------------------------------------------------------------------- pointer
def manifest_dict(repo_id: str, revision: str, token: str) -> dict:
    """The pointer manifest: exactly ``repo_id``, ``revision`` (40 hex characters) and ``token``.

    Raises:
        ValueError: for a malformed revision, an empty token or a repo id without a namespace.
    """
    if not (len(revision) == 40 and all(c in "0123456789abcdef" for c in revision.lower())):
        raise ValueError(f"revision must be a 40-character commit SHA, got {revision!r}")
    if not token:
        raise ValueError("the read token is empty")
    if "/" not in repo_id:
        raise ValueError(f"repo id must be <namespace>/<name>, got {repo_id!r}")
    return {"repo_id": repo_id, "revision": revision, "token": token}


def write_pointer_zip(zip_path: Path | str, manifest: dict) -> Path:
    """Writes ``submission_pointer.zip`` holding only ``manifest.json`` at its root.

    The manifest is written straight into the archive, never to a loose file, so the
    token exists on disk in the zip alone.
    """
    if tuple(manifest) != MANIFEST_FIELDS:
        raise ValueError(f"manifest fields must be {MANIFEST_FIELDS}, got {tuple(manifest)}")
    zip_path = Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(MANIFEST_NAME, json.dumps(manifest, indent=2))
    return zip_path


def read_pointer_zip(zip_path: Path | str) -> dict:
    """The manifest of a pointer zip; raises when the archive holds anything else."""
    with zipfile.ZipFile(zip_path) as z:
        if z.namelist() != [MANIFEST_NAME]:
            raise ValueError(f"pointer zip must hold only {MANIFEST_NAME}, found {z.namelist()}")
        return json.loads(z.read(MANIFEST_NAME))


# ------------------------------------------------------------------- MAST input checks
def solenoid_pattern_spread(table: cf.GreenTable, cfg: MachineConfig | None = None) -> dict:
    """Spread of the solenoid's flux pattern across the grid relative to its grid mean.

    The solenoid flux of a frame is the pattern times the solenoid current, so this
    ratio is the same for every frame and shot of a machine; it says how flat the
    calculated solenoid flux is (0 would be perfectly flat).
    """
    cfg = cfg or machine_config(table.machine)
    pattern = table.patterns[table.column_index([cfg.solenoid_column])[0]]
    mean = float(pattern.mean())
    return {"rel_std": float(pattern.std() / abs(mean)) if mean else float("inf"),
            "rel_range": float((pattern.max() - pattern.min()) / abs(mean)) if mean else float("inf")}


def mast_shot_checks(shot: io.Shot, table: cf.GreenTable, scales: cf.CoilScales | None = None,
                     min_sol_ka: float = 0.1) -> dict:
    """Input-only checks of one MAST shot for the run record.

    Returns the sign of Ip over the frames (``+1``, ``-1`` or ``0`` when mixed or
    absent), the median ``|Ip|`` in kA, the median solenoid current in kA, the flatness
    of the calculated solenoid flux (``solenoid_pattern_spread``, repeated per shot
    for the record), and the median over frames with a solenoid current above
    ``min_sol_ka`` of the solenoid flux's grid range as a fraction of the total coil
    flux's grid range (how much of the coil flux shape the solenoid carries).
    """
    cfg = shot.machine
    currents = cf.currents_at_frames(shot, table.columns)
    ip = cf.currents_at_frames(shot, [cfg.ip_column])[:, 0]
    finite_ip = ip[np.isfinite(ip)]
    signs = set(np.sign(finite_ip[np.abs(finite_ip) >= min_sol_ka]).astype(int).tolist())
    ip_sign = int(next(iter(signs))) if len(signs) == 1 else 0
    sol_idx = table.column_index([cfg.solenoid_column])[0]
    sol = currents[:, sol_idx]
    total = table.flux(currents, scales)
    sol_only = np.zeros_like(currents)
    sol_only[:, sol_idx] = sol
    sol_flux = table.flux(sol_only, scales)
    use = np.isfinite(sol) & (np.abs(sol) >= min_sol_ka)
    share = np.nan
    if use.any():
        total_range = total[use].max(axis=(1, 2)) - total[use].min(axis=(1, 2))
        sol_range = sol_flux[use].max(axis=(1, 2)) - sol_flux[use].min(axis=(1, 2))
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(total_range > 0, sol_range / total_range, np.nan)
        share = float(np.nanmedian(ratio)) if np.isfinite(ratio).any() else np.nan
    return {"ip_sign": ip_sign, "ip_abs_median_ka": float(np.median(np.abs(finite_ip))) if finite_ip.size else np.nan,
            "sol_median_ka": float(np.nanmedian(sol)) if np.isfinite(sol).any() else np.nan,
            "sol_flux_rel_std": solenoid_pattern_spread(table, cfg)["rel_std"],
            "sol_range_share_of_coil_range": share}


def summarize_mast_checks(checks: list[dict]) -> dict:
    """Fold-level summary: shots by sign of Ip and quantiles of the solenoid share."""
    signs = [c["ip_sign"] for c in checks]
    share = np.array([c["sol_range_share_of_coil_range"] for c in checks], dtype=np.float64)
    share = share[np.isfinite(share)]
    out = {"shots_by_ip_sign": {"positive": int(sum(s == 1 for s in signs)), "negative": int(sum(s == -1 for s in signs)),
                                "mixed_or_absent": int(sum(s == 0 for s in signs))}}
    if checks:
        out["sol_flux_rel_std"] = checks[0]["sol_flux_rel_std"]
    if share.size:
        out["sol_range_share_of_coil_range_quantiles"] = {q: float(np.quantile(share, float(q)))
                                                         for q in ("0.05", "0.5", "0.95")}
    return out
