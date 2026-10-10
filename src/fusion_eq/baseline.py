"""The plain PCA and ridge baseline, and local scoring with the challenge scorer.

Inputs are frame-table columns: the current signals in ``__mean`` form, the three
core Thomson summaries in ``__interp`` form (missing values filled with the training
median) and the ``ts_core_valid`` flag as 0 or 1, standardized with training
statistics. Targets are the principal components of the flux map plus ``efit_q95``
and ``efit_beta_n``, each fitted by ridge regression with one fixed regularization.

The flux basis is fitted from the per-pixel covariance accumulated shot by shot, so
the full set of maps is never held in memory, and the coefficients of every shot are
cached on disk for reuse. ``score_local`` calls the organizers' ``local_score.py``
functions (byte-identical to the platform scorer's path) on held-out training shots.

Choices that follow the starter kit's ``experiments.py`` and those that differ from it
are listed in ``BaselineParams``. Specification: ``docs/SPLIT.md``, "The baseline used
in B8c".
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from fusion_eq import dataset, frame_table as ft, io, starter_kit
from fusion_eq.machines import MachineConfig, machine_config

GRID = 65
N_PIXELS = GRID * GRID
THOMSON_INPUTS = ("ts_core_te_peak__interp", "ts_core_ne_peak__interp", "ts_core_pe_mean__interp")
VALID_FLAG = "ts_core_valid"
SCALAR_TARGETS = {"q95": "efit_q95", "betaN": "efit_beta_n"}
DEFAULT_CACHE_DIR = dataset.DEFAULT_PROCESSED_ROOT / "baseline"


@dataclass(frozen=True)
class BaselineParams:
    """Parameters of the baseline.

    Attributes:
        n_components: Principal components of the flux map. The starter kit uses 50
            (20 in its quick mode); the split specification names 20.
        alpha: Ridge regularization, one fixed value for every target. The starter kit
            tunes it with ``RidgeCV`` over 1e-3 to 1e3; with more than a million
            training frames and about 25 inputs the choice is immaterial.
        pca_frames: Frames the basis is fitted on, taken as whole shots in a seeded
            random order until the budget is reached (the starter kit fits on every
            training frame). None uses every frame given.
        seed: Seed of the shot order for the basis fit.

    Inputs differ from the starter kit by construction: it interpolates the raw
    current traces at frame times and uses every Thomson channel; the frame table
    holds the 2 ms windowed mean and three core summaries. Frames whose inputs are
    not finite are dropped from the fit, as in the starter kit, and predicted from the
    training mean (standardized zero) so that every frame receives a prediction.
    """

    n_components: int = 50
    alpha: float = 1.0
    pca_frames: int | None = 100_000
    seed: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def input_columns(machine: str | MachineConfig = "DIII-D") -> list[str]:
    """Input columns of the baseline, checked against the frame table's label guard."""
    cfg = machine if isinstance(machine, MachineConfig) else machine_config(machine)
    cols = [f"{name}__mean" for name in ft.current_signals(cfg)] + list(THOMSON_INPUTS) + [VALID_FLAG]
    return ft.assert_input_columns(cols, cfg)


# ------------------------------------------------------------------------ inputs
@dataclass
class InputScaler:
    """Training-set fill values and standardization of the input columns.

    Attributes:
        columns: Input columns in order.
        fill: Column to the training median used where a value is missing (Thomson).
        mean, std: Standardization statistics over the complete training rows.
    """

    columns: tuple[str, ...]
    fill: dict[str, float]
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, frames: pd.DataFrame, columns: Sequence[str]) -> "InputScaler":
        columns = tuple(columns)
        raw = _raw_matrix(frames, columns)
        fill = {c: float(np.nanmedian(raw[:, j])) for j, c in enumerate(columns) if c in THOMSON_INPUTS}
        filled = _fill(raw, columns, fill)
        complete = np.isfinite(filled).all(axis=1)
        if not complete.any():
            raise ValueError("no training row with complete inputs")
        mean = filled[complete].mean(axis=0)
        std = filled[complete].std(axis=0)
        std = np.where(std > 0, std, 1.0)
        return cls(columns=columns, fill=fill, mean=mean, std=std)

    def transform(self, frames: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Standardized inputs ``(rows, p)`` and a bool mask of the rows whose inputs were
        complete after the Thomson fill. Incomplete entries are set to 0 (the mean)."""
        raw = _raw_matrix(frames, self.columns)
        filled = _fill(raw, self.columns, self.fill)
        complete = np.isfinite(filled).all(axis=1)
        x = (np.where(np.isfinite(filled), filled, self.mean) - self.mean) / self.std
        return x, complete

    def to_dict(self) -> dict:
        return {"columns": list(self.columns), "fill": self.fill, "mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "InputScaler":
        return cls(tuple(d["columns"]), dict(d["fill"]), np.asarray(d["mean"], float), np.asarray(d["std"], float))


def _raw_matrix(frames: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    out = np.empty((len(frames), len(columns)))
    for j, c in enumerate(columns):
        col = frames[c]
        out[:, j] = col.astype(np.float64).to_numpy() if col.dtype != bool else col.to_numpy().astype(np.float64)
    return out


def _fill(raw: np.ndarray, columns: Sequence[str], fill: Mapping[str, float]) -> np.ndarray:
    out = raw.copy()
    for j, c in enumerate(columns):
        if c in fill:
            out[:, j] = np.where(np.isfinite(out[:, j]), out[:, j], fill[c])
    return out


# ------------------------------------------------------------------------- basis
@dataclass
class FluxBasis:
    """Principal components of the flux map.

    Attributes:
        mean: Pixel mean, ``(4225,)``.
        components: ``(k, 4225)`` orthonormal rows, each with its largest-magnitude
            entry positive.
        explained_variance: Variance along each component.
        total_variance: Sum of the pixel variances, for the explained share.
        n_frames: Frames the basis was fitted on.
    """

    mean: np.ndarray
    components: np.ndarray
    explained_variance: np.ndarray
    total_variance: float
    n_frames: int

    @property
    def n_components(self) -> int:
        return int(self.components.shape[0])

    @property
    def explained_ratio(self) -> np.ndarray:
        return self.explained_variance / self.total_variance if self.total_variance > 0 else np.zeros_like(self.explained_variance)

    def encode(self, maps) -> np.ndarray:
        """Coefficients ``(frames, k)`` of ``(frames, 65, 65)`` maps."""
        flat = np.asarray(maps, dtype=np.float64).reshape(-1, N_PIXELS)
        return (flat - self.mean) @ self.components.T

    def decode(self, coefficients) -> np.ndarray:
        """Maps ``(frames, 65, 65)`` from coefficients ``(frames, k)``."""
        c = np.asarray(coefficients, dtype=np.float64).reshape(-1, self.n_components)
        return (c @ self.components + self.mean).reshape(-1, GRID, GRID)

    def hash(self) -> str:
        return io.sha256_of_arrays(self.mean, self.components)

    def save(self, path: Path | str) -> None:
        np.savez(path, mean=self.mean, components=self.components, explained_variance=self.explained_variance,
                 total_variance=self.total_variance, n_frames=self.n_frames)

    @classmethod
    def load(cls, path: Path | str) -> "FluxBasis":
        with np.load(path, allow_pickle=False) as f:
            return cls(f["mean"], f["components"], f["explained_variance"], float(f["total_variance"]), int(f["n_frames"]))


class BasisAccumulator:
    """Accumulates the pixel mean and covariance of flux maps, shot by shot."""

    def __init__(self) -> None:
        self.sum = np.zeros(N_PIXELS)
        self.gram = np.zeros((N_PIXELS, N_PIXELS))
        self.n = 0

    def add(self, maps) -> None:
        flat = np.asarray(maps, dtype=np.float64).reshape(-1, N_PIXELS)
        self.sum += flat.sum(axis=0)
        self.gram += flat.T @ flat
        self.n += flat.shape[0]

    def finish(self, n_components: int) -> FluxBasis:
        """The basis from the accumulated sums, by eigendecomposition of the covariance."""
        if self.n < 2:
            raise ValueError("at least two frames are needed to fit a basis")
        mean = self.sum / self.n
        cov = self.gram / self.n - np.outer(mean, mean)
        cov = 0.5 * (cov + cov.T)
        k = min(n_components, N_PIXELS)
        values, vectors = np.linalg.eigh(cov)
        order = np.argsort(values)[::-1][:k]
        components = vectors[:, order].T.copy()
        sign = np.sign(components[np.arange(k), np.argmax(np.abs(components), axis=1)])
        components *= np.where(sign == 0, 1.0, sign)[:, None]
        return FluxBasis(mean=mean, components=components, explained_variance=np.maximum(values[order], 0.0),
                         total_variance=float(np.trace(cov)), n_frames=self.n)


def fit_flux_basis(maps_by_shot: Iterable[np.ndarray], n_components: int, max_frames: int | None = None) -> FluxBasis:
    """Fits the basis over shots' maps, stopping once ``max_frames`` have been added."""
    acc = BasisAccumulator()
    for maps in maps_by_shot:
        acc.add(maps)
        if max_frames is not None and acc.n >= max_frames:
            break
    return acc.finish(n_components)


def basis_shot_order(shot_ids: Sequence[str], seed: int) -> list[str]:
    """Seeded random order of shots for the basis fit."""
    rng = np.random.default_rng(seed)
    ids = sorted(shot_ids)
    return [ids[i] for i in rng.permutation(len(ids))]


def shot_maps(shot_ids: Iterable[str], data_root: Path | str, split: str = dataset.LABELLED_SPLITS[0]):
    """Yields ``(shot_id, maps)`` with maps ``(frames, 65, 65)`` float32, one shot at a time."""
    for shot_id in shot_ids:
        yield shot_id, ft.read_flux_maps([shot_id], split, data_root)


def encode_shots(shot_ids: Sequence[str], basis: FluxBasis, data_root: Path | str,
                 cache_dir: Path | str = DEFAULT_CACHE_DIR, split: str = dataset.LABELLED_SPLITS[0]) -> np.ndarray:
    """Coefficients of the listed shots in frame-table order (sorted ids, frames in time).

    Each shot's coefficients are cached as ``<cache_dir>/<basis hash>/<shot>.npy``
    (float32) and read from there on the next call with the same basis.
    """
    folder = Path(cache_dir) / basis.hash()[:16]
    folder.mkdir(parents=True, exist_ok=True)
    parts = []
    for shot_id in sorted(set(shot_ids)):
        path = folder / f"{shot_id}.npy"
        if path.is_file():
            parts.append(np.load(path))
            continue
        coefs = basis.encode(ft.read_flux_maps([shot_id], split, data_root)).astype(np.float32)
        tmp = path.with_suffix(".tmp.npy")
        np.save(tmp, coefs)
        tmp.replace(path)
        parts.append(coefs)
    return np.concatenate(parts, axis=0) if parts else np.zeros((0, basis.n_components), dtype=np.float32)


# ------------------------------------------------------------------------- ridge
def ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """Ridge regression with an unpenalized intercept: ``(weights (p, m), intercept (m,))``."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    y = y[:, None] if y.ndim == 1 else y
    x_mean, y_mean = x.mean(axis=0), y.mean(axis=0)
    xc, yc = x - x_mean, y - y_mean
    w = np.linalg.solve(xc.T @ xc + alpha * np.eye(x.shape[1]), xc.T @ yc)
    return w, y_mean - x_mean @ w


@dataclass
class PcaRidge:
    """The fitted baseline: scaler, basis and ridge weights for the coefficients and scalars."""

    params: BaselineParams
    scaler: InputScaler
    basis: FluxBasis
    map_weights: np.ndarray
    map_intercept: np.ndarray
    scalar_weights: dict[str, np.ndarray] = field(default_factory=dict)
    scalar_intercept: dict[str, float] = field(default_factory=dict)
    n_train_frames: int = 0

    def save(self, path: Path | str) -> None:
        """Writes an npz with the arrays and a JSON entry with the rest."""
        meta = {"params": self.params.to_dict(), "scaler": self.scaler.to_dict(),
                "scalar_intercept": self.scalar_intercept, "n_train_frames": self.n_train_frames,
                "scalars": list(self.scalar_weights)}
        np.savez(path, meta=np.array(json.dumps(meta, sort_keys=True)), basis_mean=self.basis.mean,
                 basis_components=self.basis.components, basis_explained=self.basis.explained_variance,
                 basis_total=self.basis.total_variance, basis_n=self.basis.n_frames,
                 map_weights=self.map_weights, map_intercept=self.map_intercept,
                 **{f"scalar_{k}": v for k, v in self.scalar_weights.items()})

    @classmethod
    def load(cls, path: Path | str) -> "PcaRidge":
        with np.load(path, allow_pickle=False) as f:
            meta = json.loads(str(f["meta"]))
            basis = FluxBasis(f["basis_mean"], f["basis_components"], f["basis_explained"],
                              float(f["basis_total"]), int(f["basis_n"]))
            return cls(params=BaselineParams(**meta["params"]), scaler=InputScaler.from_dict(meta["scaler"]),
                       basis=basis, map_weights=f["map_weights"], map_intercept=f["map_intercept"],
                       scalar_weights={k: f[f"scalar_{k}"] for k in meta["scalars"]},
                       scalar_intercept={k: float(v) for k, v in meta["scalar_intercept"].items()},
                       n_train_frames=int(meta["n_train_frames"]))


def fit_pca_ridge(train_frames: pd.DataFrame, coefficients: np.ndarray, basis: FluxBasis,
                  params: BaselineParams = BaselineParams(), scaler: InputScaler | None = None) -> PcaRidge:
    """Fits the ridge for the map coefficients and the two scalars.

    Args:
        train_frames: Frame-table rows of the training shots, in frame-table order, with
            the input columns and the label columns ``efit_q95`` and ``efit_beta_n``.
        coefficients: ``(rows, k)`` coefficients of the same rows (``encode_shots``).
        basis: The basis the coefficients were encoded with.
        params: Baseline parameters.
        scaler: A scaler fitted earlier; fitted on ``train_frames`` when None, so that
            arms of one experiment can share it.

    Rows with incomplete inputs are dropped from every fit; a scalar's fit also drops
    rows where that label is not finite.
    """
    columns = input_columns()
    if coefficients.shape[0] != len(train_frames):
        raise ValueError("coefficients and frames differ in row count")
    scaler = InputScaler.fit(train_frames, columns) if scaler is None else scaler
    x, complete = scaler.transform(train_frames)
    w, b = ridge_fit(x[complete], np.asarray(coefficients, dtype=np.float64)[complete], params.alpha)
    model = PcaRidge(params=params, scaler=scaler, basis=basis, map_weights=w, map_intercept=b,
                     n_train_frames=int(complete.sum()))
    for name, column in SCALAR_TARGETS.items():
        y = train_frames[column].to_numpy(dtype=np.float64)
        keep = complete & np.isfinite(y)
        ws, bs = ridge_fit(x[keep], y[keep], params.alpha)
        model.scalar_weights[name] = ws[:, 0]
        model.scalar_intercept[name] = float(bs[0])
    return model


@dataclass
class Prediction:
    """One map and two scalars per frame, in the order of the frames predicted."""

    psirz: np.ndarray
    q95: np.ndarray
    betaN: np.ndarray

    def as_scorer_dict(self) -> dict:
        return {"psirz": self.psirz.astype(np.float64), "q95": self.q95.astype(np.float64),
                "betaN": self.betaN.astype(np.float64)}


def predict(model: PcaRidge, frames: pd.DataFrame) -> Prediction:
    """Predicts the frames given, one map (float32) and two scalars each, in row order."""
    x, _ = model.scaler.transform(frames)
    coefs = x @ model.map_weights + model.map_intercept
    out = Prediction(psirz=model.basis.decode(coefs).astype(np.float32),
                     q95=x @ model.scalar_weights["q95"] + model.scalar_intercept["q95"],
                     betaN=x @ model.scalar_weights["betaN"] + model.scalar_intercept["betaN"])
    return out


# ----------------------------------------------------------------------- scoring
@lru_cache(maxsize=1)
def local_score_module() -> ModuleType:
    """Imports the organizers' ``local_score.py`` from the starter kit, unmodified."""
    starter_kit.scoring()                                   # puts fusion_scoring on sys.path
    path = starter_kit.require() / "local_score.py"
    spec = importlib.util.spec_from_file_location("starter_local_score", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _strided(a: np.ndarray, stride: int) -> np.ndarray:
    return a[::stride] if stride > 1 else a


def load_truth(shot_id: str, data_root: Path | str, split: str = dataset.LABELLED_SPLITS[0], stride: int = 1) -> dict:
    """The scorer's ground-truth record of one shot: ``psi`` (float32), ``q95``, ``betaN``."""
    shot = io.load_shot(dataset.shot_path(data_root, split, shot_id), ["efit_psirz", "efit_q95", "efit_beta_n"], apply=False)
    return {"psi": _strided(np.asarray(shot["efit_psirz"], dtype=np.float32), stride),
            "q95": _strided(np.asarray(shot["efit_q95"], dtype=np.float64), stride),
            "betaN": _strided(np.asarray(shot["efit_beta_n"], dtype=np.float64), stride)}


@dataclass
class LocalScore:
    """Per-shot scoring parts, so that any subset of the shots can be finalized.

    Attributes:
        shot_ids: Scored shots in order.
        parts: The scorer's per-shot contributions (``local_score.score_shot``).
        ref_stats: Per-shot reference sums (the scorer's ``ref_stats`` entries).
        psi_sign: The global flux sign chosen over all shots.
        frame_stride: Stride applied to the frames of every shot.
    """

    shot_ids: list[str]
    parts: list[dict]
    ref_stats: list[dict]
    psi_sign: float
    frame_stride: int

    def finalize(self, mask=None) -> dict:
        """The composite score and its terms over the shots selected by ``mask`` (all when None)."""
        ls = local_score_module()
        keep = np.ones(len(self.shot_ids), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
        acc = ls.Accum()
        acc.psi_sign = self.psi_sign
        totals: dict = {}
        for part, ref, k in zip(self.parts, self.ref_stats, keep):
            if not k:
                continue
            acc.add(part)
            for key, value in ref.items():
                totals[key] = totals.get(key, 0.0) + value
        if acc.n_shots_scored == 0:
            return {"S": np.nan, "n_shots": 0}
        return ls.finalize_machine(acc, totals)


@dataclass
class References:
    """The scorer's reference targets of a set of shots, built once and reused by every
    scoring of the same shots (the arms of one experiment).

    Attributes:
        shot_ids: Shots in order.
        refs: Per shot, ``(contours, cons, cmask, rgeo)`` from ``local_score.build_reference``.
        ref_stats: Per shot, the reference sums the scorer pools.
        frame_stride: Stride the truth was read with.
    """

    shot_ids: list[str]
    refs: list[tuple]
    ref_stats: list[dict]
    frame_stride: int

    @property
    def means(self) -> tuple:
        """``(mean_psi, mean_scal, mean_cons)`` over all shots, the scorer's fill values."""
        psi_sum = sum(r["psi_sum"] for r in self.ref_stats)
        psi_n = sum(r["psi_n"] for r in self.ref_stats)
        scal_sum = sum(r["scal_sum"] for r in self.ref_stats)
        scal_n = sum(r["scal_n"] for r in self.ref_stats)
        cons_sum = sum(r["cons_sum"] for r in self.ref_stats)
        cons_n = sum(r["cons_n"] for r in self.ref_stats)
        return (psi_sum / psi_n if psi_n else 0.0,
                np.where(scal_n > 0, scal_sum / np.where(scal_n > 0, scal_n, 1), 0.0),
                np.where(cons_n > 0, cons_sum / np.where(cons_n > 0, cons_n, 1), 0.0))


def build_references(shot_ids: Sequence[str], data_root: Path | str, split: str = dataset.LABELLED_SPLITS[0],
                     frame_stride: int = 1) -> References:
    """Builds the scorer's reference (boundary and seven derived scalars per frame, and
    the pooled sums) from the true maps of the shots, as ``local_score.py`` does."""
    ls = local_score_module()
    env = starter_kit.envelope_mask(machine_config("DIII-D"))
    refs, ref_stats = [], []
    for shot_id in shot_ids:
        truth = load_truth(shot_id, data_root, split, frame_stride)
        contours, cons, cmask, rgeo = ls.build_reference(truth["psi"], env.grid_R, env.grid_Z, env.mask_coarse, env.mask_f)
        g = truth["psi"].astype(np.float64)
        fin = np.isfinite(g)
        stats = {"psi_sum": float(g[fin].sum()), "psi_sumsq": float((g[fin] ** 2).sum()), "psi_n": int(fin.sum()),
                 "scal_sum": np.zeros(ls.N_SCALARS), "scal_sumsq": np.zeros(ls.N_SCALARS), "scal_n": np.zeros(ls.N_SCALARS),
                 "cons_sum": np.zeros(ls.N_CONS), "cons_sumsq": np.zeros(ls.N_CONS), "cons_n": np.zeros(ls.N_CONS)}
        for j, name in enumerate(ls.SCALARS):
            v = truth[name][np.isfinite(truth[name])]
            stats["scal_sum"][j], stats["scal_sumsq"][j], stats["scal_n"][j] = v.sum(), (v ** 2).sum(), v.size
        for j in range(ls.N_CONS):
            v = cons[cmask[:, j], j]
            stats["cons_sum"][j], stats["cons_sumsq"][j], stats["cons_n"][j] = v.sum(), (v ** 2).sum(), v.size
        refs.append((contours, cons, cmask, rgeo))
        ref_stats.append(stats)
    return References(shot_ids=list(shot_ids), refs=refs, ref_stats=ref_stats, frame_stride=frame_stride)


def score_local(predict_shot: Callable[[str], Prediction] | Mapping[str, Prediction], shot_ids: Sequence[str],
                data_root: Path | str, split: str = dataset.LABELLED_SPLITS[0], frame_stride: int = 1,
                references: References | None = None) -> LocalScore:
    """Scores predictions on labelled shots with the organizers' local scorer.

    The scorer's reference is built from the true maps (or taken from ``references``),
    the global flux sign is decided once over all shots from the flux residuals, and
    every shot is then scored under it, exactly as ``local_score.py`` does. True maps
    are loaded per pass rather than kept in memory.

    Args:
        predict_shot: Shot id to ``Prediction`` for that shot's (strided) frames, or a
            mapping with the same content.
        shot_ids: Shots to score.
        data_root: Raw data root.
        split: Labelled split of the shots.
        frame_stride: Keep every ``frame_stride``-th frame of each shot, in the truth
            and in the prediction alike.
        references: References built earlier for exactly these shots and stride.
    """
    ls = local_score_module()
    env = starter_kit.envelope_mask(machine_config("DIII-D"))
    get = predict_shot.__getitem__ if isinstance(predict_shot, Mapping) else predict_shot
    if references is None:
        references = build_references(shot_ids, data_root, split, frame_stride)
    if references.shot_ids != list(shot_ids) or references.frame_stride != frame_stride:
        raise ValueError("references were built for other shots or another frame stride")
    means = references.means
    # The global flux sign, from the flux residuals of every shot (the scorer's probe pass).
    predictions: dict[str, Prediction] = {}
    residual_totals = {s: 0.0 for s in ls.PSI_SIGNS}
    for shot_id in shot_ids:
        truth = load_truth(shot_id, data_root, split, frame_stride)
        pred = get(shot_id)
        predictions[shot_id] = pred
        rr, _, _ = ls.psi_residuals(truth["psi"], pred.psirz.astype(np.float64), means[0])
        for s in ls.PSI_SIGNS:
            residual_totals[s] += rr[s]
    psi_sign = min(ls.PSI_SIGNS, key=lambda s: residual_totals[s])
    parts = []
    for shot_id, ref in zip(shot_ids, references.refs):
        truth = load_truth(shot_id, data_root, split, frame_stride)
        parts.append(ls.score_shot(truth, ref, predictions[shot_id].as_scorer_dict(), env.grid_R, env.grid_Z,
                                   env.mask_coarse, env.mask_f, psi_sign, means))
    return LocalScore(shot_ids=list(shot_ids), parts=parts, ref_stats=list(references.ref_stats), psi_sign=psi_sign,
                      frame_stride=frame_stride)


def score_terms(result: dict) -> dict:
    """S and its four terms, plus the seven consistency values, from a finalized result."""
    if not np.isfinite(result.get("S", np.nan)):
        return {"S": np.nan}
    out = {"S": result["S"], "r2_psi": result["r2_psi"], "r2_qb": result["r2_qb"],
           "one_minus_dlcfs": 1.0 - min(1.0, result["dlcfs"]), "consistency": result["consistency"],
           "psi_sign": result["psi_sign"], "n_shots": result["n_shots"]}
    out.update({f"r2_{k}": (np.nan if v is None else v) for k, v in result["r2_cons_each"].items()})
    out.update({f"r2_{k}": v for k, v in result["r2_each"].items()})
    return out


def frames_of_shots(shot_ids: Sequence[str], columns: Sequence[str], table_root: Path | str = ft.DEFAULT_TABLE_ROOT,
                    split: str = dataset.LABELLED_SPLITS[0]) -> pd.DataFrame:
    """Frame-table rows of the shots with the columns needed, in frame-table order."""
    cols = list(dict.fromkeys(["shot", "frame", "time_ms", *columns]))
    return ft.read_frame_table(split, cols, shot_ids, root=table_root)


def hash_of_ids(shot_ids: Iterable[str]) -> str:
    """Short hash of a sorted id list, for naming cached fits."""
    return hashlib.sha256("\n".join(sorted(shot_ids)).encode()).hexdigest()[:12]
