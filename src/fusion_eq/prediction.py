"""Prediction chain of the plasma model and the ``rung1_v1`` predictor (``docs/PIPELINE.md``,
Part 5).

Per shot, always one shot at a time so that a prediction never depends on which other shots
are predicted with it:

1. Inputs: the frame-table rows and the unscaled version-2 features of the shot, from the
   stored tables (``source="tables"``) or rebuilt from the raw shot file through
   ``frame_table.build_shot_frames`` and ``features.build_features`` with the stored scaler
   (``source="raw"``, D7). Every frame-table column read passes the label guard.
2. Coefficients: the scaled features of the model's columns, clipped to ``loader.clip_bound``,
   through the model. Mirror averaging (``predict.mirror_average``): the prediction is averaged
   with the parity-flipped prediction of the mirrored features (``mirror_features`` before
   scaling, as in training). Seed averaging (``predict.seed_average``): the average over the
   checkpoints of the arm's seeds.
3. Map: the standardized coefficients times the component standard deviations, through the
   machine's flux basis of the target build (share, placement and plasma response in one
   matrix), plus the machine's coil flux under its adopted coil model, plus the ridge
   remainder on DIII-D (zero on MAST), assembled by ``compose_map`` with the machine's sign
   constant. The MAST constant +1 is asserted before every MAST map (F71, Z12).
   A sequence model (rung 2) runs over every frame of the shot with the time steps of the frame
   times; where only some frames are wanted (strided lists), the rows carry a ``scored`` flag,
   the coefficients come from the whole shot and the maps and scalars from the flagged frames.
4. Scalars: ``q95_from_map`` on every predicted map (the machine's verified toroidal-field
   turns, no fitted constant); betaN by the per-machine heads of ``scalar_heads``.

The model runs on the CPU by default: the outputs are then reproducible bit for bit on any
machine. Tensors leave the device before any float64 conversion, so the MPS backend (no
float64) also works.
"""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from fusion_eq import (baseline as bl, current_basis as cb, dataset, features as fe, frame_table as ft, io, loader as ld,
                       plasma_models as pm, predictors, scalar_heads as sh, scalars as sc, split as sp, starter_kit,
                       submission, target_build as tb, targets)
from fusion_eq.coil_flux import CoilScales, GreenTable, adopted_scales
from fusion_eq.feature_scaler import RobustScaler
from fusion_eq.machines import machine_config
from fusion_eq.model_config import ModelConfig

MACHINES = ("DIII-D", "MAST")
MACHINE_KEYS = {"d3d": "DIII-D", "mast": "MAST"}
SOURCES = ("tables", "raw")
KEY_COLUMNS = ("shot", "frame")
SCORED = "scored"
DEFAULT_RUN_ROOT = dataset.REPO_ROOT / "reports" / "runs"
DEFAULT_CHECKPOINT_ROOT = dataset.DEFAULT_PROCESSED_ROOT.parent / "runs"
MAST_SIGN = 1.0


# ---------------------------------------------------------------------------- guards
def assert_mast_sign() -> float:
    """The MAST sign constant, asserted to be +1 (F71, F100).

    Raises:
        ValueError: when the constant in ``targets.PSI_SIGN_RULE_C`` is not +1.
    """
    c = targets.sign_rule_c("MAST")
    if c != MAST_SIGN:
        raise ValueError(f"the MAST sign constant is {c:+g}; F71 and F100 established +1, and no MAST map is written under another")
    return c


def require_inputs(columns: Iterable[str], machine: str) -> list[str]:
    """The frame-table columns of a read, after the label guard (keys exempt).

    Raises:
        ValueError: when a column is a label (``frame_table.assert_input_columns``).
    """
    columns = list(dict.fromkeys(columns))
    ft.assert_input_columns([c for c in columns if c not in KEY_COLUMNS], machine)
    return columns


def split_of(shot: io.Shot) -> str:
    """The data split a raw shot file belongs to (its folder name)."""
    name = Path(shot.path).parent.name
    if name not in dataset.SPLIT_MACHINE:
        raise ValueError(f"{shot.path}: folder {name!r} is not a data split")
    return name


# --------------------------------------------------------------------------- members
@dataclass
class Member:
    """One trained checkpoint of an arm.

    Attributes:
        run_id: Training run.
        path: Checkpoint file.
        sha256: SHA-256 of the checkpoint file.
        model: The model in eval mode.
        meta: Checkpoint metadata (columns, parity, sigma, configuration, epoch).
    """

    run_id: str
    path: Path
    sha256: str
    model: torch.nn.Module
    meta: dict

    @property
    def config(self) -> ModelConfig:
        return ModelConfig.from_dict(self.meta["config"])


def sha256_of(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_members(run_ids: Sequence[str], checkpoint_root: Path | str = DEFAULT_CHECKPOINT_ROOT, which: str = "best",
                 device: torch.device | str = "cpu") -> list[Member]:
    """The checkpoints ``<checkpoint_root>/<run_id>/<which>.pt`` of the given runs.

    Raises:
        FileNotFoundError: for a missing checkpoint.
        ValueError: when the runs differ in anything but ``train.seed`` (columns, parity,
            standard deviations, feature and target versions, configuration).
    """
    if not run_ids:
        raise ValueError("no run given")
    members = []
    for run_id in run_ids:
        path = Path(checkpoint_root) / run_id / f"{which}.pt"
        if not path.is_file():
            raise FileNotFoundError(f"{path} not found")
        model, meta = pm.load_checkpoint(path, device)
        members.append(Member(run_id=run_id, path=path, sha256=sha256_of(path), model=model, meta=meta))
    first = members[0]
    for m in members[1:]:
        for key in ("columns", "parity", "sigma", "feature_version", "target_version"):
            if m.meta[key] != first.meta[key]:
                raise ValueError(f"runs {first.run_id} and {m.run_id} differ in {key}")
        a = first.config.with_values({"train.seed": 0}).hash()
        b = m.config.with_values({"train.seed": 0}).hash()
        if a != b:
            raise ValueError(f"runs {first.run_id} and {m.run_id} differ in configuration beyond train.seed")
    return members


# ----------------------------------------------------------------------- coefficients
class CoefficientModel:
    """Standardized coefficients of unscaled feature rows, with mirror and seed averaging.

    Args:
        members: Checkpoints; their outputs are averaged.
        fconfig: Feature configuration (mirror rules).
        scaler: The stored DIII-D scaler, applied unchanged on both machines.
        clip_bound: Bound of the scaled features.
        mirror_average: Average with the flipped-back prediction of the mirrored features.
        device: Torch device of the forward pass.
    """

    def __init__(self, members: Sequence[Member], fconfig: fe.FeatureConfig, scaler: RobustScaler, clip_bound: float,
                 mirror_average: bool = True, device: torch.device | str = "cpu") -> None:
        self.members = list(members)
        self.fconfig = fconfig
        self.scaler = scaler
        self.clip_bound = float(clip_bound)
        self.mirror_average = bool(mirror_average)
        self.device = torch.device(device)
        meta = self.members[0].meta
        self.columns = list(meta["columns"])
        self.parity = np.asarray(meta["parity"], dtype=np.float64)
        self.sigma = np.asarray(meta["sigma"], dtype=np.float64)
        self.sequence = pm.is_sequence(self.members[0].model)
        if any(pm.is_sequence(m.model) != self.sequence for m in self.members):
            raise ValueError("checkpoints mix frame and sequence models")
        self.dt_scale_ms = float(self.members[0].config.get("rung2.dt_scale_ms"))

    def scaled(self, features: pd.DataFrame, machine: str, mirror: bool = False) -> np.ndarray:
        """Scaled, clipped float32 features of the model's columns (mirrored before scaling)."""
        table = fe.mirror_features(features, self.fconfig) if mirror else features
        x = self.scaler.transform(table, machine, self.columns).astype(np.float32)
        return np.clip(x, -self.clip_bound, self.clip_bound)

    def forward(self, x: np.ndarray, x_mirror: np.ndarray | None, dt: np.ndarray | None = None) -> np.ndarray:
        """``(N, K)`` float64 standardized coefficients from scaled features (and, for a sequence
        model, the time-step input of one shot's frames in time order)."""
        out = np.zeros((len(x), len(self.sigma)))

        def run(model, features):
            return pm.predict_sequence(model, features, dt, self.device) if self.sequence else pm.predict(model, features, self.device)

        for m in self.members:
            c = run(m.model, x)
            if x_mirror is not None:
                c = 0.5 * (c + run(m.model, x_mirror) * self.parity)
            out += c
        return out / len(self.members)

    def predict(self, features: pd.DataFrame, machine: str) -> np.ndarray:
        """``(N, K)`` standardized coefficients of unscaled feature rows of one machine; a
        sequence model runs on each shot's rows (in the order given, which must be time order)
        separately, with ``time_ms``."""
        x = self.scaled(features, machine)
        x_mirror = self.scaled(features, machine, mirror=True) if self.mirror_average else None
        if not self.sequence:
            return self.forward(x, x_mirror)
        shots = features["shot"].astype(str).to_numpy()
        dt = ld.dt_input(features["time_ms"].to_numpy(dtype=np.float64), shots, self.dt_scale_ms, self.clip_bound)
        out = np.zeros((len(x), len(self.sigma)))
        for start, end in ld.shot_runs(shots):
            out[start:end] = self.forward(x[start:end], None if x_mirror is None else x_mirror[start:end], dt[start:end])
        return out


# ------------------------------------------------------------------------------- maps
class MapComposer:
    """Stored-frame maps of standardized coefficients on either machine.

    Args:
        objects: The fitted objects of the target build (flux bases, remainder model).
        tables: Machine to its coil Green's table.
        scales: Machine to its adopted coil model.
        sigma: Component standard deviations.
    """

    def __init__(self, objects: tb.TargetObjects, tables: Mapping[str, GreenTable], scales: Mapping[str, CoilScales],
                 sigma: np.ndarray) -> None:
        self.objects = objects
        self.tables = dict(tables)
        self.scales = dict(scales)
        self.sigma = np.asarray(sigma, dtype=np.float64)

    def frame_columns(self, machine: str) -> list[str]:
        """Frame-table columns the maps read on a machine."""
        cols = [*targets.current_columns_of(self.tables[machine], machine), "ip__mean"]
        if machine == tb.MACHINE:
            model = self.objects.remainder_model
            cols += list(model.ridges[model.default_window].columns)
        return list(dict.fromkeys(cols))

    def maps(self, machine: str, frames: pd.DataFrame, c_std: np.ndarray) -> np.ndarray:
        """``(N, nZ, nR)`` float64 stored-frame maps.

        Raises:
            ValueError: on MAST, when the sign constant is not +1.
        """
        if machine == "MAST":
            assert_mast_sign()
        cfg = machine_config(machine)
        ip = frames["ip__mean"].to_numpy(dtype=np.float64)
        coil = targets.coil_flux_of_frames(self.tables[machine], frames, self.scales[machine], cfg)
        remainder = self.objects.remainder_model.predict(frames) if machine == tb.MACHINE else None
        c = np.asarray(c_std, dtype=np.float64) * self.sigma
        return cb.compose_map(cfg, coil, ip, c, self.objects.flux_bases[machine], remainder)


# ---------------------------------------------------------------------------- scalars
def _map_scalars(machine: str, psi: np.ndarray, tf_ka: np.ndarray, grid_r: np.ndarray, grid_z: np.ndarray,
                 with_shapes: bool, with_q95: bool = True) -> tuple[np.ndarray, list[dict] | None]:
    """q95 of each map (NaN when not asked) and, when asked, its boundary shape (scorer
    functions). Runs in a worker process."""
    cfg = machine_config(machine)
    scorer = starter_kit.scoring()
    envelope = starter_kit.envelope_mask(cfg)
    n_turns = sc.TF_TURNS[cfg.name]
    q95 = np.array([sc.q95_from_map(p, grid_r, grid_z, t, n_turns, cfg, envelope=envelope, scorer=scorer)
                    for p, t in zip(psi, tf_ka)], dtype=np.float64) if with_q95 else np.full(len(psi), np.nan)
    shapes = [sc.shape_from_map(p, grid_r, grid_z, cfg, envelope, scorer) for p in psi] if with_shapes else None
    return q95, shapes


class ScalarWorkers:
    """Computes q95 and shapes of a shot's maps in chunks over a process pool (sequentially
    for ``workers <= 1``). The values do not depend on the number of workers."""

    def __init__(self, workers: int | None = 1, chunk: int = 64) -> None:
        self.workers = bl.default_workers() if workers is None else int(workers)
        self.chunk = int(chunk)
        self._pool: ProcessPoolExecutor | None = None

    def __call__(self, machine: str, psi: np.ndarray, tf_ka: np.ndarray, grid_r, grid_z,
                 with_shapes: bool, with_q95: bool = True) -> tuple[np.ndarray, list[dict] | None]:
        n = len(psi)
        r, z = np.asarray(grid_r, dtype=np.float64), np.asarray(grid_z, dtype=np.float64)
        if self.workers <= 1 or n <= self.chunk:
            return _map_scalars(machine, psi, tf_ka, r, z, with_shapes, with_q95)
        futures = [self.pool().submit(_map_scalars, machine, psi[s:s + self.chunk], tf_ka[s:s + self.chunk], r, z, with_shapes,
                                      with_q95) for s in range(0, n, self.chunk)]
        parts = [f.result() for f in futures]
        q95 = np.concatenate([p[0] for p in parts])
        shapes = [row for p in parts for row in p[1]] if with_shapes else None
        return q95, shapes

    def pool(self) -> ProcessPoolExecutor:
        if self._pool is None:
            self._pool = ProcessPoolExecutor(max_workers=self.workers, mp_context=get_context("spawn"))
        return self._pool

    def map_shots(self, items: Iterable[tuple]) -> Iterable[tuple[np.ndarray, list[dict] | None]]:
        """``_map_scalars`` of many shots' argument tuples, in order, a window of shots in flight
        at once (sequentially for ``workers <= 1``)."""
        if self.workers <= 1:
            for item in items:
                yield _map_scalars(*item)
            return
        pending: list = []
        for item in items:
            pending.append(self.pool().submit(_map_scalars, *item))
            if len(pending) >= 2 * self.workers:
                yield pending.pop(0).result()
        for future in pending:
            yield future.result()

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown()
            self._pool = None


def shapes_frame(shot_ids, frame_ids, shapes: Sequence[dict]) -> pd.DataFrame:
    """``shape_from_map`` results as rows ``shot, frame, R_geo, a, kappa, tri_top, tri_bot, eps,
    boundary_found``."""
    rows = [{"shot": str(s), "frame": int(f), **d} for s, f, d in zip(shot_ids, frame_ids, shapes)]
    return pd.DataFrame(rows, columns=["shot", "frame", *sc.SHAPE_COLUMNS, "boundary_found"])


# ----------------------------------------------------------------------------- inputs
class TableSource:
    """Frame-table rows and unscaled features of shots from the stored tables.

    Args:
        table_root: Frame-table build.
        feature_root: Feature builds (``<root>/v<version>``).
        version: Feature version.
        frame_columns: Machine to the frame-table columns read (label-guarded).
        feature_columns: Feature columns read.
    """

    def __init__(self, table_root: Path | str, feature_root: Path | str, version: int, frame_columns: Mapping[str, Sequence[str]],
                 feature_columns: Sequence[str]) -> None:
        self.table_root = Path(table_root)
        self.feature_root = Path(feature_root)
        self.version = int(version)
        self.frame_columns = {m: require_inputs(c, m) for m, c in frame_columns.items()}
        self.feature_columns = list(dict.fromkeys(feature_columns))
        self._cache: dict[str, dict[str, pd.DataFrame]] = {}

    def prefetch(self, split: str, shot_ids: Sequence[str]) -> None:
        """Reads the rows of many shots of a split at once (one pass over each table)."""
        rows = self._read(split, shot_ids)
        self._cache.setdefault(split, {}).update({s: g.reset_index(drop=True) for s, g in rows.groupby("shot", sort=False)})

    def _read(self, split: str, shot_ids: Sequence[str] | None) -> pd.DataFrame:
        machine = dataset.SPLIT_MACHINE[split]
        frames = ft.read_frame_table(split, ["shot", "frame", *self.frame_columns[machine]], shot_ids, root=self.table_root)
        feats = fe.read_features(split, self.version, self.feature_columns, shot_ids, self.feature_root)
        return merge_inputs(frames, feats)

    def rows(self, split: str, shot_id: str) -> pd.DataFrame:
        cached = self._cache.get(split, {}).pop(str(shot_id), None)
        rows = cached if cached is not None else self._read(split, [str(shot_id)])
        if rows.empty:
            raise KeyError(f"shot {shot_id} has no rows in the stored {split} tables")
        return rows


class RawSource:
    """Frame-table rows and unscaled features of a raw shot, built by the table builders'
    own functions.

    Args:
        params: Frame-table build parameters (those of the stored build).
        fconfig: Feature configuration.
        resources: Machine to its feature resources (point table, coil model, constants).
        frame_columns: Machine to the frame-table columns kept (label-guarded).
        feature_columns: Feature columns kept.
    """

    def __init__(self, params: ft.BuildParams, fconfig: fe.FeatureConfig, resources: Mapping[str, fe.MachineResources],
                 frame_columns: Mapping[str, Sequence[str]], feature_columns: Sequence[str]) -> None:
        self.params = params
        self.fconfig = fconfig
        self.resources = dict(resources)
        self.frame_columns = {m: require_inputs(c, m) for m, c in frame_columns.items()}
        self.feature_columns = list(dict.fromkeys(feature_columns))

    def rows_of_shot(self, shot: io.Shot, split: str) -> pd.DataFrame:
        machine = shot.machine.name
        frames = ft.build_shot_frames(submission.strip_labels(shot), split, self.params, with_labels=False)
        feats = fe.build_features(frames, machine, self.fconfig, self.resources[machine])
        columns = self.fconfig.all_columns()
        # The stored feature tables hold float32; the cast makes the two sources identical.
        feats[columns] = feats[columns].astype(np.float32)
        feats["frame"] = feats["frame"].astype(np.int32)
        return merge_inputs(frames[["shot", "frame", *self.frame_columns[machine]]], feats[["shot", "frame", *self.feature_columns]])


def merge_inputs(frames: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    """Frame-table rows joined with their feature rows on ``(shot, frame)``, in shot and frame
    order.

    Raises:
        ValueError: when a column name is in both tables or a frame lacks a feature row.
    """
    frames = frames.assign(shot=frames["shot"].astype(str))
    feats = feats.assign(shot=feats["shot"].astype(str))
    both = (set(frames.columns) & set(feats.columns)) - set(KEY_COLUMNS)
    if both:
        raise ValueError(f"columns in both the frame table and the features: {sorted(both)}")
    out = frames.merge(feats, on=list(KEY_COLUMNS), how="left", validate="one_to_one", indicator=True)
    absent = int((out.pop("_merge") != "both").sum())
    if absent:
        raise ValueError(f"{absent} frames have no feature row")
    return out.sort_values(list(KEY_COLUMNS), kind="stable").reset_index(drop=True)


# ------------------------------------------------------------------------------ chain
@dataclass
class ShotPrediction:
    """One shot's predictions.

    Attributes:
        c_std: ``(N, K)`` standardized coefficients.
        psi: ``(N, nZ, nR)`` float64 stored-frame maps.
        q95, beta_n: ``(N,)`` scalars.
        shapes: On MAST (and when asked), the boundary shapes of the maps.

    ``psi`` is None when a bulk prediction was asked not to keep the maps.
    """

    c_std: np.ndarray
    psi: np.ndarray | None
    q95: np.ndarray
    beta_n: np.ndarray
    shapes: pd.DataFrame | None = None

    def as_prediction(self) -> bl.Prediction:
        return bl.Prediction(psirz=self.psi.astype(np.float32), q95=self.q95, betaN=self.beta_n)


class PredictionChain:
    """Coefficients, maps and scalars of input rows (``merge_inputs`` rows of one shot).

    Args:
        coefficients: The model with its averaging.
        composer: The map assembly.
        heads: The betaN heads; betaN is NaN when None.
        scalar_workers: Process pool for q95 and shapes.
    """

    def __init__(self, coefficients: CoefficientModel, composer: MapComposer, heads: sh.BetanHeads | None = None,
                 scalar_workers: ScalarWorkers | None = None) -> None:
        self.coefficients = coefficients
        self.composer = composer
        self.heads = heads
        self.scalar_workers = scalar_workers or ScalarWorkers(1)

    def frame_columns(self, machine: str) -> list[str]:
        """Frame-table columns the chain reads on a machine."""
        cols = [*self.composer.frame_columns(machine), "tf__mean", "ip__mean", sh.PRESSURE_COLUMN, "time_ms"]
        if machine == "DIII-D":
            cols += list(sh.FRAME_INPUT_COLUMNS)
        return list(dict.fromkeys(cols))

    def feature_columns(self) -> list[str]:
        """Feature columns the chain reads (the model's and the MAST head's)."""
        return list(dict.fromkeys([*self.coefficients.columns, *sh.FEATURE_COLUMNS]))

    def coefficients_of(self, rows: pd.DataFrame, machine: str) -> np.ndarray:
        return self.coefficients.predict(rows, machine)

    def predict(self, rows: pd.DataFrame, machine: str, with_scalars: bool = True, with_shapes: bool | None = None,
                c_std: np.ndarray | None = None) -> ShotPrediction:
        """Predictions of one shot's rows.

        Args:
            rows: ``merge_inputs`` rows of one shot.
            machine: Machine of the rows.
            with_scalars: Compute q95 and betaN; NaN otherwise.
            with_shapes: Compute the shapes of the maps; on MAST they are always computed
                when the scalars are (the MAST head reads them).
            c_std: Coefficients to use instead of the model's (diagnostics).
        """
        c = self.coefficients_of(rows, machine) if c_std is None else np.asarray(c_std, dtype=np.float64)
        rows, c = scored_part(rows, c)
        psi = self.composer.maps(machine, rows, c)
        n = len(rows)
        q95, beta_n, shapes = np.full(n, np.nan), np.full(n, np.nan), None
        need_shapes = bool(with_shapes) or (with_scalars and machine == "MAST")
        if with_scalars or need_shapes:
            table = self.composer.tables[machine]
            q, shape_rows = self.scalar_workers(machine, psi.astype(np.float32), rows["tf__mean"].to_numpy(dtype=np.float64),
                                                table.grid_r, table.grid_z, need_shapes, with_scalars)
            if with_scalars:
                q95 = q
            if need_shapes:
                shapes = shapes_frame(rows["shot"], rows["frame"], shape_rows)
        if with_scalars and self.heads is not None:
            beta_n = np.asarray(self.heads.predict(machine, rows, shapes), dtype=np.float64)
        return ShotPrediction(c_std=c, psi=psi, q95=q95, beta_n=beta_n, shapes=shapes)

    def close(self) -> None:
        self.scalar_workers.close()


# --------------------------------------------------------------------- shared loading
@dataclass
class ChainPaths:
    """Locations of the chain's inputs."""

    table_root: Path = ft.DEFAULT_TABLE_ROOT
    feature_root: Path = fe.DEFAULT_FEATURE_ROOT
    feature_config: Path | None = None
    scaler: Path | None = None
    targets_root: Path = tb.DEFAULT_TARGET_ROOT
    green_dir: Path = predictors.DEFAULT_GREEN_DIR
    data_root: Path = dataset.DEFAULT_DATA_ROOT
    checkpoint_root: Path = DEFAULT_CHECKPOINT_ROOT


def frame_build_params(table_root: Path | str) -> tuple[ft.BuildParams, str]:
    """The parameters of the stored frame-table build, or the defaults when it has no record;
    with a note of which."""
    path = Path(table_root) / "build.json"
    if path.is_file():
        return ft.BuildParams(**json.loads(path.read_text())["params"]), f"from {path}"
    return ft.BuildParams(), "defaults (no frame-table record found)"


def build_chain(run_ids: Sequence[str], paths: ChainPaths = ChainPaths(), which: str = "best", mirror_average: bool | None = None,
                seed_average: bool | None = None, heads: sh.BetanHeads | None = None, device: str | None = "cpu",
                workers: int | None = 1) -> tuple[PredictionChain, dict]:
    """The chain of the given runs' checkpoints and the record of what it was built from.

    ``mirror_average`` and ``seed_average`` default to the first run's ``predict`` fields.
    With seed averaging off, only the first run is used.
    """
    members = load_members(run_ids, paths.checkpoint_root, which, device or "cpu")
    config = members[0].config
    mirror = bool(config.get("predict.mirror_average")) if mirror_average is None else bool(mirror_average)
    seeds = bool(config.get("predict.seed_average")) if seed_average is None else bool(seed_average)
    if not seeds:
        members = members[:1]
    version = int(members[0].meta["feature_version"])
    data_paths = ld.DataPaths(feature_root=paths.feature_root, feature_config=paths.feature_config, scaler=paths.scaler,
                              target_root=paths.targets_root, target_version=int(members[0].meta["target_version"]))
    fconfig = fe.FeatureConfig.load(data_paths.feature_config_path(version))
    scaler = RobustScaler.load(data_paths.scaler_path(version))
    coefficients = CoefficientModel(members, fconfig, scaler, float(config.get("loader.clip_bound")), mirror, device or "cpu")
    objects = tb.TargetObjects.load(data_paths.target_dir())
    tables = {m: predictors.green_table(paths.green_dir, m) for m in MACHINES}
    scales = {m: adopted_scales(m, tables[m].columns) for m in MACHINES}
    composer = MapComposer(objects, tables, scales, coefficients.sigma)
    chain = PredictionChain(coefficients, composer, heads, ScalarWorkers(workers))
    record = {"runs": [m.run_id for m in members], "checkpoints": {m.run_id: {"path": str(m.path), "sha256": m.sha256,
                                                                              "epoch": m.meta.get("epoch")} for m in members},
              "config_hash": members[0].meta["config_hash"], "which": which, "mirror_average": mirror, "seed_average": seeds,
              "device": str(coefficients.device), "feature_version": version, "columns": coefficients.columns,
              "scaler_sha256": sha256_of(data_paths.scaler_path(version)), "target_build": {
                  "version": data_paths.target_version, "config_hash": objects.record.get("config_hash"),
                  "checksums": objects.record.get("checksums", {})},
              "sign_rule_c": {m: targets.sign_rule_c(m) for m in MACHINES},
              "coil_scales": {m: scales[m].to_dict() for m in MACHINES},
              "machine_constants": {m: machine_config(m).constants_record() for m in MACHINES},
              "q95": {"method": "q95_from_map", "tf_turns": {m: sc.TF_TURNS[m] for m in MACHINES}}}
    return chain, record


def scored_part(rows: pd.DataFrame, c: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    """The rows flagged ``scored`` and their coefficients (all rows when there is no flag)."""
    if SCORED not in rows.columns:
        return rows, c
    keep = rows[SCORED].to_numpy(dtype=bool)
    return rows.loc[keep].reset_index(drop=True), np.asarray(c)[keep]


def scored(rows: pd.DataFrame) -> pd.DataFrame:
    """The rows predictions are made for: those flagged ``scored``, or all."""
    return rows if SCORED not in rows.columns else rows.loc[rows[SCORED].to_numpy(dtype=bool)].reset_index(drop=True)


def read_rows(source: TableSource, split: str, shot_ids: Sequence[str], stride: int = 1, labels: Sequence[str] = (),
              shots_per_chunk: int = 500, mark: bool = False) -> pd.DataFrame:
    """Stored rows of many shots (every ``stride``-th frame of each, read a chunk of shots at a
    time), optionally with label columns for fitting and scoring. With ``mark`` every frame is
    kept and the strided ones are flagged ``scored`` (what a sequence model needs).

    Raises:
        ValueError: when labels are asked of a split without equilibrium labels (MAST, the
            public tests).
    """
    if labels:
        tb.require_labelled(split)
    ids = sorted({str(s) for s in shot_ids})
    if not ids:
        raise ValueError("no shots to read")
    parts = []
    for start in range(0, len(ids), shots_per_chunk):
        chunk = ids[start:start + shots_per_chunk]
        rows = source._read(split, chunk)
        if mark:
            rows = rows.sort_values(list(KEY_COLUMNS), kind="stable").reset_index(drop=True)
            rows[SCORED] = rows.groupby("shot", sort=False).cumcount().to_numpy() % int(stride) == 0
        elif stride > 1:
            rows = sh.strided(rows, stride)
        if labels:
            lab = ft.read_frame_table(split, ["shot", "frame", *labels], chunk, root=source.table_root)
            lab["shot"] = lab["shot"].astype(str)
            rows = rows.merge(lab, on=list(KEY_COLUMNS), how="left", validate="one_to_one")
        parts.append(rows)
    return pd.concat(parts, ignore_index=True)


def predict_shots(chain: PredictionChain, rows: pd.DataFrame, machine: str, with_scalars: bool = True,
                  with_shapes: bool = False, keep_maps: bool = True) -> dict[str, ShotPrediction]:
    """Shot id to its predictions, shots in order of appearance.

    Coefficients and maps are computed one shot at a time, exactly as ``PredictionChain.predict``
    does; q95 and shapes of many shots go to the chain's process pool together (a shot of a
    strided list is shorter than a pool chunk). ``keep_maps`` False drops each shot's maps once
    its shapes are computed (bulk shape tables).
    """
    need_shapes = bool(with_shapes) or (with_scalars and machine == "MAST")
    groups = [(str(s), g.reset_index(drop=True)) for s, g in rows.groupby("shot", sort=False)]
    if not (with_scalars or need_shapes):
        return {s: chain.predict(g, machine, with_scalars=False, with_shapes=False) for s, g in groups}
    table = chain.composer.tables[machine]
    r, z = np.asarray(table.grid_r, dtype=np.float64), np.asarray(table.grid_z, dtype=np.float64)
    held: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def items():
        for shot, g in groups:
            c = chain.coefficients_of(g, machine)
            g, c = scored_part(g, c)
            psi = chain.composer.maps(machine, g, c)
            held[shot] = (c, psi, g)
            yield machine, psi.astype(np.float32), g["tf__mean"].to_numpy(dtype=np.float64), r, z, need_shapes, with_scalars

    out = {}
    for (shot, _), (q, shape_rows) in zip(groups, chain.scalar_workers.map_shots(items())):
        c, psi, g = held.pop(shot)
        shapes = shapes_frame(g["shot"], g["frame"], shape_rows) if need_shapes else None
        beta_n = np.full(len(g), np.nan)
        if with_scalars and chain.heads is not None:
            beta_n = np.asarray(chain.heads.predict(machine, g, shapes), dtype=np.float64)
        out[shot] = ShotPrediction(c_std=c, psi=psi if keep_maps else None, q95=q if with_scalars else np.full(len(g), np.nan),
                                   beta_n=beta_n, shapes=shapes)
    return out


# ---------------------------------------------------------------------------- heads
def heads_training_rows(source: TableSource, split_path: Path | str, train_list: str, stride: int,
                        mark: bool = False) -> pd.DataFrame:
    """Strided rows of the training list with the label filter's columns and the label (every
    frame, the strided ones flagged, with ``mark``)."""
    ld.require_readable(train_list)
    ids = sorted(sp.read_split(split_path)[train_list])
    rows = read_rows(source, tb.LABELLED_SPLIT, ids, stride, ("efit_q95", sh.LABEL), mark=mark)
    extra = ft.read_frame_table(tb.LABELLED_SPLIT, ["shot", "frame", "ip_over_peak", "ip_direction"], ids, root=source.table_root)
    extra["shot"] = extra["shot"].astype(str)
    have = [c for c in ("ip_over_peak", "ip_direction") if c in rows.columns]
    return rows.merge(extra.drop(columns=have), on=list(KEY_COLUMNS), how="left", validate="one_to_one")


def model_shapes(chain: PredictionChain, rows: pd.DataFrame, machine: str = "DIII-D") -> pd.DataFrame:
    """Boundary shapes of the chain's maps of the rows (one shot at a time)."""
    parts = [p.shapes for p in predict_shots(chain, rows, machine, with_scalars=False, with_shapes=True, keep_maps=False).values()]
    return pd.concat(parts, ignore_index=True) if parts else shapes_frame([], [], [])


def heads_r2(heads: sh.BetanHeads, rows: pd.DataFrame, shapes: pd.DataFrame, corner: np.ndarray) -> dict:
    """Pooled R2 of the MAST head (applied with the DIII-D constants, as validated in F99) on
    validation rows with both labels finite: all frames and the corner."""
    table = sh.head_table(rows, shapes, heads.q_cyl_band, "DIII-D")
    pred = heads.mast_correction.predict(table, heads.mast_constant * table[sh.BASE].to_numpy(dtype=np.float64))
    y = rows[sh.LABEL].to_numpy(dtype=np.float64)
    scored = np.isfinite(y) & np.isfinite(rows["efit_q95"].to_numpy(dtype=np.float64))
    corner = scored & np.asarray(corner, dtype=bool)
    return {"r2_all": sc.pooled_r2(y[scored], pred[scored]), "n_all": int(scored.sum()),
            "r2_corner": sc.pooled_r2(y[corner], pred[corner]) if corner.any() else float("nan"), "n_corner": int(corner.sum())}


# ------------------------------------------------------------------------ predictor
class Rung1V1:
    """Rung-1 plasma model through the full prediction chain (module docstring). Registered as
    ``rung1_v1`` in ``fusion_eq.predictors``.

    Args:
        split_path, cache_dir, green_dir, table_root, data_root: As for the other predictors
            (``make_submission.py`` passes them).
        runs: Training runs whose best checkpoints are used (one per seed).
        source: ``tables`` (stored frame-table and feature rows, looked up by shot id) or
            ``raw`` (rebuilt from the shot handed to ``predict_shot``).
        heads: The betaN heads, or a path to a file written by ``BetanHeads.save``.
        Other arguments: ``ChainPaths`` fields, averaging switches, device and the workers of
            the q95 and shape computation.
    """

    name = "rung1_v1"

    def __init__(self, split_path: Path | str = predictors.DEFAULT_SPLIT, cache_dir: Path | str = bl.DEFAULT_CACHE_DIR,
                 green_dir: Path | str = predictors.DEFAULT_GREEN_DIR, table_root: Path | str = ft.DEFAULT_TABLE_ROOT,
                 data_root: Path | str | None = None, runs: Sequence[str] = (), source: str = "raw",
                 heads: sh.BetanHeads | Path | str | None = None, feature_root: Path | str = fe.DEFAULT_FEATURE_ROOT,
                 feature_config: Path | str | None = None, scaler: Path | str | None = None,
                 targets_root: Path | str = tb.DEFAULT_TARGET_ROOT, checkpoint_root: Path | str = DEFAULT_CHECKPOINT_ROOT,
                 which: str = "best", mirror_average: bool | None = None, seed_average: bool | None = None,
                 device: str | None = "cpu", workers: int | None = 1) -> None:
        if source not in SOURCES:
            raise ValueError(f"source {source!r} is not one of {SOURCES}")
        self.split_path = Path(split_path)
        self.cache_dir = Path(cache_dir)
        self.runs = list(runs)
        self.source_name = source
        self.heads_arg = heads
        self.paths = ChainPaths(table_root=Path(table_root), feature_root=Path(feature_root),
                                feature_config=None if feature_config is None else Path(feature_config),
                                scaler=None if scaler is None else Path(scaler), targets_root=Path(targets_root),
                                green_dir=Path(green_dir), data_root=Path(data_root) if data_root else dataset.default_data_root(),
                                checkpoint_root=Path(checkpoint_root))
        self.which, self.mirror_average, self.seed_average = which, mirror_average, seed_average
        self.device, self.workers = device, workers
        self.chain: PredictionChain | None = None
        self.source: TableSource | RawSource | None = None
        self.identifiers: dict = {}

    sequence = False

    def setup(self, log: Callable[[str], None] = print) -> dict:
        if not self.runs:
            raise ValueError(f"{self.name} needs the training runs of its checkpoints (scripts/predict.py --run <run_id>)")
        heads = self.heads_arg
        heads_file = None
        if isinstance(heads, (str, Path)):
            heads_file = {"path": str(heads), "sha256": sha256_of(heads)}
            heads = sh.BetanHeads.load(heads)
        if heads is None:
            raise ValueError("rung1_v1 needs the betaN heads (scripts/predict.py fits them, or pass --heads)")
        self.chain, record = build_chain(self.runs, self.paths, self.which, self.mirror_average, self.seed_average, heads,
                                         self.device, self.workers)
        if self.chain.coefficients.sequence != self.sequence:
            kind = "a sequence model (rung 2)" if self.chain.coefficients.sequence else "a frame model (rung 1)"
            raise ValueError(f"{self.name} cannot predict with {kind}; use {'rung2_v1' if self.chain.coefficients.sequence else 'rung1_v1'}")
        assert_mast_sign()
        frame_columns = {m: self.chain.frame_columns(m) for m in MACHINES}
        features = self.chain.feature_columns()
        if self.source_name == "tables":
            self.source = TableSource(self.paths.table_root, self.paths.feature_root, record["feature_version"], frame_columns, features)
            build_note = None
        else:
            params, build_note = frame_build_params(self.paths.table_root)
            fconfig = self.chain.coefficients.fconfig
            resources = {m: fe.load_resources(m, fconfig, self.paths.data_root, self.paths.green_dir) for m in MACHINES}
            self.source = RawSource(params, fconfig, resources, frame_columns, features)
        self.identifiers = {**record, "source": self.source_name, "heads": heads.summary(), "heads_file": heads_file,
                            "mast_sign_asserted": True}
        if build_note is not None:
            self.identifiers["frame_build_params"] = {"params": self.source.params.to_dict(), "source": build_note}
        log(f"{self.name} ready: {len(record['runs'])} checkpoint(s) {', '.join(record['runs'])}, source {self.source_name}, "
            f"mirror averaging {'on' if record['mirror_average'] else 'off'}, device {record['device']}")
        return self.identifiers

    def rows(self, shot: io.Shot) -> pd.DataFrame:
        """The input rows of a raw shot from the configured source."""
        split = split_of(shot)
        if isinstance(self.source, TableSource):
            return self.source.rows(split, shot.shot_id)
        return self.source.rows_of_shot(shot, split)

    def predict_shot(self, shot: io.Shot, machine: str) -> bl.Prediction:
        if self.chain is None:
            raise RuntimeError("call setup() first")
        if machine not in MACHINES or shot.machine.name != machine:
            raise ValueError(f"shot {shot.shot_id} of {shot.machine.name} asked as {machine!r}")
        rows = self.rows(shot)
        if len(rows) != shot.n_frames:
            raise ValueError(f"shot {shot.shot_id}: {len(rows)} input rows for {shot.n_frames} frames")
        return self.chain.predict(rows, machine).as_prediction()

    def close(self) -> None:
        if self.chain is not None:
            self.chain.close()


class Rung2V1(Rung1V1):
    """The rung-2 sequence model through the same chain; each shot is predicted whole, with the
    time steps of its frames. Registered as ``rung2_v1`` in ``fusion_eq.predictors``."""

    name = "rung2_v1"
    sequence = True
