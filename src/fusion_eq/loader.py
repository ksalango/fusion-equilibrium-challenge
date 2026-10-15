"""Loader of the plasma model (``docs/PIPELINE.md``, Part 2): the frames of a list of the split
file joined from the version-2 feature table, the version-1 target table and the loss inputs,
and the batches the model trains on.

Frames are those of the target table that pass the lighter label filter (the stored flag) and,
with ``loader.require_boundary``, have a boundary in the true map (an empty inside mask leaves
every map term undefined). Features are scaled with the stored DIII-D scaler; the mirrored
features are ``mirror_features`` of the unscaled table, scaled the same way, so that the mirror
in training is exact by construction. Targets are the current coefficients divided by the
stored component standard deviations (no centring: ``c = 0`` is the mean current).

Training batches (``FrameLoader`` in mode ``train``) apply, in this order: the up-down mirror
with probability 0.5 (mirrored features, ``c`` times the parity vector, masks and the exact
reference flipped in Z), Thomson masking at ``loader.thomson_mask_rate`` (the Thomson features
set to their scaled missing value, ``ts_valid`` to 0), and clipping to ``loader.clip_bound``.
Evaluation batches apply the clipping only. Random draws come from one CPU generator seeded
with ``train.seed``, so that a run is reproducible on any device.

Tail weights (P12, off by default) multiply the frame weight of DIII-D training frames above
the training quantile of any tail column (unscaled feature values); thresholds come from the
training frames only.

Rung 2 batches by shot (``ShotLoader``): every frame of each shot enters the sequence, in time
order, padded to the longest shot of the batch; the loss reads only the frames that rung 1 trains
on (filter passed and, with ``loader.require_boundary``, a boundary), flattened. Each frame carries
the time step to the previous frame of its shot (``dt_input``: ms over ``rung2.dt_scale_ms``,
clipped to ``[0, loader.clip_bound]``, 0 at a shot's first frame). The mirror flips whole shots;
Thomson masking draws per frame. ``pad_by_shot`` and ``unpad`` are the host-side equivalents.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

from fusion_eq import (current_basis as cb, dataset, features as fe, frame_table as ft, loss as ls, loss_inputs as li,
                       split as sp, target_build as tb)
from fusion_eq.feature_scaler import RobustScaler
from fusion_eq.model_config import ModelConfig

FEATURE_SPLIT = "diii_d_train"
MACHINE = "DIII-D"
THOMSON_GROUP = "plasma_content"
VALID_FLAG = "ts_valid"
PROTECTED_LISTS = ("local_test",)
# Unscaled feature columns kept beside the scaled matrix: tail weights and the corner slice.
EXTRA_COLUMNS = ("shaping", "ts_beta_p", "q_cyl")


# ------------------------------------------------------------------------------ paths
@dataclass
class DataPaths:
    """Locations of the inputs of the model phase; defaults are the standard build locations."""

    feature_root: Path = fe.DEFAULT_FEATURE_ROOT
    feature_config: Path | None = None
    scaler: Path | None = None
    target_root: Path = tb.DEFAULT_TARGET_ROOT
    target_version: int = 1
    loss_root: Path = li.DEFAULT_ROOT
    split: Path = dataset.REPO_ROOT / "splits" / "split_v1.json"
    table_dir: Path = ft.DEFAULT_TABLE_ROOT

    def feature_config_path(self, version: int) -> Path:
        return Path(self.feature_config) if self.feature_config else fe.DEFAULT_CONFIG.with_name(f"features_v{version}.yaml")

    def scaler_path(self, version: int) -> Path:
        return Path(self.scaler) if self.scaler else fe.scaler_path(version)

    def target_dir(self) -> Path:
        return tb.target_dir(self.target_version, self.target_root)


def load_bases(paths: DataPaths) -> tuple[cb.CurrentBasis, cb.MachineFluxBasis]:
    """The current basis (variances, parity) and the DIII-D flux basis of the target build."""
    folder = paths.target_dir()
    basis = cb.CurrentBasis.load(folder / tb.OBJECT_FILES["current_basis"])
    flux = cb.MachineFluxBasis.load(folder / tb.OBJECT_FILES[f"flux_basis_{MACHINE}"])
    if flux.n_components != basis.n_components:
        raise ValueError(f"flux basis of {flux.n_components} components for a current basis of {basis.n_components}")
    return basis, flux


def component_sigma(basis: cb.CurrentBasis) -> np.ndarray:
    """Standard deviation of each current component (the standardization of ``c``).

    Raises:
        ValueError: for a component of zero variance.
    """
    v = np.asarray(basis.variances, dtype=np.float64)
    if not (v > 0).all():
        raise ValueError("a current component has zero variance; it cannot be standardized")
    return np.sqrt(v)


def model_columns(config: fe.FeatureConfig, set_name: str, drop_groups: Iterable[str] = ()) -> list[str]:
    """Columns of a feature set, less those of the dropped groups.

    Raises:
        ValueError: for an unknown group or when nothing is left.
    """
    specs = config.column_specs()
    drop = set(drop_groups)
    unknown = drop - {s.group for s in config.features}
    if unknown:
        raise ValueError(f"unknown feature groups {sorted(unknown)}")
    columns = [c for c in config.set_columns(set_name) if specs[c].group not in drop]
    if not columns:
        raise ValueError(f"set {set_name} without {sorted(drop)} has no column")
    return columns


def thomson_masking(config: fe.FeatureConfig, columns: Sequence[str], scaler: RobustScaler) -> tuple[np.ndarray, np.ndarray]:
    """Indices of the Thomson columns of the model's columns and the scaled value each takes
    when Thomson is missing: the scaler's value for NaN, and for ``ts_valid`` that of 0."""
    specs = config.column_specs()
    idx = [j for j, c in enumerate(columns) if specs[c].group == THOMSON_GROUP]
    row = pd.DataFrame({c: [0.0 if c == VALID_FLAG else np.nan] for c in columns})
    missing = scaler.transform(row, MACHINE, columns)[0]
    return np.asarray(idx, dtype=np.int64), missing[idx].astype(np.float64)


# --------------------------------------------------------------------------- frame sets
@dataclass
class FrameSet:
    """Frames of one or more lists, on the host, in table order (shots sorted, frames in order).

    Attributes:
        lists: Lists the frames come from.
        columns: Model feature columns.
        shot, frame: Keys per frame.
        x, x_mirror: ``(N, F)`` scaled features of the frame and of its mirror (not clipped).
        c: ``(N, K)`` standardized current coefficients.
        ip_a: ``(N,)`` signed plasma current in A.
        phase: ``(N,)`` phase index into ``model_config.PHASES``.
        near_repeat: ``(N,)`` whether the frame's shot has a near-repeat in the training list.
        inside, band: ``(N, bytes)`` packed masks.
        psi_ref: ``(N, P)`` exact reference (float16), or None.
        grid_shape: ``(nZ, nR)`` of the masks.
        extra: Unscaled feature columns by name (``EXTRA_COLUMNS`` present in the table).
        kept: ``(N,)`` frames the loss reads; None when every frame is kept (rung 1 sets).
        time_ms: ``(N,)`` frame times in ms (sequence sets), or None.
    """

    lists: tuple[str, ...]
    columns: list[str]
    shot: np.ndarray
    frame: np.ndarray
    x: np.ndarray
    x_mirror: np.ndarray
    c: np.ndarray
    ip_a: np.ndarray
    phase: np.ndarray
    near_repeat: np.ndarray
    inside: np.ndarray
    band: np.ndarray
    psi_ref: np.ndarray | None
    grid_shape: tuple[int, int]
    extra: dict[str, np.ndarray] = field(default_factory=dict)
    kept: np.ndarray | None = None
    time_ms: np.ndarray | None = None

    def __len__(self) -> int:
        return int(len(self.shot))

    @property
    def kept_mask(self) -> np.ndarray:
        return np.ones(len(self), dtype=bool) if self.kept is None else np.asarray(self.kept, dtype=bool)

    def subset(self, rows) -> "FrameSet":
        rows = np.asarray(rows)
        return FrameSet(lists=self.lists, columns=self.columns, shot=self.shot[rows], frame=self.frame[rows], x=self.x[rows],
                        x_mirror=self.x_mirror[rows], c=self.c[rows], ip_a=self.ip_a[rows], phase=self.phase[rows],
                        near_repeat=self.near_repeat[rows], inside=self.inside[rows], band=self.band[rows],
                        psi_ref=None if self.psi_ref is None else self.psi_ref[rows], grid_shape=self.grid_shape,
                        extra={k: v[rows] for k, v in self.extra.items()},
                        kept=None if self.kept is None else self.kept[rows], time_ms=None if self.time_ms is None else self.time_ms[rows])

    def kept_frames(self) -> "FrameSet":
        """The frames the loss reads, as a rung-1 set (no ``kept`` flag)."""
        out = self.subset(np.flatnonzero(self.kept_mask))
        out.kept = None
        return out


def concat(sets: Sequence[FrameSet]) -> FrameSet:
    """Frames of several sets, in the given order.

    Raises:
        ValueError: when the sets differ in columns or grid.
    """
    first = sets[0]
    if any(s.columns != first.columns or s.grid_shape != first.grid_shape for s in sets):
        raise ValueError("frame sets differ in columns or grid")
    cat = lambda name: np.concatenate([getattr(s, name) for s in sets])
    return FrameSet(lists=tuple(n for s in sets for n in s.lists), columns=first.columns, shot=cat("shot"), frame=cat("frame"),
                    x=cat("x"), x_mirror=cat("x_mirror"), c=cat("c"), ip_a=cat("ip_a"), phase=cat("phase"),
                    near_repeat=cat("near_repeat"), inside=cat("inside"), band=cat("band"),
                    psi_ref=None if first.psi_ref is None else cat("psi_ref"), grid_shape=first.grid_shape,
                    extra={k: np.concatenate([s.extra[k] for s in sets]) for k in first.extra},
                    kept=None if all(s.kept is None for s in sets) else np.concatenate([s.kept_mask for s in sets]),
                    time_ms=None if any(s.time_ms is None for s in sets) else cat("time_ms"))


def require_readable(list_name: str, allow_local_test: bool = False) -> None:
    """Raises for ``local_test`` unless explicitly allowed (it is read only at the final
    pre-submission check)."""
    if list_name in PROTECTED_LISTS and not allow_local_test:
        raise PermissionError(f"{list_name} is read only at the final pre-submission check")


def near_repeat_flags(split: Mapping, shots: np.ndarray) -> np.ndarray:
    """Per frame, whether its shot has a near-repeat in the training list (split file)."""
    lookup = {s: bool(v.get("near_repeat_in_train") or False) for s, v in split.get("shots", {}).items()}
    unique, inverse = np.unique(np.asarray(shots).astype(str), return_inverse=True)
    return np.array([lookup.get(s, False) for s in unique], dtype=bool)[inverse]


def load_frames(list_name: str, config: ModelConfig, paths: DataPaths = DataPaths(), split: Mapping | None = None,
                allow_local_test: bool = False, sequences: bool = False) -> FrameSet:
    """The frames of one list ready for the model.

    Args:
        list_name: List of the split file (target table and loss inputs of the same name).
        config: Model configuration.
        paths: Input locations.
        split: The split record (read from ``paths.split`` when None).
        allow_local_test: Permit ``local_test``.
        sequences: Keep every frame of each shot (rung 2) with the rung-1 selection as ``kept``,
            and the frame times from the frame table; requires a target build at stride 1.

    Raises:
        PermissionError: for ``local_test`` without permission.
        ValueError: when the inputs disagree (versions, band width, missing frames).
    """
    require_readable(list_name, allow_local_test)
    split = split if split is not None else sp.read_split(paths.split)
    version = int(config.get("features.version"))
    fconfig = fe.FeatureConfig.load(paths.feature_config_path(version))
    if fconfig.version != version:
        raise ValueError(f"feature configuration version {fconfig.version}, model configuration asks for {version}")
    scaler = RobustScaler.load(paths.scaler_path(version))
    columns = model_columns(fconfig, config.get("features.set"), config.get("features.drop_groups"))
    basis, flux = load_bases(paths)
    sigma = component_sigma(basis)
    c_columns = [f"c_{j:02d}" for j in range(basis.n_components)]

    target = pq.read_table(tb.table_path(list_name, paths.target_version, paths.target_root),
                           columns=["shot", "frame", "ip_ka", "phase", "passes_filter", "boundary_found", *c_columns]).to_pandas()
    target["shot"] = target["shot"].astype(str)
    keep = target["passes_filter"].to_numpy(dtype=bool).copy()
    if config.get("loader.require_boundary"):
        keep &= target["boundary_found"].to_numpy(dtype=bool)
    time_ms = None
    if sequences:
        record_path = paths.target_dir() / "build.json"
        stride = json.loads(record_path.read_text()).get("stride", 1) if record_path.is_file() else 1
        if int(stride) != 1:
            raise ValueError(f"the target build has stride {stride}; sequences need every frame (stride 1)")
        times = ft.read_frame_table(tb.LABELLED_SPLIT, ["shot", "frame", "time_ms"], sorted(target["shot"].unique()),
                                    root=paths.table_dir)
        times["shot"] = times["shot"].astype(str)
        joined_t = target[["shot", "frame"]].merge(times, on=["shot", "frame"], how="left", validate="one_to_one")
        if joined_t["time_ms"].isna().any():
            raise ValueError(f"{list_name}: {int(joined_t['time_ms'].isna().sum())} target frames have no frame-table time")
        time_ms = joined_t["time_ms"].to_numpy(dtype=np.float64)
    else:
        target = target.loc[keep].reset_index(drop=True)

    extra = [c for c in EXTRA_COLUMNS if c in fconfig.all_columns()]
    table = fe.read_features(FEATURE_SPLIT, version, list(dict.fromkeys([*columns, *extra])), sorted(target["shot"].unique()),
                             paths.feature_root)
    table["shot"] = table["shot"].astype(str)
    joined = target[["shot", "frame"]].merge(table, on=["shot", "frame"], how="left", indicator=True, validate="one_to_one")
    if (joined["_merge"] != "both").any():
        missing = joined.loc[joined["_merge"] != "both", ["shot", "frame"]].iloc[0].to_dict()
        raise ValueError(f"{list_name}: {(joined['_merge'] != 'both').sum()} target frames have no feature row, first {missing}")
    joined = joined.drop(columns="_merge")

    inputs = li.read_loss_inputs(list_name, paths.target_version, paths.loss_root, exact=config.get("loss.reference") == "exact")
    if int(inputs.record.get("band_cells", -1)) != int(config.get("loss.band_cells")):
        raise ValueError(f"loss inputs built with band_cells {inputs.record.get('band_cells')}, configuration asks for "
                         f"{config.get('loss.band_cells')}; rebuild with python scripts/build_loss_inputs.py --band-cells "
                         f"{config.get('loss.band_cells')}")
    if inputs.grid_shape[0] * inputs.grid_shape[1] != flux.phi.shape[1]:
        raise ValueError(f"loss-input grid {inputs.grid_shape} differs from the flux basis grid {flux.shape}")
    rows = inputs.rows_of(target["shot"], target["frame"])

    x = scaler.transform(joined, MACHINE, columns).astype(np.float32)
    x_mirror = scaler.transform(fe.mirror_features(joined, fconfig), MACHINE, columns).astype(np.float32)
    return FrameSet(lists=(list_name,), columns=columns, shot=target["shot"].to_numpy(dtype=object),
                    frame=target["frame"].to_numpy(dtype=np.int64), x=x, x_mirror=x_mirror,
                    c=(target[c_columns].to_numpy(dtype=np.float64) / sigma).astype(np.float32),
                    ip_a=target["ip_ka"].to_numpy(dtype=np.float64) * ls.A_PER_KA,
                    phase=ls.phase_index(target["phase"].to_numpy()), near_repeat=near_repeat_flags(split, target["shot"].to_numpy()),
                    inside=np.asarray(inputs.inside[rows]), band=np.asarray(inputs.band[rows]),
                    psi_ref=None if inputs.psi_ref is None else np.asarray(inputs.psi_ref[rows]), grid_shape=inputs.grid_shape,
                    extra={c: joined[c].to_numpy(dtype=np.float64) for c in extra}, kept=keep if sequences else None,
                    time_ms=time_ms)


# ---------------------------------------------------------------------------- weights
def tail_rule(train: FrameSet, config: ModelConfig) -> dict[str, float]:
    """Thresholds of the tail columns: their ``loader.tail_quantile`` over the training frames
    (finite values only).

    Raises:
        KeyError: when a tail column is not among the loaded unscaled columns.
    """
    q = float(config.get("loader.tail_quantile"))
    out = {}
    for column in config.get("loader.tail_columns"):
        if column not in train.extra:
            raise KeyError(f"tail column {column!r} is not loaded; loaded: {sorted(train.extra)}")
        v = train.extra[column][np.isfinite(train.extra[column])]
        out[column] = float(np.quantile(v, q))
    return out


def tail_flags(frames: FrameSet, rule: Mapping[str, float]) -> np.ndarray:
    """Per frame, whether any tail column lies above its threshold."""
    flags = np.zeros(len(frames), dtype=bool)
    for column, threshold in rule.items():
        with np.errstate(invalid="ignore"):
            flags |= frames.extra[column] > threshold
    return flags


def sample_weights(frames: FrameSet, config: ModelConfig, ip_med: float, rule: Mapping[str, float] | None = None) -> np.ndarray:
    """Frame weights ``w_f`` (``loss.frame_weights``), times ``loader.tail_weight`` on tail
    frames when a tail rule is given."""
    w = ls.frame_weights(frames.ip_a, frames.phase, ip_med, float(config.get("loss.frame_gamma")),
                         config.get("loss.phase_weights"))
    if rule:
        w = np.where(tail_flags(frames, rule), w * float(config.get("loader.tail_weight")), w)
    return w


def clip_rates(x: np.ndarray, bound: float) -> np.ndarray:
    """Share of values beyond ``bound`` in magnitude, per column."""
    x = np.asarray(x)
    return (np.abs(x) > bound).mean(axis=0) if len(x) else np.zeros(x.shape[1])


# ---------------------------------------------------------------------------- batches
_SHIFTS = torch.arange(7, -1, -1, dtype=torch.uint8)


def unpack_masks(packed: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    """Bool masks ``(B, nZ, nR)`` from rows packed by ``numpy.packbits`` (big bit order)."""
    bits = (packed.unsqueeze(-1) >> _SHIFTS.to(packed.device)) & 1
    n = int(shape[0]) * int(shape[1])
    return bits.reshape(packed.shape[0], -1)[:, :n].reshape(packed.shape[0], *shape).bool()


class FrameLoader:
    """Batches of frames (rung 1) on one device.

    Args:
        frames: The frames.
        parity: ``(K,)`` parity vector of the current basis.
        weights: ``(N,)`` frame weights.
        config: Model configuration (loader fields, ``train.batch_frames``, ``train.seed``).
        thomson: ``(indices, missing values)`` of ``thomson_masking``.
        mode: ``train`` (shuffled, augmented) or ``eval`` (table order, clipped only).
        device: Torch device; the frames are moved there once.
        dtype: Floating dtype of features, targets and weights.
        batch_frames: Batch size; ``train.batch_frames`` when None.
        seed: Seed of the generator; ``train.seed`` when None.
    """

    def __init__(self, frames: FrameSet, parity, weights, config: ModelConfig, thomson: tuple[np.ndarray, np.ndarray],
                 mode: str = "train", device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32,
                 batch_frames: int | None = None, seed: int | None = None) -> None:
        if mode not in ("train", "eval"):
            raise ValueError(f"mode {mode!r} is not train or eval")
        self.mode = mode
        self.device = torch.device(device)
        self.dtype = dtype
        self.n = len(frames)
        self.grid_shape = frames.grid_shape
        self.batch_frames = int(batch_frames or config.get("train.batch_frames"))
        self.mirror = bool(config.get("loader.mirror")) and mode == "train"
        self.thomson_rate = float(config.get("loader.thomson_mask_rate")) if mode == "train" else 0.0
        self.clip_bound = float(config.get("loader.clip_bound"))
        self.generator = torch.Generator(device="cpu").manual_seed(int(config.get("train.seed") if seed is None else seed))

        def to(a, dt):
            return torch.as_tensor(np.ascontiguousarray(a), dtype=dt).to(self.device)

        self.x = to(frames.x, dtype)
        self.x_mirror = to(frames.x_mirror, dtype)
        self.c = to(frames.c, dtype)
        self.ip_a = to(frames.ip_a, dtype)
        self.weight = to(weights, dtype)
        self.inside = to(frames.inside, torch.uint8)
        self.band = to(frames.band, torch.uint8)
        self.psi_ref = None if frames.psi_ref is None else to(frames.psi_ref, torch.float16)
        self.parity = to(np.asarray(parity, dtype=np.float64), dtype)
        self.thomson_idx = torch.as_tensor(thomson[0], dtype=torch.long).to(self.device)
        self.thomson_missing = to(thomson[1], dtype)

    def __len__(self) -> int:
        return (self.n + self.batch_frames - 1) // self.batch_frames

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        order = torch.randperm(self.n, generator=self.generator) if self.mode == "train" else torch.arange(self.n)
        for start in range(0, self.n, self.batch_frames):
            yield self.batch(order[start:start + self.batch_frames])

    def batch(self, index: torch.Tensor, augment: bool | None = None) -> dict[str, torch.Tensor]:
        """The batch of the given frame indices (CPU tensor); augmented in mode ``train`` unless
        ``augment`` is False (clipping always applies)."""
        b = len(index)
        augment = self.mode == "train" if augment is None else bool(augment)
        flip = (torch.rand(b, generator=self.generator) < 0.5) if (self.mirror and augment) else torch.zeros(b, dtype=torch.bool)
        masked = (torch.rand(b, generator=self.generator) < self.thomson_rate) if (self.thomson_rate > 0 and augment) \
            else torch.zeros(b, dtype=torch.bool)
        idx = index.to(self.device)
        flip, masked = flip.to(self.device), masked.to(self.device)
        x = torch.where(flip[:, None], self.x_mirror[idx], self.x[idx])
        c = torch.where(flip[:, None], self.c[idx] * self.parity, self.c[idx])
        if masked.any() and len(self.thomson_idx):
            rows = torch.nonzero(masked).squeeze(1)
            x[rows[:, None], self.thomson_idx[None, :]] = self.thomson_missing
        x = torch.clamp(x, -self.clip_bound, self.clip_bound)

        def flipped(maps: torch.Tensor) -> torch.Tensor:
            return torch.where(flip.reshape(-1, *([1] * (maps.dim() - 1))), maps.flip(1), maps)

        out = {"index": idx, "flip": flip, "thomson_masked": masked, "x": x, "c": c, "ip_a": self.ip_a[idx],
               "weight": self.weight[idx], "inside": flipped(unpack_masks(self.inside[idx], self.grid_shape)),
               "band": flipped(unpack_masks(self.band[idx], self.grid_shape))}
        if self.psi_ref is not None:
            out["psi_ref"] = flipped(self.psi_ref[idx].to(self.dtype).reshape(b, *self.grid_shape))
        return out


def dt_input(time_ms, shot, scale_ms: float, bound: float) -> np.ndarray:
    """The rung-2 time-step input per frame: the step to the previous frame of the same shot in
    ms (``time_steps``) over ``scale_ms``, clipped to ``[0, bound]``; 0 at a shot's first frame."""
    return np.clip(time_steps(time_ms, shot) / float(scale_ms), 0.0, float(bound))


class ShotLoader:
    """Padded batches of whole shots (rung 2) on one device.

    Args:
        frames: Every frame of each shot (``load_frames(..., sequences=True)``), shots contiguous
            and in time order, with ``kept`` and ``time_ms``.
        parity, weights, config, thomson, mode, device, dtype, seed: As for ``FrameLoader``.
        batch_shots: Shots per batch; ``rung2.batch_shots`` when None.

    A batch holds ``x`` ``(B, T, F)``, ``dt`` ``(B, T)``, ``lengths`` ``(B,)``, and, flattened over
    the kept frames in shot and time order, the positions ``pos_b`` and ``pos_t``, the row
    ``index`` into ``frames`` and the loss fields ``c``, ``ip_a``, ``weight``, ``inside``,
    ``band`` (and ``psi_ref``), so that the loss function of rung 1 applies unchanged.
    """

    def __init__(self, frames: FrameSet, parity, weights, config: ModelConfig, thomson: tuple[np.ndarray, np.ndarray],
                 mode: str = "train", device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32,
                 batch_shots: int | None = None, seed: int | None = None) -> None:
        if mode not in ("train", "eval"):
            raise ValueError(f"mode {mode!r} is not train or eval")
        if frames.time_ms is None:
            raise ValueError("a shot loader needs the frame times (load_frames(..., sequences=True))")
        steps = np.concatenate([np.diff(np.asarray(frames.time_ms[s:e], dtype=np.float64)) for s, e in shot_runs(frames.shot)] or [[]])
        if (steps <= 0).any():
            raise ValueError("frames of a shot are not in strictly increasing time order")
        self.frames = frames
        self.inner = FrameLoader(frames, parity, weights, config, thomson, mode, device, dtype, batch_frames=1, seed=seed)
        self.mode = mode
        self.device = self.inner.device
        self.n = len(frames)
        self.runs = shot_runs(frames.shot)
        self.batch_shots = int(batch_shots or config.get("rung2.batch_shots"))
        self.generator = self.inner.generator
        dt = dt_input(frames.time_ms, frames.shot, float(config.get("rung2.dt_scale_ms")), float(config.get("loader.clip_bound")))
        self.dt = torch.as_tensor(dt, dtype=dtype).to(self.device)
        self.kept = torch.as_tensor(frames.kept_mask).to(self.device)

    def __len__(self) -> int:
        return (len(self.runs) + self.batch_shots - 1) // self.batch_shots

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        order = torch.randperm(len(self.runs), generator=self.generator) if self.mode == "train" else torch.arange(len(self.runs))
        for start in range(0, len(self.runs), self.batch_shots):
            yield self.batch(order[start:start + self.batch_shots])

    def batch(self, shots: torch.Tensor, augment: bool | None = None) -> dict[str, torch.Tensor]:
        """The padded batch of the given shot indices (into ``runs``, a CPU tensor)."""
        inner = self.inner
        augment = self.mode == "train" if augment is None else bool(augment)
        runs = [self.runs[int(k)] for k in shots]
        lengths = torch.tensor([e - s for s, e in runs], dtype=torch.long)
        b, t = len(runs), int(lengths.max())
        steps = torch.arange(t)[None, :]
        starts = torch.tensor([s for s, _ in runs], dtype=torch.long)[:, None]
        valid = steps < lengths[:, None]
        index = torch.where(valid, starts + steps, starts.expand(b, t))
        flip = (torch.rand(b, generator=self.generator) < 0.5) if (inner.mirror and augment) else torch.zeros(b, dtype=torch.bool)
        masked = (torch.rand(b, t, generator=self.generator) < inner.thomson_rate) if (inner.thomson_rate > 0 and augment) \
            else torch.zeros(b, t, dtype=torch.bool)
        idx, flip_d = index.to(self.device), flip.to(self.device)
        valid_d, masked_d = valid.to(self.device), masked.to(self.device) & valid.to(self.device)
        x = torch.where(flip_d[:, None, None], inner.x_mirror[idx], inner.x[idx])
        if masked_d.any() and len(inner.thomson_idx):
            bb, tt = torch.nonzero(masked_d, as_tuple=True)
            x[bb[:, None], tt[:, None], inner.thomson_idx[None, :]] = inner.thomson_missing
        x = torch.clamp(x, -inner.clip_bound, inner.clip_bound) * valid_d[..., None].to(x.dtype)
        dt = self.dt[idx] * valid_d.to(self.dt.dtype)
        use = valid_d & self.kept[idx]
        pos_b, pos_t = torch.nonzero(use, as_tuple=True)
        rows = idx[pos_b, pos_t]
        frame_flip = flip_d[pos_b]
        c = inner.c[rows]
        c = torch.where(frame_flip[:, None], c * inner.parity, c)

        def flipped(maps: torch.Tensor) -> torch.Tensor:
            return torch.where(frame_flip.reshape(-1, *([1] * (maps.dim() - 1))), maps.flip(1), maps)

        out = {"x": x, "dt": dt, "lengths": lengths.to(self.device), "valid": valid_d, "pos_b": pos_b, "pos_t": pos_t,
               "index": rows, "flip": flip_d, "thomson_masked": masked_d, "c": c, "ip_a": inner.ip_a[rows],
               "weight": inner.weight[rows], "inside": flipped(unpack_masks(inner.inside[rows], inner.grid_shape)),
               "band": flipped(unpack_masks(inner.band[rows], inner.grid_shape))}
        if inner.psi_ref is not None:
            out["psi_ref"] = flipped(inner.psi_ref[rows].to(inner.dtype).reshape(len(rows), *inner.grid_shape))
        return out


def make_loader(lists: Sequence[str], set_name: str, config: ModelConfig, mode: str, paths: DataPaths = DataPaths(),
                device: torch.device | str = "cpu", ip_med: float | None = None, rule: Mapping[str, float] | None = None,
                split: Mapping | None = None, allow_local_test: bool = False) -> tuple[FrameLoader, FrameSet]:
    """``FrameLoader`` over the frames of ``lists`` with feature set ``set_name``.

    Args:
        ip_med: ``Ip_med`` of the frame weights; the median over these frames when None (pass
            the training value for validation lists).
        rule: Tail thresholds (training statistics); applied only with ``loader.tail_weights``.
    """
    config = config.with_values({"features.set": set_name}) if set_name != config.get("features.set") else config
    sets = [load_frames(name, config, paths, split, allow_local_test) for name in lists]
    frames = concat(sets) if len(sets) > 1 else sets[0]
    basis, _ = load_bases(paths)
    version = int(config.get("features.version"))
    fconfig = fe.FeatureConfig.load(paths.feature_config_path(version))
    scaler = RobustScaler.load(paths.scaler_path(version))
    ip_med = ls.ip_median(frames.ip_a) if ip_med is None else ip_med
    weights = sample_weights(frames, config, ip_med, rule if config.get("loader.tail_weights") else None)
    loader = FrameLoader(frames, basis.parity, weights, config, thomson_masking(fconfig, frames.columns, scaler), mode, device)
    return loader, frames


# ------------------------------------------------------------------------ rung 2 padding
def shot_runs(shot: np.ndarray) -> list[tuple[int, int]]:
    """``(start, end)`` of each shot's contiguous rows, in order.

    Raises:
        ValueError: when a shot's rows are not contiguous.
    """
    s = np.asarray(shot).astype(str)
    if len(s) == 0:
        return []
    starts = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])
    if len(set(s[starts])) != len(starts):
        raise ValueError("a shot's rows are not contiguous")
    ends = np.r_[starts[1:], len(s)]
    return list(zip(starts.tolist(), ends.tolist()))


def pad_by_shot(values: np.ndarray, shot: np.ndarray, pad_value: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """``(shots, T_max, ...)`` padded values and the length of each shot."""
    runs = shot_runs(shot)
    values = np.asarray(values)
    lengths = np.array([e - s for s, e in runs], dtype=np.int64)
    out = np.full((len(runs), int(lengths.max()) if len(runs) else 0, *values.shape[1:]), pad_value, dtype=values.dtype)
    for k, (s, e) in enumerate(runs):
        out[k, :e - s] = values[s:e]
    return out, lengths


def unpad(padded: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Inverse of ``pad_by_shot``: the rows in their original order."""
    return np.concatenate([padded[k, :int(n)] for k, n in enumerate(lengths)]) if len(lengths) else padded.reshape(0, *padded.shape[2:])


def time_steps(time_ms: np.ndarray, shot: np.ndarray) -> np.ndarray:
    """Step to the previous frame of the same shot in ms; 0 at each shot's first frame."""
    t = np.asarray(time_ms, dtype=np.float64)
    out = np.zeros_like(t)
    for s, e in shot_runs(shot):
        out[s + 1:e] = np.diff(t[s:e])
    return out
