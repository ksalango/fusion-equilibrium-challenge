"""The organizers' scorer on the composed validation maps of a model (``docs/PIPELINE.md``,
Part 4: the scorer at the warm start and at the best checkpoint, and the Optuna objective).

For every ``scoring.stride``-th frame of every shot of a validation list (all frames, whether or
not they pass the label filter, as the scorer sees them): the scaled, clipped features of the
model's columns; the model's standardized coefficients times the component standard deviations;
the map ``compose_map`` assembles from the frame's coil flux (adopted DIII-D coil model), the
flux basis and the remainder ridge of the target build; and q95 and betaN from the fixed scalar
ridge of ``baseline_v0`` (``ridge_signed``, fitted on the run's training list). The scalars do
not depend on the model, so the score moves with the map alone, which is what the Optuna
objective asks for. The decided scalar heads (q95 from the map, the betaN heads) belong to the
prediction chain of stage 3.

A sequence model (rung 2) is run on every frame of each validation shot, with the time steps,
and its outputs are then read at the scored frames, so that it sees the shot as it is.

Slices, each finalized by the scorer: ``all``; ``no_near_repeat`` (shots without a near-repeat in
the training list, the selection subset of D19 item 7); ``ip_negative`` (shots with negative
Ip, item 9); ``corner`` (the MAST-like corner of F99: frames with ``q_cyl`` below the training
quantile and ``ts_beta_p`` below the training median, relaxed to the training 75th percentile
when fewer than ``scoring.corner_min_frames`` validation frames qualify), scored as if only its
frames existed under the flux sign chosen over all frames.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping

import numpy as np
import pandas as pd

from fusion_eq import (baseline as bl, current_basis as cb, dataset, features as fe, frame_table as ft, loader as ld,
                       predictors, split as sp, target_build as tb, targets)
from fusion_eq.coil_flux import GreenTable, adopted_scales
from fusion_eq.feature_scaler import RobustScaler
from fusion_eq.machines import machine_config
from fusion_eq.model_config import ModelConfig
from fusion_eq.remainder_model import RemainderModel

MACHINE = "DIII-D"
SPLIT_NAME = tb.LABELLED_SPLIT
SLICES = ("all", "no_near_repeat", "ip_negative", "corner")


@dataclass
class ScoringPaths:
    """Locations the scorer reads besides the model inputs."""

    data_root: Path = dataset.DEFAULT_DATA_ROOT
    table_dir: Path = ft.DEFAULT_TABLE_ROOT
    green_dir: Path = dataset.DEFAULT_PROCESSED_ROOT / "green"
    references_dir: Path = bl.DEFAULT_REFERENCE_DIR
    baseline_cache: Path = bl.DEFAULT_CACHE_DIR


def corner_thresholds(train_ids, config: ModelConfig, paths: ld.DataPaths) -> dict[str, float]:
    """Training-list statistics of the corner slice: the ``q_cyl`` quantile, the ``ts_beta_p``
    median and its relaxed quantile, over every frame of the training list's feature rows."""
    version = int(config.get("features.version"))
    rows = fe.read_features(ld.FEATURE_SPLIT, version, ["q_cyl", "ts_beta_p"], train_ids, paths.feature_root)
    q = rows["q_cyl"].to_numpy(dtype=np.float64)
    b = rows["ts_beta_p"].to_numpy(dtype=np.float64)
    q, b = q[np.isfinite(q)], b[np.isfinite(b)]
    return {"q_cyl": float(np.quantile(q, config.get("scoring.corner_q_cyl_quantile"))),
            "ts_beta_p": float(np.quantile(b, config.get("scoring.corner_beta_p_quantile"))),
            "ts_beta_p_relaxed": float(np.quantile(b, config.get("scoring.corner_beta_p_relaxed")))}


def corner_mask(q_cyl, beta_p, thresholds: Mapping[str, float], min_frames: int) -> tuple[np.ndarray, bool]:
    """Corner frames and whether the relaxed ``ts_beta_p`` threshold was needed."""
    q_cyl, beta_p = np.asarray(q_cyl, dtype=np.float64), np.asarray(beta_p, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        strict = (q_cyl < thresholds["q_cyl"]) & (beta_p < thresholds["ts_beta_p"])
        if strict.sum() >= min_frames:
            return strict, False
        return (q_cyl < thresholds["q_cyl"]) & (beta_p < thresholds["ts_beta_p_relaxed"]), True


@dataclass
class ShotInputs:
    """One validation shot's strided frames: the rows the maps and scalars need, the model's
    features (scaled, clipped) and the fixed scalars; for sequence models, the features and the
    time-step input of every frame and the positions of the strided frames among them."""

    rows: pd.DataFrame
    x: np.ndarray
    q95: np.ndarray
    beta_n: np.ndarray
    corner: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    x_full: np.ndarray | None = None
    dt_full: np.ndarray | None = None
    positions: np.ndarray | None = None


class ValidationScorer:
    """Scores a model on a validation list with the organizers' scorer.

    Args:
        list_name: Validation list of the split file.
        train_list: Training list of the run (the scalar ridge and the corner statistics).
        config: Model configuration (feature set, clipping, scoring fields).
        paths: Model input locations.
        scoring_paths: Raw data, frame table, Green's tables, reference and baseline caches.
        workers: Scoring processes.
        log: Progress messages.

    Raises:
        FileNotFoundError: when the scalar ridge of ``baseline_v0`` for the training list is
            not in the baseline cache.
    """

    def __init__(self, list_name: str, train_list: str, config: ModelConfig, paths: ld.DataPaths = ld.DataPaths(),
                 scoring_paths: ScoringPaths = ScoringPaths(), workers: int | None = None,
                 log: Callable[[str], None] = print) -> None:
        ld.require_readable(list_name)
        self.config = config
        self.stride = int(config.get("scoring.stride"))
        self.workers = workers
        self.scoring_paths = scoring_paths
        split = sp.read_split(paths.split)
        self.shot_ids = sorted(split[list_name])
        train_ids = sorted(split[train_list])
        cfg = machine_config(MACHINE)

        version = int(config.get("features.version"))
        fconfig = fe.FeatureConfig.load(paths.feature_config_path(version))
        scaler = RobustScaler.load(paths.scaler_path(version))
        self.columns = ld.model_columns(fconfig, config.get("features.set"), config.get("features.drop_groups"))
        basis, self.flux = ld.load_bases(paths)
        self.sigma = ld.component_sigma(basis)
        self.remainder = RemainderModel.load(paths.target_dir() / tb.OBJECT_FILES["remainder_model"])
        self.table = GreenTable.load(Path(scoring_paths.green_dir) / predictors.GREEN_FILES[MACHINE])
        self.scales = adopted_scales(MACHINE, self.table.columns)

        files = predictors.model_files(scoring_paths.baseline_cache, train_ids)
        if not files["ridge_signed"].is_file():
            raise FileNotFoundError(f"{files['ridge_signed']} not found: the scalar ridge of baseline_v0 for the {train_list} list "
                                    f"(run python scripts/compare_baselines.py{' --working' if train_list.startswith('working') else ''})")
        scalar_ridge = bl.PcaRidge.load(files["ridge_signed"])
        self.scalar_source = {"model": "ridge_signed", "path": str(files["ridge_signed"])}

        ridge_columns = list(self.remainder.ridges[self.remainder.default_window].columns)
        columns = list(dict.fromkeys([*targets.current_columns_of(self.table, MACHINE), "ip__mean", *ridge_columns,
                                      *bl.input_columns(), "ip_over_peak", "n_frames"]))
        rows = bl.frames_of_shots(self.shot_ids, columns, scoring_paths.table_dir)
        rows["shot"] = rows["shot"].astype(str)
        feats = fe.read_features(ld.FEATURE_SPLIT, version, list(dict.fromkeys([*self.columns, "q_cyl", "ts_beta_p"])),
                                 self.shot_ids, paths.feature_root)
        feats["shot"] = feats["shot"].astype(str)
        self.thresholds = corner_thresholds(train_ids, config, paths)
        bound = float(config.get("loader.clip_bound"))

        self.shots: dict[str, ShotInputs] = {}
        corner_inputs = []
        for shot, full in rows.groupby("shot", sort=True):
            full = full.sort_values("frame").reset_index(drop=True)
            positions = np.arange(len(full))[::self.stride]
            g = full.iloc[positions].reset_index(drop=True)
            f_full = full[["shot", "frame"]].merge(feats, on=["shot", "frame"], how="left", validate="one_to_one")
            x_full = np.clip(scaler.transform(f_full, MACHINE, self.columns), -bound, bound)
            f = f_full.iloc[positions].reset_index(drop=True)
            dt_full = ld.dt_input(full["time_ms"].to_numpy(dtype=np.float64), full["shot"].to_numpy(),
                                  float(config.get("rung2.dt_scale_ms")), bound)
            scal = bl.predict(scalar_ridge, targets.signed_frames(g, cfg))
            self.shots[shot] = ShotInputs(rows=g, x=x_full[positions], q95=np.asarray(scal.q95), beta_n=np.asarray(scal.betaN),
                                          x_full=x_full, dt_full=dt_full, positions=positions)
            corner_inputs.append(np.column_stack([f["q_cyl"].to_numpy(dtype=np.float64), f["ts_beta_p"].to_numpy(dtype=np.float64)]))

        q_b = np.concatenate(corner_inputs)
        mask, self.corner_relaxed = corner_mask(q_b[:, 0], q_b[:, 1], self.thresholds, int(config.get("scoring.corner_min_frames")))
        start = 0
        corner_frames = {}
        for shot, s in self.shots.items():
            n = len(s.rows)
            s.corner = mask[start:start + n]
            corner_frames[shot] = np.flatnonzero(s.corner)
            start += n
        self.n_corner_frames = int(mask.sum())

        log(f"scorer: {len(self.shot_ids)} {list_name} shots at stride {self.stride}, {sum(len(s.rows) for s in self.shots.values())} frames, "
            f"{self.n_corner_frames} corner frames{' (relaxed)' if self.corner_relaxed else ''}")
        self.references = bl.build_references(self.shot_ids, scoring_paths.data_root, SPLIT_NAME, self.stride, workers=workers,
                                              cache_dir=scoring_paths.references_dir)
        self.corner_references = bl.subset_references(self.references, corner_frames, scoring_paths.data_root, SPLIT_NAME) \
            if self.n_corner_frames else None
        near = ld.near_repeat_flags(split, np.array(self.shot_ids))
        index = pd.read_parquet(Path(scoring_paths.table_dir) / "shot_index.parquet")
        index = index.loc[index["split"] == SPLIT_NAME].assign(shot=lambda t: t["shot"].astype(str)).set_index("shot")
        direction = index["ip_direction"].reindex(self.shot_ids).to_numpy()
        self.slices = {"all": np.ones(len(self.shot_ids), dtype=bool), "no_near_repeat": ~near, "ip_negative": direction == -1}

    def record(self) -> dict:
        return {"stride": self.stride, "n_shots": len(self.shot_ids), "n_frames": int(sum(len(s.rows) for s in self.shots.values())),
                "scalars": self.scalar_source, "corner_thresholds": self.thresholds, "corner_relaxed": self.corner_relaxed,
                "n_corner_frames": self.n_corner_frames, "slice_shots": {k: int(v.sum()) for k, v in self.slices.items()}}

    def maps(self, shot: str, c_std: np.ndarray) -> bl.Prediction:
        """The composed stored-frame maps and the fixed scalars of one shot."""
        s = self.shots[shot]
        cfg = machine_config(MACHINE)
        ip = s.rows["ip__mean"].to_numpy(dtype=np.float64)
        coil = targets.coil_flux_of_frames(self.table, s.rows, self.scales, cfg)
        psi = cb.compose_map(cfg, coil, ip, np.asarray(c_std, dtype=np.float64) * self.sigma, self.flux, self.remainder.predict(s.rows))
        return bl.Prediction(psirz=psi.astype(np.float32), q95=s.q95, betaN=s.beta_n)

    def __call__(self, predict_c: Callable[..., np.ndarray], sequence: bool = False) -> dict[str, dict]:
        """Scores ``predict_c``: the score terms of every slice.

        Args:
            predict_c: Scaled, clipped features ``(N, F)`` to standardized coefficients; with
                ``sequence``, one shot's features of every frame and its time-step input
                ``(T,)`` to the coefficients of every frame.
            sequence: Whether ``predict_c`` is a sequence model's.
        """
        if sequence:
            coefficients = {shot: predict_c(s.x_full, s.dt_full)[s.positions] for shot, s in self.shots.items()}
        else:
            coefficients = {shot: predict_c(s.x) for shot, s in self.shots.items()}

        def predict(shot: str) -> bl.Prediction:
            return self.maps(shot, coefficients[shot])

        sp_root, workers = self.scoring_paths.data_root, self.workers
        result = bl.score_local(predict, self.shot_ids, sp_root, SPLIT_NAME, self.stride, self.references, workers)
        out = {name: bl.score_terms(result.finalize(mask)) if mask.any() else {"S": float("nan"), "n_shots": 0}
               for name, mask in self.slices.items()}
        if self.corner_references is not None:
            part = bl.score_local(predict, self.corner_references.shot_ids, sp_root, SPLIT_NAME, self.stride, self.corner_references,
                                  workers, psi_sign=result.psi_sign)
            out["corner"] = bl.score_terms(part.finalize())
        else:
            out["corner"] = {"S": float("nan"), "n_shots": 0}
        return {name: {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v) for k, v in terms.items()}
                for name, terms in out.items()}
