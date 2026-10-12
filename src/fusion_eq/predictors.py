"""Registered predictors of the submission pipeline.

``baseline_v0``: on DIII-D the ``hybrid_ridge`` model of the baseline comparison (PCA
and ridge of the plasma target added to the coil flux under the adopted coil model)
fitted on the training shots of a split, with q95 and betaN from the ``ridge_signed``
scalar ridge; on MAST the calculated coil flux alone (MAST Green's table, every scale 1,
no flat term) on the scorer's grid, converted to the stored frame with the DIII-D sign
rule, ``-sign(Ip)`` times the physical flux, and q95 and betaN set to the DIII-D
training medians as a placeholder.

The fitted models are the files ``compare_baselines.py`` saves under the baseline cache,
named by the hash of the training list. ``setup`` loads them, or fits and saves them
once when they are missing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np

from fusion_eq import baseline as bl, coil_flux as cf, frame_table as ft, io, models, split as sp, submission, targets
from fusion_eq.dataset import DEFAULT_PROCESSED_ROOT, REPO_ROOT, default_data_root
from fusion_eq.machines import machine_config

DEFAULT_SPLIT = REPO_ROOT / "splits" / "split_v1.json"
DEFAULT_GREEN_DIR = DEFAULT_PROCESSED_ROOT / "green"
GREEN_FILES = {"DIII-D": "green_d3d.npz", "MAST": "green_mast.npz"}


def green_table(green_dir: Path | str, machine: str) -> cf.GreenTable:
    """Loads a machine's Green's table from the build directory."""
    return cf.GreenTable.load(Path(green_dir) / GREEN_FILES[machine])


def training_scalar_medians(split_path: Path | str, cache_dir: Path | str, table_root: Path | str = ft.DEFAULT_TABLE_ROOT,
                            part: str = "train") -> dict[str, float]:
    """Medians of ``efit_q95`` and ``efit_beta_n`` over the frames of the split's training
    shots, read from the frame table once and cached as JSON under ``cache_dir``."""
    split = sp.read_split(split_path)
    ids = sorted(split[part])
    path = Path(cache_dir) / f"scalar_medians_{bl.hash_of_ids(ids)}.json"
    if path.is_file():
        return {k: float(v) for k, v in json.loads(path.read_text()).items()}
    rows = ft.read_frame_table(sp.TRAIN_SPLIT, list(bl.SCALAR_TARGETS.values()), ids, root=table_root)
    medians = {name: float(np.nanmedian(rows[column].to_numpy(dtype=np.float64))) for name, column in bl.SCALAR_TARGETS.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(medians, indent=1, sort_keys=True) + "\n")
    return medians


def model_files(cache_dir: Path | str, train_ids) -> dict[str, Path]:
    """The files ``compare_baselines.py`` saves for a training list."""
    h = bl.hash_of_ids(train_ids)
    cache_dir = Path(cache_dir)
    return {"hybrid_ridge": cache_dir / f"model_hybrid_ridge_{h}.npz", "ridge_signed": cache_dir / f"model_ridge_signed_{h}.npz",
            "ridge_raw": cache_dir / f"model_ridge_raw_{h}.npz", "mean_shape": cache_dir / f"mean_shape_{h}_adopted.npz"}


def file_identifier(path: Path) -> dict:
    return {"path": str(path), "sha256": submission.sha256_of_file(path)}


@submission.register("baseline_v0")
class BaselineV0:
    """First submission predictor; see the module docstring.

    Args:
        split_path: Split file; the models are those of its ``train`` list.
        cache_dir: Baseline cache holding the fitted models.
        green_dir: Directory of the Green's tables.
        table_root: Frame-table build, read when the models or the medians must be fitted.
        data_root: Raw data root, read when the models must be fitted.
        params: Baseline parameters of the fit.
        build_params: Frame-table parameters used to build a shot's input rows; the
            defaults are those of the frame-table build the models were fitted on.
    """

    name = "baseline_v0"

    def __init__(self, split_path: Path | str = DEFAULT_SPLIT, cache_dir: Path | str = bl.DEFAULT_CACHE_DIR,
                 green_dir: Path | str = DEFAULT_GREEN_DIR, table_root: Path | str = ft.DEFAULT_TABLE_ROOT,
                 data_root: Path | str | None = None, params: bl.BaselineParams = bl.BaselineParams(),
                 build_params: ft.BuildParams = ft.BuildParams()) -> None:
        self.split_path = Path(split_path)
        self.cache_dir = Path(cache_dir)
        self.green_dir = Path(green_dir)
        self.table_root = Path(table_root)
        self.data_root = Path(data_root) if data_root is not None else default_data_root()
        self.params = params
        self.build_params = build_params
        self.model_set: models.ModelSet | None = None
        self.table_mast: cf.GreenTable | None = None
        self.scales_mast: cf.CoilScales | None = None
        self.scalar_medians: dict[str, float] | None = None
        self.identifiers: dict = {}

    def setup(self, log: Callable[[str], None] = print) -> dict:
        split = sp.read_split(self.split_path)
        train_ids = sorted(split["train"])
        files = model_files(self.cache_dir, train_ids)
        table = green_table(self.green_dir, "DIII-D")
        scales = cf.adopted_scales("DIII-D", table.columns)
        needed = [files[k] for k in ("hybrid_ridge", "ridge_signed", "mean_shape")]
        if not all(p.is_file() for p in needed):
            log(f"fitted models missing under {self.cache_dir}; fitting on {len(train_ids)} training shots")
            columns = bl.input_columns()
            frames = bl.frames_of_shots(train_ids, [*columns, *bl.SCALAR_TARGETS.values()], self.table_root)
            frames["shot"] = frames["shot"].astype(str)
            fitted = models.fit_model_set(frames, train_ids, self.data_root, table, self.params, self.cache_dir, scales, log=log)
            for name, model in (("ridge_raw", fitted.ridge_raw), ("ridge_signed", fitted.ridge_signed), ("hybrid_ridge", fitted.hybrid)):
                model.save(files[name])
            np.savez(files["mean_shape"], mean=fitted.mean_shape, n=fitted.n_mean_frames)
        hybrid = bl.PcaRidge.load(files["hybrid_ridge"])
        ridge_signed = bl.PcaRidge.load(files["ridge_signed"])
        with np.load(files["mean_shape"], allow_pickle=False) as f:
            mean_shape, n_mean = f["mean"], int(f["n"])
        # ``ModelSet.baselines()["hybrid_ridge"]`` reads the table, the scales, ``hybrid`` and
        # ``ridge_signed`` only; the other slots are filled with the loaded models so that the
        # prediction is the arithmetic of the scored comparison, not a copy of it.
        self.model_set = models.ModelSet(table=table, scales=scales, basis_psi=ridge_signed.basis, basis_plasma=hybrid.basis,
                                         ridge_raw=ridge_signed, ridge_signed=ridge_signed, hybrid=hybrid,
                                         mean_shape=mean_shape, n_mean_frames=n_mean)
        self.table_mast = green_table(self.green_dir, "MAST")
        self.scales_mast = cf.CoilScales.default("MAST", self.table_mast.columns)
        self.scalar_medians = training_scalar_medians(self.split_path, self.cache_dir, self.table_root)
        self.identifiers = {
            "split_file": str(self.split_path), "split_commit": split.get("repo_commit"), "n_train_shots": len(train_ids),
            "train_hash": bl.hash_of_ids(train_ids), "params": self.params.to_dict(), "build_params": self.build_params.to_dict(),
            "models": {k: file_identifier(files[k]) for k in ("hybrid_ridge", "ridge_signed", "mean_shape")},
            "n_mean_frames": n_mean, "scalar_medians": dict(self.scalar_medians),
            "coil_model": {"DIII-D": {"scales": scales.to_dict(), "geometry_hash": table.geometry_hash, "grid_hash": table.grid_hash,
                                      "n_filaments": table.n_filaments},
                           "MAST": {"scales": self.scales_mast.to_dict(), "geometry_hash": self.table_mast.geometry_hash,
                                    "grid_hash": self.table_mast.grid_hash, "n_filaments": self.table_mast.n_filaments}},
            "mast_sign_rule_c": targets.sign_rule_c("DIII-D"),
            "mast_scalars": "placeholder: DIII-D training medians",
        }
        log(f"baseline_v0 ready: hybrid_ridge of {len(train_ids)} training shots, medians q95 {self.scalar_medians['q95']:.3f}, "
            f"betaN {self.scalar_medians['betaN']:.3f}")
        return self.identifiers

    def _check_ready(self) -> None:
        if self.model_set is None or self.table_mast is None or self.scalar_medians is None:
            raise RuntimeError("call setup() first")

    def _frames(self, shot: io.Shot, machine: str):
        return ft.build_shot_frames(shot, submission.TEST_CONFIGS[machine], self.build_params, with_labels=False)

    def _predict_mast(self, shot: io.Shot) -> bl.Prediction:
        """Coil flux alone on the scorer's grid, DIII-D sign rule, scalars at the DIII-D medians."""
        cfg = machine_config("MAST")
        currents = cf.currents_at_frames(shot, self.table_mast.columns)
        ip = cf.currents_at_frames(shot, [cfg.ip_column])[:, 0]
        coil = self.table_mast.flux(currents, self.scales_mast)
        # Same rule as DIII-D (F45): psi_stored = c sign(Ip) psi_physical with c = -1.
        psi = targets.sign_rule_c("DIII-D") * targets.sign_of_ip(ip)[:, None, None] * coil
        n = psi.shape[0]
        return bl.Prediction(psirz=psi.astype(np.float32), q95=np.full(n, self.scalar_medians["q95"]),
                             betaN=np.full(n, self.scalar_medians["betaN"]))

    def predict_shot(self, shot: io.Shot, machine: str) -> bl.Prediction:
        self._check_ready()
        if machine == "DIII-D":
            return self.model_set.baselines()["hybrid_ridge"].predict(self._frames(shot, machine))
        if machine == "MAST":
            return self._predict_mast(shot)
        raise ValueError(f"unknown machine {machine!r}")


@submission.register("coil_only_v0")
class CoilOnlyV0(BaselineV0):
    """Calculated coil flux alone on both machines: the reference for how much of the map the
    coils explain without any plasma model.

    DIII-D: the coil flux under the adopted coil model (shaping coils as stored, flat solenoid
    term), converted to the stored frame with the sign rule, ``psi_stored = c sign(Ip) coil``;
    q95 and betaN from the ``ridge_signed`` scalar ridge, as in ``baseline_v0``. MAST: as in
    ``baseline_v0``. Same setup and files as ``baseline_v0``.
    """

    name = "coil_only_v0"

    def setup(self, log: Callable[[str], None] = print) -> dict:
        identifiers = super().setup(log)
        identifiers["d3d_flux_model"] = "coil flux only (adopted coil model), no plasma term"
        return identifiers

    def predict_shot(self, shot: io.Shot, machine: str) -> bl.Prediction:
        self._check_ready()
        if machine == "DIII-D":
            cfg = machine_config(machine)
            frames = self._frames(shot, machine)
            ip = frames["ip__mean"].to_numpy(dtype=np.float64)
            coil = targets.coil_flux_of_frames(self.model_set.table, frames, self.model_set.scales, cfg)
            psi = targets.stored_from_target(np.zeros_like(coil), coil, ip, cfg)
            q95, beta_n = self.model_set.baselines()["ridge_signed"].scalars(frames)
            return bl.Prediction(psirz=psi.astype(np.float32), q95=np.asarray(q95), betaN=np.asarray(beta_n))
        if machine == "MAST":
            return self._predict_mast(shot)
        raise ValueError(f"unknown machine {machine!r}")
