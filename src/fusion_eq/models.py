"""The four baselines of the first model comparison, behind one ``predict`` interface.

- ``ridge_raw``: the plain PCA and ridge baseline on the stored maps and raw inputs.
- ``ridge_signed``: the same with the current inputs in the physical frame
  (``targets.signed_frames``).
- ``physics_mean``: no learned flux model. In the physical frame the map is the coil
  flux under the adopted coil model plus ``Ip`` times the mean plasma shape per kA,
  the mean over training frames of ``T / |Ip|``; converted back to the stored frame.
- ``hybrid_ridge``: PCA of the plasma target ``T`` and a ridge from the signed inputs;
  the prediction is the coil flux plus the decoded ``T``, converted back.

``physics_mean`` and ``hybrid_ridge`` take q95 and betaN from the scalar ridge of
``ridge_signed``, so that the three differ in the flux map alone. Every model returns
maps in the stored frame, which the scorer compares with the stored truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from fusion_eq import baseline as bl, targets
from fusion_eq.coil_flux import ADOPTED, CoilScales, GreenTable, adopted_scales
from fusion_eq.machines import MachineConfig, machine_config

MODEL_NAMES = ("ridge_raw", "ridge_signed", "physics_mean", "hybrid_ridge")
PLASMA_TAG = "plasma"


@dataclass
class Baseline:
    """One model: a flux predictor and a scalar predictor, both from frame-table rows.

    Attributes:
        name: Model name.
        flux: Frame-table rows to stored-frame maps ``(rows, 65, 65)``.
        scalars: Frame-table rows to ``(q95, betaN)`` arrays.
        scalars_from: Name of the model whose scalar ridge is used, when borrowed.
    """

    name: str
    flux: Callable[[pd.DataFrame], np.ndarray]
    scalars: Callable[[pd.DataFrame], tuple[np.ndarray, np.ndarray]]
    scalars_from: str | None = None

    def predict(self, frames: pd.DataFrame) -> bl.Prediction:
        q95, beta_n = self.scalars(frames)
        return bl.Prediction(psirz=self.flux(frames).astype(np.float32), q95=np.asarray(q95), betaN=np.asarray(beta_n))


@dataclass
class ModelSet:
    """Everything the comparison fits on a training list.

    Attributes:
        table: DIII-D Green's table.
        scales: Coil model applied to the coil flux.
        basis_psi: Basis of the stored maps (``ridge_raw``, ``ridge_signed``).
        basis_plasma: Basis of the plasma target (``hybrid_ridge``).
        ridge_raw, ridge_signed, hybrid: Fitted ``PcaRidge`` models.
        mean_shape: Mean plasma shape per kA, ``(65, 65)``.
        n_mean_frames: Frames the mean shape averages.
    """

    table: GreenTable
    scales: CoilScales
    basis_psi: bl.FluxBasis
    basis_plasma: bl.FluxBasis
    ridge_raw: bl.PcaRidge
    ridge_signed: bl.PcaRidge
    hybrid: bl.PcaRidge
    mean_shape: np.ndarray
    n_mean_frames: int
    machine: str = "DIII-D"

    def baselines(self) -> dict[str, Baseline]:
        """The four models by name."""
        cfg = machine_config(self.machine)

        def ridge_scalars(model: bl.PcaRidge, signed: bool):
            def f(frames):
                p = bl.predict(model, targets.signed_frames(frames, cfg) if signed else frames)
                return p.q95, p.betaN
            return f

        def ridge_raw_flux(frames):
            return bl.predict(self.ridge_raw, frames).psirz

        def ridge_signed_flux(frames):
            return bl.predict(self.ridge_signed, targets.signed_frames(frames, cfg)).psirz

        def physics_mean_flux(frames):
            ip = frames["ip__mean"].to_numpy(dtype=np.float64)
            coil = targets.coil_flux_of_frames(self.table, frames, self.scales, cfg)
            target = np.abs(np.nan_to_num(ip))[:, None, None] * self.mean_shape[None]
            return targets.stored_from_target(target, coil, ip, cfg)

        def hybrid_flux(frames):
            ip = frames["ip__mean"].to_numpy(dtype=np.float64)
            coil = targets.coil_flux_of_frames(self.table, frames, self.scales, cfg)
            target = bl.predict(self.hybrid, targets.signed_frames(frames, cfg)).psirz.astype(np.float64)
            return targets.stored_from_target(target, coil, ip, cfg)

        signed_scalars = ridge_scalars(self.ridge_signed, signed=True)
        return {
            "ridge_raw": Baseline("ridge_raw", ridge_raw_flux, ridge_scalars(self.ridge_raw, signed=False)),
            "ridge_signed": Baseline("ridge_signed", ridge_signed_flux, signed_scalars),
            "physics_mean": Baseline("physics_mean", physics_mean_flux, signed_scalars, scalars_from="ridge_signed"),
            "hybrid_ridge": Baseline("hybrid_ridge", hybrid_flux, signed_scalars, scalars_from="ridge_signed"),
        }


CEILING_NAMES = ("ceiling_psi_basis", "ceiling_plasma_basis")


def ceiling_baselines(model_set: ModelSet, data_root: Path | str, frame_stride: int,
                      split: str = "diii_d_train") -> dict[str, Baseline]:
    """Two ceilings with no learned flux model: the true validation maps encoded and decoded
    with the training basis of the stored map (``ceiling_psi_basis``) and with the basis of
    the plasma target, the coil flux added back (``ceiling_plasma_basis``). Both take q95
    and betaN from the scalar ridge of ``ridge_signed``, like ``physics_mean`` and
    ``hybrid_ridge``. The frames given to ``predict`` must be the strided rows of one shot.
    """
    cfg = machine_config(model_set.machine)
    scalars = model_set.baselines()["ridge_signed"].scalars

    def truth_of(frames: pd.DataFrame) -> np.ndarray:
        shots = frames["shot"].astype(str).unique()
        if len(shots) != 1:
            raise ValueError("a ceiling predicts one shot at a time")
        psi = bl.load_truth(str(shots[0]), data_root, split, frame_stride)["psi"].astype(np.float64)
        if psi.shape[0] != len(frames):
            raise ValueError(f"{shots[0]}: {len(frames)} rows for {psi.shape[0]} true maps at stride {frame_stride}")
        return psi

    def psi_basis_flux(frames):
        psi = truth_of(frames)
        return model_set.basis_psi.decode(model_set.basis_psi.encode(psi))

    def plasma_basis_flux(frames):
        psi = truth_of(frames)
        ip = frames["ip__mean"].to_numpy(dtype=np.float64)
        coil = targets.coil_flux_of_frames(model_set.table, frames, model_set.scales, cfg)
        t = targets.plasma_target(psi, coil, ip, cfg)
        decoded = model_set.basis_plasma.decode(model_set.basis_plasma.encode(t))
        return targets.stored_from_target(decoded, coil, ip, cfg)

    return {"ceiling_psi_basis": Baseline("ceiling_psi_basis", psi_basis_flux, scalars, scalars_from="ridge_signed"),
            "ceiling_plasma_basis": Baseline("ceiling_plasma_basis", plasma_basis_flux, scalars, scalars_from="ridge_signed")}


RESIDUAL_GROUPS = ("all", "rising", "flat_top", "late")


@dataclass
class ResidualSummary:
    """Residuals of one model on the validation frames.

    Attributes:
        model: Model name.
        mean_abs: Mean absolute residual per pixel, ``(65, 65)``, by group (``all``
            and the three phases).
        counts: Frames per group.
        frames: One row per validation frame: ``shot``, ``frame``, ``phase``,
            ``ip_over_peak``, ``sol_ka``, ``ip_direction``, ``rms_residual`` (over the
            map, in Wb/rad), ``flux_range`` of the true map.
    """

    model: str
    mean_abs: dict[str, np.ndarray]
    counts: dict[str, int]
    frames: pd.DataFrame


def residual_summary(name: str, model: Baseline, val_by_shot: dict[str, pd.DataFrame], data_root: Path | str,
                     frame_stride: int, psi_sign: float, split: str = "diii_d_train") -> ResidualSummary:
    """Per-pixel and per-frame residuals of a model on the (strided) validation frames.

    The residual is ``psi_sign * prediction - truth`` in the stored frame, with the
    global flux sign the scorer chose. Phases follow ``coil_scales.phase_of`` on the
    frame table's ``ip_over_peak``; the rows of each shot must carry ``frame``,
    ``n_frames``, ``ip_over_peak``, ``ip__mean`` and ``sol__mean``.
    """
    from fusion_eq.coil_scales import phase_of
    sums = {g: np.zeros((bl.GRID, bl.GRID)) for g in RESIDUAL_GROUPS}
    counts = {g: 0 for g in RESIDUAL_GROUPS}
    rows = []
    for shot_id, frames in val_by_shot.items():
        truth = bl.load_truth(shot_id, data_root, split, frame_stride)["psi"].astype(np.float64)
        if truth.shape[0] != len(frames):
            raise ValueError(f"{shot_id}: {len(frames)} rows for {truth.shape[0]} true maps")
        pred = model.predict(frames).psirz.astype(np.float64) * psi_sign
        resid = pred - truth
        rms = np.sqrt(np.mean(resid * resid, axis=(1, 2)))
        flux_range = truth.max(axis=(1, 2)) - truth.min(axis=(1, 2))
        ip = frames["ip__mean"].to_numpy(dtype=np.float64)
        for k, (frame, over, n) in enumerate(zip(frames["frame"], frames["ip_over_peak"], frames["n_frames"])):
            phase = phase_of(float(over), int(frame), int(n))
            a = np.abs(resid[k])
            sums["all"] += a
            sums[phase] += a
            counts["all"] += 1
            counts[phase] += 1
            rows.append({"model": name, "shot": shot_id, "frame": int(frame), "phase": phase, "ip_over_peak": float(over),
                         "sol_ka": float(frames["sol__mean"].iloc[k]), "ip_direction": int(np.sign(ip[k])) if np.isfinite(ip[k]) else 0,
                         "rms_residual": float(rms[k]), "flux_range": float(flux_range[k])})
    mean_abs = {g: (sums[g] / counts[g] if counts[g] else np.full((bl.GRID, bl.GRID), np.nan)) for g in RESIDUAL_GROUPS}
    return ResidualSummary(model=name, mean_abs=mean_abs, counts=counts, frames=pd.DataFrame(rows))


def score_parts_table(name: str, result: bl.LocalScore) -> pd.DataFrame:
    """The scorer's per-shot parts of a model as one row per shot."""
    ls = bl.local_score_module()
    rows = []
    for shot_id, part in zip(result.shot_ids, result.parts):
        row = {"model": name, "shot": shot_id, "psi_sign": result.psi_sign,
               "ss_res_psi": part["ss_res_psi"][result.psi_sign], "ss_res_psi_unflipped": part["ss_res_psi"][1.0],
               "psi_pixels": part["psi_pixels"], "psi_nonfinite": part["psi_nonfinite"],
               "shot_r2_psi": part["shot_r2_psi"][result.psi_sign] if part["shot_r2_psi"] is not None else np.nan,
               "shot_dlcfs": np.nan if part["shot_dlcfs"] is None else part["shot_dlcfs"],
               "lcfs_frames": part["lcfs_frames"], "lcfs_fails": part["lcfs_fails"], "ref_lcfs_dropped": part["ref_lcfs_dropped"],
               "missing": bool(part["missing"])}
        for j, scalar in enumerate(ls.SCALARS):
            row[f"ss_res_{scalar}"] = float(part["ss_res_scal"][j])
        for j, scalar in enumerate(ls.CONS_SCALARS):
            row[f"ss_res_{scalar}"] = float(part["ss_res_cons"][j])
            row[f"cons_frames_{scalar}"] = int(part["cons_frames"][j])
            row[f"cons_fails_{scalar}"] = int(part["cons_fails"][j])
        rows.append(row)
    return pd.DataFrame(rows)


def plasma_target_fn(table: GreenTable, frames_by_shot: dict[str, pd.DataFrame], scales=ADOPTED,
                     machine: str | MachineConfig = "DIII-D") -> Callable[[str, np.ndarray], np.ndarray]:
    """``(shot_id, stored maps) -> T`` for ``baseline.encode_shots`` and the basis fit,
    from the frame-table rows of each shot (in frame order)."""
    def fn(shot_id: str, maps: np.ndarray) -> np.ndarray:
        rows = frames_by_shot[shot_id]
        if len(rows) != maps.shape[0]:
            raise ValueError(f"{shot_id}: {len(rows)} frame rows for {maps.shape[0]} maps")
        return targets.plasma_target_of_frames(maps, table, rows, scales, machine)
    return fn


def mean_plasma_shape(shot_ids, data_root: Path | str, target_fn, frames_by_shot: dict[str, pd.DataFrame],
                      cache_dir: Path | str, tag: str, split: str = "diii_d_train") -> tuple[np.ndarray, int]:
    """Mean of ``T / |Ip|`` over all frames of the shots, cached under ``cache_dir``.

    Args:
        shot_ids: Training shots.
        data_root: Raw data root.
        target_fn: ``(shot_id, stored maps) -> T`` (``plasma_target_fn``).
        frames_by_shot: Frame-table rows per shot, for ``ip__mean``.
        cache_dir, tag: Cache folder and the tag of the coil model.
        split: Labelled split of the shots.

    Returns:
        ``(mean shape, frames averaged)``.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"mean_shape_{bl.hash_of_ids(shot_ids)}_{tag}.npz"
    if path.is_file():
        with np.load(path, allow_pickle=False) as f:
            return f["mean"], int(f["n"])
    acc = targets.MeanShapeAccumulator()
    for shot_id, maps in bl.shot_maps(sorted(set(shot_ids)), data_root, split):
        acc.add(target_fn(shot_id, maps), frames_by_shot[shot_id]["ip__mean"].to_numpy(dtype=np.float64))
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, mean=acc.mean, n=acc.n)
    tmp.replace(path)
    return acc.mean, acc.n


def fit_model_set(train_frames: pd.DataFrame, train_ids: list[str], data_root: Path | str, table: GreenTable,
                  params: bl.BaselineParams, cache_dir: Path | str, scales=ADOPTED, machine: str = "DIII-D",
                  split: str = "diii_d_train", log=print) -> ModelSet:
    """Fits the four models on the training list; bases, coefficients and the mean shape are cached.

    Args:
        train_frames: Frame-table rows of the training shots (inputs and the two scalar
            labels), in frame-table order.
        train_ids: Training shot ids (sorted).
        data_root: Raw data root.
        table: DIII-D Green's table.
        params: Baseline parameters (components, alpha, basis frames, seed).
        cache_dir: Cache of bases, coefficients and the mean shape.
        scales: Coil model; the adopted DIII-D record by default.
        machine: Machine name.
        split: Labelled split of the shots.
        log: Progress printer.
    """
    cfg = machine_config(machine)
    scales = adopted_scales(machine, table.columns) if isinstance(scales, str) and scales == ADOPTED else scales
    train_ids = sorted(set(train_ids))
    frames_by_shot = {s: g for s, g in train_frames.groupby(train_frames["shot"].astype(str), sort=False)}
    if sorted(frames_by_shot) != train_ids:
        raise ValueError("training frames and training ids do not match")
    target_fn = plasma_target_fn(table, frames_by_shot, scales, cfg)
    tag = f"{PLASMA_TAG}_{scales.source.split(',')[0] or 'scales'}"

    basis_psi, s1 = bl.cached_basis(train_ids, params, data_root, cache_dir, split=split)
    log(f"basis of the stored maps: {basis_psi.n_components} components over {basis_psi.n_frames} frames, {s1:.0f} s")
    basis_plasma, s2 = bl.cached_basis(train_ids, params, data_root, cache_dir, split=split, target=target_fn, tag=tag)
    log(f"basis of the plasma target: {basis_plasma.n_components} components over {basis_plasma.n_frames} frames, {s2:.0f} s")
    coefs_psi = bl.encode_shots(train_ids, basis_psi, data_root, cache_dir, split)
    coefs_plasma = bl.encode_shots(train_ids, basis_plasma, data_root, cache_dir, split, target=target_fn, tag=tag)
    log(f"{coefs_psi.shape[0]} training frames encoded in both bases")
    mean_shape, n_mean = mean_plasma_shape(train_ids, data_root, target_fn, frames_by_shot, cache_dir, tag, split)
    log(f"mean plasma shape over {n_mean} frames")

    signed = targets.signed_frames(train_frames, cfg)
    ridge_raw = bl.fit_pca_ridge(train_frames, coefs_psi, basis_psi, params)
    ridge_signed = bl.fit_pca_ridge(signed, coefs_psi, basis_psi, params)
    hybrid = bl.fit_pca_ridge(signed, coefs_plasma, basis_plasma, params, scaler=ridge_signed.scaler)
    return ModelSet(table=table, scales=scales, basis_psi=basis_psi, basis_plasma=basis_plasma, ridge_raw=ridge_raw,
                    ridge_signed=ridge_signed, hybrid=hybrid, mean_shape=mean_shape, n_mean_frames=n_mean, machine=machine)
