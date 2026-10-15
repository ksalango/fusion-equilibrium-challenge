"""Synthetic inputs of the model phase: a grid symmetric about Z = 0, a flux basis with exact
up-down parity, ellipse masks, and a workspace holding every file the loader reads (feature
table, scaler, target table, current and flux bases, loss inputs, split file). No real data is
read."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from fusion_eq import current_basis as cb, features as fe, loss_inputs as li, target_build as tb
from fusion_eq.feature_scaler import RobustScaler
from fusion_eq.loader import DataPaths
from fusion_eq.model_config import PHASES

REPO_ROOT = Path(__file__).resolve().parents[1]
FEATURE_CONFIG = REPO_ROOT / "configs" / "features_v2.yaml"
N_GRID = 65


def symmetric_grid(n: int = N_GRID) -> tuple[np.ndarray, np.ndarray]:
    """DIII-D-like axes in metres, Z symmetric about 0."""
    return np.linspace(0.84, 2.54, n), np.linspace(-1.6, 1.6, n)


def parity_basis(k: int, shape: tuple[int, int], rng: np.random.Generator, scale: float = 1e-7) -> dict:
    """``phi_mean`` (even), ``phi`` ``(k, P)`` with exact parity, the parity vector and
    variances, in Wb/rad per ampere-like magnitudes."""
    n_z, n_r = shape
    rr, zz = np.meshgrid(np.linspace(-1, 1, n_r), np.linspace(-1, 1, n_z))
    parity = np.array([1 if j % 2 == 0 else -1 for j in range(k)], dtype=np.int64)
    phi = np.zeros((k, n_z * n_r))
    for j in range(k):
        half = rng.normal(size=(n_z, n_r)) * np.exp(-2 * (rr ** 2 + zz ** 2)) + np.cos((j + 1) * rr) * np.exp(-zz ** 2)
        field = half + parity[j] * half[::-1]
        phi[j] = field.ravel() * scale
    mean = np.exp(-3 * (rr ** 2 + zz ** 2))
    phi_mean = (mean + mean[::-1]).ravel() * scale
    variances = np.linspace(4.0, 0.5, k) ** 2 * 1e-4
    return {"phi": phi, "phi_mean": phi_mean, "parity": parity, "variances": variances}


def ellipse_masks(n: int, grid_r, grid_z, rng: np.random.Generator, empty_every: int = 0) -> np.ndarray:
    """``(n, nZ, nR)`` bool masks of random ellipses (off-centre in Z, so not symmetric);
    every ``empty_every``-th mask is empty when positive."""
    rr, zz = np.meshgrid(grid_r, grid_z)
    out = np.zeros((n, grid_z.size, grid_r.size), dtype=bool)
    for i in range(n):
        if empty_every and i % empty_every == empty_every - 1:
            continue
        r0, z0 = rng.uniform(1.5, 1.9), rng.uniform(-0.3, 0.3)
        a, b = rng.uniform(0.3, 0.55), rng.uniform(0.6, 1.1)
        out[i] = ((rr - r0) / a) ** 2 + ((zz - z0) / b) ** 2 < 1.0
    return out


def mirror_matrix(fconfig: fe.FeatureConfig, columns: list[str]) -> np.ndarray:
    """``A`` with ``mirror_features`` equal to ``x A`` on the scaled columns (sign changes of
    odd columns, exchange of permuted columns with their partners)."""
    specs = fconfig.column_specs()
    index = {c: j for j, c in enumerate(columns)}
    a = np.zeros((len(columns), len(columns)))
    for spec in fconfig.features:
        cols = [c for c in fconfig.columns_of(spec) if c in index]
        if spec.mirror == "permute":
            partner = fconfig.layout.mirror_index()
            for i, c in enumerate(cols):
                a[index[cols[partner[i]]], index[c]] = 1.0
        else:
            for c in cols:
                a[index[c], index[c]] = -1.0 if spec.mirror == "odd" else 1.0
    return a


def feature_values(fconfig: fe.FeatureConfig, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Unscaled random values of every configured column, Thomson features NaN where
    ``ts_valid`` is 0."""
    cols = fconfig.all_columns()
    data = {c: rng.normal(size=n) * (1.0 + 0.1 * j) + 0.2 * j for j, c in enumerate(cols)}
    valid = (rng.uniform(size=n) > 0.15).astype(np.float64)
    data["ts_valid"] = valid
    for c in ("ts_beta_p", "ts_pe_peaking"):
        data[c] = np.where(valid > 0, np.abs(data[c]) + 0.1, np.nan)
    data["ip_frac"] = rng.uniform(0, 1, size=n)
    return pd.DataFrame(data)


def make_model_workspace(root: Path, n_train: int = 12, n_val: int = 6, frames_per_shot: int = 20, k: int = 6,
                         seed: int = 0, band_cells: int = 2, plant: bool = True, exact: bool = False) -> dict:
    """Writes a complete synthetic workspace under ``root`` and returns its paths and facts.

    The standardized coefficients are a planted linear function of the scaled features
    (``plant``) plus small noise, so that the warm-start ridge recovers most of them. Targets
    are built for every frame; some frames fail the filter and some have no boundary (empty
    masks).
    """
    rng = np.random.default_rng(seed)
    root = Path(root)
    grid_r, grid_z = symmetric_grid()
    shape = (grid_z.size, grid_r.size)
    fconfig = fe.FeatureConfig.load(FEATURE_CONFIG)

    lists = {"train": [f"t{i:04d}" for i in range(n_train)], "validation": [f"v{i:04d}" for i in range(n_val)]}
    lists["working_train"] = lists["train"][: max(2, n_train // 2)]
    lists["working_validation"] = lists["validation"][: max(2, n_val // 2)]
    lists["local_test"] = [f"l{i:04d}" for i in range(2)]
    all_shots = sorted(lists["train"] + lists["validation"] + lists["local_test"])
    keys = pd.DataFrame({"shot": np.repeat(all_shots, frames_per_shot), "frame": np.tile(np.arange(frames_per_shot), len(all_shots))})
    n = len(keys)

    features = feature_values(fconfig, n, rng)
    table = pd.concat([keys, features], axis=1)
    feature_root = root / "features"
    (feature_root / f"v{fconfig.version}").mkdir(parents=True)
    pq.write_table(pa.Table.from_pandas(table, preserve_index=False), fe.features_path("diii_d_train", fconfig.version, feature_root))
    train_rows = table["shot"].isin(lists["train"]).to_numpy()
    scaler = RobustScaler.fit(table.loc[train_rows], fe.mirror_features(table.loc[train_rows], fconfig), fconfig)
    scaler_path = root / "scaler.json"
    scaler.save(scaler_path)

    basis = parity_basis(k, shape, rng)
    sigma = np.sqrt(basis["variances"])
    set_columns = fconfig.set_columns("set_a")
    x = scaler.transform(table, "DIII-D", set_columns)
    # Mirror-equivariant plant: with x_mirror = x A (A the signed permutation of the mirror) the
    # coefficients of the mirror are c p, so the weights satisfy A W = W diag(p).
    a = mirror_matrix(fconfig, set_columns)
    w0 = rng.normal(size=(len(set_columns), k)) * 0.3
    weights = 0.5 * (w0 + a @ w0 * basis["parity"]) if plant else np.zeros((len(set_columns), k))
    c_std = x @ weights + 0.05 * rng.normal(size=(n, k)) + (0.0 if plant else rng.normal(size=(n, k)))

    ip_ka = rng.uniform(400, 1500, size=n) * np.where(rng.uniform(size=n) < 0.2, -1.0, 1.0)
    phase = np.array(PHASES, dtype=object)[rng.integers(0, 3, size=n)]
    inside = ellipse_masks(n, grid_r, grid_z, rng, empty_every=17)
    passes = rng.uniform(size=n) > 0.1
    target = keys.assign(ip_ka=ip_ka, phase=phase, boundary_found=inside.any(axis=(1, 2)), passes_filter=passes)
    for j in range(k):
        target[f"c_{j:02d}"] = c_std[:, j] * sigma[j]
    target_root = root / "targets"
    folder = tb.target_dir(1, target_root)
    folder.mkdir(parents=True)
    for name, shots in lists.items():
        part = target.loc[target["shot"].isin(shots)].reset_index(drop=True)
        tmp = part.copy()
        tmp["phase"] = tmp["phase"].astype(str)
        pq.write_table(pa.Table.from_pandas(tmp, preserve_index=False), tb.table_path(name, 1, target_root))
    components = np.zeros((k, shape[0] * shape[1]))
    cb.CurrentBasis(mean=np.zeros(shape[0] * shape[1]), components=components, variances=basis["variances"], parity=basis["parity"],
                    total_variance=float(basis["variances"].sum()), n_frames=100, shape=shape).save(folder / tb.OBJECT_FILES["current_basis"])
    cb.MachineFluxBasis(machine="DIII-D", phi_mean=basis["phi_mean"], phi=basis["phi"], shape=shape, r0_m=1.67, a0_m=0.67) \
        .save(folder / tb.OBJECT_FILES["flux_basis_DIII-D"])

    loss_root = root / "loss_inputs"
    records = {}
    for name, shots in lists.items():
        rows = keys["shot"].isin(shots).to_numpy()
        writer = li.ListWriter(name, keys.loc[rows], shape, exact, 1, loss_root)
        for shot in shots:
            sel = (keys["shot"] == shot).to_numpy()
            result = {"shot": shot, "frames": keys.loc[sel, "frame"].to_numpy(), "inside": li.ct.pack_masks(inside[sel]),
                      "band": li.ct.pack_masks(li.band_mask(inside[sel], band_cells))}
            if exact:
                psi = (ip_ka[sel] * 1e3)[:, None] * (basis["phi_mean"] + (c_std[sel] * sigma) @ basis["phi"])
                result["psi_ref"] = psi.astype(np.float16)
            writer.add(result)
        records[name] = writer.close()
    (li.build_dir(1, loss_root) / "build.json").write_text(json.dumps(
        {"band_cells": band_cells, "grid_r": grid_r.tolist(), "grid_z": grid_z.tolist(), "lists": records,
         "reference": "exact" if exact else "basis"}))

    # Frame times (the rung-2 time steps): 20 ms apart with every seventh step doubled, per shot.
    table_dir = root / "frame_table"
    table_dir.mkdir()
    steps = np.where(keys["frame"].to_numpy() % 7 == 6, 40.0, 20.0)
    time_ms = 100.0 + keys.assign(step=steps).groupby("shot")["step"].cumsum().to_numpy() - steps
    pq.write_table(pa.Table.from_pandas(keys.assign(frame=keys["frame"].astype(np.int32), time_ms=time_ms), preserve_index=False),
                   table_dir / "diii_d_train.parquet")

    split = {name: list(shots) for name, shots in lists.items()}
    split["shots"] = {s: {"group": 0, "near_repeat_in_train": bool(i % 3 == 0)} for i, s in enumerate(all_shots)}
    split_path = root / "split.json"
    split_path.write_text(json.dumps(split))

    paths = DataPaths(feature_root=feature_root, feature_config=FEATURE_CONFIG, scaler=scaler_path, target_root=target_root,
                      target_version=1, loss_root=loss_root, split=split_path, table_dir=table_dir)
    return {"paths": paths, "lists": lists, "keys": keys, "table": table, "target": target, "inside": inside, "c_std": c_std,
            "sigma": sigma, "basis": basis, "grid_r": grid_r, "grid_z": grid_z, "scaler": scaler, "fconfig": fconfig,
            "planted": weights, "split": split, "time_ms": time_ms}
