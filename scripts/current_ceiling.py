"""Ceiling test of the current-distribution target (analysis plan F8), DIII-D only.

If a model predicted the plasma's toroidal current density exactly, how well would the
flux map rebuilt from it by physics score, and how much is lost when the current density
is compressed with PCA? Every variant ends as a full stored-frame map, the coil flux under
the adopted model plus the plasma flux rebuilt by ``current_target.PlasmaResponse``, scored
with the organizers' local scorer on the validation shots of the split (every
``--frame-stride``-th frame) with the true q95 and betaN, so that the differences in S come
from the map alone:

- ``a_masked``: true map -> plasma current density -> scorer's plasma mask -> response.
- ``b_unmasked``: the same without the mask (every interior cell), what the mask costs.
- ``c_pca_<k>``: variant a with the masked current density divided by Ip, compressed with a
  PCA basis of ``k`` components fitted on the training shots' frames (native grid).
- ``d_plasma_basis_50``: the present target, the plasma flux through the 50-component basis
  of the training list (the comparison's ``ceiling_plasma_basis``), on the same frames.

The vacuum remainder (``current_target.vacuum_remainder``) is the plasma target minus the
flux of the masked current, on the whole grid, in the target frame. Three further variants
add it back:

- ``e_remainder_exact``: variant a plus the exact remainder; reproduces the true map (a check
  of the bookkeeping).
- ``f_remainder_pca_<k>``: the current at ``--f-current-components`` (50) plus the remainder
  through a PCA basis of ``k`` components (``--remainder-components``) fitted on the training
  frames.
- ``g_remainder_ridge_<k>``: as f at ``--ridge-components`` (10), with the remainder
  coefficients predicted by a ridge on the frame-table currents and their rates (``__mean``
  and ``__rate`` of every current signal, Ip-signed as in ``targets.signed_frames``), fitted
  on the training frames.

Four variants of the target phase follow (``docs/TARGET.md``, Part 5), each with the
remainder of variant f at ``--h-remainder-components`` (10) unless stated:

- ``h_symmetric_<k>``: the share (masked current per cell over Ip) through a basis with exact
  up-down parity, fitted on the training frames that pass the lighter label filter
  (``label_filter``) and have a boundary, together with their mirrors
  (``current_basis.CurrentBasis``), at ``--symmetric-components`` (50, 60, 80). The flux of the
  current comes from the DIII-D flux basis (``current_basis.flux_basis``).
- ``i_placed_mast_<base>``: h at ``--symmetric-base`` (50) with the reconstructed share placed
  onto the lattice of MAST's nodes and brought back (``lattice.placement``,
  ``lattice.back_placement``, MAST constants of ``machines.py``) before the flux is calculated.
- ``j_rescaled_<base>``: h at the base count with the reconstructed share rescaled to a total of
  one.
- ``k_ridge_<window>_<base>``: h at the base count with the remainder from a ridge, for each
  rate window the frame table carries: 10 ms (``__rate``) always, 50 ms (``__rate50``) where
  present. The frame table has no 50 ms rate for the toroidal field, which keeps its 10 ms
  rate in the 50 ms ridge, so both ridges have 42 inputs.

Writes ``ceiling.md`` and ``ceiling.json`` to ``--out`` (``reports/current_target/``): S, the
four terms and the seven shape terms per variant, the mask statistics (share of ``|J|``
dropped, masked current over Ip, boundary failures), the remainder statistics (RMS over the
frame's flux range, overall and by phase; variance held by the leading components; the
bookkeeping check; the ridge's coefficient R2) and the runtime. ``mask_statistics.parquet``
holds one row per validation frame. Usage::

    python scripts/current_ceiling.py                  # working lists of splits/split_v1.json
    python scripts/current_ceiling.py --full           # the full training and validation lists
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from fusion_eq import (baseline as bl, current_basis as cb, current_target as ct, frame_table as ft, grad_shafranov as gs,
                       label_filter as lf, lattice as lt, models, split as sp, starter_kit, targets)
from fusion_eq.coil_flux import GreenTable, adopted_scales
from fusion_eq.coil_scales import REFERENCE_PHASES, phase_of
from fusion_eq.dataset import REPO_ROOT, default_data_root, run_info
from fusion_eq.machines import machine_config
from fusion_eq.remainder_model import available_windows, ridge_columns, ridge_frames

DEFAULT_SPLIT = REPO_ROOT / "splits" / "split_v1.json"
DEFAULT_OUT = REPO_ROOT / "reports" / "current_target"
DEFAULT_GREEN_DIR = REPO_ROOT.parent / "data" / "processed" / "green"
SPLIT_NAME = "diii_d_train"
MACHINE = "DIII-D"
TERMS = ("S", "r2_psi", "r2_qb", "one_minus_dlcfs", "consistency")
SHAPE_TERMS = ("r2_R_axis", "r2_Z_axis", "r2_kappa", "r2_tri_top", "r2_tri_bot", "r2_volume", "r2_li")
STAT_KEYS = ("dropped_abs_share", "masked_total_over_ip")
# The variance share of the remainder basis is reported at these counts as well as the f variants'.
REPORTED_REMAINDER_COMPONENTS = (5, 10, 20)
# Label columns the lighter label filter reads besides the input columns.
FILTER_LABELS = ("efit_q95", "efit_beta_n")


def log(message: str) -> None:
    print(message, flush=True)


def rate_windows(table_dir: Path) -> dict[str, list[str]]:
    """The rate windows the frame table carries, each with its ridge inputs
    (``remainder_model.available_windows``)."""
    return available_windows(pq.ParquetFile(ft.table_path(SPLIT_NAME, table_dir)).schema_arrow.names)


def strided_rows(shot_ids, table_dir: Path, stride: int, extra=()) -> dict[str, pd.DataFrame]:
    """Frame-table rows with the current columns, their rates, ``extra`` columns, the phase
    columns and the columns of the lighter label filter, every ``stride``-th frame, per shot."""
    columns = list(dict.fromkeys([*ridge_columns(), *extra, *FILTER_LABELS, "ip_over_peak", "n_frames"]))
    rows = bl.frames_of_shots(shot_ids, columns, table_dir)
    rows["shot"] = rows["shot"].astype(str)
    return {s: g.sort_values("frame").iloc[::stride] for s, g in rows.groupby("shot", sort=False)}


def _shot_setup(shot_id: str, rows: pd.DataFrame, data_root: str, stride: int, green_path: str, response_cache: str):
    """Per-shot inputs shared by every pass: true maps, coil flux, Ip, plasma current, mask, and
    the plasma target split into the masked current's flux and the vacuum remainder."""
    cfg = machine_config(MACHINE)
    table = GreenTable.load(green_path)
    scales = adopted_scales(MACHINE, table.columns)
    response = ct.PlasmaResponse.load_or_build(MACHINE, table.grid_r, table.grid_z, response_cache)
    scorer = starter_kit.scoring()
    envelope = starter_kit.envelope_mask(cfg)
    psi = bl.load_truth(shot_id, data_root, SPLIT_NAME, stride)["psi"].astype(np.float64)
    if psi.shape[0] != len(rows):
        raise ValueError(f"{shot_id}: {len(rows)} rows for {psi.shape[0]} maps at stride {stride}")
    ip = rows["ip__mean"].to_numpy(dtype=np.float64)
    coil = targets.coil_flux_of_frames(table, rows, scales, cfg)
    j = ct.plasma_current_from_stored(psi, coil, table.grid_r, table.grid_z, ip, cfg)
    masks = np.stack([ct.plasma_mask(p, table.grid_r, table.grid_z, cfg, envelope, scorer) for p in psi])
    stats = [ct.mask_statistics(j[k], masks[k], table.grid_r, table.grid_z, ip[k]) for k in range(len(ip))]
    inside, remainder = ct.vacuum_remainder(psi, coil, ip, masks, response, cfg)
    return table, scales, response, psi, ip, coil, j, masks, stats, inside, remainder


def _per_ip(j_masked: np.ndarray, ip_ka: np.ndarray) -> np.ndarray:
    """Masked current density per ampere of Ip, zero where Ip is missing or zero."""
    ip_a = np.nan_to_num(ip_ka) * ct.A_PER_KA
    with np.errstate(divide="ignore", invalid="ignore"):
        y = np.where(ip_a[:, None, None] != 0, j_masked / ip_a[:, None, None], 0.0)
    return y


def _encode(remainder: np.ndarray, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    """Coefficients ``(frames, k)`` of remainder maps in a basis given by its mean and rows."""
    return (remainder.reshape(len(remainder), -1) - mean) @ components.T


def _train_shot(shot_id: str, rows: pd.DataFrame, data_root: str, stride: int, green_path: str, response_cache: str) -> dict:
    """First training pass: the masked current density per ampere and the vacuum remainder of
    every strided frame (float32)."""
    *_, ip, coil, j, masks, stats, inside, remainder = _shot_setup(shot_id, rows, data_root, stride, green_path, response_cache)
    return {"shot": shot_id, "y": _per_ip(j * masks, ip).astype(np.float32), "remainder": remainder.astype(np.float32),
            "boundary": np.array([s["boundary_found"] for s in stats], dtype=bool),
            "boundary_failures": sum(not s["boundary_found"] for s in stats), "n_frames": len(ip)}


def _train_remainder_coefficients(shot_id: str, rows: pd.DataFrame, data_root: str, stride: int, green_path: str,
                                  response_cache: str, remainder_mean: np.ndarray, remainder_components: np.ndarray) -> dict:
    """Second training pass: the remainder coefficients of every strided frame, the ridge targets."""
    *_, remainder = _shot_setup(shot_id, rows, data_root, stride, green_path, response_cache)
    return {"shot": shot_id, "coefficients": _encode(remainder, remainder_mean, remainder_components)}


def _val_shot(shot_id: str, rows: pd.DataFrame, data_root: str, stride: int, green_path: str, response_cache: str,
              basis_mean: np.ndarray, basis_components: np.ndarray, components: tuple, plasma_basis_path: str,
              remainder_mean: np.ndarray, remainder_components: np.ndarray, remainder_ks: tuple, f_current_k: int,
              ridge: dict, target: dict) -> dict:
    """Validation pass: every variant's rebuilt stored-frame maps, the mask statistics and the
    remainder statistics per frame. ``target`` carries what variants h to k need: the
    symmetric basis (``mean``, ``components``), the DIII-D flux basis (``phi_mean``, ``phi``),
    the component counts, the MAST placement and its way back, and the ridges per window."""
    table, scales, response, psi, ip, coil, j, masks, stats, inside, remainder = _shot_setup(
        shot_id, rows, data_root, stride, green_path, response_cache)
    cfg = machine_config(MACHINE)
    sign = targets.sign_of_ip(ip)[:, None, None]

    def rebuild(current):
        return targets.to_stored(coil + response.flux(current), ip, cfg).astype(np.float32)

    def from_target(target):
        return targets.stored_from_target(target, coil, ip, cfg)

    def current_pca(k):
        basis = bl.FluxBasis(basis_mean, basis_components[:k], np.zeros(k), 0.0, 0)
        return basis.decode(basis.encode(y)) * ip_a

    out = {"shot": shot_id, "n_frames": len(ip), "maps": {}}
    j_masked = j * masks
    out["maps"]["a_masked"] = rebuild(j_masked)
    out["maps"]["b_unmasked"] = rebuild(j)
    y = _per_ip(j_masked, ip)
    ip_a = np.nan_to_num(ip)[:, None, None] * ct.A_PER_KA
    for k in components:
        out["maps"][f"c_pca_{k}"] = rebuild(current_pca(k))
    plasma_basis = bl.FluxBasis.load(plasma_basis_path)
    t = targets.plasma_target(psi, coil, ip, cfg)
    decoded = plasma_basis.decode(plasma_basis.encode(t))
    out["maps"]["d_plasma_basis_50"] = targets.stored_from_target(decoded, coil, ip, cfg).astype(np.float32)

    # Vacuum remainder: the exact bookkeeping, then the compressed and the predicted remainder
    # on top of the compressed current.
    exact = from_target(inside + remainder)
    flux_range = psi.max(axis=(1, 2)) - psi.min(axis=(1, 2))
    out["e_max_error_over_range"] = float(np.max(np.abs(exact - psi).max(axis=(1, 2)) / flux_range))
    out["maps"]["e_remainder_exact"] = exact.astype(np.float32)
    inside_c = sign * response.flux(current_pca(f_current_k))
    for k in remainder_ks:
        coef = _encode(remainder, remainder_mean, remainder_components[:k])
        out["maps"][f"f_remainder_pca_{k}"] = from_target(inside_c + (coef @ remainder_components[:k] + remainder_mean).reshape(psi.shape)).astype(np.float32)
    k = ridge["k"]
    x, _ = ridge["scaler"].transform(ridge_frames(rows))
    coef_pred = x @ ridge["weights"] + ridge["intercept"]
    out["coefficients_true"] = _encode(remainder, remainder_mean, remainder_components[:k])
    out["coefficients_pred"] = coef_pred
    out["maps"][f"g_remainder_ridge_{k}"] = from_target(inside_c + (coef_pred @ remainder_components[:k] + remainder_mean).reshape(psi.shape)).astype(np.float32)

    # Target-phase variants: the share through the symmetric basis, its flux through the flux
    # basis (physical flux Ip (phi_mean + Phi c), target frame times sign(Ip)).
    share = y * float(np.prod(gs.grid_steps(table.grid_r, table.grid_z)))
    coef_share = (share.reshape(len(ip), -1) - target["mean"]) @ target["components"].T
    kr = target["h_remainder_k"]
    remainder_h = (_encode(remainder, remainder_mean, remainder_components[:kr]) @ remainder_components[:kr]
                   + remainder_mean).reshape(psi.shape)

    def inside_basis(k):
        physical = ip_a.reshape(-1, 1) * (target["phi_mean"] + coef_share[:, :k] @ target["phi"][:k])
        return sign * physical.reshape(psi.shape)

    for k in target["ks"]:
        out["maps"][f"h_symmetric_{k}"] = from_target(inside_basis(k) + remainder_h).astype(np.float32)
    base = target["base_k"]
    s_base = (coef_share[:, :base] @ target["components"][:base] + target["mean"]).reshape(share.shape)
    placed = target["back"].apply(target["place"].apply(s_base))
    totals = s_base.sum(axis=(1, 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        out["placement_retained"] = np.where(totals != 0, placed.sum(axis=(1, 2)) / totals, np.nan)
        rescale = np.where(totals != 0, 1.0 / totals, 1.0)[:, None, None]
    out["share_total_reconstructed"] = totals
    out["maps"][f"i_placed_mast_{base}"] = from_target(sign * ip_a * cb.cell_flux(response, placed) + remainder_h).astype(np.float32)
    inside_base = inside_basis(base)
    # The flux is linear in the share, so the rescaled share's flux is the base flux rescaled.
    out["maps"][f"j_rescaled_{base}"] = from_target(rescale * inside_base + remainder_h).astype(np.float32)
    out["ridge_pred"] = {}
    for window, model in target["ridges"].items():
        x_w, _ = model["scaler"].transform(ridge_frames(rows))
        pred = x_w @ model["weights"] + model["intercept"]
        out["ridge_pred"][window] = pred
        remainder_w = (pred @ remainder_components[:model["k"]] + remainder_mean).reshape(psi.shape)
        out["maps"][f"k_ridge_{window}_{base}"] = from_target(inside_base + remainder_w).astype(np.float32)

    rms = np.sqrt(np.mean(remainder * remainder, axis=(1, 2)))
    out["stats"] = [{**s, "phase": phase_of(float(over), int(frame), int(n)), "remainder_rms": float(rms[i]),
                     "flux_range": float(flux_range[i]), "remainder_share": float(rms[i] / flux_range[i])}
                    for i, (s, over, frame, n) in enumerate(zip(stats, rows["ip_over_peak"], rows["frame"], rows["n_frames"]))]
    return out


def _train_items(train_rows: dict[str, pd.DataFrame], args, *extra) -> object:
    return ((s, g, str(args.data_root), args.frame_stride, str(args.green_dir / "green_d3d.npz"), str(args.response_cache), *extra)
            for s, g in train_rows.items())


def fit_bases(train_rows: dict[str, pd.DataFrame], args, paths: dict[str, Path], sizes: dict[str, int],
              keep: dict[str, np.ndarray], cell_area: float, workers: int) -> tuple[dict, dict]:
    """Bases over the training frames, each cached at its path; one pass over the training shots
    fits whichever is missing.

    - ``current``: PCA of the masked current density per ampere (variants c, f, g).
    - ``remainder``: PCA of the vacuum remainder.
    - ``symmetric``: the share (masked current per cell over Ip) with exact parity, over the
      frames that pass the lighter label filter (``keep``) and have a boundary, with their
      mirrors (variants h to k).
    """
    loaders = {"current": bl.FluxBasis.load, "remainder": bl.FluxBasis.load, "symmetric": cb.CurrentBasis.load}
    if all(p.is_file() for p in paths.values()):
        return {name: loaders[name](p) for name, p in paths.items()}, {"from_cache": True}
    acc = {"current": bl.BasisAccumulator(), "remainder": bl.BasisAccumulator(), "symmetric": None}
    failures, n_frames, n_symmetric = 0, 0, 0
    for k, result in enumerate(bl._map_ordered(_train_shot, _train_items(train_rows, args), workers)):
        acc["current"].add(result["y"].astype(np.float64))
        acc["remainder"].add(result["remainder"].astype(np.float64))
        chosen = keep[result["shot"]] & result["boundary"]
        if chosen.any():
            shares = result["y"][chosen].astype(np.float64) * cell_area
            acc["symmetric"] = acc["symmetric"] or cb.CurrentBasisAccumulator(shares.shape[1:])
            acc["symmetric"].add(shares)
            n_symmetric += int(chosen.sum())
        failures += result["boundary_failures"]
        n_frames += result["n_frames"]
        if (k + 1) % 50 == 0:
            log(f"  training pass: {k + 1}/{len(train_rows)} shots, {n_frames} frames")
    out = {}
    for name, path in paths.items():
        if path.is_file():
            out[name] = loaders[name](path)
            continue
        if acc[name] is None:
            raise ValueError(f"no training frame for the {name} basis")
        basis = acc[name].finish(sizes[name])
        path.parent.mkdir(parents=True, exist_ok=True)
        basis.save(path)
        out[name] = basis
    return out, {"from_cache": False, "n_frames": n_frames, "boundary_failures": failures, "n_symmetric_frames": n_symmetric}


def fit_remainder_ridge(train_rows: dict[str, pd.DataFrame], args, remainder: bl.FluxBasis, k: int, alpha: float,
                        workers: int, windows: dict[str, list[str]]) -> tuple[dict, dict]:
    """Ridges from the Ip-signed currents and rates to the first ``k`` remainder coefficients,
    on the training frames, one per rate window (``windows``: window to input columns); the
    coefficients are computed once. Returns per window the model (``k``, ``scaler``,
    ``weights``, ``intercept``) and a summary (frames used, columns, training R2)."""
    items = _train_items(train_rows, args, remainder.mean, remainder.components[:k])
    coefficients = np.concatenate([r["coefficients"] for r in bl._map_ordered(_train_remainder_coefficients, items, workers)])
    frames = ridge_frames(pd.concat(list(train_rows.values()), ignore_index=True))
    fitted, summaries = {}, {}
    for window, columns in windows.items():
        scaler = bl.InputScaler.fit(frames, columns)
        x, complete = scaler.transform(frames)
        weights, intercept = bl.ridge_fit(x[complete], coefficients[complete], alpha)
        fitted[window] = {"k": k, "scaler": scaler, "weights": weights, "intercept": intercept}
        summaries[window] = {"k": k, "alpha": alpha, "window": window, "columns": columns, "n_frames": int(len(frames)),
                             "n_complete": int(complete.sum()),
                             "train_coefficient_r2": pooled_r2(coefficients[complete], x[complete] @ weights + intercept)}
    return fitted, summaries


def pooled_r2(true: np.ndarray, pred: np.ndarray) -> float:
    """R2 pooled over the coefficients: one minus the summed squared error over the summed
    variance, i.e. the share of the remainder's variance in the subspace that is predicted."""
    t = np.asarray(true, dtype=np.float64)
    p = np.asarray(pred, dtype=np.float64)
    total = float(((t - t.mean(axis=0)) ** 2).sum())
    return 1.0 - float(((t - p) ** 2).sum()) / total if total > 0 else float("nan")


def quantiles(values) -> dict:
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"n": 0}
    return {"n": int(v.size), "median": float(np.median(v)), "q05": float(np.quantile(v, 0.05)), "q95": float(np.quantile(v, 0.95)),
            "mean": float(v.mean())}


def _quantile_line(name: str, q: dict) -> str:
    if not q.get("n"):
        return f"- {name}: no frames"
    return f"- {name}: median {q['median']:.4f}, mean {q['mean']:.4f}, 5th to 95th percentile {q['q05']:.4f} to {q['q95']:.4f} ({q['n']} frames)"


def render(table: pd.DataFrame, info: dict) -> str:
    lines = [f"# Current-distribution target ceiling, split `{Path(info['arguments']['split']).name}`", "",
             f"{info['n_train_shots']} training shots ({info['basis']['n_frames'] if 'n_frames' in info['basis'] else 'cached'} frames "
             f"for the PCA), {info['n_val_shots']} validation shots ({info['n_val_frames']} frames) scored at frame stride "
             f"{info['arguments']['frame_stride']} with the true q95 and betaN; coil model {info['coil_model']}; "
             f"{info['workers']} workers; references {'from the cache' if info['references_from_cache'] else 'built'}.", "",
             "| Variant | S | R2 psi | R2 q95,betaN | 1 - D_LCFS | Consistency | " + " | ".join(t[3:] for t in SHAPE_TERMS) + " |",
             "| --- " * (6 + len(SHAPE_TERMS)) + "|"]
    for r in table.itertuples(index=False):
        row = [r.variant] + [f"{getattr(r, t):.4f}" for t in TERMS] + [f"{getattr(r, t):.4f}" for t in SHAPE_TERMS]
        lines.append("| " + " | ".join(row) + " |")
    m = info["mask_statistics"]
    lines += ["", "## Mask statistics (validation frames)", "",
              f"Boundary found on {m['n_frames'] - m['boundary_failures']} of {m['n_frames']} frames; frames without Ip {m['n_without_ip']}.", ""]
    for key in STAT_KEYS:
        lines.append(_quantile_line(f"`{key}`", m[key]))
    rem = info["remainder"]
    ridge = rem["ridge"]
    lines += ["", "## Vacuum remainder (plasma target minus the flux of the masked current)", "",
              "RMS of the remainder over the frame's flux range (max minus min of the true map), validation frames:", "",
              *[_quantile_line(group, rem["share"][group]) for group in ("all", *REFERENCE_PHASES)], "",
              "Variance of the training remainder held by the leading components: "
              + ", ".join(f"{k}: {v:.4f}" for k, v in rem["explained_cumulative"].items()) + ".", "",
              f"Bookkeeping check (variant e against the true map): largest absolute error over the frame's flux range "
              f"{rem['e_max_error_over_range']:.2e}.", "",
              f"Ridge (variant g): {ridge['k']} coefficients from {len(ridge['columns'])} inputs (currents and rates, Ip-signed), "
              f"alpha {ridge['alpha']}, {ridge['n_complete']} of {ridge['n_frames']} training frames complete; pooled coefficient R2 "
              f"training {ridge['train_coefficient_r2']:.4f}, validation {ridge['val_coefficient_r2']:.4f}."]
    t = info.get("target_phase")
    if t:
        sym = t["symmetric_basis"]
        lines += ["", "## Target-phase variants h to k (`docs/TARGET.md`, Part 5)", "",
                  f"Symmetric basis of the share: {sym['n_components']} components over {sym['n_frames']} training frames "
                  f"(passing the lighter label filter, with a boundary) and their mirrors; even and odd components among "
                  + ", ".join(f"the first {k}: {v['even']} and {v['odd']}" for k, v in sym["parity_counts"].items())
                  + "; variance held " + ", ".join(f"{k}: {v:.6f}" for k, v in sym["explained_cumulative"].items()) + ".", "",
                  f"Placement (variant i): MAST R0 {t['placement']['r0_m']} m, a0 {t['placement']['a0_m']} m; share of the "
                  "reconstructed total kept through placement and back:"]
        lines.append(_quantile_line("retained", t["placement"]["retained"]))
        lines.append(_quantile_line("reconstructed share total (variant j rescales it to one)", t["share_total_reconstructed"]))
        lines += ["", "Remainder ridges (variant k), pooled coefficient R2:", ""]
        for window, r in t["ridges"].items():
            lines.append(f"- {window}: training {r['train_coefficient_r2']:.4f}, validation {r['val_coefficient_r2']:.4f} "
                         f"({len(r['columns'])} inputs, {r['n_complete']} of {r['n_frames']} training frames complete)")
        lines += ["", "Against the references of the specification (S, consistency, volume):", "",
                  "| Variant | Reference | S - reference | Consistency - reference | Volume - reference |", "| --- | --- | --- | --- | --- |"]
        for r in t["comparisons"]:
            lines.append(f"| {r['variant']} | {r['reference']} | {r['d_S']:+.4f} | {r['d_consistency']:+.4f} | {r['d_volume']:+.4f} |")
        lines.append("")
        lines.append("Reading rule for h: the base count is kept if S is within 0.0003 of f and consistency within 0.001.")
    lines += ["", f"Runtime: {info['runtime_s']:.0f} s (" + ", ".join(f"{k} {v:.0f}" for k, v in info["timing"].items()) + ")", ""]
    return "\n".join(lines)


def target_phase_record(table: pd.DataFrame, symmetric: cb.CurrentBasis, place: lt.Placement, ridge_infos: dict,
                        window_pred: dict, coef_true: np.ndarray, retained: np.ndarray, share_totals: np.ndarray, args) -> dict:
    """What the report says about variants h to k: the symmetric basis, the placement, the
    ridges per window with their validation R2, and each variant against its reference."""
    cumulative = np.cumsum(symmetric.explained_ratio)
    scores = table.set_index("variant")
    f_name, g_name = f"f_remainder_pca_{args.h_remainder_components}", f"g_remainder_ridge_{args.ridge_components}"
    base = args.symmetric_base
    pairs = [(f"h_symmetric_{k}", f_name) for k in args.symmetric_components]
    pairs += [(f"i_placed_mast_{base}", f"h_symmetric_{base}"), (f"j_rescaled_{base}", f"h_symmetric_{base}")]
    pairs += [(f"k_ridge_{w}_{base}", g_name) for w in ridge_infos]
    comparisons = []
    for variant, reference in pairs:
        if variant in scores.index and reference in scores.index:
            a, b = scores.loc[variant], scores.loc[reference]
            comparisons.append({"variant": variant, "reference": reference, "d_S": float(a["S"] - b["S"]),
                                "d_consistency": float(a["consistency"] - b["consistency"]),
                                "d_volume": float(a["r2_volume"] - b["r2_volume"])})
    ridges = {}
    for w, info in ridge_infos.items():
        ridges[w] = {**info, "val_coefficient_r2": pooled_r2(coef_true, np.concatenate(window_pred[w]))}
    return {"symmetric_basis": {"n_components": symmetric.n_components, "n_frames": symmetric.n_frames,
                                "parity_counts": {k: {"even": int((symmetric.parity[:k] > 0).sum()), "odd": int((symmetric.parity[:k] < 0).sum())}
                                                  for k in args.symmetric_components},
                                "explained_cumulative": {k: float(cumulative[k - 1]) for k in args.symmetric_components}},
            "placement": {"r0_m": place.r0_m, "a0_m": place.a0_m, "retained": quantiles(retained)},
            "share_total_reconstructed": quantiles(share_totals), "ridges": ridges, "comparisons": comparisons}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--full", action="store_true", help="use the full training and validation lists instead of the working ones")
    parser.add_argument("--frame-stride", type=int, default=4)
    parser.add_argument("--components", type=int, nargs="+", default=[20, 50, 100])
    parser.add_argument("--f-current-components", type=int, default=50, help="current components under variants f and g")
    parser.add_argument("--remainder-components", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument("--ridge-components", type=int, default=10, help="remainder coefficients predicted in variants g and k")
    parser.add_argument("--symmetric-components", type=int, nargs="+", default=[50, 60, 80],
                        help="components of the symmetric share basis under variant h")
    parser.add_argument("--symmetric-base", type=int, default=50, help="component count of variants i, j and k")
    parser.add_argument("--h-remainder-components", type=int, default=10, help="remainder components under variants h, i and j")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--data-root", type=Path, default=default_data_root())
    parser.add_argument("--table-dir", type=Path, default=ft.DEFAULT_TABLE_ROOT)
    parser.add_argument("--green-dir", type=Path, default=DEFAULT_GREEN_DIR)
    parser.add_argument("--cache-dir", type=Path, default=bl.DEFAULT_CACHE_DIR, help="baseline cache: scorer references and the plasma basis")
    parser.add_argument("--response-cache", type=Path, default=ct.DEFAULT_CACHE_DIR, help="cache of the response and the current and remainder bases")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    started = time.time()
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    args.out.mkdir(parents=True, exist_ok=True)
    workers = bl.default_workers() if args.workers is None else args.workers
    timing = {}

    split = sp.read_split(args.split)
    train_ids = sorted(split["train" if args.full else "working_train"])
    val_ids = sorted(split["validation" if args.full else "working_validation"])
    table = GreenTable.load(args.green_dir / "green_d3d.npz")
    scales = adopted_scales(MACHINE, table.columns)
    response = ct.PlasmaResponse.load_or_build(MACHINE, table.grid_r, table.grid_z, args.response_cache)
    log(f"{len(train_ids)} training and {len(val_ids)} validation shots, stride {args.frame_stride}, {workers} workers; "
        f"response on grid {response.grid_hash[:12]}")

    t0 = time.time()
    if args.symmetric_base not in args.symmetric_components:
        raise SystemExit("--symmetric-base must be one of --symmetric-components")
    windows = rate_windows(args.table_dir)
    extra_columns = sorted({c for columns in windows.values() for c in columns})
    train_rows = strided_rows(train_ids, args.table_dir, args.frame_stride, extra_columns)
    val_rows = strided_rows(val_ids, args.table_dir, args.frame_stride, extra_columns)
    keep = {s: lf.lighter_label_filter(g) for s, g in train_rows.items()}
    timing["frames_s"] = time.time() - t0
    log(f"rate windows: {', '.join(windows)}; training frames passing the lighter label filter: "
        f"{sum(int(k.sum()) for k in keep.values())} of {sum(len(k) for k in keep.values())}")

    # The current and remainder bases on the training frames, and the plasma-flux basis of the present target.
    t0 = time.time()
    n_current = max([*args.components, args.f_current_components])
    n_remainder = max([*args.remainder_components, args.ridge_components, args.h_remainder_components,
                       *REPORTED_REMAINDER_COMPONENTS])
    n_symmetric = max(args.symmetric_components)
    ids_hash = bl.hash_of_ids(train_ids)
    basis_path = args.response_cache / f"current_basis_{ids_hash}_k{n_current}_s{args.frame_stride}.npz"
    remainder_path = args.response_cache / f"remainder_basis_{ids_hash}_k{n_remainder}_s{args.frame_stride}.npz"
    symmetric_path = args.response_cache / f"symmetric_share_basis_{ids_hash}_k{n_symmetric}_s{args.frame_stride}.npz"
    cell_area = float(np.prod(gs.grid_steps(table.grid_r, table.grid_z)))
    bases, basis_info = fit_bases(train_rows, args, {"current": basis_path, "remainder": remainder_path, "symmetric": symmetric_path},
                                  {"current": n_current, "remainder": n_remainder, "symmetric": n_symmetric}, keep, cell_area, workers)
    basis, remainder_basis, symmetric = bases["current"], bases["remainder"], bases["symmetric"]
    timing["bases_s"] = time.time() - t0
    log(f"current basis: {basis.n_components} components, remainder basis: {remainder_basis.n_components} components, over "
        f"{basis.n_frames} frames ({'from the cache' if basis_info['from_cache'] else f'{timing['bases_s']:.0f} s'})")
    t0 = time.time()
    params = bl.BaselineParams()
    ridges, ridge_infos = fit_remainder_ridge(train_rows, args, remainder_basis, args.ridge_components, params.alpha, workers,
                                              windows)
    ridge, ridge_info = ridges["10ms"], ridge_infos["10ms"]
    timing["remainder_ridge_s"] = time.time() - t0
    log(f"remainder ridge: {ridge_info['k']} coefficients, {ridge_info['n_complete']} training frames, "
        f"training R2 {ridge_info['train_coefficient_r2']:.4f} ({timing['remainder_ridge_s']:.0f} s)")
    t0 = time.time()
    cfg = machine_config(MACHINE)
    all_train_rows = bl.frames_of_shots(train_ids, [f"{n}__mean" for n in ft.current_signals(cfg)], args.table_dir)
    all_train_rows["shot"] = all_train_rows["shot"].astype(str)
    frames_by_shot = {s: g for s, g in all_train_rows.groupby("shot", sort=False)}
    tag = f"{models.PLASMA_TAG}_{scales.source.split(',')[0] or 'scales'}"
    plasma_basis, seconds = bl.cached_basis(train_ids, params, args.data_root, args.cache_dir, split=SPLIT_NAME,
                                            target=models.plasma_target_fn(table, frames_by_shot, scales, cfg), tag=tag)
    plasma_basis_path = args.cache_dir / f"basis_{bl.hash_of_ids(train_ids)}_k{params.n_components}_f{params.pca_frames}_s{params.seed}_{tag}.npz"
    timing["plasma_basis_s"] = time.time() - t0
    log(f"plasma-flux basis: {plasma_basis.n_components} components over {plasma_basis.n_frames} frames ({seconds:.0f} s fitting)")

    # Target-phase objects: the DIII-D flux basis of the symmetric share, and the MAST placement.
    t0 = time.time()
    reference = lt.reference_lattice(table.grid_r, table.grid_z)
    d3d_flux = cb.flux_basis(MACHINE, symmetric, response, lt.placement(MACHINE, reference, reference))
    mast_table = GreenTable.load(args.green_dir / "green_mast.npz")
    mast_native = lt.native_lattice("MAST", mast_table.grid_r, mast_table.grid_z)
    mast_place = lt.placement("MAST", reference, mast_native)
    target = {"mean": symmetric.mean, "components": symmetric.components, "phi_mean": d3d_flux.phi_mean, "phi": d3d_flux.phi,
              "ks": tuple(args.symmetric_components), "base_k": args.symmetric_base,
              "h_remainder_k": args.h_remainder_components, "place": mast_place,
              "back": lt.back_placement(mast_place, reference, mast_native), "ridges": ridges}
    timing["target_objects_s"] = time.time() - t0

    # Validation pass: every variant's maps per shot.
    t0 = time.time()
    items = ((s, g, str(args.data_root), args.frame_stride, str(args.green_dir / "green_d3d.npz"), str(args.response_cache),
              basis.mean, basis.components, tuple(args.components), str(plasma_basis_path),
              remainder_basis.mean, remainder_basis.components, tuple(args.remainder_components), args.f_current_components, ridge,
              target)
             for s, g in val_rows.items())
    maps: dict[str, dict] = {}
    stats_rows = []
    coef_true, coef_pred, e_errors = [], [], []
    window_pred: dict[str, list] = {w: [] for w in ridges}
    retained, share_totals = [], []
    n_val_frames = 0
    for k, result in enumerate(bl._map_ordered(_val_shot, items, workers)):
        for variant, m in result["maps"].items():
            maps.setdefault(variant, {})[result["shot"]] = m
        stats_rows += [{"shot": result["shot"], "frame": i, **s} for i, s in enumerate(result["stats"])]
        coef_true.append(result["coefficients_true"])
        coef_pred.append(result["coefficients_pred"])
        for w, pred in result["ridge_pred"].items():
            window_pred[w].append(pred)
        retained.append(result["placement_retained"])
        share_totals.append(result["share_total_reconstructed"])
        e_errors.append(result["e_max_error_over_range"])
        n_val_frames += result["n_frames"]
        if (k + 1) % 25 == 0:
            log(f"  validation pass: {k + 1}/{len(val_rows)} shots")
    timing["validation_pass_s"] = time.time() - t0
    stats = pd.DataFrame(stats_rows)
    ip_missing = int(sum((~np.isfinite(g["ip__mean"].to_numpy(dtype=np.float64))).sum() for g in val_rows.values()))
    mask_info = {"n_frames": int(len(stats)), "boundary_failures": int((~stats["boundary_found"]).sum()), "n_without_ip": ip_missing,
                 **{key: quantiles(stats[key]) for key in STAT_KEYS}}
    log(f"mask: boundary on {mask_info['n_frames'] - mask_info['boundary_failures']} of {mask_info['n_frames']} frames; "
        f"dropped |J| share median {mask_info['dropped_abs_share'].get('median', float('nan')):.4f}; "
        f"masked current over Ip median {mask_info['masked_total_over_ip'].get('median', float('nan')):.4f}")
    cumulative = np.cumsum(remainder_basis.explained_ratio)
    ridge_info["val_coefficient_r2"] = pooled_r2(np.concatenate(coef_true), np.concatenate(coef_pred))
    remainder_info = {"share": {"all": quantiles(stats["remainder_share"]),
                                **{p: quantiles(stats.loc[stats["phase"] == p, "remainder_share"]) for p in REFERENCE_PHASES}},
                      "explained_cumulative": {k: float(cumulative[k - 1])
                                               for k in sorted({*REPORTED_REMAINDER_COMPONENTS, *args.remainder_components, args.ridge_components})},
                      "e_max_error_over_range": float(max(e_errors)), "ridge": ridge_info}
    log(f"remainder: RMS over flux range median {remainder_info['share']['all'].get('median', float('nan')):.4f}; "
        f"20 components hold {remainder_info['explained_cumulative'][20]:.4f} of its variance; "
        f"variant e error {remainder_info['e_max_error_over_range']:.2e}; ridge validation R2 {ridge_info['val_coefficient_r2']:.4f}")

    # Scoring with the true scalars.
    t0 = time.time()
    references = bl.build_references(val_ids, args.data_root, SPLIT_NAME, args.frame_stride, workers=workers,
                                     cache_dir=args.cache_dir / "references")
    timing["references_s"] = time.time() - t0
    truth = {s: bl.load_truth(s, args.data_root, SPLIT_NAME, args.frame_stride) for s in val_ids}
    rows = []
    for variant in maps:
        t0 = time.time()
        predictions = {s: bl.Prediction(maps[variant][s], truth[s]["q95"], truth[s]["betaN"]) for s in val_ids}
        result = bl.score_local(predictions, val_ids, args.data_root, SPLIT_NAME, args.frame_stride, references, workers)
        terms = bl.score_terms(result.finalize())
        timing[f"score_{variant}_s"] = time.time() - t0
        rows.append({"variant": variant, **terms})
        log(f"{variant}: S {terms['S']:.4f}, R2 psi {terms['r2_psi']:.4f}, 1 - D_LCFS {terms['one_minus_dlcfs']:.4f}, "
            f"consistency {terms['consistency']:.4f} ({timing[f'score_{variant}_s']:.0f} s)")
    table_out = pd.DataFrame(rows)
    table_out.to_parquet(args.out / "ceiling.parquet", index=False)
    target_info = target_phase_record(table_out, symmetric, mast_place, ridge_infos, window_pred, np.concatenate(coef_true),
                                      np.concatenate(retained), np.concatenate(share_totals), args)
    stats.to_parquet(args.out / "mask_statistics.parquet", index=False)
    info = run_info(argv, vars(args), timestamp, time.time() - started, workers=workers, n_train_shots=len(train_ids),
                    n_val_shots=len(val_ids), n_val_frames=n_val_frames, coil_model=scales.source, basis=basis_info,
                    current_basis_file=str(basis_path), remainder_basis_file=str(remainder_path),
                    plasma_basis_file=str(plasma_basis_path), response_grid_hash=response.grid_hash,
                    references_from_cache=references.from_cache, mask_statistics=mask_info, remainder=remainder_info,
                    symmetric_basis_file=str(symmetric_path), target_phase=target_info,
                    timing=timing, scores={r["variant"]: {k: v for k, v in r.items() if k != "variant"} for r in rows})
    (args.out / "ceiling.json").write_text(json.dumps(info, indent=2, default=float) + "\n")
    report = render(table_out, info)
    (args.out / "ceiling.md").write_text(report)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
