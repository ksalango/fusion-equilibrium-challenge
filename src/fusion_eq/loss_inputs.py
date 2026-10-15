"""Loss inputs (P3, ``docs/PIPELINE.md`` Part 1): per labelled frame of a target table, the
packed inside mask, the packed band mask and, under ``loss.reference: exact``, the exact
reference ``Ip G s`` in float16.

The inside mask is the target build's own mask: the nodes inside the scorer's boundary of the
true map (``current_target.plasma_mask`` on the stored map in float64, the call
``current_basis.current_share`` makes in the target build). The stored ``efit_lcfs`` columns
are not used (F23). The band mask holds the nodes within ``band_cells`` grid cells of that
boundary on both sides, in the Chebyshev sense: the dilation of the mask by a square of side
``2 band_cells + 1`` minus its erosion by the same square. The square is symmetric, so the band
of a mirrored mask is the mirrored band.

Files of one list under ``<root>/v<target version>/`` (rows in the target table's order):
``<list>.keys.parquet`` (``shot``, ``frame``), ``<list>.inside.npy`` and ``<list>.band.npy``
(``(rows, ceil(nZ nR / 8))`` uint8, ``current_target.pack_masks``), ``<list>.psi_ref.npy``
(``(rows, nZ nR)`` float16 in Wb/rad, physical frame; exact reference only), and ``build.json``
for the whole build (grid, band width, lists, counts, provenance).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import ndimage

from fusion_eq import current_basis as cb, current_target as ct, dataset, target_build as tb, targets
from fusion_eq.machines import machine_config

DEFAULT_ROOT = dataset.DEFAULT_PROCESSED_ROOT / "loss_inputs"
MACHINE = "DIII-D"
A_PER_KA = 1000.0


# ------------------------------------------------------------------------------ masks
def band_mask(inside, cells: int) -> np.ndarray:
    """Nodes within ``cells`` grid cells (Chebyshev distance) of the boundary of each mask,
    on both sides: ``dilate(M) & ~erode(M)`` with a square of side ``2 cells + 1``.

    Args:
        inside: ``(nZ, nR)`` or ``(frames, nZ, nR)`` bool masks.
        cells: Band half-width in grid cells, at least 1.

    Returns:
        Bool band masks of the same shape; empty for an empty mask.
    """
    if int(cells) < 1:
        raise ValueError("band width must be at least one cell")
    m = np.asarray(inside, dtype=bool)
    single = m.ndim == 2
    m = m[None] if single else m
    square = np.ones((2 * int(cells) + 1,) * 2, dtype=bool)
    out = np.zeros_like(m)
    for k in range(m.shape[0]):
        if m[k].any():
            out[k] = ndimage.binary_dilation(m[k], structure=square) & ~ndimage.binary_erosion(m[k], structure=square)
    return out[0] if single else out


def packed_width(shape: tuple[int, int]) -> int:
    """Bytes per packed mask of a grid ``(nZ, nR)``."""
    return (int(shape[0]) * int(shape[1]) + 7) // 8


def shot_loss_inputs(shot_id: str, frames: np.ndarray, data_root: str, split: str, green_dir: str, response_cache: str,
                     band_cells: int, rows: pd.DataFrame | None = None) -> dict:
    """Loss inputs of the selected frames of one labelled shot. Runs in a worker process.

    Args:
        shot_id: Shot id.
        frames: Frame indices (into the shot's EFIT timeslices) of the target table's rows.
        data_root, split: Raw data root and the labelled split.
        green_dir, response_cache: Green's tables and plasma-response cache of the target build.
        band_cells: Band half-width in grid cells.
        rows: Frame-table rows of the same frames (coil currents and ``ip__mean``), for the
            exact reference; None builds the masks only.

    Returns:
        ``{"shot", "frames", "inside", "band"}`` (packed) and ``"psi_ref"`` (float16) when
        ``rows`` is given.
    """
    table, scales, response, envelope, scorer = tb._resources(green_dir, response_cache)
    cfg = machine_config(MACHINE)
    frames = np.asarray(frames, dtype=np.int64)
    psi_all = tb.read_truth(shot_id, data_root, split, 1)["psi"]
    if frames.size and (frames.min() < 0 or frames.max() >= psi_all.shape[0]):
        raise ValueError(f"{shot_id}: frame indices outside the shot's {psi_all.shape[0]} maps")
    psi = psi_all[frames].astype(np.float64)
    out = {"shot": shot_id, "frames": frames}
    if rows is None:
        inside = np.stack([ct.plasma_mask(p, table.grid_r, table.grid_z, cfg, envelope, scorer) for p in psi]) \
            if len(psi) else np.zeros((0, table.grid_z.size, table.grid_r.size), dtype=bool)
    else:
        if not np.array_equal(rows["frame"].to_numpy(dtype=np.int64), frames):
            raise ValueError(f"{shot_id}: frame rows and target frames differ")
        ip = rows["ip__mean"].to_numpy(dtype=np.float64)
        coil = targets.coil_flux_of_frames(table, rows, scales, cfg)
        share = cb.current_share(psi, coil, ip, table.grid_r, table.grid_z, cfg, envelope=envelope, scorer=scorer)
        inside = share.mask
        flux = cb.cell_flux(response, share.share) * (np.nan_to_num(ip) * A_PER_KA)[:, None, None]
        out["psi_ref"] = flux.reshape(len(frames), -1).astype(np.float16)
    out["inside"] = ct.pack_masks(inside)
    out["band"] = ct.pack_masks(band_mask(inside, band_cells))
    return out


# ----------------------------------------------------------------------------- storage
def build_dir(target_version: int, root: Path | str = DEFAULT_ROOT) -> Path:
    return Path(root) / f"v{int(target_version)}"


def list_files(list_name: str, target_version: int, root: Path | str = DEFAULT_ROOT) -> dict[str, Path]:
    folder = build_dir(target_version, root)
    return {"keys": folder / f"{list_name}.keys.parquet", "inside": folder / f"{list_name}.inside.npy",
            "band": folder / f"{list_name}.band.npy", "psi_ref": folder / f"{list_name}.psi_ref.npy"}


class ListWriter:
    """Writes one list's arrays shot by shot into memory-mapped temporary files, then renames
    them into place (``close``).

    Args:
        list_name: List of the split file.
        keys: Target-table keys (``shot``, ``frame``) in row order.
        grid_shape: ``(nZ, nR)``.
        exact: Whether the exact reference is written.
        target_version, root: Where the files go.
    """

    def __init__(self, list_name: str, keys: pd.DataFrame, grid_shape: tuple[int, int], exact: bool, target_version: int,
                 root: Path | str = DEFAULT_ROOT) -> None:
        self.files = list_files(list_name, target_version, root)
        self.files["keys"].parent.mkdir(parents=True, exist_ok=True)
        self.keys = keys[["shot", "frame"]].reset_index(drop=True)
        self.n = len(self.keys)
        width = packed_width(grid_shape)
        self.tmp = {k: p.with_suffix(".tmp.npy") for k, p in self.files.items() if k != "keys"}
        self.arrays = {"inside": np.lib.format.open_memmap(self.tmp["inside"], mode="w+", dtype=np.uint8, shape=(self.n, width)),
                       "band": np.lib.format.open_memmap(self.tmp["band"], mode="w+", dtype=np.uint8, shape=(self.n, width))}
        if exact:
            self.arrays["psi_ref"] = np.lib.format.open_memmap(self.tmp["psi_ref"], mode="w+", dtype=np.float16,
                                                               shape=(self.n, grid_shape[0] * grid_shape[1]))
        self.position = 0
        self.empty_inside = 0

    def add(self, result: dict) -> None:
        """Appends one shot's result; its keys must be the next rows of the table.

        Raises:
            ValueError: when the shot's keys are not the next rows.
        """
        n = len(result["frames"])
        expect = self.keys.iloc[self.position:self.position + n]
        if not ((expect["shot"].astype(str) == result["shot"]).all()
                and np.array_equal(expect["frame"].to_numpy(dtype=np.int64), np.asarray(result["frames"], dtype=np.int64))):
            raise ValueError(f"{result['shot']}: result rows are not the next rows of the target table")
        for name, array in self.arrays.items():
            array[self.position:self.position + n] = result[name]
        self.empty_inside += int((np.asarray(result["inside"]) == 0).all(axis=1).sum())
        self.position += n

    def close(self) -> dict:
        """Flushes, renames into place and returns the list's record.

        Raises:
            ValueError: when fewer rows were written than the table holds.
        """
        if self.position != self.n:
            raise ValueError(f"{self.position} of {self.n} rows written")
        for name, array in self.arrays.items():
            array.flush()
            del array
        self.arrays = {}
        for name, tmp in self.tmp.items():
            if tmp.is_file():
                tmp.replace(self.files[name])
        keys_tmp = self.files["keys"].with_suffix(".tmp.parquet")
        pq.write_table(pa.Table.from_pandas(self.keys.assign(shot=self.keys["shot"].astype(str),
                                                             frame=self.keys["frame"].astype(np.int32)), preserve_index=False),
                       keys_tmp)
        keys_tmp.replace(self.files["keys"])
        return {"n_rows": int(self.n), "n_empty_inside": int(self.empty_inside),
                "files": {k: p.name for k, p in self.files.items() if p.is_file()}}


@dataclass
class LossInputs:
    """One list's loss inputs, memory-mapped.

    Attributes:
        keys: ``shot``, ``frame`` per row.
        inside, band: Packed masks ``(rows, bytes)``.
        psi_ref: Exact reference ``(rows, nZ nR)`` float16, or None.
        record: The build record.
    """

    keys: pd.DataFrame
    inside: np.ndarray
    band: np.ndarray
    psi_ref: np.ndarray | None
    record: dict = field(default_factory=dict)

    @property
    def grid_shape(self) -> tuple[int, int]:
        return (len(self.record["grid_z"]), len(self.record["grid_r"]))

    def rows_of(self, shots, frames) -> np.ndarray:
        """Row index of each ``(shot, frame)`` key.

        Raises:
            KeyError: when a key is not in the loss inputs.
        """
        index = pd.MultiIndex.from_arrays([self.keys["shot"].astype(str).to_numpy(), self.keys["frame"].to_numpy(dtype=np.int64)])
        wanted = pd.MultiIndex.from_arrays([np.asarray(shots).astype(str), np.asarray(frames, dtype=np.int64)])
        rows = index.get_indexer(wanted)
        if (rows < 0).any():
            k = int(np.flatnonzero(rows < 0)[0])
            raise KeyError(f"{int((rows < 0).sum())} frames have no loss inputs, first {wanted[k]}; rerun build_loss_inputs.py")
        return rows


def read_record(target_version: int, root: Path | str = DEFAULT_ROOT) -> dict:
    path = build_dir(target_version, root) / "build.json"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; run python scripts/build_loss_inputs.py first")
    return json.loads(path.read_text())


def read_loss_inputs(list_name: str, target_version: int = 1, root: Path | str = DEFAULT_ROOT,
                     exact: bool = False) -> LossInputs:
    """Reads one list's loss inputs (arrays memory-mapped).

    Raises:
        FileNotFoundError: when the list or the exact reference was not built.
    """
    record = read_record(target_version, root)
    files = list_files(list_name, target_version, root)
    for name in ("keys", "inside", "band") + (("psi_ref",) if exact else ()):
        if not files[name].is_file():
            hint = " with --reference exact" if name == "psi_ref" else ""
            raise FileNotFoundError(f"{files[name]} not found; run python scripts/build_loss_inputs.py{hint} for {list_name}")
    keys = pq.read_table(files["keys"]).to_pandas()
    return LossInputs(keys=keys, inside=np.load(files["inside"], mmap_mode="r"), band=np.load(files["band"], mmap_mode="r"),
                      psi_ref=np.load(files["psi_ref"], mmap_mode="r") if exact else None, record=record)
