"""Tests of ``fusion_eq.loss_inputs`` and ``scripts/build_loss_inputs.py``: the band mask, the
writer and reader, and an end-to-end build on the synthetic workspace of the target-build tests
(masks equal to the target build's own masks, the exact reference consistent with the stored
remainder, ``local_test`` refused, deterministic rebuild)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fusion_eq import baseline as bl, current_target as ct, frame_table as ft, io, loss_inputs as li, starter_kit, \
    target_build as tb, targets
from fusion_eq.coil_flux import GreenTable, adopted_scales
from fusion_eq.machines import machine_config
from tests.test_build_features import REPO_ROOT, make_workspace
from tests.test_target_build import SMALL_CONFIG, build as build_targets


# --------------------------------------------------------------------------- band mask
def test_band_of_a_square_is_two_rings_each_side():
    m = np.zeros((30, 30), dtype=bool)
    m[10:20, 10:20] = True
    band = li.band_mask(m, 2)
    assert int(band.sum()) == (100 - 36) + (196 - 100)
    assert band[10, 10] and band[11, 11] and not band[12, 12] and band[8, 8] and not band[7, 7]
    assert not li.band_mask(np.zeros((5, 5), dtype=bool), 2).any()
    with pytest.raises(ValueError):
        li.band_mask(m, 0)


def test_band_commutes_with_the_mirror():
    rng = np.random.default_rng(1)
    m = np.zeros((4, 33, 33), dtype=bool)
    for k in range(4):
        r, z = rng.integers(8, 20, size=2)
        m[k, z - 5:z + 7, r - 4:r + 6] = True
    np.testing.assert_array_equal(li.band_mask(m[:, ::-1], 2), li.band_mask(m, 2)[:, ::-1])


# ---------------------------------------------------------------------- writer, reader
def test_writer_and_reader_round_trip(tmp_path):
    shape = (9, 7)
    keys = pd.DataFrame({"shot": ["a", "a", "a", "b", "b"], "frame": [0, 2, 4, 0, 1]})
    rng = np.random.default_rng(0)
    masks = rng.uniform(size=(5, *shape)) > 0.5
    masks[3] = False
    writer = li.ListWriter("train", keys, shape, True, 1, tmp_path)
    with pytest.raises(ValueError):
        writer.add({"shot": "b", "frames": np.array([0, 1]), "inside": ct.pack_masks(masks[3:]), "band": ct.pack_masks(masks[3:]),
                    "psi_ref": np.zeros((2, 63), dtype=np.float16)})
    for shot, sel in (("a", slice(0, 3)), ("b", slice(3, 5))):
        writer.add({"shot": shot, "frames": keys["frame"].to_numpy()[sel], "inside": ct.pack_masks(masks[sel]),
                    "band": ct.pack_masks(li.band_mask(masks[sel], 1)),
                    "psi_ref": np.arange(5 * 63, dtype=np.float16).reshape(5, 63)[sel]})
    record = writer.close()
    assert record["n_rows"] == 5 and record["n_empty_inside"] == 1
    (tmp_path / "v1" / "build.json").write_text(json.dumps({"band_cells": 1, "grid_r": list(range(7)), "grid_z": list(range(9))}))
    inputs = li.read_loss_inputs("train", 1, tmp_path, exact=True)
    assert inputs.grid_shape == shape
    np.testing.assert_array_equal(ct.unpack_masks(inputs.inside, shape), masks)
    np.testing.assert_array_equal(inputs.psi_ref[4], np.arange(4 * 63, 5 * 63, dtype=np.float16))
    np.testing.assert_array_equal(inputs.rows_of(["b", "a"], [1, 2]), [4, 1])
    with pytest.raises(KeyError):
        inputs.rows_of(["a"], [1])
    with pytest.raises(FileNotFoundError):
        li.read_loss_inputs("validation", 1, tmp_path)


def test_writer_refuses_an_incomplete_list(tmp_path):
    keys = pd.DataFrame({"shot": ["a", "a"], "frame": [0, 1]})
    writer = li.ListWriter("train", keys, (3, 3), False, 1, tmp_path)
    with pytest.raises(ValueError):
        writer.add({"shot": "a", "frames": np.array([0, 1, 2]), "inside": np.zeros((3, 2), np.uint8),
                    "band": np.zeros((3, 2), np.uint8)})
    with pytest.raises(ValueError):
        writer.close()
    assert not li.list_files("train", 1, tmp_path)["inside"].is_file()


# -------------------------------------------------------------------------- end to end
def run_build(ws: dict, out: Path, *extra: str) -> subprocess.CompletedProcess:
    cmd = [sys.executable, str(REPO_ROOT / "scripts" / "build_loss_inputs.py"), "--targets-root", str(ws["targets"]),
           "--data-root", str(ws["data"]), "--table-dir", str(ws["table_dir"]), "--green-dir", str(ws["green"]),
           "--response-cache", str(ws["tmp"] / "response"), "--out", str(out), "--workers", "1", *extra]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=600)


@pytest.fixture(scope="module")
def ws(tmp_path_factory) -> dict:
    if not starter_kit.is_available():
        pytest.skip("starter kit not available")
    root = tmp_path_factory.mktemp("loss_inputs")
    ws = make_workspace(root)
    ws["config"] = root / "target.yaml"
    ws["config"].write_text(SMALL_CONFIG)
    result = build_targets(ws, root / "targets")
    assert result.returncode == 0, result.stderr
    ws["targets"] = root / "targets"
    result = run_build(ws, root / "loss", "--lists", "train", "validation", "--reference", "exact")
    assert result.returncode == 0, result.stderr
    ws["loss"] = root / "loss"
    return ws


def test_masks_are_the_target_builds_masks(ws):
    green = GreenTable.load(ws["green"] / "green_d3d.npz")
    shape = (green.grid_z.size, green.grid_r.size)
    record = li.read_record(1, ws["loss"])
    assert record["band_cells"] == 2 and record["grid_r"] == green.grid_r.tolist() and record["exact_lists"] == ["train", "validation"]
    cfg = machine_config("DIII-D")
    for name in ("train", "validation"):
        target = pd.read_parquet(tb.table_path(name, 1, ws["targets"]))
        inputs = li.read_loss_inputs(name, 1, ws["loss"], exact=True)
        assert inputs.keys["shot"].tolist() == target["shot"].astype(str).tolist()
        assert inputs.keys["frame"].tolist() == target["frame"].tolist()
        inside = ct.unpack_masks(inputs.inside, shape)
        np.testing.assert_array_equal(inside.any(axis=(1, 2)), target["boundary_found"].to_numpy())
        np.testing.assert_array_equal(ct.unpack_masks(inputs.band, shape), li.band_mask(inside, 2))
        for shot, rows in target.groupby("shot", sort=False):
            psi = bl.load_truth(str(shot), ws["data"], "diii_d_train", 1)["psi"]
            for k, frame in zip(rows.index, rows["frame"]):
                expected = ct.plasma_mask(psi[frame].astype(np.float64), green.grid_r, green.grid_z, cfg)
                np.testing.assert_array_equal(inside[k], expected)


def test_exact_reference_leaves_the_stored_remainder(ws):
    """psi_phys - coil - psi_ref is the remainder whose RMS the target build stored."""
    green = GreenTable.load(ws["green"] / "green_d3d.npz")
    scales = adopted_scales("DIII-D", green.columns)
    cfg = machine_config("DIII-D")
    target = pd.read_parquet(tb.table_path("train", 1, ws["targets"]))
    inputs = li.read_loss_inputs("train", 1, ws["loss"], exact=True)
    columns = [*targets.current_columns_of(green, "DIII-D"), "ip__mean"]
    rows = bl.frames_of_shots(sorted(target["shot"].astype(str).unique()), columns, ws["table_dir"])
    rows["shot"] = rows["shot"].astype(str)
    checked = 0
    for k, (shot, frame) in enumerate(zip(target["shot"].astype(str), target["frame"])):
        if not target["boundary_found"].iloc[k]:
            continue
        row = rows[(rows["shot"] == shot) & (rows["frame"] == frame)]
        psi = bl.load_truth(shot, ws["data"], "diii_d_train", 1)["psi"][frame].astype(np.float64)
        phys = targets.to_physical(psi[None], row["ip__mean"].to_numpy(), cfg)[0]
        coil = targets.coil_flux_of_frames(green, row, scales, cfg)[0]
        remainder = phys - coil - inputs.psi_ref[k].astype(np.float64).reshape(phys.shape)
        rms = np.sqrt(np.mean(remainder ** 2)) / target["flux_range"].iloc[k]
        assert rms == pytest.approx(float(target["remainder_rms_over_range"].iloc[k]), rel=0.05, abs=2e-4)
        checked += 1
    assert checked > 0


def test_local_test_is_refused_and_rebuild_is_identical(ws, tmp_path):
    result = run_build(ws, tmp_path / "x", "--lists", "local_test")
    assert result.returncode != 0 and "final pre-submission check" in result.stderr
    result = run_build(ws, tmp_path / "again", "--lists", "train", "validation", "--reference", "exact")
    assert result.returncode == 0, result.stderr
    for name in ("train", "validation"):
        for kind, path in li.list_files(name, 1, ws["loss"]).items():
            again = li.list_files(name, 1, tmp_path / "again")[kind]
            if kind == "keys":
                pd.testing.assert_frame_equal(pd.read_parquet(path), pd.read_parquet(again))
            else:
                assert path.read_bytes() == again.read_bytes(), (name, kind)
