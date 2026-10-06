"""Tests for the dataset layout helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fusion_eq import dataset
from tests.conftest import make_d3d_shot, make_mast_shot, write_shot


@pytest.fixture
def data_root(tmp_path) -> Path:
    root = tmp_path / "data"
    (root / "diii_d_train").mkdir(parents=True)
    (root / "mast_public_test").mkdir()
    for shot_id in ("b", "a", "c"):
        write_shot(root / "diii_d_train" / f"d3d_shot_{shot_id}.parquet", make_d3d_shot(n_frames=3))
    write_shot(root / "mast_public_test" / "mast_shot_m1.parquet", make_mast_shot(n_frames=3))
    return root


def test_list_shots_sorted_and_errors(data_root):
    files = dataset.list_shots(data_root, "diii_d_train")
    assert [p.stem for p in files] == ["d3d_shot_a", "d3d_shot_b", "d3d_shot_c"]
    with pytest.raises(FileNotFoundError):
        dataset.list_shots(data_root, "diii_d_public_test")
    with pytest.raises(ValueError):
        dataset.list_shots(data_root, "jet")


def test_select_shots_is_deterministic_and_sorted(data_root):
    files = dataset.list_shots(data_root, "diii_d_train")
    first = dataset.select_shots(files, 2, seed=3)
    assert first == dataset.select_shots(files, 2, seed=3)
    assert first == sorted(first) and len(first) == 2
    assert dataset.select_shots(files, None, 0) == files
    assert dataset.select_shots(files, 10, 0) == files


def test_shots_file_and_filter(data_root, tmp_path):
    shots_file = tmp_path / "shots.json"
    shots_file.write_text(json.dumps(["c", "a", "zzz"]))
    ids = dataset.read_shots_file(shots_file)
    files = dataset.filter_shots(dataset.list_shots(data_root, "diii_d_train"), ids)
    assert [p.stem for p in files] == ["d3d_shot_a", "d3d_shot_c"]
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"shot": 1}))
    with pytest.raises(ValueError):
        dataset.read_shots_file(bad)


def test_shot_path_uses_machine_prefix(data_root):
    assert dataset.shot_path(data_root, "diii_d_train", "a") == data_root / "diii_d_train" / "d3d_shot_a.parquet"
    assert dataset.shot_path(data_root, "mast_public_test", "m1").name == "mast_shot_m1.parquet"
    with pytest.raises(ValueError):
        dataset.shot_path(data_root, "mast_samples", "x")


def test_run_info_is_json_serialisable(tmp_path):
    info = dataset.run_info(["--n-shots", "2"], {"out": tmp_path, "n_shots": 2}, "2026-10-06T00:00:00+00:00", 1.5,
                            n_shots=2)
    text = json.dumps(info)
    assert '"args": "--n-shots 2"' in text
    assert info["arguments"]["out"] == str(tmp_path)
    assert info["n_shots"] == 2 and info["runtime_s"] == 1.5
