"""Tests for the split draw, its file and its checks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fusion_eq import split as sp


def shot_index(n_train=200, n_test=40):
    rng = np.random.default_rng(3)
    rows = []
    for i in range(n_train + n_test):
        split = "diii_d_train" if i < n_train else "diii_d_public_test"
        rows.append({"split": split, "machine": "DIII-D", "shot": f"s{i:04d}", "n_frames": int(rng.integers(30, 300)),
                     "mag_rate_hz": 20000 if rng.random() < 0.7 else 2000,
                     "ip_direction": -1 if rng.random() < 0.1 else 1,
                     "ts_core_any_valid": bool(rng.random() < 0.95), "error": None})
    return pd.DataFrame(rows)


SIZES = sp.SplitSizes(n_validation=40, n_local_test=20, n_working_train=30, n_working_validation=10)


def test_allocate_largest_remainder_and_caps():
    assert sp.allocate({"a": 70, "b": 30}, 10) == {"a": 7, "b": 3}
    out = sp.allocate({"a": 50, "b": 25, "c": 25}, 10)
    assert sum(out.values()) == 10 and out["a"] == 5
    assert sp.allocate({"a": 2, "b": 8}, 9) == {"a": 2, "b": 7}          # the larger remainder wins
    assert sp.allocate({"a": 2, "b": 98}, 50) == {"a": 1, "b": 49}
    assert sp.allocate({"a": 5}, 0) == {"a": 0}
    with pytest.raises(ValueError):
        sp.allocate({"a": 5}, 6)


def test_draw_split_is_reproducible_with_exact_counts_and_strata_shares():
    index = shot_index()
    train = index[index["split"] == "diii_d_train"]
    a = sp.draw_split(train, SIZES, seed=7)
    b = sp.draw_split(train, SIZES, seed=7)
    c = sp.draw_split(train, SIZES, seed=8)
    assert a == b and a != c
    assert sp.check_split(a, index, SIZES) == {"train": 140, "validation": 40, "local_test": 20,
                                               "working_train": 30, "working_validation": 10}
    assert all(v == sorted(v) for v in a.values())
    share = (train["mag_rate_hz"] == 20000).mean()
    for part in ("validation", "local_test", "working_train"):
        sub = train[train["shot"].isin(a[part])]
        assert abs((sub["mag_rate_hz"] == 20000).mean() - share) < 0.06


def test_draw_split_grouped_never_divides_a_group():
    index = shot_index()
    train = index[index["split"] == "diii_d_train"]
    ids = sorted(train["shot"])
    # Groups of three among the first 120 ids, singletons after: the counts stay reachable.
    groups = {s: (i // 3 if i < 120 else 40 + i) for i, s in enumerate(ids)}
    out = sp.draw_split(train, SIZES, seed=1, groups=groups)
    for part in sp.PARTS:
        members = set(out[part])
        for g in {groups[s] for s in members}:
            assert all(s in members for s in ids if groups[s] == g)
    counts = sp.check_split(out, index, SIZES, groups=groups)
    assert sum(counts[p] for p in sp.PARTS) == len(ids)
    assert any(groups[s] < 40 for s in out["validation"])            # whole triples do get drawn
    assert out == sp.draw_split(train, SIZES, seed=1, groups=groups)
    # All-triple groups (the last one a pair): the draw stops within one group of the quota,
    # never above it, and still never divides a group.
    triples = {s: i // 3 for i, s in enumerate(ids)}
    short = sp.draw_split(train, SIZES, seed=1, groups=triples)
    assert 37 < len(short["validation"]) <= 40
    for part in sp.PARTS:
        members = set(short[part])
        for g in {triples[s] for s in members}:
            assert all(s in members for s in ids if triples[s] == g)
    with pytest.raises(ValueError, match="requested"):
        sp.check_split(short, index, SIZES, groups=triples)
    with pytest.raises(ValueError, match="spans"):
        broken = {**out, "validation": out["validation"][1:], "train": out["train"] + out["validation"][:1]}
        sp.check_split(broken, index, groups=groups)


def test_check_split_raises_on_duplicate_unassigned_test_shot_and_working_outside():
    index = shot_index()
    train = index[index["split"] == "diii_d_train"]
    good = sp.draw_split(train, SIZES, seed=0)
    with pytest.raises(ValueError, match="both"):
        sp.check_split({**good, "validation": good["validation"] + good["train"][:1]}, index)
    with pytest.raises(ValueError, match="unassigned"):
        sp.check_split({**good, "train": good["train"][1:]}, index)
    with pytest.raises(ValueError, match="not training"):
        sp.check_split({**good, "train": good["train"] + ["s0205"]}, index)
    with pytest.raises(ValueError, match="outside"):
        sp.check_split({**good, "working_train": good["validation"][:5]}, index)
    with pytest.raises(ValueError, match="requested"):
        sp.check_split(good, index, sp.SplitSizes(n_validation=41, n_local_test=20, n_working_train=30, n_working_validation=10))


def test_split_file_roundtrip_balance_and_part_of(tmp_path):
    index = shot_index()
    train = index[index["split"] == "diii_d_train"]
    parts = sp.draw_split(train, SIZES, seed=0)
    groups = {s: i for i, s in enumerate(sorted(train["shot"]))}
    near = {s: False for s in train["shot"]}
    record = sp.split_record(parts, "random", 0.1, 0, SIZES, "abc123", groups, near)
    path = tmp_path / "split.json"
    sp.write_split(path, record)
    back = sp.read_split(path)
    assert back["validation"] == parts["validation"] and back["mode"] == "random" and back["repo_commit"] == "abc123"
    assert back["shots"][parts["train"][0]] == {"group": groups[parts["train"][0]], "near_repeat_in_train": False}
    assert len(back["shots"]) == len(train)
    balance = sp.split_balance(back, index)
    assert balance["list"].tolist() == ["train", "validation", "local_test", "working_train", "working_validation", "test"]
    assert balance.loc[balance["list"] == "test", "shots"].iloc[0] == 40
    assert balance["share_20khz"].between(0, 1).all() and balance["median_frames"].gt(0).all()
    empty = sp.split_balance({**back, "working_validation": []}, index)
    assert empty.loc[empty["list"] == "working_validation", "shots"].iloc[0] == 0
    assert np.isnan(empty.loc[empty["list"] == "working_validation", "share_20khz"].iloc[0])
    assert sp.part_of(back, [parts["validation"][0], "s9999"]) == ["validation", ""]
    path.write_text("{}")
    with pytest.raises(ValueError):
        sp.read_split(path)
