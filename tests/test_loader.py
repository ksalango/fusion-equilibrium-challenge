"""Tests of ``fusion_eq.loader`` on a synthetic workspace: the join of features, targets and
loss inputs, the double mirror (Z2, loader part), Thomson masking against the flag, clipping,
tail weights, determinism, and the reversible padding by shot."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fusion_eq import current_target as ct, features as fe, loader as ld, loss as ls
from fusion_eq.model_config import ModelConfig
from tests.model_fixtures import make_model_workspace, mirror_matrix


@pytest.fixture(scope="module")
def ws(tmp_path_factory) -> dict:
    return make_model_workspace(tmp_path_factory.mktemp("loader"))


@pytest.fixture(scope="module")
def config() -> ModelConfig:
    return ModelConfig.load().with_values({"train.batch_frames": 64})


@pytest.fixture(scope="module")
def train(ws, config) -> ld.FrameSet:
    return ld.load_frames("train", config, ws["paths"])


def thomson_of(ws, frames) -> tuple[np.ndarray, np.ndarray]:
    return ld.thomson_masking(ws["fconfig"], frames.columns, ws["scaler"])


def test_frames_are_filter_passing_with_a_boundary(ws, train):
    t = ws["target"]
    sel = t["shot"].isin(ws["lists"]["train"]) & t["passes_filter"] & t["boundary_found"]
    assert len(train) == int(sel.sum()) and train.lists == ("train",)
    np.testing.assert_array_equal(train.shot.astype(str), t.loc[sel, "shot"].to_numpy())
    np.testing.assert_array_equal(train.frame, t.loc[sel, "frame"].to_numpy())
    rows = np.flatnonzero(sel.to_numpy())
    np.testing.assert_allclose(train.c, ws["c_std"][rows], rtol=2e-6, atol=1e-6)
    np.testing.assert_allclose(train.ip_a, t.loc[sel, "ip_ka"].to_numpy() * 1e3)
    np.testing.assert_array_equal(ct.unpack_masks(train.inside, train.grid_shape), ws["inside"][rows])
    columns = ws["fconfig"].set_columns("set_a")
    assert train.columns == columns and train.x.shape == (len(train), 29)
    np.testing.assert_allclose(train.x, ws["scaler"].transform(ws["table"].iloc[rows], "DIII-D", columns), rtol=1e-6, atol=1e-6)
    expected_near = [ws["split"]["shots"][s]["near_repeat_in_train"] for s in train.shot]
    np.testing.assert_array_equal(train.near_repeat, expected_near)
    assert set(train.extra) == {"shaping", "ts_beta_p", "q_cyl"}


def test_local_test_is_refused(ws, config):
    with pytest.raises(PermissionError):
        ld.load_frames("local_test", config, ws["paths"])
    assert len(ld.load_frames("local_test", config, ws["paths"], allow_local_test=True)) > 0


def test_band_width_and_feature_version_must_match(ws, config):
    with pytest.raises(ValueError, match="band_cells"):
        ld.load_frames("train", config.with_values({"loss.band_cells": 3}), ws["paths"])
    with pytest.raises(ValueError, match="version"):
        ld.load_frames("train", config.with_values({"features.version": 1}), ws["paths"])


def test_drop_groups_and_set_b(ws, config):
    frames = ld.load_frames("validation", config.with_values({"features.drop_groups": ["shot_position"]}), ws["paths"])
    assert frames.columns == [c for c in ws["fconfig"].set_columns("set_a") if c not in ("ip_frac", "ip_rate_rel", "resistive_age")]
    frames = ld.load_frames("validation", config.with_values({"features.set": "set_b"}), ws["paths"])
    assert frames.columns == ws["fconfig"].set_columns("set_b")
    with pytest.raises(ValueError):
        ld.model_columns(ws["fconfig"], "set_a", ["no_such_group"])


def test_double_mirror_is_the_identity(ws, train):
    t = ws["table"].iloc[:50]
    twice = fe.mirror_features(fe.mirror_features(t, ws["fconfig"]), ws["fconfig"])
    columns = train.columns
    np.testing.assert_array_equal(ws["scaler"].transform(twice, "DIII-D", columns), ws["scaler"].transform(t, "DIII-D", columns))
    p = ws["basis"]["parity"]
    np.testing.assert_array_equal(train.c * p * p, train.c)
    m = ct.unpack_masks(train.inside, train.grid_shape)
    np.testing.assert_array_equal(m[:, ::-1][:, ::-1], m)
    # The mirrored scaled features are the sign change and permutation of the scaled features.
    specs = ws["fconfig"].column_specs()
    odd = [j for j, c in enumerate(columns) if specs[c].mirror == "odd"]
    even = [j for j, c in enumerate(columns) if specs[c].mirror == "even"]
    np.testing.assert_array_equal(train.x_mirror[:, even], train.x[:, even])
    np.testing.assert_allclose(train.x_mirror[:, odd], -train.x[:, odd])
    perm = [j for j, c in enumerate(columns) if specs[c].mirror == "permute"]
    partner = ws["fconfig"].layout.mirror_index()
    np.testing.assert_array_equal(train.x_mirror[:, perm], train.x[:, perm][:, partner])
    np.testing.assert_allclose(train.x_mirror, train.x @ mirror_matrix(ws["fconfig"], columns), atol=1e-6)


def test_mirrored_batches_flip_features_targets_and_masks(ws, config, train):
    parity = ws["basis"]["parity"]
    no_mask = config.with_values({"loader.thomson_mask_rate": 0.0, "loader.clip_bound": 1e6})
    loader = ld.FrameLoader(train, parity, np.ones(len(train)), no_mask, thomson_of(ws, train), "train")
    seen_flip = seen_plain = False
    for batch in loader:
        idx = batch["index"].numpy()
        flip = batch["flip"].numpy()
        x = np.where(flip[:, None], train.x_mirror[idx], train.x[idx])
        np.testing.assert_array_equal(batch["x"].numpy(), x)
        np.testing.assert_allclose(batch["c"].numpy(), np.where(flip[:, None], train.c[idx] * parity, train.c[idx]))
        m = ct.unpack_masks(train.inside[idx], train.grid_shape)
        np.testing.assert_array_equal(batch["inside"].numpy(), np.where(flip[:, None, None], m[:, ::-1], m))
        seen_flip |= flip.any()
        seen_plain |= (~flip).any()
    assert seen_flip and seen_plain


def test_thomson_masking_matches_the_flag(ws, config, train):
    idx, missing = thomson_of(ws, train)
    names = [train.columns[j] for j in idx]
    assert names == ["ts_beta_p", "ts_pe_peaking", "ts_valid"] and np.array_equal(missing, [0.0, 0.0, 0.0])
    # Frames whose Thomson was already invalid carry exactly the masked values.
    invalid = train.x[:, train.columns.index("ts_valid")] == 0
    assert invalid.any()
    np.testing.assert_array_equal(train.x[invalid][:, idx], np.tile(missing, (int(invalid.sum()), 1)))
    every = config.with_values({"loader.thomson_mask_rate": 1.0, "loader.mirror": False, "loader.clip_bound": 1e6})
    batch = next(iter(ld.FrameLoader(train, ws["basis"]["parity"], np.ones(len(train)), every, (idx, missing), "train")))
    rows = batch["index"].numpy()
    x = batch["x"].numpy()
    assert batch["thomson_masked"].all()
    np.testing.assert_array_equal(x[:, idx], np.tile(missing, (len(rows), 1)))
    other = [j for j in range(len(train.columns)) if j not in idx]
    np.testing.assert_array_equal(x[:, other], train.x[rows][:, other])
    never = every.with_values({"loader.thomson_mask_rate": 0.0})
    batch = next(iter(ld.FrameLoader(train, ws["basis"]["parity"], np.ones(len(train)), never, (idx, missing), "train")))
    np.testing.assert_array_equal(batch["x"].numpy(), train.x[batch["index"].numpy()])


def test_clipping_at_the_bound(ws, config, train):
    clip = config.with_values({"loader.clip_bound": 0.5})
    loader = ld.FrameLoader(train, ws["basis"]["parity"], np.ones(len(train)), clip, thomson_of(ws, train), "eval")
    x = torch.cat([b["x"] for b in loader]).numpy()
    assert np.abs(x).max() == pytest.approx(0.5) and (np.abs(train.x) > 0.5).any()
    inside = np.abs(train.x) <= 0.5
    np.testing.assert_array_equal(x[inside], train.x[inside])
    np.testing.assert_allclose(ld.clip_rates(train.x, 0.5), (np.abs(train.x) > 0.5).mean(axis=0))


def test_eval_mode_keeps_order_and_is_unaugmented(ws, config, train):
    loader = ld.FrameLoader(train, ws["basis"]["parity"], np.ones(len(train)), config, thomson_of(ws, train), "eval")
    batches = list(loader)
    assert len(batches) == len(loader)
    np.testing.assert_array_equal(torch.cat([b["index"] for b in batches]).numpy(), np.arange(len(train)))
    assert not any(b["flip"].any() or b["thomson_masked"].any() for b in batches)


def test_same_seed_same_batches(ws, config, train):
    def first(seed):
        loader = ld.FrameLoader(train, ws["basis"]["parity"], np.ones(len(train)), config, thomson_of(ws, train), "train", seed=seed)
        return [(b["index"].numpy(), b["flip"].numpy(), b["x"].numpy()) for b in loader]
    a, b, c = first(3), first(3), first(4)
    assert all(np.array_equal(u, v) for p, q in zip(a, b) for u, v in zip(p, q))
    assert not np.array_equal(a[0][0], c[0][0])


def test_tail_weights_touch_only_the_tail(ws, config, train):
    on = config.with_values({"loader.tail_weights": True, "loader.tail_weight": 3.0})
    rule = ld.tail_rule(train, on)
    assert set(rule) == {"shaping", "ts_beta_p"}
    assert rule["shaping"] == pytest.approx(np.quantile(train.extra["shaping"], 0.9))
    ip_med = ls.ip_median(train.ip_a)
    base = ld.sample_weights(train, on, ip_med)
    tailed = ld.sample_weights(train, on, ip_med, rule)
    flags = ld.tail_flags(train, rule)
    assert 0 < flags.sum() < len(train)
    np.testing.assert_array_equal(tailed[~flags], base[~flags])
    np.testing.assert_allclose(tailed[flags], 3.0 * base[flags])
    val = ld.load_frames("validation", on, ws["paths"])
    loader, frames = ld.make_loader(["validation"], "set_a", on, "eval", ws["paths"], ip_med=ip_med, rule=rule)
    np.testing.assert_allclose(loader.weight.numpy(), ld.sample_weights(val, on, ip_med, rule))
    off, _ = ld.make_loader(["validation"], "set_a", config, "eval", ws["paths"], ip_med=ip_med, rule=rule)
    np.testing.assert_allclose(off.weight.numpy(), ld.sample_weights(val, config, ip_med))


def test_make_loader_concatenates_lists(ws, config):
    loader, frames = ld.make_loader(["working_train", "working_validation"], "set_a", config, "train", ws["paths"])
    a = ld.load_frames("working_train", config, ws["paths"])
    b = ld.load_frames("working_validation", config, ws["paths"])
    assert len(frames) == len(a) + len(b) == loader.n and frames.lists == ("working_train", "working_validation")


def test_exact_reference_is_loaded_and_flipped(tmp_path, config):
    ws = make_model_workspace(tmp_path, n_train=3, n_val=2, frames_per_shot=6, exact=True)
    exact = config.with_values({"loss.reference": "exact", "loader.thomson_mask_rate": 0.0})
    frames = ld.load_frames("train", exact, ws["paths"])
    assert frames.psi_ref is not None and frames.psi_ref.dtype == np.float16
    loader = ld.FrameLoader(frames, ws["basis"]["parity"], np.ones(len(frames)), exact, thomson_of(ws, frames), "train")
    batch = next(iter(loader))
    idx, flip = batch["index"].numpy(), batch["flip"].numpy()
    ref = frames.psi_ref[idx].astype(np.float32).reshape(len(idx), *frames.grid_shape)
    np.testing.assert_array_equal(batch["psi_ref"].numpy(), np.where(flip[:, None, None], ref[:, ::-1], ref))


def test_unpack_masks_matches_numpy():
    rng = np.random.default_rng(0)
    m = rng.uniform(size=(7, 65, 65)) > 0.5
    packed = ct.pack_masks(m)
    np.testing.assert_array_equal(ld.unpack_masks(torch.as_tensor(packed), (65, 65)).numpy(), m)


def test_padding_by_shot_is_reversible():
    shot = np.array(["a", "a", "a", "b", "c", "c"])
    values = np.arange(12, dtype=np.float64).reshape(6, 2)
    padded, lengths = ld.pad_by_shot(values, shot, pad_value=-1.0)
    assert padded.shape == (3, 3, 2) and lengths.tolist() == [3, 1, 2]
    assert (padded[1, 1:] == -1.0).all()
    np.testing.assert_array_equal(ld.unpad(padded, lengths), values)
    np.testing.assert_array_equal(ld.time_steps([0.0, 20.0, 50.0, 7.0, 1.0, 3.5], shot), [0.0, 20.0, 30.0, 0.0, 0.0, 2.5])
    with pytest.raises(ValueError):
        ld.shot_runs(np.array(["a", "b", "a"]))
