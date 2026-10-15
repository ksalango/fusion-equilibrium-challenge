"""Tests of rung 2 on synthetic data: padded batching equals a per-shot loop and a causal model
ignores later frames (Z9), checkpoints reproduce outputs exactly, the warm-started linear skip
predicts what rung 1 predicts at step 0, the shot loader's batches, and short training runs of
``gru_seq`` and ``tcn_seq`` through the trainer."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fusion_eq import features as fe, loader as ld, loss as ls, plasma_models as pm, trainer as tr
from fusion_eq.model_config import ModelConfig
from tests.model_fixtures import make_model_workspace
from tests.test_trainer import SMALL, StubScorer

F, K = 7, 4
RUNG2 = {"hidden": 16, "layers": 2, "dropout": 0.0, "head_hidden": [12], "tcn_kernel": 3, "tcn_levels": 3}
VARIANTS = {"gru causal": ("gru_seq", {"causal": True, "bidirectional": False}),
            "gru bidirectional": ("gru_seq", {"causal": False, "bidirectional": True}),
            "tcn causal": ("tcn_seq", {"causal": True}), "tcn centred": ("tcn_seq", {"causal": False})}


def model_of(variant: str, seed: int = 0, dtype=torch.float64) -> torch.nn.Module:
    name, extra = VARIANTS[variant]
    torch.manual_seed(seed)
    model = pm.build_model({"name": name, "activation": "gelu", "bounded": False, "bound": 3.0}, F, K,
                           {"name": name, **RUNG2, **extra})
    with torch.no_grad():
        model.output.weight.normal_(0.0, 0.3)            # a nonzero head, so the encoder matters
    return model.to(dtype).eval()


def shots(lengths, seed: int = 0, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    return [(torch.randn(n, F, generator=g, dtype=dtype), torch.rand(n, generator=g, dtype=dtype)) for n in lengths]


def padded(items):
    lengths = torch.tensor([len(x) for x, _ in items])
    t = int(lengths.max())
    x = torch.zeros(len(items), t, F, dtype=items[0][0].dtype)
    dt = torch.zeros(len(items), t, dtype=items[0][0].dtype)
    for k, (xs, ds) in enumerate(items):
        x[k, :len(xs)], dt[k, :len(xs)] = xs, ds
    return x, dt, lengths


@pytest.mark.parametrize("variant", list(VARIANTS))
def test_padded_batching_equals_a_per_shot_loop(variant):
    model = model_of(variant)
    items = shots([9, 3, 14, 1, 6])
    x, dt, lengths = padded(items)
    # Garbage in the padding must not reach any valid frame.
    x = x + 1e3 * (torch.arange(x.shape[1])[None, :, None] >= lengths[:, None, None])
    with torch.no_grad():
        batch = model(x, dt, lengths)
        for k, (xs, ds) in enumerate(items):
            alone = model(xs[None], ds[None])[0]
            torch.testing.assert_close(batch[k, :len(xs)], alone, atol=1e-6, rtol=0)


@pytest.mark.parametrize("variant", list(VARIANTS))
def test_a_causal_output_ignores_later_frames(variant):
    model = model_of(variant)
    (xs, ds), = shots([12], seed=3)
    t = 5
    changed_x, changed_dt = xs.clone(), ds.clone()
    changed_x[t + 1:] += torch.randn_like(changed_x[t + 1:]) * 5
    changed_dt[t + 1:] += 2.0
    with torch.no_grad():
        a, b = model(xs[None], ds[None])[0], model(changed_x[None], changed_dt[None])[0]
    if model.uses_future:
        assert (a[:t + 1] - b[:t + 1]).abs().max() > 1e-6          # the check would catch a leak
    else:
        torch.testing.assert_close(a[:t + 1], b[:t + 1], atol=1e-6, rtol=0)
        assert (a[t + 1:] - b[t + 1:]).abs().max() > 1e-6


def test_variants_and_argument_checks():
    assert [model_of(v).uses_future for v in VARIANTS] == [False, True, False, True]
    with pytest.raises(ValueError):
        pm.GRUSeq(F, K, causal=True, bidirectional=True)
    with pytest.raises(ValueError):
        pm.GRUSeq(F, K, causal=False, bidirectional=False)
    with pytest.raises(ValueError):
        pm.TCNSeq(F, K, kernel=4, causal=False)
    with pytest.raises(ValueError):
        pm.build_model({"name": "gru_seq"}, F, K, {"name": "tcn_seq"})
    assert pm.is_sequence(model_of("gru causal")) and not pm.is_sequence(pm.LinearMLP(F, K))


@pytest.mark.parametrize("variant", ["gru causal", "tcn causal", "gru bidirectional"])
def test_checkpoint_reproduces_outputs_exactly(variant, tmp_path):
    model = model_of(variant, dtype=torch.float32)
    pm.save_checkpoint(tmp_path / "m.pt", model, {"sequence": True})
    loaded, meta = pm.load_checkpoint(tmp_path / "m.pt")
    assert meta == {"sequence": True} and loaded.init_args == model.init_args and type(loaded) is type(model)
    x = np.random.default_rng(0).normal(size=(20, F)).astype(np.float32)
    dt = np.random.default_rng(1).uniform(0, 2, size=20).astype(np.float32)
    assert np.array_equal(pm.predict_sequence(model, x, dt), pm.predict_sequence(loaded, x, dt))


def test_the_warm_started_skip_predicts_rung_one_at_step_zero():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(300, F))
    y = x @ rng.normal(size=(F, K)) + rng.normal(size=K)
    w, b = pm.warm_start_ridge(x, y, 1.0)
    rung1 = pm.LinearMLP(F, K, hidden=[16], dropout=0.0)
    pm.load_linear(rung1, w, b)
    xs = x[:40].astype(np.float32)
    dt = rng.uniform(0, 2, size=40).astype(np.float32)
    for variant in ("gru causal", "tcn causal"):
        name, extra = VARIANTS[variant]
        rung2 = pm.build_model({"name": name, "activation": "gelu", "bounded": False, "bound": 3.0}, F, K,
                               {"name": name, **RUNG2, **extra})
        pm.load_linear(rung2, w, b)
        np.testing.assert_allclose(pm.predict_sequence(rung2, xs, dt), pm.predict(rung1, xs), atol=1e-6)


# --------------------------------------------------------------------------- with data
@pytest.fixture(scope="module")
def ws(tmp_path_factory) -> dict:
    return make_model_workspace(tmp_path_factory.mktemp("rung2"), n_train=8, n_val=5, frames_per_shot=18)


def config_of(name: str, **extra) -> ModelConfig:
    return ModelConfig.load().with_values({**SMALL, "model.name": name, "rung2.name": name, "rung2.hidden": 16, "rung2.layers": 1,
                                           "rung2.head_hidden": [16], "rung2.dropout": 0.0, "rung2.batch_shots": 3,
                                           "rung2.tcn_levels": 2, "loss.norm_frames": 100, **extra})


def test_sequence_frames_and_the_shot_loader(ws):
    config = config_of("gru_seq")
    all_frames = ld.load_frames("train", config, ws["paths"], sequences=True)
    rung1 = ld.load_frames("train", config, ws["paths"])
    assert len(all_frames) == 8 * 18 and all_frames.kept_mask.sum() == len(rung1)
    kept = all_frames.kept_frames()
    assert np.array_equal(kept.x, rung1.x) and np.array_equal(kept.c, rung1.c) and list(kept.shot) == list(rung1.shot)
    dt = ld.dt_input(all_frames.time_ms, all_frames.shot, 20.0, 10.0)
    first = np.r_[True, all_frames.shot[1:] != all_frames.shot[:-1]]
    assert (dt[first] == 0).all() and set(np.round(dt[~first], 6)) == {1.0, 2.0}
    basis, _ = ld.load_bases(ws["paths"])
    fconfig = fe.FeatureConfig.load(ws["paths"].feature_config_path(2))
    thomson = ld.thomson_masking(fconfig, all_frames.columns, ws["scaler"])
    weights = ld.sample_weights(all_frames, config, ls.ip_median(rung1.ip_a))
    loader = ld.ShotLoader(all_frames, basis.parity, weights, config, thomson, "train")
    batch = loader.batch(torch.arange(3))
    assert batch["x"].shape == (3, 18, len(all_frames.columns)) and batch["lengths"].tolist() == [18, 18, 18]
    rows = batch["index"].numpy()
    assert np.array_equal(rows, np.flatnonzero(all_frames.kept_mask[:54]))
    for b in range(3):
        sel = batch["pos_b"].numpy() == b
        flip = bool(batch["flip"][b])
        expected = all_frames.c[rows[sel]] * (basis.parity if flip else 1.0)
        np.testing.assert_allclose(batch["c"].numpy()[sel], expected, rtol=1e-6)
        source = all_frames.x_mirror if flip else all_frames.x
        x = np.clip(source[b * 18:(b + 1) * 18], -10, 10)
        unmasked = ~batch["thomson_masked"][b].numpy()
        np.testing.assert_allclose(batch["x"][b].numpy()[unmasked], x[unmasked], rtol=1e-6)
    # The whole epoch visits every loss frame once.
    seen = np.concatenate([bt["index"].numpy() for bt in loader])
    assert np.array_equal(np.sort(seen), np.flatnonzero(all_frames.kept_mask))
    with pytest.raises(ValueError):
        ld.ShotLoader(rung1, basis.parity, weights[:len(rung1)], config, thomson)
    shuffled = all_frames.subset(np.r_[1, 0, np.arange(2, len(all_frames))])
    with pytest.raises(ValueError):
        ld.ShotLoader(shuffled, basis.parity, weights, config, thomson)


def run(ws, config, tmp_path, run_id, scorer=None):
    return tr.train(config, ("train", "validation"), ws["paths"], run_root=tmp_path / "runs", checkpoint_root=tmp_path / "ckpt",
                    device="cpu", scorer=scorer, scorer_record={"kind": "stub"}, run_id=run_id, log=lambda m: None)


class SequenceStub(StubScorer):
    """``StubScorer`` for sequence models: runs each validation shot whole, with its time steps."""

    def __init__(self, frames: ld.FrameSet, bound: float, config: ModelConfig) -> None:
        super().__init__(frames.kept_frames(), bound)
        self.all = frames
        self.dt = ld.dt_input(frames.time_ms, frames.shot, float(config.get("rung2.dt_scale_ms")), bound)
        self.bound = bound

    def __call__(self, predict_c, sequence=False):
        assert sequence
        out = np.concatenate([predict_c(np.clip(self.all.x[s:e], -self.bound, self.bound), self.dt[s:e])
                              for s, e in ld.shot_runs(self.all.shot)])
        return super().__call__(lambda x: out[self.all.kept_mask])


def test_rung_two_trains_with_the_rung_one_warm_start(ws, tmp_path):
    config1 = ModelConfig.load().with_values({**SMALL, "loss.norm_frames": 100, "train.max_epochs": 1})
    rung1 = run(ws, config1, tmp_path, "r1", scorer=StubScorer(ld.load_frames("validation", config1, ws["paths"]), 10.0))
    config = config_of("gru_seq", **{"train.max_epochs": 4, "train.lr": 3e-3})
    stub = SequenceStub(ld.load_frames("validation", config, ws["paths"], sequences=True), 10.0, config)
    result = run(ws, config, tmp_path, "gru", scorer=stub)
    rec = result.record
    # Same loss frames, same normalization, same step-0 predictions as rung 1.
    assert rec["n_frames"] == rung1.record["n_frames"]
    assert rec["normalization"] == rung1.record["normalization"]
    for group in ("all", "subset"):
        assert rec["warm_start"]["validation_loss"][group]["total"] == pytest.approx(
            rung1.record["warm_start"]["validation_loss"][group]["total"], rel=1e-5)
    for name in ("all", "no_near_repeat"):
        assert rec["warm_start"]["score"][name]["S"] == pytest.approx(rung1.record["warm_start"]["score"][name]["S"], abs=1e-6)
    losses = [m["train_total"] for m in result.metrics]
    assert losses[-1] < losses[0]
    assert rec["model"] == {**rec["model"], "name": "gru_seq", "rung": 2, "sequence": True, "uses_future": False}
    model, meta = pm.load_checkpoint(result.checkpoint_dir / "best.pt")
    assert pm.is_sequence(model) and meta["sequence"] is True and meta["uses_future"] is False
    assert np.isfinite(rec["best"]["score"]["all"]["S"])
    # A repeated run is identical.
    again = run(ws, config, tmp_path, "gru2", scorer=SequenceStub(stub.all, 10.0, config))
    assert [m["train_total"] for m in again.metrics] == losses
    for k, v in result.model.state_dict().items():
        assert torch.equal(v, again.model.state_dict()[k]), k


def test_tcn_and_a_bidirectional_gru_train_and_disclose_the_future(ws, tmp_path):
    tcn = run(ws, config_of("tcn_seq", **{"train.max_epochs": 2}), tmp_path, "tcn")
    assert tcn.record["model"]["name"] == "tcn_seq" and tcn.record["model"]["uses_future"] is False
    bi = run(ws, config_of("gru_seq", **{"train.max_epochs": 1, "rung2.causal": False, "rung2.bidirectional": True}), tmp_path, "bi")
    assert bi.record["model"]["uses_future"] is True
    terms = {**ModelConfig.load().get("loss.terms"), "grad": {"enabled": True, "weight": 1.0}, "band": {"enabled": True, "weight": 1.0}}
    g = run(ws, config_of("gru_seq", **{"train.max_epochs": 1, "loss.terms": terms}), tmp_path, "g")
    assert set(g.record["normalization"]["constants"]) == {"flux", "grad", "band"}
