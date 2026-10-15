"""Device safety of the training, prediction and evaluation paths on backends without float64
(Apple MPS). ``Float32Only`` reproduces MPS's refusal on any machine: a tensor that raises on a
float64 conversion until it has been moved with ``.cpu()``. Where MPS exists, the same paths also
run on it and agree with the CPU."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fusion_eq import features as fe, loader as ld, loss as ls, plasma_models as pm, prediction as pr, trainer as tr
from fusion_eq.model_config import ModelConfig
from tests.model_fixtures import make_model_workspace
from tests.test_trainer import SMALL, StubScorer

MPS = torch.backends.mps.is_available()


class Float32Only(torch.Tensor):
    """A tensor that refuses float64 (``double``, ``to``/``type`` with float64) until moved with
    ``.cpu()``, which returns a plain tensor, as an MPS tensor does."""

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func is torch.Tensor.cpu:
            with torch._C.DisableTorchFunctionSubclass():
                return func(*args, **kwargs).as_subclass(torch.Tensor)
        wants_64 = any(a is torch.float64 for a in [*args[1:], *kwargs.values()])
        if func is torch.Tensor.double or (func in (torch.Tensor.to, torch.Tensor.type) and wants_64):
            raise TypeError("Cannot convert a float32-only tensor to float64 dtype")
        if func is torch.Tensor.numpy:
            raise TypeError("Cannot convert a float32-only tensor to numpy; move it to the CPU first")
        return super().__torch_function__(func, types, args, kwargs)


class Wrapped(torch.nn.Module):
    """A model whose outputs are ``Float32Only``."""

    def __init__(self, inner: torch.nn.Module) -> None:
        super().__init__()
        self.inner = inner
        self.init_args = inner.init_args

    def forward(self, x):
        return self.inner(x).as_subclass(Float32Only)


@pytest.fixture(scope="module")
def ws(tmp_path_factory) -> dict:
    return make_model_workspace(tmp_path_factory.mktemp("device"), n_train=6, n_val=4, frames_per_shot=16)


@pytest.fixture(scope="module")
def config() -> ModelConfig:
    terms = {**ModelConfig.load().get("loss.terms"), "grad": {"enabled": True, "weight": 1.0}, "band": {"enabled": True, "weight": 1.0}}
    return ModelConfig.load().with_values({**SMALL, "loss.terms": terms, "train.max_epochs": 2, "loss.norm_frames": 60})


def loaders(ws, config, device="cpu"):
    frames = ld.load_frames("train", config, ws["paths"])
    basis, flux = ld.load_bases(ws["paths"])
    fconfig = fe.FeatureConfig.load(ws["paths"].feature_config_path(2))
    thomson = ld.thomson_masking(fconfig, frames.columns, ws["scaler"])
    weights = ld.sample_weights(frames, config, ls.ip_median(frames.ip_a))
    loader = ld.FrameLoader(frames, basis.parity, weights, config, thomson, "eval", device)
    op = ls.FluxOperator(flux.phi_mean, flux.phi, ld.component_sigma(basis), ws["grid_r"], ws["grid_z"]).to(device)
    return frames, loader, ls.LossFunction(config.get("loss.terms"), op, config.get("loss.reference"))


def test_the_fake_reproduces_the_mps_failure():
    t = torch.ones(3).as_subclass(Float32Only)
    with pytest.raises(TypeError):
        float(t.double())
    with pytest.raises(TypeError):
        t.to(torch.float64)
    assert ls.host_float(t * 2) == 6.0


def test_normalization_evaluation_and_prediction_without_float64(ws, config):
    frames, loader, loss_fn = loaders(ws, config)
    rows = np.arange(len(frames))
    plain = tr.compute_normalization(loader, loss_fn, rows)
    for name in ("x", "x_mirror", "c", "ip_a", "weight", "parity", "thomson_missing"):
        setattr(loader, name, getattr(loader, name).as_subclass(Float32Only))
    for name in ("phi", "phi_mean", "sigma"):
        setattr(loss_fn.operator, name, getattr(loss_fn.operator, name).as_subclass(Float32Only))
    batch = loader.batch(torch.arange(4), augment=False)
    assert isinstance(batch["c"], Float32Only)
    fake = tr.compute_normalization(loader, loss_fn, rows)
    assert fake == pytest.approx(plain, rel=1e-12)
    loss_fn.set_normalization(fake)
    torch.manual_seed(0)
    model = Wrapped(pm.LinearMLP(len(frames.columns), loader.c.shape[1], hidden=[8], dropout=0.0))
    out = tr.evaluate_loss(model, loader, loss_fn, ~frames.near_repeat)
    assert out["all"]["total"] > 0 and np.isfinite(out["subset"]["total"])
    loss, parts = loss_fn(model(batch["x"]), batch)
    assert set(parts) == {"flux", "grad", "band"} and all(isinstance(v, float) for v in parts.values())
    c = pm.predict(model, frames.x[:10].astype(np.float32))
    assert c.dtype == np.float64 and c.shape == (10, loader.c.shape[1])


@pytest.mark.skipif(not MPS, reason="MPS not available")
def test_training_and_prediction_run_on_mps(ws, config, tmp_path):
    frames, cpu_loader, cpu_loss = loaders(ws, config)
    _, mps_loader, mps_loss = loaders(ws, config, "mps")
    rows = np.arange(len(frames))
    cpu = tr.compute_normalization(cpu_loader, cpu_loss, rows)
    mps = tr.compute_normalization(mps_loader, mps_loss, rows)
    assert mps == pytest.approx(cpu, rel=1e-4)
    stub = StubScorer(ld.load_frames("validation", config, ws["paths"]), config.get("loader.clip_bound"))
    result = tr.train(config, ("train", "validation"), ws["paths"], run_root=tmp_path / "runs", checkpoint_root=tmp_path / "ckpt",
                      device="mps", scorer=stub, scorer_record={"kind": "stub"}, run_id="mps", log=lambda m: None)
    assert result.record["device"] == "mps" and len(result.metrics) == 2
    sample = tr.normalization_sample(len(frames), int(config.get("loss.norm_frames")), int(config.get("loss.norm_seed")))
    expected = tr.compute_normalization(cpu_loader, cpu_loss, sample)
    assert result.record["normalization"]["constants"] == pytest.approx(expected, rel=1e-4)
    model, meta = pm.load_checkpoint(result.checkpoint_dir / "best.pt", "mps")
    fconfig = fe.FeatureConfig.load(ws["paths"].feature_config_path(2))
    member = pr.Member(run_id="mps", path=result.checkpoint_dir / "best.pt", sha256="", model=model, meta=meta)
    on_mps = pr.CoefficientModel([member], fconfig, ws["scaler"], 10.0, True, "mps").predict(ws["table"].iloc[:40], "DIII-D")
    cpu_member = pr.Member(run_id="cpu", path=None, sha256="", model=pm.load_checkpoint(result.checkpoint_dir / "best.pt")[0], meta=meta)
    on_cpu = pr.CoefficientModel([cpu_member], fconfig, ws["scaler"], 10.0, True, "cpu").predict(ws["table"].iloc[:40], "DIII-D")
    np.testing.assert_allclose(on_mps, on_cpu, rtol=1e-4, atol=1e-5)


@pytest.mark.skipif(not MPS, reason="MPS not available")
@pytest.mark.parametrize("name", ["gru_seq", "tcn_seq"])
def test_rung_two_trains_on_mps(ws, tmp_path, name):
    config = ModelConfig.load().with_values({**SMALL, "model.name": name, "rung2.name": name, "rung2.hidden": 16, "rung2.layers": 1,
                                             "rung2.head_hidden": [8], "rung2.batch_shots": 3, "rung2.tcn_levels": 2,
                                             "train.max_epochs": 2, "loss.norm_frames": 60})
    result = tr.train(config, ("train", "validation"), ws["paths"], run_root=tmp_path / "runs", checkpoint_root=tmp_path / "ckpt",
                      device="mps", run_id=name, log=lambda m: None)
    assert result.record["device"] == "mps" and result.record["model"]["rung"] == 2
    assert all(np.isfinite(m["train_total"]) for m in result.metrics)
    model, _ = pm.load_checkpoint(result.checkpoint_dir / "best.pt", "mps")
    cpu_model, _ = pm.load_checkpoint(result.checkpoint_dir / "best.pt")
    x = np.clip(ws["table"].iloc[:18][list(result.record["features"]["columns"])].to_numpy(dtype=np.float32), -3, 3)
    dt = np.r_[0.0, np.ones(17)].astype(np.float32)
    np.testing.assert_allclose(pm.predict_sequence(model, x, dt, "mps"), pm.predict_sequence(cpu_model, x, dt), rtol=1e-4, atol=1e-5)
