"""Training of the plasma model (``docs/PIPELINE.md``, Part 4).

One run: load the training and validation frames; fix ``Ip_med`` and, with tail weights on, the
tail thresholds on the training frames; compute each loss term's normalization constant on a
fixed sample of ``loss.norm_frames`` training frames (unaugmented, seed ``loss.norm_seed``);
build the model and load the warm-start ridge into its linear path; record the warm start's
validation loss and, with a scorer, its validation score; train with AdamW, a linear warmup
then cosine schedule per step, float32 and gradient-norm clipping; stop early on the
validation loss of the frames whose shots have no near-repeat in the training list (patience
``train.patience``, at most ``train.max_epochs``); score the best checkpoint.

Rung 2 (``model.name`` a sequence model) trains on whole shots (``loader.ShotLoader``); the loss,
its normalization, the warm start and the selection read the same frames as rung 1, so the two
rungs are compared on equal terms.

``train.seed`` seeds the initialization and dropout (``torch.manual_seed``) and the loader's
generator (shuffling, mirroring, Thomson masking). Checkpoints (best and last) go under the
checkpoint root, outside git (with ``warm.pt``, the model at the warm start); the run record ``run.json`` and the per-epoch ``metrics.csv`` go
under ``reports/runs/<run_id>/``.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import torch

from fusion_eq import dataset, features as fe, frame_table as ft, loader as ld, loss as ls, loss_inputs as li, \
    plasma_models as pm, split as sp, starter_kit
from fusion_eq.feature_scaler import RobustScaler
from fusion_eq.machines import machine_config
from fusion_eq.model_config import ModelConfig

DEFAULT_RUN_ROOT = dataset.REPO_ROOT / "reports" / "runs"
DEFAULT_CHECKPOINT_ROOT = dataset.DEFAULT_PROCESSED_ROOT.parent / "runs"
LIST_PAIRS = {"full": ("train", "validation"), "working": ("working_train", "working_validation")}
Scorer = Callable[[Callable[[np.ndarray], np.ndarray]], Mapping[str, Mapping]]


def log_print(message: str) -> None:
    print(message, flush=True)


# ------------------------------------------------------------------------------ setup
def select_device(name: str | None = None) -> torch.device:
    """``name`` when given, otherwise CUDA, then MPS, then the CPU."""
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def lr_factor(step: int, total_steps: int, warmup_steps: int) -> float:
    """Linear warmup from ``1 / warmup_steps`` to 1, then a cosine decay to 0 at ``total_steps``."""
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    span = max(1, total_steps - warmup_steps)
    progress = min(1.0, (step - warmup_steps) / span)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


class EarlyStopping:
    """Stops after ``patience`` epochs without a strictly lower value."""

    def __init__(self, patience: int) -> None:
        self.patience = int(patience)
        self.best = math.inf
        self.best_epoch = -1
        self.bad_epochs = 0

    def update(self, value: float, epoch: int) -> bool:
        """Records an epoch's value; True when it is the best so far."""
        if value < self.best:
            self.best, self.best_epoch, self.bad_epochs = float(value), int(epoch), 0
            return True
        self.bad_epochs += 1
        return False

    @property
    def stop(self) -> bool:
        return self.bad_epochs >= self.patience


def sha256_or_none(path: Path) -> str | None:
    path = Path(path)
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def environment_freeze() -> list[str]:
    """Installed distributions as sorted ``name==version`` lines."""
    names = {(d.metadata["Name"] or "").lower(): d.version for d in importlib.metadata.distributions() if d.metadata["Name"]}
    return [f"{n}=={v}" for n, v in sorted(names.items())]


def data_hashes(config: ModelConfig, paths: ld.DataPaths, table_dir: Path | None) -> dict:
    """SHA-256 of the build records of the inputs (None where a record is absent)."""
    version = int(config.get("features.version"))
    return {"frame_table": sha256_or_none(Path(table_dir or ft.DEFAULT_TABLE_ROOT) / "build.json"),
            f"features_v{version}": sha256_or_none(fe.features_dir(version, paths.feature_root) / "build.json"),
            f"features_v{version}_scaler": sha256_or_none(paths.scaler_path(version)),
            f"targets_v{paths.target_version}": sha256_or_none(paths.target_dir() / "build.json"),
            "loss_inputs": sha256_or_none(li.build_dir(paths.target_version, paths.loss_root) / "build.json")}


# --------------------------------------------------------------------------- evaluation
Loader = ld.FrameLoader | ld.ShotLoader


def forward(model: torch.nn.Module, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Standardized coefficients of the batch's loss frames: the model on the frames (rung 1),
    or on the padded shots, read at the flattened loss positions (rung 2)."""
    if pm.is_sequence(model):
        return model(batch["x"], batch["dt"], batch["lengths"])[batch["pos_b"], batch["pos_t"]]
    return model(batch["x"])


def eval_batches(loader: Loader):
    """Unaugmented batches over every frame (rung 1) or shot (rung 2) in table order."""
    if isinstance(loader, ld.ShotLoader):
        for start in range(0, len(loader.runs), loader.batch_shots):
            yield loader.batch(torch.arange(start, min(len(loader.runs), start + loader.batch_shots)), augment=False)
        return
    for start in range(0, loader.n, loader.batch_frames):
        yield loader.batch(torch.arange(start, min(loader.n, start + loader.batch_frames)), augment=False)


def evaluate_loss(model: torch.nn.Module, loader: Loader, loss_fn: ls.LossFunction,
                  subset: np.ndarray | None = None) -> dict[str, dict[str, float]]:
    """Normalized term means and the total over every loss frame of ``loader`` (``all``) and over
    the frames flagged by ``subset`` (``subset``, one flag per row of the loader's frames),
    unaugmented, in eval mode."""
    model.eval()
    groups = {"all": None} if subset is None else {"all": None, "subset": torch.as_tensor(np.asarray(subset, dtype=bool))}
    sums = {g: {t: 0.0 for t in loss_fn.enabled} for g in groups}
    counts = {g: 0 for g in groups}
    with torch.no_grad():
        for batch in eval_batches(loader):
            index = batch["index"].cpu()
            if len(index) == 0:
                continue
            per = loss_fn.per_frame(forward(model, batch), batch)
            w = batch["weight"]
            for g, mask in groups.items():
                keep = torch.ones(len(index), dtype=torch.bool) if mask is None else mask[index]
                keep = keep.to(w.device)
                counts[g] += int(keep.sum())
                for t, v in per.items():
                    sums[g][t] += ls.host_float((w * v)[keep])
    out = {}
    for g in groups:
        n = counts[g]
        parts = {t: (sums[g][t] / (n * loss_fn.normalization[t]) if n else float("nan")) for t in loss_fn.enabled}
        parts["total"] = sum(loss_fn.weights[t] * parts[t] for t in loss_fn.enabled) if n else float("nan")
        parts["n_frames"] = n
        out[g] = parts
    return out


def normalization_sample(n_frames: int, size: int, seed: int) -> np.ndarray:
    """Sorted indices of the fixed normalization sample (all frames when fewer)."""
    if n_frames <= size:
        return np.arange(n_frames)
    return np.sort(np.random.default_rng(seed).choice(n_frames, size=size, replace=False))


def compute_normalization(loader: ld.FrameLoader, loss_fn: ls.LossFunction, rows: np.ndarray) -> dict[str, float]:
    """Each term's weighted mean for ``c_hat = 0`` over the given rows, unaugmented."""
    sums = {t: 0.0 for t in loss_fn.enabled}
    for start in range(0, len(rows), loader.batch_frames):
        batch = loader.batch(torch.as_tensor(rows[start:start + loader.batch_frames]), augment=False)
        with torch.no_grad():
            for t, v in loss_fn.weighted_sums(torch.zeros_like(batch["c"]), batch).items():
                sums[t] += ls.host_float(v)
    return {t: s / len(rows) for t, s in sums.items()}


# -------------------------------------------------------------------------------- run
@dataclass
class TrainResult:
    """Outcome of one run."""

    run_id: str
    run_dir: Path
    checkpoint_dir: Path
    record: dict
    metrics: list[dict] = field(default_factory=list)
    model: torch.nn.Module | None = None


def default_run_id(config: ModelConfig, timestamp: datetime) -> str:
    return f"{timestamp:%Y%m%dT%H%M%S}_{config.hash()[:8]}_s{int(config.get('train.seed'))}"


def train(config: ModelConfig, lists: Sequence[str] = LIST_PAIRS["full"], paths: ld.DataPaths = ld.DataPaths(),
          run_root: Path = DEFAULT_RUN_ROOT, checkpoint_root: Path = DEFAULT_CHECKPOINT_ROOT, device: str | None = None,
          scorer: Scorer | None = None, scorer_record: Mapping | None = None, run_id: str | None = None,
          table_dir: Path | None = None, log: Callable[[str], None] = log_print, argv: Sequence[str] | None = None,
          extra_record: Mapping | None = None) -> TrainResult:
    """One training run; see the module docstring.

    Args:
        config: Model configuration.
        lists: ``(training list, validation list)`` of the split file.
        paths: Model input locations.
        run_root: Folder of the run records (``reports/runs``).
        checkpoint_root: Folder of the checkpoints (outside git).
        device: Torch device name; automatic when None.
        scorer: Called with a function from scaled, clipped features to standardized
            coefficients, returns score terms per slice; no scoring when None.
        scorer_record: Description of the scorer for the run record.
        run_id: Name of the run; time, configuration hash and seed when None.
        table_dir: Frame-table build, for its record hash.
        argv: Command line of the run, for the record.
        extra_record: Further entries of the run record.

    Raises:
        PermissionError: when ``local_test`` is among the lists.
    """
    started = time.time()
    now = datetime.now(timezone.utc)
    train_list, val_list = lists
    for name in lists:
        ld.require_readable(name)
    run_id = run_id or default_run_id(config, now)
    run_dir = Path(run_root) / run_id
    ckpt_dir = Path(checkpoint_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    for stale in ("best.pt", "last.pt", "warm.pt"):
        (ckpt_dir / stale).unlink(missing_ok=True)
    dev = select_device(device)
    seed = int(config.get("train.seed"))
    seed_everything(seed)
    timing: dict[str, float] = {}
    log(f"run {run_id}: {train_list} / {val_list}, device {dev}, configuration {config.hash()[:12]}")

    # Frames, weights and loaders.
    t0 = time.time()
    split = sp.read_split(paths.split)
    sequence = config.get("model.name") in pm.SEQUENCE_MODELS
    if sequence:
        train_all = ld.load_frames(train_list, config, paths, split, sequences=True)
        val_all = ld.load_frames(val_list, config, paths, split, sequences=True)
        train_frames, val_frames = train_all.kept_frames(), val_all.kept_frames()
    else:
        train_frames = ld.load_frames(train_list, config, paths, split)
        val_frames = ld.load_frames(val_list, config, paths, split)
    basis, flux = ld.load_bases(paths)
    sigma = ld.component_sigma(basis)
    version = int(config.get("features.version"))
    fconfig = fe.FeatureConfig.load(paths.feature_config_path(version))
    scaler = RobustScaler.load(paths.scaler_path(version))
    thomson = ld.thomson_masking(fconfig, train_frames.columns, scaler)
    ip_med = ls.ip_median(train_frames.ip_a)
    rule = ld.tail_rule(train_frames, config) if config.get("loader.tail_weights") else None
    norm_loader = ld.FrameLoader(train_frames, basis.parity, ld.sample_weights(train_frames, config, ip_med, rule), config, thomson,
                                 "eval", dev)
    if sequence:
        train_loader = ld.ShotLoader(train_all, basis.parity, ld.sample_weights(train_all, config, ip_med, rule), config, thomson,
                                     "train", dev)
        val_loader = ld.ShotLoader(val_all, basis.parity, ld.sample_weights(val_all, config, ip_med), config, thomson, "eval", dev)
        val_subset = ~val_all.near_repeat
    else:
        train_loader = ld.FrameLoader(train_frames, basis.parity, ld.sample_weights(train_frames, config, ip_med, rule), config,
                                      thomson, "train", dev)
        val_loader = ld.FrameLoader(val_frames, basis.parity, ld.sample_weights(val_frames, config, ip_med), config, thomson, "eval",
                                    dev)
        val_subset = ~val_frames.near_repeat
    nnr = ~val_frames.near_repeat
    stop_group = "subset" if nnr.any() else "all"
    timing["load_s"] = time.time() - t0
    log(f"  {len(train_frames)} training and {len(val_frames)} validation frames ({int(nnr.sum())} without a near-repeat), "
        f"{len(train_frames.columns)} features, {flux.n_components} components; Ip_med {ip_med:.4g} A")

    # Loss and its normalization.
    t0 = time.time()
    grid = li.read_record(paths.target_version, paths.loss_root)
    operator = ls.FluxOperator(flux.phi_mean, flux.phi, sigma, grid["grid_r"], grid["grid_z"]).to(dev)
    loss_fn = ls.LossFunction(config.get("loss.terms"), operator, config.get("loss.reference"))
    sample = normalization_sample(len(train_frames), int(config.get("loss.norm_frames")), int(config.get("loss.norm_seed")))
    norms = compute_normalization(norm_loader, loss_fn, sample)
    loss_fn.set_normalization(norms)
    timing["normalization_s"] = time.time() - t0
    log("  normalization (c_hat = 0, " + f"{len(sample)} frames): " + ", ".join(f"{k} {v:.4g}" for k, v in norms.items()))

    # Model and warm start.
    t0 = time.time()
    model = pm.build_model(config.get("model"), len(train_frames.columns), flux.n_components, config.get("rung2")).to(dev)
    bound = float(config.get("loader.clip_bound"))
    warm = {"enabled": bool(config.get("model.warm_start"))}
    if warm["enabled"]:
        w, b = pm.warm_start_ridge(np.clip(train_frames.x, -bound, bound), train_frames.c, float(config.get("model.warm_start_alpha")))
        pm.load_linear(model, w, b)
        warm.update(alpha=float(config.get("model.warm_start_alpha")), n_frames=len(train_frames))
    model = model.to(dev)
    warm["validation_loss"] = evaluate_loss(model, val_loader, loss_fn, val_subset)
    timing["warm_start_s"] = time.time() - t0
    log(f"  warm start: validation loss {warm['validation_loss']['all']['total']:.5f}, "
        f"no near-repeat {warm['validation_loss'][stop_group]['total']:.5f}")

    def predict_fn(m: torch.nn.Module) -> Callable[..., np.ndarray]:
        if sequence:
            return lambda x, dt: pm.predict_sequence(m, x, dt, dev)
        return lambda x: pm.predict(m, x, dev)

    def run_scorer(m: torch.nn.Module) -> dict:
        return dict(scorer(predict_fn(m), sequence=True) if sequence else scorer(predict_fn(m)))

    if scorer is not None:
        t0 = time.time()
        warm["score"] = run_scorer(model)
        timing["warm_start_score_s"] = time.time() - t0
        log(f"  warm-start score: S {warm['score']['all']['S']:.4f} (no near-repeat {warm['score']['no_near_repeat']['S']:.4f})")

    # Training.
    t0 = time.time()
    max_epochs = int(config.get("train.max_epochs"))
    steps_per_epoch = len(train_loader)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.get("train.lr")), weight_decay=float(config.get("train.weight_decay")))
    total_steps = max_epochs * steps_per_epoch
    warmup_steps = int(config.get("train.warmup_epochs")) * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: lr_factor(s, total_steps, warmup_steps))
    stopper = EarlyStopping(int(config.get("train.patience")))
    grad_clip = float(config.get("train.grad_clip"))
    meta = {"columns": train_frames.columns, "parity": basis.parity.tolist(), "sigma": sigma.tolist(), "config_hash": config.hash(),
            "config": config.to_dict(), "run_id": run_id, "feature_version": version, "target_version": paths.target_version,
            "sequence": sequence, "uses_future": bool(getattr(model, "uses_future", False))}
    # The warm start as a checkpoint, so that it can be scored later through the prediction chain
    # (runs trained without the scorer).
    pm.save_checkpoint(ckpt_dir / "warm.pt", model, {**meta, "epoch": -1})
    metrics: list[dict] = []
    metrics_path = run_dir / "metrics.csv"
    metrics_path.unlink(missing_ok=True)
    epoch = -1
    for epoch in range(max_epochs):
        e0 = time.time()
        model.train()
        sums = {t: 0.0 for t in loss_fn.enabled}
        n_seen = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            c_hat = forward(model, batch)
            if c_hat.shape[0] == 0:
                continue
            weighted = loss_fn.weighted_sums(c_hat, batch)
            n = int(c_hat.shape[0])
            total = loss_fn.total(loss_fn.normalized_means(weighted, n))
            total.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()
            for t, v in weighted.items():
                sums[t] += ls.host_float(v)
            n_seen += n
        train_parts = {t: sums[t] / (n_seen * loss_fn.normalization[t]) for t in loss_fn.enabled}
        train_parts["total"] = sum(loss_fn.weights[t] * train_parts[t] for t in loss_fn.enabled)
        val = evaluate_loss(model, val_loader, loss_fn, val_subset)
        improved = stopper.update(val[stop_group]["total"], epoch)
        row = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"], "seconds": time.time() - e0,
               **{f"train_{k}": v for k, v in train_parts.items()},
               **{f"val_{k}": v for k, v in val["all"].items() if k != "n_frames"},
               **{f"val_nnr_{k}": v for k, v in val.get("subset", val["all"]).items() if k != "n_frames"}, "best": improved}
        metrics.append(row)
        with metrics_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            if epoch == 0:
                writer.writeheader()
            writer.writerow(row)
        if improved:
            pm.save_checkpoint(ckpt_dir / "best.pt", model, {**meta, "epoch": epoch})
        log(f"  epoch {epoch:3d}: train {train_parts['total']:.5f}, validation {val['all']['total']:.5f}, "
            f"no near-repeat {val.get('subset', val['all'])['total']:.5f}{' *' if improved else ''}")
        if stopper.stop:
            break
    pm.save_checkpoint(ckpt_dir / "last.pt", model, {**meta, "epoch": epoch})
    timing["training_s"] = time.time() - t0

    # Best checkpoint and its score.
    if not (ckpt_dir / "best.pt").is_file():
        raise RuntimeError(f"no epoch gave a finite validation loss; see {metrics_path}")
    best_model, _ = pm.load_checkpoint(ckpt_dir / "best.pt", dev)
    best_loss = evaluate_loss(best_model, val_loader, loss_fn, val_subset)
    best_score = None
    if scorer is not None:
        t0 = time.time()
        best_score = run_scorer(best_model)
        timing["best_score_s"] = time.time() - t0
        log(f"  best checkpoint (epoch {stopper.best_epoch}): S {best_score['all']['S']:.4f} "
            f"(no near-repeat {best_score['no_near_repeat']['S']:.4f}; warm start "
            f"{warm['score']['all']['S']:.4f})")

    timing["total_s"] = time.time() - started
    record = {
        "run_id": run_id, "timestamp": now.isoformat(timespec="seconds"), "argv": list(argv) if argv is not None else None,
        "repo_commit": starter_kit.git_commit(dataset.REPO_ROOT), "starter_commit": starter_kit.git_commit(starter_kit.starter_dir()),
        "config": config.to_dict(), "config_hash": config.hash(), "config_sources": list(config.sources),
        "lists": {"train": train_list, "validation": val_list}, "seed": seed, "device": str(dev),
        "python": platform.python_version(), "torch": torch.__version__, "environment": environment_freeze(),
        "data_hashes": data_hashes(config, paths, table_dir),
        "machine_constants": {"DIII-D": machine_config("DIII-D").constants_record()},
        "n_frames": {"train": len(train_frames), "validation": len(val_frames), "validation_no_near_repeat": int(nnr.sum())},
        "model": {"name": config.get("model.name"), "rung": 2 if sequence else 1, "sequence": sequence,
                  "uses_future": bool(getattr(model, "uses_future", False)), "init_args": dict(model.init_args),
                  "n_parameters": int(sum(p.numel() for p in model.parameters()))},
        "features": {"columns": train_frames.columns, "thomson_columns": [train_frames.columns[j] for j in thomson[0]]},
        "ip_med_a": ip_med, "tail_rule": rule, "normalization": {"constants": norms, "n_frames": int(len(sample)),
                                                                 "seed": int(config.get("loss.norm_seed"))},
        "early_stopping": {"monitor": f"validation {'no-near-repeat' if stop_group == 'subset' else 'all-frame'} total loss",
                           "best_epoch": stopper.best_epoch, "best_value": stopper.best, "stopped_epoch": epoch,
                           "stopped_early": stopper.stop, "patience": stopper.patience, "max_epochs": max_epochs},
        "warm_start": warm, "best": {"validation_loss": best_loss, "score": best_score},
        "scorer": dict(scorer_record) if scorer_record else None,
        "checkpoints": {"dir": str(ckpt_dir), "best_sha256": sha256_or_none(ckpt_dir / "best.pt"),
                        "warm_sha256": sha256_or_none(ckpt_dir / "warm.pt"),
                        "last_sha256": sha256_or_none(ckpt_dir / "last.pt")},
        "timing": timing,
    }
    if extra_record:
        record.update(dict(extra_record))
    (run_dir / "run.json").write_text(json.dumps(record, indent=2, default=_json_default) + "\n")
    log(f"  wrote {run_dir / 'run.json'} ({timing['total_s']:.0f} s)")
    return TrainResult(run_id=run_id, run_dir=run_dir, checkpoint_dir=ckpt_dir, record=record, metrics=metrics, model=best_model)


def _json_default(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)
