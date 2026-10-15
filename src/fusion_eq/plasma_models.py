"""Registered plasma models (``docs/PIPELINE.md``, Part 3): scaled features in, standardized
current coefficients out, and the ridge warm start of the linear path (P2).

``linear_mlp`` (rung 1) is ``c_hat = W x + b + MLP(x)``. The MLP's output layer starts at zero,
so after the warm start the model is exactly the ridge. The bounded variant replaces the MLP
output ``m`` by ``bound tanh(m / bound)`` (item 4, off by default).

Rung 2 runs over a shot's frames: ``c_hat_t = W x_t + b + head(h_t)``, where ``h_t`` is the
state of a sequence encoder fed ``(x_t, dt_t)`` (the scaled features and the scaled time step to
the previous frame). ``gru_seq`` encodes with a GRU, causal by default (bidirectional is an
option, disclosed by ``uses_future``); ``tcn_seq``, the registered alternative, with dilated
temporal convolutions, causal by default (centred kernels otherwise). The per-frame linear skip
``W x_t + b`` is warm-started with rung 1's ridge, and the head's output layer starts at zero, so
at step 0 rung 2 predicts exactly what the warm-started rung 1 predicts. Inputs are padded
batches ``(B, T, F)`` with the valid length of each shot; every output at a valid frame equals
that of the shot run alone, and for a causal model depends on no later frame.

A checkpoint holds the state dict and everything needed to rebuild the model (registered name,
constructor arguments, feature columns, the parity vector and the component standard
deviations), so that a loaded checkpoint reproduces the saved model's outputs exactly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from fusion_eq.baseline import ridge_fit

MODEL_REGISTRY: dict[str, type[nn.Module]] = {}
ACTIVATIONS: dict[str, Callable[[], nn.Module]] = {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU, "tanh": nn.Tanh}


def register_model(name: str):
    """Class decorator adding a model to ``MODEL_REGISTRY`` under ``name``."""
    def wrap(cls):
        if name in MODEL_REGISTRY:
            raise ValueError(f"model {name!r} is already registered")
        cls.registry_name = name
        MODEL_REGISTRY[name] = cls
        return cls
    return wrap


@register_model("linear_mlp")
class LinearMLP(nn.Module):
    """Rung 1: a linear path plus an MLP, summed.

    Args:
        n_in: Number of scaled feature columns.
        n_out: Number of current components.
        hidden: Width of each hidden layer.
        activation: Name in ``ACTIVATIONS``.
        dropout: Dropout after each hidden activation.
        bounded: Saturate the MLP output with ``bound tanh(m / bound)``.
        bound: Saturation level in standardized units of ``c``.
    """

    def __init__(self, n_in: int, n_out: int, hidden: Sequence[int] = (512, 512, 512), activation: str = "gelu",
                 dropout: float = 0.1, bounded: bool = False, bound: float = 3.0) -> None:
        super().__init__()
        if activation not in ACTIVATIONS:
            raise ValueError(f"activation {activation!r} is not one of {sorted(ACTIVATIONS)}")
        self.init_args = {"n_in": int(n_in), "n_out": int(n_out), "hidden": [int(h) for h in hidden], "activation": activation,
                          "dropout": float(dropout), "bounded": bool(bounded), "bound": float(bound)}
        self.linear = nn.Linear(n_in, n_out)
        layers: list[nn.Module] = []
        width = n_in
        for h in hidden:
            layers += [nn.Linear(width, int(h)), ACTIVATIONS[activation](), nn.Dropout(float(dropout))]
            width = int(h)
        self.output = nn.Linear(width, n_out)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.hidden_layers = nn.Sequential(*layers)
        self.bounded = bool(bounded)
        self.bound = float(bound)

    def mlp(self, x: torch.Tensor) -> torch.Tensor:
        m = self.output(self.hidden_layers(x))
        return self.bound * torch.tanh(m / self.bound) if self.bounded else m

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + self.mlp(x)


# ------------------------------------------------------------------------------ rung 2
SEQUENCE_MODELS = ("gru_seq", "tcn_seq")


def valid_mask(lengths: torch.Tensor | None, batch: int, steps: int, device) -> torch.Tensor:
    """``(B, T)`` bool mask of the valid frames of each padded shot."""
    if lengths is None:
        return torch.ones(batch, steps, dtype=torch.bool, device=device)
    lengths = torch.as_tensor(lengths, device=device)
    return torch.arange(steps, device=device)[None, :] < lengths[:, None]


def reverse_within(z: torch.Tensor, lengths: torch.Tensor | None) -> torch.Tensor:
    """Each shot's valid frames in reverse order, padding left in place (its own inverse)."""
    b, t = z.shape[:2]
    if lengths is None:
        return z.flip(1)
    steps = torch.arange(t, device=z.device)[None, :].expand(b, t)
    lengths = torch.as_tensor(lengths, device=z.device)[:, None]
    index = torch.where(steps < lengths, lengths - 1 - steps, steps)
    return torch.gather(z, 1, index[..., None].expand_as(z))


class SequenceModel(nn.Module):
    """Shared part of the rung-2 models: the per-frame linear skip, the head and its zero-initialized
    output layer. Subclasses implement ``encode(z, lengths) -> (B, T, H)``."""

    sequence = True

    def _init_head(self, n_in: int, n_out: int, width: int, head_hidden: Sequence[int], activation: str, dropout: float,
                   bounded: bool, bound: float) -> None:
        if activation not in ACTIVATIONS:
            raise ValueError(f"activation {activation!r} is not one of {sorted(ACTIVATIONS)}")
        self.linear = nn.Linear(n_in, n_out)
        layers: list[nn.Module] = []
        for h in head_hidden:
            layers += [nn.Linear(width, int(h)), ACTIVATIONS[activation](), nn.Dropout(float(dropout))]
            width = int(h)
        self.head_layers = nn.Sequential(*layers)
        self.output = nn.Linear(width, n_out)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.bounded = bool(bounded)
        self.bound = float(bound)

    def head(self, h: torch.Tensor) -> torch.Tensor:
        m = self.output(self.head_layers(h))
        return self.bound * torch.tanh(m / self.bound) if self.bounded else m

    def forward(self, x: torch.Tensor, dt: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        """``(B, T, K)`` standardized coefficients of padded shots.

        Args:
            x: ``(B, T, F)`` scaled, clipped features.
            dt: ``(B, T)`` scaled time step to the previous frame (0 at a shot's first frame).
            lengths: ``(B,)`` valid frames per shot; every frame valid when None.
        """
        mask = valid_mask(lengths, x.shape[0], x.shape[1], x.device)
        z = torch.cat([x, dt.unsqueeze(-1).to(x.dtype)], dim=-1) * mask[..., None].to(x.dtype)
        return self.linear(x) + self.head(self.encode(z, lengths))


@register_model("gru_seq")
class GRUSeq(SequenceModel):
    """Rung 2: a GRU over a shot's frames.

    Args:
        n_in: Number of scaled feature columns (the time step is added as one more input).
        n_out: Number of current components.
        hidden: GRU state size (each direction).
        layers: Stacked GRU layers.
        dropout: Dropout between GRU layers and in the head.
        causal: Unidirectional; must be the negation of ``bidirectional``.
        bidirectional: Add a GRU over the reversed shot (uses future frames).
        head_hidden: Widths of the head's hidden layers.
        activation, bounded, bound: As for ``LinearMLP``.
    """

    def __init__(self, n_in: int, n_out: int, hidden: int = 256, layers: int = 2, dropout: float = 0.1, causal: bool = True,
                 bidirectional: bool = False, head_hidden: Sequence[int] = (256,), activation: str = "gelu",
                 bounded: bool = False, bound: float = 3.0) -> None:
        super().__init__()
        if bool(causal) == bool(bidirectional):
            raise ValueError("gru_seq is causal or bidirectional, not both and not neither")
        self.init_args = {"n_in": int(n_in), "n_out": int(n_out), "hidden": int(hidden), "layers": int(layers),
                          "dropout": float(dropout), "causal": bool(causal), "bidirectional": bool(bidirectional),
                          "head_hidden": [int(h) for h in head_hidden], "activation": activation, "bounded": bool(bounded),
                          "bound": float(bound)}
        drop = float(dropout) if int(layers) > 1 else 0.0
        self.gru = nn.GRU(n_in + 1, int(hidden), int(layers), batch_first=True, dropout=drop)
        self.gru_reverse = nn.GRU(n_in + 1, int(hidden), int(layers), batch_first=True, dropout=drop) if bidirectional else None
        self.uses_future = bool(bidirectional)
        self._init_head(n_in, n_out, int(hidden) * (2 if bidirectional else 1), head_hidden, activation, dropout, bounded, bound)

    def encode(self, z: torch.Tensor, lengths: torch.Tensor | None) -> torch.Tensor:
        h, _ = self.gru(z)
        if self.gru_reverse is None:
            return h
        back, _ = self.gru_reverse(reverse_within(z, lengths))
        return torch.cat([h, reverse_within(back, lengths)], dim=-1)


class TemporalBlock(nn.Module):
    """One dilated convolution with activation, dropout and a residual connection; padded on
    the left (causal) or on both sides (centred)."""

    def __init__(self, c_in: int, c_out: int, kernel: int, dilation: int, causal: bool, activation: str, dropout: float) -> None:
        super().__init__()
        span = (kernel - 1) * dilation
        self.pad = (span, 0) if causal else (span // 2, span // 2)
        self.conv = nn.Conv1d(c_in, c_out, kernel, dilation=dilation)
        self.act = ACTIVATIONS[activation]()
        self.drop = nn.Dropout(float(dropout))
        self.residual = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.residual(h) + self.drop(self.act(self.conv(nn.functional.pad(h, self.pad))))


@register_model("tcn_seq")
class TCNSeq(SequenceModel):
    """Rung 2 alternative: dilated temporal convolutions over a shot's frames (dilations
    ``1, 2, 4, ...``), the state masked to the valid frames after every block so that padding
    never enters a valid frame's output.

    Args:
        n_in, n_out: As for ``GRUSeq``.
        hidden: Channels of every block.
        levels: Number of blocks.
        kernel: Kernel size; odd when not causal.
        causal: Left padding only (no future frame enters an output).
        dropout, head_hidden, activation, bounded, bound: As for ``GRUSeq``.
    """

    def __init__(self, n_in: int, n_out: int, hidden: int = 256, levels: int = 4, kernel: int = 3, causal: bool = True,
                 dropout: float = 0.1, head_hidden: Sequence[int] = (256,), activation: str = "gelu", bounded: bool = False,
                 bound: float = 3.0) -> None:
        super().__init__()
        if activation not in ACTIVATIONS:
            raise ValueError(f"activation {activation!r} is not one of {sorted(ACTIVATIONS)}")
        if not causal and kernel % 2 == 0:
            raise ValueError("a centred tcn_seq needs an odd kernel")
        self.init_args = {"n_in": int(n_in), "n_out": int(n_out), "hidden": int(hidden), "levels": int(levels),
                          "kernel": int(kernel), "causal": bool(causal), "dropout": float(dropout),
                          "head_hidden": [int(h) for h in head_hidden], "activation": activation, "bounded": bool(bounded),
                          "bound": float(bound)}
        self.blocks = nn.ModuleList([TemporalBlock(n_in + 1 if i == 0 else int(hidden), int(hidden), int(kernel), 2 ** i, causal,
                                                   activation, dropout) for i in range(int(levels))])
        self.uses_future = not bool(causal)
        self._init_head(n_in, n_out, int(hidden), head_hidden, activation, dropout, bounded, bound)

    def encode(self, z: torch.Tensor, lengths: torch.Tensor | None) -> torch.Tensor:
        mask = valid_mask(lengths, z.shape[0], z.shape[1], z.device)[:, None, :].to(z.dtype)
        h = z.transpose(1, 2)
        for block in self.blocks:
            h = block(h) * mask
        return h.transpose(1, 2)


def build_model(model_config: Mapping, n_in: int, n_out: int, rung2_config: Mapping | None = None) -> nn.Module:
    """The registered model named by ``model.name`` with the configuration's arguments: the
    ``model`` section for ``linear_mlp``; the ``rung2`` section (whose ``name`` must agree) for
    the sequence models, with ``model.activation``, ``model.bounded`` and ``model.bound``.

    Raises:
        KeyError: for an unregistered name.
        ValueError: when ``model.name`` and ``rung2.name`` name different sequence models.
    """
    name = model_config["name"]
    if name not in MODEL_REGISTRY:
        raise KeyError(f"model {name!r} is not registered; registered: {sorted(MODEL_REGISTRY)}")
    if name == "linear_mlp":
        return LinearMLP(n_in, n_out, model_config["hidden"], model_config["activation"], model_config["dropout"],
                         model_config["bounded"], model_config["bound"])
    r = dict(rung2_config or {})
    if r.get("name", name) != name:
        raise ValueError(f"model.name {name!r} and rung2.name {r.get('name')!r} disagree")
    common = {"head_hidden": r.get("head_hidden", [256]), "activation": model_config.get("activation", "gelu"),
              "bounded": bool(model_config.get("bounded", False)), "bound": float(model_config.get("bound", 3.0)),
              "dropout": float(r.get("dropout", 0.1)), "hidden": int(r.get("hidden", 256))}
    if name == "gru_seq":
        bidirectional = bool(r.get("bidirectional", False))
        return GRUSeq(n_in, n_out, layers=int(r.get("layers", 2)), causal=bool(r.get("causal", True)), bidirectional=bidirectional,
                      **common)
    if name == "tcn_seq":
        return TCNSeq(n_in, n_out, levels=int(r.get("tcn_levels", 4)), kernel=int(r.get("tcn_kernel", 3)),
                      causal=bool(r.get("causal", True)), **common)
    return MODEL_REGISTRY[name](n_in=n_in, n_out=n_out)


def is_sequence(model: nn.Module) -> bool:
    return bool(getattr(model, "sequence", False))


# ---------------------------------------------------------------------------- warm start
def warm_start_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """Ridge (unpenalized intercept, float64) from scaled features ``(N, F)`` to standardized
    coefficients ``(N, K)``: ``(W (F, K), b (K,))``."""
    return ridge_fit(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64), float(alpha))


def load_linear(model: nn.Module, weights: np.ndarray, intercept: np.ndarray) -> None:
    """Copies a ridge into the model's linear path."""
    with torch.no_grad():
        model.linear.weight.copy_(torch.as_tensor(np.asarray(weights).T, dtype=model.linear.weight.dtype))
        model.linear.bias.copy_(torch.as_tensor(np.asarray(intercept), dtype=model.linear.bias.dtype))


def predict(model: nn.Module, x: np.ndarray, device: torch.device | str = "cpu", batch: int = 65536) -> np.ndarray:
    """Model outputs ``(N, K)`` in float64 for scaled (and clipped) features, in eval mode.

    The float32 outputs are moved to the host before the float64 conversion (MPS has no float64).
    """
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(x), batch):
            xb = torch.as_tensor(np.ascontiguousarray(x[start:start + batch]), dtype=torch.float32).to(device)
            out.append(model(xb).detach().cpu().numpy().astype(np.float64))
    return np.concatenate(out) if out else np.zeros((0, model.init_args["n_out"]))


def predict_sequence(model: nn.Module, x: np.ndarray, dt: np.ndarray, device: torch.device | str = "cpu") -> np.ndarray:
    """Outputs ``(T, K)`` in float64 of one shot's scaled features ``(T, F)`` and scaled time
    steps ``(T,)``, the whole shot in one pass, in eval mode."""
    model.eval()
    with torch.no_grad():
        xb = torch.as_tensor(np.ascontiguousarray(x), dtype=torch.float32)[None].to(device)
        db = torch.as_tensor(np.ascontiguousarray(dt), dtype=torch.float32)[None].to(device)
        return model(xb, db).detach().cpu().numpy()[0].astype(np.float64)


# ---------------------------------------------------------------------------- checkpoints
def save_checkpoint(path: Path | str, model: nn.Module, meta: Mapping) -> None:
    """Writes the model and its metadata (columns, parity, sigma, epoch, configuration hash)
    atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save({"model_name": model.registry_name, "init_args": model.init_args, "state_dict": state, "meta": dict(meta)}, tmp)
    tmp.replace(path)


def load_checkpoint(path: Path | str, device: torch.device | str = "cpu") -> tuple[nn.Module, dict]:
    """The model (eval mode, on ``device``) and the metadata of a checkpoint."""
    blob = torch.load(Path(path), map_location="cpu", weights_only=False)
    model = MODEL_REGISTRY[blob["model_name"]](**blob["init_args"])
    model.load_state_dict(blob["state_dict"])
    return model.to(device).eval(), blob["meta"]
