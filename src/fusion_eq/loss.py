"""Loss of the plasma model: the flux operator, the frame weights and the registry of terms
(``docs/PIPELINE.md``, Part 1 and "Definitions").

The model predicts ``c_hat``, the 50 current coefficients standardized by the stored component
standard deviations (no centring, so ``c_hat = 0`` is the mean current). The flux of the
predicted current is ``psi_hat = Ip (phi_mean + Phi (sigma c_hat))`` in Wb/rad, Ip in A, in the
physical frame of the target build. The loss reference is ``psi_ref = Ip (phi_mean + Phi
(sigma c_true))`` by default (``basis``) or a stored ``Ip G s`` (``exact``). Every map term acts
on the error ``psi_hat - psi_ref``. Under ``basis`` the error is formed as
``Ip Phi (sigma (c_hat - c_true))``, so that it is exactly zero for ``c_hat = c_true``.

The true coil flux and the remainder are common to the predicted and the true full map, so the
error of the plasma flux is the error of the full map and the terms act on the quantities the
scorer derives from the full map (check Z3). Masks come from the true map.

Terms, each a value per frame:

- ``flux``: mean squared error over the inside mask ``M_f``.
- ``grad``: mean of the squared error of the central-difference gradient (``dR``, ``dZ``) over
  the interior nodes of ``M_f``.
- ``band``: mean squared error over the band mask ``B_f``.
- ``coeff``: weighted squared error on named components of ``c`` (standardized units).
- ``contour``, ``li_term``: registered, not implemented in version 1; enabling them raises.

A frame without nodes in a mask contributes zero to that mask's terms. The total loss is
``sum_t weight_t mean_f(w_f term_t,f) / norm_t``, where ``norm_t`` is the same weighted mean
for ``c_hat = 0`` on a fixed sample of training frames.

The module is written for torch tensors of any floating dtype; the tests run it in float64.
Sums leave the device through ``host_float``, which converts on the CPU: Apple's MPS backend
has no float64.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np
import torch

from fusion_eq.model_config import PHASES

A_PER_KA = 1000.0


# --------------------------------------------------------------------------- operator
class FluxOperator(torch.nn.Module):
    """``Ip (phi_mean + Phi (sigma c))`` on one machine's grid.

    Args:
        phi_mean: ``(P,)`` flux of the mean share, Wb/rad per ampere of Ip.
        phi: ``(K, P)`` flux of each component, Wb/rad per ampere of Ip.
        sigma: ``(K,)`` component standard deviations (standardized to raw coefficients).
        grid_r, grid_z: Grid axes in metres, ``P = len(grid_z) * len(grid_r)``.
        dtype: Floating dtype of the buffers.
    """

    def __init__(self, phi_mean, phi, sigma, grid_r, grid_z, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        grid_r = np.asarray(grid_r, dtype=np.float64)
        grid_z = np.asarray(grid_z, dtype=np.float64)
        self.shape = (int(grid_z.size), int(grid_r.size))
        phi = np.asarray(phi, dtype=np.float64)
        if phi.shape[1] != self.shape[0] * self.shape[1]:
            raise ValueError(f"flux basis of {phi.shape[1]} nodes for a grid of {self.shape}")
        self.d_r = float(np.mean(np.diff(grid_r)))
        self.d_z = float(np.mean(np.diff(grid_z)))
        self.register_buffer("phi_mean", torch.as_tensor(np.asarray(phi_mean, dtype=np.float64), dtype=dtype))
        self.register_buffer("phi", torch.as_tensor(phi, dtype=dtype))
        self.register_buffer("sigma", torch.as_tensor(np.asarray(sigma, dtype=np.float64), dtype=dtype))

    @property
    def n_components(self) -> int:
        return int(self.phi.shape[0])

    def plasma_flux(self, c_std: torch.Tensor, ip_a: torch.Tensor) -> torch.Tensor:
        """``(B, nZ, nR)`` flux in Wb/rad of standardized coefficients ``(B, K)``, Ip in A."""
        flux = self.phi_mean + (c_std * self.sigma) @ self.phi
        return (ip_a[:, None] * flux).reshape(-1, *self.shape)

    def flux_error(self, c_hat: torch.Tensor, c_true: torch.Tensor, ip_a: torch.Tensor) -> torch.Tensor:
        """``psi_hat - psi_ref`` under the basis reference, ``(B, nZ, nR)`` in Wb/rad."""
        return (ip_a[:, None] * (((c_hat - c_true) * self.sigma) @ self.phi)).reshape(-1, *self.shape)


def host_float(value: torch.Tensor) -> float:
    """Sum of a tensor as a Python float, accumulated in float64 on the host.

    The tensor is moved to the CPU before the conversion, since the MPS backend cannot hold
    float64; on CUDA and the CPU the result is the same as converting in place.
    """
    return float(value.detach().cpu().to(torch.float64).sum())


# ------------------------------------------------------------------------ frame weights
def ip_median(ip_a) -> float:
    """``Ip_med``: median of ``abs(Ip)`` in A over the given (training) frames."""
    v = np.abs(np.asarray(ip_a, dtype=np.float64))
    v = v[np.isfinite(v)]
    if v.size == 0:
        raise ValueError("no finite Ip to take the median of")
    return float(np.median(v))


def phase_index(phases) -> np.ndarray:
    """Index into ``PHASES`` of each phase name.

    Raises:
        ValueError: for an unknown phase name.
    """
    lookup = {p: i for i, p in enumerate(PHASES)}
    names = np.asarray(phases).astype(str)
    bad = sorted(set(names) - set(lookup))
    if bad:
        raise ValueError(f"unknown phases {bad}; known: {PHASES}")
    return np.array([lookup[p] for p in names], dtype=np.int64)


def frame_weights(ip_a, phases, ip_med: float, gamma: float = 2.0, phase_weights=(1.0, 1.0, 1.0)) -> np.ndarray:
    """``w_f = (abs(Ip_f) / Ip_med) ** (gamma - 2) * phase_weights[phase_f]``.

    Args:
        ip_a: Plasma current per frame in A (sign ignored).
        phases: Phase index per frame (into ``PHASES``) or phase names.
        ip_med: Median ``abs(Ip)`` of the training frames, A.
        gamma: Frame-weight exponent; 2 leaves the natural Ip-squared weight of the flux loss.
        phase_weights: Multiplier per phase, in the order of ``PHASES``.
    """
    ip = np.abs(np.asarray(ip_a, dtype=np.float64))
    idx = np.asarray(phases)
    idx = phase_index(idx) if idx.dtype.kind in "OUS" else idx.astype(np.int64)
    multipliers = np.asarray(phase_weights, dtype=np.float64)[idx]
    if float(gamma) == 2.0:
        return multipliers
    return (ip / float(ip_med)) ** (float(gamma) - 2.0) * multipliers


# ------------------------------------------------------------------------------ terms
def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-frame mean of ``values`` over the mask's nodes; zero for an empty mask."""
    m = mask.to(values.dtype)
    count = m.sum(dim=(1, 2))
    return (values * m).sum(dim=(1, 2)) / torch.clamp(count, min=1.0)


def central_gradient(field: torch.Tensor, d_r: float, d_z: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Central differences ``(d/dR, d/dZ)`` of ``(B, nZ, nR)`` maps on the interior nodes,
    each ``(B, nZ - 2, nR - 2)``."""
    g_r = (field[:, 1:-1, 2:] - field[:, 1:-1, :-2]) / (2.0 * d_r)
    g_z = (field[:, 2:, 1:-1] - field[:, :-2, 1:-1]) / (2.0 * d_z)
    return g_r, g_z


@dataclass
class TermContext:
    """What a term reads for one batch: the error map (computed once), the batch and the
    predicted and true coefficients."""

    c_hat: torch.Tensor
    batch: Mapping[str, torch.Tensor]
    operator: FluxOperator
    reference: str
    _error: torch.Tensor | None = None

    @property
    def error(self) -> torch.Tensor:
        if self._error is None:
            b = self.batch
            if self.reference == "basis":
                self._error = self.operator.flux_error(self.c_hat, b["c"], b["ip_a"])
            else:
                if "psi_ref" not in b:
                    raise KeyError("loss.reference exact needs psi_ref in the batch (build_loss_inputs.py --reference exact)")
                psi_hat = self.operator.plasma_flux(self.c_hat, b["ip_a"])
                self._error = psi_hat - b["psi_ref"].to(psi_hat.dtype).reshape(psi_hat.shape)
        return self._error


def flux_term(ctx: TermContext, params: Mapping) -> torch.Tensor:
    return masked_mean(ctx.error ** 2, ctx.batch["inside"])


def grad_term(ctx: TermContext, params: Mapping) -> torch.Tensor:
    g_r, g_z = central_gradient(ctx.error, ctx.operator.d_r, ctx.operator.d_z)
    return masked_mean(g_r ** 2 + g_z ** 2, ctx.batch["inside"][:, 1:-1, 1:-1])


def band_term(ctx: TermContext, params: Mapping) -> torch.Tensor:
    return masked_mean(ctx.error ** 2, ctx.batch["band"])


def coeff_term(ctx: TermContext, params: Mapping) -> torch.Tensor:
    idx = torch.as_tensor(list(params.get("components", [])), dtype=torch.long, device=ctx.c_hat.device)
    w = torch.as_tensor(list(params.get("component_weights", [])), dtype=ctx.c_hat.dtype, device=ctx.c_hat.device)
    diff = ctx.c_hat[:, idx] - ctx.batch["c"][:, idx]
    return (w * diff ** 2).sum(dim=1)


def not_implemented_term(name: str) -> Callable:
    def term(ctx: TermContext, params: Mapping) -> torch.Tensor:
        raise NotImplementedError(f"loss term {name!r} is registered but not implemented in version 1")
    return term


@dataclass(frozen=True)
class TermSpec:
    """A registered loss term: its function and whether version 1 implements it."""

    function: Callable[[TermContext, Mapping], torch.Tensor]
    implemented: bool = True


TERMS: dict[str, TermSpec] = {
    "flux": TermSpec(flux_term),
    "grad": TermSpec(grad_term),
    "band": TermSpec(band_term),
    "coeff": TermSpec(coeff_term),
    "contour": TermSpec(not_implemented_term("contour"), implemented=False),
    "li_term": TermSpec(not_implemented_term("li_term"), implemented=False),
}
PARAM_KEYS = ("enabled", "weight")


# ------------------------------------------------------------------------------ loss
class LossFunction:
    """The configured, normalized loss.

    Args:
        terms: ``loss.terms`` of the configuration: name to ``{enabled, weight, ...}``.
        operator: The machine's flux operator.
        reference: ``basis`` or ``exact``.
        normalization: Term name to its constant; set later by ``set_normalization``.

    Raises:
        KeyError: for an unregistered term.
        NotImplementedError: when a term version 1 does not implement is enabled.
    """

    def __init__(self, terms: Mapping[str, Mapping], operator: FluxOperator, reference: str = "basis",
                 normalization: Mapping[str, float] | None = None) -> None:
        unknown = [t for t in terms if t not in TERMS]
        if unknown:
            raise KeyError(f"unregistered loss terms {unknown}; registered: {sorted(TERMS)}")
        self.params = {name: dict(spec) for name, spec in terms.items()}
        self.enabled = [name for name in TERMS if name in terms and bool(terms[name].get("enabled", False))]
        for name in self.enabled:
            if not TERMS[name].implemented:
                raise NotImplementedError(f"loss term {name!r} is registered but not implemented in version 1")
        self.weights = {name: float(self.params[name].get("weight", 1.0)) for name in self.enabled}
        self.operator = operator
        self.reference = reference
        self.normalization = dict(normalization) if normalization is not None else None

    def per_frame(self, c_hat: torch.Tensor, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Unnormalized, unweighted value per frame ``(B,)`` of every enabled term."""
        ctx = TermContext(c_hat=c_hat, batch=batch, operator=self.operator, reference=self.reference)
        return {name: TERMS[name].function(ctx, self.params[name]) for name in self.enabled}

    def weighted_sums(self, c_hat: torch.Tensor, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """``sum_f w_f term_f`` of every enabled term over the batch (for exact accumulation
        over batches of any size)."""
        w = batch["weight"]
        return {name: (w.to(v.dtype) * v).sum() for name, v in self.per_frame(c_hat, batch).items()}

    def set_normalization(self, constants: Mapping[str, float]) -> None:
        """Sets the constants after checking that each is finite and positive.

        Raises:
            ValueError: for a missing, non-finite or non-positive constant.
        """
        for name in self.enabled:
            value = constants.get(name)
            if value is None or not np.isfinite(value) or value <= 0:
                raise ValueError(f"normalization constant of {name!r} is {value}; it must be finite and positive")
        self.normalization = {name: float(constants[name]) for name in self.enabled}

    def compute_normalization(self, batch: Mapping[str, torch.Tensor]) -> dict[str, float]:
        """Each enabled term's weighted mean ``mean_f(w_f term_f)`` for the mean-current
        prediction ``c_hat = 0`` on the batch (the fixed normalization sample)."""
        c0 = torch.zeros_like(batch["c"])
        n = batch["c"].shape[0]
        with torch.no_grad():
            sums = self.weighted_sums(c0, batch)
        return {name: host_float(v) / n for name, v in sums.items()}

    def normalized_means(self, sums: Mapping[str, torch.Tensor], n_frames: int) -> dict[str, torch.Tensor]:
        """``sum_f w_f term_f / (N norm_t)`` of each term from accumulated sums."""
        if self.normalization is None:
            raise RuntimeError("normalization constants are not set")
        return {name: sums[name] / (n_frames * self.normalization[name]) for name in self.enabled}

    def total(self, parts: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """``sum_t weight_t part_t`` of normalized parts."""
        return sum(self.weights[name] * parts[name] for name in self.enabled)

    def __call__(self, c_hat: torch.Tensor, batch: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        """The batch loss and each term's normalized batch mean (detached floats)."""
        parts = self.normalized_means(self.weighted_sums(c_hat, batch), int(c_hat.shape[0]))
        return self.total(parts), {name: host_float(v) for name, v in parts.items()}
