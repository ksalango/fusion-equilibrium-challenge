"""The physical frame: sign handling, coil flux from frame-table rows and the plasma target.

Finding F45: on DIII-D the stored flux map is the physical flux times minus the sign of
the plasma current, ``psi_stored = c sign(Ip) psi_physical`` with ``c = -1``. A model
that sees inputs and targets in one frame does not have to learn the direction of the
discharge, so inputs and targets are put in the physical frame before any model and
predictions are converted back to the stored frame for the scorer.

With ``s = sign(Ip)`` and the coil flux ``coil`` of the frame (adopted coil model):

    psi_physical = c s psi_stored
    T            = s (psi_physical - coil)        the plasma target, per frame
    psi_stored   = c s coil + c T                 the inverse

``T`` is the plasma's own flux with the direction of the current removed; divided by
``|Ip|`` it is the plasma shape per kA. Signed inputs multiply the plasma current,
the solenoid and the shaping coils by ``s`` and leave the toroidal field and the
Thomson summaries as they are.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from fusion_eq import frame_table as ft
from fusion_eq.coil_flux import ADOPTED, GreenTable, coil_flux
from fusion_eq.machines import MachineConfig, machine_config

# Sign rule constant per machine (F45); MAST is unverified until its first prediction.
PSI_SIGN_RULE_C = {"DIII-D": -1.0}


def sign_rule_c(machine: str | MachineConfig) -> float:
    name = machine if isinstance(machine, str) else machine.name
    try:
        return PSI_SIGN_RULE_C[name]
    except KeyError:
        raise KeyError(f"no sign rule established for {name}") from None


def sign_of_ip(ip_ka) -> np.ndarray:
    """``+1`` where Ip is at or above zero, ``-1`` below; NaN counts as positive."""
    ip = np.asarray(ip_ka, dtype=np.float64)
    return np.where(ip < 0, -1.0, 1.0)


def _per_frame(values, maps) -> np.ndarray:
    """Broadcasts a per-frame vector against ``(frames, nZ, nR)`` maps (or one map)."""
    v = np.asarray(values, dtype=np.float64)
    return v.reshape(-1, 1, 1) if np.ndim(maps) == 3 else v


def to_physical(psi_stored, ip_ka, machine: str | MachineConfig = "DIII-D") -> np.ndarray:
    """``psi_physical = c sign(Ip) psi_stored``."""
    return sign_rule_c(machine) * _per_frame(sign_of_ip(ip_ka), psi_stored) * np.asarray(psi_stored, dtype=np.float64)


def to_stored(psi_physical, ip_ka, machine: str | MachineConfig = "DIII-D") -> np.ndarray:
    """``psi_stored = c sign(Ip) psi_physical`` (the rule is its own inverse)."""
    return to_physical(psi_physical, ip_ka, machine)


def plasma_target(psi_stored, coil, ip_ka, machine: str | MachineConfig = "DIII-D") -> np.ndarray:
    """``T = sign(Ip) (psi_physical - coil)`` from the stored map and the coil flux."""
    s = _per_frame(sign_of_ip(ip_ka), psi_stored)
    return s * (to_physical(psi_stored, ip_ka, machine) - np.asarray(coil, dtype=np.float64))


def stored_from_target(target, coil, ip_ka, machine: str | MachineConfig = "DIII-D") -> np.ndarray:
    """``psi_stored = c sign(Ip) coil + c T``, the inverse of ``plasma_target``."""
    c = sign_rule_c(machine)
    s = _per_frame(sign_of_ip(ip_ka), target)
    return c * s * np.asarray(coil, dtype=np.float64) + c * np.asarray(target, dtype=np.float64)


# -------------------------------------------------------------------- frame-table side
def current_columns_of(table: GreenTable, machine: str | MachineConfig = "DIII-D") -> list[str]:
    """Frame-table ``__mean`` columns in the order of the Green's table's columns."""
    cfg = machine if isinstance(machine, MachineConfig) else machine_config(machine)
    names = {column: name for name, column in ft.current_signals(cfg).items()}
    return [f"{names[c]}__mean" for c in table.columns]


def signed_current_columns(machine: str | MachineConfig = "DIII-D") -> list[str]:
    """Input columns multiplied by ``sign(Ip)``: Ip, the solenoid and the shaping coils."""
    cfg = machine if isinstance(machine, MachineConfig) else machine_config(machine)
    return ["ip__mean", "sol__mean", *[f"{c}__mean" for c in cfg.shaping_coil_columns]]


def signed_frames(frames: pd.DataFrame, machine: str | MachineConfig = "DIII-D") -> pd.DataFrame:
    """A copy of frame-table rows with the current inputs in the physical frame.

    The columns of ``signed_current_columns`` are multiplied by the sign of ``ip__mean``;
    the toroidal field, the Thomson summaries and every other column stay as they are.
    """
    out = frames.copy()
    s = sign_of_ip(out["ip__mean"].to_numpy(dtype=np.float64))
    for column in signed_current_columns(machine):
        if column in out:
            out[column] = (out[column].to_numpy(dtype=np.float64) * s).astype(out[column].dtype)
    return out


def coil_flux_of_frames(table: GreenTable, frames: pd.DataFrame, scales=ADOPTED,
                        machine: str | MachineConfig = "DIII-D") -> np.ndarray:
    """Coil flux ``(frames, nZ, nR)`` of frame-table rows under the machine's coil model.
    Currents are read from the ``__mean`` columns; NaN currents count as zero."""
    currents = frames[current_columns_of(table, machine)].to_numpy(dtype=np.float64)
    return coil_flux(table, currents, scales)


def plasma_target_of_frames(psi_stored, table: GreenTable, frames: pd.DataFrame, scales=ADOPTED,
                            machine: str | MachineConfig = "DIII-D") -> np.ndarray:
    """``T`` of frame-table rows from their stored maps and currents."""
    coil = coil_flux_of_frames(table, frames, scales, machine)
    return plasma_target(psi_stored, coil, frames["ip__mean"].to_numpy(dtype=np.float64), machine)


class MeanShapeAccumulator:
    """Mean over frames of ``T / |Ip|``, the plasma shape per kA (the physics-mean model)."""

    def __init__(self, shape: Sequence[int] = (65, 65), min_ip_ka: float = 1.0) -> None:
        self.sum = np.zeros(tuple(shape))
        self.n = 0
        self.min_ip_ka = float(min_ip_ka)

    def add(self, target, ip_ka) -> None:
        """Adds frames whose ``|Ip|`` is at least ``min_ip_ka`` and whose target is finite."""
        t = np.asarray(target, dtype=np.float64).reshape(-1, *self.sum.shape)
        ip = np.abs(np.asarray(ip_ka, dtype=np.float64)).ravel()
        keep = (ip >= self.min_ip_ka) & np.isfinite(t).all(axis=(1, 2))
        if keep.any():
            self.sum += (t[keep] / ip[keep][:, None, None]).sum(axis=0)
            self.n += int(keep.sum())

    @property
    def mean(self) -> np.ndarray:
        if self.n == 0:
            raise ValueError("no frame added")
        return self.sum / self.n
