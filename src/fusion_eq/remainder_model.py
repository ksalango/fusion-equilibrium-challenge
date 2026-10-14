"""The remainder model of the target (``docs/TARGET.md``, Part 3): a basis of the vacuum remainder
and a ridge from the coil currents and their rates to its coefficients. DIII-D only.

The remainder of a frame is ``r = psi_plasma - Ip G s`` with the exact masked share (F79): the
part of the stored map, minus the coil model, that current inside the scorer's boundary does
not produce. ``current_target.vacuum_remainder`` returns it in the target frame,
``sign(Ip) r``, which is what the basis is fitted on (originals only, native DIII-D grid; the
mirror of a frame's remainder is the flip of the original's and the ridge is not retrained on
mirrors).

Ridge inputs are those of ceiling variant g: the windowed mean and a rate of every current
signal (Ip, solenoid, toroidal field, the 18 F-coils), the currents and their rates multiplied
by sign(Ip) as in ``targets.signed_frames``, standardised with ``baseline.InputScaler``. One
ridge is fitted per rate window: 10 ms (``__rate``) and, where the frame table carries it,
50 ms (``__rate50``). The frame table has no 50 ms rate for the toroidal field, which keeps its
10 ms rate in the 50 ms ridge, so both ridges have 42 inputs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from fusion_eq import baseline as bl, frame_table as ft, targets
from fusion_eq.machines import MachineConfig, machine_config

MACHINE = "DIII-D"
RATE_WINDOWS = {"10ms": "rate", "50ms": "rate50"}
WINDOW_MS = {"10ms": 10, "50ms": 50}


# ------------------------------------------------------------------------------- inputs
def ridge_columns(window: str = "10ms", available: set[str] | None = None,
                  machine: str | MachineConfig = MACHINE) -> list[str]:
    """Inputs of the remainder ridge: the ``__mean`` and a rate of every current signal, checked
    against the frame table's label guard. ``window`` picks the rate (``10ms``: ``__rate``,
    ``50ms``: ``__rate50``); a signal without that column in ``available`` (the toroidal field
    has no 50 ms rate) keeps its 10 ms rate."""
    cfg = machine if isinstance(machine, MachineConfig) else machine_config(machine)
    suffix = RATE_WINDOWS[window]
    columns = []
    for name in ft.current_signals(cfg):
        rate = f"{name}__{suffix}"
        if available is not None and rate not in available:
            rate = f"{name}__rate"
        columns += [f"{name}__mean", rate]
    return ft.assert_input_columns(columns, cfg)


def window_name(ms: int) -> str:
    """``10`` to ``"10ms"``.

    Raises:
        KeyError: for a window without a frame-table column.
    """
    for name, value in WINDOW_MS.items():
        if value == int(ms):
            return name
    raise KeyError(f"no rate column for a {ms} ms window; known: {sorted(WINDOW_MS.values())}")


def available_windows(table_columns, windows_ms=(10, 50)) -> dict[str, list[str]]:
    """The requested rate windows the frame table carries, each with its ridge inputs. The
    10 ms window is always present; the 50 ms window needs the 50 ms Ip rate."""
    names = set(table_columns)
    out = {}
    for ms in windows_ms:
        name = window_name(ms)
        if name == "10ms" or f"ip__{RATE_WINDOWS[name]}" in names:
            out[name] = ridge_columns(name, names)
    return out


def ridge_frames(rows: pd.DataFrame, machine: str | MachineConfig = MACHINE) -> pd.DataFrame:
    """Rows with the currents in the Ip-signed frame: ``targets.signed_frames`` for the
    ``__mean`` columns and the same sign on the matching rate columns of every window, so that
    the inputs follow the target frame of the remainder."""
    cfg = machine if isinstance(machine, MachineConfig) else machine_config(machine)
    out = targets.signed_frames(rows, cfg)
    s = targets.sign_of_ip(rows["ip__mean"].to_numpy(dtype=np.float64))
    for column in targets.signed_current_columns(cfg):
        for suffix in RATE_WINDOWS.values():
            rate = column.replace("__mean", f"__{suffix}")
            if rate in out:
                out[rate] = rows[rate].to_numpy(dtype=np.float64) * s
    return out


def pooled_r2(true, pred) -> float:
    """One minus the summed squared error over the summed variance, pooled over the columns."""
    t = np.asarray(true, dtype=np.float64)
    p = np.asarray(pred, dtype=np.float64)
    if t.shape[0] < 2:
        return float("nan")
    total = float(((t - t.mean(axis=0)) ** 2).sum())
    return 1.0 - float(((t - p) ** 2).sum()) / total if total > 0 else float("nan")


# -------------------------------------------------------------------------------- model
@dataclass
class WindowRidge:
    """The ridge of one rate window.

    Attributes:
        window: ``10ms`` or ``50ms``.
        scaler: Standardisation of the inputs (training statistics).
        weights, intercept: Ridge solution, ``(p, k)`` and ``(k,)``.
        scores: Pooled coefficient R2 per list (``train``, ``validation``, ...).
    """

    window: str
    scaler: bl.InputScaler
    weights: np.ndarray
    intercept: np.ndarray
    scores: dict = field(default_factory=dict)

    @property
    def columns(self) -> tuple[str, ...]:
        return self.scaler.columns

    def coefficients(self, rows: pd.DataFrame) -> np.ndarray:
        """Predicted coefficients ``(rows, k)`` of frame-table rows (signed inside)."""
        x, _ = self.scaler.transform(ridge_frames(rows))
        return x @ self.weights + self.intercept


@dataclass
class RemainderModel:
    """Basis of ``sign(Ip) r`` and the ridges to its coefficients.

    Attributes:
        basis: Remainder basis in the target frame (``mean``, ``components``); the first
            ``n_components`` rows are the model's.
        n_components: Coefficients predicted.
        explained_cumulative: Variance share held by the leading components, as fitted (up to
            the basis's length).
        ridges: Window name to its ridge.
        default_window: The window ``predict`` uses unless told otherwise.
        alpha: Ridge regularisation.
    """

    basis: bl.FluxBasis
    n_components: int
    explained_cumulative: np.ndarray
    ridges: dict[str, WindowRidge] = field(default_factory=dict)
    default_window: str = "10ms"
    alpha: float = 1.0

    def encode(self, remainder_target) -> np.ndarray:
        """Coefficients ``(frames, k)`` of remainders in the target frame."""
        r = np.asarray(remainder_target, dtype=np.float64).reshape(-1, self.basis.mean.size)
        return (r - self.basis.mean) @ self.basis.components[:self.n_components].T

    def decode(self, coefficients) -> np.ndarray:
        """Remainders in the target frame, ``(frames, nZ, nR)``, from coefficients."""
        c = np.asarray(coefficients, dtype=np.float64).reshape(-1, self.n_components)
        return (c @ self.basis.components[:self.n_components] + self.basis.mean).reshape(-1, bl.GRID, bl.GRID)

    def fit_ridge(self, window: str, columns, rows: pd.DataFrame, coefficients) -> WindowRidge:
        """Fits and stores the ridge of one window on frame-table rows and their true
        coefficients; rows with incomplete inputs are left out of the fit."""
        frames = ridge_frames(rows)
        scaler = bl.InputScaler.fit(frames, list(columns))
        x, complete = scaler.transform(frames)
        y = np.asarray(coefficients, dtype=np.float64)
        if len(y) != len(rows):
            raise ValueError(f"{len(y)} coefficient rows for {len(rows)} frames")
        weights, intercept = bl.ridge_fit(x[complete], y[complete], self.alpha)
        ridge = WindowRidge(window, scaler, weights, intercept,
                            {"train": pooled_r2(y[complete], x[complete] @ weights + intercept),
                             "n_train": int(complete.sum())})
        self.ridges[window] = ridge
        return ridge

    def score(self, window: str, name: str, rows: pd.DataFrame, coefficients) -> float:
        """Pooled coefficient R2 of a window's ridge on rows, stored under ``name``."""
        value = pooled_r2(coefficients, self.ridges[window].coefficients(rows))
        self.ridges[window].scores[name] = value
        return value

    def choose_default(self, by: str = "validation") -> str:
        """Sets the default window to the one with the best score ``by`` (the first window on
        a tie or when no score exists)."""
        scored = {w: r.scores.get(by, float("nan")) for w, r in self.ridges.items()}
        finite = {w: v for w, v in scored.items() if np.isfinite(v)}
        self.default_window = max(finite, key=finite.get) if finite else next(iter(self.ridges))
        return self.default_window

    def predict_target(self, rows: pd.DataFrame, window: str | None = None) -> np.ndarray:
        """Predicted ``sign(Ip) r`` maps of frame-table rows, Wb/rad."""
        return self.decode(self.ridges[window or self.default_window].coefficients(rows))

    def predict(self, rows: pd.DataFrame, window: str | None = None) -> np.ndarray:
        """Predicted remainder in the physical frame, ``r``, for ``current_basis.compose_map``."""
        sign = targets.sign_of_ip(rows["ip__mean"].to_numpy(dtype=np.float64))[:, None, None]
        return sign * self.predict_target(rows, window)

    # ------------------------------------------------------------------------- storage
    def save(self, path: Path | str) -> None:
        meta = {"n_components": self.n_components, "default_window": self.default_window, "alpha": self.alpha,
                "basis_n_frames": self.basis.n_frames, "basis_total_variance": self.basis.total_variance,
                "ridges": {w: {"scaler": r.scaler.to_dict(), "scores": r.scores} for w, r in self.ridges.items()}}
        arrays = {"meta": np.array(json.dumps(meta, sort_keys=True)), "basis_mean": self.basis.mean,
                  "basis_components": self.basis.components, "basis_explained": self.basis.explained_variance,
                  "explained_cumulative": self.explained_cumulative}
        for w, r in self.ridges.items():
            arrays[f"weights_{w}"] = r.weights
            arrays[f"intercept_{w}"] = r.intercept
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path: Path | str) -> "RemainderModel":
        with np.load(path, allow_pickle=False) as f:
            meta = json.loads(str(f["meta"]))
            basis = bl.FluxBasis(f["basis_mean"], f["basis_components"], f["basis_explained"],
                                 float(meta["basis_total_variance"]), int(meta["basis_n_frames"]))
            ridges = {w: WindowRidge(w, bl.InputScaler.from_dict(m["scaler"]), f[f"weights_{w}"], f[f"intercept_{w}"],
                                     dict(m["scores"])) for w, m in meta["ridges"].items()}
            return cls(basis, int(meta["n_components"]), f["explained_cumulative"], ridges, str(meta["default_window"]),
                       float(meta["alpha"]))

    @classmethod
    def from_basis(cls, basis: bl.FluxBasis, n_components: int, alpha: float = 1.0) -> "RemainderModel":
        """A model on a fitted remainder basis, before any ridge."""
        if n_components > basis.n_components:
            raise ValueError(f"{n_components} components requested of a basis of {basis.n_components}")
        return cls(basis, int(n_components), np.cumsum(basis.explained_ratio), {}, alpha=float(alpha))
