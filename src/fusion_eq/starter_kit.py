"""Access to the organizers' read-only starter kit.

The starter kit is a sibling checkout of this repository. It is never modified and
never copied; its data fixes and its scorer are imported from their original location
so that a scorer update upstream is picked up by a plain ``git pull`` there.

The scorer modules under ``fusion_scoring/`` use flat imports (``from common import
...``), so importing them requires that directory on ``sys.path``. This module does
that insertion once, exactly as the organizers' ``local_score.py`` does.

Location: the environment variable ``FUSION_STARTER_DIR`` when set, otherwise
``../fusion-equilibrium-challenge-starter`` relative to the repository root.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np

from fusion_eq.machines import MachineConfig, machine_config

ENV_VAR = "FUSION_STARTER_DIR"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STARTER_DIR = REPO_ROOT.parent / "fusion-equilibrium-challenge-starter"

SCORING_MODULES = ("common", "contour", "o_point", "lcfs", "derive", "metrics")


def starter_dir() -> Path:
    """Returns the starter-kit directory, without checking that it exists."""
    return Path(os.environ.get(ENV_VAR, DEFAULT_STARTER_DIR)).expanduser().resolve()


def is_available() -> bool:
    """True when the starter kit and its scorer are present on disk."""
    root = starter_dir()
    return (root / "data_fixes.py").is_file() and (root / "fusion_scoring" / "lcfs.py").is_file()


def require() -> Path:
    """Returns the starter-kit directory or raises with instructions."""
    root = starter_dir()
    if not is_available():
        raise FileNotFoundError(
            f"starter kit not found at {root}; clone "
            "https://github.com/Sophelio/fusion-equilibrium-challenge-starter next to this "
            f"repository or set {ENV_VAR}"
        )
    return root


@lru_cache(maxsize=1)
def data_fixes() -> ModuleType:
    """Imports ``data_fixes.py`` from the starter kit.

    Returns:
        The module with ``fix_d3d_ip_times`` and ``align_mast_thomson_core``.
    """
    path = require() / "data_fixes.py"
    spec = importlib.util.spec_from_file_location("starter_data_fixes", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def scoring() -> SimpleNamespace:
    """Imports the vendored scorer modules.

    Returns:
        A namespace with attributes ``common``, ``contour``, ``o_point``, ``lcfs``,
        ``derive`` and ``metrics``, each the imported module.
    """
    scoring_dir = str(require() / "fusion_scoring")
    if scoring_dir not in sys.path:
        sys.path.insert(0, scoring_dir)
    return SimpleNamespace(**{name: importlib.import_module(name) for name in SCORING_MODULES})


@lru_cache(maxsize=4)
def envelope_mask(machine: str | MachineConfig) -> SimpleNamespace:
    """Loads the scorer's plasma-envelope mask for one machine.

    Args:
        machine: Machine name or configuration.

    Returns:
        A namespace with ``grid_R`` and ``grid_Z`` in metres (65 each),
        ``mask_coarse`` (65 x 65 bool, rows = Z) and ``mask_f`` (the same as float64),
        the two forms the scorer's LCFS extractor expects.
    """
    cfg = machine if isinstance(machine, MachineConfig) else machine_config(machine)
    with np.load(require() / "fusion_scoring" / "masks" / cfg.mask_file) as f:
        grid_r = np.asarray(f["grid_R"], dtype=np.float64)
        grid_z = np.asarray(f["grid_Z"], dtype=np.float64)
        mask_coarse = np.asarray(f["mask_coarse"], dtype=bool)
    return SimpleNamespace(
        grid_R=grid_r, grid_Z=grid_z, mask_coarse=mask_coarse,
        mask_f=mask_coarse.astype(np.float64),
    )


def sample_shot_paths(machine: str | None = None) -> list[Path]:
    """Lists the sample shot files shipped in the starter kit, sorted by name.

    Args:
        machine: ``"DIII-D"``, ``"MAST"`` or None for both.
    """
    if not is_available():
        return []
    data_dir = starter_dir() / "parquet_data"
    if machine is None:
        pattern = "*.parquet"
    else:
        pattern = machine_config(machine).file_prefix + "*.parquet"
    return sorted(p for p in data_dir.glob(pattern) if p.stat().st_size > 10_000)


def git_commit(repo: Path) -> str | None:
    """Returns the HEAD commit hash of a git checkout, or None when unavailable."""
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None
