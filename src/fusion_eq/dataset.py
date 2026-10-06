"""Dataset layout: splits, shot file listing and selection, run provenance.

Shared by every command-line runner that walks the dataset (quality checks, frame
table build). The raw data root is read-only; nothing here writes to it.

Location of the raw data: ``--data-root`` on the command line, otherwise
``../data/raw/data`` relative to the repository root. Each split is a folder of
one-row Parquet files named ``<prefix><shot id>.parquet``.
"""

from __future__ import annotations

import json
import platform
import random
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from fusion_eq import starter_kit
from fusion_eq.io import parse_shot_filename
from fusion_eq.machines import machine_config

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = REPO_ROOT.parent / "data" / "raw" / "data"
DEFAULT_PROCESSED_ROOT = REPO_ROOT.parent / "data" / "processed"

DATA_SPLITS = ("diii_d_train", "diii_d_public_test", "mast_public_test")
# The three labelled MAST sample shots of the starter kit; diagnostics only.
SAMPLE_SPLIT = "mast_samples"
ALL_SPLITS = DATA_SPLITS + (SAMPLE_SPLIT,)
LABELLED_SPLITS = ("diii_d_train", SAMPLE_SPLIT)

SPLIT_MACHINE = {"diii_d_train": "DIII-D", "diii_d_public_test": "DIII-D", "mast_public_test": "MAST",
                 SAMPLE_SPLIT: "MAST"}

EXPECTED_SHOTS = {"diii_d_train": 7041, "diii_d_public_test": 874, "mast_public_test": 1206,
                  SAMPLE_SPLIT: 3}
EXPECTED_FRAMES = {"diii_d_train": 1_559_340, "diii_d_public_test": 193_359, "mast_public_test": 69_075}


def default_data_root() -> Path:
    """The raw data root used when none is given on the command line."""
    return DEFAULT_DATA_ROOT


def shot_path(data_root: Path | str, split: str, shot_id: str) -> Path:
    """File of one shot of a data split (not the sample split), without checking existence."""
    if split not in DATA_SPLITS:
        raise ValueError(f"unknown data split {split!r}; known: {DATA_SPLITS}")
    prefix = machine_config(SPLIT_MACHINE[split]).file_prefix
    return Path(data_root) / split / f"{prefix}{shot_id}.parquet"


def list_shots(data_root: Path | str, split: str) -> list[Path]:
    """Sorted shot files of one split; the MAST samples come from the starter kit.

    Raises:
        FileNotFoundError: when the split folder does not exist.
        ValueError: for an unknown split name.
    """
    if split == SAMPLE_SPLIT:
        return starter_kit.sample_shot_paths("MAST")
    if split not in DATA_SPLITS:
        raise ValueError(f"unknown split {split!r}; known: {ALL_SPLITS}")
    folder = Path(data_root) / split
    if not folder.is_dir():
        raise FileNotFoundError(f"split folder not found: {folder}")
    return sorted(folder.glob("*.parquet"))


def select_shots(files: list[Path], n_shots: int | None, seed: int) -> list[Path]:
    """Deterministic sample of ``n_shots`` files, sorted, or all of them when None."""
    if n_shots is None or n_shots >= len(files):
        return list(files)
    rng = random.Random(seed)
    return sorted(rng.sample(files, n_shots))


def read_shots_file(path: Path | str) -> list[str]:
    """Reads a JSON list of shot ids (file stems without the machine prefix).

    Raises:
        ValueError: when the file does not hold a list of strings.
    """
    ids = json.loads(Path(path).read_text())
    if not isinstance(ids, list) or not all(isinstance(s, str) for s in ids):
        raise ValueError(f"{path}: expected a JSON list of shot id strings")
    return ids


def filter_shots(files: Iterable[Path], shot_ids: Iterable[str]) -> list[Path]:
    """The files whose shot id is in ``shot_ids``, in the order of ``files``."""
    wanted = set(shot_ids)
    return [p for p in files if parse_shot_filename(p)[1] in wanted]


def run_info(argv: list[str] | None, arguments: dict[str, Any], timestamp: str, runtime_s: float,
             **extra: Any) -> dict[str, Any]:
    """Provenance record written next to every dataset-level output.

    Args:
        argv: Command-line arguments as given, or None to take them from ``sys.argv``.
        arguments: Parsed arguments (``vars(args)``); paths become strings.
        timestamp: ISO timestamp of the start of the run.
        runtime_s: Wall-clock runtime in seconds.
        **extra: Further entries, for example per-split counts.

    Returns:
        A JSON-serialisable dict with the repository and starter-kit commits, the
        arguments, the runtime and the versions of Python, numpy and pandas.
    """
    args_text = " ".join(sys.argv[1:]) if argv is None else " ".join(argv)
    info = {
        "timestamp": timestamp,
        "repo_commit": starter_kit.git_commit(REPO_ROOT),
        "starter_commit": starter_kit.git_commit(starter_kit.starter_dir()),
        "args": args_text,
        "arguments": {k: (str(v) if isinstance(v, Path) else v) for k, v in arguments.items()},
        "runtime_s": runtime_s,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    info.update(extra)
    return info
