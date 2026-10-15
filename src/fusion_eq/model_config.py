"""Model configuration: ``configs/model/base.yaml`` and the arm files that override it
(``docs/PIPELINE.md``, "Configuration" and Part 7).

An arm file names its parent under ``base:`` (a path relative to the arm file); the chain ends
at a file without ``base:``, the schema. Mappings merge key by key, every other value (lists
included) replaces the parent's. A key absent from the schema fails the load, so that a
misspelt field cannot be silently ignored. The configuration hash is the SHA-256 of the merged
mapping in canonical JSON, independent of the files' layout and comments.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from fusion_eq import dataset

DEFAULT_CONFIG_DIR = dataset.REPO_ROOT / "configs" / "model"
DEFAULT_BASE = DEFAULT_CONFIG_DIR / "base.yaml"
PHASES = ("rising", "flat_top", "late")
REFERENCES = ("basis", "exact")
LOSS_TERMS = ("flux", "grad", "band", "contour", "li_term", "coeff")
# Search-space keys that do not name a configuration field: together they set model.hidden.
VIRTUAL_SPACE_KEYS = ("model.depth", "model.width")
SPACE_TYPES = ("float", "log_float", "int", "categorical")


def deep_merge(parent: Mapping, child: Mapping) -> dict:
    """``child`` over ``parent``: mappings merge key by key, any other value replaces."""
    out = copy.deepcopy(dict(parent))
    for key, value in child.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


# Mappings whose keys are free (search-space entries name configuration paths).
OPEN_MAPPINGS = ("optuna.space",)


def unknown_keys(schema: Mapping, data: Mapping, prefix: str = "") -> list[str]:
    """Dotted paths of ``data`` that ``schema`` does not hold; the content of an open mapping
    (``OPEN_MAPPINGS``) is not checked."""
    found = []
    for key, value in data.items():
        path = f"{prefix}{key}"
        if key not in schema:
            found.append(path)
        elif isinstance(value, Mapping) and isinstance(schema[key], Mapping) and path not in OPEN_MAPPINGS:
            found += unknown_keys(schema[key], value, f"{path}.")
    return found


def _read_chain(path: Path) -> list[tuple[Path, dict]]:
    """``[(file, mapping)]`` from the schema down to ``path``.

    Raises:
        ValueError: for a cycle in the chain or a file that is not a mapping.
    """
    chain, seen = [], set()
    current = Path(path).resolve()
    while True:
        if current in seen:
            raise ValueError(f"configuration chain repeats {current}")
        seen.add(current)
        data = yaml.safe_load(current.read_text()) or {}
        if not isinstance(data, dict):
            raise ValueError(f"{current}: a configuration file must hold a mapping")
        chain.append((current, data))
        parent = data.get("base")
        if parent is None:
            break
        current = (current.parent / str(parent)).resolve()
    return chain[::-1]


@dataclass(frozen=True)
class ModelConfig:
    """A merged model configuration.

    Attributes:
        data: The merged mapping (no ``base`` key).
        sources: Files of the chain, schema first.
    """

    data: dict
    sources: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def load(cls, path: Path | str = DEFAULT_BASE) -> "ModelConfig":
        """Reads an arm file and its chain, merges and validates.

        Raises:
            ValueError: for a key absent from the schema or an invalid value.
        """
        chain = _read_chain(Path(path))
        schema = {k: v for k, v in chain[0][1].items() if k != "base"}
        merged = schema
        for file, data in chain[1:]:
            child = {k: v for k, v in data.items() if k != "base"}
            bad = unknown_keys(schema, child)
            if bad:
                raise ValueError(f"{file}: fields not in the base configuration: {bad}")
            merged = deep_merge(merged, child)
        config = cls(data=merged, sources=tuple(str(f) for f, _ in chain))
        config.validate()
        return config

    @classmethod
    def from_dict(cls, data: Mapping, schema: Mapping | None = None) -> "ModelConfig":
        """A configuration from a mapping, merged over ``schema`` (the base file when None)."""
        schema = schema if schema is not None else ModelConfig.load(DEFAULT_BASE).data
        bad = unknown_keys(schema, data)
        if bad:
            raise ValueError(f"fields not in the base configuration: {bad}")
        config = cls(data=deep_merge(schema, data))
        config.validate()
        return config

    def get(self, path: str) -> Any:
        """Value at a dotted path, for example ``train.lr``.

        Raises:
            KeyError: when the path does not exist.
        """
        node: Any = self.data
        for part in path.split("."):
            if not isinstance(node, Mapping) or part not in node:
                raise KeyError(f"no configuration field {path!r}")
            node = node[part]
        return node

    def with_values(self, values: Mapping[str, Any]) -> "ModelConfig":
        """A copy with dotted-path values replaced; ``model.depth`` and ``model.width`` set
        ``model.hidden`` together (the present one keeps the current value of the other).

        Raises:
            KeyError: for a path that does not exist.
        """
        data = copy.deepcopy(self.data)
        values = dict(values)
        depth = values.pop("model.depth", None)
        width = values.pop("model.width", None)
        if depth is not None or width is not None:
            hidden = list(data["model"]["hidden"])
            depth = int(depth if depth is not None else len(hidden))
            width = int(width if width is not None else (hidden[0] if hidden else 512))
            data["model"]["hidden"] = [width] * depth
        for path, value in values.items():
            parts = path.split(".")
            node = data
            for part in parts[:-1]:
                if part not in node or not isinstance(node[part], dict):
                    raise KeyError(f"no configuration field {path!r}")
                node = node[part]
            if parts[-1] not in node:
                raise KeyError(f"no configuration field {path!r}")
            node[parts[-1]] = value
        config = ModelConfig(data=data, sources=self.sources)
        config.validate()
        return config

    def validate(self) -> None:
        """Value checks beyond the key schema.

        Raises:
            ValueError: on the first invalid value.
        """
        d = self.data
        loss, loader, train = d["loss"], d["loader"], d["train"]
        if loss["reference"] not in REFERENCES:
            raise ValueError(f"loss.reference {loss['reference']!r} is not one of {REFERENCES}")
        if len(loss["phase_weights"]) != len(PHASES):
            raise ValueError(f"loss.phase_weights needs {len(PHASES)} values ({', '.join(PHASES)})")
        if any(float(w) < 0 for w in loss["phase_weights"]):
            raise ValueError("loss.phase_weights must be non-negative")
        unknown = [t for t in loss["terms"] if t not in LOSS_TERMS]
        if unknown:
            raise ValueError(f"unknown loss terms {unknown}")
        if not any(loss["terms"][t]["enabled"] for t in loss["terms"]):
            raise ValueError("no loss term is enabled")
        coeff = loss["terms"]["coeff"]
        if len(coeff["components"]) != len(coeff["component_weights"]):
            raise ValueError("loss.terms.coeff: components and component_weights differ in length")
        if int(loss["band_cells"]) < 1 or int(loss["norm_frames"]) < 1:
            raise ValueError("loss.band_cells and loss.norm_frames must be positive")
        if not 0.0 <= float(loader["thomson_mask_rate"]) <= 1.0:
            raise ValueError("loader.thomson_mask_rate must lie in [0, 1]")
        if not float(loader["clip_bound"]) > 0:
            raise ValueError("loader.clip_bound must be positive")
        if not 0.0 < float(loader["tail_quantile"]) < 1.0:
            raise ValueError("loader.tail_quantile must lie in (0, 1)")
        for key in ("batch_frames", "max_epochs", "patience"):
            if int(train[key]) < 1:
                raise ValueError(f"train.{key} must be positive")
        if int(train["warmup_epochs"]) < 0:
            raise ValueError("train.warmup_epochs must be non-negative")
        if not 0.0 <= float(d["model"]["dropout"]) < 1.0:
            raise ValueError("model.dropout must lie in [0, 1)")
        for key, spec in d["optuna"]["space"].items():
            if key not in VIRTUAL_SPACE_KEYS:
                self.get(key)
            if spec.get("type") not in SPACE_TYPES:
                raise ValueError(f"optuna.space.{key}: type must be one of {SPACE_TYPES}")

    def to_dict(self) -> dict:
        return copy.deepcopy(self.data)

    def hash(self) -> str:
        return hashlib.sha256(json.dumps(self.data, sort_keys=True).encode()).hexdigest()
