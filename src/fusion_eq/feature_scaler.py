"""Robust scaling of the features: median and interquartile range (``docs/FEATURES.md``).

The statistics are fitted on the DIII-D training frames, originals and mirrors together,
and applied unchanged to every split and machine. Features of one ``scale_group`` share
one median and one interquartile range, computed over all their values pooled, so that
a pattern keeps its shape. An optional ``transform`` (``log1p``) is applied before the
statistics and before scaling. A feature declared ``scale_with: target`` is scaled with
the statistics of the machine's own inputs, fitted by ``fit_target``; no version-1
feature does so. After scaling a missing value becomes 0, the training median.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from fusion_eq.features import FeatureConfig

TRANSFORM_FUNCTIONS = {None: lambda v: v, "log1p": np.log1p}


@dataclass
class RobustScaler:
    """Fitted statistics of the feature columns.

    Attributes:
        columns: Columns in fit order.
        rules: Column to ``{"scale", "scale_group", "scale_with", "transform"}``.
        source: Column to ``[median, interquartile range]`` on the source machine.
        target: Machine to column to ``[median, interquartile range]``, for ``scale_with:
            target`` columns.
        source_machine: Machine of the source statistics.
        n_rows: Rows of the fit (originals plus mirrors).
    """

    columns: tuple[str, ...]
    rules: dict[str, dict]
    source: dict[str, list[float]]
    target: dict[str, dict[str, list[float]]] = field(default_factory=dict)
    source_machine: str = "DIII-D"
    n_rows: int = 0

    @staticmethod
    def _rules(config: FeatureConfig, columns) -> dict[str, dict]:
        specs = config.column_specs()
        unknown = [c for c in columns if c not in specs]
        if unknown:
            raise KeyError(f"columns not in the configuration: {unknown}")
        return {c: {"scale": specs[c].scale, "scale_group": specs[c].scale_group, "scale_with": specs[c].scale_with,
                    "transform": specs[c].transform} for c in columns}

    @staticmethod
    def _groups(rules: dict[str, dict], columns) -> dict[str, list[str]]:
        """Scaled columns by their pooling group (a column alone unless it has a scale group)."""
        groups: dict[str, list[str]] = {}
        for c in columns:
            if rules[c]["scale"] == "robust":
                groups.setdefault(rules[c]["scale_group"] or c, []).append(c)
        return groups

    @staticmethod
    def _statistics(values: np.ndarray, label: str) -> list[float]:
        v = values[np.isfinite(values)]
        if v.size == 0:
            raise ValueError(f"no finite value to scale {label}")
        q25, q50, q75 = np.percentile(v, [25.0, 50.0, 75.0])
        iqr = float(q75 - q25)
        if not iqr > 0:
            raise ValueError(f"zero interquartile range for {label}")
        return [float(q50), iqr]

    @classmethod
    def fit(cls, original: pd.DataFrame, mirrored: pd.DataFrame, config: FeatureConfig, columns=None,
            source_machine: str = "DIII-D") -> "RobustScaler":
        """Fits the source statistics on originals and mirrors together.

        Args:
            original: Feature table of the training frames.
            mirrored: ``mirror_features`` of the same table.
            config: Feature configuration (scaling rules per column).
            columns: Columns to fit; every configured column present in ``original`` when None.

        Raises:
            ValueError: for a zero interquartile range or a column without finite values.
        """
        columns = tuple(c for c in config.all_columns() if c in original.columns) if columns is None else tuple(columns)
        return cls.fit_rules(original, mirrored, cls._rules(config, columns), source_machine)

    @classmethod
    def fit_rules(cls, original: pd.DataFrame, mirrored: pd.DataFrame, rules: dict[str, dict],
                  source_machine: str = "DIII-D") -> "RobustScaler":
        """Fits the source statistics on originals and mirrors together, with explicit rules.

        Serves in-memory variants of a configuration (columns outside it, changed scaling).

        Args:
            original: Feature table of the training frames.
            mirrored: Mirror of the same table.
            rules: Column to ``{"scale", "scale_group", "scale_with", "transform"}``, in the
                order of the scaler's columns.

        Raises:
            ValueError: for a zero interquartile range or a column without finite values.
        """
        columns = tuple(rules)
        rules = {c: dict(rules[c]) for c in columns}
        source: dict[str, list[float]] = {}
        for group, members in cls._groups(rules, columns).items():
            transform = TRANSFORM_FUNCTIONS[rules[members[0]]["transform"]]
            pooled = np.concatenate([transform(frame[c].to_numpy(dtype=np.float64))
                                     for frame in (original, mirrored) for c in members])
            stats = cls._statistics(pooled, f"{group} ({len(members)} columns)")
            for c in members:
                source[c] = list(stats)
        for c in columns:
            if rules[c]["scale"] == "none":
                source[c] = [0.0, 1.0]
        return cls(columns=columns, rules=rules, source=source, source_machine=source_machine,
                   n_rows=int(len(original) + len(mirrored)))

    def fit_target(self, machine: str, features: pd.DataFrame) -> None:
        """Statistics of the ``scale_with: target`` columns on a machine's own input frames."""
        stats: dict[str, list[float]] = {}
        target_columns = [c for c in self.columns if self.rules[c]["scale_with"] == "target"]
        for group, members in self._groups({c: self.rules[c] for c in target_columns}, target_columns).items():
            transform = TRANSFORM_FUNCTIONS[self.rules[members[0]]["transform"]]
            pooled = np.concatenate([transform(features[c].to_numpy(dtype=np.float64)) for c in members])
            values = self._statistics(pooled, f"{group} on {machine}")
            for c in members:
                stats[c] = list(values)
        self.target[machine] = stats

    def statistics(self, column: str, machine: str) -> list[float]:
        """``[median, interquartile range]`` used for a column on a machine."""
        if self.rules[column]["scale_with"] == "target" and machine != self.source_machine:
            if column not in self.target.get(machine, {}):
                raise KeyError(f"no {machine} statistics for the target-scaled column {column!r}")
            return self.target[machine][column]
        return self.source[column]

    def transform(self, features: pd.DataFrame, machine: str, columns=None) -> np.ndarray:
        """Scaled values ``(rows, columns)`` in float64; missing values become 0."""
        columns = self.columns if columns is None else tuple(columns)
        out = np.empty((len(features), len(columns)))
        for j, c in enumerate(columns):
            center, scale = self.statistics(c, machine)
            values = TRANSFORM_FUNCTIONS[self.rules[c]["transform"]](features[c].to_numpy(dtype=np.float64))
            out[:, j] = (values - center) / scale
        return np.where(np.isfinite(out), out, 0.0)

    def to_dict(self) -> dict:
        return {"columns": list(self.columns), "rules": self.rules, "source": self.source, "target": self.target,
                "source_machine": self.source_machine, "n_rows": self.n_rows}

    @classmethod
    def from_dict(cls, d: dict) -> "RobustScaler":
        return cls(columns=tuple(d["columns"]), rules={k: dict(v) for k, v in d["rules"].items()},
                   source={k: [float(x) for x in v] for k, v in d["source"].items()},
                   target={m: {k: [float(x) for x in v] for k, v in s.items()} for m, s in d.get("target", {}).items()},
                   source_machine=str(d.get("source_machine", "DIII-D")), n_rows=int(d.get("n_rows", 0)))

    def save(self, path: Path | str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=1, sort_keys=True) + "\n")

    @classmethod
    def load(cls, path: Path | str) -> "RobustScaler":
        return cls.from_dict(json.loads(Path(path).read_text()))
