"""Transferable features of the plasma model, built from the frame table (``docs/FEATURES.md``).

Every feature is a registered function plus one entry of the configuration file
(``configs/features_v1.yaml``). A function declares the frame-table columns it reads; the
build checks them against the label guard before computing, and the function can read no
other column. A function returns one column, or several for an entry with ``mirror:
permute`` (the coil points). Features carry no units: currents are divided by Ip, lengths
by the machine's reference radii, times by the plasma's resistive time.

Allowed reads: ``frame_table.input_columns(machine, include_flags=True)`` plus the key
column ``time_ms`` (``READABLE_KEYS``). Shots are told apart by the key ``shot``.

Units inside the formulas: currents in A, pressure in Pa, flux in Wb/rad, lengths in m,
times in s, temperature in keV. Ip is the signed ``ip__mean``; magnitudes where stated.

``build_features`` gives the unscaled features of frame-table rows, ``mirror_features``
the up-down mirror of a feature table (sign changes and a permutation only, exact because
the coil pattern is the input), ``feature_matrix`` the scaled matrix a model reads.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

from fusion_eq import coil_pattern as cp, dataset, frame_table as ft
from fusion_eq.coil_flux import MU0, CoilScales, GreenTable, PointTable, adopted_scales
from fusion_eq.machines import MachineConfig, machine_config

A_PER_KA = 1000.0
PA_PER_KPA = 1000.0
KEV_PER_EV = 1e-3
MS_PER_S = 1000.0
NE_UNIT_M3 = 1e20
# Spitzer estimate of the current-spreading time, tau_R = 7.8 a0^2 Te^1.5 s with a0 in m and
# Te in keV (Coulomb logarithm 17, effective charge 1, first radial mode).
SPITZER_TAU_COEFF = 7.8

MIRROR_RULES = ("even", "odd", "permute")
SCALE_RULES = ("robust", "none")
SCALE_WITH = ("source", "target")
TRANSFORMS = ("log1p",)
REQUIRED_FIELDS = ("name", "function", "group", "mirror", "scale", "uses_future")
OPTIONAL_FIELDS = ("params", "scale_group", "scale_with", "transform", "note")
KEY_COLUMNS = ("shot", "frame")
READABLE_KEYS = ("time_ms",)

DEFAULT_CONFIG = dataset.REPO_ROOT / "configs" / "features_v1.yaml"
DEFAULT_FEATURE_ROOT = dataset.DEFAULT_PROCESSED_ROOT / "features"
DEFAULT_REPORT_ROOT = dataset.REPO_ROOT / "reports" / "features"
DEFAULT_GREEN_DIR = dataset.DEFAULT_PROCESSED_ROOT / "green"


# ------------------------------------------------------------------------ configuration
@dataclass(frozen=True)
class FeatureSpec:
    """One feature entry of the configuration (fields as in ``configs/features_v1.yaml``)."""

    name: str
    function: str
    group: str
    mirror: str
    scale: str
    uses_future: bool
    params: dict = field(default_factory=dict)
    scale_group: str | None = None
    scale_with: str = "source"
    transform: str | None = None
    note: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "FeatureSpec":
        """Validates and reads one entry.

        Raises:
            ValueError: for a missing required field, an unknown field or an invalid value.
        """
        missing = [k for k in REQUIRED_FIELDS if k not in d]
        if missing:
            raise ValueError(f"feature {d.get('name', '?')!r} lacks {missing}")
        unknown = [k for k in d if k not in REQUIRED_FIELDS + OPTIONAL_FIELDS]
        if unknown:
            raise ValueError(f"feature {d['name']!r} has unknown fields {unknown}")
        spec = cls(name=str(d["name"]), function=str(d["function"]), group=str(d["group"]), mirror=str(d["mirror"]),
                   scale=str(d["scale"]), uses_future=d["uses_future"], params=dict(d.get("params") or {}),
                   scale_group=d.get("scale_group"), scale_with=str(d.get("scale_with", "source")),
                   transform=d.get("transform"), note=str(d.get("note", "")))
        if not isinstance(spec.uses_future, bool):
            raise ValueError(f"feature {spec.name!r}: uses_future must be true or false")
        for value, allowed, what in ((spec.mirror, MIRROR_RULES, "mirror"), (spec.scale, SCALE_RULES, "scale"),
                                     (spec.scale_with, SCALE_WITH, "scale_with")):
            if value not in allowed:
                raise ValueError(f"feature {spec.name!r}: {what} {value!r} is not one of {allowed}")
        if spec.transform is not None and spec.transform not in TRANSFORMS:
            raise ValueError(f"feature {spec.name!r}: transform {spec.transform!r} is not one of {TRANSFORMS}")
        return spec

    def to_dict(self) -> dict:
        return {"name": self.name, "function": self.function, "group": self.group, "mirror": self.mirror,
                "scale": self.scale, "uses_future": self.uses_future, "params": dict(self.params),
                "scale_group": self.scale_group, "scale_with": self.scale_with, "transform": self.transform,
                "note": self.note}


@dataclass(frozen=True)
class FeatureConfig:
    """A feature version: layout, features and sets.

    Attributes:
        version: Version number, which names the output folder ``v<version>``.
        min_abs_ip_ka: Frames with ``abs(Ip)`` below this give NaN in features divided by Ip.
        layout: Sample points of the coil pattern.
        features: Feature entries in configuration order.
        sets: Set name to its feature names.
    """

    version: int
    min_abs_ip_ka: float
    layout: cp.Layout
    features: tuple[FeatureSpec, ...]
    sets: dict[str, tuple[str, ...]]

    @classmethod
    def from_dict(cls, d: dict) -> "FeatureConfig":
        config = cls(version=int(d["version"]), min_abs_ip_ka=float(d["min_abs_ip_ka"]),
                     layout=cp.Layout.from_config(d["layout"]),
                     features=tuple(FeatureSpec.from_dict(f) for f in d["features"]),
                     sets={k: tuple(v) for k, v in d.get("sets", {}).items()})
        config.validate()
        return config

    @classmethod
    def load(cls, path: Path | str = DEFAULT_CONFIG) -> "FeatureConfig":
        return cls.from_dict(yaml.safe_load(Path(path).read_text()))

    def validate(self) -> None:
        """Every function registered, names and columns unique, sets made of configured
        features, ``permute`` only on multi-column functions over a symmetric layout.

        Raises:
            ValueError: on the first problem found.
        """
        names = [s.name for s in self.features]
        if len(set(names)) != len(names):
            raise ValueError(f"repeated feature names: {sorted({n for n in names if names.count(n) > 1})}")
        for spec in self.features:
            if spec.function not in REGISTRY:
                raise ValueError(f"feature {spec.name!r}: function {spec.function!r} is not registered")
            multi = REGISTRY[spec.function].multi_output
            if (spec.mirror == "permute") != multi:
                raise ValueError(f"feature {spec.name!r}: mirror 'permute' goes with a multi-column function "
                                 f"and only with one")
        if any(s.mirror == "permute" for s in self.features):
            self.layout.mirror_index()
        columns = self.all_columns()
        if len(set(columns)) != len(columns):
            raise ValueError("feature columns repeat")
        for set_name, members in self.sets.items():
            unknown = [m for m in members if m not in names]
            if unknown:
                raise ValueError(f"set {set_name!r} lists unconfigured features {unknown}")

    def spec(self, name: str) -> FeatureSpec:
        for s in self.features:
            if s.name == name:
                return s
        raise KeyError(f"no feature {name!r} in the configuration")

    def columns_of(self, spec: FeatureSpec | str) -> list[str]:
        """Output columns of one entry: its name, or one per layout point for a multi-column one."""
        spec = self.spec(spec) if isinstance(spec, str) else spec
        return self.layout.column_names(spec.name) if REGISTRY[spec.function].multi_output else [spec.name]

    def all_columns(self) -> list[str]:
        return [c for s in self.features for c in self.columns_of(s)]

    def set_columns(self, set_name: str) -> list[str]:
        if set_name not in self.sets:
            raise KeyError(f"no set {set_name!r}; known: {sorted(self.sets)}")
        return [c for name in self.sets[set_name] for c in self.columns_of(name)]

    def column_specs(self) -> dict[str, FeatureSpec]:
        """Column name to the entry that produces it."""
        return {c: s for s in self.features for c in self.columns_of(s)}

    def to_dict(self) -> dict:
        return {"version": self.version, "min_abs_ip_ka": self.min_abs_ip_ka,
                "layout": {"points": self.layout.points.tolist(), "min_clearance_m": self.layout.min_clearance_m},
                "features": [s.to_dict() for s in self.features], "sets": {k: list(v) for k, v in self.sets.items()}}

    def hash(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()


# ----------------------------------------------------------------------------- registry
@dataclass(frozen=True)
class FeatureFunction:
    """A registered feature function.

    Attributes:
        name: Registry name (the ``function`` field of a configuration entry).
        compute: ``compute(ctx, **params)`` returning ``(rows,)``, or ``(rows, n_points)``
            for a multi-column function.
        reads: ``reads(cfg, params)``: the frame-table columns the function reads.
        multi_output: True for a function with one column per layout point.
    """

    name: str
    compute: Callable
    reads: Callable[[MachineConfig, dict], list[str]]
    multi_output: bool = False


REGISTRY: dict[str, FeatureFunction] = {}


def register(name: str, reads: Sequence[str] | Callable[[MachineConfig, dict], list[str]], multi_output: bool = False):
    """Decorator that registers a feature function under ``name`` with its declared reads."""
    reads_fn = reads if callable(reads) else (lambda cfg, params, _r=tuple(reads): list(_r))

    def decorate(fn):
        if name in REGISTRY:
            raise ValueError(f"feature function {name!r} is registered twice")
        REGISTRY[name] = FeatureFunction(name=name, compute=fn, reads=reads_fn, multi_output=multi_output)
        return fn
    return decorate


def allowed_reads(machine: str | MachineConfig) -> set[str]:
    """Columns a feature may read: inputs and flags of the frame table, and ``time_ms``."""
    return set(ft.input_columns(machine, include_flags=True)) | set(READABLE_KEYS)


def check_reads(columns: Iterable[str], machine: str | MachineConfig) -> list[str]:
    """The label guard of the features (check X2).

    Raises:
        ValueError: when a column is a label, or is not in ``allowed_reads``.
    """
    requested = list(columns)
    labels = [c for c in requested if ft.is_label_column(c)]
    if labels:
        raise ValueError(f"label columns cannot be feature inputs: {labels}")
    allowed = allowed_reads(machine)
    unknown = [c for c in requested if c not in allowed]
    if unknown:
        raise ValueError(f"not readable by a feature (input_columns with flags, or time_ms): {unknown}")
    return requested


def feature_reads(config: FeatureConfig, machine: str | MachineConfig) -> dict[str, list[str]]:
    """Feature name to the columns its function reads on a machine (the X2 record)."""
    cfg = machine if isinstance(machine, MachineConfig) else machine_config(machine)
    return {s.name: REGISTRY[s.function].reads(cfg, s.params) for s in config.features}


# ---------------------------------------------------------------------------- resources
@dataclass
class MachineResources:
    """What the features of one machine need besides the frame rows.

    Attributes:
        cfg: Machine configuration.
        r0_m, a0_m: Reference radii in metres.
        tf_turns: Toroidal-field turns.
        point_table: Coil flux at the layout points, the reference point last.
        scales: The machine's adopted coil model.
        record: Provenance written to the build record.
    """

    cfg: MachineConfig
    r0_m: float
    a0_m: float
    tf_turns: float
    point_table: PointTable | None = None
    scales: CoilScales | None = None
    record: dict = field(default_factory=dict)

    @classmethod
    def from_machine(cls, machine: str | MachineConfig, point_table: PointTable | None = None,
                     scales: CoilScales | None = None, record: dict | None = None) -> "MachineResources":
        cfg = machine if isinstance(machine, MachineConfig) else machine_config(machine)
        return cls(cfg=cfg, r0_m=cfg.major_radius.value, a0_m=cfg.minor_radius.value, tf_turns=cfg.tf_turns.value,
                   point_table=point_table, scales=scales, record=dict(record or {}))


def load_resources(machine: str, config: FeatureConfig, data_root: Path | str = dataset.DEFAULT_DATA_ROOT,
                   green_dir: Path | str = DEFAULT_GREEN_DIR, cache_dir: Path | str | None = None) -> MachineResources:
    """Point table (from the cache, or built from the geometry of one public test shot),
    adopted coil model and constants of a machine."""
    cfg = machine_config(machine)
    grid_table = GreenTable.load(Path(green_dir) / f"green_{cfg.file_prefix.split('_')[0]}.npz")
    table, table_record = cp.load_or_build_point_table(config.layout, cfg, grid_table, data_root,
                                                       green_dir if cache_dir is None else cache_dir)
    scales = adopted_scales(machine, table.columns)
    record = {"constants": cfg.constants_record(), "coil_model": scales.to_dict(), "point_table": table_record,
              "geometry_hash": table.geometry_hash, "points_hash": table.points_hash, "n_filaments": table.n_filaments}
    return MachineResources.from_machine(cfg, table, scales, record)


# ------------------------------------------------------------------------------ context
@dataclass
class FeatureContext:
    """Frame rows of one machine, sorted by shot and frame, with the resources and a cache
    shared by the functions of one build. ``column`` serves only declared reads."""

    frames: pd.DataFrame
    config: FeatureConfig
    resources: MachineResources
    cache: dict = field(default_factory=dict)
    declared: frozenset = frozenset()

    @property
    def cfg(self) -> MachineConfig:
        return self.resources.cfg

    def column(self, name: str) -> np.ndarray:
        """A declared column as float64 (booleans as 0 and 1)."""
        if name not in self.declared:
            raise ValueError(f"column {name!r} read without being declared by the feature function")
        return self.frames[name].to_numpy(dtype=np.float64)

    def shot_starts(self) -> np.ndarray:
        """Row index of the first frame of each shot, and the row count last."""
        if "shot_starts" not in self.cache:
            shots = self.frames["shot"].to_numpy()
            change = np.flatnonzero(shots[1:] != shots[:-1]) + 1
            self.cache["shot_starts"] = np.concatenate([[0], change, [len(shots)]])
        return self.cache["shot_starts"]

    def ip_a(self) -> np.ndarray:
        return self.column("ip__mean") * A_PER_KA

    def ip_ok(self) -> np.ndarray:
        """Frames whose Ip is finite and at least ``min_abs_ip_ka`` in size."""
        ip = self.column("ip__mean")
        return np.isfinite(ip) & (np.abs(ip) >= self.config.min_abs_ip_ka) & (ip != 0)

    def thomson_ok(self) -> np.ndarray:
        return self.column("ts_core_valid") > 0


# ------------------------------------------------------------------------ shared pieces
def coil_reads(cfg: MachineConfig, params: dict | None = None) -> list[str]:
    """Ip and the ``__mean`` of every current with coil geometry (solenoid, shaping coils)."""
    return ["ip__mean", "sol__mean", *[f"{c}__mean" for c in cfg.shaping_coil_columns]]


def frame_current_column(cfg: MachineConfig, table_column: str) -> str:
    return "sol__mean" if table_column == cfg.solenoid_column else f"{table_column}__mean"


def _coil_values(ctx: FeatureContext) -> np.ndarray:
    """``coil_pt`` of every row, ``(rows, n_points)``; NaN on rows with a missing current or
    an Ip too small to divide by."""
    if "coil_pt" not in ctx.cache:
        res = ctx.resources
        if res.point_table is None:
            raise ValueError("coil features need a point table")
        currents = np.column_stack([ctx.column(frame_current_column(ctx.cfg, c)) for c in res.point_table.columns])
        values = cp.coil_point_values(res.point_table, currents, ctx.column("ip__mean"), res.r0_m, res.scales,
                                      ctx.config.min_abs_ip_ka)
        values[~np.isfinite(currents).all(axis=1)] = np.nan
        ctx.cache["coil_pt"] = values
    return ctx.cache["coil_pt"]


def _coil_terms(ctx: FeatureContext) -> np.ndarray:
    """The nine named terms of every row, ``(rows, 9)`` in ``coil_pattern.TERMS`` order."""
    if "coil_terms" not in ctx.cache:
        ctx.cache["coil_terms"] = _coil_values(ctx) @ cp.term_operator(ctx.config.layout).T
    return ctx.cache["coil_terms"]


THOMSON_TE_READS = ("ts_core_te_peak__interp", "ts_core_valid")


def _tau_r(ctx: FeatureContext) -> np.ndarray:
    """Resistive time in s per row from the core Te peak; frames without valid Thomson take
    the shot's median Te over its valid frames, and a shot with none gives NaN."""
    if "tau_r" not in ctx.cache:
        te = ctx.column("ts_core_te_peak__interp") * KEV_PER_EV
        valid = ctx.thomson_ok() & np.isfinite(te) & (te > 0)
        te_valid = np.where(valid, te, np.nan)
        median = pd.Series(te_valid).groupby(ctx.frames["shot"].to_numpy()).transform("median").to_numpy()
        te_used = np.where(valid, te, median)
        ctx.cache["tau_r"] = SPITZER_TAU_COEFF * ctx.resources.a0_m ** 2 * te_used ** 1.5
    return ctx.cache["tau_r"]


# ----------------------------------------------------------------- group 1: coil pattern
@register("coil_points", coil_reads, multi_output=True)
def coil_points(ctx: FeatureContext) -> np.ndarray:
    """Coil flux at the layout points relative to the reference point, over ``mu0 R0 Ip``."""
    return _coil_values(ctx)


@register("coil_term", coil_reads)
def coil_term(ctx: FeatureContext, term: str) -> np.ndarray:
    """One named term ``c_mn`` of the order-3 polynomial fitted to the coil points."""
    if term not in cp.TERM_NAMES:
        raise ValueError(f"unknown term {term!r}; known: {cp.TERM_NAMES}")
    return _coil_terms(ctx)[:, cp.TERM_NAMES.index(term)]


# --------------------------------------------------------------- group 2: plasma content
@register("ts_beta_p", ("ip__mean", "ts_core_pe_mean__interp", "ts_core_valid"))
def ts_beta_p(ctx: FeatureContext) -> np.ndarray:
    """Electron pressure against the poloidal field pressure, ``8 pi^2 a0^2 p_e / (mu0 Ip^2)``."""
    p_pa = ctx.column("ts_core_pe_mean__interp") * PA_PER_KPA
    ip_a = ctx.ip_a()
    with np.errstate(divide="ignore", invalid="ignore"):
        value = 8.0 * np.pi ** 2 * ctx.resources.a0_m ** 2 * p_pa / (MU0 * ip_a ** 2)
    return np.where(ctx.ip_ok() & ctx.thomson_ok(), value, np.nan)


@register("ts_pe_peaking", ("ts_core_pe_peak__interp", "ts_core_pe_mean__interp", "ts_core_valid"))
def ts_pe_peaking(ctx: FeatureContext) -> np.ndarray:
    """Peak-to-average ratio of the core electron pressure."""
    peak, mean = ctx.column("ts_core_pe_peak__interp"), ctx.column("ts_core_pe_mean__interp")
    with np.errstate(divide="ignore", invalid="ignore"):
        value = peak / mean
    return np.where(ctx.thomson_ok() & (mean > 0), value, np.nan)


@register("ts_valid", ("ts_core_valid",))
def ts_valid(ctx: FeatureContext) -> np.ndarray:
    """Core Thomson valid at the frame, 0 or 1."""
    return ctx.thomson_ok().astype(np.float64)


# ---------------------------------------------------------- group 3: field against current
@register("q_cyl", ("ip__mean", "tf__mean"))
def q_cyl(ctx: FeatureContext) -> np.ndarray:
    """Cylindrical safety factor ``a0^2 N_tf |I_tf| / (R0^2 |Ip|)`` (both currents in kA)."""
    res = ctx.resources
    with np.errstate(divide="ignore", invalid="ignore"):
        value = res.a0_m ** 2 * res.tf_turns * np.abs(ctx.column("tf__mean")) / (res.r0_m ** 2 * np.abs(ctx.column("ip__mean")))
    return np.where(ctx.ip_ok(), value, np.nan)


# ----------------------------------------------------------- group 4: position in the shot
@register("ip_frac", ("ip_over_peak",))
def ip_frac(ctx: FeatureContext) -> np.ndarray:
    """``ip_over_peak``: abs(Ip) over the shot's 99.5th percentile of abs(Ip)."""
    return ctx.column("ip_over_peak")


@register("ip_rate_rel", ("ip__mean", "ip__rate50", *THOMSON_TE_READS))
def ip_rate_rel(ctx: FeatureContext) -> np.ndarray:
    """Relative rate of change of Ip in units of the resistive time, ``(dIp/dt / Ip) tau_R``."""
    with np.errstate(divide="ignore", invalid="ignore"):
        value = ctx.column("ip__rate50") / ctx.column("ip__mean") * _tau_r(ctx)
    return np.where(ctx.ip_ok(), value, np.nan)


@register("resistive_age", ("time_ms", "ip_rise_ms", *THOMSON_TE_READS))
def resistive_age(ctx: FeatureContext) -> np.ndarray:
    """Integral of ``dt / tau_R`` from the Ip rise to the frame: a trapezoid over the shot's
    frames, starting from ``(first frame time - ip_rise_ms) / tau_R`` at the first frame."""
    t_s = ctx.column("time_ms") / MS_PER_S
    rise_s = ctx.column("ip_rise_ms") / MS_PER_S
    inv = 1.0 / _tau_r(ctx)
    out = np.full(t_s.size, np.nan)
    starts = ctx.shot_starts()
    for a, b in zip(starts[:-1], starts[1:]):
        steps = 0.5 * (inv[a + 1:b] + inv[a:b - 1]) * np.diff(t_s[a:b])
        out[a:b] = (t_s[a] - rise_s[a]) * inv[a] + np.concatenate([[0.0], np.cumsum(steps)])
    return out


# ----------------------------------------------------------------------------- candidates
@register("shafranov_lambda", coil_reads)
def shafranov_lambda(ctx: FeatureContext) -> np.ndarray:
    """Force balance, ``-4 pi (R0 / a0) c10 - ln(8 R0 / a0) + 1.5``: an estimate of
    beta_p + li / 2 from the coils' vertical field (large-aspect-ratio form)."""
    res = ctx.resources
    c10 = _coil_terms(ctx)[:, cp.TERM_NAMES.index("c10")]
    return -4.0 * np.pi * (res.r0_m / res.a0_m) * c10 - np.log(8.0 * res.r0_m / res.a0_m) + 1.5


@register("ohmic_flux_used", ("ip__mean", "sol__mean", "sol_at_rise"))
def ohmic_flux_used(ctx: FeatureContext) -> np.ndarray:
    """Solenoid flux at the reference point spent since the Ip rise, over ``mu0 R0 Ip``,
    under the machine's solenoid model (scale and flat term)."""
    res = ctx.resources
    table = res.point_table
    if table is None:
        raise ValueError("ohmic_flux_used needs a point table")
    delta = ctx.column("sol__mean") - ctx.column("sol_at_rise")
    currents = np.zeros((delta.size, len(table.columns)))
    currents[:, table.column_index([ctx.cfg.solenoid_column])[0]] = delta
    psi_ref = np.atleast_2d(table.flux(currents, res.scales))[:, -1]
    with np.errstate(divide="ignore", invalid="ignore"):
        value = psi_ref / (MU0 * res.r0_m * ctx.ip_a())
    return np.where(ctx.ip_ok() & np.isfinite(delta), value, np.nan)


@register("greenwald_frac", ("ip__mean", "ts_core_ne_peak__interp", "ts_core_valid"))
def greenwald_frac(ctx: FeatureContext) -> np.ndarray:
    """Density against current and size, ``n_e pi a0^2 / |Ip|``, n_e in 1e20 m^-3, Ip in MA."""
    ne = ctx.column("ts_core_ne_peak__interp") / NE_UNIT_M3
    ip_ma = np.abs(ctx.column("ip__mean")) / A_PER_KA
    with np.errstate(divide="ignore", invalid="ignore"):
        value = ne * np.pi * ctx.resources.a0_m ** 2 / ip_ma
    return np.where(ctx.ip_ok() & ctx.thomson_ok(), value, np.nan)


# --------------------------------------------------------------------------------- build
def build_features(frames: pd.DataFrame, machine: str | MachineConfig, config: FeatureConfig,
                   resources: MachineResources | None = None, names: Iterable[str] | None = None) -> pd.DataFrame:
    """Unscaled features of frame-table rows of one machine.

    Args:
        frames: Rows with ``shot``, ``frame`` and every column the features read; any order.
        machine: Machine of the rows.
        config: Feature configuration.
        resources: Point table, coil model and constants; ``MachineResources.from_machine``
            without a point table when None (only features without coil flux then work).
        names: Features to build; all configured features when None.

    Returns:
        ``shot``, ``frame`` and the feature columns (float64), sorted by shot and frame.

    Raises:
        ValueError: when a feature declares a column that is not readable (label guard),
            or rows repeat a (shot, frame) key.
        KeyError: when a declared column is missing from ``frames``.
    """
    cfg = machine if isinstance(machine, MachineConfig) else machine_config(machine)
    resources = resources or MachineResources.from_machine(cfg)
    specs = config.features if names is None else tuple(config.spec(n) for n in names)
    reads = {s.name: check_reads(REGISTRY[s.function].reads(cfg, s.params), cfg) for s in specs}
    needed = sorted({c for cols in reads.values() for c in cols})
    missing = [c for c in needed if c not in frames.columns]
    if missing:
        raise KeyError(f"frame rows lack the columns {missing}")
    rows = frames[list(KEY_COLUMNS) + needed].copy()
    rows["shot"] = rows["shot"].astype(str)
    rows = rows.sort_values(list(KEY_COLUMNS), kind="stable").reset_index(drop=True)
    if rows.duplicated(list(KEY_COLUMNS)).any():
        raise ValueError("frame rows repeat a (shot, frame) key")
    ctx = FeatureContext(frames=rows, config=config, resources=resources)
    out = {"shot": rows["shot"].to_numpy(), "frame": rows["frame"].to_numpy()}
    for spec in specs:
        ctx.declared = frozenset(reads[spec.name])
        values = np.asarray(REGISTRY[spec.function].compute(ctx, **spec.params), dtype=np.float64)
        columns = config.columns_of(spec)
        values = values.reshape(len(rows), len(columns))
        for j, column in enumerate(columns):
            out[column] = values[:, j]
    return pd.DataFrame(out)


def mirror_features(features: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
    """The up-down mirror of a feature table: ``odd`` columns change sign, ``permute``
    columns are exchanged with their mirror partners, every other column is unchanged.

    Raises:
        ValueError: when only part of a ``permute`` entry's columns is present.
    """
    out = features.copy()
    for spec in config.features:
        columns = [c for c in config.columns_of(spec) if c in out.columns]
        if not columns or spec.mirror == "even":
            continue
        if spec.mirror == "odd":
            out[columns] = -features[columns].to_numpy()
            continue
        if len(columns) != len(config.columns_of(spec)):
            raise ValueError(f"feature {spec.name!r}: mirroring needs all of its columns")
        partner = config.layout.mirror_index()
        out[columns] = features[columns].to_numpy()[:, partner]
    return out


# ---------------------------------------------------------------------------- reading
def features_dir(version: int, root: Path | str = DEFAULT_FEATURE_ROOT) -> Path:
    return Path(root) / f"v{version}"


def features_path(split: str, version: int, root: Path | str = DEFAULT_FEATURE_ROOT) -> Path:
    """Unscaled features of one split: ``<root>/v<version>/<split>.parquet``."""
    return features_dir(version, root) / f"{split}.parquet"


def read_features(split: str, version: int = 1, columns: Iterable[str] | None = None, shots: Iterable[str] | None = None,
                  root: Path | str = DEFAULT_FEATURE_ROOT) -> pd.DataFrame:
    """Rows of one split's feature table (``shot``, ``frame`` and the columns), in file
    order: shots sorted by id, frames in order."""
    cols = None if columns is None else list(dict.fromkeys([*KEY_COLUMNS, *columns]))
    filters = None if shots is None else [("shot", "in", sorted({str(s) for s in shots}))]
    return pq.read_table(features_path(split, version, root), columns=cols, filters=filters).to_pandas()


def scaler_path(version: int, report_root: Path | str = DEFAULT_REPORT_ROOT) -> Path:
    return Path(report_root) / f"v{version}" / "scaler.json"


def feature_matrix(split: str, shots: Iterable[str] | None, set_name: str, mirror: bool = False,
                   config: FeatureConfig | None = None, scaler=None, root: Path | str = DEFAULT_FEATURE_ROOT,
                   report_root: Path | str = DEFAULT_REPORT_ROOT) -> tuple[np.ndarray, list[str]]:
    """The scaled matrix a model reads.

    Args:
        split: Data split; its machine selects the scaling.
        shots: Shot ids, or None for the whole split.
        set_name: Feature set of the configuration.
        mirror: Whether to return the mirrored frames (mirrored before scaling).
        config: Feature configuration; the default file when None.
        scaler: A fitted ``feature_scaler.RobustScaler``; ``reports/features/v<version>/
            scaler.json`` when None.

    Returns:
        ``(X float32 (rows, columns), columns)``, rows in file order.
    """
    from fusion_eq.feature_scaler import RobustScaler

    config = config or FeatureConfig.load()
    scaler = scaler or RobustScaler.load(scaler_path(config.version, report_root))
    columns = config.set_columns(set_name)
    table = read_features(split, config.version, columns, shots, root)
    if mirror:
        table = mirror_features(table, config)
    x = scaler.transform(table, dataset.SPLIT_MACHINE[split], columns)
    return x.astype(np.float32), columns
