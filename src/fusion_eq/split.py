"""The fixed split of the DIII-D training shots: drawing, saving, reading and checking.

A split assigns every training shot to exactly one of ``train``, ``validation`` and
``local_test`` and draws two working subsets inside the first two parts. The draw is
stratified by recording rate and plasma-current direction so that each part mirrors
the test shots, and reproducible for a seed. In ``grouped`` mode whole near-repeat
groups (``fusion_eq.similarity.group_shots``) are assigned, so no group spans two
parts. Specification: ``docs/SPLIT.md``, analysis A5 and checks S5, S6.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

PARTS = ("train", "validation", "local_test")
WORKING = {"working_train": "train", "working_validation": "validation"}
STRATA = ("mag_rate_hz", "ip_direction")
MODES = ("random", "grouped")
TRAIN_SPLIT = "diii_d_train"
TEST_SPLIT = "diii_d_public_test"


@dataclass(frozen=True)
class SplitSizes:
    """Shot counts of the parts that are drawn; ``train`` takes the rest."""

    n_validation: int = 870
    n_local_test: int = 350
    n_working_train: int = 300
    n_working_validation: int = 100

    def to_dict(self) -> dict:
        return asdict(self)


# ------------------------------------------------------------------------------ draw
def allocate(counts: Mapping, total: int) -> dict:
    """Largest-remainder apportionment of ``total`` over strata in proportion to ``counts``.

    Every stratum gets at most its own count. Raises when ``total`` exceeds the sum.
    """
    keys = list(counts)
    sizes = np.array([counts[k] for k in keys], dtype=np.float64)
    if total > sizes.sum():
        raise ValueError(f"cannot allocate {total} shots over {int(sizes.sum())}")
    if total == 0 or sizes.sum() == 0:
        return {k: 0 for k in keys}
    exact = sizes * total / sizes.sum()
    floors = np.floor(exact).astype(int)
    remainder = total - floors.sum()
    # A stratum only gains the extra unit when its share has a fractional part, so the
    # result never exceeds the stratum's own count.
    order = np.argsort(-(exact - floors), kind="stable")
    for i in order[:remainder]:
        floors[i] += 1
    return {k: int(v) for k, v in zip(keys, floors)}


def _units(table: pd.DataFrame, groups: Mapping | None) -> list[tuple[tuple, list[str]]]:
    """(stratum, members) per unit: one shot, or one whole group in grouped mode.

    A group's stratum is that of its first member in sorted id order. Units are in a
    fixed order (sorted by first member) before any shuffling.
    """
    ids = sorted(table["shot"].astype(str))
    strata = {s: tuple(table.loc[table["shot"].astype(str) == s, list(STRATA)].iloc[0].tolist()) for s in ids}
    if groups is None:
        return [(strata[s], [s]) for s in ids]
    members: dict = {}
    for s in ids:
        members.setdefault(groups[s], []).append(s)
    units = sorted(members.values(), key=lambda m: m[0])
    return [(strata[m[0]], m) for m in units]


def _take(units: list[tuple[tuple, list[str]]], quota: Mapping, rng: np.random.Generator) -> tuple[list[str], list]:
    """Fills the per-stratum quota with whole units in a shuffled order; skips a unit
    that would overshoot its stratum's remaining quota. Whatever the strata could not
    fill with whole units is then filled from the leftover units of any stratum, in the
    same shuffled order, so that the total is reached whenever whole units allow it.
    Returns the chosen shot ids and the units left over."""
    remaining = dict(quota)
    by_stratum: dict = {}
    for i, (stratum, _) in enumerate(units):
        by_stratum.setdefault(stratum, []).append(i)
    chosen, used, orders = [], set(), []
    for stratum in sorted(by_stratum):
        order = rng.permutation(by_stratum[stratum])
        orders.append(order)
        for i in order:
            size = len(units[i][1])
            if size <= remaining.get(stratum, 0):
                chosen.extend(units[i][1])
                remaining[stratum] -= size
                used.add(i)
    shortfall = sum(remaining.values())
    if shortfall > 0:
        leftover = [i for order in orders for i in order if i not in used]
        for i in rng.permutation(leftover) if leftover else []:
            size = len(units[i][1])
            if size <= shortfall:
                chosen.extend(units[i][1])
                shortfall -= size
                used.add(i)
    rest = [u for i, u in enumerate(units) if i not in used]
    return chosen, rest


def draw_split(shot_table: pd.DataFrame, sizes: SplitSizes, seed: int, groups: Mapping | None = None) -> dict[str, list[str]]:
    """Draws the parts and the working subsets.

    Args:
        shot_table: One row per training shot with ``shot`` and the strata columns
            (``mag_rate_hz``, ``ip_direction``).
        sizes: Counts of the drawn parts.
        seed: Seed of the generator; the same seed gives the same split.
        groups: Shot id to group id; when given, whole groups are assigned to one part
            (grouped mode). None is random mode.

    Returns:
        Sorted shot-id lists keyed ``train``, ``validation``, ``local_test``,
        ``working_train``, ``working_validation``. In random mode the counts equal the
        requested sizes exactly; in grouped mode a part may fall short when no
        combination of whole groups fills a stratum, which ``check_split`` reports.
    """
    rng = np.random.default_rng(seed)
    units = _units(shot_table, groups)
    stratum_counts: dict = {}
    for stratum, members in units:
        stratum_counts[stratum] = stratum_counts.get(stratum, 0) + len(members)
    out: dict[str, list[str]] = {}
    pool = units
    for part, n in (("validation", sizes.n_validation), ("local_test", sizes.n_local_test)):
        out[part], pool = _take(pool, allocate(stratum_counts, n), rng)
        for stratum, members in units:
            if members[0] in out[part]:
                stratum_counts[stratum] -= len(members)
    out["train"] = [s for _, members in pool for s in members]
    for working, parent in WORKING.items():
        n = getattr(sizes, f"n_{working}")
        parent_units = [u for u in units if u[1][0] in set(out[parent])]
        counts: dict = {}
        for stratum, members in parent_units:
            counts[stratum] = counts.get(stratum, 0) + len(members)
        out[working], _ = _take(parent_units, allocate(counts, min(n, len(out[parent]))), rng)
    return {k: sorted(v) for k, v in out.items()}


# ------------------------------------------------------------------------- file
def split_record(parts: Mapping[str, list[str]], mode: str, threshold: float | None, seed: int, sizes: SplitSizes,
                 repo_commit: str | None, groups: Mapping | None = None,
                 near_repeat_in_train: Mapping | None = None) -> dict:
    """The JSON record of a split: lists, parameters, provenance and per-shot details."""
    shots = sorted(s for p in PARTS for s in parts[p])
    return {
        "mode": mode, "threshold": threshold, "seed": seed, "sizes": sizes.to_dict(), "repo_commit": repo_commit,
        **{p: list(parts[p]) for p in (*PARTS, *WORKING)},
        "shots": {s: {"group": None if groups is None else int(groups[s]),
                      "near_repeat_in_train": None if near_repeat_in_train is None else bool(near_repeat_in_train[s])}
                  for s in shots},
    }


def write_split(path: Path | str, record: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(record, indent=1, sort_keys=False) + "\n")


def read_split(path: Path | str) -> dict:
    """Reads a split file and checks that the five lists are present."""
    record = json.loads(Path(path).read_text())
    missing = [k for k in (*PARTS, *WORKING) if k not in record]
    if missing:
        raise ValueError(f"{path}: missing lists {missing}")
    return record


# ------------------------------------------------------------------------ checks
def check_split(split: Mapping, shot_index: pd.DataFrame, sizes: SplitSizes | None = None,
                groups: Mapping | None = None) -> dict:
    """Check S5: integrity of a split against the frame table's shot index.

    Raises:
        ValueError: when a shot is in two parts, a training shot is unassigned, a test
            shot (or an unknown id) is present, a working subset leaves its parent, the
            counts differ from ``sizes`` (when given), or a group spans two parts (when
            ``groups`` is given).

    Returns:
        The counts per list.
    """
    train_ids = set(shot_index.loc[shot_index["split"] == TRAIN_SPLIT, "shot"].astype(str))
    problems = []
    lists = {k: list(split[k]) for k in (*PARTS, *WORKING)}
    seen: dict[str, str] = {}
    for part in PARTS:
        for s in lists[part]:
            if s in seen:
                problems.append(f"{s} is in both {seen[s]} and {part}")
            seen[s] = part
    assigned = set(seen)
    unassigned = sorted(train_ids - assigned)
    if unassigned:
        problems.append(f"{len(unassigned)} training shots unassigned, first {unassigned[:5]}")
    foreign = sorted(assigned - train_ids)
    if foreign:
        problems.append(f"{len(foreign)} shots are not training shots (test or unknown), first {foreign[:5]}")
    for working, parent in WORKING.items():
        outside = sorted(set(lists[working]) - set(lists[parent]))
        if outside:
            problems.append(f"{working}: {len(outside)} shots outside {parent}")
        if len(set(lists[working])) != len(lists[working]):
            problems.append(f"{working}: duplicated shots")
    counts = {k: len(v) for k, v in lists.items()}
    if sizes is not None:
        for name, n in (("validation", sizes.n_validation), ("local_test", sizes.n_local_test),
                        ("working_train", sizes.n_working_train), ("working_validation", sizes.n_working_validation)):
            if counts[name] != n:
                problems.append(f"{name}: {counts[name]} shots, {n} requested")
    if groups is not None:
        part_of_group: dict = {}
        for part in PARTS:
            for s in lists[part]:
                g = groups[s]
                if g in part_of_group and part_of_group[g] != part:
                    problems.append(f"group {g} spans {part_of_group[g]} and {part}")
                part_of_group[g] = part
    if problems:
        raise ValueError("split check failed: " + "; ".join(sorted(set(problems))))
    return counts


def split_balance(split: Mapping, shot_index: pd.DataFrame) -> pd.DataFrame:
    """Check S6: per list and for the test shots, the share recorded at 20 kHz, the share
    with negative Ip, the median frames per shot and the share with no valid core
    Thomson profile."""
    idx = shot_index.set_index(shot_index["shot"].astype(str))
    rows = []
    sets = {k: list(split[k]) for k in (*PARTS, *WORKING)}
    sets["test"] = idx.loc[idx["split"] == TEST_SPLIT, "shot"].astype(str).tolist()
    for name, ids in sets.items():
        if not ids:
            rows.append({"list": name, "shots": 0, "share_20khz": np.nan, "share_negative_ip": np.nan,
                         "median_frames": np.nan, "share_no_core_thomson": np.nan})
            continue
        sub = idx.loc[ids]
        rows.append({"list": name, "shots": len(ids), "share_20khz": float((sub["mag_rate_hz"] == 20000).mean()),
                     "share_negative_ip": float((sub["ip_direction"] == -1).mean()),
                     "median_frames": float(sub["n_frames"].median()),
                     "share_no_core_thomson": float((~sub["ts_core_any_valid"].fillna(False).astype(bool)).mean())})
    return pd.DataFrame(rows)


def part_of(split: Mapping, shot_ids: Iterable[str]) -> list[str]:
    """The part (``train``, ``validation``, ``local_test``) of each shot id, or ``""``."""
    lookup = {s: p for p in PARTS for s in split[p]}
    return [lookup.get(s, "") for s in shot_ids]
