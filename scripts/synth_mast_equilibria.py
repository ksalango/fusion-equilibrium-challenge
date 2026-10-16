"""Synthetic MAST equilibria from the FreeGS free-boundary solver (``fusion_eq.synthetic_mast``).

The machine is built from the dataset's MAST coil geometry (812 elements, one FreeGS coil per
current column) under the adopted coil model (``sol_half`` by default: every solenoid element at
half the feed current). Targets cover the published MAST operating range in a scrambled Sobol
design per topology (lower single null and double null): Ip, toroidal field from the TF
conductor current (24 conductors), solenoid current (held fixed), boundary centre, minor
radius, elongation, triangularity, poloidal beta and current-profile peaking. The P-coil
currents come from FreeGS's constraint controller, with P3 and P6 at zero and P2, P4 and P5
bounded to the range of the MAST test inputs (``--no-current-limits`` frees them). No MAST
equilibrium label is read: geometry and grid come from the ``coil_*`` and ``efit_grid_*``
columns of one MAST test file.

Writes to ``--out`` (``../data/processed/synthetic/mast_v1/``):

- ``eq_<index>.npz`` per converged target (``synthetic_mast.equilibrium_record``): psi and
  FreeGS's vacuum psi on the native 65 by 65 grid, the coil currents FreeGS found in the
  dataset's units, Ip, beta_p, li, q95, betaN, the achieved shape numbers and the target.
- ``freegs_vacuum_patterns.npz``: FreeGS's vacuum flux per kA of each column on the native grid,
  for the per-coil-group check of ``synth_mast_ceiling.py``.
- ``design.json``: every target; ``build.json``: provenance, settings, counts, failures by
  reason, runtime.

Each target is solved from the same template equilibrium, so the output does not depend on the
number of workers. Existing records are kept unless ``--overwrite``. Usage::

    python scripts/synth_mast_equilibria.py --limit 8 --workers 4     # smoke
    python scripts/synth_mast_equilibria.py                           # 2 x 192 targets
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from fusion_eq import synthetic_mast as sm
from fusion_eq.coil_flux import GreenTable, coil_model
from fusion_eq.dataset import DEFAULT_PROCESSED_ROOT, default_data_root, run_info
from fusion_eq.io import sha256_of_arrays
from fusion_eq.machines import machine_config

DEFAULT_OUT = DEFAULT_PROCESSED_ROOT / "synthetic" / "mast_v1"
DEFAULT_GREEN = DEFAULT_PROCESSED_ROOT / "green" / "green_mast.npz"

_WORKER: dict = {}


def log(message: str) -> None:
    print(message, flush=True)


def _init_worker(data_root: str, coil_model_name: str, refine: int) -> None:
    """Builds the machine and the template equilibrium once per worker process."""
    import warnings

    warnings.simplefilter("ignore")
    geometry, grid_r, grid_z, _ = sm.load_geometry(data_root)
    fm = sm.freegs_machine(geometry, coil_model(sm.MACHINE, geometry.columns, coil_model_name))
    grid = sm.SolverGrid.refined(grid_r, grid_z, refine)
    _WORKER.update(fm=fm, grid_r=grid_r, grid_z=grid_z, template=sm.template_equilibrium(fm, grid), grid=grid)


def _solve(target_dict: dict, out_dir: str, rtol: float, maxits: int, limits: dict | None, meta: dict) -> dict:
    """Solves one target in a worker and writes its record when it converges."""
    target = sm.EquilibriumTarget.from_dict(target_dict)
    fm = _WORKER["fm"]
    outcome = sm.solve_target(_WORKER["template"], fm, target, rtol=rtol, maxits=maxits, limits_ka=limits)
    if outcome.converged:
        record = sm.equilibrium_record(outcome, fm, _WORKER["grid_r"], _WORKER["grid_z"], meta)
        sm.write_record(sm.record_path(out_dir, target.index), record)
    return {"index": target.index, "converged": outcome.converged, "reason": outcome.reason, "profile": outcome.profile,
            "seconds": round(outcome.iterations_s, 2)}


def reason_class(reason: str) -> str:
    """A short class of a failure reason for the summary counts."""
    for key in ("too many iterations", "No O-points", "plasma current", "no X-point", "non-finite", "no O-point"):
        if key in reason:
            return key
    return reason.split(":")[0] if reason else "converged"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data-root", type=Path, default=default_data_root(), help="raw data root (geometry from a MAST test file)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--green", type=Path, default=DEFAULT_GREEN, help="MAST Green's table, for the geometry and grid check")
    parser.add_argument("--n-per-topology", type=int, default=192, help="targets per topology (LSN, DN)")
    parser.add_argument("--seed", type=int, default=0, help="seed of the Sobol design")
    parser.add_argument("--coil-model", default="sol_half", help="MAST coil model (fusion_eq.coil_flux.COIL_MODELS)")
    parser.add_argument("--refine", type=int, default=2, help="solver grid steps per native grid step")
    parser.add_argument("--rtol", type=float, default=1e-4, help="Picard tolerance on the change of psi over its range")
    parser.add_argument("--maxits", type=int, default=80, help="Picard iterations per attempt")
    parser.add_argument("--no-current-limits", action="store_true", help="leave P2, P4 and P5 unbounded")
    parser.add_argument("--limit", type=int, default=None, help="solve only the first N targets of each topology (smoke run)")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 3) - 2))
    parser.add_argument("--overwrite", action="store_true", help="solve targets that already have a record")
    args = parser.parse_args(argv)
    t_start = time.time()
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")

    import warnings

    warnings.simplefilter("ignore")
    geometry, grid_r, grid_z, source = sm.load_geometry(args.data_root)
    table = GreenTable.load(args.green)
    if table.geometry_hash != geometry.hash() or not (np.allclose(table.grid_r, grid_r) and np.allclose(table.grid_z, grid_z)):
        raise ValueError(f"{args.green} was built from another geometry or grid than {source}")
    scales = coil_model(sm.MACHINE, geometry.columns, args.coil_model)
    log(f"Geometry from {source}: {geometry.n_elements} elements, {len(geometry.columns)} columns; coil model "
        f"{args.coil_model} (solenoid scale {scales.scales[machine_config(sm.MACHINE).solenoid_column]})")

    design = sm.target_design(args.n_per_topology, args.seed)
    if args.limit is not None:
        seen: Counter = Counter()
        kept = []
        for t in design:
            if seen[t.topology] < args.limit:
                kept.append(t)
                seen[t.topology] += 1
        design = kept
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "design.json").write_text(json.dumps([t.to_dict() for t in design], indent=1) + "\n")

    fm = sm.freegs_machine(geometry, scales)
    patterns = fm.column_patterns(grid_r, grid_z)
    np.savez(args.out / "freegs_vacuum_patterns.npz", patterns=patterns, columns=np.array(fm.columns), grid_r=grid_r,
             grid_z=grid_z, coil_model=np.array(args.coil_model))

    limits = None if args.no_current_limits else dict(sm.DEFAULT_CURRENT_LIMITS_KA)
    meta = {"coil_model": args.coil_model, "geometry_hash": geometry.hash(), "grid_hash": sha256_of_arrays(grid_r, grid_z),
            "freegs": sm.freegs_provenance(), "refine": args.refine, "rtol": args.rtol, "maxits": args.maxits,
            "current_limits_ka": limits, "zero_columns": list(sm.DEFAULT_ZERO_COLUMNS), "seed": args.seed}
    todo = [t for t in design if args.overwrite or not sm.record_path(args.out, t.index).is_file()]
    log(f"{len(design)} targets ({Counter(t.topology for t in design)}), {len(design) - len(todo)} already solved; "
        f"{len(todo)} to solve with {args.workers} workers")

    results = []
    items = [(t.to_dict(), str(args.out), args.rtol, args.maxits, limits, meta) for t in todo]
    if args.workers <= 1:
        _init_worker(str(args.data_root), args.coil_model, args.refine)
        stream = (_solve(*item) for item in items)
        pool = None
    else:
        pool = ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn"), initializer=_init_worker,
                                   initargs=(str(args.data_root), args.coil_model, args.refine))
        stream = pool.map(_solve, *zip(*items)) if items else iter(())
    try:
        for k, r in enumerate(stream, 1):
            results.append(r)
            if k % 10 == 0 or k == len(items):
                done = sum(x["converged"] for x in results)
                log(f"  {k}/{len(items)} solved, {done} converged ({time.time() - t_start:.0f} s)")
    finally:
        if pool is not None:
            pool.shutdown()

    results.sort(key=lambda r: r["index"])
    converged = [p.name for p in sm.list_records(args.out) if int(p.stem.split("_")[1]) in {t.index for t in design}]
    failures = [r for r in results if not r["converged"]]
    by_reason = Counter(reason_class(r["reason"]) for r in failures)
    by_topology = {topo: {"targets": sum(t.topology == topo for t in design),
                          "converged": sum(sm.record_path(args.out, t.index).is_file() for t in design if t.topology == topo)}
                   for topo in sm.TOPOLOGIES}
    info = run_info(argv, vars(args), timestamp, round(time.time() - t_start, 1), geometry_source=source, meta=meta,
                    n_targets=len(design), n_converged=len(converged), by_topology=by_topology,
                    failures_by_reason=dict(by_reason), failures=failures,
                    profiles=dict(Counter(r["profile"] for r in results if r["converged"])),
                    median_solve_s=float(np.median([r["seconds"] for r in results])) if results else None,
                    ranges={k: list(v) for k, v in sm.DEFAULT_RANGES.items()})
    (args.out / "build.json").write_text(json.dumps(info, indent=1, default=str) + "\n")
    log(f"Converged {len(converged)} of {len(design)} ({by_topology}); failures {dict(by_reason)}. Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
