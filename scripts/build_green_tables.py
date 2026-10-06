"""Builds the Green's tables: one coil-flux pattern per current column and machine.

Reads coil geometry and the flux grid from one shot of each machine (geometry is
identical in every shot, QA check 0.9), builds the table at ``--n-filaments`` per
element side and once more at twice that for the convergence check K2, and writes to
``--out``:

- ``green_d3d.npz`` and ``green_mast.npz``: the tables (``fusion_eq.coil_flux.GreenTable``).
- ``build.json``: commits, timestamp, arguments, geometry and grid hashes, K1 (geometry
  join), K2 (convergence, near-coil nodes and other nodes separately), the time of the
  coil flux for 100,000 frames (D8).

Nothing is written under the data root. Usage::

    python scripts/build_green_tables.py
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from fusion_eq import io, starter_kit
from fusion_eq.coil_flux import (DEFAULT_N_FILAMENTS, GEOMETRY_COLUMNS, GRID_COLUMNS, CoilGeometry, GreenTable,
                                 build_green_table, near_coil_mask)
from fusion_eq.dataset import DEFAULT_PROCESSED_ROOT, default_data_root, list_shots, run_info
from fusion_eq.machines import machine_config

DEFAULT_OUT = DEFAULT_PROCESSED_ROOT / "green"
MACHINE_SPLIT = {"DIII-D": "diii_d_train", "MAST": "mast_public_test"}
EXPECTED_ELEMENTS = {"DIII-D": 19, "MAST": 812}
EXPECTED_SOLENOID_ELEMENTS = {"DIII-D": 1, "MAST": 656}
TIMING_FRAMES = 100_000
K2_LIMIT = 1e-3


def table_file(out_dir: Path, machine: str) -> Path:
    return out_dir / f"green_{machine_config(machine).file_prefix.split('_')[0]}.npz"


def convergence(table: GreenTable, finer: GreenTable, near: np.ndarray) -> dict:
    """K2: largest change between two subdivisions, relative to each pattern's range."""
    worst_near, worst_other = 0.0, 0.0
    for k in range(len(table.columns)):
        rng = float(table.patterns[k].max() - table.patterns[k].min())
        diff = np.abs(finer.patterns[k] - table.patterns[k]) / rng if rng > 0 else np.zeros_like(near, dtype=float)
        if near.any():
            worst_near = max(worst_near, float(diff[near].max()))
        worst_other = max(worst_other, float(diff[~near].max()))
    return {"max_rel_change_near_coil": worst_near, "max_rel_change_other": worst_other,
            "n_near_coil_nodes": int(near.sum()), "status_other": "pass" if worst_other < K2_LIMIT else "fail"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data-root", type=Path, default=default_data_root())
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n-filaments", type=int, default=DEFAULT_N_FILAMENTS)
    parser.add_argument("--machines", nargs="+", default=list(MACHINE_SPLIT), choices=list(MACHINE_SPLIT))
    parser.add_argument("--no-convergence", action="store_true", help="skip the table at twice the subdivision")
    args = parser.parse_args(argv)

    starter_kit.require()
    started = time.time()
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    args.out.mkdir(parents=True, exist_ok=True)
    results = {}
    for machine in args.machines:
        cfg = machine_config(machine)
        path = list_shots(args.data_root, MACHINE_SPLIT[machine])[0]
        shot = io.load_shot(path, [*GEOMETRY_COLUMNS, *GRID_COLUMNS], apply=False)
        geometry = CoilGeometry.from_shot(shot)
        grid_r, grid_z = np.asarray(shot["efit_grid_R"], float), np.asarray(shot["efit_grid_Z"], float)
        problems = geometry.check_against_machine(cfg, shot.file_columns)
        counts = geometry.elements_per_column()
        k1_ok = (not problems and geometry.n_elements == EXPECTED_ELEMENTS[machine]
                 and counts.get(cfg.solenoid_column, 0) == EXPECTED_SOLENOID_ELEMENTS[machine])
        print(f"{machine}: {geometry.n_elements} elements over {len(counts)} columns from {path.name}; "
              f"K1 {'pass' if k1_ok else 'fail'} {problems}", flush=True)
        t0 = time.time()
        table = build_green_table(cfg, geometry, grid_r, grid_z, args.n_filaments)
        build_s = time.time() - t0
        table.save(table_file(args.out, machine))
        near = near_coil_mask(geometry, grid_r, grid_z)
        k2 = None
        if not args.no_convergence:
            finer = build_green_table(cfg, geometry, grid_r, grid_z, 2 * args.n_filaments)
            k2 = convergence(table, finer, near)
            print(f"  K2: near-coil {k2['max_rel_change_near_coil']:.2e}, other {k2['max_rel_change_other']:.2e} "
                  f"({k2['status_other']})", flush=True)
        rng = np.random.default_rng(0)
        currents = rng.normal(size=(TIMING_FRAMES, len(table.columns)))
        t0 = time.time()
        table.flux(currents)
        timing_s = time.time() - t0
        print(f"  coil flux for {TIMING_FRAMES} frames: {timing_s:.2f} s; table built in {build_s:.1f} s", flush=True)
        results[machine] = {
            "source_shot": path.name, "file": table_file(args.out, machine).name,
            "n_elements": geometry.n_elements, "elements_per_column": counts,
            "geometry_hash": table.geometry_hash, "grid_hash": table.grid_hash,
            "k1": {"status": "pass" if k1_ok else "fail", "problems": problems},
            "k2": k2, "build_s": build_s, "flux_timing_s_per_100k_frames": timing_s,
        }
    info = run_info(argv, vars(args), timestamp, time.time() - started, machines=results)
    (args.out / "build.json").write_text(json.dumps(info, indent=2) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
