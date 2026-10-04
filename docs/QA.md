# QA specification: full-dataset quality checks

Specification for the data quality phase. The repository's data rules and writing
standard apply. Check ids (0.1 to 0.15) match Part 0 of the pre-modelling analysis plan.

## Purpose

Run a fixed set of quality checks over the full dataset and produce a report plus a
per-shot results table. The phase answers one question: can the data be trusted as-is,
and if not, which cleaning rules are needed.

Out of scope: modelling, feature engineering, building the frame table, and any change
to the raw data.

## Context

- Data root: `../data/raw/data/`, read-only.
  - `diii_d_train` (7,041 files, labels present)
  - `diii_d_public_test` (874 files, labels withheld)
  - `mast_public_test` (1,206 files, labels withheld)
- One Parquet file per shot. Each file holds one row with nested arrays.
- Starter kit: `../fusion-equilibrium-challenge-starter/`.
  - `data_fixes.py` holds the organizers' fixes (Ip time axis, MAST Thomson channel
    alignment). Import and call these; do not reimplement them.
  - `fusion_scoring/` holds the scorer's functionals, including boundary extraction.
  - `parquet_data/` holds six sample shots. The three MAST sample shots are the only
    MAST files with labels.
- Existing work to reuse: `scripts/summarize_shot.py`, `docs/data_dictionary.md`.

## Deliverables

1. `fusion_eq/io.py`: shared shot loading, used by this phase and every later one. It
   reads a shot file with a chosen set of columns and applies the organizers' fixes.
   Loading logic currently inside `scripts/summarize_shot.py` moves here, and that
   script is updated to call it. No general-purpose `utils` module; shared code goes in
   a module named for what it does.
2. `fusion_eq/qa/`: one function per check. Each takes a loaded shot (or the columns it
   needs) and returns a flat dict of metrics. No printing, no file writes. Per-machine
   constants the checks need (flux sign convention, missing-value convention, solenoid column)
   live in `fusion_eq/machines.py`, the per-machine adapter the data rules call for. Access to
   the starter kit (fixes, scorer, envelope masks) goes through `fusion_eq/starter_kit.py`.
3. `scripts/run_qa.py`: the single command-line entry point. It holds no check logic;
   it selects shots, calls the check functions, and writes the outputs.
   - Arguments: `--splits` (default all three), `--n-shots` (sample size per split,
     default all), `--seed`, `--workers`, `--out` (default `internal/qa/`).
   - Reads only the columns each check needs. The flux maps are large; avoid loading
     them where a check does not need them.
   - Runs shots in parallel. One failing shot must not stop the run; record the error
     in that shot's row.
4. Outputs written to `--out`:
   - `qa_per_shot.parquet`: one row per shot, one column per metric, plus `split`,
     `machine`, `shot`, `error`.
   - `qa_report.md`: a summary table (check id, status, headline number, note), then
     one short section per check with the flagged shots (cap each list at 20; the full
     list is in the Parquet file).
   - `qa_run.json`: repo commit, starter-kit commit, timestamp, arguments, runtime.
5. `tests/test_qa.py`: pytest tests for each check function on small synthetic shots,
   including one case built to fail each check.

Status values in the report: `pass`, `fail`, `review` (no fixed criterion; the numbers
need a human read).

## Checks

| Id | Check | Metrics to record | Status rule |
| --- | --- | --- | --- |
| 0.1 | File inventory | Files per split; each file opens; row count per file; shot id; file size | Pass if counts are 7,041 / 874 / 1,206 and every file opens with one row |
| 0.12 | Split integrity | Shot ids per split; overlaps between train and test | Pass if no shot id appears in two splits |
| 0.14 | Column sets | Column list per split; columns present in train but absent in test (the label set) | Review. The label set is expected to be `efit_psirz`, `efit_q95`, `efit_beta_n`, `efit_li`, `efit_r_axis`, `efit_z_axis`, `efit_lcfs_n`, `efit_lcfs_r`, `efit_lcfs_z`, `magnetics_dsep`, `magnetics_dsep_times`. Report any difference. |
| 0.2 | Ip time-axis erratum (DIII-D) | Whether `data_fixes.py` treats the shot as affected; shift applied in ms; after the fix, whether the Ip time axis covers `efit_times`; median absolute Ip at EFIT frames; the ratio of that median to the shot's 99.5th percentile of absolute Ip | Fail for a shot if, after the fix, Ip does not cover the EFIT window or the median absolute Ip at EFIT frames is below 10 percent of the shot's 99.5th percentile of absolute Ip (the breakdown threshold `data_fixes.py` itself uses). The ratio is stored so the threshold can change without a re-run. Report the affected fraction (about 69% expected) and the distribution of shifts (about -3152.57 ms on the sample shots). |
| 0.3 | MAST Thomson channel alignment | Whether the fix changed the shot; channel counts before and after | Review |
| 0.4 | Units of current columns | For every current column, within the EFIT window: min, max, 99th percentile of absolute value | Review. Report per-column distributions per machine. Reference magnitudes from the sample shots: DIII-D F-coils up to a few hundred, Ip about 1378, ECOILA within +/-40, bcoil about -117; MAST P-coils up to about 245, Ip about 797, sol within +/-46, tf about -85. |
| 0.5 | Flux map shape and indexing (train, plus MAST sample shots) | Shape of `efit_psirz`; for a sample of frames, the grid position of the global psi extremum (minimum on DIII-D, maximum on MAST, the scorer's `AXIS_SIGN` convention) against `efit_r_axis` and `efit_z_axis` under the [t, iZ, iR] reading; the same distance under the transposed reading and for the opposite extremum, for diagnosis | Pass if the shape is [n_frames, 65, 65] and the extremum sits within one grid cell of the stored axis. The scorer locates the axis as an interior local maximum, which differs from the global extremum on about 1.4 percent of MAST frames; check 0.15 records the scorer's own axis against the stored one. |
| 0.10 | Grid | Length, monotonicity and extents of `efit_grid_R` and `efit_grid_Z`; a hash of both arrays | Pass if both are strictly increasing with length 65 and each machine has one hash. Report extents in metres. |
| 0.9 | Coil geometry | A hash of all coil geometry columns; coil names; whether every `coil_input_column` value maps to an existing current column; presence of the solenoid rows (ECOILA on DIII-D, `sol_*` on MAST); element counts | Pass if each machine has one hash. If more than one exists, report each configuration and the shots using it. |
| 0.6 | Boundary contour (train, plus MAST sample shots) | Range of `efit_lcfs_n`; padding value after the valid points; NaN fraction inside the valid points | Review. Expected padding: zeros on DIII-D, NaN on MAST. |
| 0.7 | Missing Thomson | Within the EFIT window only: missing fraction per channel (zeros on DIII-D, NaN on MAST; a channel is valid only when both Te and ne are present), stored as scalar summaries plus the full per-channel vector as a JSON string since channel counts vary per shot; fraction of EFIT frames with a valid core profile within 10 ms and within 20 ms, where a valid profile is a Thomson time slice with at least half its channels valid; the same fractions for slices with at least one valid channel, so the half-channels threshold can be revisited without a re-run | Review |
| 0.8 | `magnetics_dsep` (train, plus MAST sample shots) | Distribution; count of the -1 sentinel | Review |
| 0.11 | Label and time-base sanity | NaN and infinite counts in `efit_psirz`, `efit_q95`, `efit_beta_n`; min and max of q95 and betaN; frames per shot; `efit_times` strictly increasing; median sampling interval of each time base | Fail for a shot with non-finite labels, zero frames or non-increasing `efit_times`. The frame-count distribution of every split is review, including the number of shots with fewer than 10 frames. Extremes of q95 and betaN are review. |
| 0.11b | Test frame totals | Sum of `len(efit_times)` over each test split | Pass if DIII-D test totals 193,359 frames and MAST test totals 69,075 |
| 0.15 | Stored labels against the scorer (sample of train shots) | For about 200 shots and 5 frames each: distance between the stored contour (`efit_lcfs_r`, `efit_lcfs_z`) and the contour the scorer extracts from the true psi; difference between stored `efit_li`, `efit_r_axis`, `efit_z_axis` and the scorer's values | Review. Report the distribution of each difference. |
| 0.13 | README against data | For each of the nine contradictions listed in `docs/data_dictionary.md`: whether the full data agrees with the dictionary | Review. List proposed dictionary updates in the report; do not edit the dictionary in this phase. |

Checks that need labels run on `diii_d_train` and on the three MAST sample shots in the
starter kit. Checks on inputs run on all three splits. The runner treats the MAST sample shots
as a fourth split named `mast_samples`, included by default and never subsampled.

Locations: the data root defaults to `../data/raw/data` relative to the repository root and is
overridden with `--data-root`; the starter kit defaults to
`../fusion-equilibrium-challenge-starter` and is overridden with the environment variable
`FUSION_STARTER_DIR`.

## Procedure

1. Review `data_fixes.py`, `fusion_scoring/` and `scripts/summarize_shot.py`. The module
   layout is agreed before implementation starts.
2. Implement the check functions and tests. Run the tests.
3. Smoke test: `python scripts/run_qa.py --n-shots 20`. Record the runtime and an
   estimate for the full run.
4. Full run, launched manually from the terminal: `python scripts/run_qa.py`.

## Constraints

- Nothing is written under `../data/raw/`.
- No data is fixed or filtered in this phase. Problems are recorded, not repaired.
- Anything unexpected goes in the report under the relevant check with status `review`.
- `internal/qa/` outputs are not committed if `internal/` is git-ignored; the code and
  tests are.
- New packages are added to `requirements.txt` the same day they are installed.
