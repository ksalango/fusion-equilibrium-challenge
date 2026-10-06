# Frame table specification: data preparation and column profile

Specification for the first analysis phase. The repository's data rules and writing standard
apply. Item ids (A1 to A7, B1, B6) match the pre-modelling analysis plan. Finding ids (F01 to
F25) refer to `docs/findings.md`.

## Purpose

Build the frame table: one row per EFIT frame, with every input brought onto that frame's
time. Every later analysis and every model reads this table, so it is built once, tested,
and checked against the QA results before anything depends on it.

The phase settles four things:

1. The rule for resampling coil currents and plasma current onto frame times (A2).
2. The rule for bringing Thomson profiles onto frame times, and what marks a frame as
   having no usable Thomson (A3).
3. The window for rate-of-change features (A4).
4. Which columns are usable as they are, and how the two machines differ in raw form
   (B1, B6).

Out of scope: the train and validation split (A5, specified with the near-repeat check
B8), the label-quality flag (B5), Thomson profile descriptors on a normalized radius (B4,
Part E), coil flux (Part D), and any model.

## Context

- Shots are loaded with `fusion_eq.io.load_shot`, which applies the organizers' fixes and
  the project's Ip alignment. Nothing in this phase reads a shot file any other way.
- Data root: `../data/raw/data/`, read-only. Output root: `../data/processed/`.
- Splits: `diii_d_train` (7,041 shots, 1,559,340 frames), `diii_d_public_test` (874 shots,
  193,359 frames), `mast_public_test` (1,206 shots, 69,075 frames).
- The three MAST sample shots in the starter kit are not included in the frame table.
  They carry MAST equilibrium labels, which the challenge rules of 23 September 2026
  exclude from any use that shapes MAST predictions.
- Facts from QA that this phase depends on:
  - DIII-D magnetics were recorded at 20 kHz on 5,518 shots and 2 kHz on 2,397; MAST is
    5 kHz. DIII-D Ip is on its own 0.5 ms time base (F10).
  - 114 MAST test shots have NaN in the current columns in a regular pattern (F20).
  - DIII-D marks missing Thomson with exact zeros, MAST with NaN. Channel counts vary by
    shot on DIII-D (F16).
  - Gaps between EFIT frames reach 5.2 s, and 49 shots have fewer than 10 frames (F13).
  - MAST `efit_times` and `magnetics_time` are float32. All time axes are cast to float64
    before arithmetic.

## Deliverables

1. Package functions with tests. The layout below is a proposal; it is adjusted to fit
   the existing modules and agreed before implementation starts. No general-purpose
   `utils` module.
   - `fusion_eq/resample.py`: resampling of a sampled signal onto arbitrary target times.
     Three functions on plain arrays: nearest valid sample, mean over a centred window,
     and least-squares slope over a centred window. Windows are given in milliseconds.
     NaN samples are skipped. Target times are an argument, so the same functions later
     serve lag features (values at a frame time minus a lag).
   - `fusion_eq/thomson.py`: conversion of the machine's missing-value marker to NaN, the
     valid-slice rule, per-slice summaries, and alignment of those summaries to frame
     times. The valid-slice rule currently inside QA check 0.7 moves here, and the QA
     check calls it, so that QA and data preparation share one definition.
   - `fusion_eq/frame_table.py`: `build_shot_frames(shot, params)` returning one shot's
     rows; the column schema with a role for every column (key, input, flag, label);
     `read_frame_table(split, columns=None, shots=None)`; `input_columns(machine)`, which
     never returns a label column; and `read_flux_maps(shots)`, which returns
     `efit_psirz` for the listed training shots as a float32 array in the frame table's
     row order.
   - `fusion_eq/profile.py`: per-column summary statistics of a table, grouped by split.
   - `fusion_eq/machines.py`: extended with the per-machine column mapping for plasma
     current, solenoid, toroidal field and the list of shaping-coil columns.
2. `scripts/build_frame_table.py`: the command-line entry point for the build. It holds no
   preparation logic.
   - Arguments: `--splits` (default all three), `--n-shots`, `--shots-file` (a JSON list
     of shot ids, for the smoke set), `--seed`, `--workers`, `--data-root`, `--out`
     (default `../data/processed/frame_table`), `--window-ms` (default 2.0),
     `--rate-window-ms` (default 10.0), `--max-gap-ms` (default 5.0),
     `--thomson-tol-ms` (default 20.0), `--no-align-ip`, `--no-psi-summary`.
   - `--no-align-ip` requires an explicit `--out`, so the default table is never
     overwritten by the unaligned variant.
   - Shots run in parallel. One failing shot does not stop the run; the error is recorded
     in that shot's row of the shot index.
3. `scripts/profile_frame_table.py`: reads a built table and writes the report and the
   column profile. Separate from the build so it can be re-run without rebuilding.
4. Outputs of the build, under `--out`:
   - `diii_d_train.parquet`, `diii_d_public_test.parquet`, `mast_public_test.parquet`:
     one row per frame.
   - `shot_index.parquet`: one row per shot with split, machine, frame count, shot-level
     flags and `error`.
   - `build.json`: repository commit, starter-kit commit, timestamp, arguments, runtime.
5. Outputs of the profile, under `reports/frame_table/`:
   - `frame_table_report.md`: the acceptance checks below, then per-column missing counts.
   - `column_profile.parquet`: one row per column and split.
6. `notebooks/02_frame_table.ipynb`: the review notebook described below.
7. `tests/test_resample.py`, `tests/test_thomson.py`, `tests/test_frame_table.py`.
8. A "Frame table" section added to `docs/data_dictionary.md` listing every column, its
   role, units and how it is computed.

## Columns

Values are float32 unless stated. Label columns are present in the training table only.

### Keys

| Column | Type | Content |
| --- | --- | --- |
| `machine`, `split`, `shot` | string | Identity of the row |
| `frame` | int32 | Index of the frame within the shot, from 0 |
| `time_ms` | float64 | `efit_times[frame]` |
| `n_frames` | int32 | Frames in the shot |
| `dt_prev_ms` | float32 | Time since the previous frame; NaN on the first frame |

### Currents (A1, A2, A4)

Plasma current, solenoid and toroidal field are stored under the same names on both
machines: `ip` (`magnetics_plasma_current`), `sol` (`magnetics_ECOILA` on DIII-D,
`magnetics_sol_current` on MAST) and `tf` (`magnetics_bcoil`, `magnetics_tf_current`).
Shaping coils keep their own column names, as does `magnetics_efps_current` on MAST.

Each current signal `c` gives three columns:

| Column | Content | Units |
| --- | --- | --- |
| `c__nearest` | Nearest valid sample to the frame time. NaN if that sample is more than `--max-gap-ms` away. | kA or kA-turn |
| `c__mean` | Mean of the valid samples in a window of `--window-ms` centred on the frame time. If the window holds no valid sample, linear interpolation between the valid samples either side when they are at most `--max-gap-ms` apart, otherwise NaN. | kA or kA-turn |
| `c__rate` | Least-squares slope of the valid samples in a window of `--rate-window-ms` centred on the frame time. NaN with fewer than 3 valid samples. | kA/s |

Both resampling forms are stored so that A2 can be decided from the table. The losing
form is dropped from later builds.

On MAST each signal is resampled on its own valid samples, since the NaN pattern differs
between the toroidal field and the other currents (F20).

### Thomson (A3)

DIII-D zeros are converted to NaN first. A channel is valid in a time slice when both Te
and ne are present. A slice is valid under the QA rule (at least half of its channels
valid). Summaries are computed per valid slice from the valid channels only, then brought
to frame times in two ways: linear interpolation in time between the valid slices either
side (`__interp`; the nearest value is held before the first and after the last valid
slice), and the nearest valid slice (`__nearest`).

| Column | Content | Units |
| --- | --- | --- |
| `ts_core_te_peak`, `ts_core_ne_peak`, `ts_core_pe_peak` | 90th percentile across valid core channels of Te, ne and electron pressure (ne x Te x 1.602e-19 J/eV, divided by 1000) | eV, m^-3, kPa |
| `ts_core_pe_mean` | Mean electron pressure across valid core channels | kPa |
| `ts_edge_te_mean`, `ts_edge_ne_mean`, `ts_edge_pe_mean` | Means across valid edge channels | eV, m^-3, kPa |
| `ts_core_n_valid`, `ts_edge_n_valid` | Valid channels in the nearest valid slice | count |
| `ts_core_dt_ms`, `ts_edge_dt_ms` | Absolute time from the frame to the nearest valid slice; NaN when the shot has none | ms |
| `ts_core_gap_ms`, `ts_edge_gap_ms` | Time between the valid slices either side of the frame; NaN when the frame is not bracketed | ms |

Each summary column appears with both suffixes. The time-distance columns are stored so
that the tolerance can be changed without a rebuild.

These summaries are provisional. They exist to test the time alignment and to give the
column profile a pressure measure. Profile descriptors on a normalized radius replace
them in a later phase (B4, Part E).

### Flags (A7)

| Column | Type | Content |
| --- | --- | --- |
| `mag_rate_hz` | int32 | 20000, 2000 or 5000 |
| `mag_has_nan` | bool | The shot's current columns contain NaN (the 114 MAST shots of F20) |
| `ip_direction` | int8 | +1, -1, or 0 for mixed; per shot, computed with the function QA uses |
| `tf_direction` | int8 | Sign of the toroidal field current; per shot |
| `ip_fix` | string | `organizers`, `alignment` or `none`, from `Shot.fixes` |
| `ip_shift_ms` | float32 | Total shift applied to the Ip time axis |
| `ip_over_peak` | float32 | Absolute `ip__mean` over the shot's 99.5th percentile of absolute Ip |
| `inputs_complete` | bool | Every `__mean` current column is finite on this frame |
| `ts_core_valid`, `ts_edge_valid` | bool | Nearest valid slice within `--thomson-tol-ms` |

The label-quality flag is not written in this phase. The phase that defines it (B5)
writes it as a separate table keyed on `shot` and `frame`, which `read_frame_table` joins.

### Labels (training table only)

| Column | Content |
| --- | --- |
| `efit_q95`, `efit_beta_n`, `efit_li`, `efit_r_axis`, `efit_z_axis`, `efit_lcfs_n` | Copied from the shot file without interpolation |
| `psi_min`, `psi_max`, `psi_mean` | Minimum, maximum and mean of the frame's flux map |

Flux maps and boundary contours are not stored in the table. Flux maps are read with
`read_flux_maps`. `magnetics_dsep` is left out: it is unscored and means different things
on the two machines.

## Acceptance checks

Written to `frame_table_report.md` with status `pass`, `fail` or `review`.

| Id | Check | Status rule |
| --- | --- | --- |
| T1 | Row and shot counts per split | Pass if 1,559,340 / 193,359 / 69,075 frames over 7,041 / 874 / 1,206 shots and no shot has an error |
| T2 | Keys | Pass if (`shot`, `frame`) is unique and `time_ms` strictly increases within every shot |
| T3 | Labels untouched | Pass if, on a sample of 50 training shots, every label column equals the shot file's array exactly, and the test tables contain no label column |
| T4 | Input completeness | Review. Share of frames with `inputs_complete`, per split; shots with any incomplete frame |
| T5 | Shot-level flags against QA | Pass if recording-rate counts (5,518 and 2,397 on DIII-D), Ip direction counts (training 6,420 positive, 603 negative, 18 mixed; test 798 and 76) and the count of MAST shots with NaN (114) match the QA run |
| T6 | Ip at the first frame | Pass if `ip_over_peak` on the first frame of every DIII-D shot is at least 0.15. QA found a minimum of 0.17 after the alignment (F12); the margin allows for the windowed mean. |
| T7 | Thomson coverage against QA | Review. Share of shots with `ts_core_valid` on at least 90% of frames (about 97% in QA); shots with no valid core profile (53 training, 6 DIII-D test, 1 MAST test in QA) |
| T8 | Resampling forms agree | Review. Per current column and recording rate: the distribution of `c__nearest` minus `c__mean`, relative to the column's spread |

## Review notebook

`notebooks/02_frame_table.ipynb` calls the package and the outputs above. Sections:

1. Build summary: the acceptance table and any shots with errors.
2. Resampling (A1, A2): for one shot of each kind in the smoke set, the raw signal with
   the frame values of both forms drawn over it; then the T8 distributions by recording
   rate. The comparison of the two forms on the ridge score is made in the baseline phase.
3. Rates of change (A4): the slope drawn against the raw trace through a current ramp; the
   spread of `ip__rate` and of one shaping-coil rate on flat-top frames, compared between
   the 20 kHz and 2 kHz groups. A wider spread in one group means the window is too short.
4. Thomson (A3): `__interp` against `__nearest`; coverage as a function of the tolerance,
   per machine, from the time-distance columns; the shots with no usable Thomson.
5. Column profile (B1): for every column, range, percentiles, missing fraction and share
   of exact zeros. Histograms of the shared quantities (`ip`, `sol`, `tf`, their rates,
   the Thomson summaries, `dt_prev_ms`, `n_frames`) for DIII-D training, DIII-D test and
   MAST side by side. Shaping-coil currents per machine.
6. Relationships (B6), DIII-D training frames: flux range (`psi_max` minus `psi_min`)
   against absolute Ip; `efit_beta_n` against `ts_core_pe_mean`; `efit_q95` against
   absolute `tf` over absolute `ip`; the correlation matrix of the coil currents.
7. Findings: the decision reached on each item below, and anything unexpected.

Plots that need individual frames use a fixed-seed sample of shots. Summary statistics use
the full table.

## Decisions this phase feeds

| Decision | Options | Evidence |
| --- | --- | --- |
| Resampling of currents (A2) | Nearest sample; windowed mean, and its window length | Notebook sections 2 and 3; confirmed on the ridge score in the baseline phase |
| Rate-of-change window (A4) | 10 ms, or longer if the two recording rates differ in noise | Notebook section 3 |
| Thomson in time (A3) | Interpolation between valid slices; nearest slice within a tolerance | Notebook section 4 |
| Thomson validity tolerance | 20 ms, or another value read off the coverage curve | Notebook section 4 |
| Usable inputs (B1) | Per column: usable as is, needs scaling, or dropped | Notebook section 5 |
| First read on linearity (B6) | Which targets follow simple combinations of the inputs | Notebook section 6 |

## Tests

On small synthetic signals and shots:

- A linear ramp is reproduced by both resampling forms, and its slope by the rate.
- The same underlying signal sampled at 20 kHz and at 2 kHz gives the same mean and the
  same slope for the same window in milliseconds.
- NaN samples are skipped; a signal with no valid sample near the frame gives NaN.
- DIII-D zeros and MAST NaN are both treated as missing Thomson; shots with different
  channel counts are handled; a shot with no valid slice gives NaN summaries, NaN time
  distance and `ts_core_valid` false.
- Interpolated Thomson at a frame halfway between two valid slices is their average.
- Label columns equal the input arrays exactly; a test-split shot yields no label column.
- `input_columns` returns no label column, and requesting a label as an input raises.
- A shot that raises inside the build is recorded in the shot index and the run continues.
- `align_ip=False` is passed to the loader and recorded in `build.json`.

## Procedure

1. Review `fusion_eq/io.py`, `fusion_eq/machines.py` and the Thomson logic in
   `fusion_eq/qa/`. Agree the module layout.
2. Implement the functions and tests. Run the tests.
3. Choose the smoke set from `reports/qa/qa_per_shot.parquet`: two shots each of DIII-D
   20 kHz, DIII-D 2 kHz, DIII-D negative Ip, DIII-D with no valid Thomson, DIII-D with
   fewer than 10 frames, DIII-D test, MAST without NaN and MAST with NaN. Save the ids to
   `reports/frame_table/smoke_shots.json`.
4. Smoke run, launched manually:
   `python scripts/build_frame_table.py --shots-file reports/frame_table/smoke_shots.json --out ../data/processed/frame_table_smoke`.
   Record the runtime and an estimate for the full run.
5. Review notebook sections 1 and 2 on the smoke output.
6. Full run, launched manually: `python scripts/build_frame_table.py`.
7. Profile, launched manually: `python scripts/profile_frame_table.py`.
8. Review the remaining notebook sections. Add the findings to `docs/findings.md`.

## Constraints

- Nothing is written under `../data/raw/`.
- Targets are never interpolated or resampled.
- Every run against the data, including the smoke run, is launched manually.
- No label column is used to compute any input or flag column.
- No statistic is fitted in this phase. The table holds measured values only; scaling and
  normalization belong to later phases.
- Built tables are not committed. Code, tests, the notebook and `reports/frame_table/`
  are.
- New packages are added to `requirements.txt` the same day they are installed.
