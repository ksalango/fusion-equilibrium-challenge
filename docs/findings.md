# Data findings register

Findings from the data quality phase of the Fusion Equilibrium Challenge project, with the
evidence for each and its consequence for data preparation and modelling. Later phases add
their findings to this file.

- Period covered: 4 to 14 October 2026.
- Evidence: `reports/qa/qa_report.md` (full run over 9,124 shots, with the Ip alignment
  enabled), `reports/qa_before_ipfix/qa_report.md` (the same run before the alignment),
  `notebooks/01_qa.ipynb`; from 6 October, `reports/frame_table/frame_table_report.md`,
  `reports/frame_table/column_profile.parquet` (full build, commit 12da376) and
  `notebooks/02_frame_table.ipynb`; from 10 October, `reports/coil_flux/coil_flux_report.md`
  with its parquet tables (fit of 10 October 2026, 50 fit shots and 8 reference shots) and
  `notebooks/03_coil_flux.ipynb`; from 10 October, `reports/split/split_report.md` with
  `leak_test.json` and `near_repeats.json`, `splits/split_v1.json`, and
  `reports/baseline/compare_baselines.md` with `frame_residuals.parquet`,
  `score_parts.parquet` and the residual maps (run of 10 October 2026, commit 6b73139); from
  11 October, `reports/scalars/summary.md`, `summary.json` and `label_filter_costs.csv`
  (`notebooks/05_scalars.ipynb`, working lists, stride 4),
  `reports/submissions/2026-10-11_baseline_v0/run.json` with the Codabench result of
  submission 979130 (`codabench_scoring_result/`, scoring version 3.3.1),
  `reports/submissions/2026-10-11_coil_only_v0/run.json`, and
  `reports/current_target/ceiling.md` and `ceiling.json` (`scripts/current_ceiling.py`); from
  12 October, the same two files from the run of 12 October 2026 on the working lists, which
  adds the vacuum remainder (variants e to g) and reproduces variants a to d of F78; from
  13 October, `reports/features/v1/feature_report.md` with `profile.json` and the parquet
  tables of `scripts/profile_features.py`, `linear_check.md` and `linear_check_followup.md`
  (`scripts/feature_linear_check.py`, working lists), `backup_radii_check.md`
  (`scripts/backup_radii_check.py`) and the saved outputs of `notebooks/06_features.ipynb`; from
  14 October, `reports/features/v2/feature_report.md` and `linear_check.md` (feature version 2),
  `reports/current_target/ceiling.md` and `ceiling.json` (run of 14 October, variants h to k),
  `reports/targets/v1/target_report.md` and `target_summary.json` (profile of the full target
  build of 14 October, commit c89e246, configuration hash b2192c3a01a9), the review of
  `notebooks/07_targets.ipynb`, `reports/determinism/determinism_check.md` (D7),
  `reports/submissions/2026-10-14_mean_current_v0/run.json` and
  `reports/scalars/betan_heads.md` (`scripts/betan_heads.py`, working lists, stride 4).
- Check ids in brackets refer to `docs/QA.md` (0.x), `docs/FRAME_TABLE.md` (Tx),
  `docs/COIL_FLUX.md` (Kx), `docs/SPLIT.md` (Sx), `docs/FEATURES.md` (Xx) and
  `docs/TARGET.md` (Yx).

## Summary

The dataset is complete and internally consistent. Test shots and training shots come from
the same population on every marker examined. One defect was found and corrected: on about
30% of DIII-D shots the plasma current (Ip) time axis was late by 50 to 330 ms after the
organizers' published fix. Several properties of the data constrain how inputs, targets and
missing values must be handled; these are listed below with the action each one requires.

The frame table (one row per EFIT frame, inputs resampled onto frame times) was built over
all 9,121 shots on 6 October and reproduces the QA run. Its review settled the resampling
rules provisionally (windowed mean over 2 ms, Thomson interpolated in time) and found that
the 10 ms rate-of-change window is too short, that MAST inputs lie inside the DIII-D ranges,
and that simple physical combinations of the inputs already carry much of the signal in the
scalar labels (F26 to F44).

The coil-flux phase (F45 to F53) established that the flux of the coils can be calculated
from geometry and currents to about one percent of a frame's flux range outside the plasma,
once the solenoid is represented by a flat term fitted on DIII-D, fixed the sign rule that
relates the stored map to the physical flux, and found the physics-first structure viable.

The first submission (F68 to F72) scored 0.9249 on DIII-D and 0.0003 on MAST, where the
coil flux alone was submitted; on MAST, where every test shot has positive Ip, the stored
map has the sign opposite to DIII-D's rule. The scalar
study (F73 to F77) found that q95 calculated from the map needs no fitted constant and
confirms 144 toroidal-field turns on DIII-D, while betaN from the electron pressure stays
behind the ridge. The current-distribution ceiling (F78) and the vacuum remainder (F79)
show that a map built from the coil flux, the flux of the current inside the plasma
boundary and a remainder comes within 0.001 of the plasma-flux ceiling. On 12 October this
three-part map was adopted as the plasma model target, together with the scaling and
mirroring rules listed under "Carried into modelling".

The feature phase (F80 to F88) built a version 1 table of unitless inputs that passes its
build checks. DIII-D test matches training on every feature; MAST lies inside DIII-D's full
range but in its tails, chiefly in the stretch terms of the coil pattern, where the
extrapolation is stronger shaping on MAST's flat-top and late phases. A linear model on the
features keeps the flux map and scores 0.014 below the hybrid ridge on DIII-D, the gap in
the shape terms. Feature version 2 replaces the two stretch terms by one shaping quantity
and `coil_c10` by `shafranov_lambda` in set_b, and leaves `ip_frac` unscaled. Version 2 keeps
the linear scores, and the frame table, the version 2 features and the target tables rebuild
byte-identically (F98).

The target phase (F89 to F97) built the three-part target on the full DIII-D lists and carried
it to MAST. A symmetric basis of 50 current components meets the ceiling rule, placement on
MAST costs nothing measurable, the predicted share is not rescaled, and the remainder ridge
uses 50 ms rates; with the remainder predicted from the currents, the whole chain reaches
S 0.9972 on DIII-D, 0.0018 below the plasma-flux basis. The mean current alone scores 0.792 on
DIII-D validation and gives a closed boundary on every MAST test frame checked. The betaN
study (F99) sets one head per machine, with the MAST head built from transferable inputs.

The split phase (F54 to F58) showed that the DIII-D test shots are a random draw from the
same pool as the training shots, that near-repeats of an experiment are common on both sides
in equal measure, and that the leak they cause for a linear model is small. A random split
by shot was adopted, with scores reported for all validation shots and for those without a
near-repeat in training. The baseline phase (F59 to F67) scored four linear models and two
representation ceilings with the organizers' scorer: a signed ridge on 50 principal
components reaches S = 0.92, the 50-component basis itself allows 0.97, and nearly all of the
gap sits in the consistency term, that is in the geometry the scorer derives from the map.

## Completeness and structure

| Id | Finding | Evidence | Consequence |
| --- | --- | --- | --- |
| F01 | The download is complete and matches the published counts. | 7,041 DIII-D training shots, 874 DIII-D test shots, 1,206 MAST test shots. Test frame totals are 193,359 and 69,075. No shot id is shared between splits. No Parquet nulls. [0.1, 0.12, 0.11b] | None. |
| F02 | Labels are the columns present in training files and absent from test files. | 11 columns on DIII-D: `efit_psirz`, `efit_q95`, `efit_beta_n`, `efit_li`, `efit_r_axis`, `efit_z_axis`, `efit_lcfs_n`, `efit_lcfs_r`, `efit_lcfs_z`, `magnetics_dsep`, `magnetics_dsep_times`. [0.14] | Model inputs are restricted to columns present in the test files. `magnetics_dsep` is a label despite its name. A guard in the feature builder enforces this. |
| F03 | Shot ids are anonymized codes. | Ids have the form `21f23b2392`. | Shots cannot be ordered by date from the id. |
| F04 | The flux map is stored as [frame, Z index, R index] on one fixed grid per machine. | Shape (T, 65, 65) on all 7,044 labelled shots. DIII-D grid: R 0.84 to 2.54 m, Z -1.6 to 1.6 m. MAST grid: R 0.06 to 2.0 m, Z -2.0 to 2.0 m. The stored axis lies within half a grid cell of the flux extremum on a typical shot. [0.5, 0.10] | One grid per machine. |
| F05 | On 40 shots the lowest flux value on the map is not at the plasma centre, and on 23 of them it lies at the grid edge next to a coil. | The scorer's interior axis estimate (`find_o_point`) is within one grid cell of the stored axis on every shot. Flux maps plotted for two of the shots; the stored axis sits at the centre of the plasma in both. [0.5] | The magnetic axis must be located as the interior extremum. Coil flux is large across the whole map, so it affects the pixel score well outside the plasma. |
| F06 | Coil geometry is identical in every shot of a machine and includes the solenoid. | DIII-D: 19 elements, one per current column, including `magnetics_ECOILA`. MAST: 812 elements (156 for the ten P-coils, 656 for the solenoid). The toroidal field coils and `efps` have no geometry. [0.9] | The Green's function table is computed once per machine. Solenoid flux can be calculated on both machines. |
| F07 | Current magnitudes are consistent with kA and kA-turns. | Median 99th percentile of the plasma current: 1,060 on DIII-D, 751 on MAST. Largest F-coil value about 570. [0.4] | Confirmed by the coil-flux fit for the DIII-D shaping coils (F47) and the plasma current (F46). |

## Population: test against training

| Id | Finding | Evidence | Consequence |
| --- | --- | --- | --- |
| F08 | DIII-D test shots match training shots on every marker examined. | Share recorded at 20 kHz: 70.0% train, 67.0% test. 44-channel core Thomson: 89.9% and 90.4%. 10-channel edge Thomson: 90.7% and 91.9%. Median frames per shot: 241 and 240. Positive Ip after the alignment: 91.2% and 91.3%. Shots with no valid Thomson profile: 0.75% and 0.7%. | A random split by shot is expected to represent the leaderboard. |
| F09 | Some shots appear to be repeats of the same experiment, with members in both splits. | Several short shots share nearly the same EFIT time window across training and test. Confirmed and measured in the split phase (F56). | Near-repeats are equally common in test and training, so a random split by shot represents the leaderboard; a grouped split does not separate them (F56, F57). |

## Timing

| Id | Finding | Evidence | Consequence |
| --- | --- | --- | --- |
| F10 | DIII-D coil currents were recorded at two rates. | 20 kHz on 5,518 shots, 2 kHz on 2,397. MAST is 5 kHz throughout. Ip is sampled every 0.5 ms on all DIII-D shots. | Windows for averaging and for rates of change are defined in milliseconds and are the same for both groups. |
| F11 | After the organizers' fix, the Ip time axis was still late on the DIII-D shots that fix leaves untouched (30.3% of shots, in both training and test). | Before alignment, in that group: Ip at the first EFIT frame was under 1% of the shot's peak on a quarter of shots; Ip began to rise at a median of 188 ms against 15 ms in the corrected group; more than a quarter of shots had EFIT frames before any plasma current. The solenoid begins its swing-down at about 1 ms on the coil clock in both groups. [0.2, notebook section 4] | The Ip axis is aligned in the loader (F12). |
| F12 | Aligning each affected shot so that Ip begins to rise at 14.9 ms removes the defect. | 2,399 shots shifted; median shift -173.4 ms, range -332.9 to +55.6 ms. After alignment, across all DIII-D shots: Ip at the first EFIT frame is at least 17% of peak (median 46%), and no shot has an EFIT frame before the rise. A second rule anchored on the solenoid swing-down agrees within 10 ms on all but 88 shots. [0.2] | Implemented as `fusion_eq.ip_alignment` and applied by `load_shot` to training and test shots alike, using inputs only. The effect on model accuracy is to be measured with the baseline, with and without the alignment. |
| F13 | DIII-D shots are long and MAST shots are short. | Median 241 frames at 20 ms spacing on DIII-D (about 5 s). Median 58 frames at 5 ms spacing on MAST (about 0.3 s). 49 shots have fewer than 10 frames. Gaps between EFIT frames reach 5.2 s. [0.11] | Lag and rate-of-change features are computed from the continuous input signals, by time. MAST may contain a larger share of frames with changing current; to be checked in the analysis phase. |

## Plasma current and field direction

| Id | Finding | Evidence | Consequence |
| --- | --- | --- | --- |
| F14 | DIII-D runs the plasma current and the toroidal field in both directions; MAST runs one. | After the alignment, Ip is positive on about 91% of DIII-D shots and negative on about 9% (training 6,420 positive, 603 negative; test 798 and 76); the mixed group fell from 616 shots to 18. Toroidal field current negative on 57% and positive on 42%. The magnetic axis is at the flux minimum on every DIII-D shot. | The relation between current direction and the stored flux sign is established by the coil-flux test on one shot of each direction. Scalar formulas use the magnitudes of Ip and of the toroidal field. |
| F15 | MAST plasma current lies within the DIII-D range. | 99th percentile of the plasma current per shot: 409 to 969 kA on MAST, 231 to 2,209 kA on DIII-D. | Ip does not require extrapolation on MAST. |

## Thomson scattering

| Id | Finding | Evidence | Consequence |
| --- | --- | --- | --- |
| F16 | Channel layouts vary by shot and by machine. | DIII-D core: 40, 42, 43, 44 or 54 channels. DIII-D edge: 10 or 14. MAST: 130 core and 16 edge. | Thomson enters the model as profile descriptors, not as one column per channel. |
| F17 | Coverage is good on most shots and absent on a few. | About 97% of shots have a valid core profile within 20 ms of at least 90% of their frames. No valid profile at all: 53 training shots, 6 DIII-D test shots, 1 MAST test shot. No Thomson channel at all: 11 training shots, 1 DIII-D test shot. [0.7, notebook section 2] | Features are built from the channels that are valid. The model needs a defined behaviour when Thomson is missing, because test frames still require predictions. Options: a validity flag, and random masking of Thomson during training. |
| F18 | Coverage depends on the time tolerance, most of all on MAST. | Tightening the tolerance from 20 ms to 10 ms raises the MAST shots below 90% coverage from 49 to 331. | Thomson is interpolated in time between valid profiles, not matched to the nearest sample within a fixed tolerance. |
| F19 | The organizers' Thomson channel fix applies to every MAST shot. | 131 channels reduced to 130 on all 1,209 MAST shots; the dropped channel is entirely NaN. [0.3] | Applied in the loader. |

## Missing values in MAST currents

| Id | Finding | Evidence | Consequence |
| --- | --- | --- | --- |
| F20 | 114 of 1,206 MAST test shots contain NaN in the current columns, in a regular pattern. | One sample in six is NaN on every coil current and on Ip; five in six on the toroidal field current. These are the shots with the shorter recording (15,482 samples), where two clocks were merged into one time axis. Ip covers the EFIT window on all of them. [notebook section 7] | Each MAST signal is resampled onto the EFIT frames separately, skipping NaN. |

## Labels

| Id | Finding | Evidence | Consequence |
| --- | --- | --- | --- |
| F21 | Extreme label values are of four kinds. | Physical: q95 up to about 24 at low current, betaN up to about 4.8. Failed reconstructions: betaN down to -14.5 on the same shots where stored li reaches 8 to 21. Placeholder zeros: betaN and li exactly 0 on some frames of at least two shots. Two whole shots (`5df1630aa7`, `955c35c962`) with q95 below about 1 and betaN of 14 to 17. [0.11, notebook section 5] | The analysis phase counts the frames of each kind and checks whether the flux map is also affected. Failed and placeholder frames are to be dropped or down-weighted in training. |
| F22 | Negative betaN is not a sign convention. | 339 training shots contain some negative frames; no shot is negative throughout; it occurs for every combination of Ip and field direction. | betaN and q95 are treated as positive quantities. |
| F23 | Stored boundary and li labels differ from the values the scorer derives from the true flux map. | Over 200 shots: axis position agrees to under 1 mm; the stored contour is about 0.05 m from the scorer's contour at its worst point (mean distance under 3 mm); the scorer's li is about 1% above stored li on a typical shot and up to 30% on some. [0.15] | The score is unaffected, because the scorer applies the same functions to the predicted and the true map. Any loss term or auxiliary target for the boundary or li uses values derived with the scorer's functions, not the stored columns. |
| F24 | Boundary contours are padded arrays. | `efit_lcfs_n` gives the number of valid points (37 to 110). Padding is zero on DIII-D. [0.6] | Contours are masked with `efit_lcfs_n`; zero cannot be used as a mask because real contours cross Z = 0. |

## Documentation

| Id | Finding | Evidence | Consequence |
| --- | --- | --- | --- |
| F25 | The dataset README disagrees with the data in nine places. | All nine assessed against the full data; the data dictionary agrees with the data in each case. [0.13] | `docs/data_dictionary.md` is the reference. Two updates to it are proposed in the QA report (Thomson sample counts and channel counts). |

## Frame table: build and checks

| Id | Finding | Evidence | Consequence |
| --- | --- | --- | --- |
| F26 | The frame table is complete and reproduces the QA run. | 9,121 shots and 1,821,774 frames (1,559,340 / 193,359 / 69,075), no shot errors, 329 s. Keys unique. Labels are exact copies on 50 sampled training shots; test tables carry no label column. Recording rates (5,518 and 2,397), Ip directions (training 6,420 / 603 / 18, test 798 / 76) and MAST shots with NaN currents (114) match QA. Lowest Ip at a first frame is 0.172 of the shot's peak. [T1 to T6] | The table is the input to every later phase. |
| F27 | Every frame has complete current inputs. | 100% of frames on all three splits, including the 114 MAST shots with NaN in the raw currents. [T4] | No imputation is needed for currents. |

## Resampling of currents

| Id | Finding | Evidence | Consequence |
| --- | --- | --- | --- |
| F28 | The nearest-sample and windowed-mean forms agree. | On DIII-D the median absolute difference is 0.1% to 1.3% of the column's spread on every signal and the 99th percentile is under 7%, at both recording rates; training and test are nearly identical. Frame values lie on the raw traces for one shot of each smoke kind. [T8, notebook section 2] | The windowed mean over 2 ms is adopted provisionally. The choice is confirmed on the ridge score in the baseline phase, after which the nearest form is dropped from the build. |
| F29 | Exact zeros in the nearest-sample values are idle coils, not dropped samples. | Between 0.05% and 0.21% of training frames on F3A, F3B, F4A, F4B, F5A, F5B, F8A and F9B, under 0.04% on the other coils. On the smoke shots the affected coils carry 0.5 to 3.4 kA-turn in that shot, against hundreds when in use. | No handling needed. |
| F30 | The MAST P3 coils carry almost no current during the plasma. | `magnetics_p3l_current__mean` and `p3u`: 1st to 99th percentile about -0.6 to 1.6 kA-turn, extremes -21 to 39. The two resampling forms differ by 11% to 16% of the spread at the median, which is noise on an idle coil. | Kept in the coil-flux calculation, in the windowed-mean form. |
| F31 | The 114 MAST shots with merged clocks report a nominal recording rate of 1,900 Hz. | `mag_rate_hz` is 1,900 on 114 MAST shots, the same count as `mag_has_nan`. [T5] | 1,900 Hz is not a recording rate but the sample count of the merged axis over its span. `mag_has_nan` identifies these shots. |

## Rates of change

| Id | Finding | Evidence | Consequence |
| --- | --- | --- | --- |
| F32 | The 10 ms rate window is too short. | On flat-top frames (`ip_over_peak` above 0.9; 1,214,019 of 1,559,340 training frames) `ip__rate` has a standard deviation of 748 kA/s in the 20 kHz group and 1,067 kA/s in the 2 kHz group, as large as a typical ramp of 500 to 1,000 kA/s; the 1st to 99th percentile band is -1,940 to 1,806 kA/s and -2,179 to 2,011 kA/s. `magnetics_F1A__rate`: 162 and 212 kA-turn/s. The 2 kHz group is 1.3 to 1.4 times wider. In the 2 kHz example shot, flat-top frames are among the six largest rates. [notebook section 3] | The `__rate` columns are not used as built. A 50 ms rate is added beside the 10 ms one at the next rebuild, and both are tested in the time-dependence phase. The toroidal field rate is not a feature, since that current is constant within a shot. |
| F33 | MAST plasmas change several times faster than DIII-D plasmas. | 1st to 99th percentile of `ip__rate`: -2,166 to 2,234 kA/s on DIII-D training, -8,935 to 7,048 on MAST. `sol__rate`: -66 to 47 against -461 to 224 kA/s. | Raw rate features do not transfer. They need scaling before any use on MAST. |

## Thomson scattering in the frame table

| Id | Finding | Evidence | Consequence |
| --- | --- | --- | --- |
| F34 | Interpolation in time matters on MAST and hardly at all on DIII-D. | Median distance from a frame to the nearest valid core slice: 0.3 ms on DIII-D, 1.4 ms on MAST (99th percentile 20 ms and 17 ms). On MAST the interpolated and nearest core pressures differ visibly on frames more than 10 ms from a slice. [notebook section 4] | The interpolated form is adopted, consistent with F18. |
| F35 | Coverage has levelled off by 20 ms for the core system; the DIII-D edge system needs 25 ms. | Core within 20 ms: 98.5% / 98.7% / 99.1% of frames (training, DIII-D test, MAST); within 10 ms: 97.3% / 97.8% / 91.8%. Edge within 20 ms: 98.0% / 97.7% / 98.2%; within 10 ms: 76.3% / 75.5% / 91.0%. Median gap between valid DIII-D edge slices is 25.5 ms. Frames in shots with no valid slice: core 0.44% / 0.34% / 0.08%, edge 0.70% / 0.48% / 0%. | Validity uses 20 ms for core and 25 ms for edge, derived from the stored distance columns. No rebuild is needed. |
| F36 | 69 shots have no valid core profile within tolerance on any frame. | 62 training, 6 DIII-D test, 1 MAST. 63 shots have no valid edge profile. QA reported 53 training shots with a coverage value of zero plus 15 with no value, because no slice lies inside the EFIT window; the frame table counts slices at any time. [T7] | The frame table count stands; the difference is one of definition on one- and two-frame shots (open item 7, resolved). |
| F37 | DIII-D core Thomson summaries contain rare extreme values. | 99th percentile against maximum on training: `ts_core_te_peak` 3,967 against 12,212 eV; `ts_core_ne_peak` 1.0e20 against 8.4e20 m^-3; `ts_core_pe_mean` 17.1 against 251 kPa. MAST has none (`ts_core_pe_mean` maximum 7.5 kPa). | Thomson inputs are clipped or scaled robustly before use. |
| F38 | The "edge" Thomson system does not measure the same region on the two machines. | Median `ts_edge_te_mean`: 1,980 eV on DIII-D training (1st percentile 462), 143 eV on MAST. | Edge summaries are not used as inputs as they are. Profile descriptors on a normalized radius replace them. |

## Ranges across machines

These rows compare MAST test inputs with DIII-D training inputs. Nothing was fitted on test
inputs.

| Id | Finding | Evidence | Consequence |
| --- | --- | --- | --- |
| F39 | MAST input values lie inside the DIII-D ranges. | `ip__mean` 81 to 1,110 kA against a DIII-D magnitude up to 2,233. `sol__mean` -48 to 43 against -67 to 66 kA. Core `te_peak` median 699 against 1,513 eV, 99th percentile 1,387 against 3,967. `ne_peak` median 2.95e19 against 3.77e19 m^-3. `pe_mean` median 1.42 against 3.58 kPa, 99th percentile 4.7 against 17.1. | No extrapolation in these raw values. MAST occupies the low-temperature, low-pressure end of the DIII-D range. |
| F40 | The MAST toroidal field current is nearly constant and runs one way. | `tf__mean` 1st percentile -85.0, median -84.8, 99th percentile -69.4 kA. `ip_direction` is +1 and `tf_direction` -1 on every MAST shot (QA: Ip minimum at or above 0 and toroidal field maximum at or below 0 on all 1,206). DIII-D runs both directions of each. | On MAST, variation in q95 comes from Ip and shape. The conversion from current to field in tesla is set per machine. |

## Relationships (DIII-D training)

| Id | Finding | Evidence | Consequence |
| --- | --- | --- | --- |
| F41 | Simple physical combinations carry much of the signal. | Over 64,616 frames from 300 shots with `ip_over_peak` above 0.5: flux range against absolute Ip, r = 0.900; q95 against absolute tf over absolute Ip, r = 0.867; betaN against core `pe_mean`, r = 0.747 (63,870 frames with a valid profile). The flux range shows vertical streaks at fixed Ip. [notebook section 6] | Supports physics combinations for the scalars. The streaks are flux changing at constant plasma current, which the coil-flux subtraction is expected to remove. |
| F42 | Coil currents are strongly correlated with each other and with Ip. | Over 1,504,702 frames: F1A with F1B 0.961, F6A with F6B 0.935, Ip with F6A 0.941 and with F6B 0.926 in magnitude. The toroidal field current is uncorrelated with the rest. Every coil's 1st to 99th percentile range crosses zero. | Linear models need regularization. Dividing coil currents by Ip removes the dominant shared factor. Coil currents reversing together with Ip is consistent with a sign rule tied to the Ip direction, confirmed by the coil-flux phase (F45). |

## Labels and frame filters

| Id | Finding | Evidence | Consequence |
| --- | --- | --- | --- |
| F43 | Label extremes lie outside the 1st to 99th percentile band. | q95 2.91 to 11.76 (extremes 0.17 and 23.9); betaN 0.08 to 3.31 (-14.5 and 16.7); li 0.64 to 1.73 (0 and 20.8). | Fewer than 1% of frames on each tail. Feeds the label-quality step. |
| F44 | Some training frames have almost no plasma current. | `ip_over_peak` minimum 0.000 on training (1st percentile 0.363); minimum 0.113 on DIII-D test and 0.121 on MAST. | A minimum-Ip rule is needed for training frames, since the plasma flux is divided by Ip. The affected frames are end-of-shot ramp-downs: 13 below 0.1 and 1,303 below 0.2 (open item 9, resolved). |

## Coil flux (phase C, DIII-D training shots)

| Id | Finding | Evidence | Consequence |
| --- | --- | --- | --- |
| F45 | The stored flux map is the physical flux times minus the sign of Ip. | The current density integrated inside the scorer's boundary has the sign of Ip after that orientation on 8 of 8 reference shots (5 positive, 3 negative Ip), 24 of 24 frames of the single-frame tests and every one of the 1,438 reference frames. The scorer's `AXIS_SIGN` matches positive-Ip shots only. [K4] | Inputs and targets are put in one frame (the physical one) before any model sees them. A sign-handling step precedes every model. |
| F46 | The current carried by the stored map matches the measured Ip. | Median ratio of the integrated current density to the resampled Ip 0.997 (5th to 95th percentile 0.990 to 1.008); 0.996 on the 2 kHz shots, 0.998 at 20 kHz. [K3] | The Ip units (kA) and the Ip time alignment of the loader are confirmed at the frame level. |
| F47 | The DIII-D shaping coils are correct as stored, in kA-turn at scale 1. | With a per-frame constant every per-coil scale lies within 0.924 to 1.095. Without the constant the per-coil scales are not identifiable: F1A 0.41 trades against the solenoid (cosine 0.68 between the centred flux patterns) and the per-shot IQR reaches 3.4 (F3A). [K6] | Shaping coils enter the coil flux at scale 1. Per-coil fitted scales are not used. |
| F48 | The solenoid's flux is flat across the grid, 0.00897 Wb/rad per kA of `magnetics_ECOILA`. | A single scaled rectangle overstates its shape: K5 flat-top 0.037 with the rectangle at scale 147 against 0.0089 with the flat term. Adding the rectangle pattern to the flat term changes nothing (0.0088; pattern coefficient -9.3). [K5, K6] | Adopted DIII-D coil model `solenoid_flat`: shaping coils as stored plus the flat term, saved as `fusion_eq/resources/coil_scales_d3d.json` and applied by `coil_flux` by default. MAST keeps every scale at 1 and no flat term. |
| F49 | With the adopted model the calculated coil flux agrees with the flux outside the plasma to about one percent of the frame's range. | Median rms over the flux range on the nodes away from the coils, fit frames: flat-top 0.0089, late 0.0090, rising 0.0226; near the coils 0.0096. Reference frames 0.0121, 0.0104, 0.0222. On 50 validation shots of the provisional split that took no part in the fit (20 frames each): rising 0.0235, flat-top 0.0090, late 0.0133. Ceiling with a free constant per frame: 0.0063 (per-coil scales) to 0.0066 (solenoid only). [K5] | The physics-first structure is viable: the model learns the plasma's own flux. |
| F50 | The earliest rising frames with a large solenoid current keep a leftover of 5 to 7 percent. | Rising frames with a solenoid current above 30 kA: 0.051 on the reference frames (44 frames), 0.074 on fit frames in the 30 to 60 kA band; slope of remainder against model near 0.83. [K5] | Candidate cause: vessel currents induced by the solenoid swing. Candidate for rate-of-change features (open item 10). |
| F51 | The A and B coils of DIII-D are not exact mirrors in the shipped geometry. | F1 to F4: Z_A + Z_B = -5.4 mm; F9: +9.4 mm; F7: R_A - R_B = -10.1 mm; F6 exact; grid and solenoid symmetric about Z = 0. Swapping A and B currents and flipping in Z changes the coil flux by 1.79 percent of its range (up to 3.0 percent for F1 alone). [K8] | Mirroring is exact only when the coil flux is recalculated from the swapped currents; a flipped coil flux is not. |
| F52 | About 2 percent of DIII-D frames are double-null; two thirds are lower single-null. | Proxy (the boundary point with the smallest poloidal field) on the 50 random fit shots, 996 frames: lower 68.5 percent, upper 29.7 percent, both 1.8 percent. The 8 reference shots (upper 70.6 percent) are not representative. The scorer has no X-point finder. [D9] | Mirroring adds a comparable number of upper-null examples. The proxy is replaced by a boundary-based count when one is needed. |
| F53 | The MAST label-free checks give the same sign and order as DIII-D. | Each machine under its coil model, Ip signed: Bz ratio at the reference point, medians -1.35 (MAST) against -2.09 (DIII-D); solenoid flux ratio -0.091 (MAST) against -0.137 (DIII-D, flat term). One DIII-D Bz ratio near +1.7 in the first run was shot `1ca444c006`, a negative-Ip shot normalized by the magnitude of Ip. [K9] | MAST turn conventions are free of gross errors (factor 4 to 23 on a P-coil, 656 on the solenoid) and no more. The checks at the first MAST prediction are listed under "Carried into modelling". |

## Split and near-repeats (phase B, DIII-D)

Shots are compared through a fingerprint: 21 input signals (Ip, the solenoid, the toroidal
field and the 18 shaping coils) sampled every 100 ms from 100 to 9,000 ms, each scaled by its
training interquartile range; the distance between two shots is the root mean square
difference over their common times, so that 0.1 means the currents differ by a tenth of
their range. Shots covering fewer than five fingerprint times have no distance (65 of
7,041 training shots, 4 of 874 test shots). A near-repeat at threshold t is another shot at a
distance below t.

| Id | Finding | Evidence | Consequence |
| --- | --- | --- | --- |
| F54 | DIII-D test inputs match training inputs on all 24 quantities compared. | Per-shot medians over flat-top frames of Ip, the solenoid, the toroidal field, the 18 shaping coils and the three core Thomson summaries (6,959 training against 865 test shots; 6,927 and 861 for Thomson). Largest Kolmogorov-Smirnov statistic 0.041 (`magnetics_F4B`, p = 0.16); every median difference is within 0.054 of the training interquartile range. [S2] | Extends F08 from recording markers to the model inputs. A random draw of training shots represents the test inputs. |
| F55 | The test shots are a random draw from the same pool as the training shots. | Distance from each test shot to its nearest training shot, 5th, 50th and 95th percentiles: 0.0203, 0.0694, 0.2303. Among training shots, to the nearest other training shot: 0.0186, 0.0659, 0.2248. Kolmogorov-Smirnov statistic 0.033 (p = 0.37). [S3] | Answers the shot-order question: the test shots are not a later or otherwise separate campaign. A held-out set drawn at random from training is as far from the training shots as the test set is. |
| F56 | Near-repeats of an experiment are common, and equally common in test and training. | Share of shots with a near-repeat in training, training against test: 33.9 against 31.2 percent at threshold 0.05; 71.0 against 70.7 percent at 0.10; 92.1 against 91.9 percent at 0.20. Complete-linkage groups over all 7,915 fingerprinted shots at 0.10: 4,156 groups, 5,258 shots in a group of two or more, largest group 35; at 0.20: 2,166 groups, 7,040 shots in a group, largest 73. [S4] | Confirms F09. The grouped split mode of `make_split.py` is not used: with near-repeats this dense the groups chain together and do not separate near-repeat pairs, and the leaderboard itself contains them in the same proportion. |
| F57 | Adopted split: random by shot, seed 0, `splits/split_v1.json`. | Drawn 10 October 2026 from the 7,041 training shots, stratified by recording rate and Ip direction: train 5,821, validation 870, local_test 350; working subsets working_train 300 and working_validation 100 inside them. Integrity checks pass (disjoint lists, every shot placed, working lists inside their parents). Near-repeat flags at threshold 0.10 are stored in the file: 4,801 of the 7,041 shots have a near-repeat in `train`; 584 of the 870 validation shots do. Balance, train / validation / local_test / test: share at 20 kHz 0.700 / 0.701 / 0.700 / 0.670; share negative Ip 0.086 / 0.085 / 0.086 / 0.087; median frames 241 / 237 / 243 / 240; share with no core Thomson 0.009 / 0.007 / 0.017 / 0.007. [S5, S6] | Every model is fitted on `train` and reported on `validation`; `local_test` is kept for the final check before a submission. |
| F58 | Near-repeats leak a little into the plain ridge score, and the shots with one score higher. | Leak test with the plain ridge (50 components): arm A trained on all 5,821 training shots, arm B with the 1,974 training shots that are near-repeats of a validation shot removed, arm C three random removals of the same size. A minus B on the 870 validation shots: +0.0038 in S, against a spread of 0.0006 across the C arms (A minus the C mean +0.0002). On the working subsets: +0.0018 against a spread of 0.0018. Validation shots with a near-repeat in training score S = 0.8075 (R2 psi 0.8628), without 0.7262 (0.7591). [S7] | The leak of a linear model is a lower bound; a flexible model can memorize a near-repeat where a ridge cannot. Reporting rule: every score is given for all validation shots and for the shots without a near-repeat in training. |

## Baselines (phase B, DIII-D validation shots)

Four linear models and two representation ceilings, fitted on the 5,821 training shots
(1,289,474 frames) and scored with the organizers' scorer on the 870 validation shots at
every fourth frame (48,186 frames). The flux map is predicted through 50 principal components
of the training maps (fitted on 100,000 frames, seed 0) by a ridge with alpha 1 on the 25
frame-table inputs (21 current means, three core Thomson summaries, the Thomson validity flag);
q95 and betaN come from a separate ridge on the same inputs. The models differ in the inputs
and in the target: `ridge_raw` takes the inputs as stored and predicts the stored map;
`ridge_signed` multiplies the current inputs by sign(Ip); `physics_mean` has no learned map
and predicts the coil flux plus Ip times the mean plasma shape of training; `hybrid_ridge`
predicts the plasma target T = sign(Ip) (psi_physical minus coil flux) with signed inputs and
adds the coil flux back. The ceilings pass the true validation maps through the training basis
of the stored map and of the plasma target, with no model in between. The last four rows take
q95 and betaN from the scalar ridge of `ridge_signed`, so their differences from it come from
the flux map alone.

| Id | Finding | Evidence | Consequence |
| --- | --- | --- | --- |
| F59 | Scores on all 870 validation shots. | S, R2 psi, R2 of q95 and betaN, 1 minus D_LCFS, consistency: `ridge_raw` 0.7838, 0.8326, 0.7445, 0.9583, 0.5921; `ridge_signed` 0.9219, 0.9982, 0.8183, 0.9719, 0.7645; `physics_mean` 0.8000, 0.9910, 0.8183, 0.9285, 0.1971; `hybrid_ridge` 0.9219, 0.9982, 0.8183, 0.9719, 0.7646; `ceiling_psi_basis` 0.9717, 1.0000, 0.8183, 0.9950, 0.9972; `ceiling_plasma_basis` 0.9716, 1.0000, 0.8183, 0.9953, 0.9967. Shots without a near-repeat in training: `ridge_signed` 0.9104, `hybrid_ridge` 0.9104, ceilings 0.9686 and 0.9683. The scorer chose the positive flux sign for every model. | The signed ridge is the reference score for later models. Its gap to the ceiling is 0.050 on all shots and 0.058 on the shots without a near-repeat. |
| F60 | The plain ridge fails on negative-Ip shots; multiplying the current inputs by sign(Ip) removes the failure. | On the 74 negative-Ip validation shots `ridge_raw` scores S = 0.0862 with R2 psi = -0.916 and consistency 0; on the 794 positive-Ip shots 0.8946. `ridge_signed` scores 0.8725 and 0.9239 on the same groups and 0.9219 against 0.7838 on all shots. | Current inputs are multiplied by sign(Ip) before every model (the sign rule of F45 applied to the inputs). The plain ridge is kept as the reference for the starter-kit baseline only. |
| F61 | More training data does not help a linear model. | Plain ridge in the leak test: S = 0.7656 (R2 psi 0.8252) fitted on the 300 working training shots and scored on the 100 working validation shots at every frame; 0.7838 (0.8326) fitted on all 5,821 training shots and scored on the 870 validation shots. Signed ridge: S = 0.9186 (R2 psi 0.9985) with the 300 training shots on the 100 working validation shots; 0.9219 (0.9982) with the 5,821 training shots on the 870 validation shots. Nineteen times the shots add 0.018 to the plain ridge and 0.003 to the signed ridge. [S7] | Model development runs on the working subsets; a full fit is a confirmation, not a source of improvement, for linear models. |
| F62 | The hybrid ridge equals the signed ridge. | S 0.9219 for both; R2 psi 0.9982 for both; consistency 0.7646 against 0.7645; the same on every subset. | Expected: the coil flux is linear in the signed currents, which the ridge already has as inputs, so subtracting it from the target changes nothing a linear model can fit. The equality confirms the target machinery (coil flux, sign rule, plasma target and its inverse) round-trips without loss. The physics structure pays off only with a model that is not linear in the currents. |
| F63 | A basis of 50 principal components is not the limit. | True validation maps through the 50-component basis of the stored map: R2 psi 1.0000, 1 minus D_LCFS 0.9950, consistency 0.9972, S 0.9717 with the ridge's scalars. Through the basis of the plasma target with the coil flux added back: 1.0000, 0.9953, 0.9967, 0.9716. The two bases are equivalent. | 50 components are kept for the plasma part. The representation costs under 0.01 in S; everything above sits in the regression. |
| F64 | The gap to the ceiling is in the consistency term. | `hybrid_ridge` against `ceiling_plasma_basis` on all shots, contribution to S by term: consistency 0.7646 against 0.9967, 0.046 of the 0.050 gap; boundary 0.9719 against 0.9953, 0.002; R2 psi 0.9982 against 1.0000, 0.001; scalars shared. The scalars at R2 0.8183 cost a further 0.027 against a perfect scalar prediction. | Pixels are nearly solved for DIII-D. The score is now in the geometry the scorer derives from the map (axis position, elongation, triangularity, volume, li) and in the two scalars. Later models are judged on consistency and the scalars, not on R2 psi. |
| F65 | Negative-Ip shots score lower than positive-Ip shots under the signed ridge. | `ridge_signed`, 74 negative against 794 positive shots: S 0.8725 against 0.9239; consistency 0.5975 against 0.7716; scalars 0.7133 against 0.8224; R2 psi 0.9982 for both. | Open item 11. The map is as accurate; the derived geometry and the scalars are not. Candidates: the 8.6 percent share of negative-Ip training shots, and inputs that are not flipped (toroidal field, Thomson) interacting with the flipped ones. |
| F66 | The scalars come from separate ridges on the inputs; the physics route is untested. | R2 of q95 and betaN together 0.8183 (`ridge_signed`), 0.7986 on shots without a near-repeat, 0.7133 on negative-Ip shots. The physics combinations of F41 (q95 against the toroidal field over Ip, r = 0.867 on 300 shots) have not been scored. | Scalars are the second largest term of the gap after consistency. Physics formulas are tested with true and then predicted shapes (carried into modelling). |
| F67 | The residual of the hybrid ridge is concentrated at the plasma core. | Mean absolute residual per pixel over the 48,186 scored frames: peak 0.017 Wb/rad at R = 1.72 m, Z = 0; the region above half the peak spans 0.65 m in Z and over 1.2 m in R, reaching the outer edge of the grid on the midplane; median at the grid edge 0.003. By phase the peak is 0.015 on rising frames (8,958), 0.018 on flat-top (37,465) and 0.023 on late frames (1,763). Per-frame rms residual, median: 0.0060 Wb/rad on rising frames (1.4 percent of the frame's flux range), 0.0065 on flat-top (1.0 percent), 0.0081 on late frames (1.5 percent). The mean-shape model peaks at 0.034 at the same point. | What the ridge misses is how current and pressure are spread inside the plasma, not the coil flux outside it. Candidates are listed under "Carried into modelling". |

## Submission, scalars and the current target (phase E, 11 and 12 October)

Working lists of `split_v1.json` for the scalar study and the ceiling (300 training and 100
validation shots, every 4th frame, 16,423 and 5,551 frames with both labels finite); the
full validation list (870 shots, every 4th frame) for the local scores of the two predictors.
Pooled R2 is the scorer's, one R2 over all frames, unclipped. Readings that were not tested
are marked as readings.

| Id | Finding | Evidence | Consequence |
| --- | --- | --- | --- |
| F68 | The hybrid ridge gets the triangularity from the coil flux and the rest of the geometry from the plasma part; negative-Ip shots are weak on the axis and the elongation. | `hybrid_ridge` on the 870 validation shots: S 0.9219, R2 psi 0.9982, R2 q95 and betaN 0.8183 (q95 0.7030, betaN 0.9337), 1 minus D_LCFS 0.9719, consistency 0.7646 with R_axis 0.728, Z_axis 0.668, kappa 0.579, tri_top 0.949, tri_bot 0.950, volume 0.826, li 0.651. `physics_mean` (coil flux plus the mean shape, no learned flux): tri_top 0.702 and tri_bot 0.643 against R_axis -0.759, Z_axis -0.137, kappa -0.214, volume 0.034 and li -0.164. On the 74 negative-Ip shots the hybrid ridge gives R_axis 0.372, Z_axis 0.196 and kappa 0.361 against 0.737, 0.693 and 0.592 on the 794 positive-Ip shots; li 0.661 against 0.649; tri_top 0.890 and tri_bot 0.913 against 0.952 and 0.952. | The shape terms the coil flux already sets need no plasma model; the axis position, elongation, volume and li are the plasma part's job. Open item 11 narrows to the axis and the elongation of negative-Ip shots. |
| F69 | Storing the maps as float16 costs 0.0003 in S, all of it in the shape terms. | The same hybrid-ridge predictions on the 870 validation shots at stride 4: S 0.921537 read back from the npz file against 0.921861 in memory; R2 psi 0.9982 for both and the scalars identical (0.818345); 1 minus D_LCFS 0.971705 against 0.971869; consistency 0.763021 against 0.764562 with R_axis 0.7273 against 0.7283, Z_axis 0.6636 against 0.6677, kappa 0.5759 against 0.5791, tri_top 0.9482 against 0.9491, tri_bot 0.9494 against 0.9503, volume 0.8261 against 0.8262, li 0.6506 against 0.6512. | float16 stays the submission format. The derived geometry reacts to the third significant digit of the map, which the pixel term does not. |
| F70 | The submission's shot order is the sorted order of the local file names, the order of the Hub stream. | `check-order` compared the Hub stream of each public-test config with the local file at the same rank by the length, first and last value of `efit_times` and a hash of `efit_times` and the raw plasma current, and passed on both machines (reported by Kamia; the Hub rows carry no shot id). The build predicted 874 DIII-D shots (193,359 frames, first `0047b7837b`, last `ff6af6ea51`) and 1,206 MAST shots (69,075 frames, `0035568738` to `fffdb8dc56`); Codabench scored 874 and 1,206 shots with none missing, and the lowest per-shot R2 psi among the DIII-D worst 20 is 0.9427, which a shifted order could not give. | Key index = rank of the file name in sorted order of `data/raw/data/<config>/`. No further order check is needed for later submissions built by the same pipeline. |
| F71 | First submission (Codabench 979130): DIII-D 0.9249 as the local validation score predicted; MAST 0.0003 with the coil flux alone; on MAST, where every test shot has positive Ip, the stored map has the sign opposite to DIII-D's rule. | DIII-D (874 shots, scoring version 3.3.1): S 0.9249, R2 psi 0.9981, R2 q95 and betaN 0.8219 (q95 0.7093, betaN 0.9346), D_LCFS 0.0290, consistency 0.7777 (R_axis 0.720, Z_axis 0.707, kappa 0.648, tri_top 0.950, tri_bot 0.961, volume 0.825, li 0.633), no boundary failure, flux sign +1. Local validation of the same model (870 shots, stride 4, through the npz file): 0.9215, 0.9982, 0.8183, 0.0283, 0.7630. MAST (1,206 shots): S 0.0003, R2 psi -1.357 after the scorer applied the sign -1 (-12.85 as submitted), R2 q95 and betaN -0.404 (q95 -0.747, betaN -0.060 at the DIII-D medians 4.95 and 1.21), D_LCFS 1.000 with the boundary failing on 98.3 percent of frames, consistency 0.0013 with the derivation failing on 70.6 percent; G_ratio 0.000272. Every MAST test shot has positive Ip, and no MAST shot with negative Ip exists in the data; the calculated solenoid flux has a relative spread of 0.40 across the grid and carries a median 0.24 of the coil flux range (5th to 95th percentile 0.11 to 0.40). | The local validation score is a reliable estimate of the DIII-D leaderboard score (0.003 below it). Shown: for positive Ip, the MAST stored map is the physical flux itself, where DIII-D's rule (`-sign(Ip)` times the physical flux) gives its negative; this agrees with the scorer's `AXIS_SIGN` (+1 MAST, -1 DIII-D). Not shown: how MAST stores a negative-Ip shot, so the form `+sign(Ip)` is carried over from DIII-D's rule and not tested. The sign rule constant becomes per machine, +1 for MAST, established on positive-Ip shots only. The scorer normalizes the sign, so the placement cost nothing, but every MAST target and sanity check uses the MAST constant from now on. |
| F72 | The coil flux alone scores below zero on DIII-D as on MAST; the MAST value is in the range a correct coil flux without a plasma gives. | `coil_only_v0` on the 870 DIII-D validation shots at stride 4 (adopted coil model with the flat solenoid term, ridge scalars): R2 psi -1.914, 1 minus D_LCFS 0.000, consistency 0.000, S 0.1228 (the 0.8183 scalar term carries all of it). MAST on Codabench with every scale 1 and no flat term: -1.357 after sign normalization. | Shown: the stored map's variance between frames is dominated by the plasma on both machines, so a negative pooled R2 is what the coil flux alone gives and the MAST value is not evidence against the MAST coil model. Not shown: whether the MAST coil flux is right to the one percent of F47 or whether MAST needs a flat solenoid term; that needs the plasma-residual test on the three sample shots or a first plasma model, and the sign constant of F71 applied. |
| F73 | The q95 formula needs a constant of 0.77 and still drifts with the inverse aspect ratio; a correction on quantities without units closes most of the gap. | Validation frames of the working list (5,551). Least-squares multiplier on the formula with the true shapes: 0.771 on the training frames (111 turns if read as a turn count), 0.785 on validation. Stage 1 (true shapes): formula alone R2 0.307, times the constant 0.948, with the ridge correction 0.967, with the gradient-boosted correction 0.978; the ridge baseline on the inputs 0.736 on the same frames. Stage 2 (shapes from the hybrid-ridge maps, constant and corrections refitted on predicted training shapes): 0.326, 0.937, 0.950, 0.969; the stage-1 corrections applied to the predicted shapes 0.951 and 0.961. Negative-Ip frames, stage 1: constant 0.986, gradient-boosted 0.983, ridge baseline 0.744. Label over formula, bin medians: from 0.83 at eps 0.33 to 0.76 at eps 0.37 and from 0.83 at delta 0.3 to 0.77 at delta 0.8; flat in kappa (0.78 to 0.81) and in the Thomson peak-to-mean ratio; from 0.77 at half the peak current to 0.80 at the peak. | The formula's aspect-ratio and triangularity factors over-correct on DIII-D shapes (reading). With a correction the formula beats the ridge by 0.23 in R2 at both stages, but the direct calculation of F74 does as well at stage 1 without any fit. |
| F74 | q95 calculated from the map needs no constant: the median of label over calculated is 0.997 with 144 turns, which confirms the DIII-D turn count. | Field-line integral on the 0.95 surface of the scorer's boundary, 5,551 validation frames, no failure. Stage 1 (true maps): R2 0.9685, negative-Ip frames 0.9942, median label over calculated 0.9974. At 0.94: R2 0.9660, median 1.0391; at 0.96: 0.9293 and 0.9523. Stage 2 (hybrid-ridge maps): R2 0.9388, negative-Ip 0.9722, median 0.9996, no failure. | 144 turns is confirmed for DIII-D to 0.3 percent at the median; the 111 turns of F73 was the formula's shape factor, not the field. The surface matters: 0.01 in psi_N moves the median by 4 to 5 percent. At stage 2 the calculation (0.939) sits below the corrected formula (0.969) because it inherits the shape errors of the predicted map (F76); on negative-Ip frames it is the best at both stages. |
| F75 | betaN from the electron pressure stays behind the ridge on the inputs; predicting betaN times Ip does not help. | Constant factor label over formula (core `pe_mean`): 3.67 on training, 3.50 on validation. Stage 1: formula alone R2 -0.840, times the constant 0.790, ridge correction 0.813, gradient-boosted correction 0.851; the ridge baseline 0.940 on the same frames (0.941 refitted on them). Stage 2: 0.800, 0.850, 0.899; stage-1 corrections on predicted shapes 0.824 and 0.840. Negative-Ip frames: gradient-boosted 0.797 at stage 1 and 0.829 at stage 2 against the ridge's 0.891. A ridge on the inputs predicting betaN times the absolute Ip, divided afterwards: 0.924 against 0.941 direct. | One electron-pressure summary carries less of betaN than the full input set does (reading: the factor of 3.7 is more than the electron share of the pressure, so the summary also under-reads the stored pressure). betaN keeps a learned correction; the question is which of its inputs transfer. |
| F76 | Shapes read from the hybrid-ridge maps agree with the true shapes to R2 0.56 to 0.92. | Pooled R2 of the predicted against the true shape quantities on the 5,551 validation frames: R_geo 0.682, a 0.627, kappa 0.557, tri_top 0.921, tri_bot 0.903, eps 0.747. | The same ranking as the scorer's consistency terms (F68: kappa 0.58, triangularity 0.95). Any scalar derived from the predicted map is capped by this agreement until the plasma part improves. |
| F77 | The candidate label-filter rules cost under 5 percent of the frames on the working lists and are frame filters, not shot filters. | Frames with both labels finite: 16,423 of 16,423 on the 300 training shots, 5,551 of 5,551 on the 100 validation shots. Toroidal field under 50 kA: 10 training frames in 1 shot, none on validation. Ip under 0.1 of the peak: 1 training frame. q95 outside the F43 band: 259 training frames (1.6 percent) in 113 shots, 112 validation frames in 44 shots. betaN outside the band: 338 (2.1 percent) in 88 shots, 161 in 30. The two flagged shots are not in the working lists. Union: 568 training frames (3.5 percent) in 150 shots, 263 validation frames (4.7 percent) in 53 shots. No rule was applied in the scalar study. | The band rules touch half the shots at a few frames each, so they remove ramp frames rather than shots (reading). The lighter filter (impossible values, field, Ip fraction, the two shots) costs almost nothing on these lists; the bands are the open choice (carried into modelling). |
| F78 | Current-distribution ceiling: a map rebuilt by physics from the exact masked current density scores S 0.978, the unmasked current 0.991, the present plasma-flux target 0.999; the PCA of the current costs 0.003 at 20 components and nothing at 50. | Working lists, 100 validation shots, 5,551 frames at stride 4, true q95 and betaN for every variant, coil flux under the adopted model plus the rebuilt plasma flux. S, R2 psi, 1 minus D_LCFS, consistency, then R_axis, Z_axis, kappa, tri_top, tri_bot, volume, li: (a) masked 0.9784, 0.9984, 0.9844, 0.9041; 0.934, 0.999, 0.955, 0.987, 0.983, 0.485, 0.985. (b) unmasked 0.9913, 0.9945, 0.9905, 0.9763; 0.997, 1.000, 0.902, 0.990, 0.983, 0.967, 0.996. (c) masked, divided by Ip, PCA on the native grid fitted on 16,423 training frames: 20 components 0.9755, 0.9984, 0.9790, 0.8924; 0.933, 0.996, 0.939, 0.965, 0.939, 0.492, 0.982. 50 components 0.9780, 0.9984, 0.9826, 0.9029; 0.936, 0.999, 0.953, 0.981, 0.974, 0.493, 0.985. 100 components 0.9780, 0.9984, 0.9832, 0.9026; 0.934, 0.999, 0.953, 0.982, 0.977, 0.488, 0.985. (d) plasma flux through the 50-component basis 0.9990, 1.0000, 0.9953, 0.9971; 1.000, 1.000, 0.991, 0.996, 0.995, 1.000, 0.998. Mask: boundary found on 5,551 of 5,551 frames; share of the absolute current outside the boundary, median 0.063 (5th to 95th percentile 0.038 to 0.114); masked current over the measured Ip, median 0.9975 (0.989 to 1.007). Runtime 85 s with 8 workers. | Recorded without a decision on the target. The numbers to weigh: the mask drops 6 percent of the current and with it the volume term (0.485 against 0.967 unmasked) while the unmasked current keeps the R2 psi lower (0.9945 against 0.9984); the plasma-flux target has no such loss at 50 components (0.9990). |
| F79 | The loss of the masked current (variant a of F78) is the vacuum remainder, the plasma target minus the flux of the masked current: adding it back restores the map exactly, it sits in a few components, and a ridge on the currents predicts most of it. | Run of 12 October on the working lists, the same 5,551 validation frames as F78, true q95 and betaN for every variant. (e) variant a plus the exact remainder: 1.0000 on every term; largest error 1.33e-16 of the frame's flux range. Remainder RMS over the frame's flux range (max minus min of the true map), median: all frames 0.0105 (5th to 95th percentile 0.0040 to 0.0305), rising 0.0193 (1,103 frames), flat-top 0.0093 (4,308), late 0.0143 (140). Variance of the training remainder (16,423 frames) held by the leading components: 5 components 0.9978, 10 components 0.9998, 20 components 1.0000. S, consistency, volume: (f) current at 50 components plus the remainder through its PCA: 5 components 0.9978, 0.9927, 0.996; 10 components 0.9982, 0.9944, 0.998; 20 components 0.9983, 0.9947, 0.998. (g) as f at 10 components with the coefficients predicted by a ridge (42 inputs: Ip, solenoid, toroidal field and the 18 F-coils, `__mean` and `__rate`, currents Ip-signed; alpha 1.0): 0.9973, 0.9911, 0.991. (d) plasma-flux basis at 50: 0.9990, 0.9971, 1.000. (a) 0.9784, 0.9041, 0.485. Ridge, pooled R2 of the 10 coefficients: training 0.8885, validation 0.8842. Runtime 155 s with 8 workers. | The loss of variant a is the remainder. The three-part map (coil flux, flux of the predicted current, remainder) reaches within 0.001 of the plasma-flux ceiling at 10 components (S 0.9982 against 0.9990) and within 0.002 with the remainder predicted by a ridge (0.9973); volume returns from 0.485 to 0.998 and 0.991. Reading, not tested: the remainder is coil-model error plus vessel currents; its median by phase (rising 0.019, flat-top 0.009, late 0.014) matches the coil-model accuracy of F49 on shots outside the fit (0.0235, 0.0090, 0.0133). |

## Features (feature phase, 13 October)

Feature version 1 (`configs/features_v1.yaml`, `docs/FEATURES.md`): 22 coil points and nine
named terms of the coil flux divided by mu0 R0 Ip, Thomson, field and shot-position
features, robust-scaled with DIII-D training statistics of originals and mirrors together.
Reference of every comparison: the 1,289,474 frames of the 5,821 shots of the `train` list
of `split_v1.json`. Medians are over all frames of a split unless a phase is named (rising,
flat-top, late; the rule of `coil_scales.phase_of`). IQR is the training interquartile
range. Readings that were not tested are marked as readings. Feature version 2
(`configs/features_v2.yaml`) is profiled against version 1 in F98, with the same reference.

| Id | Finding | Evidence | Consequence |
| --- | --- | --- | --- |
| F80 | The version 1 feature table is sound: rows and keys, inputs, completeness, layout, mirror, current direction and force-balance sign raise nothing to act on. | Profile of 13 October. X1: rows and keys equal the frame table on every split (1,559,340 DIII-D training, 193,359 DIII-D test and 69,075 MAST rows). X2: 41 columns read, each in `input_columns(machine, include_flags=True)` or `time_ms`. X3: the 20 kA Ip threshold removes 7 training frames and no test frame; the lowest finite share is `ts_beta_p` (0.985 on DIII-D training, 0.991 on MAST), on frames without valid Thomson. X4: smallest clearance 0.279 m on DIII-D and 0.199 m on MAST (rule 0.15 m). X5: mirroring twice returns the input on every split; against points calculated with the A and B currents exchanged (4,522 frames of 20 training shots), the mirrored points differ by 0.21 percent of the pattern's range at the median, 0.41 percent at the 95th percentile and 0.51 percent at most. X6: 96.0 to 99.9 percent of the negative-Ip training values of each named term lie inside the positive-Ip 1st to 99th percentile band; largest median shift `coil_c11`, -0.54 IQR. X7: median 4 pi (R0/a0) `coil_c10` on flat-top frames -2.259 on DIII-D and -1.485 on MAST (K9: -2.096 and -1.351); signs and order match, the order Shafranov's ln(8 R0/a0) predicts. | The version 1 table is used as built. |
| F81 | DIII-D test matches training on every feature. MAST lies inside DIII-D's full range almost everywhere but in its tails, mainly in the even coil terms, and is far from DIII-D in the combined feature space. Reading, for review. | X8 (all frames): DIII-D test outside the training 1st to 99th percentile band at most 0.031 per feature, median shifts within 0.05 IQR. MAST outside the training range at most 0.013 (`coil_c30`) and at most 0.001 on every other feature; outside the band `coil_c20` 0.874 (median shift -3.97 IQR), `coil_pt_12` 0.800, `coil_pt_11` 0.757, `coil_c02` 0.693 (-2.28 IQR), `coil_pt_18` 0.468, `resistive_age` 0.373, `q_cyl` 0.283. X9, MAST outside the band by phase: `coil_c20` rising 0.527, flat-top 0.938, late 0.812; `coil_c02` 0.992, 0.709, 0.104. X10 (scaled features, nearest of 200,000 training frames, 5,000 frames per split, seed 0), median distance on validation, DIII-D test and MAST: set_a 0.230, 0.236 and 1.027, with 0.435 of MAST frames beyond the validation 99th percentile; set_b 0.421, 0.426 and 4.431, with 1.000 beyond. With the mirrored training frames added, MAST's median is 1.026 and 4.426. The coil groups carry 0.67 (set_a) and 0.83 (set_b) of MAST's squared distance at the median. | For review. The model extrapolates on MAST in shape. set_a is the default coil form. The bounded-network and clipping options stay as configuration switches. |
| F82 | The current-free (vacuum) relation `coil_c20` + `coil_c02` = (a0 / 2 R0) `coil_c10` holds on both machines to within the leakage of the order-3 fit. Expressed as one shaping quantity, the stretch terms show that MAST's rising phase matches DIII-D's and that the extrapolation is stronger shaping on MAST's flat-top and late phases. | Notebook 06 section 6b. Median abs(residual) over median abs(`coil_c20`): DIII-D training 0.028, DIII-D test 0.027, MAST 0.081 (0.074 to 0.084 by phase), against 0.026 and 0.061 for the current-free field R^4 - 4 R^2 Z^2 sampled at the layout points and fitted the same way at each machine's radii. Residual median and IQR: DIII-D training -0.00073 and 0.00065, MAST -0.00345 and 0.00074; 0.970 of MAST frames outside the training band (median shift -4.20 IQR). `shaping` = (`coil_c02` - `coil_c20`) / 2: MAST outside the band 0.388 (median shift +1.59 IQR), against 0.874 for `coil_c20` and 0.693 for `coil_c02`; by phase rising 0.031 (-0.08 IQR), flat-top 0.460 (+1.66), late 0.370 (+2.16); DIII-D test 0.025. X10 with `ip_frac` unscaled, set_b median distance on MAST 4.348, and 2.582 with `shaping` in place of `coil_c20` and `coil_c02`; on validation 0.382 and 0.371; MAST beyond the validation 99th percentile 1.000 and 0.932. | Reading: the residual is a machine signature (constant offset, same spread); at the backup radii of F88 it falls inside DIII-D's band. Feature version 2 replaces `coil_c20` and `coil_c02` by `shaping` in set_b and drops the residual; set_b's MAST median distance falls 41 percent (4.35 to 2.58, `ip_frac` unscaled in both; 42 percent against version 1's 4.43). |
| F83 | `ip_frac`, bounded in [0, 1], puts 2 percent of its robust-scaled values beyond 10 on every split, training included; unscaled, it tightens DIII-D's own distances. | X11: share of scaled `ip_frac` beyond 10 in magnitude 0.0215 on training, 0.0219 on validation, 0.0208 on DIII-D test, 0.0205 on MAST (training median 0.982 and IQR 0.054, `scaler.json`). X10 (notebook 06 section 6b), set_a median distance with `ip_frac` unscaled against version 1: validation 0.194 against 0.230, DIII-D test 0.202 against 0.236, MAST 0.956 against 1.027; the validation 99th percentile falls from 1.060 to 0.807, so MAST's share beyond it rises from 0.435 to 0.859. set_b: validation 0.382 against 0.421, MAST 4.348 against 4.431. X16: the scores of both sets are unchanged to the fourth decimal (0.9050, 0.9037), as for any affine rescaling of one ridge input. | `ip_frac` enters unscaled in version 2. |
| F84 | Dividing the coil flux by mu0 R0 Ip removes the factor the raw currents share with Ip, and the shared scale keeps the pattern. | X12 on 200,000 training frames: median abs(r) with signed Ip 0.045 for the coil features (largest 0.154, `coil_c11`) against 0.795 for the raw currents (largest 0.942, `magnetics_F6A`); with abs(Ip) 0.177 against 0.273; median abs(r) between columns 0.346 against 0.532. X13 (notebook 06 section 3): the coil points share one median (0.02165) and one IQR (0.07182); the scaled pattern correlates with the unscaled one at 1.0 on every frame shown. | The scaling decision (unitless by physics first, one shared scale per pattern group) is confirmed. |
| F85 | `shafranov_lambda` against `ts_beta_p` has the same intercept on both machines, but MAST's slope is half to two-thirds of DIII-D's. Reading, for review. | X14, line over the frames inside both 1st to 99th percentile bands; slope, intercept, r, residual standard deviation, median offset from the training line. All frames: DIII-D training 2.964, 0.473, 0.736, 0.229, -0.006; DIII-D test 2.879, 0.473, 0.730, 0.223, -0.014; MAST 1.536, 0.466, 0.607, 0.109, -0.174. Flat-top: 3.043, 0.436, 0.746, 0.222, 0.000; 2.973, 0.433, 0.738, 0.214, -0.011; 1.960, 0.418, 0.715, 0.092, -0.151. Candidate causes, not tested: the plasma's size relative to the nominal a0, the large-aspect-ratio form of the formula at MAST's aspect ratio, and the Thomson viewing geometry. | For review. No feature is scaled with MAST's own statistics. `shafranov_lambda` is `coil_c10` with a per-machine aspect-ratio correction and enters set_b in place of `coil_c10` in version 2. |
| F86 | In plasma-time units MAST's current changes faster than DIII-D's and MAST shots end before the current profile relaxes. Reading. | X15: band width (1st to 99th percentile) of the 50 ms Ip slope on MAST over DIII-D training 9.75 in kA/s and 4.30 as `ip_rate_rel` (the slope over Ip times the resistive time), medians 0.105 on DIII-D training and -0.018 on MAST. `resistive_age` median 0.805 on DIII-D training and 0.181 on MAST (band width 0.21 of DIII-D's); time since the Ip rise, median 2.63 s and 0.25 s. Share of frames with abs(`ip_rate_rel`) above 1, the current changing by more than its own size per resistive time: 0.595 on MAST, 0.392 on DIII-D training (above 0.3: 0.824 and 0.687). Frames off the flat-top 0.248 and 0.222. `ip_rise_ms` median 2.2 ms on MAST and 14.9 ms on DIII-D. | The shot-position group is kept, with arms with and without it compared by submission. Open item 6 is answered. |
| F87 | A linear model on the features keeps the flux map but loses part of the shape: both sets score 0.014 to 0.015 below `hybrid_ridge` on the plasma-flux target, the gap in the shape terms. On the masked-current target the features score the same while the refitted raw-input arm drops. | X16, working lists (300 training shots, 65,262 frames; 100 validation shots at stride 4), 50 components, ridge alpha 1, scalars shared by every arm. Plasma-flux target over mu0 R0 abs(Ip): S 0.9050 (set_a), 0.9037 (set_b), 0.9186 (`hybrid_ridge`); R2 psi 0.9953, 0.9951, 0.9985; consistency 0.6759, 0.6699, 0.7331, with volume 0.673, 0.647, 0.812 and li 0.570, 0.547, 0.652. S by phase, set_a against `hybrid_ridge`: rising 0.8731 and 0.8731, flat-top 0.9042 and 0.9246, late 0.8583 and 0.8942. Masked-current target (the remainder of F79 added back before scoring): set_a 0.9055, set_b 0.9048, `hybrid_ridge` refitted on the same per-Ip target 0.8947; late-phase volume of set_a 0.506 on the plasma-flux target and 0.815 on the masked-current target (set_b 0.444 and 0.791). Coefficient R2 on validation: set_a 0.4953, without the shot-position group 0.5226 (set_b 0.4730 and 0.4970), the same with `ip_frac` unscaled. Share of the per-Ip target's training variance by phase: rising 0.114, flat-top 0.333, late 0.553 (validation 0.249, 0.683, 0.067); coefficient R2 of the raw inputs on training frames 0.390 overall and 0.086 on late frames. | Reading: the raw-input arm's drop on the masked target is confounded by its per-Ip form; the clean facts are that the features carry no remainder information and that late-phase volume recovers when the remainder leaves the target. The shot-position group overfits linearly, independent of `ip_frac` scaling. Features only (no raw currents) stands; the loss weighting across frames is an explicit choice in the model phase. |
| F88 | At the backup MAST radii (R0 0.70 m, a0 0.50 m, Morris 1998) the even coil terms centre on DIII-D's, but `coil_c02`, `coil_c12` and the inboard points move further out, `shaping` flips to below DIII-D's, and the X14 intercept and `shafranov_lambda` drop; set_a's MAST distance is unchanged. | `backup_radii_check.md`: MAST test features rebuilt with the radii passed as an override (DIII-D, `machines.py` and the scaler unchanged); layout clearance 0.162 m (rule 0.15 m). Version 1 radii against backup radii, against DIII-D training: outside the band `coil_c20` 0.874 and 0.012 (median shift -3.97 and -0.29 IQR), `coil_c30` 0.272 and 0.012, `coil_c02` 0.693 and 0.874 (-2.28 and -2.78 IQR); median shift of `coil_c12` +1.76 and +2.58 IQR; inboard points move out (`coil_pt_05` 0.274 and 0.967, `coil_pt_03` 0.047 and 0.751), outboard points move in (`coil_pt_14` 0.444 and 0.008). `shaping` median 0.0235 and 0.0151 against DIII-D's 0.0187 (median shift +1.59 and -1.17 IQR). `vacuum_residual` outside the band 0.970 and 0.000; ratio to `coil_c20` 0.081 and 0.026, against the R^4 - 4 R^2 Z^2 benchmark 0.061 and 0.057. X14 intercept 0.466 and 0.128 on all frames, 0.418 and 0.075 on flat-top; slope 1.536 and 1.730, 1.960 and 2.556; median offset from the training line -0.174 and -0.429. `shafranov_lambda` median 0.633 and 0.236 (training 1st percentile 0.240), outside the band 0.002 and 0.514. X10 with `ip_frac` unscaled, MAST median distance: set_a 0.956 and 0.982; set_b 4.348 and 3.132; set_b with `shaping` 2.582 and 2.209. | The reference radii stay R0 0.85 m and a0 0.65 m (Sykes et al. 2001, Table 1); the backup radii are a case in the target phase's Y11. |
| F98 | Feature version 2 changes what it was meant to change: set_a scores as in version 1, set_b scores 0.0005 lower with 15 inputs, the scaled extremes of `ip_frac` are gone and `shaping` reproduces notebook 06 section 6b. With `shafranov_lambda` in place of `coil_c10`, set_b's MAST distance is 2.72 against 2.58 with `coil_c10` kept. The frame table, the version 2 features and the version 1 targets rebuild byte-identically. | `reports/features/v2/feature_report.md` (configuration hash 01263fe0c4f1) and `linear_check.md` (working lists, as F87; version 1 beside it on the same lists, frames, basis and ridge). X16: set_a S 0.9050 with 29 inputs (version 1 0.9050), set_b 0.9032 with 15 inputs (version 1 0.9037 with 16), `hybrid_ridge` 0.9186. set_b against version 1: consistency 0.6678 and 0.6699, Z_axis 0.5928 and 0.5457, kappa 0.4626 and 0.4900, tri_bot 0.9106 and 0.8934, volume 0.6368 and 0.6473, li 0.5044 and 0.5470; coefficient R2 on validation 0.4634 and 0.4730 (set_a 0.4953 in both). X11: share of scaled `ip_frac` beyond 10 in magnitude 0 on every split (version 1 0.0205 to 0.0219). X9, `shaping` outside the training band on MAST: all frames 0.388 (median shift +1.59 IQR), rising 0.031, flat-top 0.460, late 0.370, as F82. X10, set_b median distance: validation 0.371, DIII-D test 0.381, MAST 2.717 with 0.962 of MAST frames beyond the validation 99th percentile (version 1: 4.431 and 1.000; `shaping` with `coil_c10` kept, F82: 2.582 and 0.932); set_a: validation 0.194, MAST 0.956. X1 to X7 as F80. Determinism check (D7, `reports/determinism/determinism_check.md`): 20 DIII-D training shots and 5 MAST test shots rebuilt in a temporary directory with the functions and settings of the full builds; SHA-256 of the row bytes and every value equal for the frame table (4,585 and 295 rows), the version 2 features (4,585 and 295 rows) and the version 1 target tables (train 2,951, validation 1,481, local_test 153 rows); the fitted target objects match their checksums. The stored builds were made at commits 393318b (frame table), e596e2b (features) and c89e246 (targets); the rebuild ran at c89e246. | Version 2 is used as built. Open reading: `shafranov_lambda` places MAST further from DIII-D in set_b than `coil_c10` does (2.72 against 2.58), and its MAST slope against `ts_beta_p` is half to two-thirds of DIII-D's (F85); which of the two transfers better is for the model phase to show by submission. The competition's determinism rule is met on the three layers checked, and the code changes between the stored builds' commits and c89e246 leave their output unchanged. |

## Targets (target phase, from 13 October)

Specification `docs/TARGET.md` (checks Y1 to Y16). F89 records a property of the plasma
response established at the layout review. F90 to F97 come from the ceiling run of 14 October
on the working lists (variants h to k), the full target build of 14 October and its profile,
and the review of notebook 07; ceiling variants are compared in S, consistency and volume, in
that order. Readings that were not tested are marked as readings.

| Id | Finding | Evidence | Consequence |
| --- | --- | --- | --- |
| F89 | The plasma response (`current_target.PlasmaResponse`) is a discrete solver: it inverts the five-point Delta* stencil on the interior nodes, with the flux on the border nodes from the Green's function of each interior cell's current (J dR dZ at its node). Its difference from the continuous filament Green's function is discretisation error, larger on MAST because MAST's cells are larger relative to R. The response is less symmetric in its two arguments for cells in the first interior rings, next to the border, where the interior solve meets the Green's border: a property of the border treatment, not of the interior solve. | Layout review of 13 October on the two native grids (65 by 65; DIII-D R 0.84 to 2.54 m, Z -1.6 to 1.6 m; MAST R 0.06 to 2.0 m, Z -2.0 to 2.0 m), one ampere in a single interior cell against a filament at the cell's node. Beyond five cells, the largest difference over the largest reference value there, worst of 121 source cells: 8e-3 on DIII-D and 4.9e-2 on MAST; per node the median relative difference is 3e-5 to 8e-5 on DIII-D and 1.5e-4 to 3.9e-4 on MAST (five source cells). Beyond ten cells the largest relative difference per node, worst of the same 121 source cells (every sixth interior node), is 3.9e-3 on DIII-D and 1.8e-2 on MAST. The mutual flux of two of those cells agrees both ways to a median of 3.6e-5 (DIII-D) and 8.8e-5 (MAST); the largest asymmetry, 4.2e-3 and 1.6e-2, involves a cell of the first interior ring next to the border, where the interior solve meets the Green's border; without that ring it is 5.0e-4 and 1.35e-3, and between the 81 source cells outside the first three interior rings 1.7e-4 and 3.1e-4. The response to a current flipped in Z is the flipped response to 3e-15. Border cells carry no current. Variant e of F79 closes to 1e-16 because the current is taken with the same stencil the response inverts. | The response is kept as it is. Check Y9 passes at the discretisation level: against a filament beyond five cells, the largest difference at most 2e-2 on DIII-D and 1e-1 on MAST; beyond ten cells at most 5e-3 and 3e-2; symmetry in the two arguments to 1e-3 between source cells outside the first three interior rings, with the value over all source cells reported beside it; the Z flip exact. Guard for the border treatment: the share of the absolute current of the fit sample in the first three interior rings of each machine's grid (the share as it is on DIII-D, placed on MAST), largest over the frames, at most 1e-4 (recorded by the target build). The profile reports the measured values beside the thresholds. |
| F90 | A current basis of 50 components fitted on originals and mirrors meets the ceiling rule of `docs/TARGET.md` Part 5; placing the share on MAST's lattice and back costs nothing measurable; rescaling the share to a total of one costs volume. | `reports/current_target/ceiling.md`, run of 14 October: 300 training shots, 100 validation shots (5,551 frames at stride 4), true q95 and betaN, 10 remainder components. Symmetric basis on 16,397 training frames and their mirrors; variance held at 50, 60 and 80 components 0.9852, 0.9872 and 0.9898, with 30 and 20, 35 and 25, 44 and 36 even and odd components. Against f (S 0.9982, consistency 0.9944, volume 0.998): h at 50 -0.0002, -0.0006, -0.0006; h at 60 -0.0000, -0.0001, -0.0002; h at 80 +0.0004, +0.0012, +0.0005. i (h at 50 with the reconstructed share placed onto MAST's lattice in scaled units and back, two remaps where MAST has one) against h at 50: -0.0001, -0.0004, +0.0002; share of the total retained through placement and back, median 1.0000 (5th to 95th percentile 1.0000 to 1.0000). j (h at 50 rescaled to a total of one) against h at 50: -0.0010, -0.0040, -0.0226 (volume 0.9748 against 0.9974); total of the reconstructed share, median 0.9976 (5th to 95th percentile 0.9890 to 1.0071). | 50 components are kept (rule: S within 0.0003 of f and consistency within 0.001); 80, the only count above f, is a configuration option. The conservative remap is kept for placement. The predicted share is not rescaled. Reading, not tested: the total of the share departs from one by about one percent either way and carries information on the plasma's size, which rescaling removes from the volume. |
| F91 | For the remainder ridge the 50 ms rates are better than the 10 ms rates on validation, on the working lists and on the full build; in score the difference is 0.0001. With the remainder from the ridge the whole target chain on DIII-D stays within 0.0001 of g and 0.0018 of the plasma-flux basis. | Pooled R2 of the 10 remainder coefficients, training and validation. Working lists (ceiling, variant k): 10 ms 0.8885 and 0.8842, 50 ms 0.9084 and 0.9055. Full build (Y14; 1,286,602 training frames passing the lighter label filter, 870 validation shots): 10 ms 0.8831 and 0.8862, 50 ms 0.9041 and 0.9057. Ceiling against g (S 0.9973, consistency 0.9911, volume 0.991): k with 10 ms rates -0.0002, -0.0006, -0.0009 (S 0.9971); with 50 ms rates -0.0001, -0.0004, -0.0004 (S 0.9972). d (plasma-flux basis at 50): S 0.9990. Both ridges have 42 inputs: the frame table has no 50 ms rate for the toroidal field, whose current is constant within a shot, so the 50 ms ridge keeps `tf__rate` (10 ms). | The build's default window is 50 ms, the better one on validation under the rule of Part 3; the toroidal field keeps its 10 ms rate. Target B stands: h and k are within 0.0002 of f and g. The fallback stays as defined on 12 October. |
| F92 | The full target build passes its build checks, and the full lists reproduce the working-list values of F79. | Build of 14 October (`../data/processed/targets/v1`, commit c89e246, configuration hash b2192c3a01a9, runtime 4,270 s, of which 4,069 s for the tables), profile in `reports/targets/v1/target_report.md`. Y1: train 1,289,474 rows (5,821 shots), validation 191,411 (870), local_test 78,455 (350), keys equal to the frame table; frames passing the lighter label filter 1,286,602, 191,150 and 78,368; no frame without a boundary. Fit sample: 100,000 of the 1,286,602 filter-passing training frames (seed 0) and their mirrors; variance held by the 50 components 0.9836. Y2 (1,000 frames of 50 training shots): coil flux plus Ip G s plus r against the physical map 4.6e-15, `compose_map` against the stored map 3.1e-16 of the frame's flux range (rule 1e-10). Y3, total of the share, median (5th to 95th percentile): train 0.9976 (0.9886 to 1.0064), validation 0.9976 (0.9887 to 1.0067), local_test 0.9977 (0.9891 to 1.0062). Y4: 30 even and 20 odd components, departure from parity 0. Y5: coefficients of the flipped share against c times p, 3.0e-15 of the largest coefficient. Y6: relative error of the share through the basis, median 0.0333 (train), 0.0323 (validation), 0.0336 (local_test), the same on mirrors. Y7: 2.5e-15 (DIII-D) and 1.9e-15 (MAST) of the largest value. Y8: placement on MAST conserves the total to 7.5e-16; mean share on MAST border cells 0. Y12, median RMS of the remainder over the flux range: train 0.0095 (rising 0.0194, flat-top 0.0084, late 0.0116), validation 0.0095 (0.0199, 0.0084, 0.0109), local_test 0.0093 (0.0191, 0.0081, 0.0121); F79 0.0105 (0.0193, 0.0093, 0.0143). Y13: variance of the remainder held by 5, 10 and 20 components 0.9976, 0.9998 and 1.0000 (F79 0.9978, 0.9998, 1.0000). | The build is used as built. Its target rows rebuild byte-identically (F98). |
| F93 | Y9 passes on the build's grids, and the fit sample puts no measurable current in the border rings where the response is less symmetric. | Profile of 14 October, 121 source cells (every sixth interior node); rule in brackets. DIII-D: beyond five cells 1.05e-2 (2e-2), beyond ten cells 3.90e-3 (5e-3), symmetry between the 81 source cells outside the first three interior rings 1.70e-4 (1e-3; over all 121 source cells 4.19e-3), Z flip 2.4e-15 (1e-12). MAST: 3.81e-2 (1e-1), 2.02e-2 (3e-2), 3.09e-4 (1e-3; all source cells 1.64e-2), 3.2e-15. Ring guard, share of the absolute current in the first three interior rings over the 100,000 frames of the fit sample (DIII-D as it is, MAST placed), rule 1e-4: DIII-D largest 0; MAST largest 2.2e-6, 99th percentile 0, pooled 1.9e-9; border cells 0 on both. The layout review of F89 gave 8e-3 and 4.9e-2 beyond five cells and 3.9e-3 and 1.8e-2 beyond ten cells; the source of the difference is not traced. | The rule and guard of F89 stand. The asymmetry of the border rings does not reach the targets. The smallest margin is DIII-D beyond ten cells, at 0.78 of the rule. |
| F94 | On MAST with the version 1 radii the placed current lies inside the grid and clear of every coil element. Its outer edge follows R0 + a0 under each set of constants: 1.47 m with the version 1 radii, 1.20 m with the backup radii. | Y10 and Y11: envelope of the cells inside the 99 percent support of at least 1 percent of 20,000 reconstructed training shares placed on MAST, and centroid of the placed mean share. R0 0.85 m, a0 0.65 m: centroid R 0.865 m, Z 0.000 m; envelope R 0.257 to 1.470 m, height 2.250 m; nearest coil element 0.109 m, grid edge 0.212 m. Centroid, envelope R, nearest coil element: R0 +5 percent 0.907 m, 0.287 to 1.530 m, 0.101 m; R0 -5 percent 0.822 m, 0.196 to 1.439 m, 0.048 m; a0 +5 percent 0.865 m, 0.227 to 1.500 m (height 2.375 m), 0.076 m; a0 -5 percent 0.864 m, 0.287 to 1.439 m (height 2.188 m), 0.139 m. Backup radii (R0 0.70 m, a0 0.50 m, Morris 1998): 0.711 m, 0.227 to 1.197 m (height 1.750 m), 0.079 m, grid edge 0.182 m. No coil element lies inside the envelope in any case. | The reference radii stay R0 0.85 m and a0 0.65 m (Sykes et al. 2001, Table 1); the backup radii stay a sensitivity case. Reading: placement works in scaled units, so the centroid and envelope follow the constants by construction and cannot test them; a label-free test from MAST inputs is the size of the boundary the scorer finds in the mean-current maps (Y16, added on 14 October; the profile is to be rerun). A 5 percent smaller R0 brings the current within 0.048 m of a coil element. |
| F95 | The mean current (`mean_current_v0`, no model) scores S 0.792 on DIII-D validation, with R2 psi 0.991 and consistency terms below zero. On the public test frames the scorer finds a closed boundary in every composed map; on MAST the axis lies at 0.97 m and kappa is 1.87. | Y16, profile of 14 October, 870 validation shots at stride 4, scalars of `baseline_v0`. With the remainder: S 0.7924, R2 psi 0.9913, R2 q95 and betaN 0.8183, 1 - D_LCFS 0.9161, consistency 0.1642 (R_axis -0.743, Z_axis -0.438, kappa -0.385, tri_top 0.609, tri_bot 0.540, volume -0.001, li -0.197). Without the remainder: 0.7911, 0.9903, 0.8183, 0.9151, 0.1608 (volume -0.171). `physics_mean` of F59: R2 psi 0.991, S 0.800. Through the submission file (`reports/submissions/2026-10-14_mean_current_v0/run.json`, local score on the same shots and stride): S 0.7927, R2 psi 0.9913. 2,000 public test frames per machine (seed 0), remainder on DIII-D and none on MAST: boundary found on 1.000 of the frames on both machines; median R_axis 1.737 m (DIII-D) and 0.968 m (MAST), Z_axis +0.003 m and +0.000 m, kappa 1.778 and 1.873. | The MAST path of the target (sign constant +1, placement, machine constants, MAST response) runs end to end and gives maps in which the scorer finds a boundary. The DIII-D score is below the 0.85 gate of Challenge 2, so `mean_current_v0` is a diagnostic of the MAST path, not a ranked entry. Reading: one mean shape per machine has no geometry that varies from frame to frame, so shape terms near or below zero are expected; the remainder adds 0.0013 in S and 0.17 in volume. Reading: R_axis lies outside R0 on both machines (by 0.07 m on DIII-D and 0.12 m on MAST), in the direction of the Shafranov shift. |
| F96 | Reading, from notebook 07 section 3: the leading components of the current basis act as handles on li, R_axis and Z_axis. | Review of notebook 07 section 3 (the mean share and the leading components of the full build's basis, with their parity and share of the variance). Basis: 50 components, 30 even and 20 odd, variance held 0.9836. Under the mirror Z_axis changes sign and li and R_axis do not (`docs/TARGET.md`, Part 4), so a handle on Z_axis is an odd component and handles on li and R_axis are even ones. | Not tested. The test is the correlation of each leading coefficient with li, R_axis and Z_axis over the training frames, which the stored shape numbers of the target tables allow. A component that tracks one of these quantities is a candidate for a loss term on it in the model phase; until tested, the reading does not shape the model. |
| F97 | Reading, from notebook 07 section 5: the remainder consists of blobs localised at the coils and a near-uniform offset over the grid. | Review of notebook 07 section 5 (leading components of the remainder basis, target frame sign(Ip) r in Wb/rad). From the build: median RMS of the remainder 0.0095 of the flux range on the training frames (rising 0.0194, flat-top 0.0084, late 0.0116); 5 components hold 0.9976 of its variance (Y13); the 50 ms ridge on the currents and their rates predicts its coefficients with R2 0.906 on validation (Y14). | Not tested. Reading: the blobs fit the coil-model error read in F79 (a coil whose flux is misplaced or misscaled leaves its error near that coil), and the offset fits a flux constant over the grid such as the flat solenoid term of F47. A uniform offset changes R2 psi and none of the shape quantities: the scorer finds the axis and the boundary by comparing flux levels and takes li from the flux gradient. The MAST remainder stays at zero. A test of the reading is the remainder's mean over the grid against its spread within the frame, by phase. |

### Carried into modelling

- DIII-D coil model: shaping coils as stored plus the flat solenoid term (F47, F48).
- Sign handling: on DIII-D the stored map is `-sign(Ip)` times the physical flux (F45; MAST
  in the first-submission item below); inputs and targets are put in one frame before any
  model sees them. Current inputs (Ip, the solenoid
  and the shaping coils) are multiplied by sign(Ip); the toroidal field and Thomson are not
  (F60).
- PCA with 50 components for the plasma part (F63).
- Split `splits/split_v1.json`; every score reported for all validation shots and for the
  shots without a near-repeat in training (F57, F58).
- Up-down mirroring, to give the model symmetric behaviour; DIII-D has about 2 percent
  double-null frames, 1.8 percent on the fit shots (F51, F52). The method is the feature
  flip of the decisions of 12 October below, corrected on 13 October (F80).
- Earliest rising frames: 5 to 6 percent leftover, a candidate for rate-of-change features
  (F50).
- The core residual (F67) points at how current and pressure are spread inside the plasma.
  Candidates: more of the Thomson profile than three summaries, history and rate-of-change
  features, and a current-distribution target.
- Scalars: test the physics formulas with true and then with predicted shapes (F66). The
  toroidal-field turn count on MAST, needed to turn the coil current into a field, has to
  come from published specifications.
- Transfer needs inputs that mean the same on both machines and an output on a grid scaled
  to each machine. Not yet tested.
- First submission made: the hybrid ridge on DIII-D, the coil flux alone on MAST (F71). The
  three MAST checks of F53 are answered in F71 and F72: solenoid flux relative spread 0.40;
  on MAST, where every test shot has positive Ip, the stored map has the sign opposite to
  DIII-D's rule (sign constant per machine: +1 for MAST, established on positive-Ip shots
  only, with the `sign(Ip)` dependence carried over from DIII-D untested; -1 for DIII-D);
  the coil flux alone scores R2 psi -1.357 on MAST against -1.914 on DIII-D.
- Scalars, from the study of 11 October (F73 to F77):
  - q95 is calculated from the predicted map (field-line integral on the 0.95 surface, 144
    turns on DIII-D, no fitted constant; F74). On MAST the same calculation without a learned
    correction; the MAST turn count is to be verified from a published source before it is
    used, since the calculation is linear in it.
  - betaN stays formula times correction (F75). The correction is to be given transferable
    coil features (quantities without units, and the coil flux where a feature needs it)
    instead of the raw-input ridge; which inputs transfer is untested (reading). Superseded
    on 14 October by one head per machine (F99, below).
  - Training uses the lighter label filter: impossible values, toroidal field under 50 kA,
    Ip under 0.1 of its peak, the two flagged shots (`5df1630aa7`, `955c35c962`). The
    percentile bands of F43 are to be checked against it on the full lists before they are
    added (F77).
- Decisions of 12 October (F78, F79):
  - Plasma model target: the toroidal current density inside the plasma boundary, as a share
    of Ip, turned into flux by each machine's Green's function, plus the remainder, learned on
    DIII-D and set to zero on MAST. Fallback: the plasma-flux target. Before falling back,
    the current model is trained with the loss computed on the flux its predicted current
    produces.
  - Scaling: features are made unitless by physics first. A robust scaler (median and
    interquartile range) is fitted on the DIII-D training frames, originals and mirrors
    together, and applied unchanged to MAST, with one shared scale per pattern group (the
    coil-pattern sample points). MAST's own statistics are used only for a feature where the
    evidence points to an instrument difference.
  - Mirroring is on from the first model, applied on the fly in the data loader. It is a
    flip of the coil-pattern features and the target together: the sample points trade
    places across the midplane and the odd terms change sign; Z_axis changes sign, tri_top
    and tri_bot exchange, q95 and betaN are unchanged. PCA bases are fitted on originals and
    mirrors together. The A/B coil-current swap is not exact (the shipped coils sit 5 to
    10 mm off a mirror, F51) and differs from the feature flip by 0.4 percent of the
    pattern's range at the 95th percentile (X5, F80). Corrected on 13 October; the earlier
    text described the current swap with the coil flux recalculated.
  - The MAST remainder stays at zero for this phase; a physics model of the vessel currents
    is a later step.
- Decisions of 13 October (F80 to F88):
  - Feature version 2 scope: in set_b, `shaping` replaces `coil_c20` and `coil_c02` and
    `shafranov_lambda` replaces `coil_c10`; the vacuum residual is not a feature; `ip_frac`
    enters unscaled; `ts_te_peaking` is a candidate if a mean Te summary exists. set_a
    remains the default coil form (F81 to F83, F85).
  - Machine constants verified from published sources: MAST R0 0.85 m and a0 0.65 m (Sykes
    et al. 2001, Table 1); MAST toroidal-field turns 24 (Morris 1998, consistent with Sykes
    et al. 2001 and Berkery et al. 2021); DIII-D R0 1.67 m and a0 0.67 m (National Academies
    2019 report); DIII-D toroidal-field turns 144 (F74). The source and `verified` fields in
    `fusion_eq/machines.py` carry these references, as the target build records (14 October).
  - Loss: the flux of the predicted current is compared with the flux of the true current
    inside the boundary (Ip G s), not with the full map.
  - Loss weighting: the per-Ip target's training variance sits mostly in low-Ip frames (late
    frames hold 0.553 of it on the working lists, F87); this is the evidence for an explicit
    weighting across frames in the model phase.
- Decisions of 14 October (F90 to F99):
  - Current basis: 50 components fitted on originals and mirrors (30 even, 20 odd), on
    100,000 training frames; 80 components is a configuration option (F90, F92).
  - The predicted share is not rescaled to a total of one (F90).
  - Placement on MAST: the conservative remap (F90).
  - Remainder ridge: 50 ms rates, with the toroidal field at its 10 ms rate; zero on MAST
    (F91).
  - Target B stands; the fallback stays as defined on 12 October (F91).
  - Y9: the rule and the border-ring guard of F89, recorded by every target build (F93).
  - MAST reference radii: R0 0.85 m and a0 0.65 m stand; the backup radii stay a sensitivity
    case. The label-free check that follows is the size of the boundary found in the MAST
    mean-current maps (F94, F95).
  - `mean_current_v0` is the reference of the MAST path; with S 0.792 on DIII-D it is below
    the 0.85 gate of Challenge 2 and is not a ranked entry (F95).
  - Candidate loss terms on li, R_axis and Z_axis through the leading current components,
    once the reading of F96 is tested.
  - Feature version 2 as built; `shafranov_lambda` against `coil_c10` in set_b stays open
    for the model phase (F98).
  - betaN: the raw-input ridge on DIII-D; on MAST the formula with explicit geometry times
    the gradient-boosted correction on the transferable inputs. The constant 3.71 is fitted
    on DIII-D and does not transfer. Before any submission that uses the MAST head, stage 2
    is rerun with the shapes of the maps that submission carries (F99).
  - q95 unchanged: calculated from the predicted map (F74).

## Scalars: betaN heads (14 October)

`reports/scalars/betan_heads.md` (`scripts/betan_heads.py`): four ways of predicting betaN on
the working lists at stride 4, in the two stages of F73 to F75 (stage 1 with the shape from
the true maps, the label-side bound; stage 2 with the shape from the `hybrid_ridge` maps).

| Id | Finding | Evidence | Consequence |
| --- | --- | --- | --- |
| F99 | betaN takes one head per machine: on DIII-D the ridge on the raw inputs (R2 0.941); on MAST the formula with explicit geometry times a gradient-boosted correction on transferable inputs, which scores 0.911 on DIII-D validation and 0.873 on the MAST-like corner. The formula's fitted constant, 3.71, is the part that does not transfer. | 16,397 training frames after the lighter label filter (26 of 16,423 removed), 5,551 validation frames with both labels finite. Base `100 (2 mu0 p_e / Bt0^2) a Bt0 / abs(Ip)`, with p_e the core Thomson mean pressure (Pa), Bt0 the vacuum field at R0 (144 turns, R0 1.67 m), `a` the stage's minor radius (m) and Ip in MA; elongation enters the corrections, not the base. One constant per stage, least squares on the training frames: 3.7139 (stage 1), 3.7093 (stage 2). Transferable inputs (v2 features where MAST sits inside the DIII-D band, X8, plus map shapes): `ts_pe_peaking`, `ts_valid`, `shafranov_lambda`, `greenwald_frac`, `ip_frac`, `ohmic_flux_used`, `shaping`, `q_cyl` clipped to the training band, the stage's kappa and a / a0. Corner, a label-free transfer proxy: `q_cyl` below the training 10th percentile (1.780) and `ts_beta_p` below the training median (0.1088), 302 validation frames. Stage 2 R2 on all frames, the 423 negative-Ip frames and the corner: ridge on the 25 raw inputs 0.9408, 0.8926, 0.8740; ridge on the 10 transferable inputs 0.8524, 0.7024, 0.6801; formula times a ridge correction 0.8017, 0.8510, 0.8558; formula times the gradient-boosted correction 0.9108, 0.9133, 0.8729; base times the constant alone 0.8130, 0.6062, 0.8147. Stage 1, formula times the gradient-boosted correction: 0.9004, 0.9089, 0.8874; the raw-input ridge reads no shape and is the same at both stages. | DIII-D keeps the raw-input ridge. MAST takes the formula times the gradient-boosted correction on the transferable inputs. Reading: the two are level on the corner (0.873 and 0.874) and the formula arm leads on negative-Ip frames (0.913 against 0.893), while the raw currents do not mean the same on MAST. The constant 3.71 carries what one electron-pressure summary misses of the stored pressure on DIII-D (F75) and cannot be checked on MAST without labels. Before any submission that uses the MAST head, stage 2 is rerun with the shapes of the maps that submission carries, since the heads were scored on `hybrid_ridge` shapes. q95 is unchanged: calculated from the predicted map (F74). |

## Use of MAST labels

The three MAST sample shots in the starter kit carry equilibrium labels. In this phase they
were read only to confirm storage conventions (array shapes, padding, the dsep sentinel).
Under the challenge rules of 23 September 2026, MAST equilibrium data is not used to train,
calibrate, validate or select any component that produces MAST predictions, and no later
phase of this project uses it for those purposes.

## Open items

1. 135 DIII-D shots are listed for review by check 0.2: 88 where the two alignment rules
   differ by more than 10 ms, 2 with no solenoid anchor, and the remainder corrected by the
   organizers' fix but with an Ip rise outside 0 to 60 ms.
2. Resolved. After the alignment, Ip is positive on 6,420 training and 798 test shots and
   negative on 603 and 76; the mixed group fell from 616 shots to 18 (F14).
3. Resolved with item 7. The 15 training shots have no computable QA coverage value because
   no Thomson slice lies inside their EFIT window; 10 of them have no valid slice within
   20 ms of any frame either, and 5 are one-frame shots with a valid slice within 1.2 ms.
4. Resolved with item 8. On the two shots with q95 below 1 the toroidal field signal is
   implausible for their plasma current.
5. Frame-level counts of failed and placeholder labels (F21).
6. The repeat-shot hypothesis (F09) is resolved by the split phase (F56): near-repeats are
   common and equally common in test and training. The share of changing-current frames on
   MAST (F13) is answered by F86: 0.595 of MAST frames have the current changing by more than
   its own size per resistive time, against 0.392 on DIII-D training.
7. Resolved. The 62 training shots without a valid core Thomson profile (frame table) and
   the 53 reported by QA differ by definition only, on one- and two-frame shots. Frame table:
   `ts_core_any_valid` is false when no frame has a valid slice (at least half the channels
   with Te and ne present) within 20 ms, counting slices at any time. QA 0.7: the coverage
   value counts only slices inside the EFIT window and is absent when none lies there
   (15 shots). Shot by shot: 52 shots have a zero QA value and no valid slice under either
   definition; 10 of the 15 shots without a QA value have no valid slice within 20 ms either
   (52 + 10 = 62); 5 of the 15 are one-frame shots whose single frame is within 1.2 ms of a
   valid slice (`316f541fd6`, `8c2c362dfd`, `98106629ab`, `b722acb466`, `d4fc5902f4`); and one
   two-frame shot, `f0a33e7fe4`, has five slices inside its 20 ms window, none valid, with
   valid slices 8 to 9 ms outside it. The frame table count of 62 stands. [notebook 02
   section 8]
8. Resolved. 1,592 training frames have `tf__mean` below 50 kA in magnitude, in 11 shots.
   Nine are real low-field or field-ramp shots: `d597b8ec22`, `8dffbfc783` and `e869356e6e`
   run the whole shot at 46 to 50 kA, `6fb57bed55` and `4173ac3d16` at 30 kA, `c667d63ad1`
   at 34 to 43 kA, and `341645ad3f`, `2c59574ab1` and `2da8d37fa8` fall to 44 to 50 kA over
   their last 13 to 39 frames; their q95 is 1.9 to 3.9. The other two are the shots with q95
   below 1, `5df1630aa7` and `955c35c962`: the field signal runs at 3 to 27 kA (median about
   18 kA) through the whole shot while the plasma current is about 1,150 kA, which is
   implausible for that current, and their labels scale with the signal (q95 median 0.64 and
   0.72, maximum about 1.0; betaN median 6.0 and 6.2). The two shots are candidates for
   exclusion from scalar training; their flux maps are still to be checked; the decision
   belongs to the label-quality step (B5). They are left out of the label-quality statistics
   (F21). [notebook 02 section 8]
9. Resolved. Frames with `ip_over_peak` below 0.1: 13 in 11 shots, all in the last tenth of
   their shot. Below 0.2: 1,303 frames (0.084% of training) in 277 shots, 1,229 in the last
   tenth, 63 elsewhere and 11 in the first tenth; at most 45 frames in one shot. They are
   end-of-shot frames where the plasma current is ramping down. A minimum-Ip rule for
   training frames (F44) removes under 0.1% of frames. [notebook 02 section 8]
10. The leftover of 5 to 7 percent on the earliest rising frames with a large solenoid current
    (F50): whether it is induced vessel current, and whether rate-of-change inputs or an
    explicit vessel model remove it.
11. Negative-Ip shots score lower than positive-Ip shots under the signed ridge, in the
    consistency term and the scalars with an equal R2 psi (F65): whether the cause is the
    small share of negative-Ip training shots or the inputs that are not flipped.
