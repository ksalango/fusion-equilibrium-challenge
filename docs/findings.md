# Data findings register

Findings from the data quality phase of the Fusion Equilibrium Challenge project, with the
evidence for each and its consequence for data preparation and modelling. Later phases add
their findings to this file.

- Period covered: 4 to 6 October 2026.
- Evidence: `reports/qa/qa_report.md` (full run over 9,124 shots, with the Ip alignment
  enabled), `reports/qa_before_ipfix/qa_report.md` (the same run before the alignment),
  `notebooks/01_qa.ipynb`; from 6 October, `reports/frame_table/frame_table_report.md`,
  `reports/frame_table/column_profile.parquet` (full build, commit 12da376) and
  `notebooks/02_frame_table.ipynb`.
- Check ids in brackets refer to `docs/QA.md` (0.x) and `docs/FRAME_TABLE.md` (Tx).

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

## Completeness and structure

| Id | Finding | Evidence | Consequence |
| --- | --- | --- | --- |
| F01 | The download is complete and matches the published counts. | 7,041 DIII-D training shots, 874 DIII-D test shots, 1,206 MAST test shots. Test frame totals are 193,359 and 69,075. No shot id is shared between splits. No Parquet nulls. [0.1, 0.12, 0.11b] | None. |
| F02 | Labels are the columns present in training files and absent from test files. | 11 columns on DIII-D: `efit_psirz`, `efit_q95`, `efit_beta_n`, `efit_li`, `efit_r_axis`, `efit_z_axis`, `efit_lcfs_n`, `efit_lcfs_r`, `efit_lcfs_z`, `magnetics_dsep`, `magnetics_dsep_times`. [0.14] | Model inputs are restricted to columns present in the test files. `magnetics_dsep` is a label despite its name. A guard in the feature builder enforces this. |
| F03 | Shot ids are anonymized codes. | Ids have the form `21f23b2392`. | Shots cannot be ordered by date from the id. |
| F04 | The flux map is stored as [frame, Z index, R index] on one fixed grid per machine. | Shape (T, 65, 65) on all 7,044 labelled shots. DIII-D grid: R 0.84 to 2.54 m, Z -1.6 to 1.6 m. MAST grid: R 0.06 to 2.0 m, Z -2.0 to 2.0 m. The stored axis lies within half a grid cell of the flux extremum on a typical shot. [0.5, 0.10] | One grid per machine. |
| F05 | On 40 shots the lowest flux value on the map is not at the plasma centre, and on 23 of them it lies at the grid edge next to a coil. | The scorer's interior axis estimate (`find_o_point`) is within one grid cell of the stored axis on every shot. Flux maps plotted for two of the shots; the stored axis sits at the centre of the plasma in both. [0.5] | The magnetic axis must be located as the interior extremum. Coil flux is large across the whole map, so it affects the pixel score well outside the plasma. |
| F06 | Coil geometry is identical in every shot of a machine and includes the solenoid. | DIII-D: 19 elements, one per current column, including `magnetics_ECOILA`. MAST: 812 elements (156 for the ten P-coils, 656 for the solenoid). The toroidal field coils and `efps` have no geometry. [0.9] | The Green's function table is computed once per machine. Solenoid flux can be calculated on both machines. |
| F07 | Current magnitudes are consistent with kA and kA-turns. | Median 99th percentile of the plasma current: 1,060 on DIII-D, 751 on MAST. Largest F-coil value about 570. [0.4] | The scale is confirmed by the coil-flux test in the analysis phase. |

## Population: test against training

| Id | Finding | Evidence | Consequence |
| --- | --- | --- | --- |
| F08 | DIII-D test shots match training shots on every marker examined. | Share recorded at 20 kHz: 70.0% train, 67.0% test. 44-channel core Thomson: 89.9% and 90.4%. 10-channel edge Thomson: 90.7% and 91.9%. Median frames per shot: 241 and 240. Positive Ip after the alignment: 91.2% and 91.3%. Shots with no valid Thomson profile: 0.75% and 0.7%. | A random split by shot is expected to represent the leaderboard. |
| F09 | Some shots appear to be repeats of the same experiment, with members in both splits. | Several short shots share nearly the same EFIT time window across training and test. Hypothesis, not yet tested. | A split that keeps similar shots together is the stricter test of generalization. To be checked in the analysis phase. |

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
| F42 | Coil currents are strongly correlated with each other and with Ip. | Over 1,504,702 frames: F1A with F1B 0.961, F6A with F6B 0.935, Ip with F6A 0.941 and with F6B 0.926 in magnitude. The toroidal field current is uncorrelated with the rest. Every coil's 1st to 99th percentile range crosses zero. | Linear models need regularization. Dividing coil currents by Ip removes the dominant shared factor. Coil currents reversing together with Ip is consistent with a sign rule tied to the Ip direction, to be confirmed by the coil-flux phase. |

## Labels and frame filters

| Id | Finding | Evidence | Consequence |
| --- | --- | --- | --- |
| F43 | Label extremes lie outside the 1st to 99th percentile band. | q95 2.91 to 11.76 (extremes 0.17 and 23.9); betaN 0.08 to 3.31 (-14.5 and 16.7); li 0.64 to 1.73 (0 and 20.8). | Fewer than 1% of frames on each tail. Feeds the label-quality step. |
| F44 | Some training frames have almost no plasma current. | `ip_over_peak` minimum 0.000 on training (1st percentile 0.363); minimum 0.113 on DIII-D test and 0.121 on MAST. | A minimum-Ip rule is needed for training frames, since the plasma flux is divided by Ip. The affected frames are end-of-shot ramp-downs: 13 below 0.1 and 1,303 below 0.2 (open item 9, resolved). |

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
6. The repeat-shot hypothesis (F09) and the share of changing-current frames on MAST (F13).
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
