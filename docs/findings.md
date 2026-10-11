# Data findings register

Findings from the data quality phase of the Fusion Equilibrium Challenge project, with the
evidence for each and its consequence for data preparation and modelling. Later phases add
their findings to this file.

- Period covered: 4 to 10 October 2026.
- Evidence: `reports/qa/qa_report.md` (full run over 9,124 shots, with the Ip alignment
  enabled), `reports/qa_before_ipfix/qa_report.md` (the same run before the alignment),
  `notebooks/01_qa.ipynb`; from 6 October, `reports/frame_table/frame_table_report.md`,
  `reports/frame_table/column_profile.parquet` (full build, commit 12da376) and
  `notebooks/02_frame_table.ipynb`; from 10 October, `reports/coil_flux/coil_flux_report.md`
  with its parquet tables (fit of 10 October 2026, 50 fit shots and 8 reference shots) and
  `notebooks/03_coil_flux.ipynb`; from 10 October, `reports/split/split_report.md` with
  `leak_test.json` and `near_repeats.json`, `splits/split_v1.json`, and
  `reports/baseline/compare_baselines.md` with `frame_residuals.parquet`,
  `score_parts.parquet` and the residual maps (run of 10 October 2026, commit 6b73139).
- Check ids in brackets refer to `docs/QA.md` (0.x), `docs/FRAME_TABLE.md` (Tx),
  `docs/COIL_FLUX.md` (Kx) and `docs/SPLIT.md` (Sx).

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

### Carried into modelling

- DIII-D coil model: shaping coils as stored plus the flat solenoid term (F47, F48).
- Sign handling: the stored map is `-sign(Ip)` times the physical flux; inputs and targets
  are put in one frame before any model sees them (F45). Current inputs (Ip, the solenoid
  and the shaping coils) are multiplied by sign(Ip); the toroidal field and Thomson are not
  (F60).
- PCA with 50 components for the plasma part (F63).
- Split `splits/split_v1.json`; every score reported for all validation shots and for the
  shots without a near-repeat in training (F57, F58).
- Up-down mirroring with recalculated coil flux, to give the model symmetric behaviour;
  DIII-D has about 2 percent double-null frames, 1.8 percent on the fit shots (F51, F52).
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
- First submission: the hybrid ridge on DIII-D, the coil flux alone on MAST.
- To check at the first MAST prediction: how flat MAST's calculated solenoid flux is across
  the plasma region; the stored-flux sign on MAST; a first MAST score with the coil flux
  alone (F53).

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
   MAST (F13) remains open.
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
