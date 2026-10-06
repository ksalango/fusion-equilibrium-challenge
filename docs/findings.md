# Data findings register

Findings from the data quality phase of the Fusion Equilibrium Challenge project, with the
evidence for each and its consequence for data preparation and modelling. Later phases add
their findings to this file.

- Period covered: 4 to 5 October 2026.
- Evidence: `reports/qa/qa_report.md` (full run over 9,124 shots, with the Ip alignment
  enabled), `reports/qa_before_ipfix/qa_report.md` (the same run before the alignment) and
  `notebooks/01_qa.ipynb`.
- Check ids in brackets refer to `docs/QA.md`.

## Summary

The dataset is complete and internally consistent. Test shots and training shots come from
the same population on every marker examined. One defect was found and corrected: on about
30% of DIII-D shots the plasma current (Ip) time axis was late by 50 to 330 ms after the
organizers' published fix. Several properties of the data constrain how inputs, targets and
missing values must be handled; these are listed below with the action each one requires.

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
3. 15 training shots have no computable Thomson coverage value.
4. The toroidal field current of the two shots with q95 below 1 is to be compared with the
   typical value.
5. Frame-level counts of failed and placeholder labels (F21).
6. The repeat-shot hypothesis (F09) and the share of changing-current frames on MAST (F13).
