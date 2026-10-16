# MAST boundary diagnostic (v1)

Run of 2026-10-16T05:22:11+00:00, repository commit 64d128ebfbd029545619ea24f935ce16ac87af20. 40 shots, 302 frames per set at stride 8. Map sets: arm_g (`/Users/kamia/Desktop/PROJECTS/fusion-challenge/fusion-equilibrium-challenge/submissions/2026-10-15_arm_g_solhalf/mast_public_test.npz`), set_b (`/Users/kamia/Desktop/PROJECTS/fusion-challenge/fusion-equilibrium-challenge/submissions/2026-10-15_arm_setb_g_solhalf/mast_public_test.npz`), control (coil flux of sol_half plus Ip phi_mean, c = 0, no remainder). No MAST equilibrium label read.

## Method

- Boundary and axis: the organizers' `extract_lcfs_with_sign` and `derive_frame`, unmodified, machine MAST, axis sign +1, on the float16 maps read as float64 (as the scorer reads them).
- Wall: the 0.5 isoline of the scorer's MAST envelope mask (the boundary must stay inside it); centre column: its points at R <= 0.3 m. `machines.py` holds no wall or centre column.
- Saddles: zeros of the gradient of a bicubic spline of the map (Newton from minima of |grad psi|^2), classified by the Hessian; the scorer has no saddle finder. dpsi_sep uses the two saddles inside the envelope closest in flux to the boundary.
- Phase: `coil_scales.phase_of` on the frame table's ip_over_peak (flat-top at or above its level of peak Ip; rising and falling by position in the shot).
- Clip counts: each arm's model columns scaled by its stored DIII-D scaler, beyond its training clip bound.

## Flag definitions and frozen thresholds

Configuration `diag_mast_boundary_v1.yaml`, SHA-256 `7bfd7b0ba4c4843894ec65d3431bde5bc4302d2e818470c19fbfde907cca5c44`, written before any comparison with known shots; the overlap step reads flags only.

| threshold | value |
| --- | --- |
| axis_sign | 1.0 |
| robust_z_max | 3.5 |
| a_band_m | (0.35, 0.75) |
| area_band_m2 | (0.5, 4.2) |
| dpsi_sep_max | 0.02 |
| xpoint_flux_tol | 0.02 |
| xpoint_contour_max_m | 0.08 |
| wall_tol_m | 0.03 |
| centre_column_max_r_m | 0.3 |
| grid_edge_m | 0.04 |
| da_max_m | 0.05 |
| dr0_max_m | 0.05 |
| flag_fraction | 0.2 |
| coil_active_ka | 5.0 |

- F1 size: |robust z| of a or area above robust_z_max (over the set's frames), or a or area outside its band.
- F2 topology: |dpsi_sep| below dpsi_sep_max, or the defining saddle changes side between consecutive frames.
- F3 limiting: limited on the centre column, within grid_edge_m of the grid edge, or no boundary found.
- F4 arm disagreement: |a_g - a_b| above da_max_m or |R0_g - R0_b| above dr0_max_m, or only one found.
- F5 control agreement: F1, F2 or F3 in arm_g, in set_b and in the control.
- A shot is flagged for a flag when at least 0.2 of its frames carry it.

## Boundaries per set

| set | frames | found | type_diverted | type_limited | type_unresolved | type_none | limited_cc | fail_axis_on_rim | fail_open_at_axis | two_peaks | second_gap_median | a_median_m | r0_median_m | kappa_median |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| arm_g | 302 | 1.000 | 0.887 | 0.113 | 0.000 | 0.000 | 0.113 | 0.000 | 0.000 | 0.000 | n/a | 0.598 | 0.843 | 1.799 |
| set_b | 302 | 1.000 | 0.897 | 0.103 | 0.000 | 0.000 | 0.103 | 0.000 | 0.000 | 0.000 | n/a | 0.620 | 0.851 | 1.745 |
| control | 302 | 1.000 | 0.901 | 0.099 | 0.000 | 0.000 | 0.099 | 0.000 | 0.000 | 0.000 | n/a | 0.604 | 0.861 | 1.846 |

Frame share per flag:

| set | F1 | F2 | F3 | any_flag | F4 | F5 |
| --- | --- | --- | --- | --- | --- | --- |
| arm_g | 0.000 | 0.894 | 0.113 | 0.934 | n/a | n/a |
| set_b | 0.007 | 0.927 | 0.103 | 0.947 | n/a | n/a |
| control | 0.000 | 0.940 | 0.099 | 0.957 | n/a | n/a |
| arm pair | n/a | n/a | n/a | n/a | 0.023 | 0.921 |

Shots flagged per flag:

| flag | shots |
| --- | --- |
| arm_g_F1 | 0 |
| arm_g_F2 | 39 |
| arm_g_F3 | 7 |
| arm_g_any_flag | 40 |
| arm_g_fail | 0 |
| set_b_F1 | 1 |
| set_b_F2 | 40 |
| set_b_F3 | 5 |
| set_b_any_flag | 40 |
| set_b_fail | 0 |
| control_F1 | 0 |
| control_F2 | 40 |
| control_F3 | 4 |
| control_any_flag | 40 |
| control_fail | 0 |
| F4 | 0 |
| F5 | 40 |

## Flags by phase (share of frames)

| set | phase | frames | F1 | F2 | F3 | F4 | F5 | fail |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| arm_g | falling | 24 | 0.000 | 0.792 | 0.042 | 0.042 | 0.792 | 0.000 |
| arm_g | flat-top | 221 | 0.000 | 0.919 | 0.018 | 0.000 | 0.919 | 0.000 |
| arm_g | rising | 57 | 0.000 | 0.842 | 0.509 | 0.105 | 0.982 | 0.000 |
| set_b | falling | 24 | 0.083 | 0.958 | 0.042 | 0.042 | 0.792 | 0.000 |
| set_b | flat-top | 221 | 0.000 | 0.932 | 0.009 | 0.000 | 0.919 | 0.000 |
| set_b | rising | 57 | 0.000 | 0.895 | 0.491 | 0.105 | 0.982 | 0.000 |
| control | falling | 24 | 0.000 | 0.875 | 0.000 | 0.042 | 0.792 | 0.000 |
| control | flat-top | 221 | 0.000 | 0.950 | 0.014 | 0.000 | 0.919 | 0.000 |
| control | rising | 57 | 0.000 | 0.930 | 0.474 | 0.105 | 0.982 | 0.000 |

## Flagged against unflagged shots (medians of shot-level inputs)

| flag | n_flagged | ip_mean_ka flagged / not | ip_max_ka flagged / not | ip_rate_rel_max flagged / not | sol_swing_ka flagged / not | clip_rate_arm_g flagged / not | clip_rate_set_b flagged / not |
| --- | --- | --- | --- | --- | --- | --- | --- |
| arm_g_F1 | 0 | n/a / 627.298 | n/a / 712.803 | n/a / 9.397 | n/a / 47.918 | n/a / 0.000 | n/a / 0.000 |
| arm_g_F2 | 39 | 630.350 / 616.963 | 736.077 / 634.221 | 9.398 / 7.747 | 47.956 / 41.847 | 0.000 / 0.000 | 0.000 / 0.000 |
| arm_g_F3 | 7 | 610.465 / 662.276 | 650.076 / 745.208 | 7.747 / 10.153 | 41.847 / 47.956 | 0.000 / 0.083 | 0.000 / 0.083 |
| arm_g_any_flag | 40 | 627.298 / n/a | 712.803 / n/a | 9.397 / n/a | 47.918 / n/a | 0.000 / n/a | 0.000 / n/a |
| arm_g_fail | 0 | n/a / 627.298 | n/a / 712.803 | n/a / 9.397 | n/a / 47.918 | n/a / 0.000 | n/a / 0.000 |
| set_b_F1 | 1 | 566.585 / 630.350 | 750.735 / 689.529 | 6.818 / 9.398 | 57.245 / 47.880 | 0.222 / 0.000 | 0.222 / 0.000 |
| set_b_F2 | 40 | 627.298 / n/a | 712.803 / n/a | 9.397 / n/a | 47.918 / n/a | 0.000 / n/a | 0.000 / n/a |
| set_b_F3 | 5 | 610.465 / 662.276 | 650.076 / 745.208 | 7.747 / 9.677 | 41.847 / 47.956 | 0.000 / 0.000 | 0.000 / 0.000 |
| set_b_any_flag | 40 | 627.298 / n/a | 712.803 / n/a | 9.397 / n/a | 47.918 / n/a | 0.000 / n/a | 0.000 / n/a |
| set_b_fail | 0 | n/a / 627.298 | n/a / 712.803 | n/a / 9.397 | n/a / 47.918 | n/a / 0.000 | n/a / 0.000 |
| control_F1 | 0 | n/a / 627.298 | n/a / 712.803 | n/a / 9.397 | n/a / 47.918 | n/a / 0.000 | n/a / 0.000 |
| control_F2 | 40 | 627.298 / n/a | 712.803 / n/a | 9.397 / n/a | 47.918 / n/a | 0.000 / n/a | 0.000 / n/a |
| control_F3 | 4 | 613.714 / 646.313 | 647.689 / 740.642 | 7.602 / 9.915 | 47.573 / 47.918 | 0.100 / 0.000 | 0.100 / 0.000 |
| control_any_flag | 40 | 627.298 / n/a | 712.803 / n/a | 9.397 / n/a | 47.918 / n/a | 0.000 / n/a | 0.000 / n/a |
| control_fail | 0 | n/a / 627.298 | n/a / 712.803 | n/a / 9.397 | n/a / 47.918 | n/a / 0.000 | n/a / 0.000 |
| F4 | 0 | n/a / 627.298 | n/a / 712.803 | n/a / 9.397 | n/a / 47.918 | n/a / 0.000 | n/a / 0.000 |
| F5 | 40 | 627.298 / n/a | 712.803 / n/a | 9.397 / n/a | 47.918 / n/a | 0.000 / n/a | 0.000 / n/a |

## Figures

- 0035568738 (flagged): [0035568738_maps.png](figures/0035568738_maps.png), [0035568738_traces.png](figures/0035568738_traces.png)
- 0052369c79 (flagged): [0052369c79_maps.png](figures/0052369c79_maps.png), [0052369c79_traces.png](figures/0052369c79_traces.png)
- 009ba0ad8d (flagged): [009ba0ad8d_maps.png](figures/009ba0ad8d_maps.png), [009ba0ad8d_traces.png](figures/009ba0ad8d_traces.png)
- 00db024f42 (flagged): [00db024f42_maps.png](figures/00db024f42_maps.png), [00db024f42_traces.png](figures/00db024f42_traces.png)
- 018ab61eba (flagged): [018ab61eba_maps.png](figures/018ab61eba_maps.png), [018ab61eba_traces.png](figures/018ab61eba_traces.png)
- 01c7c14ce8 (flagged): [01c7c14ce8_maps.png](figures/01c7c14ce8_maps.png), [01c7c14ce8_traces.png](figures/01c7c14ce8_traces.png)
- 02a43a2db9 (flagged): [02a43a2db9_maps.png](figures/02a43a2db9_maps.png), [02a43a2db9_traces.png](figures/02a43a2db9_traces.png)
- 02d9558a2b (flagged): [02d9558a2b_maps.png](figures/02d9558a2b_maps.png), [02d9558a2b_traces.png](figures/02d9558a2b_traces.png)
- 02f0948202 (flagged): [02f0948202_maps.png](figures/02f0948202_maps.png), [02f0948202_traces.png](figures/02f0948202_traces.png)
- 033d230355 (flagged): [033d230355_maps.png](figures/033d230355_maps.png), [033d230355_traces.png](figures/033d230355_traces.png)

## Readings

Left for the review of the full run.
