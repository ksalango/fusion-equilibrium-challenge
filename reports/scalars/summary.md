# Scalar formulas, working lists of `split_v1.json`, stride 4

300 training and 100 validation shots; kept frames 16423 and 5551; filter: none; N_TURNS 144; pressure `ts_core_pe_mean__interp`.

## betaN, stage 1

| Method | R2 all | R2 negative Ip | frames | missing pred | calc failed |
| --- | --- | --- | --- | --- | --- |
| ridge baseline (inputs) | 0.9396 | 0.8909 | 5551 | 0 |  |
| formula | -0.8403 | -0.7819 | 5551 | 0 |  |
| formula x constant | 0.7895 | 0.6097 | 5551 | 0 |  |
| formula x ridge correction | 0.8131 | 0.6469 | 5551 | 0 |  |
| formula x gbm correction | 0.8513 | 0.7973 | 5551 | 0 |  |
| ridge on inputs, refit on these frames | 0.9409 | 0.8930 | 5551 | 0 |  |
| ridge on inputs, betaN x Ip then divided | 0.9241 | 0.8436 | 5551 | 0 |  |

## betaN, stage 2

| Method | R2 all | R2 negative Ip | frames | missing pred | calc failed |
| --- | --- | --- | --- | --- | --- |
| ridge baseline (inputs) | 0.9396 | 0.8909 | 5551 | 0 |  |
| formula | -0.8383 | -0.7799 | 5551 | 0 |  |
| formula x constant | 0.7998 | 0.6020 | 5551 | 0 |  |
| formula x ridge correction | 0.8498 | 0.6779 | 5551 | 0 |  |
| formula x gbm correction | 0.8985 | 0.8285 | 5551 | 0 |  |
| formula x ridge correction (stage-1 fit) | 0.8244 | 0.6663 | 5551 | 0 |  |
| formula x gbm correction (stage-1 fit) | 0.8396 | 0.7347 | 5551 | 0 |  |
| ridge on inputs, refit on these frames | 0.9409 | 0.8930 | 5551 | 0 |  |
| ridge on inputs, betaN x Ip then divided | 0.9241 | 0.8436 | 5551 | 0 |  |

## q95, stage 1

| Method | R2 all | R2 negative Ip | frames | missing pred | calc failed |
| --- | --- | --- | --- | --- | --- |
| ridge baseline (inputs) | 0.7355 | 0.7435 | 5551 | 0 |  |
| formula | 0.3073 | -0.0860 | 5551 | 0 |  |
| formula x constant | 0.9478 | 0.9860 | 5551 | 0 |  |
| formula x ridge correction | 0.9672 | 0.9700 | 5551 | 0 |  |
| formula x gbm correction | 0.9778 | 0.9831 | 5551 | 0 |  |
| calculated from map (psi_N 0.95) | 0.9685 | 0.9942 | 5551 | 0 | 0 |
| calculated from map, failures filled with formula x constant | 0.9685 | 0.9942 | 5551 | 0 | 0 |
| calculated from map (psi_N 0.94) | 0.9660 | 0.9819 | 5551 | 0 | 0 |
| calculated from map (psi_N 0.96) | 0.9293 | 0.9611 | 5551 | 0 | 0 |

## q95, stage 2

| Method | R2 all | R2 negative Ip | frames | missing pred | calc failed |
| --- | --- | --- | --- | --- | --- |
| ridge baseline (inputs) | 0.7355 | 0.7435 | 5551 | 0 |  |
| formula | 0.3255 | 0.0703 | 5551 | 0 |  |
| formula x constant | 0.9371 | 0.9750 | 5551 | 0 |  |
| formula x ridge correction | 0.9501 | 0.9641 | 5551 | 0 |  |
| formula x gbm correction | 0.9686 | 0.9806 | 5551 | 0 |  |
| calculated from map (psi_N 0.95) | 0.9388 | 0.9722 | 5551 | 0 | 0 |
| calculated from map, failures filled with formula x constant | 0.9388 | 0.9722 | 5551 | 0 | 0 |
| formula x ridge correction (stage-1 fit) | 0.9506 | 0.9621 | 5551 | 0 |  |
| formula x gbm correction (stage-1 fit) | 0.9613 | 0.9693 | 5551 | 0 |  |
