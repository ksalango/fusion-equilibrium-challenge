"""The lighter label filter: which labelled frames are fit for training (F77, "Carried into
modelling").

A frame is removed when its labels are impossible, when the toroidal field is under 50 kA,
when Ip is under 0.1 of the shot's peak (``ip_over_peak``), or when it belongs to one of the
two shots whose field signal is implausible for their plasma current (open item 8). The
percentile bands of F43 are not part of it. The rule masks are those of
``scalars.candidate_rules``; "impossible" extends that function's non-finite rule with values
no equilibrium can have: q95 at or below zero, and betaN at or below zero (negative values are
failed reconstructions and exact zeros placeholders, F21).

Targets are built for every labelled frame; the filter result is stored beside them as a flag
and used to choose fit samples and training frames.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from fusion_eq.scalars import candidate_rules

# The two DIII-D shots with q95 below 1 and a toroidal-field signal implausible for their current.
FLAGGED_SHOTS = ("5df1630aa7", "955c35c962")
TF_MIN_KA = 50.0
IP_OVER_PEAK_MIN = 0.1
# Frame-table columns the filter reads.
FILTER_COLUMNS = ("shot", "efit_q95", "efit_beta_n", "tf__mean", "ip_over_peak")
RULES = ("impossible_label", "tf_under_threshold", "ip_under_fraction_of_peak", "flagged_shot")


def lighter_filter_rules(frames: pd.DataFrame, tf_min_ka: float = TF_MIN_KA, ip_over_peak_min: float = IP_OVER_PEAK_MIN,
                         flagged_shots: Sequence[str] = FLAGGED_SHOTS) -> pd.DataFrame:
    """One bool column per rule, True where the rule removes the frame.

    Args:
        frames: Frame-table rows with the columns of ``FILTER_COLUMNS``.

    Raises:
        KeyError: when a column is missing.
    """
    missing = [c for c in FILTER_COLUMNS if c not in frames.columns]
    if missing:
        raise KeyError(f"the label filter needs the columns {missing}")
    rules = candidate_rules(frames, tf_min_ka=tf_min_ka, ip_over_peak_min=ip_over_peak_min, excluded_shots=flagged_shots)
    q95 = frames["efit_q95"].to_numpy(dtype=np.float64)
    betan = frames["efit_beta_n"].to_numpy(dtype=np.float64)
    with np.errstate(invalid="ignore"):
        impossible = rules["invalid_label"] | (q95 <= 0) | (betan <= 0)
    return pd.DataFrame({"impossible_label": impossible, "tf_under_threshold": rules["tf_under_threshold"],
                         "ip_under_fraction_of_peak": rules["ip_under_fraction_of_peak"],
                         "flagged_shot": rules["excluded_shots"]}, index=frames.index)


def lighter_label_filter(frames: pd.DataFrame, **kwargs) -> np.ndarray:
    """Bool per frame, True where the frame passes the lighter label filter."""
    return ~lighter_filter_rules(frames, **kwargs).any(axis=1).to_numpy()
