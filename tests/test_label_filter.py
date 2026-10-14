"""Tests of the lighter label filter on synthetic frame rows."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fusion_eq import label_filter as lf


def frames() -> pd.DataFrame:
    return pd.DataFrame({
        "shot": ["a", "a", "a", "b", "b", "5df1630aa7", "c", "c", "c", "c"],
        "efit_q95": [4.0, np.nan, 3.5, 4.2, 0.0, 0.7, 5.0, 4.0, 4.0, 4.0],
        "efit_beta_n": [1.5, 1.2, -2.0, 1.1, 1.0, 15.0, 0.0, 1.0, 1.0, 1.0],
        "tf__mean": [-120.0, -120.0, -120.0, 120.0, 120.0, -20.0, -120.0, -30.0, -120.0, np.nan],
        "ip_over_peak": [0.9, 0.9, 0.9, 0.95, 0.95, 0.9, 0.9, 0.9, 0.05, 0.9],
    })


def test_each_rule_removes_its_frames():
    rules = lf.lighter_filter_rules(frames())
    assert list(rules.columns) == list(lf.RULES)
    # Non-finite q95, negative betaN, q95 at zero, betaN at zero (a placeholder).
    assert rules["impossible_label"].tolist() == [False, True, True, False, True, False, True, False, False, False]
    # The field is taken in magnitude; a missing field does not remove the frame.
    assert rules["tf_under_threshold"].tolist() == [False] * 5 + [True, False, True, False, False]
    assert rules["ip_under_fraction_of_peak"].tolist() == [False] * 8 + [True, False]
    assert rules["flagged_shot"].tolist() == [False] * 5 + [True] + [False] * 4
    keep = lf.lighter_label_filter(frames())
    assert keep.tolist() == [True, False, False, True, False, False, False, False, False, True]


def test_thresholds_are_parameters_and_columns_are_required():
    keep = lf.lighter_label_filter(frames(), tf_min_ka=10.0, ip_over_peak_min=0.01, flagged_shots=())
    assert keep.tolist() == [True, False, False, True, False, True, False, True, True, True]
    with pytest.raises(KeyError, match="ip_over_peak"):
        lf.lighter_label_filter(frames().drop(columns="ip_over_peak"))
