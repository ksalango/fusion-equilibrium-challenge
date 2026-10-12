"""Tests of the scalar formulas, the boundary shapes from the scorer's functions, the pooled
R2 and the learned correction, on synthetic inputs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fusion_eq import scalars as sc
from tests.conftest import gaussian_psi, make_d3d_shot

GRID_R = np.linspace(0.84, 2.54, 65)
GRID_Z = np.linspace(-1.6, 1.6, 65)


def test_toroidal_field_is_linear_in_current_and_ignores_its_sign():
    bt = sc.toroidal_field(117.0, 1.67, 144)
    assert bt == pytest.approx(2e-7 * 144 * 117_000 / 1.67) and bt == pytest.approx(2.0176, abs=1e-3)
    assert sc.toroidal_field(-117.0, 1.67, 144) == bt and sc.toroidal_field(234.0, 1.67, 144) == pytest.approx(2 * bt)
    assert sc.toroidal_field(117.0, 3.34, 144) == pytest.approx(bt / 2)
    assert sc.TF_TURNS["DIII-D"] == 144 and "MAST" not in sc.TF_TURNS
    assert np.isnan(sc.toroidal_field(np.nan, 1.67, 144))


def test_q95_formula_closed_form_scalings_and_sign():
    # Circular, small aspect-ratio correction: 5 a^2 Bt / (R Ip) x (1 + 1) / 2 x (1.17 - 0.65 eps) / (1 - eps^2)^2.
    a, r, bt, ip = 0.6, 1.7, 2.0, 1.2
    eps = a / r
    expected = 5 * a ** 2 * bt / (r * ip) * (1.17 - 0.65 * eps) / (1 - eps ** 2) ** 2
    assert sc.q95_formula(a, r, 1.0, 0.0, bt, ip) == pytest.approx(expected)
    # Elongation 1.8 and triangularity 0.4 raise q95 by the shape factor.
    k, d = 1.8, 0.4
    shape = (1 + k ** 2 * (1 + 2 * d ** 2 - 1.2 * d ** 3)) / 2
    assert sc.q95_formula(a, r, k, d, bt, ip) == pytest.approx(expected * shape)
    # Inverse in Ip, linear in Bt, positive whatever the directions of Ip and Bt.
    assert sc.q95_formula(a, r, 1.0, 0.0, bt, 2 * ip) == pytest.approx(expected / 2)
    assert sc.q95_formula(a, r, 1.0, 0.0, 2 * bt, ip) == pytest.approx(2 * expected)
    assert sc.q95_formula(a, r, 1.0, 0.0, -bt, -ip) == pytest.approx(expected)
    out = sc.q95_formula(np.array([a, a]), np.array([r, r]), np.array([1.0, 1.0]), np.array([0.0, 0.0]), np.array([bt, bt]), np.array([ip, np.nan]))
    assert out.shape == (2,) and out[0] == pytest.approx(expected) and np.isnan(out[1])


def test_beta_n_formula_closed_form_and_linearity():
    p, a, bt, ip = 20_000.0, 0.6, 2.0, 1.2
    beta_percent = 100 * 2 * sc.MU0 * p / bt ** 2
    assert sc.beta_n_formula(p, a, bt, ip) == pytest.approx(beta_percent * a * bt / ip)
    assert sc.beta_n_formula(2 * p, a, bt, ip) == pytest.approx(2 * sc.beta_n_formula(p, a, bt, ip))
    assert sc.beta_n_formula(p, a, bt, -ip) == pytest.approx(sc.beta_n_formula(p, a, bt, ip))
    # 20 kPa at 2 T and 1.2 MA with a = 0.6 m: beta 1.26 percent, betaN 1.26.
    assert sc.beta_n_formula(p, a, bt, ip) == pytest.approx(1.2566, abs=1e-3)


def test_pooled_r2_matches_the_scorer_definition_and_best_multiplier_recovers_a_factor():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert sc.pooled_r2(y, y) == 1.0 and sc.pooled_r2(y, np.full(4, y.mean())) == 0.0
    pred = np.array([1.5, 2.0, 2.5, 4.5])
    assert sc.pooled_r2(y, pred) == pytest.approx(1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2))
    # Pooled over frames, not averaged per shot; a non-finite prediction counts as the pooled mean.
    assert sc.pooled_r2(y, np.array([1.0, np.nan, 3.0, 4.0])) == pytest.approx(1 - (2 - 2.5) ** 2 / 5.0)
    assert sc.pooled_r2(np.array([1.0, np.nan, 3.0]), np.array([1.0, 1.0, 3.0])) == 1.0
    assert sc.pooled_r2(y, pred) < sc.pooled_r2(y, y) and np.isnan(sc.pooled_r2([np.nan], [1.0]))
    f = np.array([2.0, 4.0, 6.0, np.nan])
    assert sc.best_multiplier(y, f) == pytest.approx(0.5)
    assert sc.best_multiplier(1.3 * f[:3] + np.array([0.01, -0.01, 0.0]), f[:3]) == pytest.approx(1.3, abs=1e-3)


def test_shape_from_map_recovers_the_synthetic_boundary(starter):
    shot = make_d3d_shot(n_frames=2)
    shape = sc.shape_from_map(shot["efit_psirz"][0], GRID_R, GRID_Z, "DIII-D")
    assert shape["boundary_found"] and set(shape) == {*sc.SHAPE_COLUMNS, "boundary_found"}
    # Gaussian centred at R 1.85, Z -0.30 with widths 0.35 and 0.6: an up-down symmetric ellipse.
    assert shape["R_geo"] == pytest.approx(1.85, abs=0.06) and 0 < shape["a"] < 0.8
    assert 1.3 < shape["kappa"] < 2.1 and abs(shape["tri_top"]) < 0.15 and abs(shape["tri_bot"]) < 0.15
    assert shape["eps"] == pytest.approx(shape["a"] / shape["R_geo"])
    # The scorer's own derivation gives the same kappa and triangularities.
    scorer = sc.starter_kit.scoring()
    env = sc.starter_kit.envelope_mask("DIII-D")
    derived = scorer.derive.derive_frame(shot["efit_psirz"][0], GRID_R, GRID_Z, "DIII-D", env.mask_coarse, env.mask_f, with_li=False)
    assert derived["kappa"] == shape["kappa"] and derived["tri_top"] == shape["tri_top"]
    # A flat map has no boundary: NaN everywhere, flagged.
    flat = sc.shape_from_map(np.zeros((65, 65)), GRID_R, GRID_Z, "DIII-D")
    assert not flat["boundary_found"] and all(np.isnan(flat[k]) for k in sc.SHAPE_COLUMNS)
    # A more elongated gaussian gives a larger kappa and the same R_geo.
    tall = gaussian_psi(GRID_R, GRID_Z, 1.85, -0.3, 0.3, 0.8, -0.5, 0.1)
    taller = sc.shape_from_map(tall, GRID_R, GRID_Z, "DIII-D")
    assert taller["kappa"] > shape["kappa"] and taller["R_geo"] == pytest.approx(shape["R_geo"], abs=0.06)


def test_shape_table_keeps_order_and_matches_single_frames(starter):
    maps = make_d3d_shot(n_frames=3)["efit_psirz"]
    items = [("s2", np.array([0, 2]), maps[[0, 2]]), ("s1", np.array([1]), maps[[1]])]
    table = sc.shape_table(items, GRID_R, GRID_Z, workers=1)
    assert table["shot"].tolist() == ["s2", "s2", "s1"] and table["frame"].tolist() == [0, 2, 1]
    one = sc.shape_from_map(maps[2], GRID_R, GRID_Z)
    row = table[(table["shot"] == "s2") & (table["frame"] == 2)].iloc[0]
    assert row["kappa"] == one["kappa"] and row["R_geo"] == one["R_geo"] and bool(row["boundary_found"])
    assert sc.shape_table([], GRID_R, GRID_Z, workers=1).empty


def test_cached_shape_table_is_read_back(starter, tmp_path):
    maps = make_d3d_shot(n_frames=2)["efit_psirz"]
    calls = []

    def items():
        calls.append(1)
        return [("s", np.array([0, 1]), maps)]

    path = tmp_path / "shapes.parquet"
    first = sc.cached_shape_table(path, items, GRID_R, GRID_Z, workers=1)
    second = sc.cached_shape_table(path, items, GRID_R, GRID_Z, workers=1)
    assert len(calls) == 1 and path.is_file() and first.equals(second) and len(first) == 2


def circular_map(r0, k, n, half_width=2.0):
    """Uniform current density: ``psi = R0 k r^2 / 2`` so that ``Bp = k r`` on circles about
    ``(R0, 0)``; the axis sits on a grid node for odd ``n``."""
    r = np.linspace(r0 - half_width, r0 + half_width, n)
    z = np.linspace(-half_width, half_width, n)
    rr, zz = np.meshgrid(r, z)
    return r0 * k * ((rr - r0) ** 2 + zz ** 2) / 2, r, z


def test_q95_from_map_matches_the_circular_closed_form(starter):
    k, bt, n_turns = 0.5, 2.0, 144.0
    # Large aspect ratio: q = r Bt / (R Bp) = Bt / (R0 k), the same on every surface.
    r0 = 1000.0
    tf_ka = r0 * bt / (2e-7 * n_turns) / 1000.0            # F = R0 Bt = 2e-7 n_turns |I_tf|
    psi, r, z = circular_map(r0, k, 129)
    q = sc.q95_from_map(psi, r, z, tf_ka, n_turns, use_envelope=False)
    assert q == pytest.approx(bt / (r0 * k), rel=1e-3)
    # Finite aspect ratio: the closed integral gives Bt / (k sqrt(R0^2 - r^2)) on the surface of radius r,
    # above the large-aspect-ratio value; the surface sits near sqrt(0.95) of the boundary radius.
    r0 = 10.0
    tf_ka = r0 * bt / (2e-7 * n_turns) / 1000.0
    psi, r, z = circular_map(r0, k, 129)
    q = sc.q95_from_map(psi, r, z, tf_ka, n_turns, use_envelope=False)
    r_surface = np.sqrt(0.95) * 2.0
    assert bt / (r0 * k) < q < 1.001 * bt / (k * np.sqrt(r0 ** 2 - r_surface ** 2))
    assert q == pytest.approx(bt / (k * np.sqrt(r0 ** 2 - r_surface ** 2)), rel=5e-3)
    # Linear in the field, sign of the current ignored, NaN without a field or a boundary.
    assert sc.q95_from_map(psi, r, z, 2 * tf_ka, n_turns, use_envelope=False) == pytest.approx(2 * q)
    assert sc.q95_from_map(psi, r, z, -tf_ka, n_turns, use_envelope=False) == pytest.approx(q)
    assert np.isnan(sc.q95_from_map(psi, r, z, np.nan, n_turns, use_envelope=False))
    assert np.isnan(sc.q95_from_map(np.zeros_like(psi), r, z, tf_ka, n_turns, use_envelope=False))
    assert np.isnan(sc.q95_from_map(np.where(psi > 1, np.nan, psi), r, z, tf_ka, n_turns, use_envelope=False))


def test_q95_from_map_is_sign_insensitive_and_converges_with_resolution(starter):
    k, bt, n_turns, r0 = 0.5, 2.0, 144.0, 1000.0
    tf_ka = r0 * bt / (2e-7 * n_turns) / 1000.0
    expected = bt / (r0 * k)
    errors = []
    for n in (33, 65, 129):
        psi, r, z = circular_map(r0, k, n)
        q = sc.q95_from_map(psi, r, z, tf_ka, n_turns, use_envelope=False)
        assert sc.q95_from_map(-psi, r, z, tf_ka, n_turns, use_envelope=False) == pytest.approx(q, rel=1e-9)
        errors.append(abs(q / expected - 1))
    assert errors[0] > errors[1] > errors[2] and errors[-1] < 2e-4
    # The surface chosen matters only through the aspect ratio here: 0.94 and 0.96 agree within 0.1 percent.
    psi, r, z = circular_map(r0, k, 65)
    q94 = sc.q95_from_map(psi, r, z, tf_ka, n_turns, psi_n=0.94, use_envelope=False)
    q96 = sc.q95_from_map(psi, r, z, tf_ka, n_turns, psi_n=0.96, use_envelope=False)
    assert q94 == pytest.approx(q96, rel=1e-3)


def test_q95_from_map_on_the_synthetic_shot_and_the_table(starter):
    shot = make_d3d_shot(n_frames=3)
    maps = shot["efit_psirz"]
    q = sc.q95_from_map(maps[0], GRID_R, GRID_Z, 117.0, 144)
    assert np.isfinite(q) and q > 0
    assert sc.q95_from_map(-maps[0], GRID_R, GRID_Z, 117.0, 144) == pytest.approx(q, rel=1e-9)
    items = [("s2", np.array([0, 2]), maps[[0, 2]], np.array([117.0, 117.0])), ("s1", np.array([1]), maps[[1]], np.array([-117.0]))]
    table = sc.q95_table(items, GRID_R, GRID_Z, 144, levels=(0.95, 0.94), workers=1)
    assert list(table.columns) == ["shot", "frame", "q95_calc_095", "q95_calc_094"]
    assert table["shot"].tolist() == ["s2", "s2", "s1"] and table["frame"].tolist() == [0, 2, 1]
    assert table["q95_calc_095"].iloc[0] == pytest.approx(q) and np.isfinite(table["q95_calc_094"]).all()
    assert sc.q95_table([], GRID_R, GRID_Z, 144, workers=1).empty


def frame_rows(n=6, shot="s"):
    rng = np.random.default_rng(1)
    return pd.DataFrame({
        "shot": shot, "frame": np.arange(n), "ip__mean": np.linspace(900.0, 1200.0, n), "tf__mean": np.full(n, -117.0),
        "ip_over_peak": np.linspace(0.6, 1.0, n), "ip_direction": 1,
        "ts_core_pe_peak__interp": 30.0 + rng.random(n), "ts_core_pe_mean__interp": np.full(n, 20.0),
        "efit_q95": np.full(n, 4.0), "efit_beta_n": np.full(n, 1.5),
    })


def shape_rows(n=6, shot="s"):
    return pd.DataFrame({"shot": shot, "frame": np.arange(n), "R_geo": 1.7, "a": 0.6, "kappa": 1.8, "tri_top": 0.5,
                         "tri_bot": 0.3, "eps": 0.6 / 1.7, "boundary_found": True})


def test_frame_quantities_joins_shapes_and_applies_the_formulas():
    frames, shapes = frame_rows(), shape_rows()
    shapes = shapes.iloc[:-1]                                    # the last frame has no shape
    q = sc.frame_quantities(frames, shapes, "DIII-D")
    assert len(q) == 6 and q["frame"].tolist() == list(range(6)) and np.isnan(q["q95_formula"].iloc[-1])
    row = q.iloc[0]
    assert row["ip_ma"] == pytest.approx(0.9) and row["delta"] == pytest.approx(0.4) and row["p_pa"] == pytest.approx(20_000.0)
    assert row["bt"] == pytest.approx(sc.toroidal_field(117.0, 1.7, 144))
    assert row["peak_to_mean"] == pytest.approx(frames["ts_core_pe_peak__interp"].iloc[0] / 20.0)
    assert row["q95_formula"] == pytest.approx(sc.q95_formula(0.6, 1.7, 1.8, 0.4, row["bt"], 0.9))
    assert row["betan_formula"] == pytest.approx(sc.beta_n_formula(20_000.0, 0.6, row["bt"], 0.9))
    other = sc.frame_quantities(frames, shape_rows(), "DIII-D", n_turns=72, pressure_column="ts_core_pe_peak__interp")
    assert other["bt"].iloc[0] == pytest.approx(row["bt"] / 2) and other["p_pa"].iloc[0] == pytest.approx(1000 * frames["ts_core_pe_peak__interp"].iloc[0])


def test_candidate_rules_count_each_filter():
    frames = pd.concat([frame_rows(4, "a"), frame_rows(3, "5df1630aa7")], ignore_index=True)
    frames.loc[0, "efit_q95"] = np.nan
    frames.loc[1, "tf__mean"] = 20.0
    frames.loc[2, "ip_over_peak"] = 0.05
    frames.loc[3, "efit_q95"] = 20.0
    frames.loc[4, "efit_beta_n"] = -3.0
    rules = sc.candidate_rules(frames)
    assert rules["invalid_label"].sum() == 1 and rules["tf_under_threshold"].sum() == 1
    assert rules["ip_under_fraction_of_peak"].sum() == 1 and rules["q95_outside_band"].sum() == 1
    assert rules["betan_outside_band"].sum() == 1 and rules["excluded_shots"].sum() == 3
    assert not rules["q95_outside_band"][0]                     # a NaN label is not "outside the band"


def test_correction_model_learns_a_log_linear_ratio_and_is_deterministic():
    rng = np.random.default_rng(0)
    n = 400
    table = pd.DataFrame({"kappa": rng.uniform(1.2, 2.0, n), "tri_top": rng.uniform(0, 0.6, n), "tri_bot": rng.uniform(0, 0.6, n),
                          "eps": rng.uniform(0.3, 0.4, n), "peak_to_mean": rng.uniform(1.0, 2.0, n), "ip_over_peak": rng.uniform(0.5, 1.0, n)})
    formula = rng.uniform(3.0, 6.0, n)
    ratio = np.exp(0.3 * table["kappa"].to_numpy() - 0.2 * table["eps"].to_numpy() + 0.1)
    label = formula * ratio
    ridge = sc.CorrectionModel(method="ridge", alpha=1e-6).fit(table, label, formula)
    assert ridge.n_train == n and np.allclose(ridge.factor(table), ratio, rtol=1e-3)
    assert sc.pooled_r2(label, ridge.predict(table, formula)) > 0.999 and sc.pooled_r2(label, formula) < 0.9
    # A non-linear ratio: the boosted model fits what the ridge cannot, and refits identically.
    bumpy = np.exp(0.5 * np.sin(6 * table["kappa"].to_numpy()))
    gbm = sc.CorrectionModel(method="gbm", max_iter=100).fit(table, formula * bumpy, formula)
    lin = sc.CorrectionModel(method="ridge").fit(table, formula * bumpy, formula)
    assert sc.pooled_r2(formula * bumpy, gbm.predict(table, formula)) > sc.pooled_r2(formula * bumpy, lin.predict(table, formula))
    again = sc.CorrectionModel(method="gbm", max_iter=100).fit(table, formula * bumpy, formula)
    assert np.array_equal(again.factor(table), gbm.factor(table))
    # Missing features are filled with the training median; non-positive rows are left out of the fit.
    holes = table.copy()
    holes.loc[:9, "peak_to_mean"] = np.nan
    label2 = label.copy()
    label2[:5] = np.nan
    label2[5:8] = -1.0
    m = sc.CorrectionModel(method="ridge").fit(holes, label2, formula)
    assert m.n_train == n - 8 and m.fill["peak_to_mean"] == pytest.approx(float(np.nanmedian(holes["peak_to_mean"][8:])))
    assert np.isfinite(m.factor(holes)).all()
    with pytest.raises(ValueError):
        sc.CorrectionModel(method="other").fit(table, label, formula)
    with pytest.raises(RuntimeError):
        sc.CorrectionModel().factor(table)
