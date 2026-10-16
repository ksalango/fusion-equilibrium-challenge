"""Tests of the MAST boundary diagnostic on analytic maps built here (a plasma of filaments in a
vertical field, with off-envelope coils of the same sign for X-points): topology, limiting and
open-contour cases, the clean single null, the control map, the frozen thresholds, the label
guard and the script end to end on synthetic inputs."""

from __future__ import annotations

import dataclasses
import importlib.util
import inspect
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from fusion_eq import current_basis as cb, starter_kit, submission, targets
from fusion_eq.coil_flux import GreenTable, coil_model, filament_flux
from fusion_eq.dataset import REPO_ROOT
from fusion_eq.diagnostics import boundary as bd

pytestmark = pytest.mark.skipif(not starter_kit.is_available(), reason="starter kit not found")

CONFIG = REPO_ROOT / "configs" / "diag_mast_boundary_v1.yaml"
CONFIG_V2 = REPO_ROOT / "configs" / "diag_mast_boundary_v2.yaml"
# SHA-256 of version 1 as printed by the quick run of 15 October 2026: version 1 stays untouched.
V1_SHA256 = "7bfd7b0ba4c4843894ec65d3431bde5bc4302d2e818470c19fbfde907cca5c44"
R, Z = np.linspace(0.06, 2.0, 65), np.linspace(-2.0, 2.0, 65)
RR, ZZ = np.meshgrid(R, Z)
MAST_COLUMNS = (*[f"magnetics_p{i}{s}_current" for i in range(2, 7) for s in "lu"], "magnetics_sol_current")


def plasma(r0=0.95, a=0.4, kappa=1.8, ip_a=7e5) -> np.ndarray:
    """Flux of a uniform elliptic current of filaments on a fine lattice, Wb/rad."""
    pts = [(r0 + x, y) for x in np.linspace(-a, a, 17) for y in np.linspace(-kappa * a, kappa * a, 29)
           if (x / a) ** 2 + (y / (kappa * a)) ** 2 <= 1]
    return sum(filament_flux(RR, ZZ, r, z, ip_a / len(pts), 0.03) for r, z in pts)


def vertical_field(b) -> np.ndarray:
    return -b * RR ** 2 / 2


def coil(z, current_a) -> np.ndarray:
    return filament_flux(RR, ZZ, 0.6, z, current_a)


@pytest.fixture(scope="module")
def maps() -> dict[str, np.ndarray]:
    core = plasma() + vertical_field(0.1)
    return {"dn": core + coil(1.7, 8e5) + coil(-1.7, 8e5),
            "dn_low": core + coil(1.7, 8e5) + coil(-1.7, 8.08e5),
            "dn_up": core + coil(1.7, 8.08e5) + coil(-1.7, 8e5),
            "sn": core + coil(-1.7, 8e5),
            "limited": plasma(0.45, 0.25, 1.6, 5e5) + vertical_field(0.25),
            "open": 0.1 * RR + 0.05 * ZZ}


@pytest.fixture(scope="module")
def setup():
    thr = bd.load_thresholds(CONFIG)
    env = starter_kit.envelope_mask("MAST")
    return thr, env, bd.scorer_wall(env, thr.centre_column_max_r_m)


def records(names_maps, setup, shot="0000000001"):
    thr, env, wall = setup
    rows = []
    for k, m in enumerate(names_maps):
        rec, _ = bd.analyze_frame(m, R, Z, env, wall, thr)
        rows.append({"shot": shot, "frame": k, **rec})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------- topology
def test_critical_points_of_closed_forms():
    saddle = bd.critical_points((RR - 1.0) ** 2 - (ZZ - 0.2) ** 2, R, Z)
    assert [q.kind for q in saddle] == ["saddle"]
    assert saddle[0].r == pytest.approx(1.0, abs=1e-6) and saddle[0].z == pytest.approx(0.2, abs=1e-6)
    peak = bd.critical_points(-((RR - 0.9) ** 2 + 2 * (ZZ + 0.3) ** 2), R, Z)
    assert [q.kind for q in peak] == ["maximum"]
    assert (peak[0].r, peak[0].z) == pytest.approx((0.9, -0.3), abs=1e-6)


def test_symmetric_double_null_flips_with_a_small_perturbation_and_fires_f2(maps, setup):
    thr = setup[0]
    sym = records([maps["dn"]], setup).iloc[0]
    assert sym["boundary_type"] == "diverted" and abs(sym["dpsi_sep"]) < 1e-3
    frames = bd.set_flags(records([maps["dn_low"], maps["dn_up"]], setup), thr)
    assert list(frames["def_xpt_side"]) == ["lower", "upper"]
    assert list(frames["dominant"]) == ["lower", "upper"]
    assert (frames["dpsi_sep"].abs() < thr.dpsi_sep_max).all()
    assert list(frames["xpt_switch"]) == [False, True]
    assert frames["F2"].all() and not frames["F3"].any()
    # The two X-points are mirror images.
    assert frames["def_xpt_z"].iloc[0] == pytest.approx(-frames["def_xpt_z"].iloc[1], abs=1e-3)


def test_centre_column_limited_plasma_fires_f3(maps, setup):
    frames = bd.set_flags(records([maps["limited"]], setup), setup[0])
    row = frames.iloc[0]
    assert row["found"] and row["boundary_type"] == "limited" and row["limited_on"] == "centre_column"
    assert row["cc_distance"] <= setup[0].wall_tol_m
    assert row["F3"]


def test_open_contour_fires_f3(maps, setup):
    frames = bd.set_flags(records([maps["open"]], setup), setup[0])
    row = frames.iloc[0]
    assert not row["found"] and row["fail_reason"] in ("axis_on_rim", "open_at_axis")
    assert row["boundary_type"] == "none" and np.isnan(row["a"])
    assert row["F3"] and not row["F1"] and not row["F2"]


def test_clean_single_null_raises_no_flag(setup):
    thr = setup[0]
    core = [plasma(ip_a=ip) + vertical_field(0.1) + coil(-1.7, 8e5) for ip in (6.8e5, 6.9e5, 7.0e5, 7.1e5, 7.2e5)]
    frames = bd.set_flags(records(core, setup), thr)
    assert (frames["boundary_type"] == "diverted").all() and (frames["def_xpt_side"] == "lower").all()
    assert (frames["dpsi_sep"].isna() | (frames["dpsi_sep"].abs() > thr.dpsi_sep_max)).all()
    assert not frames[["F1", "F2", "F3"]].any().any()
    lo, hi = thr.a_band_m
    assert ((frames["a"] > lo) & (frames["a"] < hi)).all()


def test_size_flag_on_an_outlier(setup):
    thr = setup[0]
    df = pd.DataFrame({"shot": "s", "frame": range(8), "found": True, "a": [0.5] * 7 + [0.72],
                       "area": [1.5, 1.52, 1.49, 1.51, 1.5, 1.48, 1.53, 1.5], "dpsi_sep": np.nan, "def_xpt_side": "",
                       "boundary_type": "limited", "limited_on": "", "touches_edge": False})
    df.loc[[1, 3, 5], "a"] = [0.51, 0.49, 0.505]
    flags = bd.set_flags(df, thr)
    assert list(flags["F1"]) == [False] * 7 + [True]


def test_geometry_of_a_circle():
    t = np.linspace(0, 2 * np.pi, 4000, endpoint=False)
    g = bd.contour_geometry(np.column_stack([1.0 + 0.5 * np.cos(t), 0.2 + 0.5 * np.sin(t)]))
    assert g["a"] == pytest.approx(0.5, rel=1e-6) and g["r0"] == pytest.approx(1.0, rel=1e-6)
    assert g["area"] == pytest.approx(np.pi * 0.25, rel=1e-5) and g["length"] == pytest.approx(np.pi, rel=1e-5)
    assert bd.robust_z([1.0, 1.0, 1.0])[0] == 0.0 and np.isnan(bd.robust_z([np.nan, 1.0])[0])


# ------------------------------------------------------------------------------ arms, shots
def test_arm_flags_and_overlap(setup):
    thr = setup[0]
    base = pd.DataFrame({"shot": ["a", "a", "b"], "frame": [0, 1, 0], "found": [True, True, True], "a": [0.5, 0.5, 0.5],
                         "r0": [0.9, 0.9, 0.9], "any_flag": [True, False, True]})
    other = base.assign(a=[0.5, 0.58, 0.5], found=[True, True, False])
    control = base.assign(any_flag=[True, True, False])
    pair = bd.arm_flags(base, other, control, thr)
    assert list(pair["F4"]) == [False, True, True]
    assert list(pair["F5"]) == [True, False, False]
    shots = bd.shot_flag_table(pair, ["F4", "F5"], thr)
    assert list(shots["shot_F4"]) == [True, True] and list(shots["frac_F5"]) == [0.5, 0.0]
    ov = bd.overlap(shots, ["a", "zz"], ["shot_F4", "shot_F5"]).set_index("flag")
    assert ov.loc["shot_F4", "known_caught"] == 1 and ov.loc["shot_F4", "other_raised"] == 1
    assert ov.loc["shot_F5", "known_caught_share"] == 1.0 and ov.loc["shot_F5", "known_not_in_table"] == 1


def test_thresholds_load_from_the_config_and_overlap_cannot_alter_them(setup):
    import yaml

    thr = bd.load_thresholds(CONFIG)
    raw = yaml.safe_load(CONFIG.read_text())
    assert thr.dpsi_sep_max == raw["topology"]["dpsi_sep_max"] and thr.a_band_m == tuple(raw["size"]["a_band_m"])
    before = (thr.record(), CONFIG.read_bytes())
    with pytest.raises(dataclasses.FrozenInstanceError):
        thr.dpsi_sep_max = 1.0                                       # type: ignore[misc]
    assert "thr" not in inspect.signature(bd.overlap).parameters
    shots = pd.DataFrame({"shot": ["a", "b"], "shot_F1": [True, False]})
    bd.overlap(shots, ["a"], ["shot_F1"])
    assert (thr.record(), CONFIG.read_bytes()) == before
    assert bd.load_thresholds(CONFIG).sha256 == thr.sha256


def test_unknown_threshold_entries_are_refused(tmp_path):
    text = CONFIG.read_text().replace("dpsi_sep_max: 0.02", "dpsi_sep_max: 0.02\n  tuned_on_known: 0.5")
    path = tmp_path / "bad.yaml"
    path.write_text(text)
    with pytest.raises(KeyError):
        bd.load_thresholds(path)


def test_phases_follow_the_frame_table_rule():
    df = pd.DataFrame({"ip_over_peak": [0.3, 0.95, 0.4], "frame": [0, 5, 9], "n_frames": [10, 10, 10]})
    assert list(bd.phases(df)) == ["rising", "flat-top", "falling"]


# ------------------------------------------------------------------------------- guards
def test_label_paths_are_refused(tmp_path):
    with pytest.raises(ValueError, match="sample"):
        bd.refuse_label_paths([starter_kit.starter_dir() / "parquet_data" / "anything.parquet"])
    with pytest.raises(ValueError, match="raw shot"):
        bd.refuse_label_paths([tmp_path / "mast_shot_0001.parquet"])
    labelled = tmp_path / "table.parquet"
    pq.write_table(pa.table({"shot": ["1"], "efit_psirz": [1.0]}), labelled)
    with pytest.raises(ValueError, match="label"):
        bd.refuse_label_paths([labelled])
    np.savez(tmp_path / "other.npz", efit_lcfs_r=np.zeros(3))
    with pytest.raises(ValueError, match="not a submission"):
        bd.refuse_label_paths([tmp_path / "other.npz"])
    clean = tmp_path / "frames.parquet"
    pq.write_table(pa.table({"shot": ["1"], "efit_times": [1.0], "ip__mean": [1.0]}), clean)
    np.savez(tmp_path / "sub.npz", shot_0000_psirz=np.zeros((1, 2, 2)), shot_0000_q95=np.zeros(1))
    bd.refuse_label_paths([clean, tmp_path / "sub.npz"])


# ------------------------------------------------------------------------------ control
def synthetic_green(scale=1e-5) -> GreenTable:
    patterns = np.stack([scale * np.exp(-((RR - 1.0) ** 2 + (ZZ - 0.1 * k) ** 2)) for k in range(len(MAST_COLUMNS))])
    return GreenTable(machine="MAST", columns=MAST_COLUMNS, patterns=patterns, grid_r=R, grid_z=Z, n_filaments=1,
                      geometry_hash="x", grid_hash="y", conventions={c: "shared" for c in MAST_COLUMNS},
                      elements_per_column={c: 1 for c in MAST_COLUMNS})


def frame_rows(n, ip=650.0) -> pd.DataFrame:
    table = synthetic_green()
    rng = np.random.default_rng(0)
    rows = pd.DataFrame({c: rng.normal(0, 50, n) for c in targets.current_columns_of(table, "MAST")})
    rows["ip__mean"] = ip + rng.normal(0, 5, n)
    return rows


def test_control_equals_compose_map_with_zero_coefficients():
    table = synthetic_green()
    scales = coil_model("MAST", table.columns, "sol_half")
    rng = np.random.default_rng(1)
    fb = cb.MachineFluxBasis("MAST", rng.normal(size=R.size * Z.size) * 1e-7, rng.normal(size=(5, R.size * Z.size)) * 1e-7,
                             (Z.size, R.size), 0.85, 0.65)
    objects = SimpleNamespace(flux_bases={"MAST": fb}, remainder_model=None)
    rows = frame_rows(4)
    got = bd.control_maps(rows, table, scales, objects)
    coil = targets.coil_flux_of_frames(table, rows, scales, "MAST")
    expected = cb.compose_map("MAST", coil, rows["ip__mean"].to_numpy(), np.zeros((4, 5)), fb)
    np.testing.assert_array_equal(got, expected)
    # sol_half: the solenoid enters at half its value.
    rows2 = rows.copy()
    rows2["sol__mean"] += 10.0
    delta = bd.control_maps(rows2, table, scales, objects) - got
    np.testing.assert_allclose(delta, np.broadcast_to(5.0 * table.patterns[-1], delta.shape), atol=1e-15)


# ------------------------------------------------------------------------ script end to end
def load_script():
    spec = importlib.util.spec_from_file_location("diag_mast_boundary", REPO_ROOT / "scripts" / "diag_mast_boundary.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_script_end_to_end_on_synthetic_inputs(tmp_path, monkeypatch, maps):
    script = load_script()
    shots = ["0000000011", "0000000022"]
    n = 4
    ip = 700.0
    sn = maps["sn"]
    shot_maps = {"0000000011": [maps["dn_low"], maps["dn_up"], maps["dn_low"], maps["dn_up"]], "0000000022": [sn] * n}
    for arm, folder in (("arm_g", "sub_g"), ("set_b", "sub_b")):
        arrays = {}
        for i, s in enumerate(shots):
            arrays[submission.shot_key(i, "psirz")] = np.stack(shot_maps[s]).astype(np.float16)
            arrays[submission.shot_key(i, "q95")] = np.full(n, 5.0)
            arrays[submission.shot_key(i, "betaN")] = np.full(n, 2.0)
        submission.write_npz(tmp_path / folder / "mast_public_test.npz", arrays)
        record = {"folds": {"MAST": {"shot_checks": [{"index": i, "shot": s} for i, s in enumerate(shots)]}},
                  "predictor_identifiers": {"coil_models": {"MAST": "sol_half"}}}
        (tmp_path / "records" / folder).mkdir(parents=True)
        (tmp_path / "records" / folder / "run.json").write_text(json.dumps(record))
    table = synthetic_green(scale=0.0)
    table.save(tmp_path / "green_mast.npz")
    rows = pd.DataFrame({"shot": np.repeat(shots, n), "frame": np.tile(np.arange(n), 2).astype(np.int32),
                         "time_ms": np.tile(100.0 + 10 * np.arange(n), 2), "n_frames": np.int32(n),
                         "ip_over_peak": np.tile([0.5, 0.95, 0.97, 0.6], 2).astype(np.float32), "ip__mean": ip})
    for c in targets.current_columns_of(table, "MAST"):
        rows[c] = 0.0
    rows["tf__mean"] = -84.0
    (tmp_path / "table").mkdir()
    pq.write_table(pa.Table.from_pandas(rows, preserve_index=False), tmp_path / "table" / "mast_public_test.parquet")
    (tmp_path / "features").mkdir()
    feats = rows[["shot", "frame"]].assign(ip_rate_rel=0.1)
    pq.write_table(pa.Table.from_pandas(feats, preserve_index=False), tmp_path / "features" / "mast_public_test.parquet")
    # The control: zero coil flux plus Ip phi_mean, with phi_mean the single-null plasma per ampere.
    fb = cb.MachineFluxBasis("MAST", sn.ravel() / (ip * 1e3), np.zeros((1, sn.size)), sn.shape, 0.85, 0.65)
    monkeypatch.setattr(script.tb.TargetObjects, "load", classmethod(lambda cls, d: SimpleNamespace(
        flux_bases={"MAST": fb}, remainder_model=None)))
    dummy = SimpleNamespace(transform=lambda df, machine, cols: df[list(cols)].to_numpy(dtype=float) * 100.0)
    monkeypatch.setattr(script, "clip_setup", lambda record, root: {"columns": ["ip_rate_rel"], "scaler": dummy,
                                                                     "bound": 5.0, "run": "r"})
    known = tmp_path / "known.csv"
    known.write_text("key,d_lcfs\nshot_0001,0.40\nshot_0000,0.70\nshot_0999,0.5\n")
    out = tmp_path / "out"
    code = script.main(["--arm-g", str(tmp_path / "sub_g"), "--set-b", str(tmp_path / "sub_b"),
                        "--records", str(tmp_path / "records"), "--features", str(tmp_path / "features"),
                        "--table-dir", str(tmp_path / "table"), "--green", str(tmp_path / "green_mast.npz"),
                        "--targets", str(tmp_path), "--out", str(out), "--workers", "1", "--known-shots", str(known)])
    assert code == 0
    frames = pd.read_parquet(out / "per_frame.parquet")
    assert len(frames) == 3 * 2 * n and set(frames["set"]) == set(bd.SET_NAMES)
    dn = frames[(frames["shot"] == shots[0]) & (frames["set"] == "arm_g")].sort_values("frame")
    # Version 2 (default): F2 is the change of side only; version 1 adds the near-double-null bound.
    assert dn["xpt_switch"].tolist() == [False, True, True, True]
    assert dn["F2"].tolist() == [False, True, True, True] and dn["F2_v1"].all()
    assert not frames[frames["shot"] == shots[1]][["F1", "F2", "F3"]].any().any()
    # The control is the single null on every frame, so F5 never fires; F4 never fires (equal arms).
    assert not frames["F5"].any() and not frames["F4"].any()
    assert (frames.loc[frames["set"] == "arm_g", "clip_arm_g"] == 1).all()
    per_shot = pd.read_csv(out / "per_shot.csv", dtype={"shot": str})
    assert per_shot.set_index("shot").loc[shots[0], "shot_arm_g_F2"]
    assert per_shot.set_index("shot").loc[shots[0], "shot_arm_g_F2_v1"]
    assert per_shot.set_index("shot").loc[shots[0], "known_key"] == "shot_0000"
    report = (out / "report.md").read_text()
    assert bd.load_thresholds(CONFIG_V2).sha256 in report and V1_SHA256 in report
    assert "### Flags v2" in report and "### Flags v1" in report and "## dpsi_sep by set and phase" in report
    assert "3 entries, 2 matched to a shot of the submission, 2 analysed" in report and "shot_0999" in report
    assert "(reading)" in report
    # Known shots in order of d_lcfs: shot_0001 (0.40, cluster 0.3-0.45 m) first, shot_0000 (0.70) second.
    known_dir = out / "figures" / "known"
    tables = sorted(p.name for p in known_dir.glob("*_frames.md"))
    assert tables == [f"k01_shot_0001_{shots[1]}_frames.md", f"k02_shot_0000_{shots[0]}_frames.md"]
    text = (known_dir / tables[1]).read_text()
    for col in ("phase", "arm_g type", "set_b a", "control R0", "arm_g kappa", "set_b dpsi_sep", "control limited"):
        assert col in text
    assert "cluster 0.6-0.86 m" in text and len(text.strip().splitlines()) == 2 + 2 + n
    assert sorted(p.name for p in known_dir.glob("k0*_maps.png")) == [f"k01_shot_0001_{shots[1]}_maps.png",
                                                                       f"k02_shot_0000_{shots[0]}_maps.png"]
    assert len(list(known_dir.glob("k0*_traces.png"))) == 2
    assert list((out / "figures").glob("*_maps.png"))


def test_script_refuses_a_labelled_input(tmp_path):
    script = load_script()
    labelled = tmp_path / "features"
    labelled.mkdir()
    pq.write_table(pa.table({"shot": ["1"], "frame": [0], "efit_q95": [3.0]}), labelled / "mast_public_test.parquet")
    missing = tmp_path / "absent"
    with pytest.raises(ValueError, match="label"):
        script.main(["--arm-g", str(missing / "g"), "--set-b", str(missing / "b"), "--table-dir", str(missing),
                     "--features", str(labelled), "--workers", "1", "--no-figures"])


def test_second_peak_is_reported_and_float16_ties_are_not(maps, setup):
    thr, env, wall = setup
    tied, _ = bd.analyze_frame(maps["sn"].astype(np.float16).astype(np.float64), R, Z, env, wall, thr)
    assert tied["n_maxima"] == 1 and np.isnan(tied["second_max_gap"])
    # Two current filaments 1 m apart, 3 percent apart in current: the scorer takes the stronger
    # as the axis and the other is reported as a separate peak with a small flux gap.
    two = filament_flux(RR, ZZ, 1.0, 0.5, 3.0e5, 0.05) + filament_flux(RR, ZZ, 1.0, -0.5, 2.9e5, 0.05)
    rec, _ = bd.analyze_frame(two, R, Z, env, wall, thr)
    assert rec["n_maxima"] == 2
    assert rec["Z_axis"] == pytest.approx(0.5, abs=0.07) and rec["second_max_z"] == pytest.approx(-0.5, abs=0.07)
    assert 0 < rec["second_max_gap"] < 0.2


def test_script_stops_when_no_known_shot_matches(tmp_path, monkeypatch):
    script = load_script()
    for folder in ("sub_g", "sub_b"):
        submission.write_npz(tmp_path / folder / "mast_public_test.npz", {"shot_0000_psirz": np.zeros((1, 2, 2), np.float16)})
        (tmp_path / "records" / folder).mkdir(parents=True)
        (tmp_path / "records" / folder / "run.json").write_text(json.dumps(
            {"folds": {"MAST": {"shot_checks": [{"index": 0, "shot": "0000000011"}]}},
             "predictor_identifiers": {"coil_models": {"MAST": "sol_half"}}}))
    missing = tmp_path / "absent"
    with pytest.raises(ValueError, match="none of the 2 known shots"):
        script.main(["--arm-g", str(tmp_path / "sub_g"), "--set-b", str(tmp_path / "sub_b"), "--records",
                     str(tmp_path / "records"), "--table-dir", str(missing), "--features", str(missing),
                     "--known-shots", "shot_0005,0035568738", "--workers", "1", "--no-figures"])


# --------------------------------------------------------------------------- version 2
def test_version_1_is_untouched_and_version_2_differs_only_in_f2():
    import hashlib

    assert hashlib.sha256(CONFIG.read_bytes()).hexdigest() == V1_SHA256
    v1, v2 = bd.load_thresholds(CONFIG), bd.load_thresholds(CONFIG_V2)
    assert (v1.version, v2.version) == (1, 2) and v1.dpsi_sep_max == 0.02 and v2.dpsi_sep_max is None
    assert "before any overlap" in v2.note and "quick run" in v2.note
    differ = {k for k, v in v1.record().items() if v2.record()[k] != v} - {"version", "note", "source", "sha256"}
    assert differ == {"dpsi_sep_max"}
    assert v1.analysis_record() == v2.analysis_record()


def test_f2_of_version_2_fires_on_a_side_change_only(maps, setup):
    v1, v2 = bd.load_thresholds(CONFIG), bd.load_thresholds(CONFIG_V2)
    steady = records([maps["dn_low"]] * 3, setup)
    assert bd.set_flags(steady, v1)["F2"].all()
    assert not bd.set_flags(steady, v2)["F2"].any()
    flip = records([maps["dn_low"], maps["dn_low"], maps["dn_up"]], setup)
    assert bd.set_flags(flip, v2)["F2"].tolist() == [False, False, True]


def test_known_keys_map_through_the_writing_order():
    order = ["0035568738", "0035568800", "0035569001"]
    entries = pd.DataFrame({"key": ["shot_0002", "shot_0000", "35568800", "shot_0007", "nonsense"],
                            "d_lcfs": [0.65, 0.35, np.nan, 0.5, 0.2]})
    m = bd.map_known(entries, order)
    assert m["shot"].tolist() == ["0035569001", "0035568738", "0035568800", "", ""]
    assert m["matched"].tolist() == [True, True, True, False, False]
    assert m["group"].tolist() == ["cluster 0.6-0.86 m", "cluster 0.3-0.45 m", "no d_lcfs", "other", "other"]
    # The writer's key of position i is shot_key(i, field) without the field.
    assert submission.shot_key(2, "psirz").rsplit("_", 1)[0] == "shot_0002"
    with pytest.raises(ValueError, match="none of the 2 known shots"):
        bd.map_known(pd.DataFrame({"key": ["shot_0009", "0000000001"]}), order)


def test_dpsi_sep_distribution():
    df = pd.DataFrame({"set": ["a"] * 5 + ["b"] * 2, "phase": ["rising"] * 7,
                       "dpsi_sep": [-0.01, 0.02, 0.03, np.nan, 0.05, np.nan, np.nan]})
    t = bd.dpsi_sep_distribution(df).set_index("set")
    assert t.loc["a", "frames"] == 5 and t.loc["a", "with_value"] == pytest.approx(0.8)
    assert t.loc["a", "p50"] == pytest.approx(0.025) and np.isnan(t.loc["b", "p50"])
