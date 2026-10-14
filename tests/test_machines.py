"""Tests of the machine constants and their provenance records."""

from __future__ import annotations

from fusion_eq import scalars
from fusion_eq.machines import DIII_D, MAST, MACHINES


def test_constants_carry_values_sources_and_verified_status():
    assert (DIII_D.major_radius.value, DIII_D.minor_radius.value, DIII_D.tf_turns.value) == (1.67, 0.67, 144.0)
    assert (MAST.major_radius.value, MAST.minor_radius.value, MAST.tf_turns.value) == (0.85, 0.65, 24.0)
    assert DIII_D.tf_turns.verified and "F74" in DIII_D.tf_turns.source
    # Every constant is verified against a published source (13 October 2026).
    assert all(c.verified for m in MACHINES.values() for c in (m.major_radius, m.minor_radius, m.tf_turns))
    assert "Sykes" in MAST.major_radius.source and "Table 1" in MAST.minor_radius.source
    assert "Morris" in MAST.tf_turns.source and "National Academies" in DIII_D.major_radius.source
    record = MAST.constants_record()
    assert set(record) == {"major_radius", "minor_radius", "tf_turns"}
    assert record["tf_turns"]["value"] == 24.0 and record["tf_turns"]["verified"] is True


def test_reference_radius_is_the_major_radius_and_turns_reach_the_scalars_only_when_verified():
    assert DIII_D.reference_radius_m == 1.67 and MAST.reference_radius_m == 0.85
    assert scalars.TF_TURNS == {"DIII-D": 144, "MAST": 24}
