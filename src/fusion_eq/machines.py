"""Per-machine constants.

Everything that differs between DIII-D and MAST at the level of conventions, rather
than data values, is collected here so that the rest of the code can be written once
and parameterised by machine. Column names, units and missing-value conventions are
documented in ``docs/data_dictionary.md``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MachineConfig:
    """Conventions of one tokamak in the challenge dataset.

    Attributes:
        name: Value of the ``source`` column, ``"DIII-D"`` or ``"MAST"``.
        file_prefix: Prefix of the shot file names, for example ``"d3d_shot_"``.
        axis_sign: Sign ``s`` such that ``s * efit_psirz`` has its magnetic axis at a
            maximum. Matches the scorer's ``AXIS_SIGN``. The axis is the minimum of
            psi on DIII-D and the maximum on MAST.
        thomson_missing: Missing-value convention of the Thomson data, ``"zero"``
            (exact 0 marks a missing point) or ``"nan"``.
        solenoid_column: Current column of the central solenoid.
        ip_has_own_time_axis: True when the plasma current sits on
            ``magnetics_plasma_current_times`` instead of ``magnetics_time``.
        lcfs_padding: Expected padding value of ``efit_lcfs_r/z`` beyond
            ``efit_lcfs_n``, ``"zero"`` or ``"nan"``.
        mask_file: File name of the scorer's plasma-envelope mask for this machine.
    """

    name: str
    file_prefix: str
    axis_sign: float
    thomson_missing: str
    solenoid_column: str
    ip_has_own_time_axis: bool
    lcfs_padding: str
    mask_file: str


DIII_D = MachineConfig(
    name="DIII-D",
    file_prefix="d3d_shot_",
    axis_sign=-1.0,
    thomson_missing="zero",
    solenoid_column="magnetics_ECOILA",
    ip_has_own_time_axis=True,
    lcfs_padding="zero",
    mask_file="d3d_envelope.npz",
)

MAST = MachineConfig(
    name="MAST",
    file_prefix="mast_shot_",
    axis_sign=1.0,
    thomson_missing="nan",
    solenoid_column="magnetics_sol_current",
    ip_has_own_time_axis=False,
    lcfs_padding="nan",
    mask_file="mast_envelope.npz",
)

MACHINES: dict[str, MachineConfig] = {m.name: m for m in (DIII_D, MAST)}


def machine_config(name: str) -> MachineConfig:
    """Returns the configuration for a ``source`` value.

    Args:
        name: ``"DIII-D"`` or ``"MAST"``.

    Raises:
        KeyError: for any other machine name.
    """
    try:
        return MACHINES[name]
    except KeyError:
        raise KeyError(f"unknown machine {name!r}; known: {sorted(MACHINES)}") from None


def machine_from_filename(stem: str) -> MachineConfig:
    """Infers the machine from a shot file stem such as ``d3d_shot_203702``.

    Raises:
        ValueError: when the stem carries no known prefix.
    """
    for m in MACHINES.values():
        if stem.startswith(m.file_prefix):
            return m
    raise ValueError(f"file stem {stem!r} has no known machine prefix")
