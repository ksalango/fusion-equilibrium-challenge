"""Per-machine constants.

Everything that differs between DIII-D and MAST at the level of conventions, rather
than data values, is collected here so that the rest of the code can be written once
and parameterised by machine. Column names, units and missing-value conventions are
documented in ``docs/data_dictionary.md``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

# The plasma current column carries the same name on both machines.
IP_COLUMN = "magnetics_plasma_current"
MAGNETICS_TIME_COLUMN = "magnetics_time"
IP_TIME_COLUMN = "magnetics_plasma_current_times"


@dataclass(frozen=True)
class MachineConstant:
    """A physical constant of one machine with its provenance.

    Attributes:
        value: The value in ``units``.
        units: Physical units.
        source: Where the value comes from.
        verified: True once the value is confirmed against a published source or the data.
    """

    value: float
    units: str
    source: str
    verified: bool

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class CoilConvention:
    """How a current column's value is divided among its geometry elements.

    Attributes:
        turns: ``"shared"`` when the column holds the total current of all its elements
            (kA-turn, split equally over the elements), ``"per_element"`` when every
            element carries the column's value (kA per turn).
        scale: Multiplier applied to the column's current before the flux calculation.
            1.0 until a fit says otherwise; the DIII-D solenoid scale is the open
            quantity of the coil-flux phase.
    """

    turns: str
    scale: float = 1.0


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
        solenoid_column: Current column of the central solenoid, in kA.
        tf_column: Current column of the toroidal field coils, in kA. No poloidal
            flux contribution and no geometry rows.
        shaping_coil_columns: Current columns of the poloidal field (shaping) coils,
            in kA-turn, in the order used throughout the frame table.
        extra_current_columns: Current columns that are neither Ip, solenoid, toroidal
            field nor shaping coil (MAST ``magnetics_efps_current``), in kA.
        ip_has_own_time_axis: True when the plasma current sits on
            ``magnetics_plasma_current_times`` instead of ``magnetics_time``.
        lcfs_padding: Expected padding value of ``efit_lcfs_r/z`` beyond
            ``efit_lcfs_n``, ``"zero"`` or ``"nan"``.
        mask_file: File name of the scorer's plasma-envelope mask for this machine.
        solenoid_turns: Turn convention of the solenoid column, ``"shared"`` or
            ``"per_element"`` (see ``CoilConvention``). Shaping coils are always
            ``"shared"``: their columns hold total ampere-turns.
        major_radius: Reference major radius R0 in metres: the reference point (R0, 0) of
            the coil-flux checks and the origin of the normalized coordinates of the
            features, ``x = (R - R0) / a0``.
        minor_radius: Reference minor radius a0 in metres, the length scale of the
            normalized coordinates, ``y = Z / a0``.
        tf_turns: Turns of the toroidal-field coil set, giving ``Bt = mu0 N I_tf / (2 pi R)``.
    """

    name: str
    file_prefix: str
    axis_sign: float
    thomson_missing: str
    solenoid_column: str
    tf_column: str
    shaping_coil_columns: tuple[str, ...]
    extra_current_columns: tuple[str, ...]
    ip_has_own_time_axis: bool
    lcfs_padding: str
    mask_file: str
    solenoid_turns: str
    major_radius: MachineConstant
    minor_radius: MachineConstant
    tf_turns: MachineConstant

    @property
    def reference_radius_m(self) -> float:
        """Reference major radius R0 in metres (``major_radius``)."""
        return self.major_radius.value

    def constants_record(self) -> dict:
        """The machine constants with their sources and verified status, for build records."""
        return {"major_radius": self.major_radius.to_dict(), "minor_radius": self.minor_radius.to_dict(),
                "tf_turns": self.tf_turns.to_dict()}

    def coil_convention(self, current_column: str) -> CoilConvention:
        """Turn convention of a current column that has coil geometry.

        Raises:
            KeyError: for a column without geometry (plasma current, toroidal field,
                extra currents) or an unknown column.
        """
        if current_column in self.shaping_coil_columns:
            return CoilConvention(turns="shared")
        if current_column == self.solenoid_column:
            return CoilConvention(turns=self.solenoid_turns)
        raise KeyError(f"{current_column!r} has no coil geometry on {self.name}")

    @property
    def ip_column(self) -> str:
        """Plasma current column, in kA."""
        return IP_COLUMN

    @property
    def current_columns(self) -> tuple[str, ...]:
        """Every current column: Ip, solenoid, toroidal field, shaping coils, extras."""
        return (self.ip_column, self.solenoid_column, self.tf_column,
                *self.shaping_coil_columns, *self.extra_current_columns)

    def time_axis_column(self, current_column: str) -> str:
        """Time-axis column of a current column.

        DIII-D Ip sits on its own axis (``magnetics_plasma_current_times``); every
        other current on both machines sits on ``magnetics_time``.
        """
        if self.ip_has_own_time_axis and current_column == self.ip_column:
            return IP_TIME_COLUMN
        return MAGNETICS_TIME_COLUMN


DIII_D = MachineConfig(
    name="DIII-D",
    file_prefix="d3d_shot_",
    axis_sign=-1.0,
    thomson_missing="zero",
    solenoid_column="magnetics_ECOILA",
    tf_column="magnetics_bcoil",
    shaping_coil_columns=tuple(f"magnetics_F{i}{s}" for i in range(1, 10) for s in "AB"),
    extra_current_columns=(),
    ip_has_own_time_axis=True,
    lcfs_padding="zero",
    mask_file="d3d_envelope.npz",
    # One lumped rectangle; the README gives the column in kA and calls the turn count
    # ambiguous (48 single-turn elements in EFIT). Its scale is the quantity the coil-flux
    # phase fits, so the convention here only fixes what "scale 1" means.
    solenoid_turns="per_element",
    major_radius=MachineConstant(1.67, "m", "National Academies of Sciences, Engineering, and Medicine (2019), Final "
                                 "Report of the Committee on a Strategic Plan for U.S. Burning Plasma Research, "
                                 "Appendix G", True),
    minor_radius=MachineConstant(0.67, "m", "National Academies of Sciences, Engineering, and Medicine (2019), Final "
                                 "Report of the Committee on a Strategic Plan for U.S. Burning Plasma Research, "
                                 "Appendix G", True),
    tf_turns=MachineConstant(144.0, "turns", "Confirmed from the data: q95 calculated from the map against the "
                             "label, median ratio 0.997 (F74)", True),
)

MAST = MachineConfig(
    name="MAST",
    file_prefix="mast_shot_",
    axis_sign=1.0,
    thomson_missing="nan",
    solenoid_column="magnetics_sol_current",
    tf_column="magnetics_tf_current",
    shaping_coil_columns=tuple(f"magnetics_p{i}{s}_current" for i in range(2, 7) for s in "lu"),
    extra_current_columns=("magnetics_efps_current",),
    ip_has_own_time_axis=False,
    lcfs_padding="nan",
    mask_file="mast_envelope.npz",
    # 656 single-turn elements sharing one column in kA: every element carries the value.
    solenoid_turns="per_element",
    # Alternative published nominal set: R0 0.70 m, a0 0.50 m (Morris 1998), a sensitivity case
    # of the target phase (docs/TARGET.md, Y11) and of reports/features/v1/backup_radii_check.md.
    major_radius=MachineConstant(0.85, "m", "Sykes et al. (2001), Nucl. Fusion 41 1423, Table 1", True),
    minor_radius=MachineConstant(0.65, "m", "Sykes et al. (2001), Nucl. Fusion 41 1423, Table 1", True),
    tf_turns=MachineConstant(24.0, "turns", "Morris (1998), Europhysics News 29(6) 233: 24 centre-column conductors; "
                             "consistent with the 2.1 MA rod current of Sykes et al. (2001) and the -86 kA per "
                             "conductor of Berkery et al. (2021)", True),
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
