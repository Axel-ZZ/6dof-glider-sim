from dataclasses import dataclass


@dataclass
class AircraftParameters:
    fuselage_length: float
    fuselage_radius: float

    wing_span: float
    wing_root_chord: float
    wing_x: float
    wing_thickness: float = 0.08
    wing_sweep: float     = 0.0
    wing_dihedral: float  = 0.05

    hstab_span: float       = 2.0
    hstab_root_chord: float = 0.6
    hstab_x: float          = 0.3

    vstab_height: float     = 0.8
    vstab_root_chord: float = 0.7
    vstab_x: float          = 0.3


default_params = AircraftParameters(
    fuselage_length = 5.3,
    fuselage_radius = 0.25,
    wing_span       = 6.0,
    wing_root_chord = 1.1,
    wing_x          = 2.5,
    wing_thickness  = 0.10,
    wing_sweep      = 0.15,
    wing_dihedral   = 0.06,
    hstab_span      = 2.2,
    hstab_root_chord = 0.55,
    hstab_x         = 0.35,
    vstab_height     = 0.75,
    vstab_root_chord = 0.65,
    vstab_x         = 0.35,
)