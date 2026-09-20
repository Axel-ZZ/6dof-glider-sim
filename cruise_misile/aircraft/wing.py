from __future__ import annotations
import numpy as np
from geometry.mesh import Mesh
from geometry.primitives import wing_panel
from aircraft.parameters import AircraftParameters


def build_wings(params: AircraftParameters) -> Mesh:
    kwargs = dict(
        root_chord    = params.wing_root_chord,
        half_span     = params.wing_span / 2,
        thickness_frac = params.wing_thickness,
        sweep         = params.wing_sweep,
        dihedral      = params.wing_dihedral,
        x_offset      = params.wing_x,
    )
    rv, rf = wing_panel(**kwargs, flip_y=False)
    lv, lf = wing_panel(**kwargs, flip_y=True)
    lf = lf + len(rv)
    return Mesh(np.vstack([rv, lv]), np.vstack([rf, lf]))