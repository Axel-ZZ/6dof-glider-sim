from __future__ import annotations
import numpy as np
from geometry.mesh import Mesh
from geometry.primitives import wing_panel
from aircraft.parameters import AircraftParameters


def build_tail(params: AircraftParameters) -> tuple[Mesh, Mesh]:
    hkwargs = dict(
        root_chord    = params.hstab_root_chord,
        half_span     = params.hstab_span / 2,
        thickness_frac = 0.09,
        sweep         = params.hstab_root_chord * 0.5,
        dihedral      = 0.02,
        x_offset      = params.hstab_x,
    )
    hrv, hrf = wing_panel(**hkwargs, flip_y=False)
    hlv, hlf = wing_panel(**hkwargs, flip_y=True)
    hlf = hlf + len(hrv)
    h_verts = np.vstack([hrv, hlv])
    h_faces = np.vstack([hrf, hlf])

    vv, vf = wing_panel(
        root_chord    = params.vstab_root_chord,
        half_span     = params.vstab_height,
        thickness_frac = 0.10,
        sweep         = params.vstab_root_chord * 0.5,
        dihedral      = 0.0,
        x_offset      = params.vstab_x,
        flip_y        = False,
    )
    # Rotate the panel so span points up
    vv = vv[:, [0, 2, 1]].copy()
    vv[:, 1] = 0.0                      

    return Mesh(h_verts, h_faces), Mesh(vv, vf)