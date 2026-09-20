from __future__ import annotations
import numpy as np
from geometry.mesh import Mesh
from geometry.primitives import tube, hemisphere
from aircraft.parameters import AircraftParameters


def build_fuselage(params: AircraftParameters, segments: int = 24) -> Mesh:
    L = params.fuselage_length
    r = params.fuselage_radius
    nose_len = r * 3.0
    tail_len = r * 1.2

    y_tail      = 0.0
    y_body_start = tail_len
    y_body_end  = L - nose_len
    y_nose_ring = L - nose_len * 0.15

    all_verts, all_faces = [], []
    offset = 0

    def add_tube(r0, r1, y0, y1):
        nonlocal offset
        v, f = tube(r0, r1, y0, y1, segments, offset)
        all_verts.append(v)
        all_faces.append(f)
        offset += len(v)

    add_tube(r * 0.3, r,      y_tail,       y_body_start)
    add_tube(r,       r,      y_body_start, y_body_end)
    add_tube(r,       r * 0.6, y_body_end,  y_nose_ring)

    tip_verts, tip_faces = hemisphere(
        radius=r * 0.6, y_base=y_nose_ring,
        segments=segments, rings=8, offset=offset,
    )
    all_verts.append(tip_verts)
    all_faces.append(tip_faces)

    return Mesh(
        np.vstack(all_verts).astype(np.float32),
        np.vstack(all_faces).astype(np.uint32),
    )