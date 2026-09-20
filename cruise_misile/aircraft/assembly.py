from __future__ import annotations
import numpy as np
from geometry.mesh import Mesh
from aircraft.parameters import AircraftParameters, default_params
from aircraft.fuselage import build_fuselage
from aircraft.wing import build_wings
from aircraft.tail import build_tail


def build_aircraft(params: AircraftParameters | None = None) -> dict[str, Mesh]:
    if params is None:
        params = default_params
    
    fuselage = build_fuselage(params)
    wings = build_wings(params)
    hstab, vstab = build_tail(params)

    offset = 0
    body_verts, body_faces = [], []
    for m in [fuselage, wings, vstab]:
        body_verts.append(m.vertices)
        body_faces.append(m.faces + offset)
        offset += len(m.vertices)

    body = Mesh(
        np.vstack(body_verts).astype(np.float32),
        np.vstack(body_faces).astype(np.uint32),
    )

    return {
        "body": body,
        "elevator": hstab,
    }