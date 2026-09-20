"""
export_czml.py
==============
Converts F5B OCP solution arrays into a CZML file for Cesium visualisation.

Coordinate conventions
-----------------------
  OCP: local NED frame, origin at (REF_LAT, REF_LON, REF_ALT_M).
       alt = -pd, positive upward.
  Cesium CZML position: WGS-84 cartographicDegrees [lon, lat, height].
  Cesium CZML orientation: quaternion that rotates body axes INTO ECEF
       (the "Earth-fixed" frame).  This quaternion is position-dependent.

Attitude — correct derivation
------------------------------
  R_ECEF_body = R_ECEF_ENU(lat, lon)  @  R_ENU_NED  @  R_NED_body(phi, theta, psi)

  where:
    R_ECEF_ENU(lat, lon)  — standard geodetic ENU-to-ECEF rotation at the
                            aircraft's current geodetic position.
    R_ENU_NED             — fixed frame swap: ENU-East=NED-East, ENU-North=NED-North,
                            ENU-Up=-NED-Down.
    R_NED_body(phi,theta,psi) — ZYX Euler rotation matrix (body axes in NED).

  This is the only correct approach; using a fixed ENU approximation or
  Cesium's HPR shortcut both produce wrong results at non-trivial headings
  because they ignore the position-dependent ECEF rotation.

Usage
-----
  from utils.export_czml import export_czml

  export_czml(
      pn=pn_sol, pe=pe_sol, alt=alt_sol,
      phi=X_sol[:, 6], theta=X_sol[:, 7], psi=X_sol[:, 8],
      t_mesh=t_mesh,
      output_path="trajectory.czml",
      label="F5B Trajectory",
      waypoints=[0.0, 150.0],
  )
"""

import json
import math
import numpy as np
from datetime import datetime, timedelta, timezone
# 47°21'52.5"N 8°32'33.2"E

# ─── Reference geodetic origin ────────────────────────────────────────────────
# REF_LAT_DEG = 47.3769   # Zürich area — change to your actual airfield
# REF_LON_DEG = 8.5417
# REF_LAT_DEG = 47.364583   # Lake Zurich, Zurich city
# REF_LON_DEG = 8.542556
# REF_ALT_M   = 440.0     # MSL height of the NED origin [m]

# Genèvesjön
# REF_LAT_DEG = 46.498649
# REF_LON_DEG = 6.629862
# REF_ALT_M   = 500.0     # MSL height of the NED origin [m]


REF_LAT_DEG = 57.635344   # Önnered Sweden
REF_LON_DEG = 11.870261
REF_ALT_M   = 500.0 

REF_LAT_DEG = 57.682300   # Inloppet goteborg
REF_LON_DEG = 11.845190
REF_ALT_M   = 500.0 

# REF_LAT_DEG = 45.976357   # Matterhorn
# REF_LON_DEG = 7.658490
# REF_ALT_M   = 4500.0     # MSL height of the NED origin [m]

M_PER_DEG_LAT = 111_320.0
M_PER_DEG_LON = 111_320.0 * math.cos(math.radians(REF_LAT_DEG))

# Default aircraft model — CDN hosted, no local server needed.
_DEFAULT_GLB = (
    "https://cdn.jsdelivr.net/npm/cesium@1.115.0"
    "/Apps/SampleData/models/CesiumAir/Cesium_Air.glb"
)

# ─── Fixed frame-swap matrix: ENU axes in NED coordinates ────────────────────
# ENU-East  = NED-East  (NED index 1) → row [0, 1, 0]
# ENU-North = NED-North (NED index 0) → row [1, 0, 0]
# ENU-Up    = -NED-Down (NED index 2, negated) → row [0, 0, -1]
_R_ENU_NED = np.array([
    [0,  1,  0],
    [1,  0,  0],
    [0,  0, -1],
], dtype=float)


def _R_ecef_enu(lat_deg: float, lon_deg: float) -> np.ndarray:
    """
    3×3 rotation matrix whose columns are the ENU unit vectors expressed in ECEF.
    Multiplying by this gives:  v_ECEF = R @ v_ENU.
    """
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    sl, cl = math.sin(lat), math.cos(lat)
    slo, clo = math.sin(lon), math.cos(lon)

    east  = np.array([-slo,      clo,    0.0])
    north = np.array([-sl*clo,  -sl*slo,  cl])
    up    = np.array([ cl*clo,   cl*slo,  sl])

    return np.column_stack([east, north, up])   # det = +1


def _R_ned_body(phi: float, theta: float, psi: float) -> np.ndarray:
    """
    ZYX Euler rotation: columns are body-frame axes expressed in NED.
    v_NED = R @ v_body.
    """
    cp, sp = math.cos(phi),   math.sin(phi)
    ct, st = math.cos(theta), math.sin(theta)
    cy, sy = math.cos(psi),   math.sin(psi)
    return np.array([
        [ ct*cy,  sp*st*cy - cp*sy,  cp*st*cy + sp*sy],
        [ ct*sy,  sp*st*sy + cp*cy,  cp*st*sy - sp*cy],
        [-st,     sp*ct,             cp*ct            ],
    ])


def _mat_to_quat_xyzw(M: np.ndarray) -> list[float]:
    """Stable Shepperd method: rotation matrix → unit quaternion [x, y, z, w]."""
    tr = M[0,0] + M[1,1] + M[2,2]
    if tr > 0:
        s = 0.5 / math.sqrt(tr + 1.0)
        w = 0.25 / s
        x = (M[2,1] - M[1,2]) * s
        y = (M[0,2] - M[2,0]) * s
        z = (M[1,0] - M[0,1]) * s
    elif M[0,0] > M[1,1] and M[0,0] > M[2,2]:
        s = 2.0 * math.sqrt(1.0 + M[0,0] - M[1,1] - M[2,2])
        w = (M[2,1] - M[1,2]) / s;  x = 0.25 * s
        y = (M[0,1] + M[1,0]) / s;  z = (M[0,2] + M[2,0]) / s
    elif M[1,1] > M[2,2]:
        s = 2.0 * math.sqrt(1.0 + M[1,1] - M[0,0] - M[2,2])
        w = (M[0,2] - M[2,0]) / s;  x = (M[0,1] + M[1,0]) / s
        y = 0.25 * s;                z = (M[1,2] + M[2,1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + M[2,2] - M[0,0] - M[1,1])
        w = (M[1,0] - M[0,1]) / s;  x = (M[0,2] + M[2,0]) / s
        y = (M[1,2] + M[2,1]) / s;  z = 0.25 * s
    norm = math.sqrt(x*x + y*y + z*z + w*w)
    return [x/norm, y/norm, z/norm, w/norm]


def _body_to_ecef_quat(lat_deg, lon_deg, phi, theta, psi) -> list[float]:
    """
    Full body→ECEF quaternion for CZML orientation.
    Computed at each timestep because R_ECEF_ENU depends on position.
    """
    R = _R_ecef_enu(lat_deg, lon_deg) @ _R_ENU_NED @ _R_ned_body(phi, theta, psi)
    return _mat_to_quat_xyzw(R)


def _ned_to_geodetic(pn_m, pe_m, alt_m):
    lat = REF_LAT_DEG + pn_m / M_PER_DEG_LAT
    lon = REF_LON_DEG + pe_m / M_PER_DEG_LON
    h   = REF_ALT_M  + alt_m
    return lat, lon, h


def _offset_iso(iso_start: str, seconds: float) -> str:
    dt  = datetime.fromisoformat(iso_start.replace("Z", "+00:00"))
    dt2 = dt + timedelta(seconds=seconds)
    return dt2.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def export_czml(
    pn,
    pe,
    alt,
    phi,
    theta,
    psi,
    t_mesh,
    output_path: str = "trajectory.czml",
    label: str = "F5B Trajectory",
    start_iso: str | None = None,
    model_uri: str | None = None,
    model_scale: float = 10.0,
    waypoints: list | None = None,
):
    """
    Write a CZML file from OCP solution arrays.

    Parameters
    ----------
    pn, pe, alt : array-like (N+1,)   NED position; alt = -pd, positive up [m].
    phi, theta, psi : array-like (N+1,)   ZYX Euler angles [rad].
    t_mesh : array-like (N+1,)   Time stamps [s] from t=0.
    output_path : str
    label : str   Display name in Cesium.
    start_iso : str   ISO-8601 UTC start time.
    model_uri : str or None
        Path/URL to a .glb glider model.  None → CDN Cesium_Air fallback.
    model_scale : float
        Uniform scale applied to the glTF model.  Tune per model.
        The viewer also has live scale buttons for quick adjustment.
    waypoints : list of float or None
        pn values [m] of waypoint planes to draw as orange wireframe walls.
    """
    pn    = np.asarray(pn,     dtype=float)
    pe    = np.asarray(pe,     dtype=float)
    alt   = np.asarray(alt,    dtype=float)
    phi   = np.asarray(phi,    dtype=float)
    theta = np.asarray(theta,  dtype=float)
    psi   = np.asarray(psi,    dtype=float)
    t     = np.asarray(t_mesh, dtype=float)

    N = len(t) - 1

    if start_iso is None:
        start_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT12:00:00Z")

    total_dur    = float(t[-1])
    availability = f"{start_iso}/{_offset_iso(start_iso, total_dur)}"

    # ── Position list: [t, lon, lat, h, ...]  ─────────────────────────────────
    position_list: list = []
    geodetic: list = []           # store (lat, lon, h) for orientation reuse
    for i in range(N + 1):
        lat, lon, h = _ned_to_geodetic(pn[i], pe[i], alt[i])
        geodetic.append((lat, lon, h))
        position_list += [round(float(t[i]), 4),
                          round(lon, 8), round(lat, 8), round(h, 3)]

    # ── Orientation list: [t, x, y, z, w, ...]  ───────────────────────────────
    # Full body→ECEF rotation, computed at each geodetic position.
    orientation_list: list = []
    for i in range(N + 1):
        lat, lon, _ = geodetic[i]
        q = _body_to_ecef_quat(lat, lon,
                               float(phi[i]), float(theta[i]), float(psi[i]))
        orientation_list += [round(float(t[i]), 4)] + [round(v, 8) for v in q]

    # ── Model URI  ─────────────────────────────────────────────────────────────
    effective_model_uri = model_uri if model_uri else _DEFAULT_GLB

    # ── Glider packet  ────────────────────────────────────────────────────────
    glider_packet = {
        "id": "glider",
        "name": label,
        "availability": availability,
        "position": {
            "epoch": start_iso,
            "cartographicDegrees": position_list,
            "interpolationAlgorithm": "LINEAR",
        },
        "orientation": {
            "epoch": start_iso,
            "unitQuaternion": orientation_list,
            "interpolationAlgorithm": "LINEAR",
        },
        "model": {
            "gltf": effective_model_uri,
            "scale": model_scale,
            "minimumPixelSize": 32,
            "maximumScale": 200_000,
            "runAnimations": False,
        },
    }

    # ── Static trajectory polyline  ───────────────────────────────────────────
    static_pos: list = []
    for lat, lon, h in geodetic:
        static_pos += [round(lon, 8), round(lat, 8), round(h, 3)]

    polyline_packet = {
        "id": "trajectory_line",
        "name": "Full trajectory",
        "polyline": {
            "positions": {"cartographicDegrees": static_pos},
            "width": 1.5,
            "material": {"solidColor": {"color": {"rgba": [0, 200, 255, 60]}}},
            "clampToGround": False,
        },
    }

    # ── Waypoint plane wireframes  ────────────────────────────────────────────
    extra_packets: list = []
    if waypoints:
        pe_min   = float(pe.min()) - 80
        pe_max   = float(pe.max()) + 80
        alt_base = float(alt.min()) - 10
        alt_top  = float(alt.max()) + 10

        for wp_i, wp_pn in enumerate(waypoints):
            for side_label, pe_val in [("L", pe_min), ("R", pe_max)]:
                lat_b, lon_s, h_b = _ned_to_geodetic(wp_pn, pe_val, alt_base)
                lat_t, _,     h_t = _ned_to_geodetic(wp_pn, pe_val, alt_top)
                extra_packets.append({
                    "id": f"waypoint_{wp_i}_{side_label}",
                    "name": f"Waypoint {wp_i + 1}",
                    "polyline": {
                        "positions": {"cartographicDegrees": [
                            round(lon_s, 8), round(lat_b, 8), round(h_b, 3),
                            round(lon_s, 8), round(lat_t, 8), round(h_t, 3),
                        ]},
                        "width": 3.0,
                        "material": {"solidColor": {"color": {"rgba": [255, 180, 0, 200]}}},
                    },
                })
            for a_val, edge_lbl in [(alt_top, "top"), (alt_base, "bot")]:
                lat_l, lon_l, h_e = _ned_to_geodetic(wp_pn, pe_min, a_val)
                lat_r, lon_r, _   = _ned_to_geodetic(wp_pn, pe_max, a_val)
                extra_packets.append({
                    "id": f"waypoint_{wp_i}_{edge_lbl}",
                    "polyline": {
                        "positions": {"cartographicDegrees": [
                            round(lon_l, 8), round(lat_l, 8), round(h_e, 3),
                            round(lon_r, 8), round(lat_r, 8), round(h_e, 3),
                        ]},
                        "width": 3.0,
                        "material": {"solidColor": {"color": {"rgba": [255, 180, 0, 200]}}},
                    },
                })

    # ── Assemble  ─────────────────────────────────────────────────────────────
    czml = [
        {
            "id": "document",
            "name": label,
            "version": "1.0",
            "clock": {
                "interval":    f"{start_iso}/{_offset_iso(start_iso, total_dur)}",
                "currentTime": start_iso,
                "multiplier":  1.0,
                "range":       "LOOP_STOP",
                "step":        "SYSTEM_CLOCK_MULTIPLIER",
            },
        },
        glider_packet,
        polyline_packet,
        *extra_packets,
    ]

    with open(output_path, "w") as f:
        json.dump(czml, f, indent=2)

    print(f"[export_czml] Written {len(czml)} packets → {output_path}")
    print(f"  Duration : {total_dur:.2f} s  |  Points : {N + 1}")
    print(f"  Origin   : {REF_LAT_DEG}°N  {REF_LON_DEG}°E  +{REF_ALT_M} m MSL")
    print(f"  Model    : {effective_model_uri[:70]}")