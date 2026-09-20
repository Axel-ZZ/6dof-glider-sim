"""
Orbit camera.

Controls
--------
  Left-drag   : rotate (azimuth / elevation)
  Scroll      : zoom
  Middle-drag : pan
"""
from __future__ import annotations
import math
import numpy as np


class OrbitCamera:
    def __init__(
        self,
        target:    tuple[float, float, float] = (3.0, 0.0, 0.0),
        distance:  float = 12.0,
        azimuth:   float = 30.0,   # degrees
        elevation: float = 15.0,   # degrees
        fov:       float = 45.0,
        near:      float = 0.1,
        far:       float = 500.0,
    ):
        self.target    = np.array(target, dtype=np.float32)
        self.distance  = distance
        self.azimuth   = azimuth
        self.elevation = elevation
        self.fov       = fov
        self.near      = near
        self.far       = far

    # ── user controls ────────────────────────────────────────────────────────

    def orbit(self, dx: float, dy: float, sensitivity: float = 0.4) -> None:
        self.azimuth   -= dx * sensitivity
        self.elevation  = float(np.clip(self.elevation + dy * sensitivity, -89.0, 89.0))

    def zoom(self, delta: float, factor: float = 1.1) -> None:
        self.distance = float(np.clip(
            self.distance / factor if delta > 0 else self.distance * factor,
            0.5, 300.0,
        ))

    def pan(self, dx: float, dy: float, sensitivity: float = 0.005) -> None:
        right, up, _ = self._basis()
        self.target -= right * dx * self.distance * sensitivity
        self.target += up    * dy * self.distance * sensitivity

    def set_eye_target(self, eye: np.ndarray, target: np.ndarray) -> None:
        """
        Drive the camera to an arbitrary world-space (eye, target) pair.
        Converts to spherical coords so orbit/zoom/pan keep working afterward.
        """
        self.target   = np.asarray(target, dtype=np.float32)
        offset        = np.asarray(eye, dtype=np.float32) - self.target
        self.distance = float(np.linalg.norm(offset))
        if self.distance < 1e-6:
            return
        # _basis defines forward = direction camera looks = (target - eye) / distance = -offset/distance
        # eye = target - forward * distance
        # So we must store azimuth/elevation of the FORWARD vector, not the offset.
        fwd = -offset / self.distance
        self.elevation = float(math.degrees(math.asin(float(np.clip(fwd[2], -1.0, 1.0)))))
        self.azimuth   = float(math.degrees(math.atan2(float(fwd[1]), float(fwd[0]))))

    # ── internals ────────────────────────────────────────────────────────────

    def _basis(self):
        az  = math.radians(self.azimuth)
        el  = math.radians(self.elevation)
        forward = np.array([
            math.cos(el) * math.cos(az),
            math.cos(el) * math.sin(az),
            math.sin(el),
        ], dtype=np.float32)
        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        right    = np.cross(forward, world_up)
        right   /= np.linalg.norm(right)
        up       = np.cross(right, forward)
        return right, up, forward

    def eye(self) -> np.ndarray:
        _, _, fwd = self._basis()
        return self.target - fwd * self.distance

    def view_matrix(self) -> np.ndarray:
        right, up, forward = self._basis()
        eye = self.target - forward * self.distance
        Rot = np.eye(4, dtype=np.float32)
        Rot[0, :3] =  right
        Rot[1, :3] =  up
        Rot[2, :3] = -forward
        T = np.eye(4, dtype=np.float32)
        T[:3, 3] = -eye
        return Rot @ T

    def projection_matrix(self, aspect: float) -> np.ndarray:
        f  = 1.0 / math.tan(math.radians(self.fov) / 2)
        nf = 1.0 / (self.near - self.far)
        return np.array([
            [f / aspect, 0,  0,                                0],
            [0,          f,  0,                                0],
            [0,          0,  (self.far + self.near) * nf,      2 * self.far * self.near * nf],
            [0,          0, -1,                                0],
        ], dtype=np.float32)