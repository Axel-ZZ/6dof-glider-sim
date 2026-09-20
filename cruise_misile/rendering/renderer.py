"""
ModernGL renderer — Blinn-Phong shading via pygame/moderngl.

World convention (matches mesh):
  +X  forward (nose direction)
  +Z  up
  +Y  left wing

Camera modes  (Tab to cycle, Space to pause/resume):
  FOLLOW   — chase-cam: sits behind the tail (+Z offset, -X offset), yaw-relative
  OVERVIEW — fit-to-view of the generated flight path
  ORBIT    — free mouse orbit; any drag switches here
"""
from __future__ import annotations
from enum import Enum, auto
import math

import numpy as np
import moderngl
import pygame
from scipy.spatial.transform import Rotation as R

from geometry.mesh import Mesh
from rendering.camera import OrbitCamera
from rendering.types import CameraMode
from rendering.shaders import (
    VERT_SHADER, FRAG_SHADER, 
    LINE_VERT_SHADER, LINE_FRAG_SHADER, 
    UI_VERT_SHADER, UI_FRAG_SHADER,
    GRID_VERT_SHADER, GRID_FRAG_SHADER
)
from rendering.ui import UIManager


# ── tuning knobs ─────────────────────────────────────────────────────────────
# FOLLOW offset in the aircraft's yaw-rotated frame.
# Aircraft nose = +X, so behind the tail = -X direction.
# We rotate this local offset by yaw so the camera always trails the nose.
FOLLOW_BACK     = 16.0    # metres behind the nose (along -X local)
FOLLOW_UP       =  4.0    # metres above (world +Z)
FOLLOW_LAG      =  0.06   # exponential lag factor; smaller = tighter

OVERVIEW_MARGIN =  1.4    # bbox scale factor — breathing room around path
OVERVIEW_LERP   =  0.03   # how fast the camera glides to overview position

# Ground plane visual
GROUND_Z        = 0.0     # Z height of the ground grid
GROUND_HALF     = 120.0   # half-extent of the grid in world units
GROUND_STEP     = 10.0    # grid line spacing
GROUND_COLOR    = (0.25, 0.28, 0.25)   # dark green-grey
# ─────────────────────────────────────────────────────────────────────────────


class Renderer:
    def __init__(self, width: int = 1280, height: int = 720,
                 title: str = "Aircraft Simulator"):
        pygame.init()
        pygame.display.set_caption(title)
        # Start in Fullscreen
        self.width, self.height = pygame.display.Info().current_w, pygame.display.Info().current_h
        pygame.display.set_mode(
            (self.width, self.height), pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE | pygame.FULLSCREEN)

        self.ctx = moderngl.create_context()
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.CULL_FACE)

        self.prog      = self.ctx.program(vertex_shader=VERT_SHADER,
                                          fragment_shader=FRAG_SHADER)
        self.line_prog = self.ctx.program(vertex_shader=LINE_VERT_SHADER,
                                          fragment_shader=LINE_FRAG_SHADER)
        self.grid_prog = self.ctx.program(vertex_shader=GRID_VERT_SHADER,
                                          fragment_shader=GRID_FRAG_SHADER)

        self.body_vao     = None
        self.elevator_vao = None
        self.hstab_x      = 0.0

        # OrbitCamera: eye = target - forward*distance
        # forward = (cos(el)*cos(az), cos(el)*sin(az), sin(el))
        # Nose = +X.  For eye behind tail (eye at -X of target):
        #   forward must point in +X  →  az=0°
        # For eye ABOVE: forward must tilt upward → but that pushes eye DOWN.
        #   So elevation must be NEGATIVE to get eye above the aircraft.
        self.camera = OrbitCamera(
            target   = (3.0, 0.0, 0.5),
            distance = 14.0,
            azimuth  =   0.0,    # eye at target.x - dist  (behind tail) ✓
            elevation= -15.0,    # negative → eye above aircraft ✓
        )

        # ── static geometry ──────────────────────────────────────────────────
        self.axes_vao   = self._build_axes()
        
        # Infinite Grid Quad (Full-screen for raycasting)
        grid_data = np.array([
            -1.0, -1.0, 0.0,
             1.0, -1.0, 0.0,
            -1.0,  1.0, 0.0,
             1.0,  1.0, 0.0,
        ], dtype='f4')
        self.grid_vbo = self.ctx.buffer(grid_data.tobytes())
        self.grid_vao = self.ctx.vertex_array(self.grid_prog, [(self.grid_vbo, '3f', 'in_position')])

        # ── flight history / trail ───────────────────────────────────────────
        self.history : list[np.ndarray] = []
        self._trail_vbo  = None
        self._trail_vao  = None
        self._init_trail_buffer(5000)   # enough for ~1.5 mins at 60fps

        # ── camera state ─────────────────────────────────────────────────────
        self.camera_mode        = CameraMode.FOLLOW
        self._cam_pos_smooth    = np.array([-16.0, 0.0, 24.0], dtype=np.float32)
        self._cam_target_smooth = np.array([  0.0, 0.0, 20.0], dtype=np.float32)
        self._overview_eye      = np.array([-30.0, 0.0, 20.0], dtype=np.float32)
        self._overview_target   = np.zeros(3, dtype=np.float32)

        # ── live aircraft state (physics based) ───────────────────────────────
        self._mouse_btn  = [False, False, False]
        self._last_mouse = (0, 0)
        self.aircraft_pos   = np.zeros(3, dtype=np.float32)
        self.aircraft_rpy   = np.zeros(3, dtype=np.float32)
        self.elevator_pitch = 0.0
        self._paused        = False

        # ── telemetry history (timed buffers) ────────────────────────────────
        self.telemetry_history = {
            "time": [],
            "va":   [],
            "alt":  [],
            "pos":  []  # (pn, pe)
        }
        self._telemetry_timer = 0.0

        # ── UI Overlay ───────────────────────────────────────────────────────
        self.ui = UIManager(self.ctx, self.width, self.height, on_action=self._handle_ui_action)

        # ── physics state ────────────────────────────────────────────────────
        self.state      = None  # (12,) state vector [u,v,w, p,q,r, phi,theta,psi, pn,pe,pd]
        self.u_ctrl     = np.array([0.0, 0.0, 0.0, 0.5], dtype=np.float32)

    # ── static geometry builders ─────────────────────────────────────────────

    def _build_axes(self):
        """XYZ axes at origin: X=red, Y=green, Z=blue. Length=2 units."""
        data = np.array([
            0,0,0, 1,0,0,  2,0,0, 1,0,0,   # X red
            0,0,0, 0,1,0,  0,2,0, 0,1,0,   # Y green
            0,0,0, 0,0,1,  0,0,2, 0,0,1,   # Z blue
        ], dtype=np.float32)
        vbo = self.ctx.buffer(data.tobytes())
        return self.ctx.vertex_array(
            self.line_prog, [(vbo, "3f 3f", "in_position", "in_color")])

    # ── mesh upload ───────────────────────────────────────────────────────────

    def _create_vao(self, mesh: Mesh):
        vbo = self.ctx.buffer(mesh.interleaved().tobytes())
        ibo = self.ctx.buffer(mesh.faces.flatten().tobytes())
        return self.ctx.vertex_array(
            self.prog, [(vbo, "3f 3f", "in_position", "in_normal")], ibo)

    def upload_mesh(self, meshes: dict[str, Mesh]) -> None:
        self.body_vao     = self._create_vao(meshes["body"])
        self.elevator_vao = self._create_vao(meshes["elevator"])
        # hstab pivot is near X=0 (tail); mean X of elevator verts confirms this
        self.hstab_x = float(np.mean(meshes["elevator"].vertices[:, 0]))

    def _init_trail_buffer(self, capacity: int) -> None:
        pos_vbo   = self.ctx.buffer(
            np.zeros(capacity * 3, dtype=np.float32).tobytes(), dynamic=True)
        color_vbo = self.ctx.buffer(
            np.tile([0.35, 0.75, 1.0], capacity).astype(np.float32).tobytes())
        self._trail_vbo = pos_vbo
        self._trail_vao = self.ctx.vertex_array(
            self.line_prog,
            [(pos_vbo,   "3f", "in_position"),
             (color_vbo, "3f", "in_color")])

    # ── main loop ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        clock = pygame.time.Clock()
        while True:
            dt = clock.tick(60) / 1000.0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); return
                self._handle_event(event)
            self._update(dt)
            self._draw()
            pygame.display.flip()

    # ── update ────────────────────────────────────────────────────────────────

    def _update(self, dt: float) -> None:
        if not self._paused:
            self._update_physics(dt)
        self._update_camera(dt)

    def _update_physics(self, dt: float) -> None:
        keys = pygame.key.get_pressed()

        if self.state is not None:
            # ── Capture controls ─────────────────────────────────────────────
            # W/S or U/J: Elevator
            if keys[pygame.K_w] or keys[pygame.K_u]: self.u_ctrl[0] += 1.0 * dt
            if keys[pygame.K_s] or keys[pygame.K_j]: self.u_ctrl[0] -= 1.0 * dt
            # A/D: Aileron
            if keys[pygame.K_a]: self.u_ctrl[1] -= 1.0 * dt
            if keys[pygame.K_d]: self.u_ctrl[1] += 1.0 * dt
            # Q/E: Rudder
            if keys[pygame.K_q]: self.u_ctrl[2] += 1.0 * dt
            if keys[pygame.K_e]: self.u_ctrl[2] -= 1.0 * dt
            # I/K: Throttle
            if keys[pygame.K_i]: self.u_ctrl[3] += 0.5 * dt
            if keys[pygame.K_k]: self.u_ctrl[3] -= 0.5 * dt

            self.u_ctrl[0:3] = np.clip(self.u_ctrl[0:3], -0.5, 0.5)
            self.u_ctrl[3]   = np.clip(self.u_ctrl[3], 0.0, 1.0)

            # Auto-center controls
            if not (keys[pygame.K_w] or keys[pygame.K_s] or 
                    keys[pygame.K_u] or keys[pygame.K_j]): self.u_ctrl[0] *= (1.0 - 2.5 * dt)
            if not (keys[pygame.K_a] or keys[pygame.K_d]): self.u_ctrl[1] *= (1.0 - 2.5 * dt)
            if not (keys[pygame.K_q] or keys[pygame.K_e]): self.u_ctrl[2] *= (1.0 - 2.5 * dt)

            # ── Step dynamics ────────────────────────────────────────────────
            from params.dynamics import update
            self.state = update(self.state, self.u_ctrl, dt)

            # Map state to renderer visuals
            self.aircraft_pos[0] = self.state[9]   # pn
            self.aircraft_pos[1] = self.state[10]  # pe
            self.aircraft_pos[2] = -self.state[11] # -pd (altitude)

            self.aircraft_rpy[0] = self.state[6]   # phi
            self.aircraft_rpy[1] = self.state[7]   # theta
            self.aircraft_rpy[2] = self.state[8]   # psi

            self.elevator_pitch = self.u_ctrl[0]
            
            # Update history
            self.history.append(self.aircraft_pos.copy())
            if len(self.history) > 5000:
                self.history.pop(0)

            # Update telemetry history (~10Hz for smooth graphs)
            self._telemetry_timer += dt
            if self._telemetry_timer > 0.1:
                self._telemetry_timer = 0.0
                self.telemetry_history["va"].append(float(np.linalg.norm(self.state[:3])))
                self.telemetry_history["alt"].append(float(-self.state[11]))
                self.telemetry_history["pos"].append((float(self.state[9]), float(self.state[10])))
                if len(self.telemetry_history["va"]) > 200:
                    for k in ["va", "alt", "pos"]: self.telemetry_history[k].pop(0)

        # ── Legacy Interactive Mode (no physics, only used if state is None) ──────
        if self.state is None:
            move_speed = 10.0
            rot_speed  = 1.0
            if keys[pygame.K_UP]:    self.aircraft_pos[0] += move_speed * dt
            if keys[pygame.K_DOWN]:  self.aircraft_pos[0] -= move_speed * dt
            if keys[pygame.K_LEFT]:  self.aircraft_pos[1] += move_speed * dt
            if keys[pygame.K_RIGHT]: self.aircraft_pos[1] -= move_speed * dt
            if keys[pygame.K_w]: self.aircraft_rpy[1] += rot_speed * dt
            if keys[pygame.K_s]: self.aircraft_rpy[1] -= rot_speed * dt
            if keys[pygame.K_a]: self.aircraft_rpy[0] -= rot_speed * dt
            if keys[pygame.K_d]: self.aircraft_rpy[0] += rot_speed * dt
            if keys[pygame.K_q]: self.aircraft_rpy[2] += rot_speed * dt
            if keys[pygame.K_e]: self.aircraft_rpy[2] -= rot_speed * dt

    def _update_camera(self, dt: float) -> None:
        if self.camera_mode == CameraMode.ORBIT:
            return

        if self.camera_mode == CameraMode.FOLLOW:
            desired_eye = self._follow_eye(self.aircraft_pos, self.aircraft_rpy[2])
            desired_tgt = self.aircraft_pos
            lag = FOLLOW_LAG
        else:  # OVERVIEW
            if not self.history:
                desired_eye, desired_tgt = self._cam_pos_smooth, self._cam_target_smooth
            else:
                pts = np.array(self.history)
                mn, mx = pts.min(axis=0), pts.max(axis=0)
                center = (mn + mx) / 2
                extent = np.linalg.norm(mx - mn)
                dist   = max(extent * OVERVIEW_MARGIN, 20.0)
                desired_tgt = center.astype(np.float32)
                desired_eye = (center + np.array([-dist*0.35, -dist*0.55, dist*0.50])).astype(np.float32)
            lag = OVERVIEW_LERP

        alpha = float(np.clip(1.0 - lag ** (max(dt, 1e-4) * 60), 0.0, 1.0))
        self._cam_pos_smooth    += alpha * (desired_eye - self._cam_pos_smooth)
        self._cam_target_smooth += alpha * (desired_tgt - self._cam_target_smooth)
        self.camera.set_eye_target(self._cam_pos_smooth, self._cam_target_smooth)

    def _follow_eye(self, pos: np.ndarray, yaw: float) -> np.ndarray:
        cy, sy = math.cos(yaw), math.sin(yaw)
        return pos + np.array([-cy * FOLLOW_BACK, -sy * FOLLOW_BACK, FOLLOW_UP], dtype=np.float32)

    # ── events ────────────────────────────────────────────────────────────────

    def _handle_event(self, event) -> None:
        if self.ui.handle_event(event):
            return

        if event.type == pygame.VIDEORESIZE:
            self.width, self.height = event.w, event.h

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button in (1, 2, 3):
                self._mouse_btn[event.button - 1] = True
                self.camera_mode = CameraMode.ORBIT
            elif event.button == 4: self.camera.zoom(1)
            elif event.button == 5: self.camera.zoom(-1)
            self._last_mouse = event.pos

        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button in (1, 2, 3):
                self._mouse_btn[event.button - 1] = False

        elif event.type == pygame.MOUSEMOTION:
            x, y   = event.pos
            dx, dy = x - self._last_mouse[0], y - self._last_mouse[1]
            if self._mouse_btn[0]:   self.camera.orbit(dx, dy)
            elif self._mouse_btn[2]: self.camera.pan(dx, -dy)
            self._last_mouse = event.pos

        elif event.type == pygame.KEYDOWN:
            if   event.key == pygame.K_ESCAPE: pygame.quit(); raise SystemExit
            elif event.key == pygame.K_TAB:
                modes = list(CameraMode)
                self.camera_mode = modes[(modes.index(self.camera_mode) + 1) % len(modes)]
    def _handle_ui_action(self, action: str) -> None:
        if action == "Exit":
            pygame.quit(); raise SystemExit
        elif action == "Pause/Resume":
            self._paused = not self._paused
            print(f"Simulation {'PAUSED' if self._paused else 'RESUMED'}")
        elif action == "Restart":
            self.restart_simulation()
        elif action == "Reset Camera":
            self.reset_camera()
        elif action == "Camera: Follow":
            self.camera_mode = CameraMode.FOLLOW
        elif action == "Camera: Overview":
            self.camera_mode = CameraMode.OVERVIEW
        elif action == "Camera: Orbit":
            self.camera_mode = CameraMode.ORBIT

    def restart_simulation(self) -> None:
        """Reset physics state and history."""
        from params.dynamics import init_state
        self.state = init_state()
        self.u_ctrl = np.array([0.0, 0.0, 0.0, 0.5], dtype=np.float32)
        self.history = []
        for k in self.telemetry_history: self.telemetry_history[k] = []
        print("Simulation RESTARTED")

    def reset_camera(self) -> None:
        """Revert camera to default FOLLOW position."""
        self.camera_mode = CameraMode.FOLLOW
        self.camera.azimuth = 0.0
        self.camera.elevation = -15.0
        self.camera.distance = 14.0
        # Instant snap or smooth? Smooth will happen in _update_camera.
        print("Camera RESET")
    # ── draw ──────────────────────────────────────────────────────────────────
    def _draw(self) -> None:
        if self.body_vao is None:
            return
        
        # Clear color and depth
        self.ctx.clear(0.12, 0.13, 0.15, 1.0, depth=1.0)

        # Total width depends on whether panels are shown
        view_w = self.width - (self.ui.panel_width if self.ui.show_panels else 0)
        aspect = view_w / max(self.height, 1)
        view   = self.camera.view_matrix()
        proj   = self.camera.projection_matrix(aspect)
        eye    = self.camera.eye()

        # Aircraft model matrix: RPY rotation then translation
        rot   = R.from_euler('xyz', self.aircraft_rpy).as_matrix()
        model = np.eye(4, dtype=np.float32)
        model[:3, :3] = rot
        model[:3,  3] = self.aircraft_pos
        n_mat = np.linalg.inv(model[:3, :3]).T.astype(np.float32)

        light_pos = np.array([20.0, -15.0, 30.0], dtype=np.float32)

        def upload(m, n, color):
            self.prog["u_model"].write(m.T.tobytes())
            self.prog["u_view"].write(view.T.tobytes())
            self.prog["u_proj"].write(proj.T.tobytes())
            self.prog["u_normal_mat"].write(n.T.tobytes())
            self.prog["u_light_pos"].write(light_pos.tobytes())
            self.prog["u_view_pos"].write(eye.tobytes())
            self.prog["u_color"].write(np.array(color, dtype=np.float32).tobytes())

        self.ctx.viewport = (0, 0, view_w, self.height)

        # Identity for static world objects
        identity = np.eye(4, dtype=np.float32)
        def upload_line(m):
            self.line_prog["u_model"].write(m.T.tobytes())
            self.line_prog["u_view"].write(view.T.tobytes())
            self.line_prog["u_proj"].write(proj.T.tobytes())

        # ── Render Infinite Grid ──
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self.grid_prog["u_view"].write(view.T.tobytes())
        self.grid_prog["u_proj"].write(proj.T.tobytes())
        self.grid_vao.render(moderngl.TRIANGLE_STRIP)
        self.ctx.disable(moderngl.BLEND)

        # ── Axes ──
        upload_line(identity)
        self.axes_vao.render(moderngl.LINES)

        # ── Aircraft ──
        # Body
        upload(model, n_mat, [0.72, 0.76, 0.82])
        self.body_vao.render(moderngl.TRIANGLES)

        # Elevator (pivot around hstab_x)
        t0 = np.eye(4, dtype=np.float32); t0[0, 3] = -self.hstab_x
        er = np.eye(4, dtype=np.float32)
        er[:3, :3] = R.from_euler('y', self.elevator_pitch).as_matrix()
        t1 = np.eye(4, dtype=np.float32); t1[0, 3] =  self.hstab_x
        em = model @ t1 @ er @ t0
        en = np.linalg.inv(em[:3, :3]).T.astype(np.float32)
        upload(em, en, [0.82, 0.46, 0.42])
        self.elevator_vao.render(moderngl.TRIANGLES)

        self._draw_trail(view, proj)
        self.ctx.viewport = (0, 0, self.width, self.height)
        self.ui.draw(self.telemetry_history, self.camera_mode)

    def _draw_trail(self, view, proj) -> None:
        if self._trail_vao is None or not self.history:
            return
        n   = len(self.history)
        pts = np.array(self.history, dtype=np.float32).flatten()
        self._trail_vbo.write(pts.tobytes()[:n*3*4]) 

        identity = np.eye(4, dtype=np.float32)
        self.line_prog["u_model"].write(identity.T.tobytes())
        self.line_prog["u_view"].write(view.T.tobytes())
        self.line_prog["u_proj"].write(proj.T.tobytes())

        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self._trail_vao.render(moderngl.LINE_STRIP, vertices=n)
        self.ctx.disable(moderngl.BLEND)

