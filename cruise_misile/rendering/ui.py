import pygame
import numpy as np
import moderngl
from rendering.shaders import UI_VERT_SHADER, UI_FRAG_SHADER

class UIManager:
    def __init__(self, ctx, width, height, on_action=None):
        self.ctx = ctx
        self.width = width
        self.height = height
        self.on_action = on_action # Callback to Renderer
        
        # ── Texture & Surface ───────────────────────────────────────────────
        self.surf = pygame.Surface((width, height), pygame.SRCALPHA)
        # Ping-Pong textures to avoid read-write hazards causing flicker
        self.texs = [self.ctx.texture((width, height), 4) for _ in range(2)]
        for t in self.texs:
            t.filter = (moderngl.NEAREST, moderngl.NEAREST)
            t.swizzle = 'RGBA'
        self.tex_idx = 0

        # ── Shader Program ──
        self.prog = self.ctx.program(vertex_shader=UI_VERT_SHADER,
                                     fragment_shader=UI_FRAG_SHADER)
        
        # ── Quad Geometry (Full screen) ──
        # x, y, u, v
        data = np.array([
            -1.0,  1.0,  0.0, 0.0,
            -1.0, -1.0,  0.0, 1.0,
             1.0,  1.0,  1.0, 0.0,
             1.0, -1.0,  1.0, 1.0,
        ], dtype='f4')
        self.vbo = self.ctx.buffer(data.tobytes())
        self.vao = self.ctx.vertex_array(self.prog, [(self.vbo, '2f 2f', 'in_position', 'in_texcoord')])

        self.show_panels = True
        self.panel_width = 300
        self.menu_items = [
            {"label": "Simulation", "rect": pygame.Rect(0,0,0,0), "subs": ["Pause/Resume", "Restart", "Exit"]},
            {"label": "View", "rect": pygame.Rect(0,0,0,0), "subs": [
                "Telemetry Panels", "Reset Camera", "-", "Camera: Follow", "Camera: Overview", "Camera: Orbit"
            ]},
        ]
        self.bar_height = 25
        self.font = pygame.font.SysFont("Arial", 14)
        self.active_menu = None # Index of open menu

    def handle_event(self, event):
        if event.type == pygame.VIDEORESIZE:
            if event.w == self.width and event.h == self.height:
                return False
            self.width, self.height = event.w, event.h
            self.surf = pygame.Surface((event.w, event.h), pygame.SRCALPHA)
            for t in self.texs: t.release()
            self.texs = [self.ctx.texture((event.w, event.h), 4) for _ in range(2)]
            for t in self.texs:
                t.filter = (moderngl.NEAREST, moderngl.NEAREST)
                t.swizzle = 'RGBA'
            return False # Let renderer also handle resize

        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                # Check top bar
                if event.pos[1] < self.bar_height:
                    for i, m in enumerate(self.menu_items):
                        if m["rect"].collidepoint(event.pos):
                            self.active_menu = i if self.active_menu != i else None
                            return True # Consumed
                    self.active_menu = None
                elif self.active_menu is not None:
                    # Check sub-menu
                    m = self.menu_items[self.active_menu]
                    for j, sub in enumerate(m["subs"]):
                        sub_rect = pygame.Rect(m["rect"].left, self.bar_height + j*20, 120, 20)
                        if sub_rect.collidepoint(event.pos):
                            self._handle_action(sub)
                            self.active_menu = None
                            return True
                    self.active_menu = None
        return False

    def _handle_action(self, action):
        if action == "-": return
        if action == "Telemetry Panels":
            self.show_panels = not self.show_panels
        elif self.on_action:
            self.on_action(action)

    def draw(self, telemetry_data, camera_mode):
        self.surf.fill((0, 0, 0, 0)) # Clear with transparency
        
        # ── Top Bar ──
        pygame.draw.rect(self.surf, (45, 48, 52), (0, 0, self.width, self.bar_height))
        x_offset = 10
        for i, m in enumerate(self.menu_items):
            label_s = self.font.render(m["label"], True, (220, 220, 220))
            m["rect"] = pygame.Rect(x_offset, 0, label_s.get_width() + 20, self.bar_height)
            if self.active_menu == i:
                pygame.draw.rect(self.surf, (60, 65, 75), m["rect"])
            self.surf.blit(label_s, (x_offset + 10, 5))
            
            # Sub-menu
            if self.active_menu == i:
                for j, sub in enumerate(m["subs"]):
                    sub_rect = pygame.Rect(m["rect"].left, self.bar_height + j*20, 150, 20)
                    pygame.draw.rect(self.surf, (45, 48, 52), sub_rect)
                    text = self.font.render(sub, True, (200, 200, 200))
                    self.surf.blit(text, (sub_rect.left + 5, sub_rect.top + 2))

            x_offset += m["rect"].width

        # Status text on the right side of the top bar
        cam_label = {0: "FOLLOW", 1: "OVERVIEW", 2: "ORBIT"}.get(camera_mode.value-1, "ORBIT") # Quick fix for enum
        # Actually CameraMode is an Enum, let's use its name or value.
        # It's better to just pass the string from Renderer or let UIManager handle the Enum.
        from rendering.types import CameraMode
        cam_text = {CameraMode.FOLLOW: "FOLLOW", CameraMode.OVERVIEW: "OVERVIEW", CameraMode.ORBIT: "ORBIT"}.get(camera_mode, "ORBIT")
        
        status = f"CAM: {cam_text} | "
        if telemetry_data["va"] and telemetry_data["alt"]:
            status += f"ALT: {telemetry_data['alt'][-1]:.1f}m | SPD: {telemetry_data['va'][-1]:.1f}m/s"
        
        status_s = self.font.render(status, True, (150, 200, 255))
        self.surf.blit(status_s, (self.width - status_s.get_width() - 10, 5))

        # ── Telemetry Panels (Sidebar) ──
        if self.show_panels:
            self._draw_side_panels(telemetry_data)

        # ── Upload to Texture (Ping-Pong) ──
        self.tex_idx = (self.tex_idx + 1) % 2
        tex = self.texs[self.tex_idx]
        
        try:
            data = self.surf.get_view('raw')
            tex.write(data)
        except:
            data = pygame.image.tostring(self.surf, "RGBA", False)
            tex.write(data)

        # ── Render Overlay ──
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.CULL_FACE)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        tex.use(0)
        self.vao.render(moderngl.TRIANGLE_STRIP)
        self.ctx.disable(moderngl.BLEND)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.CULL_FACE)
        self.ctx.finish() # Ensure GPU finished before flip

    def _draw_side_panels(self, history):
        margin = 15
        # Fill the entire sidebar area exactly
        rect = pygame.Rect(self.width - self.panel_width, self.bar_height, self.panel_width, self.height - self.bar_height)
        
        if rect.width < 50 or rect.height < 100:
            return # Too small to draw panels

        # Background
        bg = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
        bg.fill((30, 32, 35, 200)) # Slightly more opaque
        self.surf.blit(bg, rect.topleft)

        p_h = (rect.height - 4 * margin) // 3
        if p_h < 20: return
        
        def draw_plot(idx, title, data, color):
            top = rect.top + margin + idx * (p_h + margin)
            r = pygame.Rect(rect.left + 5, top, rect.width - 10, p_h)
            if r.height < 5: return
            pygame.draw.rect(self.surf, (45, 48, 52), r, border_radius=4)
            
            t_s = self.font.render(title, True, (200, 200, 200))
            self.surf.blit(t_s, (r.left + 5, r.top + 2))
            
            if len(data) < 2: return
            pts = []
            d_min, d_max = min(data), max(data)
            span = max(d_max - d_min, 0.1)
            for i, v in enumerate(data):
                x = r.left + (i / (len(data)-1)) * r.width
                y = r.bottom - 5 - ((v - d_min) / span) * (r.height - 25)
                pts.append((x, y))
            pygame.draw.lines(self.surf, color, False, pts, 2)

        draw_plot(0, f"AIRSPEED (m/s) {history['va'][-1]:.1f}" if history['va'] else "AIRSPEED", history['va'], (100, 200, 255))
        draw_plot(1, f"ALTITUDE (m) {history['alt'][-1]:.1f}" if history['alt'] else "ALTITUDE", history['alt'], (255, 200, 100))
        
        # Trajectory
        idx = 2
        top = rect.top + margin + idx * (p_h + margin)
        r = pygame.Rect(rect.left + 5, top, rect.width - 10, p_h)
        pygame.draw.rect(self.surf, (45, 48, 52), r, border_radius=4)
        self.surf.blit(self.font.render("TRAJECTORY", True, (200, 200, 200)), (r.left + 5, r.top + 2))
        
        pos_data = history["pos"]
        if len(pos_data) >= 2:
            pn, pe = [p[0] for p in pos_data], [p[1] for p in pos_data]
            n_min, n_max = min(pn), max(pn)
            e_min, e_max = min(pe), max(pe)
            span = max(n_max - n_min, e_max - e_min, 10.0)
            c_n, c_e = (n_min + n_max)/2, (e_min + e_max)/2
            pts = []
            for p in pos_data:
                x = r.centerx + (p[1] - c_e) / span * (r.width - 20)
                y = r.centery - (p[0] - c_n) / span * (r.height - 40)
                pts.append((x, y))
            pygame.draw.lines(self.surf, (150, 255, 150), False, pts, 2)
            pygame.draw.circle(self.surf, (255, 100, 100), (int(pts[-1][0]), int(pts[-1][1])), 3)
