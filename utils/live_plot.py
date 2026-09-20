import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import trimesh




def _load_stl(stl_path: Path) -> np.ndarray:
    """Load an STL and return vertices as (N*3, 3) float64 (flat over triangles)."""
    mesh = trimesh.load(str(stl_path), force='mesh')
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces)
    # shape (n_faces, 3, 3) → (n_faces*3, 3)
    return verts[faces].reshape(-1, 3)


def _prepare_mesh(raw_flat: np.ndarray, body_length: float, stl_scale: float):
    """
    Centre and scale the flat vertex array, applying the same orientation
    fixes as the original live_plot (flip Z upright, flip X forward).

    Returns
    -------
    tris_flat  : (N*3, 3) centred + oriented vertices, ready to rotate each frame
    n_faces    : int
    scale      : float
    """
    center = raw_flat.mean(axis=0)
    centered = raw_flat - center

    # Original orientation fixes — proven to work, don't touch
    centered[..., 2] *= -1.0   # flip Z so model is upright
    centered[..., 0] *= -1.0   # flip X so nose points forward

    extent = np.ptp(centered, axis=0)
    norm   = max(extent.max(), 1e-9)
    scale  = body_length * 0.12 * stl_scale / norm

    n_faces = centered.shape[0] // 3
    return centered, n_faces, scale


def _normalize_waypoints(raw):
    if raw is None:
        return []
    arr = np.asarray(raw, dtype=float)
    if arr.size == 0:
        return []
    if arr.ndim == 1:
        if arr.shape[0] < 3:
            raise ValueError("Each waypoint must have at least [pn, pe, pd].")
        return [arr[:3]]
    if arr.ndim == 2:
        if arr.shape[1] < 3:
            raise ValueError("Each waypoint must have at least [pn, pe, pd].")
        return [row[:3] for row in arr]
    raise ValueError("waypoints must be a 1-D or 2-D array.")


def _body_to_ned(phi, theta, psi):
    """Direction-cosine matrix: body → NED."""
    cp, sp = np.cos(phi),   np.sin(phi)
    ct, st = np.cos(theta), np.sin(theta)
    cs, ss = np.cos(psi),   np.sin(psi)
    return np.array([
        [ct*cs,  sp*st*cs - cp*ss,  cp*st*cs + sp*ss],
        [ct*ss,  sp*st*ss + cp*cs,  cp*st*ss - sp*cs],
        [-st,    sp*ct,             cp*ct           ],
    ], dtype=np.float64)


def live_plot_3d(
    pn,
    pe,
    alt,
    phi,
    theta,
    psi,
    title,
    save_animation: bool = True,
    sample_time: float | None = None,
    waypoints=None,
    waypoint_labels=None,
    path_3D=None,
    stl_path: str | Path | None = None,
    stl_scale: float = 8.0,
    display_frame_skip: int = 2,
    gif_frame_skip: int = 2,
):
    """
    Animated 3-D trajectory viewer.

    Parameters
    ----------
    gif_frame_skip : int
        Save every Nth frame to the GIF.  gif_frame_skip=2 halves the file
        size and render time with negligible perceptual difference at 30 fps.
    stl_scale : float
        Visual size multiplier for the aircraft mesh.
    stl_path : path-like or None
        Explicit STL file.  Defaults to 'f5b_glider_v2.stl' next to this
        file.
    """
    # ── inputs ──────────────────────────────────────────────────────────────
    pn  = np.asarray(pn,    dtype=float).ravel()
    pe  = np.asarray(pe,    dtype=float).ravel()
    alt = np.asarray(alt,   dtype=float).ravel()
    phi = np.asarray(phi,   dtype=float).ravel()
    theta = np.asarray(theta, dtype=float).ravel()
    psi = np.asarray(psi,   dtype=float).ravel()

    waypoint_points = _normalize_waypoints(waypoints)

    # ── STL loading ─────────────────────────────────────────────────────────
    if stl_path is None:
        here = Path(__file__).resolve().parent
        stl_path = here.parent / "f5b_glider_v2.stl"

    # Pre-compute sizing constants (needed for mesh scaling)
    pe_all  = np.concatenate([pe,  [float(p[1]) for p in waypoint_points]] if waypoint_points else [pe])
    pn_all  = np.concatenate([pn,  [float(p[0]) for p in waypoint_points]] if waypoint_points else [pn])
    alt_all = np.concatenate([alt, [float(-p[2]) for p in waypoint_points]] if waypoint_points else [alt])

    span        = max(np.ptp(pe_all), np.ptp(pn_all), np.ptp(alt_all), 1e-3)
    body_length = max(6.0, 0.10 * span)

    raw = _load_stl(Path(stl_path))
    tris_flat, n_faces, mesh_scale = _prepare_mesh(raw, body_length, stl_scale)
    print(f"[live_plot] STL loaded: {Path(stl_path).name}  "
          f"({n_faces} faces, display scale ×{mesh_scale:.2f})")

    # ── figure / axes ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(7.5, 5.5))
    ax  = fig.add_subplot(111, projection='3d')

    min_e,  max_e  = pe_all.min(),  pe_all.max()
    min_n,  max_n  = pn_all.min(),  pn_all.max()
    min_alt, max_alt = alt_all.min(), alt_all.max()

    north_range = max_n   - min_n
    east_range  = max_e   - min_e
    alt_range   = max_alt - min_alt

    pad_east  = (north_range - east_range) / 2
    pad_north = 0.10 * (north_range + 1e-6)
    pad_alt   = 0.50 * (alt_range  + 1e-6)

    ax.set_xlim(min_e  - pad_east,  max_e  + pad_east)
    ax.set_ylim(min_n  - pad_north, max_n  + pad_north)
    ax.set_zlim(min_alt - pad_alt,  max_alt + pad_alt)
    ax.set_xlabel('East [m]')
    ax.set_ylabel('North [m]')
    ax.set_zlabel('Altitude [m]')
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.view_init(elev=0.0, azim=90.0)
    try:
        ax.set_box_aspect((1.0, 1.0, 1.0))
    except Exception:
        pass

    # ── static artists (drawn once) ──────────────────────────────────────────
    ax.plot(pe, pn, alt, color='royalblue', linewidth=1.8, alpha=0.4, zorder=1)
    ax.scatter(pe[0],  pn[0],  alt[0],  color='green', s=35, zorder=2)
    ax.scatter(pe[-1], pn[-1], alt[-1], color='black', s=35, zorder=2)

    # Waypoint planes (North = 0 m and North = 150 m)
    for pn_plane in (0.0, 150.0):
        verts_plane = np.array([
            [min_e - pad_east,  pn_plane, min_alt - pad_alt],
            [max_e + pad_east,  pn_plane, min_alt - pad_alt],
            [max_e + pad_east,  pn_plane, max_alt + pad_alt],
            [min_e - pad_east,  pn_plane, max_alt + pad_alt],
        ])
        ax.add_collection3d(Poly3DCollection(
            [verts_plane], facecolors="orange", alpha=0.20,edgecolors='none'))

    # ── dynamic artists ──────────────────────────────────────────────────────
    trail, = ax.plot([], [], [], color='royalblue', linewidth=2.2, alpha=1.0)

    # STL mesh collection
    aircraft_collection = Poly3DCollection(
        [], facecolors="#FF0000", edgecolors='none', linewidths=0.0, alpha=1.0)
    ax.add_collection3d(aircraft_collection)

    fig.tight_layout()

    # ── per-frame helpers ────────────────────────────────────────────────────

    def init():
        trail.set_data([], [])
        trail.set_3d_properties([])
        aircraft_collection.set_verts([])
        return (trail,) + (aircraft_collection,)

    def update(i):
        pos = np.array([pe[i], pn[i], alt[i]])
        rot = _body_to_ned(phi[i], theta[i], psi[i])

        # Original rotation approach: rotate flat verts into NED, then
        # swap axes to plot frame exactly as the original code did
        rotated_ned = tris_flat @ rot.T                        # (N*3, 3) in NED
        rotated_plot = np.column_stack([
            rotated_ned[:, 1],    # plot X = NED East
            rotated_ned[:, 0],    # plot Y = NED North
            -rotated_ned[:, 2],   # plot Z = NED Up
        ])
        rotated_plot = rotated_plot.reshape(n_faces, 3, 3) * mesh_scale + pos
        aircraft_collection.set_verts(rotated_plot)

        trail.set_data(pe[:i+1], pn[:i+1])
        trail.set_3d_properties(alt[:i+1])

        return (trail,) + (aircraft_collection,)

    # ── animation ────────────────────────────────────────────────────────────
    interval_ms = (1000 / 30) if sample_time is None else (1000 * sample_time)
    display_skip = max(1, int(display_frame_skip))
    display_frames = list(range(0, len(pe), display_skip))

    ani = animation.FuncAnimation(
        fig, update,
        frames=display_frames,
        init_func=init,
        blit=False,
        interval=interval_ms,
        repeat=True,
    )
    fig._live_animation = ani  # keep reference so GC doesn't collect it

    # ── save GIF ─────────────────────────────────────────────────────────────
    if save_animation:
        plots_dir = Path(__file__).resolve().parents[1] / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        save_dir = plots_dir if path_3D is None else plots_dir / Path(path_3D)
        save_dir.mkdir(parents=True, exist_ok=True)
        gif_path = save_dir / f"{title.replace(' ', '_').lower()}_3D_live.gif"

        skip = max(1, int(gif_frame_skip))
        gif_frames = list(range(0, len(pe), skip))

        try:
            fps = 30 if sample_time is None else max(1, int(round(1.0 / sample_time)))
            gif_fps = max(1, fps // skip)

            ani_gif = animation.FuncAnimation(
                fig, update,
                frames=gif_frames,
                init_func=init,
                blit=False,
                interval=interval_ms,
            )
            ani_gif.save(gif_path, writer=animation.PillowWriter(fps=gif_fps))
            print(f"[live_plot] Saved ({len(gif_frames)} frames @ {gif_fps} fps): {gif_path}")
        except Exception as exc:
            print(f"[live_plot] Could not save GIF: {exc}")

    try:
        fig.canvas.manager.set_window_title(f"{title} - 3D Live View")
    except Exception:
        pass