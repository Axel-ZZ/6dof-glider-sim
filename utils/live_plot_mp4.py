from pathlib import Path

import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import trimesh

matplotlib.rcParams["animation.ffmpeg_path"] = str(
    Path.home()
    / r"AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1-full_build\bin\ffmpeg.exe"
)


def _load_stl(stl_path: Path) -> np.ndarray:
    mesh = trimesh.load(str(stl_path), force="mesh")
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces)
    return verts[faces].reshape(-1, 3)


def _prepare_mesh(raw_flat: np.ndarray, body_length: float, stl_scale: float):
    center = raw_flat.mean(axis=0)
    centered = raw_flat - center

    centered[..., 2] *= -1.0
    centered[..., 0] *= -1.0

    extent = np.ptp(centered, axis=0)
    norm = max(extent.max(), 1e-9)
    scale = body_length * 0.12 * stl_scale / norm

    n_faces = centered.shape[0] // 3
    return centered, n_faces, scale


class LivePlot3DMp4Creator:
    def __init__(self, title, fps=30):
        self.title = title
        self.fps = int(max(1, fps))

    @staticmethod
    def _body_to_ned(phi_i, theta_i, psi_i):
        cphi, sphi = np.cos(phi_i), np.sin(phi_i)
        ctheta, stheta = np.cos(theta_i), np.sin(theta_i)
        cpsi, spsi = np.cos(psi_i), np.sin(psi_i)
        return np.array([
            [ctheta * cpsi, sphi * stheta * cpsi - cphi * spsi, cphi * stheta * cpsi + sphi * spsi],
            [ctheta * spsi, sphi * stheta * spsi + cphi * cpsi, cphi * stheta * spsi - sphi * cpsi],
            [-stheta, sphi * ctheta, cphi * ctheta],
        ])

    def create(
        self,
        pn,
        pe,
        alt,
        phi,
        theta,
        psi,
        time_mesh,
        path_3d=None,
        stl_path: str | Path | None = None,
        stl_scale: float = 8.0,
    ):
        pn = np.asarray(pn).reshape(-1)
        pe = np.asarray(pe).reshape(-1)
        alt = np.asarray(alt).reshape(-1)
        phi = np.asarray(phi).reshape(-1)
        theta = np.asarray(theta).reshape(-1)
        psi = np.asarray(psi).reshape(-1)
        time_mesh = np.asarray(time_mesh, dtype=float).reshape(-1)

        n_points = len(pe)
        if not (len(pn) == n_points == len(alt) == len(phi) == len(theta) == len(psi) == len(time_mesh)):
            raise ValueError("All state arrays and time_mesh must have the same length.")
        if n_points < 2:
            raise ValueError("Need at least two points to build MP4 animation.")
        if np.any(np.diff(time_mesh) <= 0.0):
            raise ValueError("time_mesh must be strictly increasing.")

        if stl_path is None:
            here = Path(__file__).resolve().parent
            stl_path = here.parent / "f5b_glider_v2.stl"

        fig = plt.figure(figsize=(7.5, 5.5))
        ax = fig.add_subplot(111, projection="3d")

        min_e, max_e = np.min(pe), np.max(pe)
        min_n, max_n = np.min(pn), np.max(pn)
        min_alt, max_alt = np.min(alt), np.max(alt)

        north_range = max_n - min_n
        east_range = max_e - min_e
        alt_range = max_alt - min_alt

        pad_east = (north_range - east_range) / 2
        pad_north = 0.10 * (north_range + 1e-6)
        pad_alt = 0.50 * (alt_range + 1e-6)

        span = max(north_range, east_range, alt_range, 1e-3)
        body_length = max(6.0, 0.10 * span)

        raw = _load_stl(Path(stl_path))
        tris_flat, n_faces, mesh_scale = _prepare_mesh(raw, body_length, stl_scale)
        print(
            f"[live_plot_mp4] STL loaded: {Path(stl_path).name}  "
            f"({n_faces} faces, display scale x{mesh_scale:.2f})"
        )

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

        ax.set_xlim(min_e - pad_east, max_e + pad_east)
        ax.set_ylim(min_n - pad_north, max_n + pad_north)
        ax.set_zlim(min_alt - pad_alt, max_alt + pad_alt)
        ax.set_xlabel("East [m]")
        ax.set_ylabel("North [m]")
        ax.set_zlabel("Altitude [m]")
        ax.set_title(self.title)
        ax.grid(True, alpha=0.25)
        ax.view_init(elev=10, azim=-20)
        try:
            ax.set_box_aspect((1.0, 1.0, 1.0))
        except Exception:
            pass

        ax.plot(pe, pn, alt, color="royalblue", linewidth=1.8, alpha=0.35)
        ax.scatter(pe[0], pn[0], alt[0], color="green", s=35)
        ax.scatter(pe[-1], pn[-1], alt[-1], color="black", s=35)

        trail, = ax.plot([], [], [], color="royalblue", linewidth=2.2, alpha=0.8)
        aircraft_collection = Poly3DCollection(
            [], facecolors="#FF0000", edgecolors="none", linewidths=0.0, alpha=1.0
        )
        ax.add_collection3d(aircraft_collection)

        total_time = float(time_mesh[-1] - time_mesh[0])
        frame_times = np.arange(0.0, total_time + 1e-12, 1.0 / self.fps)
        if frame_times[-1] < total_time:
            frame_times = np.append(frame_times, total_time)

        shifted_mesh = time_mesh - time_mesh[0]
        state_idx = np.searchsorted(shifted_mesh, frame_times, side="right") - 1
        state_idx = np.clip(state_idx, 0, n_points - 1)

        def init():
            trail.set_data([], [])
            trail.set_3d_properties([])
            aircraft_collection.set_verts([])
            return trail, aircraft_collection

        def update(frame_i):
            i = int(state_idx[frame_i])
            pos = np.array([pe[i], pn[i], alt[i]])
            rot = self._body_to_ned(phi[i], theta[i], psi[i])

            trail.set_data(pe[: i + 1], pn[: i + 1])
            trail.set_3d_properties(alt[: i + 1])

            rotated_ned = tris_flat @ rot.T
            rotated_plot = np.column_stack([
                rotated_ned[:, 1],
                rotated_ned[:, 0],
                -rotated_ned[:, 2],
            ])
            rotated_plot = rotated_plot.reshape(n_faces, 3, 3) * mesh_scale + pos
            aircraft_collection.set_verts(rotated_plot)
            return trail, aircraft_collection

        ani = animation.FuncAnimation(
            fig,
            update,
            frames=len(frame_times),
            init_func=init,
            blit=False,
            interval=1000.0 / self.fps,
            repeat=False,
        )

        videos_dir = Path(__file__).resolve().parents[1] / "videos"
        videos_dir.mkdir(parents=True, exist_ok=True)
        if path_3d is None:
            save_dir = videos_dir
        else:
            requested_dir = Path(path_3d)
            save_dir = requested_dir if requested_dir.is_absolute() else videos_dir / requested_dir
        save_dir.mkdir(parents=True, exist_ok=True)
        mp4_path = save_dir / f"{self.title.replace(' ', '_').lower()}_3D_live.mp4"

        try:
            writer = animation.FFMpegWriter(fps=self.fps, bitrate=1800)
            ani.save(mp4_path, writer=writer)
            print(f"{self.title} MP4 saved to: {mp4_path}")
        except Exception as exc:
            print("Could not save 3D MP4 animation. Make sure ffmpeg is installed and on PATH.")
            print(f"Could not save 3D MP4 animation: {exc}")

        # Save-only behavior: do not display the MP4 figure.
        plt.close(fig)
        return mp4_path