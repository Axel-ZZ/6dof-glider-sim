import trimesh
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pathlib import Path


def main():
    stl_path = Path(__file__).resolve().parents[1] / "f5b_glider_v2.stl"
    if not stl_path.exists():
        print(f"STL not found at: {stl_path}")
        return

    mesh = trimesh.load(stl_path, force='mesh')
    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    print(f"Loaded mesh: verts={verts.shape}, faces={faces.shape}")

    # triangles as (n,3,3)
    triangles = verts[faces]

    # center and scale for nice viewing
    all_pts = triangles.reshape(-1, 3)
    center = all_pts.mean(axis=0)
    triangles_centered = triangles - center
    extent = np.ptp(triangles_centered.reshape(-1, 3), axis=0)
    scale = 1.0 / (np.max(extent) + 1e-9)
    # flip mesh forward axis if model points backwards
    try:
        triangles_centered[..., 0] *= -1.0
    except Exception:
        pass
    # make the drone larger for visibility
    SCALE_MULTIPLIER = 3.0
    triangles_centered *= scale * SCALE_MULTIPLIER

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Render faces without heavy edges so the red color is clear
    poly = Poly3DCollection(triangles_centered, facecolor=(1.0, 0.0, 0.0, 1.0),
                            edgecolor='none', linewidths=0.0)
    ax.add_collection3d(poly)

    # autoscale to the triangles
    pts = triangles_centered.reshape(-1, 3)
    max_range = (pts.max(axis=0) - pts.min(axis=0)).max()
    mid = (pts.max(axis=0) + pts.min(axis=0)) / 2.0
    ax.set_xlim(mid[0] - max_range / 2, mid[0] + max_range / 2)
    ax.set_ylim(mid[1] - max_range / 2, mid[1] + max_range / 2)
    ax.set_zlim(mid[2] - max_range / 2, mid[2] + max_range / 2)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('STL Render Demo (triangles)')

    print(f"Number of triangles: {triangles.shape[0]}")
    plt.show()


if __name__ == '__main__':
    main()
