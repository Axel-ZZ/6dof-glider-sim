from __future__ import annotations
import numpy as np


class Mesh:
    def __init__(self, vertices: np.ndarray, faces: np.ndarray):
        self.vertices = np.asarray(vertices, dtype=np.float32)
        self.faces    = np.asarray(faces,    dtype=np.uint32)
        self.normals  = self._compute_normals()

    def _compute_normals(self) -> np.ndarray:
        v, f = self.vertices, self.faces
        face_normals = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
        normals = np.zeros_like(v)
        for i in range(3):
            np.add.at(normals, f[:, i], face_normals)
        lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        lengths = np.where(lengths == 0, 1.0, lengths)
        return (normals / lengths).astype(np.float32)

    def interleaved(self) -> np.ndarray:
        return np.hstack([self.vertices, self.normals]).astype(np.float32)

    def __repr__(self) -> str:
        return f"Mesh(vertices={len(self.vertices)}, faces={len(self.faces)})"