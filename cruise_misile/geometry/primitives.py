from __future__ import annotations
import numpy as np

def ring(radius: float, y: float, segments: int) -> np.ndarray:
    angles = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    return np.stack([
        np.full(segments, y),
        np.cos(angles) * radius,
        np.sin(angles) * radius,
    ], axis=1).astype(np.float32)


def tube(
    r_start: float,
    r_end: float,
    y_start: float,
    y_end: float,
    segments: int,
    offset: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    ra = ring(r_start, y_start, segments)
    rb = ring(r_end,   y_end,   segments)
    verts = np.vstack([ra, rb])
    faces = []
    for i in range(segments):
        j = (i + 1) % segments
        a, b = i + offset,            j + offset
        c, d = i + segments + offset, j + segments + offset
        faces += [[a, b, c], [b, d, c]]
    return verts, np.array(faces, dtype=np.uint32)


def hemisphere(
    radius: float,
    y_base: float,
    segments: int,
    rings: int = 8,
    offset: int = 0,
    flip: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    sign = -1 if flip else 1
    thetas = np.linspace(0, np.pi / 2, rings + 1)

    ring_verts = [
        ring(radius * np.cos(t), y_base + sign * radius * np.sin(t), segments)
        for t in thetas
    ]

    faces = []
    for i in range(rings - 1):
        base = offset + i * segments
        for j in range(segments):
            k = (j + 1) % segments
            a, b = base + j,            base + k
            c, d = base + segments + j, base + segments + k
            if flip:
                faces += [[a, c, b], [b, c, d]]
            else:
                faces += [[a, b, c], [b, d, c]]

    tip = np.array([y_base + sign * radius, 0.0, 0.0], dtype=np.float32)
    tip_idx        = offset + rings * segments
    last_ring_base = offset + (rings - 1) * segments
    for i in range(segments):
        j = (i + 1) % segments
        if flip:
            faces.append([tip_idx, last_ring_base + j, last_ring_base + i])
        else:
            faces.append([tip_idx, last_ring_base + i, last_ring_base + j])

    verts = np.vstack(ring_verts + [tip[None]]).astype(np.float32)
    return verts, np.array(faces, dtype=np.uint32)


def naca_profile(thickness: float, n: int = 16) -> np.ndarray:
    x = np.linspace(0, 1, n + 1)
    y = (thickness / 0.2) * (
        0.2969 * np.sqrt(x)
        - 0.1260 * x
        - 0.3516 * x ** 2
        + 0.2843 * x ** 3
        - 0.1015 * x ** 4
    )
    return np.stack([x, y], axis=1)


def wing_panel(
    root_chord: float,
    half_span: float,
    thickness_frac: float,
    sweep: float,
    dihedral: float,
    x_offset: float,
    chordwise_pts: int = 16,
    flip_y: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    profile = naca_profile(thickness_frac, chordwise_pts)
    n = len(profile)

    all_verts = []

    # Assuming te_x is constant both for root and tip (no backward sweep) 
    te_x = x_offset

    # Root
    le_x_root = te_x + root_chord
    
    upper = [[le_x_root - px * root_chord, 0, +py * root_chord] for px, py in profile]
    lower = [[le_x_root - px * root_chord, 0, -py * root_chord] for px, py in profile]
    all_verts.append(np.array(upper + lower, dtype=np.float32))

    # Tip
    sw = half_span * np.tan(sweep)  # Dx / s = tan(sweep)
    dih = half_span * np.tan(dihedral) # Dz / s = tan(dihedral)
    
    tip_chord = root_chord - sw
    le_x_tip = te_x + tip_chord

    span_y = -half_span if flip_y else half_span

    upper = [[le_x_tip - px * tip_chord, span_y, +py * tip_chord + dih] for px, py in profile]
    lower = [[le_x_tip - px * tip_chord, span_y, -py * tip_chord + dih] for px, py in profile]
    all_verts.append(np.array(upper + lower, dtype=np.float32))

    verts = np.vstack(all_verts)
    faces = []
    s0u, s0l = 0, n
    s1u, s1l = 2 * n, 3 * n

    def quad(a, b, c, d):
        nonlocal faces
        if flip_y:
            faces.extend([[a, b, c], [a, c, d]])
        else:
            faces.extend([[a, c, b], [a, d, c]])

    for i in range(n - 1):
        quad(s0u + i, s0u + i + 1, s1u + i + 1, s1u + i)   # upper skin
        quad(s0l + i + 1, s0l + i, s1l + i, s1l + i + 1)   # lower skin

    quad(s0u, s0l, s1l, s1u)  # LE
    te = n - 1
    quad(s0l + te, s0u + te, s1u + te, s1l + te)  # TE

    for i in range(n - 1):
        quad(s1u + i, s1u + i + 1, s1l + i + 1, s1l + i)  # Tip

    return verts, np.array(faces, dtype=np.uint32)