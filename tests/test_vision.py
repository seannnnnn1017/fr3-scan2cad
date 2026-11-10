"""Geometry helpers on synthetic data: projection, visual-hull carving, upright rotation."""
import math

import numpy as np

import _path  # noqa: F401
import vision as vz

K = np.array([[900.0, 0, 640], [0, 900.0, 360], [0, 0, 1]])
D = np.zeros(5)
SIZE = (720, 1280)


def sphere_mask(T, c, r):
    """Silhouette of a sphere: project dense surface samples."""
    u = np.random.default_rng(1).normal(size=(20000, 3))
    P = c + r * u / np.linalg.norm(u, axis=1, keepdims=True)
    uv, _ = vz.project(T, K, D, P)
    m = np.zeros(SIZE, bool)
    uv = np.round(uv).astype(int)
    ok = (uv[:, 0] >= 0) & (uv[:, 0] < SIZE[1]) & (uv[:, 1] >= 0) & (uv[:, 1] < SIZE[0])
    m[uv[ok, 1], uv[ok, 0]] = True
    import cv2
    return cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)).astype(bool)


def test_project_center():
    T = vz.look_at_cam([0.3, 0.0, 0.4], [0.3, 0.0, 0.0])
    uv, depth = vz.project(T, K, D, [[0.3, 0.0, 0.0]])
    assert np.allclose(uv[0], [640, 360], atol=1e-6) and abs(depth[0] - 0.4) < 1e-9


def test_carve_recovers_sphere():
    c, r = np.array([0.25, 0.1, 0.045]), 0.045
    eyes = [c + [0, 0, 0.35]] + [c + 0.4 * np.array([math.cos(a) * math.cos(0.6), math.sin(a) * math.cos(0.6), math.sin(0.6)])
                                 for a in (0.5, 2.6)]
    views = []
    for e in eyes:
        T = vz.look_at_cam(e, c)
        views.append((T, sphere_mask(T, c, r)))
    P = vz.carve(views, K, D, c - 0.12 + [0, 0, 0.12 - c[2]], c + 0.12, voxel=0.004, min_seen=3)
    centre = (np.percentile(P, 0.5, 0) + np.percentile(P, 99.5, 0)) / 2
    assert np.linalg.norm(centre[:2] - c[:2]) < 0.006, centre
    top = np.percentile(P[:, 2], 99.5)
    assert 2 * r - 0.006 < top < 2 * r + 0.03, top  # hull may only over-estimate the height


def test_upright_rotates_world_up_to_image_up():
    T = vz.look_at_cam([0.6, 0.0, 0.3], [0.3, 0.0, 0.05], roll_deg=70)
    img = np.zeros((*SIZE, 3), np.uint8)
    out, ang = vz.upright(img, T, K, D, [0.3, 0.0, 0.05])
    uv, _ = vz.project(T, K, D, [[0.3, 0, 0.05], [0.3, 0, 0.10]])
    v = uv[1] - uv[0]
    a = math.radians(ang)
    # cv2 rotation by +ang (counter-clockwise in the image) must map v onto straight up (0, -1)
    R = np.array([[math.cos(a), math.sin(a)], [-math.sin(a), math.cos(a)]])
    w = R @ v
    assert w[1] < 0 and abs(w[0]) < 1e-6 * np.linalg.norm(v) + 1e-6, (ang, w)


if __name__ == "__main__":
    test_project_center()
    test_carve_recovers_sphere()
    test_upright_rotates_world_up_to_image_up()
    print("vision OK")
