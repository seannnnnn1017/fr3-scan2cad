"""Nominal Franka FR3 kinematics (published modified-DH parameters) + damped-least-squares IK.

Only datasheet knowledge is used here, exactly what you would have for the real arm.
Frame 8 is the flange (fr3_link8); the wrist camera pose relative to it comes from
hand-eye calibration, not from this file.
"""
from __future__ import annotations

import numpy as np

# (a_{i-1}, d_i, alpha_{i-1}); last row is the flange (fixed).
DH = [(0, 0.333, 0), (0, 0, -np.pi / 2), (0, 0.316, np.pi / 2), (0.0825, 0, np.pi / 2),
      (-0.0825, 0.384, -np.pi / 2), (0, 0, np.pi / 2), (0.088, 0, np.pi / 2), (0, 0.107, 0)]
# FR3 datasheet joint limits (rad).
LIMITS = np.array([[-2.7437, 2.7437], [-1.7837, 1.7837], [-2.9007, 2.9007], [-3.0421, -0.1518],
                   [-2.8065, 2.8065], [0.5445, 4.5169], [-3.0159, 3.0159]])
MID = LIMITS.mean(1)


def _dh(a, d, al, th):
    ca, sa, ct, st = np.cos(al), np.sin(al), np.cos(th), np.sin(th)
    return np.array([[ct, -st, 0, a], [st * ca, ct * ca, -sa, -d * sa], [st * sa, ct * sa, ca, d * ca], [0, 0, 0, 1]])


def fk_all(q):
    """Base-frame transforms of frames 1..8 (8 = flange)."""
    T = np.eye(4)
    out = []
    for i, (a, d, al) in enumerate(DH):
        T = T @ _dh(a, d, al, q[i] if i < 7 else 0.0)
        out.append(T.copy())
    return out


def fk_link8(q):
    return fk_all(q)[-1]


def rot_err(R, Rd):
    """Rotation vector (base frame) taking R to Rd."""
    dR = Rd @ R.T
    angle = np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))
    v = np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]])
    if angle < 1e-9:
        return 0.5 * v
    if np.pi - angle < 1e-4:  # near pi: use the symmetric part
        w, V = np.linalg.eigh((dR + dR.T) / 2)
        return V[:, -1] * angle
    return v * angle / (2 * np.sin(angle))


def jacobian(q, tool=np.eye(4)):
    Ts = fk_all(q)
    pe = (Ts[-1] @ tool)[:3, 3]
    J = np.zeros((6, 7))
    for i in range(7):
        z, o = Ts[i][:3, 2], Ts[i][:3, 3]
        J[:3, i] = np.cross(z, pe - o)
        J[3:, i] = z
    return J


def random_q(rng):
    return LIMITS[:, 0] + rng.random(7) * (LIMITS[:, 1] - LIMITS[:, 0])


def ik(T_target, q_seed, tool=np.eye(4), limits=LIMITS, iters=150, tol_p=5e-4, tol_r=np.radians(0.3)):
    """Solve base_T_link8 @ tool = T_target.  Returns (q, pos_err_m, rot_err_rad)."""
    q = np.clip(np.asarray(q_seed, float), limits[:, 0] + 1e-3, limits[:, 1] - 1e-3)
    mid = limits.mean(1)
    span = limits[:, 1] - limits[:, 0]
    lam = 0.05
    for _ in range(iters):
        T = fk_link8(q) @ tool
        ep = T_target[:3, 3] - T[:3, 3]
        er = rot_err(T[:3, :3], T_target[:3, :3])
        if np.linalg.norm(ep) < tol_p and np.linalg.norm(er) < tol_r:
            break
        e = np.concatenate([np.clip(ep, -0.05, 0.05), np.clip(er, -0.3, 0.3)])
        J = jacobian(q, tool)
        JJt = J @ J.T + lam**2 * np.eye(6)
        Jp = J.T @ np.linalg.inv(JJt)
        dq = Jp @ e
        # null-space pull toward the middle of the joint range
        dq += (np.eye(7) - Jp @ J) @ (-(q - mid) / span * 0.2)
        q = np.clip(q + np.clip(dq, -0.2, 0.2), limits[:, 0] + 1e-4, limits[:, 1] - 1e-4)
    T = fk_link8(q) @ tool
    return q, float(np.linalg.norm(T_target[:3, 3] - T[:3, 3])), float(np.linalg.norm(rot_err(T[:3, :3], T_target[:3, :3])))


def ik_multi(T_target, seeds, tool=np.eye(4), limits=LIMITS, rng=None, n_random=4):
    """Try several seeds; return the converged solution closest to seeds[0] (current q)."""
    rng = rng or np.random.default_rng(0)
    # reach pre-check: flange must lie within the FR3 workspace sphere around the shoulder
    if np.linalg.norm(T_target[:3, 3] - [0, 0, 0.333]) > 0.86 or T_target[2, 3] < -0.05:
        return None
    best = None
    for s in list(seeds):
        q, ep, er = ik(T_target, s, tool, limits)
        if ep < 1e-3 and er < np.radians(0.5):
            return q  # nearest-seed solution: smooth, predictable motion
    cands = [random_q(rng) * 0.5 + limits.mean(1) * 0.5 for _ in range(n_random)]
    for s in cands:
        q, ep, er = ik(T_target, s, tool, limits)
        if ep < 1e-3 and er < np.radians(0.5):
            margin = np.min(np.minimum(q - limits[:, 0], limits[:, 1] - q))
            cost = np.linalg.norm((q - seeds[0])[:6]) - 2.0 * min(margin, 0.3)
            if best is None or cost < best[0]:
                best = (cost, q)
    return None if best is None else best[1]


def link_points(q, tool_points_link8=()):
    """Sample points along the arm (for crude clearance checks): joint origins, link midpoints, tool points."""
    Ts = fk_all(q)
    pts = [T[:3, 3] for T in Ts]
    pts += [(Ts[i][:3, 3] + Ts[i + 1][:3, 3]) / 2 for i in range(len(Ts) - 1)]
    for p in tool_points_link8:
        pts.append((Ts[-1] @ np.r_[p, 1.0])[:3])
    return np.array(pts)
