"""AprilTag geometry, camera intrinsic calibration, hand-eye calibration (eye-in-hand)."""
from __future__ import annotations

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

_params = cv2.aruco.DetectorParameters()
_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
_params.cornerRefinementWinSize = 5
_DET = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), _params)


# ----------------------------------------------------------------------------- SE(3) helpers
def T_from(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.ravel(t)
    return T


def inv(T):
    R, t = T[:3, :3], T[:3, 3]
    return T_from(R.T, -R.T @ t)


def T_from_rvec(rvec, tvec):
    return T_from(cv2.Rodrigues(np.asarray(rvec, float).reshape(3))[0], tvec)


def to_vec6(T):
    return np.r_[Rotation.from_matrix(T[:3, :3]).as_rotvec(), T[:3, 3]]


def from_vec6(v):
    return T_from(Rotation.from_rotvec(v[:3]).as_matrix(), v[3:])


def average_T(Ts):
    R = Rotation.from_matrix(np.array([T[:3, :3] for T in Ts])).mean().as_matrix()
    return T_from(R, np.mean([T[:3, 3] for T in Ts], 0))


def rot_deg(Ra, Rb):
    return float(np.degrees(np.linalg.norm(Rotation.from_matrix(Ra.T @ Rb).as_rotvec())))


# ----------------------------------------------------------------------------- tags
def tag_corners_board(board, tag_id):
    """Outer black-boundary corners in the board frame, order matching OpenCV's detection."""
    s = float(board["tag_size_m"]) / 2
    x, y = board["centers_m"][str(tag_id)]
    return np.array([[x - s, y + s, 0], [x + s, y + s, 0], [x + s, y - s, 0], [x - s, y - s, 0]], np.float64)


def detect_tags(rgb, board):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    corners, ids, _ = _DET.detectMarkers(gray)
    out = []
    if ids is None:
        return out
    for c, i in zip(corners, ids.ravel()):
        if str(int(i)) in board["centers_m"]:
            out.append({"id": int(i), "corners": c.reshape(4, 2).astype(np.float64)})
    return out


def correspondences(board, dets):
    obj = np.concatenate([tag_corners_board(board, d["id"]) for d in dets])
    img = np.concatenate([d["corners"] for d in dets])
    return obj, img


def pnp(board, K, dist, dets, min_tags=2):
    """camera_T_board from tag corners.  Returns (T, rmse_px) or (None, None)."""
    if len(dets) < min_tags:
        return None, None
    obj, img = correspondences(board, dets)
    ok, r, t = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_IPPE if len(dets) == 1 else cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None, None
    r, t = cv2.solvePnPRefineLM(obj, img, K, dist, r, t)
    proj = cv2.projectPoints(obj, r, t, K, dist)[0].reshape(-1, 2)
    return T_from_rvec(r, t), float(np.sqrt(np.mean(np.sum((proj - img) ** 2, 1))))


def draw_tags(rgb, dets, K=None, dist=None, cam_T_board=None, axis_len=0.08):
    vis = rgb.copy()
    for d in dets:
        c = d["corners"].astype(int)
        cv2.polylines(vis, [c.reshape(-1, 1, 2)], True, (0, 255, 0), 3)
        cv2.putText(vis, str(d["id"]), tuple(c.mean(0).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 40, 40), 3)
    if cam_T_board is not None:
        r = cv2.Rodrigues(cam_T_board[:3, :3])[0]
        cv2.drawFrameAxes(vis, K, dist, r, cam_T_board[:3, 3], axis_len, 4)
    return vis


# ----------------------------------------------------------------------------- calibration
def calibrate_intrinsics(board, views, image_size, K0):
    """views: list of detection lists.  Planar tag board, several orientations (Zhang)."""
    objs, imgs = [], []
    for dets in views:
        if len(dets) >= 2:
            o, i = correspondences(board, dets)
            objs.append(o.astype(np.float32))
            imgs.append(i.astype(np.float32))
    if len(objs) < 6:
        raise RuntimeError(f"too few calibration views with >=2 tags: {len(objs)}")
    flags = cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_ASPECT_RATIO  # square pixels
    rms, K, dist, _, _ = cv2.calibrateCamera(objs, imgs, tuple(image_size), K0.copy(), np.zeros(5), flags=flags)
    return K, dist.ravel(), float(rms), len(objs)


def hand_eye(base_T_link8_list, cam_T_board_list):
    """Eye-in-hand: returns link8_T_cam and base_T_board (Park + averaged)."""
    Rg = [T[:3, :3] for T in base_T_link8_list]
    tg = [T[:3, 3] for T in base_T_link8_list]
    Rt = [T[:3, :3] for T in cam_T_board_list]
    tt = [T[:3, 3] for T in cam_T_board_list]
    R, t = cv2.calibrateHandEye(Rg, tg, Rt, tt, method=cv2.CALIB_HAND_EYE_PARK)
    X = T_from(R, t)
    B = average_T([G @ X @ C for G, C in zip(base_T_link8_list, cam_T_board_list)])
    return X, B


def refine_hand_eye(board, K, dist, base_T_link8_list, dets_list, X0, B0):
    """Joint bundle adjustment of link8_T_cam (X) and base_T_board (B) on tag-corner reprojection."""
    data = [(G, *correspondences(board, d)) for G, d in zip(base_T_link8_list, dets_list) if len(d) >= 1]

    def resid(v):
        X, B = from_vec6(v[:6]), from_vec6(v[6:])
        res = []
        for G, obj, img in data:
            cam_T_board = inv(G @ X) @ B
            r = cv2.Rodrigues(cam_T_board[:3, :3])[0]
            proj = cv2.projectPoints(obj, r, cam_T_board[:3, 3], K, dist)[0].reshape(-1, 2)
            res.append((proj - img).ravel())
        return np.concatenate(res)

    v0 = np.r_[to_vec6(X0), to_vec6(B0)]
    sol = least_squares(resid, v0, loss="huber", f_scale=2.0)
    r = resid(sol.x).reshape(-1, 2)
    return from_vec6(sol.x[:6]), from_vec6(sol.x[6:]), float(np.sqrt(np.mean(np.sum(r**2, 1))))


# ----------------------------------------------------------------------------- camera geometry
def look_at_cam(eye, target, up=(0, 0, 1), roll_deg=0.0, nadir_hint=(1, 0, 0)):
    """board_T_cam for an OpenCV camera (x right, y down, z forward) at eye looking at target."""
    eye, target, up = (np.asarray(v, float) for v in (eye, target, up))
    z = target - eye
    z /= np.linalg.norm(z)
    y0 = -up
    y = y0 - (y0 @ z) * z
    if np.linalg.norm(y) < 1e-6:
        h = np.asarray(nadir_hint, float)
        y = h - (h @ z) * z
    y /= np.linalg.norm(y)
    x = np.cross(y, z)
    R = np.column_stack([x, y, z])
    if roll_deg:
        R = R @ Rotation.from_euler("z", roll_deg, degrees=True).as_matrix()
    return T_from(R, eye)


def pixel_ray(K, dist, uv):
    p = cv2.undistortPoints(np.asarray(uv, np.float64).reshape(-1, 1, 2), K, dist).reshape(-1, 2)
    return np.c_[p, np.ones(len(p))]


def ray_plane(board_T_cam, K, dist, uv, z=0.0):
    """Intersect pixel rays with the board-frame plane Z=z."""
    d = (board_T_cam[:3, :3] @ pixel_ray(K, dist, uv).T).T
    o = board_T_cam[:3, 3]
    s = (z - o[2]) / d[:, 2]
    return o + d * s[:, None]


def project(board_T_cam, K, dist, pts):
    cam_T_board = inv(board_T_cam)
    r = cv2.Rodrigues(cam_T_board[:3, :3])[0]
    uv = cv2.projectPoints(np.ascontiguousarray(np.asarray(pts, np.float64).reshape(-1, 3)), r, cam_T_board[:3, 3], K, dist)[0].reshape(-1, 2)
    depth = (np.asarray(pts).reshape(-1, 3) @ cam_T_board[:3, :3].T + cam_T_board[:3, 3])[:, 2]
    return uv, depth


def carve(views, K, dist, lo, hi, voxel=0.004, dilate_px=6, min_seen=2):
    """Silhouette carving.  views: [(board_T_cam, mask_bool)].  Returns occupied voxel centres.

    A voxel survives if it projects inside the (dilated) mask in every view that sees it, and is
    seen by at least min_seen views (voxels outside every frustum carry no evidence).
    """
    axes = [np.arange(lo[i] + voxel / 2, hi[i], voxel) for i in range(3)]
    P = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    keep = np.ones(len(P), bool)
    seen = np.zeros(len(P), int)
    kernel = np.ones((2 * dilate_px + 1,) * 2, np.uint8)
    for T, mask in views:
        m = cv2.dilate(mask.astype(np.uint8), kernel) > 0
        uv, depth = project(T, K, dist, P)
        h, w = m.shape
        u, v = np.round(uv[:, 0]).astype(int), np.round(uv[:, 1]).astype(int)
        inside = (u >= 0) & (u < w) & (v >= 0) & (v < h) & (depth > 0)
        ok = np.ones(len(P), bool)
        ok[inside] = m[v[inside], u[inside]]
        keep &= ok
        seen += inside
    return P[keep & (seen >= min(min_seen, len(views)))]


def fill_object_holes(rgb, mask, min_lab_dist=18.0, ring_px=12):
    """Segmenters sometimes leave the *inside* of an object out of its mask (e.g. looking into a mug).
    A hole is filled unless it looks like the background just outside the object (then it is a real
    see-through gap, like a mug handle).  Returns (mask, number of pixels filled)."""
    m = mask.astype(np.uint8)
    n, lab = cv2.connectedComponents(1 - m, connectivity=4)  # label 0 = the mask itself
    border = set(np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]]).tolist())
    holes = [i for i in range(1, n) if i not in border]
    if not holes:
        return mask, 0
    solid = m.copy()
    for i in holes:
        solid[lab == i] = 1
    ring = (cv2.dilate(solid, np.ones((2 * ring_px + 1,) * 2, np.uint8)) > 0) & (solid == 0)
    L = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(float)
    bg = np.median(L[ring], 0) if ring.any() else None
    out, filled = mask.copy(), 0
    for i in holes:
        h = lab == i
        if bg is None or h.sum() < 30 or np.linalg.norm(np.median(L[h], 0) - bg) > min_lab_dist:
            out[h] = True
            filled += int(h.sum())
    return out, filled


def upright(img, board_T_cam, K, dist, center, up=(0, 0, 1)):
    """Rotate an image about the projected object centre so the world up axis points up in the image
    (image generators expect upright objects; the planner may have rolled the camera).
    Returns (rotated image on an enlarged canvas, applied angle in degrees)."""
    p = np.asarray(center, float)
    uv, _ = project(board_T_cam, K, dist, np.stack([p, p + 0.05 * np.asarray(up, float)]))
    v = uv[1] - uv[0]
    ang = float(np.degrees(np.arctan2(v[0], -v[1])))  # angle of the projected up axis from image-up (clockwise)
    view_dir = np.asarray(board_T_cam)[:3, 2]  # optical axis in the world frame
    if np.degrees(np.arccos(np.clip(-view_dir @ np.asarray(up, float), -1, 1))) < 20:
        ang = 0.0  # (near) top-down view: "up" projects to a point, leave the image as it is
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((float(uv[0][0]), float(uv[0][1])), ang, 1.0)
    c, s = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * s + w * c), int(h * c + w * s)
    M[0, 2] += nw / 2 - uv[0][0]
    M[1, 2] += nh / 2 - uv[0][1]
    out = cv2.warpAffine(img, M, (nw, nh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return out, ang
