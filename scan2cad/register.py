"""Metric 6-D registration of the TRELLIS mesh to the visual hull, using only the scan's own evidence.

TRELLIS returns a shape in a normalised, unit-less frame.  The visual hull (carved from the same
calibrated photos) says where the object is and how big it is, in metres.  Find the similarity
transform (scale s, rotation R, translation t) that puts the mesh into the hull:

  align = "hull"             3-D only: mesh surface inside the hull and filling it
  align = "hull+silhouette"  (default) the same, plus the mesh's projected silhouettes must match each
                             photo's mask -- the masks the hull was carved from, so this is the exact
                             form of "consistent with the visual hull"

always with the object resting on the table (z_min = 0).  Both fits are run and saved; the configured
one becomes the CAD.  Outputs: the metric CAD + 6-D pose (board and robot base frame) and the visual
hull itself (points, box, centre) in both frames.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import trimesh
from PIL import Image
from scipy.ndimage import binary_erosion, distance_transform_edt, map_coordinates
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vision as vz  # noqa: E402

DOWN = 4  # silhouette resolution divisor for the fit


def decimate(mesh, n):
    if len(mesh.faces) <= n:
        return mesh
    return mesh.simplify_quadric_decimation(face_count=n)


def similarity(p):
    """p = [log s, rx, ry, yaw, tx, ty, tz]; R = Rz(yaw) Ry(ry) Rx(rx)."""
    R = Rotation.from_euler("zyx", [p[3], p[2], p[1]]).as_matrix()
    return math.exp(p[0]), R, np.asarray(p[4:7])


def apply(V, p):
    s, R, t = similarity(p)
    return s * V @ R.T + t


def silhouette(V, F, T, K, dist, shape, down=DOWN):
    h, w = shape
    uv, depth = vz.project(T, K, dist, V)
    uv = uv / down
    img = np.zeros((h, w), np.uint8)
    ok = (depth[F] > 0.02).all(1) & np.isfinite(uv[F]).all((1, 2)) & (np.abs(uv[F]) < 4 * max(h, w)).all((1, 2))
    tri = np.zeros(F.shape + (2,), np.int32)
    tri[ok] = np.round(uv[F[ok]] * 4).astype(np.int32)  # 2 bits of sub-pixel precision
    for t in tri[ok]:
        cv2.fillConvexPoly(img, t, 1, lineType=cv2.LINE_8, shift=2)
    return img.astype(bool)


class HullField:
    """Visual hull voxels -> distance-outside field (for 'mesh inside hull') and hull surface points
    (for 'mesh fills hull')."""

    def __init__(self, P, voxel=0.004):
        self.v = voxel
        self.lo = P.min(0) - 4 * voxel
        idx = np.round((P - self.lo) / voxel).astype(int)
        occ = np.zeros(idx.max(0) + 5, bool)
        occ[tuple(idx.T)] = True
        self.outside = distance_transform_edt(~occ) * voxel
        surf = occ & ~binary_erosion(occ)
        S = self.lo + np.argwhere(surf) * voxel
        self.S = S[np.random.default_rng(0).permutation(len(S))[:2500]]

    def dist_out(self, X):
        return map_coordinates(self.outside, ((X - self.lo) / self.v).T, order=1, mode="nearest")

    def loss(self, X):
        """mean metres: mesh points outside the hull + 0.5 * hull surface not reached by the mesh."""
        fill = cKDTree(X).query(self.S)[0]
        return self.dist_out(X).mean() + 0.5 * fill.mean()


def fit(loss, s0, placed, label):
    grid = []
    for yaw in np.radians(np.arange(0, 360, 15)):
        for s in (s0 * 0.9, s0, s0 * 1.1):  # scale is the least constrained: try a couple of starts
            p = placed(yaw, s)
            grid.append((loss(p), p))
    grid.sort(key=lambda g: g[0])
    # Multi-start: near-symmetric objects (mug handle!) have deep wrong-yaw minima, so refine the
    # best few distinct yaws (and their 180-degree flips) and keep the lowest final loss.
    starts = []
    for L, p in grid:
        if all(abs((p[3] - q[3] + math.pi) % (2 * math.pi) - math.pi) > math.radians(40) for q in starts):
            starts.append(p)
        if len(starts) == 3:
            break
    starts += [placed(p[3] + math.pi, math.exp(p[0])) for p in starts]
    res = None
    for p in starts:
        bounds = [(math.log(s0) - 0.5, math.log(s0) + 0.5), (-0.3, 0.3), (-0.3, 0.3), (p[3] - math.pi, p[3] + math.pi),
                  (p[4] - 0.06, p[4] + 0.06), (p[5] - 0.06, p[5] + 0.06), (p[6] - 0.06, p[6] + 0.06)]
        r = minimize(loss, p, method="Powell", bounds=bounds, options={"maxiter": 3000, "xtol": 1e-4, "ftol": 1e-6})
        print(f"[register:{label}] start yaw {math.degrees(p[3]) % 360:5.0f}° -> loss {r.fun:.4f}", flush=True)
        if res is None or r.fun < res.fun:
            res = r
    return res


def export_hull(P, B, out):
    """The visual hull as an output in its own right: voxel centres + box + centre, board and base frames."""
    Pb = (B @ np.c_[P, np.ones(len(P))].T).T[:, :3]
    trimesh.PointCloud(P).export(out / "visual_hull_board_frame.ply")
    trimesh.PointCloud(Pb).export(out / "visual_hull_robot_base_frame.ply")
    lo, hi = np.percentile(P, 0.5, 0), np.percentile(P, 99.5, 0)
    C = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    info = {
        "voxel_m": 0.004, "n_voxels": int(len(P)), "volume_m3": float(len(P) * 0.004 ** 3),
        "board_frame": {"center_m": ((lo + hi) / 2).tolist(), "bbox_lo_m": lo.tolist(), "bbox_hi_m": hi.tolist(),
                        "size_m": (hi - lo).tolist()},
        "robot_base_frame": {"center_m": (B @ np.r_[(lo + hi) / 2, 1])[:3].tolist(),
                             "bbox_corners_m": (B @ np.c_[C, np.ones(8)].T).T[:, :3].tolist()},
        "note": "Silhouette-carved hull (superset of the object; concavities and unseen regions are filled). "
                "Points: visual_hull_*_frame.ply, metres.",
    }
    (out / "visual_hull.json").write_text(json.dumps(info, indent=1))
    return info


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--align", choices=["hull", "hull+silhouette", "silhouette"], default=None,
                    help="default: register.align from the scan config, else hull+silhouette")
    a = ap.parse_args()
    run = a.run.resolve()
    cfgp = run / "inputs_config/scan.json"
    align = a.align or (json.loads(cfgp.read_text()).get("register", {}).get("align") if cfgp.exists() else None) or "hull+silhouette"
    meta = json.loads((run / "views/views.json").read_text())
    K, dist = np.array(meta["K"]), np.array(meta["dist"])
    B = np.array(meta["base_T_board"])
    obj = meta["object_estimate"]
    views = meta["views"]
    out = run / "cad"
    out.mkdir(exist_ok=True)

    mesh = trimesh.load(run / "trellis/mesh_native.glb", force="mesh", process=False)
    fit_mesh = decimate(mesh, 6000)
    V0, F0 = np.asarray(fit_mesh.vertices), np.asarray(fit_mesh.faces)
    W, H = meta["image_size"]
    shape = (H // DOWN, W // DOWN)
    data = []
    for v in views:
        m = np.array(Image.open(run / f"views/{v['id']}_mask.png")) > 127
        data.append((np.array(v["board_T_cam"]), cv2.resize(m.astype(np.uint8), shape[::-1], interpolation=cv2.INTER_AREA) > 0))

    def iou_all(p):
        V = apply(V0, p)
        ious = []
        for T, m in data:
            s = silhouette(V, F0, T, K, dist, shape)
            ious.append((s & m).sum() / max(1, (s | m).sum()))
        return np.array(ious), V

    P_hull = np.load(run / "locate/visual_hull_points_board.npy")
    hull = HullField(P_hull)
    Vs = np.asarray(trimesh.sample.sample_surface(mesh, 2500, seed=0)[0])

    def prior(p, V):  # resting on the table, nearly upright
        return 4.0 * (V[:, 2].min() / 0.01) ** 2 * 0.01 + 0.5 * (p[1] ** 2 + p[2] ** 2)

    def loss_sil(p):
        ious, V = iou_all(p)
        return (1 - ious.mean()) + prior(p, V)

    def loss_hull(p):
        X = apply(Vs, p)
        return hull.loss(X) / 0.01 + prior(p, X)  # 1.0 == 1 cm mean misfit

    def loss_both(p):
        return loss_sil(p) + 0.3 * hull.loss(apply(Vs, p)) / 0.01

    # --- initialisation from the visual hull box: scale from its extents, centre over its centre
    lo, hi = np.array(obj["lo"]), np.array(obj["hi"])
    nb = mesh.bounds
    ext_n = nb[1] - nb[0]
    ext_m = hi - lo
    s0 = float(np.median([ext_m[2] / ext_n[2], np.linalg.norm(ext_m[:2]) / np.linalg.norm(ext_n[:2])]))
    c_n = (nb[0] + nb[1]) / 2

    def placed(yaw, s):
        p = np.array([math.log(s), 0, 0, yaw, 0, 0, 0.0])
        _, R, _ = similarity(p)
        p[4:] = np.r_[(lo[:2] + hi[:2]) / 2, 0] - s * R @ c_n  # mesh centre over the hull centre
        p[6] -= apply(V0, p)[:, 2].min()  # resting on the table
        return p

    losses = {"hull": loss_hull, "hull+silhouette": loss_both, "silhouette": loss_sil}
    runs = {"hull": fit(loss_hull, s0, placed, "hull")}  # hull-only is always computed, for comparison
    if align != "hull":
        runs[align] = fit(losses[align], s0, placed, align)
    res = runs[align]
    p = res.x
    (out / "alignments").mkdir(exist_ok=True)
    fits = {}
    for name, r in runs.items():
        ii, _ = iou_all(r.x)
        sh, Rh, th = similarity(r.x)
        Sh = np.eye(4)
        Sh[:3, :3], Sh[:3, 3] = sh * Rh, th
        m = mesh.copy()
        m.apply_transform(Sh)
        m.export(out / f"alignments/{name.replace('+', '_')}_board_frame.glb")
        fits[name] = {"scale_native_to_m": sh, "silhouette_iou_mean": float(ii.mean()),
                      "hull_misfit_mm": float(hull.loss(apply(Vs, r.x)) * 1000),
                      "yaw_deg": float(np.degrees(r.x[3]) % 360), "t_m": th.tolist(), "loss": float(r.fun)}
    ious, _ = iou_all(p)
    s, R, t = similarity(p)
    S = np.eye(4)
    S[:3, :3], S[:3, 3] = s * R, t  # board_T_native (similarity)

    # --- exports
    full = mesh.copy()
    full.apply_transform(S)
    full.export(out / "object_board_frame.glb")
    full.export(out / "object_board_frame.stl")
    full.export(out / "object_board_frame.obj")
    base = full.copy()
    base.apply_transform(B)
    base.export(out / "object_robot_base_frame.glb")
    base.export(out / "object_robot_base_frame.stl")
    # object-centred CAD: origin at footprint centre on the table, axes = board axes
    lo_m, hi_m = full.bounds
    centred = full.copy()
    centred.apply_translation(-np.r_[(lo_m[:2] + hi_m[:2]) / 2, lo_m[2]])
    mm = centred.copy()
    mm.apply_scale(1000.0)
    mm.export(out / "object_cad_mm.stl")
    centred.export(out / "object_cad_m.glb")

    pose_board = np.eye(4)
    pose_board[:3, :3], pose_board[:3, 3] = R, t
    hull_info = export_hull(P_hull, B, out)
    result = {
        "align": align,
        "alignments_compared": fits,
        "visual_hull": hull_info,
        "silhouette_iou_per_view": dict(zip([v["id"] for v in views], ious.round(4).tolist())),
        "silhouette_iou_mean": float(ious.mean()),
        "scale_native_to_m": s,
        "pose_board": {"R": R.tolist(), "t_m": t.tolist(), "euler_zyx_deg": np.degrees([p[3], p[2], p[1]]).tolist()},
        "pose_robot_base": {"T": (B @ pose_board).tolist()},
        "board_T_native_similarity": S.tolist(),
        "dimensions_m_board_axes": (hi_m - lo_m).tolist(),
        "dimensions_m_oriented": (np.ptp(np.asarray(full.vertices) @ R, 0)).tolist(),
        "optimizer": {"success": bool(res.success), "nfev": int(res.nfev), "final_loss": float(res.fun)},
        "note": "Pose of TRELLIS' canonical frame in the AprilTag board frame; object-centred CAD has origin at footprint centre on table.",
        "files": sorted(p.name for p in out.iterdir()),
    }
    (out / "pose_and_scale.json").write_text(json.dumps(result, indent=1))
    print(f"[register] align={align}  " + "  ".join(f"{k}: s={v['scale_native_to_m']:.4f} IoU={v['silhouette_iou_mean']:.3f} hull={v['hull_misfit_mm']:.1f}mm" for k, v in fits.items()), flush=True)
    print(f"[register] mean IoU {ious.mean():.3f}  scale {s:.4f}  dims(mm) {np.round((hi_m - lo_m) * 1000, 1)}", flush=True)

    # --- overlays: fitted mesh silhouette (green) vs observed mask (magenta)
    small = decimate(full, 30000)
    Vf, Ff = np.asarray(small.vertices), np.asarray(small.faces)
    for v in views:
        rgb = np.array(Image.open(run / f"views/{v['id']}_rgb.png"))
        m = np.array(Image.open(run / f"views/{v['id']}_mask.png")) > 127
        sil = silhouette(Vf, Ff, np.array(v["board_T_cam"]), K, dist, m.shape, down=1)
        vis = rgb.copy()
        vis[sil] = (0.5 * vis[sil] + 0.5 * np.array([60, 220, 90])).astype(np.uint8)
        uvh, dh = vz.project(np.array(v["board_T_cam"]), K, dist, P_hull[:: max(1, len(P_hull) // 3000)])
        for u_, v_ in uvh[dh > 0].astype(int):
            if 0 <= u_ < vis.shape[1] and 0 <= v_ < vis.shape[0]:
                cv2.circle(vis, (u_, v_), 1, (0, 230, 255), -1)
        for mk, col in ((m, (255, 0, 255)), (sil, (0, 255, 0))):
            cnts, _ = cv2.findContours(mk.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(vis, cnts, -1, col, 3)
        # draw the object pose axes
        r = cv2.Rodrigues(vz.inv(np.array(v["board_T_cam"]))[:3, :3] @ R)[0]
        tt = (vz.inv(np.array(v["board_T_cam"])) @ np.r_[t, 1])[:3]
        cv2.drawFrameAxes(vis, K, dist, r, tt, 0.06, 4)
        Image.fromarray(vis).save(out / f"overlay_{v['id']}.jpg", quality=90)


if __name__ == "__main__":
    main()
