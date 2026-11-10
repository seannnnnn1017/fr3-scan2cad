"""Post-hoc verification clip: the registered CAD overlaid on the simulator ground truth.

Runs after eval/evaluate.py (needs eval/gt_mesh_board_frame.glb + eval/evaluation.json).  Writes
eval/verify.mp4 and process_with_eval.mp4 (= process.mp4 + this clip).  Ground truth is only drawn
here, after every brain output has been frozen by evaluate.py.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import trimesh
from PIL import Image
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scan2cad"))
import vision as vz  # noqa: E402
from register import silhouette  # noqa: E402
from video import BG, ACCENT, F_BIG, F_MID, F_SMALL, W, H, Writer, fit, text  # noqa: E402

GREEN, RED = np.array([70, 220, 100.0]), np.array([235, 70, 70.0])
ERR_MAX_MM = 10.0


def small(m, n=15000):
    return m.simplify_quadric_decimation(face_count=n) if len(m.faces) > n else m


def render(V, F, fcol, center, r, yaw, elev=22.0, size=720):
    """Painter's-algorithm render with a shared camera (so two meshes line up).  Returns img, mask."""
    a, e = np.radians(yaw), np.radians(elev)
    Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, np.cos(e), -np.sin(e)], [0, np.sin(e), np.cos(e)]])
    P = (V - center) @ Rz.T @ Rx.T
    f = size * 0.42 / r
    uv = np.c_[size / 2 + P[:, 0] * f, size / 2 - P[:, 2] * f]
    n = np.cross(P[F[:, 1]] - P[F[:, 0]], P[F[:, 2]] - P[F[:, 0]])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    light = np.array([-0.4, -0.7, 0.6]) / np.linalg.norm([-0.4, -0.7, 0.6])
    col = np.clip(fcol * (0.35 + 0.65 * np.abs(n @ light))[:, None], 0, 255).astype(np.uint8)
    img = np.zeros((size, size, 3), np.uint8)
    mask = np.zeros((size, size), np.uint8)
    tri = np.round(uv[F] * 4).astype(np.int32)
    for i in np.argsort(-P[F][:, :, 1].mean(1)):
        cv2.fillConvexPoly(img, tri[i], col[i].tolist(), cv2.LINE_AA, 2)
        cv2.fillConvexPoly(mask, tri[i], 1, cv2.LINE_8, 2)
    return img, mask.astype(bool)


def colorbar(h=360, w=26):
    v = np.linspace(255, 0, h).astype(np.uint8)[:, None].repeat(w, 1)
    return cv2.applyColorMap(v, cv2.COLORMAP_TURBO)[..., ::-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    a = ap.parse_args()
    run = a.run.resolve()
    ev = json.loads((run / "eval/evaluation.json").read_text())
    cad = trimesh.load(run / "cad/object_board_frame.glb", force="mesh", process=False)
    gt = trimesh.load(run / "eval/gt_mesh_board_frame.glb", force="mesh", process=False)
    pose = json.loads((run / "cad/pose_and_scale.json").read_text())

    # per-face error of the CAD against the GT surface (mm)
    gt_pts, _ = trimesh.sample.sample_surface(gt, 200000, seed=0)
    cs, gs = small(cad), small(gt)
    Vc, Fc, Vg, Fg = (np.asarray(x, float if i % 2 == 0 else int) for i, x in enumerate((cs.vertices, cs.faces, gs.vertices, gs.faces)))
    err = cKDTree(gt_pts).query(Vc[Fc].mean(1))[0] * 1000
    heat = cv2.applyColorMap(np.clip(err / ERR_MAX_MM * 255, 0, 255).astype(np.uint8)[:, None], cv2.COLORMAP_TURBO)[:, 0, ::-1].astype(float)

    lo = np.minimum(Vc.min(0), Vg.min(0))
    hi = np.maximum(Vc.max(0), Vg.max(0))
    center, r = (lo + hi) / 2, np.linalg.norm(hi - lo) / 2

    fps = 15
    out = run / "eval/verify.mp4"
    wr = Writer(out, fps)
    c3 = ev["cad_vs_gt"]
    de, dg = c3["bbox_dims_mm_est"], c3["bbox_dims_mm_gt"]

    def footer(c):
        text(c, (40, 960), f"Chamfer {c3['chamfer_mean_mm']:.2f} mm   ·   F@5mm {c3['fscore_5mm']:.3f}   ·   "
                           f"中心誤差 {c3['centroid_err_mm']:.1f} mm   ·   CAD 誤差中位數 {np.median(err):.1f} mm", F_MID)
        text(c, (40, 1005), f"尺寸 估計 {de[0]:.0f}×{de[1]:.0f}×{de[2]:.0f} mm   /   真值 {dg[0]:.0f}×{dg[1]:.0f}×{dg[2]:.0f} mm"
                            f"   （對齊方式：{pose.get('align', 'silhouette')}）", F_MID, (200, 200, 205))

    # --- 8a: turntable, CAD vs GT overlapped in the same (AprilTag board) frame
    n = fps * 8
    for k in range(n):
        yaw = k * 360 / n
        ic, mc = render(Vc, Fc, np.tile(GREEN, (len(Fc), 1)), center, r, yaw)
        ig, mg = render(Vg, Fg, np.tile(RED, (len(Fg), 1)), center, r, yaw)
        ov = np.full_like(ic, (38, 41, 47))
        ov[mg] = ig[mg]
        ov[mc] = ic[mc]
        both = mc & mg
        ov[both] = (0.5 * ic[both] + 0.5 * ig[both]).astype(np.uint8)  # overlap -> yellowish
        ih, mh = render(Vc, Fc, heat, center, r, yaw)
        hm = np.full_like(ih, (38, 41, 47))
        hm[mh] = ih[mh]
        cnt, _ = cv2.findContours(mg.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(hm, cnt, -1, (255, 255, 255), 2, cv2.LINE_AA)
        c = np.full((H, W, 3), BG, np.uint8)
        text(c, (40, 24), "8  驗證：輸出 CAD 與模擬真值在同一座標重疊 (僅流程結束後)", F_BIG, ACCENT)
        c[150:870, 60:780] = fit(ov, 720, 720)
        c[150:870, 900:1620] = fit(hm, 720, 720)
        text(c, (60, 105), "綠 = CAD   紅 = 真值   黃 = 重疊", F_MID)
        text(c, (900, 105), "CAD 表面到真值的距離 (白線 = 真值輪廓)", F_MID)
        c[300:660, 1660:1686] = colorbar()
        text(c, (1700, 288), f"≥{ERR_MAX_MM:.0f} mm", F_SMALL)
        text(c, (1700, 470), f"{ERR_MAX_MM / 2:.0f} mm", F_SMALL)
        text(c, (1700, 645), "0 mm", F_SMALL)
        footer(c)
        wr.put(c)

    # --- 8b: GT vs CAD projected into the wrist photos (camera poses = brain's own calibration)
    meta = json.loads((run / "views/views.json").read_text())
    K, dist = np.array(meta["K"]), np.array(meta["dist"])
    tiles = []
    for v in meta["views"][:4]:
        rgb = np.array(Image.open(run / f"views/{v['id']}_rgb.png").convert("RGB"))
        T = np.array(v["board_T_cam"])
        sc = silhouette(Vc, Fc, T, K, dist, rgb.shape[:2], down=1)
        sg = silhouette(Vg, Fg, T, K, dist, rgb.shape[:2], down=1)
        vis = rgb.copy()
        vis[sc] = (0.6 * vis[sc] + 0.4 * GREEN).astype(np.uint8)
        for m, col in ((sg, (255, 50, 50)), (sc, (60, 255, 90))):
            cnt, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(vis, cnt, -1, col, 3, cv2.LINE_AA)
        ys, xs = np.nonzero(sc | sg)
        mg_ = int(0.6 * max(np.ptp(xs), np.ptp(ys))) + 20
        crop = vis[max(0, ys.min() - mg_):ys.max() + mg_, max(0, xs.min() - mg_):xs.max() + mg_]
        iou = (sc & sg).sum() / max(1, (sc | sg).sum())
        tiles.append((fit(crop, 430, 430), v["id"], iou))
    c = np.full((H, W, 3), BG, np.uint8)
    text(c, (40, 24), "8  驗證：把真值投影回腕部照片 (紅 = 真值輪廓，綠 = CAD)", F_BIG, ACCENT)
    for i, (t, name, iou) in enumerate(tiles):
        x = 40 + i * 465
        c[180:610, x:x + 430] = t
        text(c, (x, 625), f"{name}   CAD/真值 輪廓 IoU {iou:.3f}", F_SMALL)
    text(c, (40, 700), "相機位姿用的是 brain 自己的標定 (AprilTag + 手眼)，所以這裡的偏差 = 標定誤差 + CAD 誤差。", F_SMALL, (200, 200, 205))
    vh = ev.get("visual_hull_vs_gt")
    if vh:
        text(c, (40, 740), f"visual hull 中心水平誤差 {vh['center_xy_err_mm']:.1f} mm   ·   hull 尺寸 "
                           f"{vh['bbox_dims_mm_hull'][0]:.0f}×{vh['bbox_dims_mm_hull'][1]:.0f}×{vh['bbox_dims_mm_hull'][2]:.0f} mm", F_SMALL)
    footer(c)
    wr.put(c, fps * 5)
    wr.close()

    # process.mp4 + verify.mp4 (same encoder settings -> stream copy)
    if (run / "process.mp4").exists():
        lst = run / "eval/concat.txt"
        lst.write_text(f"file '{run / 'process.mp4'}'\nfile '{out}'\n")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy",
                        str(run / "process_with_eval.mp4")], check=True)
        lst.unlink()
    print(f"[verify] {wr.n} frames -> {out}; {run / 'process_with_eval.mp4'}")


if __name__ == "__main__":
    main()
