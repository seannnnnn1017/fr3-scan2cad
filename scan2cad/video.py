"""Build the full-process video from the recorder frames + brain annotations + final CAD.

Layout (1920x1080):  top bar = phase | left 1280x720 = lab camcorder | right = wrist live + latest
perception overlay | bottom strip = captured views + event log.  Then: TRELLIS inputs, metric
CAD turntable, silhouette-fit overlays, summary.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont

FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
W, H = 1920, 1080
BG = (24, 26, 30)
ACCENT = (255, 196, 64)


def font(size):
    try:
        return ImageFont.truetype(FONT, size, index=2)  # Noto Sans CJK TC
    except Exception:
        return ImageFont.load_default()


F_BIG, F_MID, F_SMALL = font(34), font(24), font(19)


def fit(img, w, h, bg=(0, 0, 0)):
    ih, iw = img.shape[:2]
    s = min(w / iw, h / ih)
    r = cv2.resize(img, (max(1, int(iw * s)), max(1, int(ih * s))), interpolation=cv2.INTER_AREA)
    out = np.full((h, w, 3), bg, np.uint8)
    y, x = (h - r.shape[0]) // 2, (w - r.shape[1]) // 2
    out[y:y + r.shape[0], x:x + r.shape[1]] = r
    return out


def text(canvas, xy, s, f=F_MID, fill=(235, 235, 235)):
    im = Image.fromarray(canvas)
    ImageDraw.Draw(im).text(xy, s, font=f, fill=fill)
    canvas[:] = np.asarray(im)


def label(img, s, f=F_SMALL):
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    b = d.textbbox((0, 0), s, font=f)
    d.rectangle((0, 0, b[2] + 16, b[3] + 10), fill=(0, 0, 0))
    d.text((8, 3), s, font=f, fill=ACCENT)
    return np.asarray(im)


class Writer:
    def __init__(self, path, fps):
        self.p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                                   "-s", f"{W}x{H}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "medium",
                                   "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)],
                                  stdin=subprocess.PIPE)
        self.n = 0

    def put(self, frame, repeat=1):
        b = np.ascontiguousarray(frame).tobytes()
        for _ in range(int(repeat)):
            self.p.stdin.write(b)
            self.n += 1

    def close(self):
        self.p.stdin.close()
        self.p.wait()


# ----------------------------------------------------------------------------- software turntable
def render_mesh(mesh, yaw, size=720, elev=25.0, dims=None):
    V = np.asarray(mesh.vertices, float)
    F = np.asarray(mesh.faces)
    c = (V.min(0) + V.max(0)) / 2
    V = V - c
    r = np.linalg.norm(V, axis=1).max()
    a, e = np.radians(yaw), np.radians(elev)
    Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, np.cos(e), -np.sin(e)], [0, np.sin(e), np.cos(e)]])
    P = V @ Rz.T @ Rx.T  # x right, z up, y depth
    f = size * 0.42 / r
    uv = np.c_[size / 2 + P[:, 0] * f, size * 0.55 - P[:, 2] * f]
    depth = P[F][:, :, 1].mean(1)
    n = np.cross(P[F[:, 1]] - P[F[:, 0]], P[F[:, 2]] - P[F[:, 0]])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    light = np.array([-0.4, -0.7, 0.6])
    light /= np.linalg.norm(light)
    shade = 0.35 + 0.65 * np.abs(n @ light)
    if mesh.visual.kind == "vertex" and len(mesh.visual.vertex_colors):
        col = np.asarray(mesh.visual.vertex_colors)[:, :3][F].mean(1)
    else:
        col = np.full((len(F), 3), 200.0)
    col = np.clip(col * shade[:, None], 0, 255).astype(np.uint8)
    img = np.full((size, size, 3), (38, 41, 47), np.uint8)
    # ground ellipse = table
    cv2.ellipse(img, (size // 2, int(size * 0.55 - (V[:, 2].min()) * f * np.cos(e))), (int(size * 0.36), int(size * 0.36 * np.sin(e))),
                0, 0, 360, (60, 64, 72), -1, cv2.LINE_AA)
    tri = np.round(uv[F] * 4).astype(np.int32)
    for i in np.argsort(-depth):  # painter's algorithm, far to near (y = depth into screen)
        cv2.fillConvexPoly(img, tri[i], col[i].tolist(), cv2.LINE_AA, 2)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    a = ap.parse_args()
    run = a.run.resolve()
    fps = 15
    frames = [json.loads(l) for l in (run / "rec/frames.jsonl").read_text().splitlines() if l.strip()]
    events = [json.loads(l) for l in (run / "rec/events.jsonl").read_text().splitlines() if l.strip()]
    wr = Writer(run / "process.mp4", fps)

    overlay = np.zeros((360, 640, 3), np.uint8)
    thumbs = {}
    log_lines = []
    ev_i = 0
    phase = ""
    src = json.loads((run / "views/views.json").read_text()).get("source", "") if (run / "views/views.json").exists() else ""
    title = ("FR3 三爪手臂 · AprilTag 定位 · 開放詞彙偵測 · visual hull 拍攝 = TRELLIS 輸入 · 對齊 hull → CAD"
             if src.startswith("visual hull") else "FR3 三爪手臂 · AprilTag 定位 · 開放詞彙偵測 · IK 環繞拍攝 · TRELLIS → CAD")

    # Persistent visual-hull overlay on the live wrist view, from the moment it is carved.  Uses only the
    # brain's own outputs: its calibration (K, dist, hand-eye, board pose), FK of the logged joint
    # encoders, and the carved voxels.
    hull = None
    hp = run / "locate/visual_hull_points_board.npy"
    cp = run / "calib/calibration.json"
    if hp.exists() and cp.exists():
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import kinematics as kin
        import vision as vz
        cal = json.loads(cp.read_text())
        P = np.load(hp)
        est = json.loads((run / "locate/object_estimate.json").read_text())
        lo, hi = np.array(est["lo"]), np.array(est["hi"])
        hull = {"P": P[:: max(1, len(P) // 3000)], "lo": lo, "hi": hi, "K": np.array(cal["K"]), "dist": np.array(cal["dist"]),
                "X": np.array(cal["link8_T_cam"]), "Binv": vz.inv(np.array(cal["base_T_board"])), "kin": kin, "vz": vz,
                "cal_w": cal.get("image_size", [1280, 720])[0]}
    hull_on = False

    def draw_hull(wrist, q):
        h = hull
        s = wrist.shape[1] / h["cal_w"]
        K = h["K"].copy()
        K[:2] *= s
        T = h["Binv"] @ h["kin"].fk_link8(np.asarray(q)) @ h["X"]
        vis = wrist.copy()
        uv, depth = h["vz"].project(T, K, h["dist"], h["P"])
        ok = (depth > 0.02) & np.isfinite(uv).all(1)
        layer = vis.copy()
        for u, v in uv[ok].astype(int):
            if 0 <= u < vis.shape[1] and 0 <= v < vis.shape[0]:
                cv2.circle(layer, (u, v), 1, (0, 255, 255), -1)
        vis = cv2.addWeighted(layer, 0.75, vis, 0.25, 0)
        lo, hi = h["lo"], h["hi"]
        C = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (0.0, hi[2])])
        uv, depth = h["vz"].project(T, K, h["dist"], C)
        if (depth > 0.02).all() and np.isfinite(uv).all() and (np.abs(uv) < 5000).all():
            uv = uv.astype(int)
            for a, b in [(0, 1), (0, 2), (1, 3), (2, 3), (4, 5), (4, 6), (5, 7), (6, 7), (0, 4), (1, 5), (2, 6), (3, 7)]:
                cv2.line(vis, tuple(uv[a]), tuple(uv[b]), (255, 60, 0), 2, cv2.LINE_AA)
        return vis

    def compose(fr, hold_note=None):
        canvas = np.full((H, W, 3), BG, np.uint8)
        ext = cv2.imread(str(run / f"rec/ext/{fr['frame']:06d}.jpg"))[..., ::-1]
        wrist = cv2.imread(str(run / f"rec/wrist/{fr['frame']:06d}.jpg"))[..., ::-1]
        if hull_on and fr.get("q") is not None:
            wrist = draw_hull(wrist, fr["q"])
        canvas[60:780, 0:1280] = fit(ext, 1280, 720)
        canvas[60:420, 1280:1920] = label(fit(wrist, 640, 360), "腕部相機 (即時) + 3D 外殼 (visual hull)" if hull_on else "腕部相機 (即時)")
        canvas[420:780, 1280:1920] = label(overlay.copy(), "感知結果")
        text(canvas, (20, 10), title, F_MID, (200, 200, 205))
        text(canvas, (1300, 10), phase, F_MID, ACCENT)
        for k in range(4):
            x = 10 + k * 330
            if k in thumbs:
                canvas[790:1070, x:x + 320] = fit(thumbs[k], 320, 280, (40, 40, 40))
            else:
                cv2.rectangle(canvas, (x, 790), (x + 320, 1070), (60, 60, 66), 2)
                text(canvas, (x + 110, 915), f"視角 {k + 1}", F_SMALL, (110, 110, 115))
        y = 792
        for line in log_lines[-6:]:
            text(canvas, (1340, y), line[:44], F_SMALL, (225, 225, 225))
            y += 46
        if hold_note:
            text(canvas, (20, 735), hold_note, F_SMALL, (150, 220, 255))
        text(canvas, (1150, 735), f"t = {fr['t']:.1f}s (sim)", F_SMALL, (170, 170, 170))
        return canvas

    for fr in frames:
        while ev_i < len(events) and events[ev_i]["frame"] <= fr["frame"]:
            ev = events[ev_i]
            ev_i += 1
            if ev.get("phase"):
                phase = ev["phase"]
            img = None
            if ev.get("image"):
                img = np.array(Image.open(run / ev["image"]).convert("RGB"))
            slot = ev.get("slot", "overlay")
            if slot.startswith("thumb") and img is not None:
                ys, xs = np.nonzero(img.max(2) > 8)  # masked photo: crop to the object (+ margin)
                if len(xs):
                    m = int(0.25 * max(np.ptp(xs), np.ptp(ys))) + 10
                    img = img[max(0, ys.min() - m):ys.max() + m, max(0, xs.min() - m):xs.max() + m]
                thumbs[int(slot[5:])] = img
                continue
            if img is not None:
                overlay = fit(img, 640, 360)
            if ev.get("text"):
                log_lines.append(ev["text"])
                if hull is not None and ev["text"].startswith("物件位置"):
                    hull_on = True
            if ev.get("hold_s", 0) > 0:
                wr.put(compose(fr, "（計算中：模擬時間暫停）"), repeat=int(ev["hold_s"] * fps))
        wr.put(compose(fr))

    # ---------------------------------------------------------------- results
    def card(title_s, body=None):
        c = np.full((H, W, 3), BG, np.uint8)
        text(c, (40, 24), title_s, F_BIG, ACCENT)
        if body is not None:
            c[100:100 + body.shape[0], (W - body.shape[1]) // 2:(W - body.shape[1]) // 2 + body.shape[1]] = body
        return c

    tdir = run / "trellis"
    if (tdir / "result.json").exists():
        ins = [np.array(Image.open(p).convert("RGBA")) for p in sorted(tdir.glob("*_model_input.png"))]
        tiles = []
        for im in ins:
            rgb = im[..., :3].astype(float)
            al = im[..., 3:4].astype(float) / 255
            checker = ((np.indices(im.shape[:2]).sum(0) // 24) % 2 * 40 + 200)[..., None]
            tiles.append(fit((rgb * al + checker * (1 - al)).astype(np.uint8), 440, 440))
        if tiles:
            body = np.concatenate(tiles, 1)
            src = json.loads((run / "views/views.json").read_text()).get("source", "")
            what = f"visual hull 的 {len(tiles)} 張照片直接沿用" if src.startswith("visual hull") else f"遮罩去背後的 {len(tiles)} 張腕部照片"
            c = card(f"5  TRELLIS v1 多圖輸入 ({what}；模型不接收相機姿態)", body)
            res = json.loads((tdir / "result.json").read_text())
            text(c, (40, 600), f"推論 {res['inference_seconds']:.1f}s  ·  {res['triangles']:,} 三角面  ·  seed {res['parameters']['seed']}", F_MID)
            wr.put(c, fps * 3)
    cad = run / "cad"
    if (cad / "pose_and_scale.json").exists():
        pose = json.loads((cad / "pose_and_scale.json").read_text())
        ovs = [np.array(Image.open(p)) for p in sorted(cad.glob("overlay_*.jpg"))]
        grid = np.zeros((720, 1280, 3), np.uint8)
        for k, o in enumerate(ovs[:4]):
            grid[(k // 2) * 360:(k // 2) * 360 + 360, (k % 2) * 640:(k % 2) * 640 + 640] = fit(o, 640, 360)
        how = {"hull": "對齊 visual hull (青點)", "hull+silhouette": "對齊 visual hull (青點) 與其遮罩 (洋紅)",
               "silhouette": "對齊遮罩 (洋紅)"}.get(pose.get("align", "silhouette"))
        c = card(f"6  公尺尺度 + 6D 位姿：TRELLIS 網格 (綠) {how}", grid)
        hm = pose.get("alignments_compared", {}).get(pose.get("align"), {}).get("hull_misfit_mm")
        text(c, (40, 840), f"平均輪廓 IoU {pose['silhouette_iou_mean']:.3f}   ·   尺度 {pose['scale_native_to_m']:.4f} m/unit"
                           + (f"   ·   與 visual hull 平均偏差 {hm:.1f} mm" if hm is not None else ""), F_MID)
        t = pose["pose_board"]["t_m"]
        text(c, (40, 890), f"物件位姿 (AprilTag 桌面座標)：t = ({t[0] * 100:.1f}, {t[1] * 100:.1f}, {t[2] * 100:.1f}) cm   "
                           f"yaw = {pose['pose_board']['euler_zyx_deg'][0]:.1f}°", F_MID)
        wr.put(c, fps * 4)
        mesh = trimesh.load(cad / "object_cad_m.glb", force="mesh", process=False)
        if len(mesh.faces) > 20000:
            mesh = mesh.simplify_quadric_decimation(face_count=20000)
        d = np.array(pose["dimensions_m_board_axes"]) * 1000
        for k in range(fps * 6):
            img = render_mesh(mesh, yaw=k * 360 / (fps * 6))
            c = card("7  輸出 CAD (公尺尺度，原點 = 桌面上物件底部中心)", fit(img, 860, 860, (38, 41, 47)))
            text(c, (40, 300), f"寬 {d[0]:.1f} mm", F_MID)
            text(c, (40, 345), f"深 {d[1]:.1f} mm", F_MID)
            text(c, (40, 390), f"高 {d[2]:.1f} mm", F_MID)
            text(c, (40, 470), "檔案: object_cad_mm.stl", F_SMALL)
            text(c, (40, 500), "      object_cad_m.glb", F_SMALL)
            text(c, (40, 530), "      object_robot_base_frame.glb", F_SMALL)
            if (cad / "visual_hull.json").exists():
                vh = json.loads((cad / "visual_hull.json").read_text())["robot_base_frame"]["center_m"]
                text(c, (40, 590), "visual hull (基座座標) 中心", F_SMALL)
                text(c, (40, 620), f"  ({vh[0] * 100:.1f}, {vh[1] * 100:.1f}, {vh[2] * 100:.1f}) cm", F_SMALL)
                text(c, (40, 650), "      visual_hull.json / .ply", F_SMALL)
            wr.put(c)
    wr.close()
    print(f"[video] {wr.n} frames -> {run / 'process.mp4'}")


if __name__ == "__main__":
    main()
