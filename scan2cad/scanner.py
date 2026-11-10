"""The scanning "brain": everything a real deployment would run on the robot PC.

Inputs:  wrist RGB frames + joint encoders (from the driver), the printed tag-board layout,
         nominal FR3 kinematics, a rough CAD guess of the camera bracket, a text prompt.
Outputs: calibration, object 3-D box, four object photos (RGBA) with calibrated camera poses.

It never opens simulator files; an audit hook enforces that (see _guard).
"""
from __future__ import annotations

import io
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from common import protocol  # noqa: E402
import kinematics as kin  # noqa: E402
import vision as vz  # noqa: E402

FORBIDDEN = ("sim_private", "configs/scenes", "scene_", "/rec/")


def _guard(event, args):
    """Audit hook: the brain may not read simulator ground truth or the world description."""
    if event == "open" and args and isinstance(args[0], (str, bytes, Path)):
        p = str(args[0])
        mode = args[1] if len(args) > 1 else "r"
        if any(f in p for f in FORBIDDEN) and (mode is None or "r" in str(mode) and "+" not in str(mode) and "w" not in str(mode)):
            raise PermissionError(f"brain tried to read simulator-only file: {p}")


class Robot:
    """Client for the driver.  Same calls would wrap franky/libfranka + a UVC camera on hardware."""

    def __init__(self, port):
        for _ in range(600):
            try:
                self.conn = protocol.connect(port)
                break
            except (ConnectionRefusedError, OSError):
                time.sleep(1)
        else:
            raise RuntimeError("driver not reachable")
        self.info = self.call("hello")[0]

    def call(self, cmd, blob=None, **kw):
        protocol.send(self.conn, {"cmd": cmd, **kw}, blob)
        h, b = protocol.recv(self.conn)
        if not h.get("ok"):
            raise RuntimeError(f"driver error on {cmd}: {h.get('error')}")
        return h, b

    def q(self):
        return np.array(self.call("get_joints")[0]["q"])

    def move(self, waypoints, durations, phase=None):
        h, _ = self.call("move", waypoints=[np.asarray(w).tolist() for w in waypoints],
                         durations=list(map(float, durations)), phase=phase)
        return np.array(h["q"]), h

    def capture(self):
        h, png = self.call("capture")
        rgb = np.array(Image.open(io.BytesIO(png)).convert("RGB"))
        return rgb, np.array(h["q"])

    def annotate(self, text, image=None, phase=None, hold_s=0.0, slot="overlay"):
        blob = None
        if image is not None:
            buf = io.BytesIO()
            Image.fromarray(image).save(buf, format="JPEG", quality=88)
            blob = buf.getvalue()
        self.call("annotate", blob, text=text, phase=phase, hold_s=hold_s, slot=slot)

    def idle(self, seconds):
        self.call("idle", seconds=seconds)

    def shutdown(self):
        return self.call("shutdown")[0]


def rpy_T(xyz, rpy_deg):
    return vz.T_from(Rotation.from_euler("xyz", rpy_deg, degrees=True).as_matrix(), xyz)


class Scanner:
    def __init__(self, cfg, out: Path, robot: Robot, segmenter):
        self.cfg, self.out, self.robot, self.seg = cfg, out, robot, segmenter
        self.board = cfg["board"]
        W, H = robot.info["camera_resolution"]
        self.size = (W, H)
        f = (W / 2) / math.tan(math.radians(cfg["camera_prior"]["hfov_deg"]) / 2)
        self.K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])  # datasheet guess until calibrated
        self.dist = np.zeros(5)
        self.X = rpy_T(cfg["camera_prior"]["mount_xyz_m"], cfg["camera_prior"]["mount_rpy_deg"])  # link8_T_cam guess
        self.B = None  # base_T_board
        lim = np.array(robot.info["arm_limits"])
        self.limits = np.c_[np.maximum(lim[:, 0], kin.LIMITS[:, 0]), np.minimum(lim[:, 1], kin.LIMITS[:, 1])] + [[0.02, -0.02]]
        self.home = np.array(cfg["home_q"])
        self.log = {"config": cfg, "steps": []}
        self.obj = None
        self.safe_q = self.home
        self.robot_q_cache = robot.q()
        self.nadir_hint = np.array([-1.0, 0, 0])
        self.rng = np.random.default_rng(cfg.get("seed", 0))
        for d in ("calib", "locate", "views", "inputs"):
            (out / d).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ utilities
    def note(self, **kw):
        kw["t_wall"] = time.time()
        self.log["steps"].append(kw)
        (self.out / "scan_log.json").write_text(json.dumps(self.log, indent=1, default=_json))
        print("[brain]", {k: v for k, v in kw.items() if k != "t_wall"}, flush=True)

    def board_T_cam(self, q):
        return vz.inv(self.B) @ kin.fk_link8(q) @ self.X

    def tool_points(self):
        pts = [(0, 0, z) for z in (0.04, 0.09, 0.14, 0.18)] + [(sx * 0.045, sy * 0.045, 0.13) for sx in (-1, 1) for sy in (-1, 1)]
        pts.append(tuple(self.X[:3, 3]))
        return pts

    def clearance_ok(self, q, strict=True):
        """Crude collision check in the board frame: table plane + object keep-out cylinder."""
        if self.B is None:
            return True
        P = kin.link_points(q, self.tool_points())[2:]  # skip base links
        P = (vz.inv(self.B) @ np.c_[P, np.ones(len(P))].T).T[:, :3]
        if (P[:, 2] < 0.035).any():
            return False
        if self.obj is not None:
            c, r, top = self.obj["center"], self.obj["radius_xy"] + 0.045, self.obj["top"] + 0.04
            d = np.linalg.norm(P[:, :2] - c[:2], axis=1)
            if ((d < r) & (P[:, 2] < top)).any():
                return False
        return True

    def path_ok(self, q0, q1, n=24):
        return all(self.clearance_ok(q0 + (q1 - q0) * s) for s in np.linspace(0, 1, n))

    def duration(self, q0, q1, speed=0.55):
        return max(1.6, float(np.abs(np.asarray(q1) - np.asarray(q0)).max() / speed) + 0.6)

    def go(self, q_target, phase=None, via=None):
        q_now = self.robot.q()
        legs = [q_target] if via is None else [via, q_target]
        seq, prev = [], q_now
        for leg in legs:
            seq.append(leg)
            prev = leg
        if via is None and not self.path_ok(q_now, q_target):
            safe = self.safe_q
            if self.path_ok(q_now, safe) and self.path_ok(safe, q_target):
                seq = [safe, q_target]
            else:
                self.note(event="warning", msg="no collision-free joint path found by the crude checker; moving anyway via safe pose")
                seq = [safe, q_target]
        durs, prev = [], q_now
        for s in seq:
            durs.append(self.duration(prev, s))
            prev = s
        q, h = self.robot.move(seq, durs, phase=phase)
        return q

    def solve_cam(self, board_T_cam_target, seeds=None):
        """IK for a desired camera pose (board frame).  Returns q or None."""
        target = self.B @ board_T_cam_target @ vz.inv(self.X)
        seeds = seeds or [self.robot_q_cache, self.home]
        return kin.ik_multi(target, seeds, limits=self.limits, rng=self.rng)

    def cam_pose_ik(self, eye, look, rolls=(0, 30, -30, 60, -60, 90, -90, 135, -135, 180), check=True):
        best = None
        for roll in rolls:
            T = vz.look_at_cam(eye, look, roll_deg=roll, nadir_hint=self.nadir_hint)
            q = self.solve_cam(T)
            if q is None or (check and not self.clearance_ok(q)):
                continue
            cost = np.linalg.norm((q - self.robot_q_cache)[:6]) + abs(roll) / 90
            if best is None or cost < best[0]:
                best = (cost, q, T, roll)
            if best is not None and abs(best[3]) <= 30:
                break
        return (None, None, None) if best is None else best[1:]

    # ------------------------------------------------------------------ 1. calibration
    def calibrate(self):
        R = self.robot
        R.annotate("1 相機內參 + 手眼校正 (AprilTag)  |  Camera intrinsics + hand-eye calibration", phase="1 校正 Calibration")
        q = self.go(self.home, phase="1 校正 Calibration")
        self.robot_q_cache = q
        rgb, q = R.capture()
        dets = vz.detect_tags(rgb, self.board)
        T_cb, rmse = vz.pnp(self.board, self.K, self.dist, dets, min_tags=1)
        if T_cb is None:
            raise RuntimeError("No AprilTag visible from home pose; place the board in front of the robot")
        self.B = kin.fk_link8(q) @ self.X @ T_cb  # rough board pose from datasheet K + CAD mount
        self.nadir_hint = vz.inv(self.B)[:3, :3] @ np.array([-1.0, 0, 0])  # image-down = toward robot, in board frame
        self.note(event="rough_board", tags=[d["id"] for d in dets], base_T_board=self.B)
        R.annotate("粗略定位桌面 (規格書內參 + CAD 支架)", vz.draw_tags(rgb, dets, self.K, self.dist, T_cb), hold_s=1.0)

        c = self.cfg["calibration"]
        poses, qs = [], []
        n = c["n_views"]
        for i in range(n * 3):
            if len(poses) >= n:
                break
            tilt = math.radians(self.rng.uniform(*c["tilt_deg"]))
            az = 2 * math.pi * (i * 0.618034 % 1.0)
            d = self.rng.uniform(*c["distance_m"])
            eye = d * np.array([math.sin(tilt) * math.cos(az), math.sin(tilt) * math.sin(az), math.cos(tilt)])
            look = np.r_[self.rng.uniform(-1, 1, 2) * c.get("look_offset_m", 0.04), 0]
            roll = self.rng.uniform(*c["roll_deg"])
            T = vz.look_at_cam(eye, look, roll_deg=roll, nadir_hint=self.nadir_hint)
            qq = self.solve_cam(T)
            if qq is None or not self.clearance_ok(qq):
                continue
            poses.append(T)
            qs.append(qq)
        # visit in a short tour
        order, left, cur = [], list(range(len(qs))), q
        while left:
            j = min(left, key=lambda k: np.abs(qs[k] - cur).max())
            order.append(j)
            left.remove(j)
            cur = qs[j]
        Gs, dets_all, rgbs = [], [], []
        for k, j in enumerate(order):
            qm = self.go(qs[j], phase="1 校正 Calibration")
            self.robot_q_cache = qm
            rgb, qm = R.capture()
            dets = vz.detect_tags(rgb, self.board)
            Gs.append(kin.fk_link8(qm))
            dets_all.append(dets)
            rgbs.append(rgb)
            cv2.imwrite(str(self.out / f"calib/view_{k:02d}.jpg"), rgb[..., ::-1])
            R.annotate(f"校正影像 {k + 1}/{len(order)}：偵測到 tag {[d['id'] for d in dets]}", vz.draw_tags(rgb, dets))
        K, dist, rms, nv = vz.calibrate_intrinsics(self.board, dets_all, self.size, self.K)
        self.K, self.dist = K, dist
        usable = [(G, d) for G, d in zip(Gs, dets_all) if len(d) >= 2]
        cams = [vz.pnp(self.board, K, dist, d)[0] for _, d in usable]
        X0, B0 = vz.hand_eye([G for G, _ in usable], cams)
        X, B, ba_rmse = vz.refine_hand_eye(self.board, K, dist, [G for G, _ in usable], [d for _, d in usable], X0, B0)
        spread = [np.linalg.norm((G @ X @ C)[:3, 3] - B[:3, 3]) * 1000 for (G, _), C in zip(usable, cams)]
        self.X, self.B = X, B
        self.nadir_hint = vz.inv(self.B)[:3, :3] @ np.array([-1.0, 0, 0])
        calib = {"K": K, "dist": dist, "intrinsic_rms_px": rms, "views_used": nv, "link8_T_cam": X, "base_T_board": B,
                 "handeye_park": X0, "bundle_rmse_px": ba_rmse, "per_view_board_spread_mm": spread,
                 "image_size": self.size}
        (self.out / "calib/calibration.json").write_text(json.dumps(calib, indent=1, default=_json))
        self.note(event="calibrated", intrinsic_rms_px=rms, fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2],
                  bundle_rmse_px=ba_rmse, board_spread_mm_median=float(np.median(spread)))
        # verification overlay: re-project the board using FK + calibration only
        vis = rgbs[-1].copy()
        cam_T_board = vz.inv(Gs[-1] @ X) @ B
        for tid in self.board["centers_m"]:
            uv, _ = vz.project(vz.inv(cam_T_board), K, dist, vz.tag_corners_board(self.board, tid))
            cv2.polylines(vis, [uv.astype(int).reshape(-1, 1, 2)], True, (255, 0, 0), 3)
        vis = vz.draw_tags(vis, dets_all[order.index(order[-1])] if False else dets_all[-1], K, dist, cam_T_board)
        R.annotate(f"校正完成  fx={K[0, 0]:.1f}  重投影 {ba_rmse:.2f}px  (藍框 = 由關節角+手眼推算)", vis, hold_s=2.0)

    def load_calibration(self, path):
        c = json.loads(Path(path).read_text())
        self.K, self.dist = np.array(c["K"]), np.array(c["dist"])
        self.X, self.B = np.array(c["link8_T_cam"]), np.array(c["base_T_board"])
        self.nadir_hint = vz.inv(self.B)[:3, :3] @ np.array([-1.0, 0, 0])
        self.robot_q_cache = self.robot.q()
        (self.out / "calib/calibration.json").write_text(Path(path).read_text())
        self.note(event="calibration_loaded", source=str(path))

    # ------------------------------------------------------------------ 2-3. detect + localise
    def look_and_segment(self, q, label, expect_xyz=None):
        rgb, q = self.robot.capture()
        T = self.board_T_cam(q)
        dets = vz.detect_tags(rgb, self.board)
        T_pnp, rmse = vz.pnp(self.board, self.K, self.dist, dets)
        if T_pnp is not None:  # local refinement: tags in this very image
            T_pnp = vz.inv(T_pnp)
            dev = (np.linalg.norm(T_pnp[:3, 3] - T[:3, 3]) * 1000, vz.rot_deg(T_pnp[:3, :3], T[:3, :3]))
        else:
            dev = None
        expect = None
        if expect_xyz is not None:
            expect = vz.project(T, self.K, self.dist, expect_xyz)[0][0]
        res = self.seg.segment(rgb, expect_uv=expect)
        if res is not None:
            res["mask"], filled = vz.fill_object_holes(rgb, res["mask"])
            if filled:
                self.note(event="mask_holes_filled", label=label, pixels=filled)
        cv2.imwrite(str(self.out / f"locate/{label}.jpg"), rgb[..., ::-1])
        self.note(event="look", label=label, q=q, board_T_cam=T, tags=[d["id"] for d in dets], pnp_vs_fk=dev,
                  score=None if res is None else res["score"])
        if res is not None:
            cv2.imwrite(str(self.out / f"locate/{label}_mask.png"), res["mask"].astype(np.uint8) * 255)
        return rgb, q, T, res, dets, dev

    def locate(self):
        R = self.robot
        phase = "2 物件偵測 Detect"
        prompt = self.cfg["prompt"]
        R.annotate(f"2 開放詞彙偵測：\"{prompt}\"  ({self.seg.name})", phase=phase)
        # survey: high nadir views over the board (and around it, if the object is not found in the
        # first one or is cut by the image border)
        h = self.cfg["locate"]["survey_height_m"]
        best, out = None, None
        # each point is [x, y] (board frame, at survey_height_m) or [x, y, height]; e.g. one high
        # overview first, then closer views only if the object is not found cleanly
        for i, pt in enumerate(self.cfg["locate"].get("survey_points_m", [[0, 0]])):
            sx, sy = pt[0], pt[1]
            eye = np.array([sx, sy, pt[2] if len(pt) > 2 else h])
            qs, _, _ = self.cam_pose_ik(eye, eye * [1, 1, 0])
            if qs is None:
                self.note(event="survey_unreachable", point=[sx, sy])
                continue
            if i == 0:
                self.safe_q = qs
            qs = self.go(qs, phase=phase)
            self.robot_q_cache = qs
            out = self.look_and_segment(qs, "survey" if i == 0 else f"survey_{i}")
            r = out[3]
            if r is None:
                R.annotate(f"俯視 {i + 1}：沒有 \"{prompt}\"", out[0])
                continue
            ys, xs = np.nonzero(r["mask"])
            edge = xs.min() <= 2 or ys.min() <= 2 or xs.max() >= self.size[0] - 3 or ys.max() >= self.size[1] - 3
            rank = r["score"] - (0.5 if edge else 0.0)
            R.annotate(f"俯視 {i + 1}：{prompt} score={r['score']:.2f}" + ("（被畫面邊緣切到）" if edge else ""),
                       _draw_det(out[0], r, prompt))
            if best is None or rank > best[0]:
                best = (rank, out)
            if not edge and r["score"] >= 0.5:
                break
        if best is None:
            if out is not None:
                R.annotate(f"找不到 \"{prompt}\"", out[0], hold_s=2)
            raise RuntimeError(f"object '{prompt}' not found in survey views")
        rgb, q, T, res, dets, dev = best[1]
        R.annotate(f"偵測到 {prompt}  score={res['score']:.2f}  [{res['backend']}]",
                   _draw_det(rgb, res, prompt), hold_s=1.5)
        m = res["mask"]
        ys, xs = np.nonzero(m)
        p0 = vz.ray_plane(T, self.K, self.dist, [[xs.mean(), ys.mean()]])[0]
        self.note(event="survey_detection", score=res["score"], centroid_ray_on_table=p0, pnp_vs_fk=dev)

        phase = "3 3D 定位 Localize"
        R.annotate("3 移到物件正上方 + 斜視，以多視角輪廓雕刻 (visual hull) 求 3D 位置/大小", phase=phase)
        L = self.cfg["locate"]
        as_inputs = L.get("hull_views_as_trellis_inputs", False)
        views = [(T, m)]
        self.hull_shots = []  # nadir + oblique photos; with hull_views_as_trellis_inputs they are also TRELLIS' inputs

        def shot(rgb, q, T, res, dets, dev, label, info):
            views.append((T, res["mask"]))
            self.hull_shots.append({"label": label, "rgb": rgb, "q": q, "T": T, "res": res, "dets": dets, "dev": dev, "plan": info})
            k = len(self.hull_shots) - 1
            if as_inputs and k < 4:
                R.annotate(f"{'正上方' if label == 'nadir' else '斜視'}視角 → visual hull + TRELLIS 輸入 {k + 1}",
                           _draw_det(rgb, res, prompt), slot=f"view{k}")
                R.annotate("", rgb * res["mask"][..., None], slot=f"thumb{k}")
            else:
                R.annotate(f"{'正上方' if label == 'nadir' else '斜視'}視角", _draw_det(rgb, res, prompt))

        # nadir over the first estimate (far objects: lower, or tilted slightly from the robot side)
        q = None
        base_dir = vz.inv(self.B)[:3, 3][:2] - p0[:2]
        base_dir = base_dir / np.linalg.norm(base_dir)
        for hgt in [L["nadir_height_m"]] + list(L.get("nadir_fallback_heights_m", [])):
            for tilt in [0] + list(L.get("nadir_fallback_tilts_deg", [])):
                eye = np.r_[p0[:2] + base_dir * hgt * math.tan(math.radians(tilt)), hgt]
                q, _, _ = self.cam_pose_ik(eye, np.r_[p0[:2], 0.0])
                if q is not None:
                    break
            if q is not None:
                break
        if q is None:
            self.note(event="nadir_unreachable")
        else:
            q = self.go(q, phase=phase)
            self.robot_q_cache = q
            rgb, q, T, res, dets, dev = self.look_and_segment(q, "nadir", np.r_[p0[:2], 0.03])
            if res is not None:
                ys, xs = np.nonzero(res["mask"])
                p0 = vz.ray_plane(T, self.K, self.dist, [[xs.mean(), ys.mean()]])[0]
                shot(rgb, q, T, res, dets, dev, "nadir", {"height_m": L["nadir_height_m"]})
        # oblique views around the object.  Each requested azimuth is tried at several elevations /
        # distances / small azimuth shifts until IK reaches it and a prior object box stays in frame.
        base_dir = vz.inv(self.B)[:3, 3][:2] - p0[:2]
        phi0 = math.degrees(math.atan2(base_dir[1], base_dir[0]))
        look = np.r_[p0[:2], 0.04]
        bx = np.array(L.get("object_prior_box_m", [0.14, 0.14, 0.16]))
        corners = np.array([[p0[0] + sx * bx[0] / 2, p0[1] + sy * bx[1] / 2, z] for sx in (-1, 1) for sy in (-1, 1) for z in (0, bx[2])])
        els = [L["oblique_elevation_deg"]] + list(L.get("oblique_fallback_elevations_deg", []))
        dists = [L["oblique_distance_m"]] + list(L.get("oblique_fallback_distances_m", []))
        shifts = [0] + list(L.get("oblique_fallback_azimuth_shifts_deg", []))
        taken = []
        for k, offs in enumerate(L["oblique_azimuth_offsets_deg"]):
            # an entry may list alternatives (e.g. [0, 180]: robot side or far side); the lowest
            # reachable elevation wins, so a near object gets a back view instead of a steep one
            offs = offs if isinstance(offs, list) else [offs]
            off = offs[0]
            found = None
            for el in els:
                for az in [o + sh for sh in shifts for o in offs]:  # listed azimuths first, then shifts
                    if any(abs((az - t + 180) % 360 - 180) < L.get("min_azimuth_gap_deg", 40) for t in taken):
                        continue
                    for d in dists:
                        a, e = math.radians(phi0 + az), math.radians(el)
                        eye = look + d * np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])
                        q, Tp, roll = self.cam_pose_ik(eye, look, check=False)
                        if q is None:
                            continue
                        uv, depth = vz.project(Tp, self.K, self.dist, corners)
                        mg = 0.04 * np.array(self.size)
                        if (depth <= 0).any() or (uv < mg).any() or (uv > np.array(self.size) - mg).any():
                            continue
                        found = (q, {"azimuth_offset_deg": az, "elevation_deg": el, "distance_m": d, "roll_deg": roll})
                        break
                    if found:
                        break
                if found:
                    break
            if found is None:
                self.note(event="oblique_unreachable", azimuth_offset_deg=off)
                continue
            taken.append(found[1]["azimuth_offset_deg"])
            if found[1]["azimuth_offset_deg"] != off or found[1]["elevation_deg"] != els[0]:
                self.note(event="oblique_fallback", requested_offset_deg=off, **found[1])
            q = self.go(found[0], phase=phase)
            self.robot_q_cache = q
            rgb, q, T, res, dets, dev = self.look_and_segment(q, f"oblique_{k}", look)
            if res is not None:
                shot(rgb, q, T, res, dets, dev, f"oblique_{k}", found[1])
        span = self.cfg["locate"]["carve_half_span_m"]
        lo = np.array([p0[0] - span, p0[1] - span, 0.0])
        hi = np.array([p0[0] + span, p0[1] + span, self.cfg["locate"]["carve_max_height_m"]])
        P = vz.carve(views, self.K, self.dist, lo, hi, voxel=0.004, min_seen=max(2, len(views) - 1))
        if len(P) < 20:
            raise RuntimeError("visual hull empty: masks inconsistent")
        P = P[P[:, 2] > 0.004]  # drop the table-plane slab
        plo, phi_ = np.percentile(P, 0.5, 0), np.percentile(P, 99.5, 0)
        center = (plo + phi_) / 2
        rect = cv2.minAreaRect(P[:, :2].astype(np.float32))
        rxy = float(np.max(np.linalg.norm(P[:, :2] - center[:2], axis=1)))
        self.obj = {"center": center, "lo": plo, "hi": phi_, "top": float(phi_[2]), "radius_xy": rxy,
                    "footprint_rect": {"center": rect[0], "size": rect[1], "angle_deg": rect[2]},
                    "n_voxels": int(len(P)), "views_used": len(views)}
        np.save(self.out / "locate/visual_hull_points_board.npy", P)
        (self.out / "locate/object_estimate.json").write_text(json.dumps(self.obj, indent=1, default=_json))
        self.note(event="localized", **{k: v for k, v in self.obj.items()})
        R.annotate(f"物件位置 (桌面座標) x={center[0] * 100:.1f}cm y={center[1] * 100:.1f}cm  高 {phi_[2] * 100:.1f}cm  "
                   f"佔地 {rect[1][0] * 100:.1f}x{rect[1][1] * 100:.1f}cm",
                   _draw_hull(rgb, T, self.K, self.dist, P, plo, phi_), hold_s=2.5)

    # ------------------------------------------------------------------ 4. plan + capture
    def plan_views(self):
        s = self.cfg["scan"]
        c = self.obj["center"].copy()
        r = 0.5 * float(np.linalg.norm(self.obj["hi"] - self.obj["lo"]))
        vfov = 2 * math.atan(self.size[1] / 2 / self.K[1, 1])
        d0 = float(np.clip(r / math.sin(s["fill"] * vfov / 2), s["min_distance_m"], s["max_distance_m"]))
        base_dir = vz.inv(self.B)[:3, 3][:2] - c[:2]
        phi0 = math.degrees(math.atan2(base_dir[1], base_dir[0]))
        corners = np.array([[x, y, z] for x in (self.obj["lo"][0], self.obj["hi"][0])
                            for y in (self.obj["lo"][1], self.obj["hi"][1]) for z in (0, self.obj["hi"][2])])
        plans = []

        def az_gap(a, b):
            return abs((a - b + 180) % 360 - 180)

        offsets = list(s["azimuth_offsets_deg"])
        for k, off in enumerate(offsets):
            found = self._plan_one(k, off, (0, 12, -12, 25, -25), s["elevation_candidates_deg"], c, d0, corners)
            if found is None:
                # far objects: the far-side views are out of reach -> try other azimuths and steeper
                # (more top-down) elevations, keeping views >= min_gap apart so the object is still circled
                gap = s.get("min_azimuth_gap_deg", 45)
                taken = [p["azimuth_offset_deg"] for p in plans] + [o for o in offsets[k + 1:]]
                alts = [off + d for d in s.get("fallback_azimuth_shifts_deg", (35, -35, 55, -55, 75, -75))]
                alts = [a for a in alts if all(az_gap(a, t) >= gap for t in taken)]
                els = list(s["elevation_candidates_deg"]) + list(s.get("fallback_elevations_deg", (65, 75)))
                for a in alts:
                    # closer camera = flange nearer the shoulder; still checked for full framing
                    found = self._plan_one(k, a, (0,), els, c, d0, corners, scales=(1.0, 0.9, 0.8, 0.7))
                    if found:
                        self.note(event="view_fallback", requested_offset_deg=off, used_offset_deg=a, elevation_deg=found["elevation_deg"])
                        break
            if found is None:
                self.note(event="view_unreachable", azimuth_offset_deg=off)
            else:
                plans.append(found)
        self.note(event="views_planned", n=len(plans), distance_m=d0,
                  plan=[{k: v for k, v in p.items() if k in ("azimuth_offset_deg", "elevation_deg", "distance_m", "roll_deg")} for p in plans])
        return plans

    def _plan_one(self, k, off, dazs, elevations, c, d0, corners, scales=(1.0, 1.15, 1.3, 0.9)):
        base_dir = vz.inv(self.B)[:3, 3][:2] - c[:2]
        phi0 = math.degrees(math.atan2(base_dir[1], base_dir[0]))
        found = None
        for daz in dazs:
            for el in elevations:
                for scale in scales:
                    a, e, d = math.radians(phi0 + off + daz), math.radians(el), d0 * scale
                    eye = c + d * np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])
                    q, T, roll = self.cam_pose_ik(eye, c)
                    if q is None:
                        continue
                    uv, depth = vz.project(T, self.K, self.dist, corners)
                    m = 0.04 * np.array(self.size)
                    if (depth <= 0).any() or (uv < m).any() or (uv > np.array(self.size) - m).any():
                        continue
                    found = {"index": k, "azimuth_offset_deg": off + daz, "elevation_deg": el, "distance_m": d,
                             "roll_deg": roll, "q": q, "board_T_cam_plan": T}
                    break
                if found:
                    break
            if found:
                break
        return found

    def capture_views(self, plans):
        R = self.robot
        phase = "4 環繞拍攝 Capture"
        R.annotate(f"4 IK 規劃 {len(plans)} 個環繞視角並拍攝 (方位 {[round(p['azimuth_offset_deg']) for p in plans]}°)", phase=phase)
        high = None
        views = []
        names = ["front", "left", "back", "right", "v4", "v5", "v6", "v7"]
        for i, p in enumerate(plans):
            q = self.go(p["q"], phase=phase)
            self.robot_q_cache = q
            rgb, q, T, res, dets, dev = self.look_and_segment(q, f"scan_{i}", self.obj["center"])
            if res is None:
                self.note(event="view_no_detection", view=i)
                continue
            name = names[len(views)]
            views.append(self._save_view(name, rgb, res, q, T, dets, dev, {k: v for k, v in p.items() if k != "q"}))
            vis = _draw_det(rgb, res, self.cfg["prompt"])
            R.annotate(f"視角 {len(views)}/{len(plans)}  方位 {p['azimuth_offset_deg']:.0f}°  仰角 {p['elevation_deg']}°  距離 {p['distance_m'] * 100:.0f}cm",
                       vis, slot=f"view{len(views) - 1}")
            R.annotate("", rgb * res["mask"][..., None], slot=f"thumb{len(views) - 1}")
        self.go(self.home, phase="5 回到原位 Home")
        self._write_views(views, source="orbit capture")
        return views

    def export_hull_views(self):
        """Use the localisation (visual hull) photos themselves as TRELLIS inputs: no second orbit."""
        views = []
        for s in self.hull_shots[: self.cfg["locate"].get("max_trellis_inputs", 4)]:
            views.append(self._save_view(s["label"], s["rgb"], s["res"], s["q"], s["T"], s["dets"], s["dev"], s["plan"]))
        self.go(self.home, phase="5 回到原位 Home")
        self._write_views(views, source="visual hull photos (nadir + obliques)")
        return views

    def _save_view(self, name, rgb, res, q, T, dets, dev, plan):
        mask = res["mask"]
        rgba = np.dstack([rgb, (mask * 255).astype(np.uint8)])
        Image.fromarray(rgb).save(self.out / f"views/{name}_rgb.png")
        Image.fromarray((mask * 255).astype(np.uint8)).save(self.out / f"views/{name}_mask.png")
        up_rgba, roll_fix = vz.upright(rgba, T, self.K, self.dist, self.obj["center"])
        Image.fromarray(up_rgba, "RGBA").save(self.out / f"inputs/{name}.png")  # upright for TRELLIS
        return {"id": name, "q": q, "board_T_cam": T, "tags_seen": [d["id"] for d in dets],
                "pnp_vs_fk_mm_deg": dev, "score": res["score"], "upright_rotation_deg": roll_fix, "plan": plan}

    def _write_views(self, views, source):
        meta = {"K": self.K, "dist": self.dist, "image_size": self.size, "views": views, "source": source,
                "object_estimate": self.obj, "frame": "board (AprilTag) frame, metres, Z up = table normal",
                "base_T_board": self.B}
        (self.out / "views/views.json").write_text(json.dumps(meta, indent=1, default=_json))
        self.note(event="captured", n=len(views), source=source)


def _json(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(type(o))


def _draw_det(rgb, res, label):
    from detect import draw_mask
    return draw_mask(rgb, res["mask"], res["box"], f"{label} {res['score']:.2f}")


def _draw_hull(rgb, T, K, dist, P, lo, hi):
    vis = rgb.copy()
    sub = P[:: max(1, len(P) // 4000)]
    uv, _ = vz.project(T, K, dist, sub)
    for u, v in uv.astype(int):
        if 0 <= u < vis.shape[1] and 0 <= v < vis.shape[0]:
            cv2.circle(vis, (u, v), 2, (0, 255, 255), -1)
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (0.0, hi[2])])
    uv, _ = vz.project(T, K, dist, corners)
    uv = uv.astype(int)
    for a, b in [(0, 1), (0, 2), (1, 3), (2, 3), (4, 5), (4, 6), (5, 7), (6, 7), (0, 4), (1, 5), (2, 6), (3, 7)]:
        cv2.line(vis, tuple(uv[a]), tuple(uv[b]), (255, 60, 0), 3)
    return vis


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--port", type=int, default=6200)
    ap.add_argument("--calibration", type=Path, default=None, help="reuse a previous calib/calibration.json")
    a = ap.parse_args()
    cfg = json.loads(a.config.read_text())
    out = a.out.resolve()
    sys.addaudithook(_guard)
    from detect import Segmenter
    seg = Segmenter(cfg["prompt"], **cfg.get("detector", {}))
    robot = Robot(a.port)
    sc = Scanner(cfg, out, robot, seg)
    try:
        robot.annotate(f"三爪 FR3 自動掃描 → TRELLIS  |  目標：\"{cfg['prompt']}\"", phase="0 開始 Start", hold_s=1.5)
        if a.calibration:
            sc.load_calibration(a.calibration)
            robot.annotate("沿用既有校正檔", phase="1 校正 Calibration")
        else:
            sc.calibrate()
        sc.locate()
        if cfg["locate"].get("hull_views_as_trellis_inputs", False):
            views = sc.export_hull_views()
            if len(views) < cfg["locate"].get("min_trellis_inputs", 3):
                raise RuntimeError(f"only {len(views)} usable visual-hull photos")
        else:
            plans = sc.plan_views()
            if len(plans) < cfg["scan"]["min_views"]:
                raise RuntimeError(f"only {len(plans)} reachable views")
            views = sc.capture_views(plans)
        robot.annotate("拍攝完成，關閉模擬器並啟動 TRELLIS", phase="5 回到原位 Home", hold_s=1.0)
        sc.note(event="done")
    except Exception as e:
        sc.note(event="error", error=f"{type(e).__name__}: {e}")
        try:
            robot.annotate(f"錯誤: {e}", hold_s=2)
        except Exception:
            pass
        raise
    finally:
        try:
            robot.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
