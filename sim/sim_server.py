"""Isaac Sim stand-in for the real lab hardware: FR3 + three-finger hand, wrist RGB camera,
a wood table with four AprilTags, an unknown object, and a camcorder that films everything.

The brain talks to this process only through common/protocol.py.  It receives what a real
robot driver / USB camera would give: joint encoder readings and RGB frames.  Object pose,
board pose, camera intrinsics, camera mount, depth, segmentation... are never sent.  They are
written to <out>/sim_private/ at shutdown, for post-hoc evaluation only.

Run with Isaac Sim's python.sh:
  python.sh sim/sim_server.py --scene configs/scenes/mug.json --out runs/X --port 6200
  python.sh sim/sim_server.py --scene configs/scenes/mug.json --out runs/X --probe   # dev check
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import protocol  # noqa: E402
from common import paths  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--scene", required=True, type=Path)
ap.add_argument("--out", required=True, type=Path)
ap.add_argument("--port", type=int, default=6200)
ap.add_argument("--gui", action="store_true")
ap.add_argument("--probe", action="store_true", help="developer check: render a few frames, compare FK, exit")
args, _ = ap.parse_known_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": not args.gui, "renderer": "RayTracedLighting", "anti_aliasing": 0})
import carb.settings  # noqa: E402

_s = carb.settings.get_settings()
_s.set("/rtx/post/aa/op", 0)  # no temporal AA/DLSS: frames must not blend the previous camera pose
_s.set("/rtx/post/tonemap/enableAutoExposure", False)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.prims import Articulation, XFormPrim  # noqa: E402
from isaacsim.core.utils.stage import add_reference_to_stage  # noqa: E402
from isaacsim.sensors.camera import Camera  # noqa: E402

ROBOT = "/World/FR3_ThreeFinger/fr3"
ARM_N = 7
FINGERS = np.arange(9, 17)
PHYSICS_HZ = 120


def resolve(base: Path, p: str) -> str:
    return paths.resolve_asset(p, base)  # relative / ${VAR} / isaac://  (see common/paths.py)


# ----------------------------------------------------------------------------- scene building
def srgb_to_linear(c):
    c = np.asarray(c, float)
    return np.where(c <= .04045, c / 12.92, ((c + .055) / 1.055) ** 2.4)


def preview_material(stage, path, color=None, texture=None, roughness=.7, uv_scale=1.0):
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, path + "/Surface")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
    if texture:
        tex = UsdShade.Shader.Define(stage, path + "/Texture")
        tex.CreateIdAttr("UsdUVTexture")
        tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(texture)
        tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
        for w in ("wrapS", "wrapT"):
            tex.CreateInput(w, Sdf.ValueTypeNames.Token).Set("repeat")
        tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        rd = UsdShade.Shader.Define(stage, path + "/UV")
        rd.CreateIdAttr("UsdPrimvarReader_float2")
        rd.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        rd.CreateOutput("result", Sdf.ValueTypeNames.Float2)
        tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(rd.ConnectableAPI(), "result")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), "rgb")
    else:
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*srgb_to_linear(color).tolist()))
    sh.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat


def quad(stage, path, corners_xyz, uvs, mat):
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([Gf.Vec3f(*map(float, c)) for c in corners_xyz])
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDoubleSidedAttr(True)
    UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray,
                                            UsdGeom.Tokens.faceVarying).Set([Gf.Vec2f(*u) for u in uvs])
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)
    return mesh


def rot_z(deg):
    a = math.radians(deg)
    return np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1.0]])


def rot_xyz(deg):
    x, y, z = np.radians(deg)
    rx = np.array([[1, 0, 0], [0, math.cos(x), -math.sin(x)], [0, math.sin(x), math.cos(x)]])
    ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    rz = np.array([[math.cos(z), -math.sin(z), 0], [math.sin(z), math.cos(z), 0], [0, 0, 1]])
    return rz @ ry @ rx


def build_table_and_board(stage, cfg, base):
    t = cfg["table"]
    cx, cy = t["center_xy_m"]
    sx, sy = t["size_m"]
    top = UsdGeom.Cube.Define(stage, "/World/Table")
    top.CreateSizeAttr(1.0)
    top.AddTranslateOp().Set(Gf.Vec3d(cx, cy, -0.02))
    top.AddScaleOp().Set(Gf.Vec3f(sx, sy, 0.04))
    UsdPhysics.CollisionAPI.Apply(top.GetPrim())
    top.CreateVisibilityAttr("invisible")  # collider only; the textured quad below is what cameras see
    tile = float(t.get("texture_tile_m", 0.45))
    wood = preview_material(stage, "/World/Looks/Wood", texture=resolve(base, t["texture"]), roughness=.62)
    h = [(cx - sx / 2, cy - sy / 2), (cx + sx / 2, cy - sy / 2), (cx + sx / 2, cy + sy / 2), (cx - sx / 2, cy + sy / 2)]
    quad(stage, "/World/TableTop", [(x, y, 0.0) for x, y in h], [(x / tile, y / tile) for x, y in h], wood)
    legs = preview_material(stage, "/World/Looks/Floor", color=[.30, .31, .33], roughness=.9)
    floor = UsdGeom.Cube.Define(stage, "/World/Floor")
    floor.CreateSizeAttr(1.0)
    floor.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.80))
    floor.AddScaleOp().Set(Gf.Vec3f(8, 8, 0.04))
    UsdShade.MaterialBindingAPI.Apply(floor.GetPrim()).Bind(legs)

    b = cfg["board"]
    R = rot_z(b["yaw_deg"])
    origin = np.array([*b["center_xy_m"], 0.0])
    tex_dir = Path(resolve(base, b["tag_textures_dir"]))
    half = b["tag_size_m"] / 0.8 / 2  # texture = 8-cell black boundary + 1 white cell each side
    paper = preview_material(stage, "/World/Looks/Paper", color=[.93, .93, .91], roughness=.95)
    for key, (x, y) in b["centers_m"].items():
        ph = b["tag_size_m"] * b.get("paper_scale", 1.6) / 2  # printed sheet glued on the wood
        sheet = [origin + R @ np.array([u, v, 0.0002]) for u, v in
                 [(x - ph, y - ph), (x + ph, y - ph), (x + ph, y + ph), (x - ph, y + ph)]]
        quad(stage, f"/World/TagPaper{key}", sheet, [(0, 0), (1, 0), (1, 1), (0, 1)], paper)
        local = [(x - half, y - half), (x + half, y - half), (x + half, y + half), (x - half, y + half)]
        world = [origin + R @ np.array([u, v, 0.0004]) for u, v in local]
        mat = preview_material(stage, f"/World/Looks/Tag{key}", texture=str(tex_dir / f"tag{key}.png"), roughness=.9)
        quad(stage, f"/World/Tag{key}", world, [(0, 0), (1, 0), (1, 1), (0, 1)], mat)
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, origin
    return T  # world_from_board (private)


def build_lights(stage, cfg):
    dome = UsdLux.DomeLight.Define(stage, "/World/Lights/Dome")
    dome.CreateIntensityAttr(float(cfg.get("dome_intensity", 650)))
    dome.CreateColorAttr(Gf.Vec3f(.96, .97, 1.0))
    for i, (pos, size, power, color) in enumerate(cfg.get("rect_lights", [])):
        light = UsdLux.RectLight.Define(stage, f"/World/Lights/Rect{i}")
        light.CreateWidthAttr(size[0])
        light.CreateHeightAttr(size[1])
        light.CreateIntensityAttr(float(power))
        light.CreateColorAttr(Gf.Vec3f(*color))
        target = np.array(cfg.get("light_target", [0.45, 0, 0]), float)
        p = np.array(pos, float)
        f = (target - p) / np.linalg.norm(target - p)
        r = np.cross(f, [0, 0, 1.0])
        r /= np.linalg.norm(r)
        u = np.cross(r, f)
        m = np.eye(4)
        m[:3, :3] = np.column_stack([r, u, -f])
        m[:3, 3] = p
        UsdGeom.Xformable(light.GetPrim()).AddTransformOp().Set(Gf.Matrix4d(m.T.tolist()))


def bbox_of(prim):
    b = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"], useExtentsHint=False)
    r = b.ComputeWorldBound(prim).ComputeAlignedRange()
    return np.array(r.GetMin()), np.array(r.GetMax())


def build_object(stage, o, base, path="/World/Object"):
    usd = resolve(base, o["usd"])
    src = Usd.Stage.Open(usd)
    root = src.GetPrimAtPath(o["source_prim"]) if o.get("source_prim") else src.GetDefaultPrim()
    xf = UsdGeom.Xform.Define(stage, path)
    child = stage.DefinePrim(f"{path}/Asset")
    child.GetReferences().AddReference(usd, str(root.GetPath()))
    M = np.eye(4)
    if str(UsdGeom.GetStageUpAxis(src)) == "Y":
        M[:3, :3] = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    M[:3, :3] = rot_z(o.get("yaw_deg", 0)) @ rot_xyz(o.get("rotation_xyz_deg", [0, 0, 0])) @ M[:3, :3] \
        * UsdGeom.GetStageMetersPerUnit(src) * float(o.get("scale", 1.0))
    op = xf.AddTransformOp()
    op.Set(Gf.Matrix4d(M.T.tolist()))
    lo, hi = bbox_of(xf.GetPrim())
    x, y = o["position_xy_m"]
    M[:3, 3] = np.array([x, y, 0.001]) - np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]])
    op.Set(Gf.Matrix4d(M.T.tolist()))
    if o.get("dynamic", True):  # a collision would knock it over, as in the real world
        UsdPhysics.RigidBodyAPI.Apply(xf.GetPrim())
        UsdPhysics.MassAPI.Apply(xf.GetPrim()).CreateMassAttr(float(o.get("mass_kg", 0.25)))
        if o.get("angular_damping"):  # rolling resistance (a ball on a real table does not drift away)
            PhysxSchema.PhysxRigidBodyAPI.Apply(xf.GetPrim()).CreateAngularDampingAttr(float(o["angular_damping"]))
        changed = True
        while changed:  # instanceable meshes cannot take collision APIs: de-instance first
            changed = False
            for prim in Usd.PrimRange(child):
                if prim.IsInstanceable():
                    prim.SetInstanceable(False)
                    changed = True
        n_col = 0
        for prim in Usd.PrimRange(child):
            if prim.IsA(UsdGeom.Mesh):
                n_col += 1
                UsdPhysics.CollisionAPI.Apply(prim)
                UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("convexHull")
        print(f"[sim] {path}: {n_col} collision meshes", flush=True)
    return xf.GetPrim()


def object_mesh_world(stage, prim):
    verts, faces, off = [], [], 0
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    for p in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
        if not p.IsA(UsdGeom.Mesh) or UsdGeom.Imageable(p).ComputeVisibility() == "invisible":
            continue
        m = UsdGeom.Mesh(p)
        pts = np.asarray(m.GetPointsAttr().Get(), float)
        counts = np.asarray(m.GetFaceVertexCountsAttr().Get(), int)
        idx = np.asarray(m.GetFaceVertexIndicesAttr().Get(), int)
        if pts.size == 0:
            continue
        W = np.array(cache.GetLocalToWorldTransform(p)).T  # column-vector convention
        pts = (np.c_[pts, np.ones(len(pts))] @ W.T)[:, :3]
        k = 0
        for c in counts:
            for j in range(1, c - 1):
                faces.append([idx[k] + off, idx[k + j] + off, idx[k + j + 1] + off])
            k += c
        verts.append(pts)
        off += len(pts)
    return np.concatenate(verts), np.asarray(faces, np.int64)


# ----------------------------------------------------------------------------- server
class Rig:
    def __init__(self, cfg, base, out):
        self.cfg, self.base, self.out = cfg, base, out
        self.world = World(stage_units_in_meters=1.0, physics_dt=1.0 / PHYSICS_HZ, rendering_dt=1.0 / PHYSICS_HZ)
        stage = self.stage = omni.usd.get_context().get_stage()
        UsdGeom.SetStageUpAxis(stage, "Z")
        self.world_from_board = build_table_and_board(stage, cfg, base)
        build_lights(stage, cfg["lights"])
        self.object_prim = build_object(stage, cfg["object"], base)
        # other objects on the table (clutter); unknown to the brain, like everything else
        self.clutter = [build_object(stage, d, base, f"/World/Clutter_{i}") for i, d in enumerate(cfg.get("clutter", []))]
        rcfg = cfg["robot"]
        add_reference_to_stage(resolve(base, rcfg["usd"]), "/World/FR3_ThreeFinger")

        # Wrist camera rigidly bolted to the flange (fr3_link8).  The brain does NOT know this
        # transform; it estimates it by hand-eye calibration.
        wc = cfg["wrist_camera"]
        self.wrist_path = f"{ROBOT}/fr3_link8/wrist_cam"
        cam = UsdGeom.Camera.Define(stage, self.wrist_path)
        M = np.eye(4)
        M[:3, :3] = rot_xyz(wc["mount_rpy_deg"])
        M[:3, 3] = wc["mount_xyz_m"]
        UsdGeom.Xformable(cam.GetPrim()).AddTransformOp().Set(Gf.Matrix4d(M.T.tolist()))
        self._intrinsics(self.wrist_path, wc["focal_mm"], wc["h_aperture_mm"], wc["v_aperture_mm"])
        self.wrist_mount = M

        self.robot = self.world.scene.add(Articulation(ROBOT, name="fr3"))
        self.link8 = XFormPrim(f"{ROBOT}/fr3_link8")
        rec = cfg["recorder"]
        self.rec_res = tuple(rec["resolution"])
        self.wrist_res = tuple(wc["resolution"])
        self.ext = Camera("/World/Camcorder", resolution=self.rec_res)
        self.wrist = Camera(self.wrist_path, resolution=self.wrist_res)
        self.world.reset()
        self.ext.initialize()
        self.wrist.initialize()
        eye, look = np.array(rec["eye"], float), np.array(rec["look_at"], float)
        self.ext.set_world_pose(eye, self._lookat_quat(eye, look), camera_axes="usd")
        self._intrinsics("/World/Camcorder", rec["focal_mm"], rec["h_aperture_mm"], rec["h_aperture_mm"] * self.rec_res[1] / self.rec_res[0])

        n = len(self.robot.dof_names)
        self.dof_names = list(self.robot.dof_names)
        kps, kds = np.full((1, n), 400.0, np.float32), np.full((1, n), 80.0, np.float32)
        kps[0, FINGERS], kds[0, FINGERS] = 14.0, 2.0
        self.robot.set_gains(kps=kps, kds=kds)
        self.limits = np.asarray(self.robot.get_dof_limits()[0], float)
        poses = json.loads(Path(resolve(base, rcfg["gripper_poses"])).read_text())
        finger = np.array([poses[rcfg.get("finger_pose", "closed")][self.dof_names[i]] for i in FINGERS], np.float32)
        self.cmd = np.zeros(n, np.float32)
        self.cmd[:ARM_N] = np.array(rcfg["home_q"], np.float32)
        self.cmd[FINGERS] = finger
        self.robot.set_joint_positions(self.cmd[None])
        self.robot.set_joint_velocities(np.zeros((1, n), np.float32))
        self.rng = np.random.default_rng(int(wc.get("noise_seed", 0)))

        self.frame_idx = 0
        self.sim_time = 0.0
        self.status = "boot"
        for d in ("rec/ext", "rec/wrist", "rec/overlays", "sim_private"):
            (out / d).mkdir(parents=True, exist_ok=True)
        self.frames_log = (out / "rec/frames.jsonl").open("a")
        self.events_log = (out / "rec/events.jsonl").open("a")
        self.object_start = self._object_pose()
        self.clutter_start = [self._object_pose(p) for p in self.clutter]
        self._apply()
        for _ in range(60):
            self._step(render=False)
        for _ in range(6):
            self._step(render=True)

    # --- helpers
    @staticmethod
    def _lookat_quat(eye, target, up=(0, 0, 1)):
        z = eye - target
        z = z / np.linalg.norm(z)
        x = np.cross(up, z)
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        m = Gf.Matrix4d(1.0)
        for i, row in enumerate((x, y, z)):
            m.SetRow3(i, Gf.Vec3d(*row))
        q = m.ExtractRotationQuat()
        return np.array([q.GetReal(), *q.GetImaginary()])

    def _intrinsics(self, path, focal, hap, vap):
        c = UsdGeom.Camera(self.stage.GetPrimAtPath(path))
        c.GetFocalLengthAttr().Set(float(focal))
        c.GetHorizontalApertureAttr().Set(float(hap))
        c.GetVerticalApertureAttr().Set(float(vap))
        c.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 20.0))

    def _object_pose(self, prim=None):
        M = np.array(UsdGeom.Xformable(prim or self.object_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
        return M

    def _apply(self):
        self.robot.set_joint_position_targets(self.cmd[None])
        tau = self.robot.get_generalized_gravity_forces(joint_indices=np.arange(ARM_N))
        self.robot.set_joint_efforts(tau, joint_indices=np.arange(ARM_N))

    def _step(self, render):
        self._apply()
        self.world.step(render=render)
        self.sim_time += 1.0 / PHYSICS_HZ

    def encoders(self):
        return np.asarray(self.robot.get_joint_positions()[0][:ARM_N], float)

    @staticmethod
    def _rgb(cam, res):
        rgba = cam.get_rgba()
        if rgba is None or rgba.size == 0:
            return np.zeros((res[1], res[0], 3), np.uint8)
        return np.ascontiguousarray(rgba[:, :, :3].astype(np.uint8))

    def record_frame(self):
        """Grab camcorder + live wrist view; called at the recording rate."""
        ext = self._rgb(self.ext, self.rec_res)
        wr = self._rgb(self.wrist, self.wrist_res)
        i = self.frame_idx
        cv2.imwrite(str(self.out / f"rec/ext/{i:06d}.jpg"), ext[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 90])
        cv2.imwrite(str(self.out / f"rec/wrist/{i:06d}.jpg"), cv2.resize(wr, (640, 360))[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 88])
        # joint encoders are ordinary robot telemetry (a real arm streams them too); the video uses them
        # to re-project the brain's own estimates onto the live wrist view.
        self.frames_log.write(json.dumps({"frame": i, "t": round(self.sim_time, 4), "status": self.status,
                                          "q": np.round(self.encoders(), 5).tolist()}) + "\n")
        self.frame_idx += 1

    def run_steps(self, n_steps):
        every = PHYSICS_HZ // int(self.cfg["recorder"]["fps"])
        for k in range(n_steps):
            frame = (k % every) == every - 1
            self._step(render=frame)
            if frame:
                self.record_frame()

    # --- commands
    def move(self, waypoints, durations):
        every = PHYSICS_HZ // int(self.cfg["recorder"]["fps"])
        for wp, dur in zip(waypoints, durations):
            q0 = self.cmd[:ARM_N].astype(float).copy()
            q1 = np.clip(np.asarray(wp, float), self.limits[:ARM_N, 0], self.limits[:ARM_N, 1])
            n = max(1, int(round(dur * PHYSICS_HZ)))
            for k in range(1, n + 1):
                s = k / n
                s = 10 * s**3 - 15 * s**4 + 6 * s**5  # min-jerk
                self.cmd[:ARM_N] = q0 + (q1 - q0) * s
                frame = (k % every) == 0
                self._step(render=frame)
                if frame:
                    self.record_frame()
        # settle
        for k in range(int(1.5 * PHYSICS_HZ)):
            frame = (k % every) == every - 1
            self._step(render=frame)
            if frame:
                self.record_frame()
            vel = np.abs(self.robot.get_joint_velocities()[0][:ARM_N]).max()
            if k > 20 and vel < 2e-3:
                break
        q = self.encoders()
        return {"q": q.tolist(), "max_tracking_error_rad": float(np.abs(q - self.cmd[:ARM_N]).max()),
                "sim_time": self.sim_time}

    def capture(self):
        for _ in range(int(0.3 * PHYSICS_HZ)):
            self._step(render=False)
        for _ in range(8):  # let the renderer converge on the still pose
            self._step(render=True)
        self.record_frame()
        q = self.encoders()
        rgb = self._rgb(self.wrist, self.wrist_res).astype(np.float32)
        sigma = float(self.cfg["wrist_camera"].get("noise_std", 0.0))
        if sigma > 0:  # simple sensor read noise
            rgb = rgb + self.rng.normal(0, sigma, rgb.shape)
        rgb = np.clip(rgb + 0.5, 0, 255).astype(np.uint8)
        ok, png = cv2.imencode(".png", rgb[..., ::-1])
        assert ok
        return {"q": q.tolist(), "sim_time": self.sim_time, "width": rgb.shape[1], "height": rgb.shape[0]}, png.tobytes()

    def annotate(self, header, blob):
        ev = {"frame": self.frame_idx, "t": self.sim_time, "text": header.get("text", ""),
              "phase": header.get("phase"), "hold_s": float(header.get("hold_s", 0.0)),
              "slot": header.get("slot", "overlay")}
        if header.get("phase"):
            self.status = header["phase"]
        if blob is not None:
            p = self.out / f"rec/overlays/{self.frame_idx:06d}_{len(list((self.out / 'rec/overlays').iterdir())):03d}.png"
            p.write_bytes(blob)
            ev["image"] = str(p.relative_to(self.out))
        self.events_log.write(json.dumps(ev, ensure_ascii=False) + "\n")
        self.events_log.flush()
        return {"frame": self.frame_idx}

    def write_private(self):
        """Ground truth for evaluation only.  Never sent over the socket."""
        end = self._object_pose()
        lo, hi = bbox_of(self.object_prim)
        V, F = object_mesh_world(self.stage, self.object_prim)
        np.savez_compressed(self.out / "sim_private/object_mesh_world.npz", vertices=V, faces=F)
        wc = self.cfg["wrist_camera"]
        W, H = wc["resolution"]
        fx = wc["focal_mm"] / wc["h_aperture_mm"] * W
        fy = wc["focal_mm"] / wc["v_aperture_mm"] * H
        pos8, quat8 = self.link8.get_world_poses()
        data = {
            "note": "Simulator ground truth. Written at shutdown; the brain never receives it.",
            "world_from_board": self.world_from_board.tolist(),
            "object_world_from_local_start": self.object_start.tolist(),
            "object_world_from_local_end": end.tolist(),
            "object_moved_m": float(np.linalg.norm(end[:3, 3] - self.object_start[:3, 3])),
            "object_bbox_world_m": [lo.tolist(), hi.tolist()],
            "clutter_moved_m": {d.get("name", str(i)): float(np.linalg.norm(self._object_pose(p)[:3, 3] - s0[:3, 3]))
                                for i, (d, p, s0) in enumerate(zip(self.cfg.get("clutter", []), self.clutter, self.clutter_start))},
            "wrist_K_ideal": [[fx, 0, W / 2], [0, fy, H / 2], [0, 0, 1]],
            "link8_from_wrist_camera_usd": self.wrist_mount.tolist(),
            "usd_camera_axes_note": "USD camera looks along -Z with +Y up; OpenCV camera = USD @ diag(1,-1,-1)",
            "robot_base_world": [0, 0, 0],
            "link8_world_final": {"pos": np.asarray(pos8)[0].tolist(), "quat_wxyz": np.asarray(quat8)[0].tolist()},
            "scene_config": self.cfg,
            "frames_recorded": self.frame_idx,
        }
        (self.out / "sim_private/ground_truth.json").write_text(json.dumps(data, indent=2))


def probe(rig: Rig):
    """Developer check of the kinematic model the brain will assume (nominal FR3 DH)."""
    sys.path.insert(0, str(HERE.parent / "scan2cad"))
    import kinematics as kin
    errs = []
    rng = np.random.default_rng(0)
    qs = [np.array(rig.cfg["robot"]["home_q"])] + [kin.random_q(rng) * 0.6 + kin.MID * 0.4 for _ in range(4)]
    for q in qs:
        rig.move([q], [2.0])
        qe = rig.encoders()
        pos, quat = rig.link8.get_world_poses()
        w, x, y, z = np.asarray(quat)[0]
        R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                      [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                      [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        T = kin.fk_link8(qe)
        dr = np.degrees(np.arccos(np.clip((np.trace(T[:3, :3].T @ R) - 1) / 2, -1, 1)))
        errs.append({"q": qe.round(3).tolist(), "pos_err_mm": float(np.linalg.norm(T[:3, 3] - np.asarray(pos)[0]) * 1000),
                     "rot_err_deg": float(dr)})
        print("[probe]", errs[-1], flush=True)
    meta = {"dof_names": rig.dof_names, "limits": rig.limits.tolist(), "fk_check": errs}
    (rig.out / "probe.json").write_text(json.dumps(meta, indent=2))
    rig.move([rig.cfg["robot"]["home_q"]], [2.0])
    _, png = rig.capture()
    (rig.out / "probe_wrist_home.png").write_bytes(png)
    cv2.imwrite(str(rig.out / "probe_ext.png"), rig._rgb(rig.ext, rig.rec_res)[..., ::-1])


def serve(rig: Rig, port: int):
    listener = protocol.listen(port)
    print(f"[sim] READY port={port}", flush=True)
    conn = listener.accept()
    print("[sim] brain connected", flush=True)
    while True:
        header, blob = protocol.recv(conn)
        cmd = header.get("cmd")
        if cmd not in protocol.COMMANDS:
            protocol.send(conn, {"ok": False, "error": f"unknown command {cmd}"})
            continue
        try:
            if cmd == "hello":
                lim = rig.limits[:ARM_N]
                protocol.send(conn, {"ok": True, "robot": "FR3 + three-finger hand (sim)",
                                     "arm_joint_names": rig.dof_names[:ARM_N], "arm_limits": lim.tolist(),
                                     "camera_resolution": list(rig.wrist_res), "record_fps": rig.cfg["recorder"]["fps"]})
            elif cmd == "get_joints":
                protocol.send(conn, {"ok": True, "q": rig.encoders().tolist()})
            elif cmd == "move":
                if header.get("phase"):
                    rig.status = header["phase"]
                protocol.send(conn, {"ok": True, **rig.move(header["waypoints"], header["durations"])})
            elif cmd == "capture":
                meta, png = rig.capture()
                protocol.send(conn, {"ok": True, **meta}, png)
            elif cmd == "annotate":
                protocol.send(conn, {"ok": True, **rig.annotate(header, blob)})
            elif cmd == "idle":
                rig.run_steps(int(float(header.get("seconds", 1.0)) * PHYSICS_HZ))
                protocol.send(conn, {"ok": True})
            elif cmd == "shutdown":
                rig.write_private()
                protocol.send(conn, {"ok": True, "frames": rig.frame_idx})
                break
        except Exception as e:  # report to brain and keep serving
            traceback.print_exc()
            protocol.send(conn, {"ok": False, "error": f"{type(e).__name__}: {e}"})
    conn.close()
    listener.close()


def main():
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(args.scene.read_text())
    rig = Rig(cfg, args.scene.resolve().parent, out)
    if args.probe:
        probe(rig)
    else:
        serve(rig, args.port)
    rig.frames_log.close()
    rig.events_log.close()


code = 1
try:
    main()
    code = 0
except Exception:
    traceback.print_exc()
finally:
    app.close(exit_code=code) if hasattr(app, "close") else None
