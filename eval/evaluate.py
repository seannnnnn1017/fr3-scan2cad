"""Post-hoc evaluation against simulator ground truth.

Runs only after the brain, TRELLIS and registration are finished: it first hashes every result
file (freezing them in eval/frozen_outputs.json), then opens sim_private/.  Nothing here feeds back.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

CV_FROM_USD = np.diag([1.0, -1.0, -1.0, 1.0])


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rot_deg(A, B):
    return float(np.degrees(np.linalg.norm(Rotation.from_matrix(A.T @ B).as_rotvec())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    a = ap.parse_args()
    run = a.run.resolve()
    out = run / "eval"
    out.mkdir(exist_ok=True)
    frozen = {str(p.relative_to(run)): sha(p) for d in ("calib", "locate", "views", "trellis", "cad")
              for p in sorted((run / d).rglob("*")) if p.is_file()}
    (out / "frozen_outputs.json").write_text(json.dumps(frozen, indent=1))

    gt = json.loads((run / "sim_private/ground_truth.json").read_text())
    mesh_gt = np.load(run / "sim_private/object_mesh_world.npz")
    calib = json.loads((run / "calib/calibration.json").read_text())
    est = json.loads((run / "locate/object_estimate.json").read_text())
    R = {}

    # --- calibration
    K, Kgt = np.array(calib["K"]), np.array(gt["wrist_K_ideal"])
    R["intrinsics"] = {"fx": [K[0, 0], Kgt[0, 0]], "fy": [K[1, 1], Kgt[1, 1]], "cx": [K[0, 2], Kgt[0, 2]], "cy": [K[1, 2], Kgt[1, 2]],
                       "fx_err_pct": float(abs(K[0, 0] / Kgt[0, 0] - 1) * 100), "dist": calib["dist"]}
    X, Xgt = np.array(calib["link8_T_cam"]), np.array(gt["link8_from_wrist_camera_usd"]) @ CV_FROM_USD
    R["hand_eye"] = {"trans_err_mm": float(np.linalg.norm(X[:3, 3] - Xgt[:3, 3]) * 1000), "rot_err_deg": rot_deg(X[:3, :3], Xgt[:3, :3])}
    B, Bgt = np.array(calib["base_T_board"]), np.array(gt["world_from_board"])
    R["board_pose"] = {"trans_err_mm": float(np.linalg.norm(B[:3, 3] - Bgt[:3, 3]) * 1000), "rot_err_deg": rot_deg(B[:3, :3], Bgt[:3, :3])}

    # --- object in the board frame
    V = mesh_gt["vertices"]
    Vb = (np.linalg.inv(Bgt) @ np.c_[V, np.ones(len(V))].T).T[:, :3]
    glo, ghi = Vb.min(0), Vb.max(0)
    elo, ehi = np.array(est["lo"]), np.array(est["hi"])
    R["localisation_visual_hull"] = {
        "gt_bbox_center_board_m": ((glo + ghi) / 2).tolist(), "est_bbox_center_board_m": ((elo + ehi) / 2).tolist(),
        "center_xy_err_mm": float(np.linalg.norm(((glo + ghi) / 2 - (elo + ehi) / 2)[:2]) * 1000),
        "height_gt_mm": float(ghi[2] * 1000), "height_est_mm": float(ehi[2] * 1000)}
    R["object_disturbed_by_arm_m"] = gt["object_moved_m"]

    # --- visual hull itself vs ground truth
    g = trimesh.Trimesh(Vb, mesh_gt["faces"], process=False)
    pb, _ = trimesh.sample.sample_surface(g, 30000, seed=0)
    hp = run / "locate/visual_hull_points_board.npy"
    if hp.exists():
        P = np.load(hp)
        hlo, hhi = np.percentile(P, 0.5, 0), np.percentile(P, 99.5, 0)
        d_gt_to_hull = cKDTree(P).query(pb)[0]
        R["visual_hull_vs_gt"] = {
            "center_err_mm": float(np.linalg.norm((hlo + hhi) / 2 - (glo + ghi) / 2) * 1000),
            "center_xy_err_mm": float(np.linalg.norm(((hlo + hhi) / 2 - (glo + ghi) / 2)[:2]) * 1000),
            "bbox_dims_mm_hull": ((hhi - hlo) * 1000).tolist(), "bbox_dims_mm_gt": ((ghi - glo) * 1000).tolist(),
            "gt_surface_within_4mm_of_hull_pct": float((d_gt_to_hull < 0.004 * 1.01).mean() * 100),
            "note": "A visual hull is a superset of the object: dims >= GT and coverage near 100% are expected."}

    # --- registered CAD (and every alignment variant) vs ground-truth mesh (board frame)
    def cad_metrics(path):
        m = trimesh.load(path, force="mesh", process=False)
        pa, _ = trimesh.sample.sample_surface(m, 30000, seed=0)
        da, db = cKDTree(pb).query(pa)[0], cKDTree(pa).query(pb)[0]
        lo, hi = m.bounds
        f = lambda th: float(2 * (da < th).mean() * (db < th).mean() / max(1e-9, (da < th).mean() + (db < th).mean()))
        return {
            "chamfer_mean_mm": float((da.mean() + db.mean()) / 2 * 1000),
            "accuracy_median_mm": float(np.median(da) * 1000), "completeness_median_mm": float(np.median(db) * 1000),
            "fscore_5mm": f(.005), "fscore_10mm": f(.01),
            "bbox_dims_mm_est": ((hi - lo) * 1000).tolist(), "bbox_dims_mm_gt": ((ghi - glo) * 1000).tolist(),
            "bbox_dims_err_pct": (np.abs((hi - lo) / (ghi - glo) - 1) * 100).tolist(),
            "centroid_err_mm": float(np.linalg.norm(((lo + hi) / 2 - (glo + ghi) / 2)) * 1000),
        }

    cad = run / "cad/object_board_frame.glb"
    if cad.exists():
        R["cad_vs_gt"] = cad_metrics(cad)
        R["cad_vs_gt"]["note"] = "Bottom of the object is never photographed; TRELLIS hallucinates it. Chamfer uses the full GT surface."
        R["alignments_vs_gt"] = {p.stem.replace("_board_frame", ""): cad_metrics(p)
                                 for p in sorted((run / "cad/alignments").glob("*_board_frame.glb"))}
        g.export(out / "gt_mesh_board_frame.glb")
    (out / "evaluation.json").write_text(json.dumps(R, indent=1, default=float))
    print(json.dumps(R, indent=1, default=float))


if __name__ == "__main__":
    main()
