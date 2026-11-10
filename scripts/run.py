"""One-command pipeline:  Isaac Sim (as the real world) + scanning brain -> TRELLIS -> metric CAD -> video.

  .venv/bin/python scripts/run.py --scene configs/scenes/mug.json --scan configs/scans/mug.json

Stages (each in its own process / Python env):
  sim      Isaac Sim python.sh  sim/sim_server.py      robot + camera driver, films everything
  brain    .venv                scan2cad/scanner.py    calibrate, detect, visual hull (its photos = TRELLIS inputs)
  trellis  TRELLIS venv         generate/trellis_run.py RGBA photos -> mesh (after Isaac exits, frees VRAM)
  register .venv                scan2cad/register.py   align mesh to the visual hull -> metric CAD + 6-D pose + hull export
  video    .venv                scan2cad/video.py      full-process MP4
  evaluate .venv                eval/evaluate.py       (optional) compare with simulator ground truth
                                eval/verify_video.py   CAD vs GT overlay clip -> process_with_eval.mp4
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]  # repo root
sys.path.insert(0, str(HERE))
from common import paths  # noqa: E402

def ISAAC():
    return paths.get("ISAAC_PYTHON")  # looked up only when the sim stage runs
BRAIN_PY = paths.get("BRAIN_PYTHON", str(HERE / ".venv/bin/python"))
def MODEL_PY():
    return paths.get("TRELLIS_PYTHON")


def stage(name, cmd, log, **kw):
    print(f"\n=== {name}: {' '.join(map(str, cmd))}\n    log -> {log}", flush=True)
    t = time.time()
    with open(log, "w") as f:
        r = subprocess.run(list(map(str, cmd)), stdout=f, stderr=subprocess.STDOUT, **kw)
    print(f"=== {name}: exit {r.returncode} in {time.time() - t:.0f}s", flush=True)
    if r.returncode != 0:
        print(Path(log).read_text()[-3000:])
        raise SystemExit(f"stage {name} failed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=Path, default=HERE / "configs/scenes/mug.json")
    ap.add_argument("--scan", type=Path, default=HERE / "configs/scans/mug.json")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--port", type=int, default=6200)
    ap.add_argument("--calibration", type=Path, default=None, help="reuse calib/calibration.json from a previous run")
    ap.add_argument("--from-stage", choices=["scan", "trellis", "register", "video", "evaluate"], default="scan")
    ap.add_argument("--no-evaluate", action="store_true")
    ap.add_argument("--gui", action="store_true", help="show the Isaac Sim window while scanning")
    a = ap.parse_args()
    scan_cfg = json.loads(a.scan.read_text())
    out = (a.out or HERE / "runs" / f"{datetime.now():%Y%m%d_%H%M%S}_{scan_cfg['prompt'].replace(' ', '_')}").resolve()
    order = ["scan", "trellis", "register", "video", "evaluate"]
    todo = order[order.index(a.from_stage):]
    if a.from_stage == "scan":
        if out.exists():
            raise SystemExit(f"{out} exists; choose a new --out")
        out.mkdir(parents=True)
        (out / "inputs_config").mkdir()
        (out / "inputs_config/scan.json").write_text(a.scan.read_text())

    if "scan" in todo:
        sim_cmd = [ISAAC(), HERE / "sim/sim_server.py", "--scene", a.scene.resolve(), "--out", out, "--port", a.port]
        if a.gui:
            sim_cmd.append("--gui")
        print(f"=== sim: starting Isaac Sim (log -> {out / 'sim.log'})", flush=True)
        sim_log = open(out / "sim.log", "w")
        sim = subprocess.Popen(list(map(str, sim_cmd)), stdout=sim_log, stderr=subprocess.STDOUT)
        try:
            for _ in range(900):
                if "[sim] READY" in (out / "sim.log").read_text(errors="ignore"):
                    break
                if sim.poll() is not None:
                    raise SystemExit("Isaac Sim exited during startup; see sim.log")
                time.sleep(1)
            brain_cmd = [BRAIN_PY, HERE / "scan2cad/scanner.py", "--config", a.scan.resolve(), "--out", out, "--port", a.port]
            if a.calibration:
                brain_cmd += ["--calibration", a.calibration.resolve()]
            stage("brain (calibrate -> detect -> visual hull capture)", brain_cmd, out / "brain.log")
            sim.wait(timeout=300)
        finally:
            if sim.poll() is None:
                time.sleep(5)
                sim.kill()
            sim_log.close()

    t = scan_cfg.get("trellis", {})
    def trellis(seed):
        stage(f"trellis (seed {seed})", [MODEL_PY(), HERE / "generate/trellis_run.py", "--inputs", out / "inputs", "--out", out / "trellis",
                                         "--seed", seed, "--steps", t.get("steps", 12),
                                         "--ss-cfg", t.get("ss_cfg", 7.5), "--slat-cfg", t.get("slat_cfg", 3.0)], out / "trellis.log")

    def register():
        stage("register", [BRAIN_PY, HERE / "scan2cad/register.py", "--run", out], out / "register.log")
        return json.loads((out / "cad/pose_and_scale.json").read_text())["silhouette_iou_mean"]

    if "trellis" in todo:
        # TRELLIS sometimes generates a broken shape.  The brain can tell without any ground truth: the
        # aligned mesh then fails to explain its own masks.  Retry with other seeds, keep the best.
        min_iou = scan_cfg.get("register", {}).get("min_silhouette_iou", 0.85)
        seeds = [t.get("seed", 42) + i for i in range(t.get("max_attempts", 3))]
        tries = []
        for seed in seeds:
            trellis(seed)
            iou = register()
            print(f"=== seed {seed}: silhouette IoU {iou:.3f} (accept >= {min_iou})", flush=True)
            tries.append((iou, seed))
            if iou >= min_iou or seed == seeds[-1]:
                break
            for d in ("trellis", "cad"):
                (out / d).rename(out / f"{d}_rejected_seed{seed}")
        best = max(tries)
        if best[1] != tries[-1][1]:  # every seed failed: restore the best one
            for d in ("trellis", "cad"):
                (out / d).rename(out / f"{d}_rejected_seed{tries[-1][1]}")
                (out / f"{d}_rejected_seed{best[1]}").rename(out / d)
        (out / "trellis_attempts.json").write_text(json.dumps({"min_silhouette_iou": min_iou, "attempts":
                                                               [{"seed": s_, "silhouette_iou": i_} for i_, s_ in tries], "kept_seed": best[1]}, indent=1))
    elif "register" in todo:
        register()
    if "video" in todo:
        stage("video", [BRAIN_PY, HERE / "scan2cad/video.py", "--run", out], out / "video.log")
    if "evaluate" in todo and not a.no_evaluate:
        stage("evaluate", [BRAIN_PY, HERE / "eval/evaluate.py", "--run", out], out / "evaluate.log")
        stage("verify video", [BRAIN_PY, HERE / "eval/verify_video.py", "--run", out], out / "verify_video.log")
    print(f"\nDone. Run folder: {out}\n  video: {out / 'process.mp4'}\n  CAD:   {out / 'cad'}")


if __name__ == "__main__":
    main()
