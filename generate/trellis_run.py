"""Multi-image TRELLIS v1 (offline, local weights).  Runs in the TRELLIS venv (TRELLIS_PYTHON in .env).

python trellis_run.py --inputs RUN/inputs --out RUN/trellis [--seed 42 --steps 12]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))  # offline_encoder
from common import paths  # noqa: E402

MODEL_ROOT = Path(paths.get("TRELLIS_ROOT"))  # holds TRELLIS/, cache/, trellis_weights.json, cuda-12.4/


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--ss-cfg", type=float, default=7.5)
    ap.add_argument("--slat-cfg", type=float, default=3.0)
    ap.add_argument("--formats", default="mesh,gaussian")
    a = ap.parse_args()
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    cache = MODEL_ROOT / "cache"
    os.environ.update(HF_HUB_CACHE=str(cache / "huggingface"), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      TORCH_HOME=str(cache / "torch"), XDG_CACHE_HOME=str(cache), ATTN_BACKEND="xformers",
                      SPARSE_ATTN_BACKEND="xformers", SPCONV_ALGO="native")
    os.environ.setdefault("CUDA_HOME", str(MODEL_ROOT / "cuda-12.4"))
    import numpy as np
    import torch
    import trimesh
    from PIL import Image
    sys.path.insert(0, str(MODEL_ROOT / "TRELLIS"))
    from trellis import models
    from trellis.pipelines import TrellisImageTo3DPipeline, samplers
    from offline_encoder import load_local_encoder

    weights = json.loads((MODEL_ROOT / "trellis_weights.json").read_text())
    snap = Path(weights["snapshot"])
    conf = json.loads((snap / "pipeline.json").read_text())["args"]
    formats = a.formats.split(",")
    names = ["sparse_structure_decoder", "sparse_structure_flow_model", "slat_decoder_mesh", "slat_flow_model"]
    if "gaussian" in formats:
        names.append("slat_decoder_gs")
    loaded = {n: models.from_pretrained(str(snap / conf["models"][n])) for n in names}
    info = {}

    class Pipe(TrellisImageTo3DPipeline):
        def _init_image_cond_model(self, name):
            from torchvision import transforms
            enc, prov = load_local_encoder(MODEL_ROOT, name)
            self.models["image_cond_model"] = enc
            self.image_cond_model_transform = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            info["image_encoder"] = prov

    pipe = Pipe(models=loaded,
                sparse_structure_sampler=getattr(samplers, conf["sparse_structure_sampler"]["name"])(**conf["sparse_structure_sampler"]["args"]),
                slat_sampler=getattr(samplers, conf["slat_sampler"]["name"])(**conf["slat_sampler"]["args"]),
                slat_normalization=conf["slat_normalization"], image_cond_model=conf["image_cond_model"])
    pipe.sparse_structure_sampler_params = conf["sparse_structure_sampler"]["params"]
    pipe.slat_sampler_params = conf["slat_sampler"]["params"]
    pipe.cuda()
    orig = pipe.decode_slat

    def decode(self, slat, fmts):
        idle = [k for k in self.models if not k.startswith("slat_decoder")]
        for k in idle:
            self.models[k].cpu()
        torch.cuda.empty_cache()
        try:
            return orig(slat, fmts)
        finally:
            for k in idle:
                self.models[k].cuda()

    pipe.decode_slat = types.MethodType(decode, pipe)
    files = sorted(p for p in a.inputs.iterdir() if p.suffix == ".png")
    order = {"front": 0, "left": 1, "back": 2, "right": 3}
    files.sort(key=lambda p: order.get(p.stem, 9))
    images = []
    for f in files:
        im = pipe.preprocess_image(Image.open(f).convert("RGBA"))
        im.save(out / f"{f.stem}_model_input.png")
        images.append(im)
    params = {"num_samples": 1, "seed": a.seed,
              "sparse_structure_sampler_params": {"steps": a.steps, "cfg_strength": a.ss_cfg},
              "slat_sampler_params": {"steps": a.steps, "cfg_strength": a.slat_cfg},
              "formats": formats, "preprocess_image": False, "mode": "stochastic"}
    t = time.perf_counter()
    with torch.inference_mode():
        res = pipe.run_multi_image(images, **params)
    sec = time.perf_counter() - t
    raw = res["mesh"][0]
    mesh = trimesh.Trimesh(raw.vertices.cpu().numpy(), raw.faces.cpu().numpy(), process=False)
    # vertex colours from the mesh decoder (if present) make the CAD preview nicer
    if getattr(raw, "vertex_attrs", None) is not None and raw.vertex_attrs.shape[1] >= 3:
        col = (raw.vertex_attrs[:, :3].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        mesh.visual.vertex_colors = np.c_[col, np.full(len(col), 255, np.uint8)]
    mesh.export(out / "mesh_native.ply")
    mesh.export(out / "mesh_native.glb")
    if "gaussian" in res:
        res["gaussian"][0].save_ply(str(out / "gaussians_native.ply"))
    result = {"status": "complete", "model": "microsoft/TRELLIS-image-large (v1)", "weights_revision": weights["revision"],
              "inputs": [{"file": f.name, "sha256": sha(f)} for f in files], "parameters": params,
              "inference_seconds": sec, "vertices": len(mesh.vertices), "triangles": len(mesh.faces),
              "bounds_native": mesh.bounds.tolist(), "native_up_axis": "Z", "metric_scale": False,
              "cameras_supplied_to_model": False, **info}
    (out / "result.json").write_text(json.dumps(result, indent=1))
    print(f"[trellis] {len(mesh.faces)} triangles in {sec:.1f}s", flush=True)


if __name__ == "__main__":
    main()
