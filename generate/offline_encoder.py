"""DINOv2 encoder for TRELLIS: local architecture and local state dict, never a Torch Hub download.
(Vendored from pipelines/isaac_trellis_mv4/offline_encoder.py.)"""
import hashlib
from pathlib import Path


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_local_encoder(root, name):
    if name != "dinov2_vitl14_reg":
        raise ValueError("Unsupported cached encoder: " + name)
    hub = Path(root) / "cache/torch/hub"
    repo = hub / "facebookresearch_dinov2_main"
    weight = hub / "checkpoints/dinov2_vitl14_reg4_pretrain.pth"
    for path in [repo / "hubconf.py", weight]:
        if not path.is_file():
            raise FileNotFoundError("Missing local encoder cache: " + str(path))
    import torch
    model = torch.hub.load(str(repo), name, source="local", pretrained=False)
    state = torch.load(weight, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    code_files = sorted(repo.rglob("*.py"))
    provenance = {"name": name, "weight": str(weight), "weight_sha256": sha256(weight),
                  "architecture_source": "local cached DINOv2 repository; no network loading",
                  "code_sha256": {str(p.relative_to(repo)): sha256(p) for p in code_files}}
    return model, provenance
