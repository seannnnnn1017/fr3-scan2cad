"""Open-vocabulary object segmentation from a text prompt.

backend "yoloe": YOLOE-26 text-prompted detection+segmentation, then (optional) SAM 2.1
                 box-prompted refinement for crisp silhouettes (TRELLIS / carving inputs).
backend "sam3":  SAM 3 concept segmentation straight from the text prompt
                 (needs weights/sam3.pt, gated on Hugging Face).
"""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

WEIGHTS = Path(__file__).resolve().parents[1] / "weights"
os.environ.setdefault("YOLO_VERBOSE", "False")
os.environ.setdefault("YOLO_AUTOINSTALL", "False")


class Segmenter:
    """backend + optional fallback backend (used when the primary finds nothing above min_score)."""

    def __init__(self, prompt: str, backend: str = "yoloe", model: str = "yoloe-26l-seg.pt",
                 refine: str | None = "sam2.1_b.pt", conf: float = 0.05, aliases=(), fallback: str | None = "sam3",
                 min_score: float = 0.3):
        self.min_score = min_score
        self.primary = _Backend(prompt, backend, model, refine, conf, aliases)
        self.fallback = _Backend(prompt, fallback, model, None, 0.3, aliases) if fallback and fallback != backend else None
        self.name = self.primary.name + (f" → fallback {self.fallback.name}" if self.fallback else "")
        self.last_backend = None

    def segment(self, rgb, expect_uv=None, exclude=None):
        res = self.primary.segment(rgb, expect_uv, exclude)
        self.last_backend = self.primary.name
        if (res is None or res["score"] < self.min_score) and self.fallback is not None:
            alt = self.fallback.segment(rgb, expect_uv, exclude)
            if alt is not None and (res is None or alt["score"] > res["score"]):
                res, self.last_backend = alt, self.fallback.name
        if res is not None:
            res["backend"] = self.last_backend
        return res


class _Backend:
    def __init__(self, prompt, backend, model, refine, conf, aliases):
        self.prompt, self.backend, self.conf = prompt, backend, conf
        self.classes = [prompt] + [a for a in aliases if a != prompt]
        cwd = os.getcwd()
        os.chdir(WEIGHTS)  # ultralytics downloads next to the CWD
        try:
            if backend == "yoloe":
                from ultralytics import YOLOE
                self.model = YOLOE(str(WEIGHTS / model))
                self.model.set_classes(self.classes)
                self.refiner = None
                if refine:
                    from ultralytics import SAM
                    self.refiner = SAM(str(WEIGHTS / refine) if (WEIGHTS / refine).exists() else refine)
            elif backend == "sam3":
                from ultralytics.models.sam import SAM3SemanticPredictor
                self.model = SAM3SemanticPredictor(overrides=dict(conf=conf, task="segment", mode="predict",
                                                                  model=str(WEIGHTS / "sam3.pt"), half=True, verbose=False))
                self.refiner = None
            else:
                raise ValueError(backend)
        finally:
            os.chdir(cwd)
        self.name = (f"YOLOE-26 {self.classes}" + (" + SAM2.1" if refine else "")) if backend == "yoloe" else "SAM3"

    def candidates(self, rgb):
        """List of {mask(H,W bool), box xyxy, score}."""
        h, w = rgb.shape[:2]
        bgr = np.ascontiguousarray(rgb[..., ::-1])
        out = []
        if self.backend == "yoloe":
            r = self.model.predict(bgr, conf=self.conf, verbose=False, retina_masks=True)[0]
            if r.boxes is None or len(r.boxes) == 0:
                return out
            for i in range(len(r.boxes)):
                m = r.masks.data[i].cpu().numpy() > 0.5 if r.masks is not None else None
                if m is not None and m.shape != (h, w):
                    m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
                out.append({"mask": m, "box": r.boxes.xyxy[i].cpu().numpy(), "score": float(r.boxes.conf[i])})
        else:
            self.model.set_image(bgr)
            r = self.model(text=[self.prompt])[0]
            if r.masks is None:
                return out
            for i in range(len(r.masks.data)):
                m = r.masks.data[i].cpu().numpy() > 0.5
                if m.shape != (h, w):
                    m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
                box = r.boxes.xyxy[i].cpu().numpy() if r.boxes is not None else _box(m)
                score = float(r.boxes.conf[i]) if r.boxes is not None else 1.0
                out.append({"mask": m, "box": box, "score": score})
        return out

    def refine(self, rgb, box):
        if self.refiner is None:
            return None
        bgr = np.ascontiguousarray(rgb[..., ::-1])
        r = self.refiner.predict(bgr, bboxes=[box.tolist()], verbose=False)[0]
        if r.masks is None:
            return None
        m = r.masks.data[0].cpu().numpy() > 0.5
        h, w = rgb.shape[:2]
        if m.shape != (h, w):
            m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
        return m

    def segment(self, rgb, expect_uv=None, exclude=None):
        """Best instance (closest to expect_uv if given, else highest score), refined.

        exclude: bool mask of pixels that cannot be the object (e.g. AprilTag paper).
        """
        cands = self.candidates(rgb)
        if not cands:
            return None
        if expect_uv is not None:
            def dist(c):
                b = c["box"]
                return np.hypot((b[0] + b[2]) / 2 - expect_uv[0], (b[1] + b[3]) / 2 - expect_uv[1]) - 400 * c["score"]
            best = min(cands, key=dist)
        else:
            best = max(cands, key=lambda c: c["score"])
        mask = best["mask"]
        refined = self.refine(rgb, best["box"])
        if refined is not None and mask is not None:
            iou = (refined & mask).sum() / max(1, (refined | mask).sum())
            if iou > 0.5:  # accept SAM only if it agrees with the open-vocab instance
                mask = refined
        elif refined is not None:
            mask = refined
        if exclude is not None:
            mask = mask & ~exclude
        mask = clean_mask(mask)
        return {"mask": mask, "box": best["box"], "score": best["score"], "n_candidates": len(cands)}


def _box(m):
    ys, xs = np.nonzero(m)
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], float)


def clean_mask(mask):
    """Keep the largest connected component, fill holes."""
    m = mask.astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n > 2:
        m = (lab == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled = np.zeros_like(m)
    cv2.drawContours(filled, cnts, -1, 1, -1)
    # keep genuine through-holes (mug handle) visible from the original mask
    holes = (filled > 0) & (m == 0)
    nh, hl, hs, _ = cv2.connectedComponentsWithStats(holes.astype(np.uint8), 8)
    out = filled.copy()
    for i in range(1, nh):
        if hs[i, cv2.CC_STAT_AREA] > 400:
            out[hl == i] = 0
    return out > 0


def draw_mask(rgb, mask, box=None, label=None, color=(255, 0, 255)):
    vis = rgb.copy()
    if mask is not None:
        tint = vis.copy()
        tint[mask] = (0.45 * tint[mask] + 0.55 * np.array(color)).astype(np.uint8)
        vis = tint
        cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(vis, cnts, -1, color, 3)
    if box is not None:
        b = np.asarray(box).astype(int)
        cv2.rectangle(vis, tuple(b[:2]), tuple(b[2:]), (255, 255, 0), 3)
        if label:
            cv2.putText(vis, label, (b[0], max(30, b[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 3)
    return vis
