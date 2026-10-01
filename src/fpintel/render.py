"""Loading uploaded drawings (image or PDF) and drawing result overlays."""
from __future__ import annotations

import cv2
import numpy as np

from .classes import CLASS_NAMES, PALETTE
from .inference import Predictor
from .postprocess import extract_rooms


def load_drawing(data: bytes, filename: str, dpi: int = 150) -> np.ndarray:
    """Return an RGB uint8 image. PDFs are rasterised (first page)."""
    if filename.lower().endswith(".pdf"):
        try:
            import pymupdf
        except ImportError:  # older PyMuPDF versions
            import fitz as pymupdf

        with pymupdf.open(stream=data, filetype="pdf") as doc:
            pix = doc[0].get_pixmap(dpi=dpi, alpha=False)
            img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)
            return np.ascontiguousarray(img[:, :, :3])
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def analyse(predictor: Predictor, image_rgb: np.ndarray, mc_samples: int = 8,
            review_threshold: float = 0.8) -> tuple[dict, np.ndarray]:
    """Full pipeline: returns (structured result, label map at original resolution)."""
    pred = predictor.predict(image_rgb, mc_samples=mc_samples)
    label_map = pred.probs.argmax(0).astype(np.uint8)
    result = extract_rooms(label_map, pred.probs, pred.entropy, scale=pred.scale,
                           review_threshold=review_threshold)
    result["model"] = {"temperature": predictor.temperature, "mc_samples": mc_samples}
    h, w = image_rgb.shape[:2]
    full_labels = cv2.resize(label_map, (w, h), interpolation=cv2.INTER_NEAREST)
    return result, full_labels


def draw_overlay(image_rgb: np.ndarray, label_map: np.ndarray, result: dict, alpha: float = 0.45) -> np.ndarray:
    colour = np.array(PALETTE, np.uint8)[label_map]
    out = cv2.addWeighted(image_rgb, 1 - alpha, colour, alpha, 0)
    thick = max(2, round(max(out.shape[:2]) / 400))
    for room in result["rooms"]:
        pts = np.array(room["polygon"], np.int32)
        if len(pts) < 3:
            continue
        flagged = room.get("needs_review", False)
        cv2.polylines(out, [pts], True, (220, 30, 30) if flagged else (30, 120, 30), thick)
        cx, cy = pts.mean(0).astype(int)
        label = f"#{room['id']} {room['type']}"
        if "confidence" in room:
            label += f" {room['confidence']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        x, y = int(cx) - tw // 2, int(cy) + th // 2
        cv2.rectangle(out, (x - 3, y - th - 3), (x + tw + 3, y + 4), (255, 255, 255), -1)
        cv2.putText(out, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return out


__all__ = ["load_drawing", "analyse", "draw_overlay", "CLASS_NAMES"]
