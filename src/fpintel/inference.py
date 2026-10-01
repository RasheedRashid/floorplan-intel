"""Inference: calibrated, MC-dropout predictions for a single floor-plan image."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .data import normalise, pad_to_multiple, resize_long_side
from .model import enable_mc_dropout, load_checkpoint


@dataclass
class Prediction:
    probs: np.ndarray      # C,H,W mean calibrated class probabilities (at working resolution)
    entropy: np.ndarray    # H,W predictive entropy (nats)
    scale: float           # working resolution / original resolution


class Predictor:
    def __init__(self, checkpoint: str | Path, calibration: str | Path | None = None,
                 device: str | None = None, max_side: int = 1024):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, ckpt = load_checkpoint(checkpoint, self.device)
        self.max_side = max_side
        self.temperature = 1.0
        if calibration and Path(calibration).exists():
            self.temperature = float(json.loads(Path(calibration).read_text())["temperature"])

    @torch.no_grad()
    def predict(self, image_rgb: np.ndarray, mc_samples: int = 8) -> Prediction:
        image, _, scale = resize_long_side(image_rgb, self.max_side)
        h, w = image.shape[:2]
        padded, _ = pad_to_multiple(image, None)
        x = normalise(padded).unsqueeze(0).to(self.device)

        if mc_samples > 1:
            enable_mc_dropout(self.model)
        else:
            self.model.eval()
        n = max(1, mc_samples)
        probs = None
        for _ in range(n):
            p = torch.softmax(self.model(x).float() / self.temperature, dim=1)[0, :, :h, :w]
            probs = p if probs is None else probs + p
        probs = (probs / n).cpu().numpy()
        self.model.eval()

        entropy = -(probs * np.log(np.clip(probs, 1e-12, 1.0))).sum(axis=0)
        return Prediction(probs=probs, entropy=entropy, scale=scale)
