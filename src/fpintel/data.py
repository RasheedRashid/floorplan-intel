"""Dataset for prepared floor-plan image / label-mask pairs.

Expected layout (produced by scripts/prepare_cubicasa.py or scripts/make_synthetic.py):

    <root>/<split>/images/<id>.png   RGB floor-plan image
    <root>/<split>/masks/<id>.png    uint8 label map (values = class ids, 255 = ignore)
"""
from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .classes import IGNORE_INDEX

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def normalise(image_rgb: np.ndarray) -> torch.Tensor:
    """HxWx3 uint8 RGB -> 3xHxW float tensor, ImageNet-normalised."""
    x = image_rgb.astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    return torch.from_numpy(x.transpose(2, 0, 1).copy())


def pad_to_multiple(image: np.ndarray, mask: np.ndarray | None, multiple: int = 32):
    """Pad bottom/right so both sides are divisible by `multiple` (needed by U-Net)."""
    h, w = image.shape[:2]
    ph = (multiple - h % multiple) % multiple
    pw = (multiple - w % multiple) % multiple
    if ph or pw:
        image = cv2.copyMakeBorder(image, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        if mask is not None:
            mask = cv2.copyMakeBorder(mask, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=IGNORE_INDEX)
    return image, mask


def resize_long_side(image: np.ndarray, max_side: int, mask: np.ndarray | None = None):
    """Downscale so the longest side is <= max_side. Returns (image, mask, scale)."""
    h, w = image.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        size = (round(w * scale), round(h * scale))
        image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        if mask is not None:
            mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
    return image, mask, scale


class FloorPlanDataset(Dataset):
    def __init__(self, root: str | Path, split: str, train: bool = False,
                 crop_size: int = 512, max_side: int = 1024):
        self.dir = Path(root) / split
        self.ids = sorted(p.stem for p in (self.dir / "images").glob("*.png"))
        if not self.ids:
            raise FileNotFoundError(f"No images found in {self.dir / 'images'}")
        self.train = train
        self.crop_size = crop_size
        self.max_side = max_side

    def __len__(self) -> int:
        return len(self.ids)

    def _load(self, idx: int):
        sid = self.ids[idx]
        image = cv2.cvtColor(cv2.imread(str(self.dir / "images" / f"{sid}.png")), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(self.dir / "masks" / f"{sid}.png"), cv2.IMREAD_GRAYSCALE)
        return image, mask

    def _random_crop(self, image, mask):
        c = self.crop_size
        h, w = mask.shape
        ph, pw = max(0, c - h), max(0, c - w)
        if ph or pw:
            image = cv2.copyMakeBorder(image, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=(255, 255, 255))
            mask = cv2.copyMakeBorder(mask, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=IGNORE_INDEX)
            h, w = mask.shape
        y = random.randint(0, h - c)
        x = random.randint(0, w - c)
        return image[y:y + c, x:x + c], mask[y:y + c, x:x + c]

    def __getitem__(self, idx: int):
        image, mask = self._load(idx)
        image, mask, _ = resize_long_side(image, self.max_side, mask)
        if self.train:
            image, mask = self._random_crop(image, mask)
            # Floor plans have no canonical orientation, so flips/rotations are safe.
            if random.random() < 0.5:
                image, mask = image[:, ::-1], mask[:, ::-1]
            if random.random() < 0.5:
                image, mask = image[::-1], mask[::-1]
            k = random.randint(0, 3)
            image, mask = np.rot90(image, k), np.rot90(mask, k)
        else:
            image, mask = pad_to_multiple(image, mask)
        return normalise(np.ascontiguousarray(image)), torch.from_numpy(np.ascontiguousarray(mask)).long()
