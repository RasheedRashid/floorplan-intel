"""Evaluation metrics: confusion matrix, IoU, room counting."""
from __future__ import annotations

import numpy as np

from .classes import IGNORE_INDEX, NUM_CLASSES


def update_confusion(conf: np.ndarray, pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    valid = target != IGNORE_INDEX
    idx = NUM_CLASSES * target[valid].astype(np.int64) + pred[valid].astype(np.int64)
    conf += np.bincount(idx, minlength=NUM_CLASSES**2).reshape(NUM_CLASSES, NUM_CLASSES)
    return conf


def iou_from_confusion(conf: np.ndarray) -> tuple[float, np.ndarray]:
    tp = np.diag(conf).astype(np.float64)
    denom = conf.sum(0) + conf.sum(1) - tp
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(denom > 0, tp / denom, np.nan)
    return float(np.nanmean(iou)), iou
