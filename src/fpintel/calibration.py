"""Temperature scaling (Guo et al., 2017) and expected calibration error for pixel classification."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .classes import IGNORE_INDEX


@torch.no_grad()
def collect_logits(model, loader, device: str, max_pixels_per_image: int = 20000, seed: int = 0):
    """Run the model over a loader and return a random sample of (logits, labels) pixels."""
    gen = torch.Generator().manual_seed(seed)
    model.eval()
    all_logits, all_labels = [], []
    for images, masks in loader:
        logits = model(images.to(device)).float().cpu()          # B,C,H,W
        c = logits.shape[1]
        logits = logits.permute(0, 2, 3, 1).reshape(-1, c)
        labels = masks.reshape(-1)
        valid = labels != IGNORE_INDEX
        logits, labels = logits[valid], labels[valid]
        n = min(max_pixels_per_image * images.shape[0], labels.numel())
        idx = torch.randperm(labels.numel(), generator=gen)[:n]
        all_logits.append(logits[idx])
        all_labels.append(labels[idx])
    return torch.cat(all_logits), torch.cat(all_labels)


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 200) -> float:
    """Find T > 0 minimising NLL of softmax(logits / T) on held-out data."""
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().item())


def reliability(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15):
    """Returns ECE and per-bin (confidence, accuracy, count) for a reliability diagram."""
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins, ece = [], 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (conf > lo) & (conf <= hi)
        count = int(in_bin.sum())
        if count:
            acc, avg_conf = correct[in_bin].mean(), conf[in_bin].mean()
            ece += count / len(conf) * abs(acc - avg_conf)
            bins.append((float(avg_conf), float(acc), count))
        else:
            bins.append((float((lo + hi) / 2), float("nan"), 0))
    return float(ece), bins


def nll(logits: torch.Tensor, labels: torch.Tensor, temperature: float = 1.0) -> float:
    return float(F.cross_entropy(logits / temperature, labels).item())
