"""U-Net segmentation model with an EfficientNet encoder and a dropout layer for MC-dropout."""
from __future__ import annotations

from pathlib import Path

import segmentation_models_pytorch as smp
import torch
from torch import nn

from .classes import NUM_CLASSES


def build_model(encoder: str = "efficientnet-b0", encoder_weights: str | None = "imagenet",
                num_classes: int = NUM_CLASSES, dropout: float = 0.2) -> nn.Module:
    model = smp.Unet(encoder_name=encoder, encoder_weights=encoder_weights, classes=num_classes)
    # Dropout before the segmentation head; kept active at inference for MC-dropout uncertainty.
    model.segmentation_head = nn.Sequential(nn.Dropout2d(dropout), model.segmentation_head)
    return model


def enable_mc_dropout(model: nn.Module) -> None:
    """Eval mode everywhere (frozen BatchNorm) except dropout layers."""
    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            m.train()


def save_checkpoint(path: str | Path, model: nn.Module, config: dict, metrics: dict | None = None) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": config, "metrics": metrics or {}}, path)


def load_checkpoint(path: str | Path, device: str = "cpu") -> tuple[nn.Module, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = build_model(cfg["encoder"], encoder_weights=None,
                        num_classes=cfg.get("num_classes", NUM_CLASSES), dropout=cfg.get("dropout", 0.2))
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt
