"""Train the U-Net floor-plan segmentation model.

    python scripts/train.py --data data/processed --epochs 40 --out checkpoints/best.pth
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fpintel.classes import CLASS_NAMES, IGNORE_INDEX, NUM_CLASSES  # noqa: E402
from fpintel.data import FloorPlanDataset  # noqa: E402
from fpintel.metrics import iou_from_confusion, update_confusion  # noqa: E402
from fpintel.model import build_model, save_checkpoint  # noqa: E402


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    conf = np.zeros((NUM_CLASSES, NUM_CLASSES), np.int64)
    for images, masks in loader:
        pred = model(images.to(device)).argmax(1).cpu().numpy()
        conf = update_confusion(conf, pred, masks.numpy())
    return iou_from_confusion(conf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--out", default="checkpoints/best.pth")
    ap.add_argument("--encoder", default="efficientnet-b0")
    ap.add_argument("--weights", default="imagenet", help="'imagenet' or 'none'")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--max-side", type=int, default=1024)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    train_ds = FloorPlanDataset(args.data, "train", train=True, crop_size=args.crop, max_side=args.max_side)
    val_ds = FloorPlanDataset(args.data, "val", train=False, max_side=args.max_side)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                          drop_last=len(train_ds) > args.batch_size, pin_memory=device == "cuda")
    val_dl = DataLoader(val_ds, batch_size=1, num_workers=args.workers)  # variable image sizes

    weights = None if args.weights.lower() == "none" else args.weights
    model = build_model(args.encoder, weights, NUM_CLASSES, args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")

    config = {"encoder": args.encoder, "num_classes": NUM_CLASSES, "dropout": args.dropout,
              "max_side": args.max_side, "class_names": CLASS_NAMES}
    best = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, total = time.time(), 0.0
        for images, masks in train_dl:
            images, masks = images.to(device), masks.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, enabled=device == "cuda"):
                loss = loss_fn(model(images), masks)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += loss.item() * images.size(0)
        sched.step()
        miou, _ = validate(model, val_dl, device)
        print(f"epoch {epoch:3d} | loss {total / len(train_ds):.4f} | val mIoU {miou:.4f} | {time.time() - t0:.0f}s")
        if miou > best:
            best = miou
            save_checkpoint(args.out, model, config, {"val_miou": miou, "epoch": epoch})
            print(f"  saved new best to {args.out}")
    print(f"Best val mIoU: {best:.4f}")


if __name__ == "__main__":
    main()
