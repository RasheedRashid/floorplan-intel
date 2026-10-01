"""Evaluate on the test split: segmentation IoU, room counting, and calibration.

    python scripts/evaluate.py --data data/processed --checkpoint checkpoints/best.pth
Writes results/metrics.json and results/reliability.png.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fpintel.calibration import collect_logits, nll, reliability  # noqa: E402
from fpintel.classes import CLASS_NAMES, IGNORE_INDEX, NUM_CLASSES  # noqa: E402
from fpintel.data import FloorPlanDataset  # noqa: E402
from fpintel.metrics import iou_from_confusion, update_confusion  # noqa: E402
from fpintel.model import load_checkpoint  # noqa: E402
from fpintel.postprocess import extract_rooms  # noqa: E402


def plot_reliability(bins_before, bins_after, ece_b, ece_a, path):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="grey", label="perfect calibration")
    for bins, label in [(bins_before, f"before (ECE {ece_b:.3f})"), (bins_after, f"after T-scaling (ECE {ece_a:.3f})")]:
        pts = [(c, a) for c, a, n in bins if n > 0]
        ax.plot(*zip(*pts), "o-", label=label)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title("Pixel-level reliability (test split)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--checkpoint", default="checkpoints/best.pth")
    ap.add_argument("--calibration", default=None, help="defaults to <checkpoint dir>/calibration.json")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ckpt = load_checkpoint(args.checkpoint, device)
    cal_path = Path(args.calibration) if args.calibration else Path(args.checkpoint).parent / "calibration.json"
    t = json.loads(cal_path.read_text())["temperature"] if cal_path.exists() else 1.0
    ds = FloorPlanDataset(args.data, "test", train=False, max_side=ckpt["config"].get("max_side", 1024))
    dl = DataLoader(ds, batch_size=1)

    conf = np.zeros((NUM_CLASSES, NUM_CLASSES), np.int64)
    count_errors, exact = [], 0
    with torch.no_grad():
        for images, masks in dl:
            pred = model(images.to(device)).argmax(1)[0].cpu().numpy()
            gt = masks[0].numpy()
            conf = update_confusion(conf, pred[None], gt[None])
            valid = gt != IGNORE_INDEX
            n_pred = extract_rooms(np.where(valid, pred, 0))["summary"]["room_count"]
            n_gt = extract_rooms(np.where(valid, gt, 0))["summary"]["room_count"]
            count_errors.append(abs(n_pred - n_gt))
            exact += int(n_pred == n_gt)
    miou, ious = iou_from_confusion(conf)

    logits, labels = collect_logits(model, dl, device)
    ece_b, bins_b = reliability(torch.softmax(logits, 1).numpy(), labels.numpy())
    ece_a, bins_a = reliability(torch.softmax(logits / t, 1).numpy(), labels.numpy())

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    metrics = {
        "test_images": len(ds),
        "mIoU": round(miou, 4),
        "per_class_IoU": {n: (None if np.isnan(v) else round(float(v), 4)) for n, v in zip(CLASS_NAMES, ious)},
        "room_count_exact_match": round(exact / len(ds), 4),
        "room_count_MAE": round(float(np.mean(count_errors)), 3),
        "temperature": t,
        "pixel_ECE_before": round(ece_b, 4),
        "pixel_ECE_after": round(ece_a, 4),
        "pixel_NLL_before": round(nll(logits, labels), 4),
        "pixel_NLL_after": round(nll(logits, labels, t), 4),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    plot_reliability(bins_b, bins_a, ece_b, ece_a, out / "reliability.png")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
