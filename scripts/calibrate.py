"""Fit a temperature on the validation split and save it next to the checkpoint.

    python scripts/calibrate.py --data data/processed --checkpoint checkpoints/best.pth
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fpintel.calibration import collect_logits, fit_temperature, nll, reliability  # noqa: E402
from fpintel.data import FloorPlanDataset  # noqa: E402
from fpintel.model import load_checkpoint  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--checkpoint", default="checkpoints/best.pth")
    ap.add_argument("--out", default=None, help="defaults to <checkpoint dir>/calibration.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ckpt = load_checkpoint(args.checkpoint, device)
    ds = FloorPlanDataset(args.data, "val", train=False, max_side=ckpt["config"].get("max_side", 1024))
    logits, labels = collect_logits(model, DataLoader(ds, batch_size=1), device)

    t = fit_temperature(logits, labels)
    before, _ = reliability(torch.softmax(logits, 1).numpy(), labels.numpy())
    after, _ = reliability(torch.softmax(logits / t, 1).numpy(), labels.numpy())
    result = {"temperature": t, "fitted_on": "val", "pixels": int(labels.numel()),
              "val_ece_before": before, "val_ece_after": after,
              "val_nll_before": nll(logits, labels), "val_nll_after": nll(logits, labels, t)}

    out = Path(args.out) if args.out else Path(args.checkpoint).parent / "calibration.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"Saved to {out}. Report calibration on the TEST split with scripts/evaluate.py.")


if __name__ == "__main__":
    main()
