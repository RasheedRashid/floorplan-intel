"""Generate simple synthetic floor plans to smoke-test the whole pipeline end to end.

These are NOT a substitute for real data: they only check that training, calibration,
evaluation and the apps run. Results on them mean nothing.

    python scripts/make_synthetic.py --out data/synthetic
"""
import argparse
import random
from pathlib import Path

import cv2
import numpy as np

LABELS = {3: "KITCHEN", 4: "LIVING", 5: "BED", 6: "BATH", 7: "ENTRY", 9: "STORE"}


def split_rooms(x0, y0, x1, y1, min_size, depth=0):
    w, h = x1 - x0, y1 - y0
    if depth > 3 or (w < 2 * min_size and h < 2 * min_size) or (depth > 1 and random.random() < 0.3):
        return [(x0, y0, x1, y1)]
    if w >= h and w >= 2 * min_size:
        s = random.randint(x0 + min_size, x1 - min_size)
        return split_rooms(x0, y0, s, y1, min_size, depth + 1) + split_rooms(s, y0, x1, y1, min_size, depth + 1)
    if h >= 2 * min_size:
        s = random.randint(y0 + min_size, y1 - min_size)
        return split_rooms(x0, y0, x1, s, min_size, depth + 1) + split_rooms(x0, s, x1, y1, min_size, depth + 1)
    return [(x0, y0, x1, y1)]


def make_plan(rng: random.Random):
    random.seed(rng.random())
    H, W = random.randint(480, 720), random.randint(560, 900)
    image = np.full((H, W, 3), 255, np.uint8)
    mask = np.zeros((H, W), np.uint8)
    m = 40
    bx0, by0 = random.randint(m, 80), random.randint(m, 80)
    bx1, by1 = W - random.randint(m, 80), H - random.randint(m, 80)
    for x0, y0, x1, y1 in split_rooms(bx0, by0, bx1, by1, min_size=110):
        cls = random.choice(list(LABELS))
        mask[y0:y1, x0:x1] = cls
        cv2.rectangle(mask, (x0, y0), (x1, y1), 2, 6)
        cv2.rectangle(image, (x0, y0), (x1, y1), (0, 0, 0), 6)
        text = LABELS[cls]
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cx, cy = (x0 + x1) // 2 - tw // 2, (y0 + y1) // 2 + th // 2
        cv2.putText(image, text, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.rectangle(mask, (bx0, by0), (bx1, by1), 2, 10)
    cv2.rectangle(image, (bx0, by0), (bx1, by1), (0, 0, 0), 10)
    noise = np.random.default_rng(rng.randint(0, 10**9)).normal(0, 8, image.shape)
    image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return image, mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--n", type=int, nargs=3, default=[60, 15, 15], metavar=("TRAIN", "VAL", "TEST"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    for split, n in zip(["train", "val", "test"], args.n):
        (Path(args.out) / split / "images").mkdir(parents=True, exist_ok=True)
        (Path(args.out) / split / "masks").mkdir(parents=True, exist_ok=True)
        for i in range(n):
            image, mask = make_plan(rng)
            cv2.imwrite(str(Path(args.out) / split / "images" / f"{i:04d}.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(Path(args.out) / split / "masks" / f"{i:04d}.png"), mask)
    print(f"Wrote synthetic dataset to {args.out}")


if __name__ == "__main__":
    main()
