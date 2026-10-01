"""Convert CubiCasa5K (SVG annotations) into simple image / label-mask PNG pairs.

This uses the SVG parser from the official CubiCasa5K code repository, so clone it first:

    git clone https://github.com/CubiCasa/CubiCasa5k.git external/CubiCasa5k

Then download and extract the dataset (see the CubiCasa5k README for the link and licence),
so that you have <cubicasa_root>/train.txt, val.txt, test.txt and the sample folders.

    python scripts/prepare_cubicasa.py --cubicasa-root data/cubicasa5k \
        --repo external/CubiCasa5k --out data/processed

If the parser import fails, check external/CubiCasa5k/floortrans/loaders/svg_loader.py:
this script mirrors how that loader builds its label tensors.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fpintel.data import resize_long_side  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cubicasa-root", required=True, help="folder containing train.txt / val.txt / test.txt")
    ap.add_argument("--repo", required=True, help="path to a clone of github.com/CubiCasa/CubiCasa5k")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--max-side", type=int, default=1024, help="downscale longest side to save disk and memory")
    ap.add_argument("--limit", type=int, default=0, help="only convert the first N samples per split (0 = all)")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(args.repo).resolve()))
    try:
        from floortrans.loaders.house import House
    except ImportError as e:
        sys.exit(f"Could not import the CubiCasa5K parser from {args.repo}: {e}")

    root = Path(args.cubicasa_root)
    for split in ["train", "val", "test"]:
        folders = [ln.strip().strip("/") for ln in (root / f"{split}.txt").read_text().splitlines() if ln.strip()]
        if args.limit:
            folders = folders[: args.limit]
        out_img = Path(args.out) / split / "images"
        out_mask = Path(args.out) / split / "masks"
        out_img.mkdir(parents=True, exist_ok=True)
        out_mask.mkdir(parents=True, exist_ok=True)
        ok = failed = 0
        for folder in folders:
            sample = root / folder
            sid = folder.replace("/", "_")
            try:
                image = cv2.cvtColor(cv2.imread(str(sample / "F1_scaled.png")), cv2.COLOR_BGR2RGB)
                h, w = image.shape[:2]
                house = House(str(sample / "model.svg"), h, w)
                rooms = house.get_segmentation_tensor()[0].astype(np.uint8)  # channel 0 = walls/rooms
                image, rooms, _ = resize_long_side(image, args.max_side, rooms)
                cv2.imwrite(str(out_img / f"{sid}.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(out_mask / f"{sid}.png"), rooms)
                ok += 1
            except Exception as e:  # a few samples in public datasets are usually malformed
                failed += 1
                print(f"  skipped {folder}: {e}")
        print(f"{split}: {ok} converted, {failed} skipped")


if __name__ == "__main__":
    main()
