#!/usr/bin/env python3
"""
Terroir AI — AHDB/Kaggle Augmentation Script

Applies realistic farm conditions:
  - Drop brightness by 30% (shadows)
  - Motion blur (conveyor belt)
  - Random rotations

Example:
  python augment_ahdb.py --input data/ahdb --output data/ahdb_aug --num 3
"""

import os
import argparse
import shutil
import random
import numpy as np
import cv2

try:
    import albumentations as A
except ImportError as exc:
    raise SystemExit(
        "ERROR: albumentations is required. Install with: pip install albumentations"
    ) from exc

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def _build_augmenter(brightness_drop: float, blur_limit: int, rotate_limit: int):
    # brightness_drop = 0.30 means 30% darker
    return A.Compose([
        A.RandomBrightnessContrast(
            brightness_limit=(-brightness_drop, -brightness_drop),
            contrast_limit=(0.0, 0.0),
            p=1.0,
        ),
        A.MotionBlur(blur_limit=blur_limit, p=1.0),
        A.Rotate(limit=rotate_limit, border_mode=cv2.BORDER_REFLECT_101, p=1.0),
    ])


def _iter_images(root_dir: str):
    for root, _, files in os.walk(root_dir):
        for fn in files:
            if fn.lower().endswith(IMG_EXTS):
                yield root, fn


def main():
    p = argparse.ArgumentParser(description="AHDB/Kaggle dataset augmentation")
    p.add_argument("--input", required=True, help="Input dataset root (class subfolders)")
    p.add_argument("--output", required=True, help="Output dataset root")
    p.add_argument("--num", type=int, default=3, help="Augmented copies per image")
    p.add_argument("--brightness-drop", type=float, default=0.30, help="Brightness drop fraction")
    p.add_argument("--blur-limit", type=int, default=9, help="Motion blur kernel size (odd int)")
    p.add_argument("--rotate", type=int, default=15, help="Random rotation limit (degrees)")
    p.add_argument("--copy-originals", action="store_true", help="Copy originals into output")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    aug = _build_augmenter(args.brightness_drop, args.blur_limit, args.rotate)
    in_root = os.path.abspath(args.input)
    out_root = os.path.abspath(args.output)

    if not os.path.exists(in_root):
        raise SystemExit(f"ERROR: input not found: {in_root}")

    os.makedirs(out_root, exist_ok=True)

    n_in = 0
    n_out = 0
    for root, fn in _iter_images(in_root):
        n_in += 1
        src = os.path.join(root, fn)
        rel_dir = os.path.relpath(root, in_root)
        dst_dir = os.path.join(out_root, rel_dir)
        os.makedirs(dst_dir, exist_ok=True)

        if args.copy_originals:
            shutil.copy2(src, os.path.join(dst_dir, fn))

        img = cv2.imread(src)
        if img is None:
            continue

        stem, ext = os.path.splitext(fn)
        for i in range(args.num):
            out = aug(image=img)["image"]
            out_name = f"{stem}_aug{i+1:02d}{ext}"
            cv2.imwrite(os.path.join(dst_dir, out_name), out)
            n_out += 1

    print(f"Done. Images read: {n_in} | Augmented: {n_out}")


if __name__ == "__main__":
    main()
