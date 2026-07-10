"""
Run this ONCE after downloading the Montgomery dataset.
It combines left + right lung masks into one mask per image,
then copies everything into the correct data/ folders.

Usage:
    python prepare_data.py --dataset_path "path/to/NLM-MontgomeryCXRSet"
"""

import os
import argparse
import shutil
import numpy as np
from PIL import Image


def combine_masks(left_mask_path, right_mask_path):
    """Combine left and right lung masks into one binary mask."""
    left  = np.array(Image.open(left_mask_path).convert("L"))
    right = np.array(Image.open(right_mask_path).convert("L"))
    combined = np.clip(left + right, 0, 255).astype(np.uint8)
    return Image.fromarray(combined)


def prepare(dataset_path, output_image_dir, output_mask_dir):
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_mask_dir,  exist_ok=True)

    image_dir      = os.path.join(dataset_path, "CXR_png")
    left_mask_dir  = os.path.join(dataset_path, "ManualMask", "leftMask")
    right_mask_dir = os.path.join(dataset_path, "ManualMask", "rightMask")

    images = sorted([f for f in os.listdir(image_dir) if f.endswith(".png")])
    print(f"Found {len(images)} images. Processing...")

    ok, skipped = 0, 0
    for fname in images:
        left_path  = os.path.join(left_mask_dir,  fname)
        right_path = os.path.join(right_mask_dir, fname)

        if not os.path.exists(left_path) or not os.path.exists(right_path):
            print(f"  SKIP {fname} — mask not found")
            skipped += 1
            continue

        # Copy image
        shutil.copy(os.path.join(image_dir, fname),
                    os.path.join(output_image_dir, fname))

        # Combine and save mask
        mask = combine_masks(left_path, right_path)
        mask.save(os.path.join(output_mask_dir, fname))

        ok += 1
        if ok % 20 == 0:
            print(f"  Processed {ok}/{len(images)}...")

    print(f"\nDone! {ok} image+mask pairs ready. {skipped} skipped.")
    print(f"Images → {output_image_dir}")
    print(f"Masks  → {output_mask_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True,
                        help="Path to the unzipped NLM-MontgomeryCXRSet folder")
    args = parser.parse_args()

    prepare(
        dataset_path     = args.dataset_path,
        output_image_dir = "data/images",
        output_mask_dir  = "data/masks"
    )
