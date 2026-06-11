#!/usr/bin/env python3
"""
Generate train_list.txt / test_list.txt for scenes without a predefined split.
Reads the COLMAP images.bin to get all image basenames, then shuffles and splits.

Usage:
    python configs/generate_split.py /path/to/scene --ratio 0.8 --seed 42
"""

import argparse
import os
import random
import struct
from pathlib import Path


def read_image_names_from_colmap(images_bin_path: str):
    """Read image basenames from COLMAP images.bin."""
    names = []
    with open(images_bin_path, "rb") as f:
        num_reg_images = struct.unpack("Q", f.read(8))[0]
        for _ in range(num_reg_images):
            image_id = struct.unpack("I", f.read(4))[0]
            qvec = struct.unpack("dddd", f.read(32))
            tvec = struct.unpack("ddd", f.read(24))
            camera_id = struct.unpack("I", f.read(4))[0]
            name = b""
            while True:
                ch = f.read(1)
                if ch == b"\x00":
                    break
                name += ch
            name = name.decode("utf-8", errors="replace")
            basename = Path(name).stem
            num_points2D = struct.unpack("Q", f.read(8))[0]
            # Skip 2D points to avoid overflow on large values
            f.read(24 * num_points2D)
            names.append(basename)
    return names


def main():
    parser = argparse.ArgumentParser(description="Generate train/test split for COLMAP scene")
    parser.add_argument("scene_dir", type=str, help="Path to COLMAP scene directory")
    parser.add_argument("--ratio", type=float, default=0.8, help="Train ratio (default: 0.8)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    scene_dir = args.scene_dir
    images_bin = os.path.join(scene_dir, "colmap", "sparse", "0", "images.bin")
    if not os.path.exists(images_bin):
        images_bin = os.path.join(scene_dir, "sparse", "0", "images.bin")

    if not os.path.exists(images_bin):
        print(f"Error: images.bin not found at expected paths.")
        return

    names = read_image_names_from_colmap(images_bin)
    print(f"Found {len(names)} images in COLMAP reconstruction.")

    random.seed(args.seed)
    random.shuffle(names)

    split_idx = int(len(names) * args.ratio)
    train_names = sorted(names[:split_idx])
    test_names = sorted(names[split_idx:])

    train_path = os.path.join(scene_dir, "train_list.txt")
    test_path = os.path.join(scene_dir, "test_list.txt")

    with open(train_path, "w") as f:
        for n in train_names:
            f.write(n + "\n")

    with open(test_path, "w") as f:
        for n in test_names:
            f.write(n + "\n")

    print(f"Train: {len(train_names)} -> {train_path}")
    print(f"Test:  {len(test_names)} -> {test_path}")


if __name__ == "__main__":
    main()
