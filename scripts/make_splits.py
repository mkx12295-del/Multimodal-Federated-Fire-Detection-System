"""Create reproducible train/val/test splits for the YOLO image dataset.

The original setup validated on the training images (val: train/images), which
inflates every reported metric. This script writes image-list files that
Ultralytics reads directly, so no images need to be moved on disk.
"""
import argparse
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", default="train/images")
    parser.add_argument("--out", default="splits")
    parser.add_argument("--val", type=float, default=0.1)
    parser.add_argument("--test", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    images = sorted(Path(args.images).glob("*.jpg"))
    if not images:
        raise SystemExit(f"No images found in {args.images}")
    random.Random(args.seed).shuffle(images)

    n_val = int(len(images) * args.val)
    n_test = int(len(images) * args.test)
    splits = {
        "val": images[:n_val],
        "test": images[n_val:n_val + n_test],
        "train": images[n_val + n_test:],
    }

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    for name, files in splits.items():
        # Absolute paths: the lists are regenerated deterministically from the
        # seed on each machine, so they are not committed.
        (out / f"{name}.txt").write_text("\n".join(p.resolve().as_posix() for p in files) + "\n")
        print(f"{name:5s}: {len(files)} images")


if __name__ == "__main__":
    main()
