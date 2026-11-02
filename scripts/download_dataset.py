"""Download the Roboflow fire dataset (CC BY 4.0) into train/images and train/labels.

Needs a free Roboflow account: set ROBOFLOW_API_KEY, then

    pip install roboflow
    python scripts/download_dataset.py

Alternatively download the "YOLOv11" export by hand from
https://universe.roboflow.com/fire-dataset-tp9jt/fire-detection-sejra/dataset/1
and unzip it into a folder passed with --from-dir.

Every image of every exported split goes into train/; scripts/make_splits.py
then creates the seeded train/val/test split used by this project.
"""
import argparse
import os
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORKSPACE, PROJECT, VERSION = "fire-dataset-tp9jt", "fire-detection-sejra", 1


def fetch(dest):
    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        raise SystemExit("Set ROBOFLOW_API_KEY, or download manually and use --from-dir.")
    from roboflow import Roboflow

    project = Roboflow(api_key=key).workspace(WORKSPACE).project(PROJECT)
    project.version(VERSION).download("yolov11", location=str(dest), overwrite=True)


def collect(src):
    out_img, out_lbl = ROOT / "train" / "images", ROOT / "train" / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)
    n = 0
    for img_dir in Path(src).rglob("images"):
        for img in img_dir.iterdir():
            label = img_dir.parent / "labels" / f"{img.stem}.txt"
            if not label.exists():
                continue
            shutil.copy2(img, out_img / img.name)
            shutil.copy2(label, out_lbl / label.name)
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from-dir", help="an already downloaded and unzipped export")
    args = ap.parse_args()

    if args.from_dir:
        n = collect(args.from_dir)
    else:
        with tempfile.TemporaryDirectory() as tmp:
            fetch(tmp)
            n = collect(tmp)
    print(f"{n} labelled images in train/ (expected 8939). "
          f"Next: python scripts/make_splits.py")


if __name__ == "__main__":
    main()
