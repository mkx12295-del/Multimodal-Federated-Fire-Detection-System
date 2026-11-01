"""Train the YOLO vision model on the held-out splits and evaluate on the test split."""
import argparse
import json
import shutil
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--name", default="fire_detection_v2")
    args = parser.parse_args()

    data = str(ROOT / "data.yaml")
    if not (ROOT / "splits" / "train.txt").exists():
        raise SystemExit("Run `python scripts/make_splits.py` first.")

    # A bare name such as yolo11n.pt is downloaded by Ultralytics if not present.
    local = ROOT / args.model
    model = YOLO(str(local) if local.exists() else args.model)
    model.train(
        data=data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(ROOT / "runs" / "detect"),
        name=args.name,
        exist_ok=True,
        workers=0,  # avoid multiprocessing spawn issues on Windows
        seed=42,
    )

    best = ROOT / "runs" / "detect" / args.name / "weights" / "best.pt"
    metrics = YOLO(str(best)).val(data=data, split="test", imgsz=args.imgsz, device=args.device,
                                  project=str(ROOT / "runs" / "detect"), name=f"{args.name}_test",
                                  exist_ok=True)
    summary = {
        "weights": str(best.relative_to(ROOT).as_posix()),
        "split": "test",
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
        "per_class_mAP50": {metrics.names[int(c)]: float(m) for c, m in
                            zip(metrics.box.ap_class_index, metrics.box.ap50)},
    }
    # Copy to the path the node and benchmark load (config/default.yaml).
    deploy = ROOT / "models" / "vision_yolo11n.pt"
    deploy.parent.mkdir(exist_ok=True)
    shutil.copy2(best, deploy)

    out = ROOT / "results" / "vision_test_metrics.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
