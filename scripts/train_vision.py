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
    parser.add_argument("--workers", type=int, default=0,
                        help="dataloader workers (0 avoids spawn issues on Windows CPU runs)")
    parser.add_argument("--patience", type=int, default=15, help="early-stopping patience")
    parser.add_argument("--resume", action="store_true", help="continue an interrupted run")
    parser.add_argument("--test-only", action="store_true", help="skip training")
    args = parser.parse_args()

    data = str(ROOT / "data.yaml")
    if not (ROOT / "splits" / "train.txt").exists():
        raise SystemExit("Run `python scripts/make_splits.py` first.")

    run_dir = ROOT / "runs" / "detect" / args.name
    last = run_dir / "weights" / "last.pt"
    if args.test_only:
        pass
    elif args.resume and last.exists():
        YOLO(str(last)).train(resume=True)
    else:
        YOLO(str(ROOT / args.model)).train(
            data=data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            project=str(ROOT / "runs" / "detect"),
            name=args.name,
            exist_ok=True,
            workers=args.workers,
            patience=args.patience,
            seed=42,
        )

    best = run_dir / "weights" / "best.pt"
    deploy = ROOT / "models" / "vision_yolo11n.pt"
    deploy.parent.mkdir(exist_ok=True)
    shutil.copy2(best, deploy)
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
    out = ROOT / "results" / "vision_test_metrics.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
