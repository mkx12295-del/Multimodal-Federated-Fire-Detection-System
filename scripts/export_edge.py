"""Export the vision model to edge formats and measure size, CPU latency and accuracy.

Formats: PyTorch (reference), ONNX fp32, NCNN fp32, NCNN fp16 (the format
Ultralytics recommends for Raspberry Pi) and OpenVINO INT8 (calibrated on the
validation split). The process pins itself to --threads CPU cores so desktop
numbers approximate a 4-core Pi 5; run the same script on the Pi for real
Pi latency:

    python scripts/export_edge.py --threads 4
"""
import os

# CPU-only benchmark: hide any GPU before torch loads, so every format runs on
# the same cores and Ultralytics does not try to install GPU runtimes.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import statistics  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import psutil  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

FORMATS = [
    ("pytorch_fp32", None, {}),
    ("onnx_fp32", "onnx", {}),
    ("ncnn_fp32", "ncnn", {}),
    ("ncnn_fp16", "ncnn", {"half": True}),
    ("openvino_int8", "openvino", {"int8": True}),
]


def limit_onnxruntime_threads(n):
    """onnxruntime starts one thread per core by default; cap it like the other runtimes."""
    import onnxruntime as ort

    original = ort.InferenceSession.__init__

    def init(self, path_or_bytes, sess_options=None, *a, **kw):
        sess_options = sess_options or ort.SessionOptions()
        sess_options.intra_op_num_threads = n
        sess_options.inter_op_num_threads = 1
        original(self, path_or_bytes, sess_options, *a, **kw)

    ort.InferenceSession.__init__ = init


def size_mb(path):
    p = Path(path)
    total = sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.is_dir() else p.stat().st_size
    return round(total / 2 ** 20, 2)


def export(weights, fmt, kw, imgsz, tag, out_root):
    """Export into <out_root>/<tag> so variants of one format do not overwrite each other."""
    import shutil

    from ultralytics import YOLO

    out = Path(out_root) / tag
    if out.exists():
        return next(out.iterdir())
    out.mkdir(parents=True)
    src = out / Path(weights).name
    shutil.copy2(weights, src)
    if kw.get("int8"):
        kw = dict(kw, data=str(ROOT / "data.yaml"))  # calibration images
    path = Path(YOLO(str(src)).export(format=fmt, imgsz=imgsz, device="cpu", **kw))
    src.unlink()
    return path


def latency(model_path, image, imgsz, runs, warmup=5):
    from ultralytics import YOLO

    m = YOLO(str(model_path), task="detect")
    for _ in range(warmup):
        m.predict(image, imgsz=imgsz, device="cpu", verbose=False)
    times = []
    for _ in range(runs):
        t = time.perf_counter()
        m.predict(image, imgsz=imgsz, device="cpu", verbose=False)
        times.append((time.perf_counter() - t) * 1000)
    return round(statistics.median(times), 1), round(sorted(times)[int(0.95 * len(times)) - 1], 1)


def accuracy(model_path, imgsz):
    from ultralytics import YOLO

    m = YOLO(str(model_path), task="detect").val(
        data=str(ROOT / "data.yaml"), split="test", imgsz=imgsz, batch=1, device="cpu", plots=False,
        verbose=False, project=str(ROOT / "runs" / "edge"), name="val", exist_ok=True)
    return round(float(m.box.map50), 4), round(float(m.box.map), 4)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default=str(ROOT / "models" / "vision_yolo11n.pt"))
    ap.add_argument("--imgsz", type=int, default=416)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--skip-accuracy", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "models" / "edge"),
                    help="export folder (use a short path on Windows: NCNN tools fail on long paths)")
    args = ap.parse_args()

    psutil.Process().cpu_affinity(list(range(args.threads)))
    import torch
    torch.set_num_threads(args.threads)
    limit_onnxruntime_threads(args.threads)

    image = [l for l in (ROOT / "splits" / "test.txt").read_text().splitlines() if l][0]
    rows = []
    for tag, fmt, kw in FORMATS:
        path = Path(args.weights) if fmt is None else export(args.weights, fmt, kw, args.imgsz,
                                                             tag, args.out)
        med, p95 = latency(path, image, args.imgsz, args.runs)
        row = {"format": tag, "file": path.name,
               "size_mb": size_mb(path), "latency_ms_median": med, "latency_ms_p95": p95}
        if not args.skip_accuracy:
            row["test_mAP50"], row["test_mAP50_95"] = accuracy(path, args.imgsz)
        rows.append(row)
        print(row, flush=True)

    out = {"device": f"{platform.processor() or platform.machine()} ({platform.system()})",
           "cpu_threads": args.threads, "imgsz": args.imgsz, "runs": args.runs,
           "note": "end-to-end predict() time per image incl. pre/post-processing",
           "results": rows}
    name = "edge_benchmark.json" if platform.machine().lower() in ("amd64", "x86_64") \
        else f"edge_benchmark_{platform.machine().lower()}.json"
    (ROOT / "results" / name).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
