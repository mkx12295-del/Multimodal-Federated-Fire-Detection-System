"""Federated training of the YOLO vision model across non-IID camera nodes.

The training split is divided among N nodes with a Dirichlet(alpha) split over
each image's dominant class, so some nodes see mostly fire, others mostly
smoke or "other". Each round, every node trains one local epoch starting from
the global weights; the server averages the weights (FedAvg, weighted by image
count). A centralized model trained with the same optimizer and the same
number of passes over the data is the baseline. Both are scored on the same
held-out test split.

Needs a CUDA PyTorch environment for reasonable speed (about 1 h on an
RTX 5060 laptop GPU with the defaults).
"""
import argparse
import json
import math
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel

ROOT = Path(__file__).resolve().parent.parent
NAMES = {0: "fire", 1: "other", 2: "smoke"}
WORK = ROOT / "runs" / "fedvision"


def dominant_class(image):
    label = Path(image.replace("/images/", "/labels/")).with_suffix(".txt")
    counts = Counter(line.split()[0] for line in label.read_text().splitlines() if line.strip())
    return int(counts.most_common(1)[0][0]) if counts else 1


def partition(images, n_clients, alpha, seed):
    rng = np.random.default_rng(seed)
    by_class = {}
    for img in images:
        by_class.setdefault(dominant_class(img), []).append(img)
    clients = [[] for _ in range(n_clients)]
    for c, imgs in sorted(by_class.items()):
        rng.shuffle(imgs)
        cuts = (np.cumsum(rng.dirichlet([alpha] * n_clients)) * len(imgs)).astype(int)[:-1]
        for k, part in enumerate(np.split(np.array(imgs), cuts)):
            clients[k].extend(part.tolist())
    return clients


def write_yaml(path, train_list):
    path.write_text(yaml.safe_dump({
        "train": str(train_list), "val": str(ROOT / "splits" / "val.txt"),
        "test": str(ROOT / "splits" / "test.txt"), "nc": 3, "names": list(NAMES.values())}))


def init_model(path):
    """3-class YOLO11n with COCO-pretrained backbone; every node starts from this."""
    y = YOLO(str(ROOT / "yolo11n.pt"))
    m = DetectionModel("yolo11n.yaml", nc=3, verbose=False)
    m.load(y.model)
    m.names = NAMES
    y.model = m
    y.save(str(path))


def load_state(ckpt_path):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = ck.get("ema") or ck["model"]
    return {k: v.float() if v.is_floating_point() else v for k, v in model.state_dict().items()}


def fedavg_torch(states, weights):
    """Sample-weighted average of float tensors; integer buffers come from the first node."""
    total = float(sum(weights))
    out = {}
    for k, v in states[0].items():
        if v.is_floating_point():
            out[k] = sum(w * s[k] for s, w in zip(states, weights)) / total
        else:
            out[k] = v.clone()
    return out


def round_lr(r, rounds, lr0=0.01, lr_min=0.0005):
    """Cosine decay across rounds, matching the centralized cos_lr schedule."""
    return lr_min + 0.5 * (lr0 - lr_min) * (1 + math.cos(math.pi * (r - 1) / rounds))


def train_args(device, imgsz, batch, workers):
    return dict(device=device, imgsz=imgsz, batch=batch, workers=workers, optimizer="SGD",
                momentum=0.937, warmup_epochs=0, close_mosaic=0, plots=False, seed=42,
                exist_ok=True, verbose=False)


def evaluate(weights, data, split, device, imgsz):
    m = YOLO(str(weights)).val(data=str(data), split=split, device=device, imgsz=imgsz,
                               plots=False, verbose=False, project=str(WORK), name="val",
                               exist_ok=True)
    return {"mAP50": round(float(m.box.map50), 4), "mAP50_95": round(float(m.box.map), 4),
            "per_class_mAP50": {m.names[int(c)]: round(float(a), 4)
                                for c, a in zip(m.box.ap_class_index, m.box.ap50)}}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clients", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--alpha", type=float, default=0.5, help="Dirichlet non-IID strength")
    ap.add_argument("--imgsz", type=int, default=416)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-central", action="store_true")
    args = ap.parse_args()

    WORK.mkdir(parents=True, exist_ok=True)
    images = [l for l in (ROOT / "splits" / "train.txt").read_text().splitlines() if l]
    clients = partition(images, args.clients, args.alpha, args.seed)
    data_files, sizes, composition = [], [], {}
    for k, imgs in enumerate(clients):
        lst = WORK / f"client_{k}.txt"
        lst.write_text("\n".join(imgs) + "\n")
        y = WORK / f"client_{k}.yaml"
        write_yaml(y, lst)
        data_files.append(y)
        sizes.append(len(imgs))
        dom = Counter(NAMES[dominant_class(i)] for i in imgs)
        composition[f"node_{k}"] = {"images": len(imgs), **dict(dom)}
        print(f"node_{k}: {len(imgs)} images {dict(dom)}")
    full_yaml = WORK / "central.yaml"
    write_yaml(full_yaml, ROOT / "splits" / "train.txt")

    global_w = WORK / "global_r0.pt"
    init_model(global_w)
    common = train_args(args.device, args.imgsz, args.batch, args.workers)
    history = []
    for r in range(1, args.rounds + 1):
        lr = round_lr(r, args.rounds)
        states = []
        for k, data in enumerate(data_files):
            name = f"node_{k}"
            YOLO(str(global_w)).train(data=str(data), epochs=1, lr0=lr, lrf=1.0, val=False,
                                      project=str(WORK), name=name, **common)
            states.append(load_state(WORK / name / "weights" / "last.pt"))
        g = YOLO(str(global_w))
        g.model.load_state_dict(fedavg_torch(states, sizes))
        new_w = WORK / f"global_r{r}.pt"
        g.save(str(new_w))
        if r > 1:
            global_w.unlink(missing_ok=True)
        global_w = new_w
        val = evaluate(global_w, full_yaml, "val", args.device, args.imgsz)
        history.append({"round": r, "lr": round(lr, 5), "val_mAP50": val["mAP50"]})
        print(f"round {r}/{args.rounds}  lr={lr:.4f}  val mAP50={val['mAP50']:.4f}", flush=True)

    shutil.copy2(global_w, ROOT / "models" / "vision_yolo11n_fedavg.pt")
    results = {"clients": composition, "rounds": args.rounds, "alpha": args.alpha,
               "local_epochs_per_round": 1, "history": history,
               "fedavg_test": evaluate(global_w, full_yaml, "test", args.device, args.imgsz)}

    if not args.skip_central:
        # Same optimizer, schedule and number of passes over the training data.
        init_model(WORK / "central_init.pt")
        YOLO(str(WORK / "central_init.pt")).train(
            data=str(full_yaml), epochs=args.rounds, lr0=0.01, lrf=0.05, cos_lr=True,
            project=str(WORK), name="central", **common)
        results["centralized_same_budget_test"] = evaluate(
            WORK / "central" / "weights" / "best.pt", full_yaml, "test", args.device, args.imgsz)
    deployed = json.loads((ROOT / "results" / "vision_test_metrics.json").read_text())
    results["centralized_50_epochs_test_mAP50"] = round(deployed["mAP50"], 4)

    (ROOT / "results" / "federated_vision.json").write_text(json.dumps(results, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot([h["round"] for h in history], [h["val_mAP50"] for h in history], marker="o",
            ms=3, label=f"FedAvg, {args.clients} non-IID nodes")
    if "centralized_same_budget_test" in results:
        ax.axhline(results["centralized_same_budget_test"]["mAP50"], ls="--", c="gray",
                   label=f"Centralized, {args.rounds} epochs (test)")
    ax.set_xlabel("Round (1 local epoch per node)")
    ax.set_ylabel("mAP50")
    ax.set_title("Federated YOLO11n fire/smoke detector")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "federated_vision.png", dpi=120)
    print(json.dumps({k: v for k, v in results.items() if k != "history"}, indent=2))


if __name__ == "__main__":
    sys.exit(main())
