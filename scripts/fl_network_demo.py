"""Networked federated training: one aggregator and five node processes over HTTP.

Each node is a separate OS process that loads only its own recording session,
keeps the latest 20% of its readings for testing, and talks to the aggregator
through the JSON/HTTP protocol in federated/server.py. On a real deployment
the same commands run on separate Raspberry Pis with --server pointing at the
aggregator's address.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from federated.model import predict_proba  # noqa: E402
from firenode.features import chronological_split, load_iot_csv, xy  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rounds", type=int, default=200)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--clip", type=float, help="use clipped equal-weight aggregation")
    ap.add_argument("--noise", type=float, default=0.0)
    args = ap.parse_args()
    extra = ["--clip", str(args.clip), "--noise", str(args.noise)] if args.clip else []

    csv = ROOT / "smoke_detection_iot.csv"
    df = load_iot_csv(csv)
    sessions = sorted(df["session"].unique())
    out_dir = ROOT / "results" / "fl_network"
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    server = subprocess.Popen([sys.executable, "-m", "federated.server", "--clients",
                               str(len(sessions)), "--rounds", str(args.rounds),
                               "--host", "127.0.0.1", "--port", str(args.port), *extra], cwd=ROOT)
    time.sleep(2)
    nodes = [subprocess.Popen([sys.executable, "-m", "federated.client",
                               "--server", f"http://127.0.0.1:{args.port}",
                               "--name", f"node_{s}", "--csv", str(csv), "--session", str(s),
                               "--holdout", "0.2", "--out", str(out_dir / f"node_{s}.npz")],
                              cwd=ROOT) for s in sessions]
    codes = [n.wait(timeout=3600) for n in nodes]
    server.wait(timeout=60)
    elapsed = time.time() - t0
    if any(codes):
        raise SystemExit(f"node exit codes: {codes}")

    models = [np.load(out_dir / f"node_{s}.npz") for s in sessions]
    same = all(np.allclose(models[0][k], m[k]) for m in models[1:] for k in ("W1", "b1", "W2", "b2"))
    m = models[0]
    params = {k: m[k] for k in ("W1", "b1", "W2", "b2")}

    per_node, ys, ps = {}, [], []
    for s in sessions:
        X, y = xy(chronological_split(df[df["session"] == s], val=0.0, test=0.2)["test"])
        pred = (predict_proba(params, (X - m["mean"]) / m["std"]) >= 0.5).astype(int)
        per_node[f"node_{s}"] = round(float((pred == y).mean()), 4)
        ys.append(y)
        ps.append(pred)
    y, p = np.concatenate(ys), np.concatenate(ps)

    result = {"nodes": len(sessions), "rounds": args.rounds,
              "aggregation": f"clipped(clip={args.clip}, noise={args.noise})" if args.clip
              else "fedavg", "wall_clock_s": round(elapsed, 1),
              "all_nodes_hold_identical_model": bool(same),
              "per_node_test_accuracy": per_node,
              "pooled_test_accuracy": round(float((y == p).mean()), 4)}
    name = "federated_network_demo" + ("_clipped" if args.clip else "") + ".json"
    (ROOT / "results" / name).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
