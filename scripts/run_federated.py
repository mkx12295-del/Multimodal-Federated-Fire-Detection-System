"""Federated learning experiment on the sensor dataset.

Each recording session in the IoT dataset is treated as a separate fire node
(a different site), which gives naturally non-IID clients: some sites never
see a fire, others are mostly fire. Every client is split chronologically
into train / test. Compared methods:

* local-only  : each node trains on its own data and never shares anything
* FedAvg      : nodes share weights only, aggregated each round
* FedProx     : FedAvg plus a proximal term for non-IID clients
* centralized : all raw data pooled in one place (upper bound, no privacy)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from federated.fedavg import (Client, local_stats, merge_stats, run_centralized,  # noqa: E402
                              run_federated, run_local_only)
from federated.model import predict_proba  # noqa: E402
from firenode.features import chronological_split, load_iot_csv, xy  # noqa: E402


def evaluate(params, tests):
    out, ys, ps = {}, [], []
    for name, (X, y) in tests.items():
        pred = (predict_proba(params, X) >= 0.5).astype(int)
        out[name] = round(accuracy_score(y, pred), 4)
        ys.append(y)
        ps.append(pred)
    y, p = np.concatenate(ys), np.concatenate(ps)
    out["pooled_accuracy"] = round(accuracy_score(y, p), 4)
    out["pooled_f1"] = round(f1_score(y, p), 4)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rounds", type=int, default=200)
    ap.add_argument("--local-epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--mu", type=float, default=0.1, help="FedProx proximal weight")
    ap.add_argument("--local-steps", type=int, default=20,
                    help="cap on SGD steps per node per round")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = load_iot_csv(ROOT / "smoke_detection_iot.csv")
    raw_train, raw_test = {}, {}
    for s, g in df.groupby("session"):
        parts = chronological_split(g, val=0.0, test=0.2)
        raw_train[f"node_{s}"] = xy(parts["train"])
        raw_test[f"node_{s}"] = xy(parts["test"])

    # Global standardization from per-node sums (what the HTTP protocol does).
    mean, std = merge_stats([local_stats(X) for X, _ in raw_train.values()])
    norm = lambda X: (X - mean) / std  # noqa: E731
    clients = [Client(n, norm(X), y.astype(float)) for n, (X, y) in raw_train.items()]
    tests = {n: (norm(X), y) for n, (X, y) in raw_test.items()}

    results = {"clients": {n: {"train_rows": int(len(y)), "test_rows": int(len(raw_test[n][1])),
                               "train_fire_rate": round(float(y.mean()), 4)}
                           for n, (_, y) in raw_train.items()}}

    eval_fn = lambda p: {"pooled_accuracy": evaluate(p, tests)["pooled_accuracy"]}  # noqa: E731
    fedavg_params, fedavg_hist = run_federated(clients, args.rounds, args.local_epochs, args.lr,
                                               mu=0.0, seed=args.seed, eval_fn=eval_fn,
                                               local_steps=args.local_steps)
    fedprox_params, fedprox_hist = run_federated(clients, args.rounds, args.local_epochs, args.lr,
                                                 mu=args.mu, seed=args.seed, eval_fn=eval_fn,
                                                 local_steps=args.local_steps)
    X_all = np.concatenate([c.X for c in clients])
    y_all = np.concatenate([c.y for c in clients])
    central = run_centralized(X_all, y_all, args.rounds * args.local_epochs, args.lr,
                              seed=args.seed)
    local = run_local_only(clients, args.rounds * args.local_epochs, args.lr, seed=args.seed)

    results["fedavg"] = evaluate(fedavg_params, tests)
    results["fedprox"] = evaluate(fedprox_params, tests)
    results["centralized"] = evaluate(central, tests)
    # A local-only model is used at its own site, and would also be what a new
    # site gets if it copied another node's model; report both.
    local_eval = {n: evaluate(p, tests) for n, p in local.items()}
    results["local_only"] = {
        "own_site_accuracy": {n: local_eval[n][n] for n in local},
        "mean_pooled_accuracy": round(float(np.mean([e["pooled_accuracy"]
                                                     for e in local_eval.values()])), 4),
        "worst_pooled_accuracy": round(float(min(e["pooled_accuracy"]
                                                 for e in local_eval.values())), 4),
    }
    results["config"] = vars(args)

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "federated_metrics.json").write_text(json.dumps(results, indent=2))
    (out_dir / "federated_history.json").write_text(json.dumps(
        {"fedavg": fedavg_hist, "fedprox": fedprox_hist}, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 3.5))
    for name, hist in (("FedAvg", fedavg_hist), ("FedProx", fedprox_hist)):
        ax.plot([h["round"] for h in hist], [h["pooled_accuracy"] for h in hist], label=name)
    ax.axhline(results["centralized"]["pooled_accuracy"], ls="--", c="gray",
               label="Centralized (pooled raw data)")
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Test accuracy (all nodes)")
    ax.set_title("Federated sensor model, 5 non-IID nodes")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "federated_convergence.png", dpi=120)

    print(json.dumps({k: v for k, v in results.items() if k != "clients"}, indent=2))


if __name__ == "__main__":
    main()
