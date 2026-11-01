"""Aggregation primitives and an in-process federated training loop."""
from dataclasses import dataclass

import numpy as np

from .model import copy_params, init_params, log_loss, train_local


def local_stats(X):
    """Sufficient statistics a node shares instead of raw readings."""
    return {"n": int(len(X)), "sum": X.sum(0).tolist(), "sumsq": (X ** 2).sum(0).tolist()}


def merge_stats(stats):
    """Global mean/std from per-node sums, so no node reveals its rows."""
    n = sum(s["n"] for s in stats)
    total = np.sum([s["sum"] for s in stats], axis=0)
    total_sq = np.sum([s["sumsq"] for s in stats], axis=0)
    mean = total / n
    std = np.sqrt(np.maximum(total_sq / n - mean ** 2, 1e-12))
    return mean, std


def fedavg(updates):
    """Weighted average of node parameters; ``updates`` is [(n_samples, params)]."""
    total = sum(n for n, _ in updates)
    keys = updates[0][1].keys()
    return {k: sum(n * p[k] for n, p in updates) / total for k in keys}


@dataclass
class Client:
    name: str
    X: np.ndarray
    y: np.ndarray


def run_federated(clients, rounds=30, local_epochs=1, lr=0.05, mu=0.0, hidden=16,
                  seed=0, eval_fn=None, local_steps=None):
    """Simulate FedAvg (mu=0) or FedProx (mu>0). Inputs must already be standardized."""
    params = init_params(clients[0].X.shape[1], hidden, seed)
    history = []
    for r in range(1, rounds + 1):
        updates = [(len(c.y), train_local(params, c.X, c.y, local_epochs, lr, mu=mu,
                                          seed=seed * 1000 + r * 10 + i,
                                          max_steps=local_steps))
                   for i, c in enumerate(clients)]
        params = fedavg(updates)
        row = {"round": r,
               "train_loss": float(np.average([log_loss(params, c.X, c.y) for c in clients],
                                              weights=[len(c.y) for c in clients]))}
        if eval_fn:
            row.update(eval_fn(params))
        history.append(row)
    return params, history


def run_centralized(X, y, epochs=30, lr=0.05, hidden=16, seed=0):
    params = init_params(X.shape[1], hidden, seed)
    for e in range(epochs):
        params = train_local(params, X, y, 1, lr, seed=seed * 1000 + e)
    return params


def run_local_only(clients, epochs=30, lr=0.05, hidden=16, seed=0):
    out = {}
    for c in clients:
        p = init_params(c.X.shape[1], hidden, seed)
        out[c.name] = train_local(copy_params(p), c.X, c.y, epochs, lr, seed=seed)
    return out
