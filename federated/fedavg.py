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


def _flat(params):
    return np.concatenate([params[k].ravel() for k in sorted(params)])


def _unflat(vec, like):
    out, i = {}, 0
    for k in sorted(like):
        n = like[k].size
        out[k] = vec[i:i + n].reshape(like[k].shape)
        i += n
    return out


def dp_fedavg(global_params, local_params, clip, noise_multiplier, rng):
    """Client-level DP-FedAvg (McMahan et al., 2018).

    Each node's update (local - global) is clipped to L2 norm ``clip``; the
    updates are averaged with equal weight and Gaussian noise with standard
    deviation ``noise_multiplier * clip / num_nodes`` is added, so the
    aggregate hides whether any single node took part.
    """
    g = _flat(global_params)
    deltas = []
    for p in local_params:
        d = _flat(p) - g
        norm = np.linalg.norm(d)
        deltas.append(d * min(1.0, clip / (norm + 1e-12)))
    k = len(deltas)
    avg = np.mean(deltas, axis=0)
    avg = avg + rng.normal(0, noise_multiplier * clip / k, avg.shape)
    return _unflat(g + avg, global_params)


def gaussian_dp_epsilon(noise_multiplier, rounds, delta):
    """(epsilon, delta) for ``rounds`` adaptive Gaussian mechanisms with full participation.

    Uses Renyi DP: one round gives alpha / (2 sigma^2) at order alpha, rounds
    compose additively, and epsilon = RDP + log(1/delta) / (alpha - 1),
    minimized over alpha. No subsampling amplification is assumed, because
    every node takes part in every round.
    """
    if noise_multiplier <= 0:
        return float("inf")
    alphas = np.concatenate([np.linspace(1.01, 10, 200), np.arange(11, 2000)])
    eps = rounds * alphas / (2 * noise_multiplier ** 2) + np.log(1 / delta) / (alphas - 1)
    return float(eps.min())


@dataclass
class Client:
    name: str
    X: np.ndarray
    y: np.ndarray


def run_federated(clients, rounds=30, local_epochs=1, lr=0.05, mu=0.0, hidden=16,
                  seed=0, eval_fn=None, local_steps=None, dp_clip=None, dp_noise=0.0):
    """Simulate FedAvg (mu=0), FedProx (mu>0) or DP-FedAvg (dp_clip set).

    Inputs must already be standardized.
    """
    params = init_params(clients[0].X.shape[1], hidden, seed)
    rng = np.random.default_rng(seed + 12345)
    history = []
    for r in range(1, rounds + 1):
        updates = [(len(c.y), train_local(params, c.X, c.y, local_epochs, lr, mu=mu,
                                          seed=seed * 1000 + r * 10 + i,
                                          max_steps=local_steps))
                   for i, c in enumerate(clients)]
        if dp_clip:
            params = dp_fedavg(params, [p for _, p in updates], dp_clip, dp_noise, rng)
        else:
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
