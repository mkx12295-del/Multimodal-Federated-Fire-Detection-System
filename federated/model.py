"""A small NumPy MLP whose weights can be averaged across nodes.

Tree ensembles such as XGBoost cannot be combined by averaging parameters, so
the federated sensor model is a 2-layer MLP. Written in plain NumPy so it runs
on a Raspberry Pi without PyTorch.
"""
import numpy as np


def init_params(n_in, n_hidden=16, seed=0):
    rng = np.random.default_rng(seed)
    return {
        "W1": rng.normal(0, np.sqrt(2 / n_in), (n_in, n_hidden)),
        "b1": np.zeros(n_hidden),
        "W2": rng.normal(0, np.sqrt(1 / n_hidden), (n_hidden, 1)),
        "b2": np.zeros(1),
    }


def copy_params(p):
    return {k: v.copy() for k, v in p.items()}


def _sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def predict_proba(params, X):
    h = np.maximum(0, X @ params["W1"] + params["b1"])
    return _sigmoid(h @ params["W2"] + params["b2"]).ravel()


def _grads(params, X, y):
    z1 = X @ params["W1"] + params["b1"]
    h = np.maximum(0, z1)
    p = _sigmoid(h @ params["W2"] + params["b2"]).ravel()
    d2 = ((p - y) / len(y))[:, None]          # dBCE/dlogit, averaged over the batch
    dh = (d2 @ params["W2"].T) * (z1 > 0)
    return {"W2": h.T @ d2, "b2": d2.sum(0), "W1": X.T @ dh, "b1": dh.sum(0)}


def log_loss(params, X, y):
    p = np.clip(predict_proba(params, X), 1e-7, 1 - 1e-7)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def train_local(params, X, y, epochs=1, lr=0.05, batch_size=64, mu=0.0, seed=0, max_steps=None):
    """Mini-batch SGD on one node's data.

    ``mu > 0`` adds the FedProx proximal term (mu/2)||w - w_global||^2, which
    keeps local models close to the global one when node data is non-IID.
    ``max_steps`` caps the SGD steps per call, so a node with far more data
    than the others cannot drag the global model towards its own optimum.
    """
    params = copy_params(params)
    anchor = copy_params(params)
    rng = np.random.default_rng(seed)
    n = len(y)
    steps = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for i in range(0, n, batch_size):
            if max_steps is not None and steps >= max_steps:
                return params
            steps += 1
            idx = order[i:i + batch_size]
            g = _grads(params, X[idx], y[idx])
            for k in params:
                step = g[k] + (mu * (params[k] - anchor[k]) if mu else 0)
                params[k] -= lr * step
    return params


def to_lists(params):
    return {k: v.tolist() for k, v in params.items()}


def from_lists(d):
    return {k: np.asarray(v, dtype=np.float64) for k, v in d.items()}
