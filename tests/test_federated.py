import threading

import numpy as np

from federated.client import FedClient
from federated.fedavg import Client, fedavg, local_stats, merge_stats, run_federated
from federated.model import init_params, log_loss, predict_proba, train_local
from federated.server import FedState, serve


def toy_data(n=400, shift=0.0, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(shift, 1, (n, 4))
    y = (X[:, 0] + X[:, 2] > 2 * shift).astype(float)
    return X, y


def test_fedavg_is_sample_weighted():
    a = {"w": np.array([0.0])}
    b = {"w": np.array([10.0])}
    assert fedavg([(1, a), (3, b)])["w"][0] == 7.5


def test_merged_stats_match_pooled_data():
    parts = [toy_data(100, shift, i)[0] for i, shift in enumerate((0.0, 1.0, 3.0))]
    mean, std = merge_stats([local_stats(X) for X in parts])
    pooled = np.concatenate(parts)
    np.testing.assert_allclose(mean, pooled.mean(0))
    np.testing.assert_allclose(std, pooled.std(0))


def test_local_training_reduces_loss():
    X, y = toy_data()
    p0 = init_params(4)
    p1 = train_local(p0, X, y, epochs=20, lr=0.1)
    assert log_loss(p1, X, y) < log_loss(p0, X, y)


def test_fedprox_stays_closer_to_global_model():
    X, y = toy_data()
    p0 = init_params(4)
    dist = lambda p: sum(np.sum((p[k] - p0[k]) ** 2) for k in p)  # noqa: E731
    free = train_local(p0, X, y, epochs=5, lr=0.1)
    prox = train_local(p0, X, y, epochs=5, lr=0.1, mu=5.0)
    assert dist(prox) < dist(free)


def test_federated_training_learns_across_clients():
    clients = [Client(f"c{i}", *toy_data(300, 0.0, i)) for i in range(3)]
    params, history = run_federated(clients, rounds=15, lr=0.1)
    X, y = toy_data(500, 0.0, 99)
    acc = ((predict_proba(params, X) >= 0.5) == y).mean()
    assert acc > 0.9
    assert history[-1]["train_loss"] < history[0]["train_loss"]


def test_http_round_trip_between_server_and_nodes():
    state = FedState(num_clients=2, rounds=3, n_features=4)
    server = serve(state, "127.0.0.1", 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}"

    results = {}

    def node(name, seed):
        X, y = toy_data(200, 0.0, seed)
        results[name] = FedClient(url, name, X, y, lr=0.1, poll_s=0.01).run()

    threads = [threading.Thread(target=node, args=(f"n{i}", i)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    server.shutdown()

    assert state.done and state.round == 4
    assert set(results) == {"n0", "n1"}
    np.testing.assert_allclose(results["n0"]["W1"], results["n1"]["W1"])
