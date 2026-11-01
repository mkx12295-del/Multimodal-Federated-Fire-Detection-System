"""Federated aggregation server (standard library only).

Protocol, for a fixed set of ``num_clients`` nodes:

1. Each node POSTs /stats with its feature sums; once all have reported,
   GET /stats returns the global mean and std used for standardization.
2. Each round, nodes GET /model, train locally, and POST /update with their
   sample count and new weights. When every node has reported, the server
   runs FedAvg and moves to the next round.

Raw sensor readings never leave a node; only sums and weights do.
"""
import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .fedavg import fedavg, merge_stats
from .model import from_lists, init_params, to_lists


class FedState:
    def __init__(self, num_clients, rounds, n_features, hidden=16, seed=0):
        self.num_clients, self.rounds = num_clients, rounds
        self.params = init_params(n_features, hidden, seed)
        self.round = 1
        self.stats = {}
        self.norm = None
        self.pending = {}
        self.lock = threading.Lock()

    @property
    def done(self):
        return self.round > self.rounds

    def add_stats(self, client, stats):
        with self.lock:
            self.stats[client] = stats
            if len(self.stats) == self.num_clients:
                mean, std = merge_stats(list(self.stats.values()))
                self.norm = {"mean": mean.tolist(), "std": std.tolist()}

    def add_update(self, client, rnd, n, params):
        with self.lock:
            if rnd != self.round:
                return False
            self.pending[client] = (n, from_lists(params))
            if len(self.pending) == self.num_clients:
                self.params = fedavg(list(self.pending.values()))
                self.pending = {}
                self.round += 1
            return True


def make_handler(state):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, code, body):
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _body(self):
            return json.loads(self.rfile.read(int(self.headers["Content-Length"])))

        def do_GET(self):
            if self.path == "/stats":
                self._send(200 if state.norm else 202, state.norm or {})
            elif self.path == "/model":
                with state.lock:
                    self._send(200, {"round": state.round, "done": state.done,
                                     "params": to_lists(state.params)})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            body = self._body()
            if self.path == "/stats":
                state.add_stats(body["client"], body["stats"])
                self._send(200, {"ok": True})
            elif self.path == "/update":
                ok = state.add_update(body["client"], body["round"], body["n"], body["params"])
                self._send(200 if ok else 409, {"ok": ok})
            else:
                self._send(404, {"error": "not found"})

    return Handler


def serve(state, host="0.0.0.0", port=8080):
    server = ThreadingHTTPServer((host, port), make_handler(state))
    return server


def main():
    p = argparse.ArgumentParser(description="Run the federated aggregation server.")
    p.add_argument("--clients", type=int, required=True)
    p.add_argument("--rounds", type=int, default=20)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8080)
    args = p.parse_args()

    from firenode.features import FEATURES
    state = FedState(args.clients, args.rounds, len(FEATURES))
    server = serve(state, args.host, args.port)
    print(f"Aggregator on {args.host}:{args.port}, waiting for {args.clients} nodes "
          f"x {args.rounds} rounds")
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        while not state.done:
            threading.Event().wait(1.0)
        print("All rounds complete.")
        threading.Event().wait(5.0)  # let nodes fetch the final model
    finally:
        server.shutdown()


if __name__ == "__main__":
    main()
