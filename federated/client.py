"""Federated node client: trains on local readings and exchanges weights only."""
import argparse
import json
import time
import urllib.error
import urllib.request

import numpy as np

from .fedavg import local_stats
from .model import from_lists, to_lists, train_local


class FedClient:
    def __init__(self, server_url, name, X, y, local_epochs=1, lr=0.05, mu=0.0, poll_s=0.2,
                 local_steps=None):
        self.url = server_url.rstrip("/")
        self.name, self.X_raw, self.y = name, X, y
        self.local_epochs, self.lr, self.mu, self.poll_s = local_epochs, lr, mu, poll_s
        self.local_steps = local_steps

    def _req(self, method, path, body=None):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.url + path, data=data, method=method,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read() or b"{}")

    def run(self):
        self._req("POST", "/stats", {"client": self.name, "stats": local_stats(self.X_raw)})
        while True:
            code, norm = self._req("GET", "/stats")
            if code == 200:
                break
            time.sleep(self.poll_s)
        self.mean, self.std = np.array(norm["mean"]), np.array(norm["std"])
        X = (self.X_raw - self.mean) / self.std

        last_round = 0
        while True:
            _, msg = self._req("GET", "/model")
            if msg["done"]:
                return from_lists(msg["params"])
            if msg["round"] == last_round:
                time.sleep(self.poll_s)  # waiting for the other nodes
                continue
            params = train_local(from_lists(msg["params"]), X, self.y, self.local_epochs,
                                 self.lr, mu=self.mu, seed=msg["round"],
                                 max_steps=self.local_steps)
            self._req("POST", "/update", {"client": self.name, "round": msg["round"],
                                          "n": int(len(self.y)), "params": to_lists(params)})
            last_round = msg["round"]


def main():
    p = argparse.ArgumentParser(description="Run one federated fire node.")
    p.add_argument("--server", default="http://localhost:8080")
    p.add_argument("--name", required=True)
    p.add_argument("--csv", required=True, help="this node's local readings (IoT CSV format)")
    p.add_argument("--session", type=int, help="use only this recording session from the CSV")
    p.add_argument("--holdout", type=float, default=0.0,
                   help="keep the latest fraction of readings out of training")
    p.add_argument("--local-steps", type=int, default=20)
    p.add_argument("--poll", type=float, default=0.05)
    p.add_argument("--out", help="where to save the final global model (.npz)")
    args = p.parse_args()

    from firenode.features import chronological_split, load_iot_csv, xy
    df = load_iot_csv(args.csv)
    if args.session is not None:
        df = df[df["session"] == args.session]
    if args.holdout:
        df = chronological_split(df, val=0.0, test=args.holdout)["train"]
    X, y = xy(df)
    client = FedClient(args.server, args.name, X, y, poll_s=args.poll,
                       local_steps=args.local_steps)
    params = client.run()
    print(f"{args.name}: received final global model")
    if args.out:
        np.savez(args.out, mean=client.mean, std=client.std, **params)


if __name__ == "__main__":
    main()
