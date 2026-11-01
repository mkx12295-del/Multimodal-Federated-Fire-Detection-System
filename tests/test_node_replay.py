"""End-to-end smoke test: run the node loop on replayed sensor data, no hardware."""
import csv

import pytest

from firenode.config import load_config, resolve
from firenode.node import run


@pytest.mark.skipif(not resolve("models/sensor_xgb.json").exists(),
                    reason="train the sensor model first")
def test_node_runs_in_replay_mode(tmp_path):
    cfg = load_config()
    cfg["loop"]["interval_s"] = 0
    cfg["loop"]["log_csv"] = str(tmp_path / "events.csv")
    run(cfg, replay=True, no_camera=True, max_steps=10)

    rows = list(csv.DictReader(open(tmp_path / "events.csv")))
    assert len(rows) == 10
    assert all(r["state"] in {"NORMAL", "WARNING", "ALARM"} for r in rows)
    assert all(0.0 <= float(r["sensor_prob"]) <= 1.0 for r in rows)
