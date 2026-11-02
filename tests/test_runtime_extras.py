import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from federated.model import init_params, predict_proba
from firenode.alerts import Alerter
from firenode.config import load_config
from firenode.fusion import State
from firenode.models import FedSensorModel, SensorModel, load_sensor_model


def test_fed_sensor_model_matches_federated_predictions(tmp_path):
    params = init_params(4, seed=3)
    mean, std = np.array([20.0, 50.0, 100.0, 940.0]), np.array([2.0, 5.0, 50.0, 1.0])
    path = tmp_path / "fl.npz"
    np.savez(path, mean=mean, std=std, **params)
    m = FedSensorModel(path)
    reading = {"Temperature": 25.0, "Humidity": 40.0, "TVOC": 900.0, "Pressure": 939.0}
    X = np.array([[25.0, 40.0, 900.0, 939.0]])
    np.testing.assert_allclose(m.predict_proba(reading), predict_proba(params, (X - mean) / std))


def test_sensor_model_selected_from_config(tmp_path):
    cfg = load_config(tuned=False)
    assert isinstance(load_sensor_model(cfg), SensorModel)
    np.savez(tmp_path / "fl.npz", mean=np.zeros(4), std=np.ones(4), **init_params(4))
    cfg["models"].update(sensor_type="fl_mlp", sensor_fl=str(tmp_path / "fl.npz"))
    assert isinstance(load_sensor_model(cfg), FedSensorModel)


class _Sink(BaseHTTPRequestHandler):
    events = []

    def log_message(self, *a):
        pass

    def do_POST(self):
        _Sink.events.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
        self.send_response(200)
        self.end_headers()


def test_alerts_fire_on_rising_and_clearing_only():
    _Sink.events = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Sink)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        a = Alerter("n1", webhook_url=f"http://127.0.0.1:{server.server_address[1]}/")
        row = {"Temperature": 30}
        seq = [State.NORMAL, State.WARNING, State.WARNING, State.ALARM, State.ALARM,
               State.WARNING, State.NORMAL]
        queued = [a.update(s, row) for s in seq]
        assert queued == [False, True, False, True, False, False, True]
        for _ in range(100):
            if len(_Sink.events) == 3:
                break
            time.sleep(0.02)
        assert [e["state"] for e in _Sink.events] == ["WARNING", "ALARM", "NORMAL"]
        assert _Sink.events[0]["node"] == "n1"
    finally:
        server.shutdown()


def test_alert_failures_do_not_raise():
    a = Alerter("n1", webhook_url="http://127.0.0.1:9/", timeout_s=0.5)
    assert a.update(State.ALARM, {}) is True
    for _ in range(100):
        if a.failed:
            break
        time.sleep(0.02)
    assert a.failed == 1


def test_alerter_disabled_without_env(monkeypatch):
    for k in ("FIRENODE_WEBHOOK_URL", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"):
        monkeypatch.delenv(k, raising=False)
    assert Alerter.from_env("n") is None
