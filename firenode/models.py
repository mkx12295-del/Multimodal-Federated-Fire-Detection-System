"""Model wrappers for the sensor (XGBoost) and vision (YOLO) pathways."""
import numpy as np
import pandas as pd

from .features import FEATURES


class SensorModel:
    def __init__(self, path):
        import xgboost as xgb

        self.model = xgb.XGBClassifier()
        self.model.load_model(str(path))

    def predict_proba(self, reading):
        """Probability of fire for one reading dict or a DataFrame of readings."""
        if isinstance(reading, dict):
            reading = pd.DataFrame([reading])
        return self.model.predict_proba(reading[FEATURES])[:, 1]


class FedSensorModel:
    """The federated MLP written by federated/client.py (weights + global mean/std)."""

    def __init__(self, path):
        from federated.model import predict_proba

        d = np.load(str(path))
        self._predict = predict_proba
        self.mean, self.std = d["mean"], d["std"]
        self.params = {k: d[k] for k in ("W1", "b1", "W2", "b2")}

    def predict_proba(self, reading):
        if isinstance(reading, dict):
            reading = pd.DataFrame([reading])
        X = reading[FEATURES].to_numpy(dtype=np.float64)
        return self._predict(self.params, (X - self.mean) / self.std)


def load_sensor_model(cfg):
    from .config import resolve

    kind = cfg["models"].get("sensor_type", "xgboost")
    if kind == "fl_mlp":
        return FedSensorModel(resolve(cfg["models"]["sensor_fl"]))
    if kind == "xgboost":
        return SensorModel(resolve(cfg["models"]["sensor"]))
    raise ValueError(f"unknown sensor_type {kind!r}")


class VisionModel:
    def __init__(self, path, hazard_classes=("fire", "smoke"), imgsz=320, conf=0.25):
        from ultralytics import YOLO

        self.model = YOLO(str(path))
        names = self.model.names
        self.hazard_ids = [i for i, n in names.items() if n in set(hazard_classes)]
        self.imgsz, self.conf = imgsz, conf

    def hazard_confidence(self, frames):
        """Highest fire/smoke box confidence per frame (0.0 when none)."""
        single = not isinstance(frames, list)
        results = self.model.predict(frames if not single else [frames], imgsz=self.imgsz,
                                     conf=self.conf, classes=self.hazard_ids, verbose=False)
        scores = [float(r.boxes.conf.max()) if len(r.boxes) else 0.0 for r in results]
        return scores[0] if single else np.array(scores)
