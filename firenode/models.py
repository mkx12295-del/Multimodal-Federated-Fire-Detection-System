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
