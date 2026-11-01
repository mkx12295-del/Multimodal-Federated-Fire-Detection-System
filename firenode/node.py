"""Main inference loop for the fire node."""
import argparse
import csv
import time
from datetime import datetime, timezone

from .actuators import Actuators
from .config import load_config, resolve
from .fusion import FusionConfig, FusionEngine
from .models import SensorModel, VisionModel
from .sensors import open_source


class Camera:
    def __init__(self, index, width, height):
        import cv2

        self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self.cap.isOpened():
            raise RuntimeError(f"camera {index} could not be opened")

    def read(self):
        ok, frame = self.cap.read()
        return frame if ok else None

    def close(self):
        self.cap.release()


class EventLog:
    FIELDS = ["time", "Temperature", "Humidity", "TVOC", "Pressure",
              "sensor_prob", "vision_conf", "fused", "smoothed", "state"]

    def __init__(self, path):
        self.path = resolve(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        new = not self.path.exists()
        self.f = open(self.path, "a", newline="")
        self.w = csv.DictWriter(self.f, fieldnames=self.FIELDS)
        if new:
            self.w.writeheader()

    def write(self, row):
        self.w.writerow(row)
        self.f.flush()

    def close(self):
        self.f.close()


def run(cfg, replay=False, no_camera=False, max_steps=None):
    sensor_model = SensorModel(resolve(cfg["models"]["sensor"]))
    source = open_source(cfg, force_replay=replay)

    vision = camera = None
    vcfg = cfg["vision"]
    if vcfg["enabled"] and not no_camera:
        try:
            camera = Camera(vcfg["camera_index"], vcfg["width"], vcfg["height"])
            vision = VisionModel(resolve(cfg["models"]["vision"]), vcfg["hazard_classes"],
                                 vcfg["imgsz"], conf=0.05)
        except Exception as e:
            print(f"Vision pathway disabled: {e}")
            camera = None

    engine = FusionEngine(FusionConfig.from_dict(cfg["fusion"]))
    actuators = Actuators(cfg["hardware"]["buzzer_pin"], cfg["hardware"]["relay_pin"],
                          use_gpio=not replay)
    log = EventLog(cfg["loop"]["log_csv"])
    print("\n--- Fire node active (Ctrl+C to stop) ---\n")

    step = 0
    try:
        while max_steps is None or step < max_steps:
            t0 = time.monotonic()
            reading = source.read()
            sensor_prob = float(sensor_model.predict_proba(reading)[0])

            vision_conf = None
            if camera is not None:
                frame = camera.read()
                if frame is not None:
                    vision_conf = vision.hazard_confidence(frame)

            d = engine.update(sensor_prob, vision_conf)
            actuators.apply(d.state)
            log.write({"time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                       **{k: round(v, 3) for k, v in reading.items()},
                       "sensor_prob": round(sensor_prob, 4),
                       "vision_conf": "" if vision_conf is None else round(vision_conf, 4),
                       "fused": round(d.fused, 4), "smoothed": round(d.smoothed, 4),
                       "state": d.state.name})

            vis = "n/a" if vision_conf is None else f"{vision_conf:.2f}"
            print(f"T={reading['Temperature']:.1f}C H={reading['Humidity']:.1f}% "
                  f"TVOC={reading['TVOC']:.0f} | sensor={sensor_prob:.2f} vision={vis} "
                  f"| score={d.smoothed:.2f} -> {d.state.name}")

            step += 1
            time.sleep(max(0.0, cfg["loop"]["interval_s"] - (time.monotonic() - t0)))
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        actuators.close()
        log.close()
        if camera is not None:
            camera.close()
    return engine.state


def main(argv=None):
    p = argparse.ArgumentParser(description="Run the fire detection node.")
    p.add_argument("--config", help="YAML file overriding config/default.yaml")
    p.add_argument("--replay", action="store_true",
                   help="replay recorded sensor data instead of reading the BME680")
    p.add_argument("--no-camera", action="store_true", help="run the sensor pathway only")
    p.add_argument("--steps", type=int, help="stop after N iterations")
    p.add_argument("--interval", type=float, help="seconds between iterations")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    if args.interval is not None:
        cfg["loop"]["interval_s"] = args.interval
    run(cfg, replay=args.replay, no_camera=args.no_camera, max_steps=args.steps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
