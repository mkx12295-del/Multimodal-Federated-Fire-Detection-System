"""Main inference loop for the fire node."""
import argparse
import csv
import json
import queue
import threading
import time
from collections import deque
from datetime import datetime, timezone

from .actuators import Actuators
from .config import load_config, resolve
from .fusion import FusionConfig, FusionEngine, State
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


class ReasoningWorker:
    """Runs the reasoner off the main loop so a slow LLM never delays sensing."""

    def __init__(self, reasoner, cooldown_s, log_path):
        self.reasoner, self.cooldown_s = reasoner, cooldown_s
        self.log_path = resolve(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._jobs, self._done = queue.Queue(maxsize=1), queue.Queue()
        self._busy = threading.Event()
        self._last = -float("inf")
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        from .reasoning import verdict_record

        while True:
            ctx = self._jobs.get()
            try:
                v = self.reasoner.reason(ctx)
                with open(self.log_path, "a") as f:
                    f.write(json.dumps({"time": datetime.now(timezone.utc).isoformat(
                        timespec="seconds"), **verdict_record(ctx, v)}) + "\n")
                self._done.put(v)
            finally:
                self._busy.clear()

    def submit(self, ctx):
        now = time.monotonic()
        if self._busy.is_set() or now - self._last < self.cooldown_s:
            return False
        self._busy.set()
        self._last = now
        self._jobs.put(ctx)
        return True

    def results(self):
        out = []
        while not self._done.empty():
            out.append(self._done.get())
        return out


def run(cfg, replay=False, no_camera=False, max_steps=None, status=None):
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

    rcfg = cfg.get("reasoning", {})
    worker = None
    if rcfg.get("enabled"):
        from .reasoning import load_reasoner

        reasoner = load_reasoner(rcfg)
        print(f"Reasoning backend: {reasoner.name}")
        worker = ReasoningWorker(reasoner, rcfg.get("cooldown_s", 30), rcfg["log_jsonl"])
    window = deque(maxlen=rcfg.get("window", 30))
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
            window.append((reading, sensor_prob, vision_conf))

            if worker is not None:
                if d.state in (State.WARNING, State.ALARM):
                    from .reasoning import Context

                    readings, probs, confs = zip(*window)
                    worker.submit(Context.from_window(readings, probs, confs, d.state.name,
                                                      cfg["fusion"]["vision_threshold"]))
                for v in worker.results():
                    print(f"  [reasoning/{v.backend} {v.latency_s:.1f}s] {v.verdict} "
                          f"({v.confidence:.2f}): {v.explanation}")
                    if (rcfg.get("escalate") and v.is_fire
                            and v.confidence >= rcfg.get("escalate_confidence", 0.8)
                            and engine.escalate()):
                        print("  [reasoning] escalated WARNING -> ALARM")
                    if status is not None:
                        status.update(last_verdict={"verdict": v.verdict,
                                                    "confidence": v.confidence,
                                                    "explanation": v.explanation,
                                                    "backend": v.backend})

            actuators.apply(engine.state)
            row = {"time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                   **{k: round(v, 3) for k, v in reading.items()},
                   "sensor_prob": round(sensor_prob, 4),
                   "vision_conf": "" if vision_conf is None else round(vision_conf, 4),
                   "fused": round(d.fused, 4), "smoothed": round(d.smoothed, 4),
                   "state": engine.state.name}
            log.write(row)
            if status is not None:
                status.push(row)

            vis = "n/a" if vision_conf is None else f"{vision_conf:.2f}"
            print(f"T={reading['Temperature']:.1f}C H={reading['Humidity']:.1f}% "
                  f"TVOC={reading['TVOC']:.0f} | sensor={sensor_prob:.2f} vision={vis} "
                  f"| score={d.smoothed:.2f} -> {engine.state.name}")

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
    p.add_argument("--no-llm", action="store_true", help="disable the reasoning module")
    p.add_argument("--dashboard", type=int, metavar="PORT",
                   help="serve a live status page on this port")
    p.add_argument("--steps", type=int, help="stop after N iterations")
    p.add_argument("--interval", type=float, help="seconds between iterations")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    if args.interval is not None:
        cfg["loop"]["interval_s"] = args.interval
    if args.no_llm:
        cfg.setdefault("reasoning", {})["enabled"] = False
    status = None
    if args.dashboard:
        from .dashboard import Status, start_dashboard

        status = Status()
        start_dashboard(status, port=args.dashboard)
        print(f"Dashboard on http://localhost:{args.dashboard}")
    run(cfg, replay=args.replay, no_camera=args.no_camera, max_steps=args.steps, status=status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
