"""Replay benchmark comparing alarm policies on held-out data.

No public dataset pairs camera frames with gas readings from the same fire, so
this builds synthetic episodes from real held-out pieces:

* sensor windows: consecutive readings from the chronological test split,
  scored by the trained XGBoost model;
* camera frames: images from the YOLO test split, scored by the trained
  detector (fire/smoke confidence). Frames are sampled independently, so this
  is a sequence of stills, not video.

Episode types (60 frames each):
  fire_both      sensor fire window + hazard images
  fire_no_camera sensor fire window + non-hazard images (smoke out of view)
  fire_no_gas    clean-air sensor window + hazard images (open flame, no smoke yet)
  no_fire        clean-air sensor window + non-hazard images

"fire_no_gas" episodes are scored as fires; whether a node should alarm on
vision alone is a policy choice, and the table makes that trade-off visible.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from firenode.config import load_config, resolve  # noqa: E402
from firenode.features import FEATURES, TARGET, chronological_split, load_iot_csv  # noqa: E402
from firenode.fusion import AndRuleBaseline, FusionConfig, FusionEngine, State  # noqa: E402
from firenode.models import SensorModel, VisionModel  # noqa: E402

EPISODE_LEN = 60


class Threshold:
    """Single-modality per-frame rule."""

    def __init__(self, use, threshold):
        self.use, self.t = use, threshold

    def reset(self):
        pass

    def update(self, s, v):
        x = s if self.use == "sensor" else v
        return type("D", (), {"state": State.ALARM if x >= self.t else State.NORMAL})


def score_images(vision_path, split, cfg):
    cache = ROOT / "results" / f"vision_{split}_scores.csv"
    if cache.exists() and cache.stat().st_mtime > Path(vision_path).stat().st_mtime:
        return pd.read_csv(cache)
    split_file = ROOT / "splits" / f"{split}.txt"
    images = [l for l in split_file.read_text().splitlines() if l]
    model = VisionModel(vision_path, cfg["vision"]["hazard_classes"], cfg["vision"]["imgsz"],
                        conf=0.05)
    rows = []
    for i in range(0, len(images), 32):
        batch = images[i:i + 32]
        for img, conf in zip(batch, model.hazard_confidence(batch)):
            label = Path(img.replace("/images/", "/labels/")).with_suffix(".txt")
            classes = {line.split()[0] for line in label.read_text().splitlines() if line.strip()}
            rows.append({"image": Path(img).name, "hazard": int(bool(classes & {"0", "2"})),
                         "conf": float(conf)})
    df = pd.DataFrame(rows)
    cache.parent.mkdir(exist_ok=True)
    df.to_csv(cache, index=False)
    return df


def sensor_windows(df, label, n, rng):
    """n windows of EPISODE_LEN consecutive readings whose labels are all `label`."""
    starts = []
    for _, g in df.groupby("session"):
        y = g[TARGET].to_numpy()
        ok = np.convolve(y == label, np.ones(EPISODE_LEN, int), "valid") == EPISODE_LEN
        idx = np.flatnonzero(ok)
        starts += [(g, i) for i in idx]
    picks = rng.choice(len(starts), n, replace=len(starts) < n)
    return [starts[p][0].iloc[starts[p][1]:starts[p][1] + EPISODE_LEN] for p in picks]


def run_policy(policy, sensor, vision):
    policy.reset()
    states = [policy.update(s, v).state for s, v in zip(sensor, vision)]
    alarm = [st == State.ALARM for st in states]
    first = alarm.index(True) if any(alarm) else None
    toggles = sum(1 for a, b in zip(alarm, alarm[1:]) if a != b)
    return first, toggles, float(np.mean(alarm))


def build_episodes(split, cfg, sensor_model, n, rng):
    parts = chronological_split(load_iot_csv(ROOT / "smoke_detection_iot.csv"))
    data = parts[split].copy()
    data["prob"] = sensor_model.predict_proba(data)
    img = score_images(resolve(cfg["models"]["vision"]), split, cfg)
    hazard_conf = img[img.hazard == 1].conf.to_numpy()
    clear_conf = img[img.hazard == 0].conf.to_numpy()

    def make(label, pool):
        # (sensor probabilities, camera confidences, raw readings)
        return [(w["prob"].to_numpy(), rng.choice(pool, EPISODE_LEN), w[FEATURES])
                for w in sensor_windows(data, label, n, rng)]

    episodes = {
        "fire_both": make(1, hazard_conf),
        "fire_no_camera": make(1, clear_conf),
        "fire_no_gas": make(0, hazard_conf),
        "no_fire": make(0, clear_conf),
    }
    vision_stats = {"images": int(len(img)), "hazard_images": int(img.hazard.sum()),
                    "frame_recall@0.45": round(float((hazard_conf >= 0.45).mean()), 3),
                    "frame_false_positive_rate@0.45": round(float((clear_conf >= 0.45).mean()), 3)}
    return episodes, vision_stats


def evaluate(policy, episodes):
    row = {}
    for ename, eps in episodes.items():
        res = [run_policy(policy, s, v) for s, v, _ in eps]
        lat = [r[0] for r in res if r[0] is not None]
        row[ename] = {
            "alarm_rate": round(float(np.mean([r[0] is not None for r in res])), 3),
            "median_latency_frames": float(np.median(lat)) if lat else None,
            "mean_toggles": round(float(np.mean([r[1] for r in res])), 2),
            "alarm_time_fraction": round(float(np.mean([r[2] for r in res])), 3),
        }
    return row


def tune(cfg, episodes, max_false_alarm=0.01):
    """Grid search on the validation episodes: catch the most fires of all three
    kinds while keeping the no-fire false-alarm rate at or below the limit."""
    best, best_key = None, None
    for ws in (0.4, 0.5, 0.6, 0.7, 0.8):
        for alarm_on in (0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6):
            for vt in (0.35, 0.45, 0.55):
                fc = dict(cfg["fusion"], sensor_weight=ws, vision_weight=round(1 - ws, 2),
                          alarm_on=alarm_on, vision_threshold=vt,
                          warning_on=round(alarm_on - 0.1, 2),
                          clear_below=round(min(0.25, alarm_on - 0.15), 2))
                row = evaluate(FusionEngine(FusionConfig.from_dict(fc)), episodes)
                if row["no_fire"]["alarm_rate"] > max_false_alarm:
                    continue
                detect = np.mean([row[e]["alarm_rate"]
                                  for e in ("fire_both", "fire_no_camera", "fire_no_gas")])
                toggles = np.mean([row[e]["mean_toggles"] for e in row])
                key = (round(float(detect), 3), -toggles)
                if best_key is None or key > best_key:
                    best, best_key = fc, key
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=int, default=200, help="per episode type")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tune", action="store_true",
                    help="grid-search fusion settings on the validation split and save them "
                         "to config/tuned_fusion.yaml before testing")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    cfg = load_config(tuned=False)
    tuned_path = ROOT / "config" / "tuned_fusion.yaml"
    sensor_model = SensorModel(resolve(cfg["models"]["sensor"]))

    if args.tune:
        val_eps, _ = build_episodes("val", cfg, sensor_model, args.episodes, rng)
        best = tune(cfg, val_eps)
        if best is None:
            raise SystemExit("no setting met the false-alarm limit on validation")
        import yaml
        tuned_path.write_text("# Written by scripts/benchmark_fusion.py --tune "
                              "(validation split)\n" + yaml.safe_dump({"fusion": best}))
        print(f"tuned fusion settings -> {tuned_path.relative_to(ROOT)}: {best}")
    if tuned_path.exists():
        cfg = load_config(tuned_path)

    episodes, vision_stats = build_episodes("test", cfg, sensor_model, args.episodes, rng)
    policies = {
        "sensor_only": Threshold("sensor", 0.5),
        "vision_only": Threshold("vision", 0.45),
        "v1_and_rule": AndRuleBaseline(0.5, 0.45),
        "v2_fusion": FusionEngine(FusionConfig.from_dict(cfg["fusion"])),
    }
    table = {name: evaluate(p, episodes) for name, p in policies.items()}

    out = {"split": "test", "episode_len": EPISODE_LEN, "episodes_per_type": args.episodes,
           "fusion_config": cfg["fusion"], "vision_frames": vision_stats, "policies": table}
    (ROOT / "results" / "fusion_benchmark.json").write_text(json.dumps(out, indent=2))

    print(json.dumps(vision_stats))
    print(f"\n{'policy':14s}" + "".join(f"{e:>18s}" for e in episodes))
    for pname, row in table.items():
        print(f"{pname:14s}" + "".join(
            f"{row[e]['alarm_rate']:>9.2f} ({row[e]['mean_toggles']:>4.1f}t)" for e in episodes))
    print("\ncells: fraction of episodes that reached ALARM (mean ALARM on/off toggles)")


if __name__ == "__main__":
    main()
