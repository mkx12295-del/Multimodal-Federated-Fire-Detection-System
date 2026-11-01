"""Evaluate the reasoning module on the held-out replay episodes.

Two measurements, for the local LLM and for the rule-based fallback:

1. Verdict accuracy: summarize the last 30 frames of each episode and ask for
   fire / no_fire. Episode types are the same as in benchmark_fusion.py.
2. Escalation value: run the fusion engine on each episode; the first time it
   enters WARNING without having alarmed, ask the reasoner and escalate to
   ALARM on a "fire" verdict with confidence >= the configured threshold.
   Reports the alarm rate per episode type with and without the reasoner, so
   any extra fires caught and any extra false alarms are both visible.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_fusion import build_episodes  # noqa: E402
from firenode.config import load_config, resolve  # noqa: E402
from firenode.fusion import FusionConfig, FusionEngine, State  # noqa: E402
from firenode.models import SensorModel  # noqa: E402
from firenode.reasoning import Context, RuleReasoner, load_reasoner  # noqa: E402

WINDOW = 30
FIRE_TYPES = ("fire_both", "fire_no_camera", "fire_no_gas")


def context(ep, end, state, vision_threshold):
    probs, confs, raw = ep
    lo = max(0, end - WINDOW)
    rows = raw.iloc[lo:end].to_dict("records")
    return Context.from_window(rows, probs[lo:end], list(confs[lo:end]), state, vision_threshold)


def run_escalation(ep, cfg, reasoner, threshold):
    probs, confs, _ = ep
    engine = FusionEngine(FusionConfig.from_dict(cfg["fusion"]))
    asked = None
    for i, (s, v) in enumerate(zip(probs, confs)):
        st = engine.update(s, v).state
        if st == State.ALARM:
            return True, asked
        if st == State.WARNING and asked is None:
            verdict = reasoner.reason(context(ep, i + 1, "WARNING", cfg["fusion"]["vision_threshold"]))
            asked = verdict
            if verdict.is_fire and verdict.confidence >= threshold:
                return True, asked
    return False, asked


def evaluate(reasoner, episodes, cfg, threshold):
    out = {"verdict_accuracy": {}, "alarm_rate_with_reasoner": {}, "parse_rate": None,
           "latency_s": None}
    lat, parsed = [], []
    for etype, eps in episodes.items():
        truth = etype in FIRE_TYPES
        correct = []
        for ep in eps:
            v = reasoner.reason(context(ep, len(ep[0]), "WARNING", cfg["fusion"]["vision_threshold"]))
            correct.append(v.is_fire == truth)
            lat.append(v.latency_s)
            parsed.append(v.parsed)
        out["verdict_accuracy"][etype] = round(float(np.mean(correct)), 3)
        alarms = [run_escalation(ep, cfg, reasoner, threshold)[0] for ep in eps]
        out["alarm_rate_with_reasoner"][etype] = round(float(np.mean(alarms)), 3)
    out["verdict_accuracy"]["overall"] = round(float(np.mean(list(out["verdict_accuracy"].values()))), 3)
    out["parse_rate"] = round(float(np.mean(parsed)), 3)
    out["latency_s"] = {"median": round(float(np.median(lat)), 3),
                        "p95": round(float(np.percentile(lat, 95)), 3)}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=int, default=25, help="per episode type")
    ap.add_argument("--backend", default="auto")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    cfg = load_config()
    rng = np.random.default_rng(args.seed)
    sensor_model = SensorModel(resolve(cfg["models"]["sensor"]))
    episodes, _ = build_episodes("test", cfg, sensor_model, args.episodes, rng)
    threshold = cfg["reasoning"]["escalate_confidence"]

    baseline = {}
    for etype, eps in episodes.items():
        alarms = []
        for probs, confs, _ in eps:
            e = FusionEngine(FusionConfig.from_dict(cfg["fusion"]))
            alarms.append(any(e.update(s, v).state == State.ALARM for s, v in zip(probs, confs)))
        baseline[etype] = round(float(np.mean(alarms)), 3)

    t = time.time()
    llm = load_reasoner(dict(cfg["reasoning"], backend=args.backend))
    load_s = round(time.time() - t, 1)
    results = {
        "episodes_per_type": args.episodes,
        "escalate_confidence": threshold,
        "fusion_only_alarm_rate": baseline,
        "rules": evaluate(RuleReasoner(), episodes, cfg, threshold),
        llm.name: {"model": Path(cfg["reasoning"]["model_path"]).name, "load_s": load_s,
                   **evaluate(llm, episodes, cfg, threshold)},
    }
    (ROOT / "results" / "reasoning_eval.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
