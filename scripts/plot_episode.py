"""Plot how the fusion engine reacts over time on held-out replay episodes.

One panel per episode type: sensor fire probability, camera fire/smoke
confidence, the smoothed fused score and the resulting state. Uses the same
test episodes and tuned settings as scripts/benchmark_fusion.py.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_fusion import build_episodes  # noqa: E402
from firenode.config import load_config, resolve  # noqa: E402
from firenode.fusion import AndRuleBaseline, FusionConfig, FusionEngine, State  # noqa: E402
from firenode.models import SensorModel  # noqa: E402

COLORS = {State.NORMAL: "#2e7d32", State.WARNING: "#ef8f00", State.ALARM: "#c62828"}
TITLES = {"fire_no_camera": "Fire, camera blind (smoke out of view)",
          "fire_no_gas": "Fire seen only by the camera (no gas yet)",
          "no_fire": "No fire"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = load_config()
    rng = np.random.default_rng(args.seed)
    episodes, _ = build_episodes("test", cfg, SensorModel(resolve(cfg["models"]["sensor"])), 5, rng)

    fig, axes = plt.subplots(len(TITLES), 1, figsize=(8, 7.5), sharex=True)
    for ax, (etype, title) in zip(axes, TITLES.items()):
        probs, confs, _ = episodes[etype][0]
        engine = FusionEngine(FusionConfig.from_dict(cfg["fusion"]))
        v1 = AndRuleBaseline(0.5, 0.45)
        smoothed, states, v1_alarm = [], [], []
        for s, v in zip(probs, confs):
            d = engine.update(s, v)
            smoothed.append(d.smoothed)
            states.append(d.state)
            v1_alarm.append(v1.update(s, v).state == State.ALARM)
        t = np.arange(len(probs))
        for i, st in enumerate(states):
            ax.axvspan(i - 0.5, i + 0.5, color=COLORS[st], alpha=0.12, lw=0)
        ax.plot(t, probs, lw=1, label="sensor probability")
        ax.plot(t, confs, lw=0.8, alpha=0.7, label="camera confidence")
        ax.plot(t, smoothed, lw=2, c="black", label="v2 smoothed score")
        ax.scatter(t[np.array(v1_alarm)], np.full(sum(v1_alarm), 1.05), marker="|", c="#7b1fa2",
                   label="v1 AND-rule alarm frames")
        ax.axhline(cfg["fusion"]["alarm_on"], ls=":", c="gray", lw=1)
        ax.set_ylim(-0.05, 1.12)
        ax.set_title(f"{title}: v2 ends in {states[-1].name}", fontsize=10, loc="left")
    axes[0].legend(fontsize=7, ncol=4, loc="lower right")
    axes[-1].set_xlabel("frame (1 per second); background = v2 state (green NORMAL, "
                        "orange WARNING, red ALARM)")
    fig.tight_layout()
    out = ROOT / "results" / "fusion_timeline.png"
    fig.savefig(out, dpi=120)
    print(f"saved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
