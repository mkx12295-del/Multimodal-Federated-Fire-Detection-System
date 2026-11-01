"""Full training pipeline: data splits, sensor model, vision model, FL experiment, fusion benchmark.

    python train.py              # everything (YOLO on CPU takes ~2 h)
    python train.py sensor fl    # only the listed stages
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STAGES = {
    "splits": ["scripts/make_splits.py"],
    "sensor": ["scripts/train_sensor.py"],
    "vision": ["scripts/train_vision.py"],
    "fl": ["scripts/run_federated.py"],
    "fusion": ["scripts/benchmark_fusion.py", "--tune"],
}


def main(argv):
    stages = argv or list(STAGES)
    unknown = set(stages) - set(STAGES)
    if unknown:
        raise SystemExit(f"unknown stage(s): {', '.join(sorted(unknown))}; "
                         f"choose from {', '.join(STAGES)}")
    for name in stages:
        print(f"\n===== {name} =====")
        subprocess.run([sys.executable, *STAGES[name]], cwd=ROOT, check=True)


if __name__ == "__main__":
    main(sys.argv[1:])
