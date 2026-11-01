# Multimodal Federated Fire Detection Node

Edge fire detection for a Raspberry Pi 5 that fuses a gas/temperature sensor
(BME680 + XGBoost) with a camera (YOLO11n). The sensor models are trained
across sites with federated learning, and a small local LLM explains alerts.
Every number below is produced by a script in this repo, on held-out data.

## What it does

| Part | Implementation |
|---|---|
| Sensor pathway | XGBoost on temperature, humidity, TVOC, pressure (`models/sensor_xgb.json`) |
| Vision pathway | YOLO11n fire/smoke detector, 416 px (`scripts/train_vision.py`) |
| Fusion | Weighted late fusion, EMA smoothing, NORMAL→WARNING→ALARM state machine with hysteresis (`firenode/fusion.py`) |
| Federated learning | NumPy MLP, FedAvg / FedProx / clipped equal-weight aggregation / DP-FedAvg; HTTP aggregator and node client (`federated/`) |
| Local LLM reasoning | Qwen2.5-0.5B-Instruct (GGUF, Q4_K_M) explains WARNING/ALARM events; may only escalate, never cancel (`firenode/reasoning.py`) |
| Dashboard | Live status page served by the node (`--dashboard PORT`) |
| Hardware | Buzzer GPIO27 (pin 13), relay GPIO17 (pin 11), BME680 on I2C |

## Results

### Vision (YOLO11n, held-out test split, 893 images)

| Version | Training | Evaluated on | mAP50 | mAP50-95 |
|---|---|---|---|---|
| v1 | 5 epochs, CPU, 320 px | its own training images | 0.373 | 0.182 |
| **v2** | 50 epochs, GPU, 416 px | **held-out test split** | **0.427** | **0.208** |

Per class mAP50 (v2): fire 0.62, smoke 0.43, other 0.23. Inference about 1.2 ms
per image on an RTX 5060 laptop GPU. Raspberry Pi latency has not been measured yet.

### Fusion (replay benchmark, test split)

No public dataset pairs camera frames and gas readings from the same fire,
so episodes (60 frames each, 200 per type) combine held-out sensor windows
with held-out test images. Fusion thresholds were tuned on the validation
split (`config/tuned_fusion.yaml`), then evaluated once on test.

Cells: share of episodes that reached ALARM, with the mean number of times the
alarm switched on or off per episode in brackets.

| Policy | Fire, both see it | Fire, camera blind | Fire, no gas yet | No fire (false alarm) |
|---|---|---|---|---|
| Sensor only | 1.00 (0.0) | 1.00 (0.0) | 0.03 (0.0) | 0.02 (0.0) |
| Vision only | 1.00 (21.6) | 0.96 (5.5) | 1.00 (21.4) | 0.94 (5.3) |
| v1 AND rule | 1.00 (21.6) | 0.96 (5.5) | 0.03 (0.3) | 0.01 (0.0) |
| **v2 fusion** | **1.00 (1.0)** | **1.00 (1.0)** | **0.92 (0.9)** | **0.01 (0.0)** |

v2 catches 92% of fires visible only to the camera (v1: 3%) at the same 1%
false-alarm rate, and does not flicker the alarm on and off.

### Sensor model (XGBoost)

| Split | Accuracy |
|---|---|
| Random row split (v1 protocol) | 99.5% |
| Chronological split per session | 99.8% |
| **Leave one session out (new site)** | **3% – 88%** depending on session |

The model is near-perfect on the sites it trained on but does not transfer to
an unseen site. This motivates training across sites with federated learning.

### Federated learning (5 non-IID nodes = 5 recording sessions)

| Method | Pooled test accuracy |
|---|---|
| Local only (each node alone, evaluated on all nodes) | 42.8% mean |
| FedAvg | 93.5% |
| FedProx (mu = 0.1) | 93.3% |
| Centralized (pooled raw data, no privacy) | 95.5% |
| **Clipped equal-weight aggregation** | **98.2% ± 1.5** (3 seeds) |
| Same, over HTTP with 5 separate processes | 98.8% |

DP-FedAvg (clip 1.0, delta 1e-3, 200 rounds, 3 seeds):

| Noise multiplier | epsilon | Accuracy |
|---|---|---|
| 0.1 | 10677 | 97.2% |
| 0.5 | 506 | 84.5% |
| 1.0 | 153 | 70.1% |
| 2.0 | 51 | 65.7% |

With only 5 nodes and full participation, a meaningful privacy budget costs a
lot of accuracy. More nodes per round would be needed for strong guarantees.

![FL convergence](results/federated_convergence.png)
![DP trade-off](results/dp_tradeoff.png)

### Local LLM reasoning: limitations seen so far

The 0.5B model runs locally (12 s per call on desktop CPU through
transformers). In a replay test it labeled a clean-air window "fire" with high
confidence and invented a TVOC reading, so it is advisory only. The
escalate-only policy keeps it from cancelling alarms. A rule-based fallback is
used when no model is available. `scripts/eval_reasoning.py` measures verdict
accuracy and escalation impact; its results are not yet reported here.

## Setup

```bash
pip install -r requirements.txt
python scripts/download_dataset.py      # needs ROBOFLOW_API_KEY (dataset: CC BY 4.0)
python train.py                         # splits, sensor, vision, FL, fusion stages
pytest -q                               # 30 tests, no hardware needed
```

Run the node without hardware (replays recorded sensor data):

```bash
python inference.py --replay --no-camera --dashboard 8000
```

Optional LLM model (about 470 MB):

```bash
python -c "from huggingface_hub import hf_hub_download; hf_hub_download('Qwen/Qwen2.5-0.5B-Instruct-GGUF','qwen2.5-0.5b-instruct-q4_k_m.gguf',local_dir='models')"
```

Networked federated training (one aggregator, one process per node):

```bash
python scripts/fl_network_demo.py --clip 1.0
```

Deploy to a Raspberry Pi 5 (SSH key auth, installs a systemd service):

```bash
python deploy.py --host <pi-address> --user pi
```

## Project structure

```
firenode/     node runtime: fusion, sensors, models, actuators, reasoning, dashboard
federated/    FL model, aggregation (FedAvg/FedProx/DP), HTTP server and client
scripts/      dataset, training, FL experiments, benchmarks
config/       default.yaml (+ tuned_fusion.yaml from the benchmark)
results/      metrics JSON and plots produced by the scripts
tests/        pytest suite (runs in CI)
deploy/       systemd unit
```

## Limitations

- Hardware code (BME680, GPIO, camera) has not been re-tested on a Pi after the v2 rewrite.
- No dataset pairs camera frames with gas readings from the same fire; fusion is
  evaluated on replay episodes assembled from held-out sensor windows and test images.
- Federated "nodes" are recording sessions of one public dataset, not separate deployments.
- The BME680 gives gas resistance, not TVOC; the conversion is approximate.

## Data

- Images: [Roboflow Fire Detection v1](https://universe.roboflow.com/fire-dataset-tp9jt/fire-detection-sejra/dataset/1), CC BY 4.0, 8,939 images.
- Sensor readings: `smoke_detection_iot.csv`, public smoke detection IoT dataset (62,630 rows).
