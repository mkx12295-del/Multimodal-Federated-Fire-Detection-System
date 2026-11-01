#!/bin/sh
# One fire node in the Docker demo:
#   1. join federated training with only this node's recording session
#      (the latest 20% of readings are held out),
#   2. run the node on the held-out readings with the federated model.
set -e
: "${NODE_ID:?}" "${SESSION:?}" "${AGGREGATOR:=http://aggregator:8080}"

cat > /tmp/node.yaml <<EOF
node:
  id: ${NODE_ID}
models:
  sensor_type: fl_mlp
  sensor_fl: /tmp/fl_node.npz
sensor:
  replay_session: ${SESSION}
  replay_start_fraction: 0.8
vision:
  enabled: false
reasoning:
  backend: rules
  cooldown_s: 20
  log_jsonl: /tmp/logs/incidents.jsonl
loop:
  log_csv: /tmp/logs/events.csv
EOF

python -m federated.client --server "${AGGREGATOR}" --name "${NODE_ID}" \
  --csv smoke_detection_iot.csv --session "${SESSION}" --holdout 0.2 --out /tmp/fl_node.npz

exec python inference.py --config /tmp/node.yaml --replay --no-camera --dashboard 8000
