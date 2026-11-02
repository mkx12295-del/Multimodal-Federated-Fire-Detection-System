# Image for the federated aggregator and sensor-only fire nodes (no camera, no torch).
FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1

COPY requirements-node.txt .
RUN pip install --no-cache-dir -r requirements-node.txt

COPY firenode firenode
COPY federated federated
COPY config/default.yaml config/tuned_fusion.yaml config/
COPY models/sensor_xgb.json models/
COPY deploy deploy
COPY inference.py smoke_detection_iot.csv ./

CMD ["python", "-m", "federated.server", "--help"]
