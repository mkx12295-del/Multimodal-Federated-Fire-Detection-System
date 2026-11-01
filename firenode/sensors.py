"""Sensor sources: a real BME680 on the Pi, or a CSV replay for desktop runs."""
import itertools

import numpy as np

from .features import FEATURES, estimate_tvoc, load_iot_csv


class BME680Source:
    """Reads the BME680 over I2C and converts gas resistance to a TVOC proxy."""

    def __init__(self, burn_in_samples=30):
        import adafruit_bme680
        import board
        import busio

        i2c = busio.I2C(board.SCL, board.SDA)
        self.dev = adafruit_bme680.Adafruit_BME680_I2C(i2c)
        self.dev.sea_level_pressure = 1013.25
        self.baseline = self._calibrate(burn_in_samples)

    def _calibrate(self, n):
        # The gas heater needs a few readings to settle; use the median of the
        # burn-in window as the clean-air baseline.
        readings = [self.dev.gas for _ in range(max(1, n))]
        return float(np.median(readings))

    def read(self):
        d = self.dev
        return {
            "Temperature": d.temperature,
            "Humidity": d.relative_humidity,
            "TVOC": estimate_tvoc(d.gas, self.baseline),
            "Pressure": d.pressure,
        }


class ReplaySource:
    """Replays recorded readings from the IoT dataset, looping at the end."""

    def __init__(self, csv_path, session=None):
        df = load_iot_csv(csv_path)
        if session is not None:
            df = df[df["session"] == session]
        self._rows = itertools.cycle(df[FEATURES].to_dict("records"))

    def read(self):
        return next(self._rows)


def open_source(cfg, force_replay=False):
    if not force_replay:
        try:
            return BME680Source(cfg["sensor"]["burn_in_samples"])
        except (ImportError, NotImplementedError, OSError, ValueError, RuntimeError) as e:
            print(f"BME680 unavailable ({e.__class__.__name__}); replaying recorded data.")
    from .config import resolve
    return ReplaySource(resolve(cfg["sensor"]["replay_csv"]))
