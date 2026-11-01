"""Sensor feature schema and dataset handling shared by training, FL and the node."""
import numpy as np
import pandas as pd

FEATURES = ["Temperature", "Humidity", "TVOC", "Pressure"]
CSV_COLUMNS = {
    "Temperature[C]": "Temperature",
    "Humidity[%]": "Humidity",
    "TVOC[ppb]": "TVOC",
    "Pressure[hPa]": "Pressure",
}
TARGET = "Fire Alarm"


def load_iot_csv(path):
    """Load the smoke-detection IoT dataset with a recording-session id.

    The dataset is a concatenation of recording sessions; the CNT column
    restarts at 0 at the start of each one.
    """
    df = pd.read_csv(path)
    df = df.rename(columns=CSV_COLUMNS)
    df["session"] = (df["CNT"] == 0).cumsum() - 1
    return df


def chronological_split(df, val=0.15, test=0.15):
    """Split every session in time order: first part train, then val, then test.

    A random row split leaks information because neighbouring one-second
    readings are nearly identical; splitting in time order does not.
    """
    parts = {"train": [], "val": [], "test": []}
    for _, g in df.groupby("session", sort=True):
        g = g.sort_values("UTC")
        n = len(g)
        n_test, n_val = int(n * test), int(n * val)
        n_train = n - n_val - n_test
        parts["train"].append(g.iloc[:n_train])
        parts["val"].append(g.iloc[n_train:n_train + n_val])
        parts["test"].append(g.iloc[n_train + n_val:])
    return {k: pd.concat(v) for k, v in parts.items()}


def xy(df):
    return df[FEATURES].to_numpy(dtype=np.float64), df[TARGET].to_numpy(dtype=np.int64)


def estimate_tvoc(gas_resistance_ohms, baseline_ohms):
    """Rough TVOC (ppb) proxy from BME680 gas resistance.

    The model is trained on TVOC from an SGP30-type sensor, while the BME680
    only reports gas resistance (which drops as VOCs rise). This maps the
    resistance drop relative to a clean-air baseline onto a ppb-like scale.
    It is an approximation; Bosch's BSEC library gives a calibrated IAQ.
    """
    if gas_resistance_ohms <= 0 or baseline_ohms <= 0:
        return 0.0
    ratio = baseline_ohms / gas_resistance_ohms
    return max(0.0, (ratio - 1.0) * 100.0)
