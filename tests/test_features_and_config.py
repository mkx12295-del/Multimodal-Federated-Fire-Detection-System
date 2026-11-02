import pandas as pd
import pytest

from firenode.config import load_config
from firenode.features import chronological_split, estimate_tvoc


def test_tvoc_proxy_rises_as_resistance_drops():
    baseline = 50_000
    assert estimate_tvoc(baseline, baseline) == 0
    assert estimate_tvoc(25_000, baseline) == pytest.approx(100)
    assert estimate_tvoc(10_000, baseline) > estimate_tvoc(25_000, baseline)
    assert estimate_tvoc(0, baseline) == 0
    assert estimate_tvoc(80_000, baseline) == 0


def test_chronological_split_keeps_time_order_per_session():
    df = pd.DataFrame({"session": [0] * 10 + [1] * 10, "UTC": list(range(10)) * 2,
                       "Fire Alarm": [0] * 20})
    parts = chronological_split(df, val=0.2, test=0.2)
    for s in (0, 1):
        tr = parts["train"][parts["train"].session == s].UTC
        va = parts["val"][parts["val"].session == s].UTC
        te = parts["test"][parts["test"].session == s].UTC
        assert tr.max() < va.min() and va.max() < te.min()
        assert (len(tr), len(va), len(te)) == (6, 2, 2)


def test_config_override_merges(tmp_path):
    override = tmp_path / "o.yaml"
    override.write_text("fusion:\n  alarm_on: 0.9\n")
    cfg = load_config(override, tuned=False)
    assert cfg["fusion"]["alarm_on"] == 0.9
    assert cfg["fusion"]["clear_below"] == 0.25
    assert cfg["hardware"]["buzzer_pin"] == 27
