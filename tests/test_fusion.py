from firenode.fusion import AndRuleBaseline, FusionConfig, FusionEngine, State


def feed(engine, frames):
    return [engine.update(s, v).state for s, v in frames]


def test_quiet_inputs_stay_normal():
    states = feed(FusionEngine(), [(0.02, 0.0)] * 20)
    assert set(states) == {State.NORMAL}


def test_both_modalities_agreeing_raises_alarm_quickly():
    cfg = FusionConfig(confirm_frames=2)
    states = feed(FusionEngine(cfg), [(0.9, 0.8)] * 3)
    assert states[1] == State.ALARM


def test_single_noisy_frame_only_warns():
    states = feed(FusionEngine(), [(0.02, 0.0)] * 5 + [(0.95, 0.9)] + [(0.02, 0.0)] * 5)
    assert State.ALARM not in states
    assert State.WARNING in states


def test_sensor_only_fire_still_alarms():
    # Camera sees nothing (smoke out of view) but gas readings stay high.
    states = feed(FusionEngine(FusionConfig(sensor_weight=0.7, vision_weight=0.3)),
                  [(1.0, 0.0)] * 30)
    assert State.ALARM in states


def test_alarm_latches_until_clear_frames():
    cfg = FusionConfig(clear_frames=5)
    engine = FusionEngine(cfg)
    feed(engine, [(0.95, 0.9)] * 5)
    assert engine.state == State.ALARM
    states = feed(engine, [(0.0, 0.0)] * 20)
    first_normal = states.index(State.NORMAL)
    assert all(s == State.ALARM for s in states[:first_normal])
    assert first_normal >= cfg.clear_frames - 1


def test_alarm_does_not_flap_on_alternating_frames():
    engine = FusionEngine()
    feed(engine, [(0.95, 0.9)] * 5)
    states = feed(engine, [(0.95, 0.9), (0.1, 0.0)] * 10)
    assert set(states) == {State.ALARM}


def test_missing_camera_uses_sensor_score():
    engine = FusionEngine()
    d = engine.update(0.8, None)
    assert d.fused == 0.8 and not d.vision_alert


def test_weights_are_normalized():
    d = FusionEngine(FusionConfig(sensor_weight=3, vision_weight=1)).update(1.0, 0.0)
    assert abs(d.fused - 0.75) < 1e-9


def test_and_rule_baseline_matches_v1_behaviour():
    b = AndRuleBaseline()
    assert b.update(0.9, 0.9).state == State.ALARM
    assert b.update(0.9, 0.0).state == State.WARNING
    assert b.update(0.1, 0.0).state == State.NORMAL
