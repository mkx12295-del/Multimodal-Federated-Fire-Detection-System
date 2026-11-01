"""Late-fusion decision engine with temporal smoothing and hysteresis.

The original node raised the alarm only when the sensor model and the camera
agreed on the same frame, and cleared it on the very next frame that did not.
That misses fires one modality cannot see (smoke out of camera view, flame
before gas builds up) and makes the buzzer flap on noisy frames.

This engine fuses both scores into a weighted value, smooths it over time,
and drives a NORMAL -> WARNING -> ALARM state machine:

* ALARM when the smoothed score stays above ``alarm_on`` for ``alarm_frames``
  frames, or both modalities agree for ``confirm_frames`` frames.
* ALARM latches until the smoothed score stays below ``clear_below`` for
  ``clear_frames`` frames, so it does not toggle on and off.
"""
from dataclasses import dataclass
from enum import IntEnum


class State(IntEnum):
    NORMAL = 0
    WARNING = 1
    ALARM = 2


@dataclass
class FusionConfig:
    sensor_weight: float = 0.5
    vision_weight: float = 0.5
    sensor_threshold: float = 0.5
    vision_threshold: float = 0.35
    ema_alpha: float = 0.4
    warning_on: float = 0.35
    alarm_on: float = 0.6
    clear_below: float = 0.25
    alarm_frames: int = 3
    confirm_frames: int = 2
    clear_frames: int = 5

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class Decision:
    state: State
    fused: float
    smoothed: float
    sensor_alert: bool
    vision_alert: bool


class FusionEngine:
    def __init__(self, config=None):
        self.cfg = config or FusionConfig()
        total = self.cfg.sensor_weight + self.cfg.vision_weight
        if total <= 0:
            raise ValueError("fusion weights must sum to a positive value")
        self._ws = self.cfg.sensor_weight / total
        self._wv = self.cfg.vision_weight / total
        self.reset()

    def reset(self):
        self.state = State.NORMAL
        self.smoothed = 0.0
        self._above_alarm = 0
        self._agree = 0
        self._below_clear = 0

    def update(self, sensor_prob, vision_conf=None):
        """Feed one frame. ``vision_conf`` is None when no camera is available."""
        sensor_alert = sensor_prob >= self.cfg.sensor_threshold
        if vision_conf is None:
            fused = sensor_prob
            vision_alert = False
        else:
            fused = self._ws * sensor_prob + self._wv * vision_conf
            vision_alert = vision_conf >= self.cfg.vision_threshold

        a = self.cfg.ema_alpha
        self.smoothed = a * fused + (1 - a) * self.smoothed

        self._above_alarm = self._above_alarm + 1 if self.smoothed >= self.cfg.alarm_on else 0
        self._agree = self._agree + 1 if (sensor_alert and vision_alert) else 0
        self._below_clear = self._below_clear + 1 if self.smoothed < self.cfg.clear_below else 0

        escalate = (self._above_alarm >= self.cfg.alarm_frames
                    or self._agree >= self.cfg.confirm_frames)

        if self.state == State.ALARM:
            if self._below_clear >= self.cfg.clear_frames:
                self.state = State.NORMAL
        elif escalate:
            self.state = State.ALARM
        elif self.smoothed >= self.cfg.warning_on or sensor_alert or vision_alert:
            self.state = State.WARNING
        else:
            self.state = State.NORMAL

        return Decision(self.state, fused, self.smoothed, sensor_alert, vision_alert)


class AndRuleBaseline:
    """The original v1 rule: alarm only while both modalities fire on the same frame."""

    def __init__(self, sensor_threshold=0.5, vision_threshold=0.45):
        self.st, self.vt = sensor_threshold, vision_threshold

    def reset(self):
        pass

    def update(self, sensor_prob, vision_conf=None):
        s = sensor_prob > self.st
        v = vision_conf is not None and vision_conf >= self.vt
        state = State.ALARM if (s and v) else State.WARNING if (s or v) else State.NORMAL
        return Decision(state, float(s and v), float(s and v), s, v)
