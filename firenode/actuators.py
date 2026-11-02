"""Buzzer and relay outputs. Falls back to console output off the Pi."""
from .fusion import State


class _ConsoleOutput:
    def __init__(self, name):
        self.name, self.is_active = name, False

    def on(self):
        if not self.is_active:
            print(f"[{self.name}] ON")
        self.is_active = True

    def off(self):
        if self.is_active:
            print(f"[{self.name}] OFF")
        self.is_active = False

    def close(self):
        self.off()


class Actuators:
    """Buzzer sounds on WARNING and ALARM; the suppression relay only on ALARM."""

    def __init__(self, buzzer_pin, relay_pin, use_gpio=True):
        self.buzzer = self.relay = None
        if use_gpio:
            try:
                from gpiozero import Buzzer, OutputDevice

                self.buzzer = Buzzer(buzzer_pin)
                self.relay = OutputDevice(relay_pin, active_high=True, initial_value=False)
            except Exception as e:  # gpiozero raises several backend-specific errors
                print(f"GPIO unavailable ({e.__class__.__name__}); using console outputs.")
        if self.buzzer is None:
            self.buzzer, self.relay = _ConsoleOutput("buzzer"), _ConsoleOutput("relay")
        self._state = State.NORMAL

    def apply(self, state):
        if state == self._state:
            return
        self._state = state
        if state == State.ALARM:
            self.buzzer.on()
            self.relay.on()
        elif state == State.WARNING:
            if hasattr(self.buzzer, "beep"):
                self.buzzer.beep(on_time=0.1, off_time=0.9)
            else:
                self.buzzer.on()
            self.relay.off()
        else:
            self.buzzer.off()
            self.relay.off()

    def close(self):
        self.buzzer.off()
        self.relay.off()
