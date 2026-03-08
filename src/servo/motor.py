"""Servo motor controller for vibe feedback."""
from __future__ import annotations

import time
import threading


class ServoController:
    """
    Controls a servo motor via gpiozero on a PWM-capable GPIO pin.

    Good vibe: slow sweep then stop.
    Bad vibe: fast aggressive sweep.
    """

    def __init__(self, gpio_pin: int = 12) -> None:
        from gpiozero import Servo
        from gpiozero.pins.lgpio import LGPIOFactory

        factory = LGPIOFactory()
        self._servo = Servo(gpio_pin, pin_factory=factory)
        self._servo.mid()
        self._lock = threading.Lock()

    def good_vibe(self) -> None:
        """Slow gentle sweep: min → max → center, then stop."""
        with self._lock:
            self._servo.min()
            time.sleep(0.8)
            for pos in _interpolate(-1.0, 1.0, steps=20):
                self._servo.value = pos
                time.sleep(0.05)
            time.sleep(0.3)
            self._servo.mid()

    def bad_vibe(self) -> None:
        """Fast aggressive back-and-forth sweep."""
        with self._lock:
            for _ in range(3):
                self._servo.min()
                time.sleep(0.1)
                self._servo.max()
                time.sleep(0.1)
            self._servo.mid()

    def stop(self) -> None:
        self._servo.mid()
        self._servo.detach()

    def __enter__(self) -> "ServoController":
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()


def _interpolate(start: float, end: float, steps: int) -> list[float]:
    """Generate evenly spaced values from start to end."""
    return [start + (end - start) * i / steps for i in range(steps + 1)]
