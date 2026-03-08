"""Servo motor controller for vibe feedback."""
from __future__ import annotations

import threading


class ServoController:
    """
    Controls a continuous rotation servo.

    Value controls speed and direction:
      1.0  = full speed one direction
      0.0  = stop
     -1.0  = full speed other direction

    Idle: medium speed
    Good: slow
    Bad:  fast
    """

    def __init__(self, gpio_pin: int = 12) -> None:
        from gpiozero import Servo
        from gpiozero.pins.lgpio import LGPIOFactory

        factory = LGPIOFactory()
        self._servo = Servo(gpio_pin, pin_factory=factory)
        self._lock = threading.Lock()
        # Start at medium speed
        self._servo.value = 0.3

    def good_vibe(self) -> None:
        """Slow rotation."""
        with self._lock:
            self._servo.value = 0.1

    def bad_vibe(self) -> None:
        """Fast rotation."""
        with self._lock:
            self._servo.value = 0.8

    def stop(self) -> None:
        self._servo.value = 0.0
        self._servo.detach()

    def __enter__(self) -> "ServoController":
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()
