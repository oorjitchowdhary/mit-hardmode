"""Servo motor controller for vibe feedback."""
from __future__ import annotations

import time
import threading


# gpiozero Servo range: -1.0 = 0°, 0.0 = 90°, 1.0 = 180°
_POS_0 = -1.0    # 0 degrees
_POS_180 = 1.0   # 180 degrees
_POS_90 = 0.0    # 90 degrees (center)
_STEPS = 30      # number of steps per sweep


class ServoController:
    """
    Sweeps a servo continuously from 0° to 180° and back.

    Idle:  medium speed
    Good:  slow down and stop at 90°
    Bad:   speed up
    """

    def __init__(self, gpio_pin: int = 12) -> None:
        from gpiozero import Servo
        from gpiozero.pins.lgpio import LGPIOFactory

        factory = LGPIOFactory()
        self._servo = Servo(gpio_pin, pin_factory=factory)
        self._servo.value = _POS_90
        self._lock = threading.Lock()
        self._running = True
        self._speed = 0.04  # seconds per step (medium idle)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        """Continuous 0°→180° sweep, then snap back to 0° and repeat."""
        while self._running:
            with self._lock:
                speed = self._speed
            if speed is None:
                time.sleep(0.1)
                continue
            # Sweep 0° → 180°
            for pos in _interpolate(_POS_0, _POS_180, _STEPS):
                if not self._running:
                    return
                with self._lock:
                    speed = self._speed
                if speed is None:
                    break
                self._servo.value = pos
                time.sleep(speed)
            # Snap back to 0°
            self._servo.value = _POS_0
            time.sleep(0.05)

    def good_vibe(self) -> None:
        """Slow down and stop."""
        with self._lock:
            self._speed = 0.1

    def bad_vibe(self) -> None:
        """Speed up the sweep."""
        with self._lock:
            self._speed = 0.015

    def stop(self) -> None:
        self._running = False
        self._thread.join(timeout=2.0)
        self._servo.value = _POS_90
        self._servo.detach()

    def __enter__(self) -> "ServoController":
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()


def _interpolate(start: float, end: float, steps: int) -> list[float]:
    """Generate evenly spaced values from start to end."""
    return [start + (end - start) * i / steps for i in range(steps + 1)]
