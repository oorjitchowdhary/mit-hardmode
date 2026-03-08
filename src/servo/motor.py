"""Servo motor controller for vibe feedback."""
from __future__ import annotations

import time
import threading


class ServoController:
    """
    Controls a servo motor via gpiozero on a PWM-capable GPIO pin.

    Starts with a medium-speed idle sweep. On vibe updates:
      Good → slow gentle sweep
      Bad  → fast aggressive sweep
    Then returns to idle.
    """

    def __init__(self, gpio_pin: int = 12) -> None:
        from gpiozero import Servo
        from gpiozero.pins.lgpio import LGPIOFactory

        factory = LGPIOFactory()
        self._servo = Servo(gpio_pin, pin_factory=factory)
        self._servo.mid()
        self._lock = threading.Lock()
        self._running = True
        # Delay between sweep steps — lower = faster
        self._speed = 0.04  # medium idle speed
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        """Continuous sweep at current speed."""
        while self._running:
            with self._lock:
                speed = self._speed
            if speed is None:
                # Stopped (good vibe) — hold center
                self._servo.mid()
                time.sleep(0.1)
                continue
            for pos in _interpolate(-1.0, 1.0, steps=30):
                if not self._running:
                    return
                with self._lock:
                    speed = self._speed
                if speed is None:
                    break
                self._servo.value = pos
                time.sleep(speed)
            for pos in _interpolate(1.0, -1.0, steps=30):
                if not self._running:
                    return
                with self._lock:
                    speed = self._speed
                if speed is None:
                    break
                self._servo.value = pos
                time.sleep(speed)

    def good_vibe(self) -> None:
        """Slow down and stop."""
        with self._lock:
            self._speed = None

    def bad_vibe(self) -> None:
        """Speed up."""
        with self._lock:
            self._speed = 0.015

    def stop(self) -> None:
        self._running = False
        self._reaction_event.set()  # unblock loop
        self._thread.join(timeout=2.0)
        self._servo.mid()
        self._servo.detach()

    def __enter__(self) -> "ServoController":
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()


def _interpolate(start: float, end: float, steps: int) -> list[float]:
    """Generate evenly spaced values from start to end."""
    return [start + (end - start) * i / steps for i in range(steps + 1)]
