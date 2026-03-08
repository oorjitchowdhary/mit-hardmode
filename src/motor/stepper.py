"""
Stepper motor controller — 28BYJ-48 + ULN2003.

Wiring (ULN2003 → Pi 5):
    IN1  → GPIO 17 (Pin 11)
    IN2  → GPIO 27 (Pin 13)
    IN3  → GPIO 22 (Pin 15)
    IN4  → GPIO 5  (Pin 29)
    5V   → Pin 2 or 4
    GND  → Pin 14

Runs a background thread that continuously ticks like a clock second hand.
Vibe reactions change the speed, then it returns to normal.
"""
from __future__ import annotations

import threading
import time


HALF_STEP_SEQ = [
    [1, 0, 0, 0],
    [1, 1, 0, 0],
    [0, 1, 0, 0],
    [0, 1, 1, 0],
    [0, 0, 1, 0],
    [0, 0, 1, 1],
    [0, 0, 0, 1],
    [1, 0, 0, 1],
]

STEPS_PER_REV = 4096  # half-step mode
TICK_STEPS = STEPS_PER_REV // 60  # one "second" tick = 6°

# Delays
NORMAL_DELAY = 0.002   # normal clock tick speed
SLOW_DELAY = 0.006     # good vibe — slow
FAST_DELAY = 0.001     # bad vibe — fast


class StepperClock:
    """
    Stepper motor that ticks like a clock second hand.

    Default: ticks at normal speed continuously.
    Good vibe: gradually slows down over 5 seconds, then stops.
    Bad vibe: speeds up to 2x for 10 full revolutions, then back to normal.
    """

    def __init__(self, pins: tuple[int, ...] = (17, 27, 22, 5)) -> None:
        from gpiozero import OutputDevice
        from gpiozero.pins.lgpio import LGPIOFactory

        factory = LGPIOFactory()
        self._coils = [OutputDevice(p, pin_factory=factory) for p in pins]
        self._lock = threading.Lock()
        self._running = True
        self._step_idx = 0

        # Motor state: "normal", "good", "bad"
        self._mode = "normal"
        self._mode_lock = threading.Lock()

        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _do_step(self, delay: float) -> None:
        """Advance one half-step."""
        pattern = HALF_STEP_SEQ[self._step_idx % len(HALF_STEP_SEQ)]
        for i, coil in enumerate(self._coils):
            coil.value = pattern[i]
        self._step_idx += 1
        time.sleep(delay)

    def _release(self) -> None:
        """Turn off all coils to prevent heating."""
        for c in self._coils:
            c.off()

    def _loop(self) -> None:
        """Main loop: tick like a clock, react to vibe changes."""
        while self._running:
            with self._mode_lock:
                mode = self._mode

            if mode == "normal":
                self._tick_normal()
            elif mode == "good":
                self._tick_good()
                # After good finishes, stay stopped until next vibe check
                with self._mode_lock:
                    if self._mode == "good":
                        self._mode = "stopped"
            elif mode == "bad":
                self._tick_bad()
                # After bad finishes, return to normal
                with self._mode_lock:
                    if self._mode == "bad":
                        self._mode = "normal"
            elif mode == "stopped":
                self._release()
                time.sleep(0.1)

    def _tick_normal(self) -> None:
        """One tick at normal speed."""
        for _ in range(TICK_STEPS):
            if not self._running or self._mode_changed():
                return
            self._do_step(NORMAL_DELAY)
        # Pause between ticks (like a second hand pausing)
        self._release()
        self._sleep_interruptible(0.3)

    def _tick_good(self) -> None:
        """Gradually slow down over 5 seconds, then stop."""
        # 5 seconds of gradually slowing ticks
        total_ticks = 10
        for t in range(total_ticks):
            if not self._running:
                return
            # Linearly increase delay from normal to very slow
            frac = t / total_ticks
            delay = NORMAL_DELAY + (SLOW_DELAY - NORMAL_DELAY) * frac
            for _ in range(TICK_STEPS):
                if not self._running:
                    return
                self._do_step(delay)
            self._release()
            # Pause grows longer as it slows
            self._sleep_interruptible(0.3 + frac * 0.7)
        # Fully stopped
        self._release()

    def _tick_bad(self) -> None:
        """Speed up to 2x for 10 full revolutions, then back to normal."""
        total_steps = STEPS_PER_REV * 10
        for _ in range(total_steps):
            if not self._running:
                return
            self._do_step(FAST_DELAY)
        self._release()

    def _mode_changed(self) -> bool:
        """Check if mode was changed externally."""
        with self._mode_lock:
            return self._mode != "normal"

    def _sleep_interruptible(self, duration: float) -> None:
        """Sleep that can be interrupted by mode change or shutdown."""
        end = time.time() + duration
        while time.time() < end:
            if not self._running:
                return
            with self._mode_lock:
                if self._mode not in ("normal", "good", "stopped"):
                    return
            time.sleep(0.05)

    def good_vibe(self) -> None:
        """Gradually slow down and stop."""
        with self._mode_lock:
            self._mode = "good"

    def bad_vibe(self) -> None:
        """Speed up for 10 revolutions."""
        with self._mode_lock:
            self._mode = "bad"

    def stop(self) -> None:
        self._running = False
        self._thread.join(timeout=3.0)
        self._release()

    def __enter__(self) -> "StepperClock":
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()
