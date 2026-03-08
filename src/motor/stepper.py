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
Vibe reactions change the speed, then returns to normal after the reaction.
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

# Delays (seconds per half-step)
NORMAL_DELAY = 0.002   # normal clock tick speed
GOOD_STOP_DELAY = 0.012  # very slow before stopping (6x slower than normal)
FAST_DELAY = 0.0005    # bad vibe — 4x faster than normal


class StepperClock:
    """
    Stepper motor that ticks like a clock second hand.

    Default: ticks at normal speed continuously.
    Good vibe: gradually slows over 5s then stops. Returns to normal after 15s.
    Bad vibe: 4x speed for 10 revolutions. Returns to normal after.
    """

    def __init__(self, pins: tuple[int, ...] = (17, 27, 22, 5)) -> None:
        from gpiozero import OutputDevice
        from gpiozero.pins.lgpio import LGPIOFactory

        factory = LGPIOFactory()
        self._coils = [OutputDevice(p, pin_factory=factory) for p in pins]
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

    def _get_mode(self) -> str:
        with self._mode_lock:
            return self._mode

    def _set_mode(self, mode: str) -> None:
        with self._mode_lock:
            self._mode = mode

    def _loop(self) -> None:
        """Main loop: tick like a clock, react to vibe changes."""
        while self._running:
            mode = self._get_mode()

            if mode == "normal":
                self._tick_normal()
            elif mode == "good":
                self._do_good_reaction()
            elif mode == "bad":
                self._do_bad_reaction()

    def _tick_normal(self) -> None:
        """One tick at normal speed (like a second hand)."""
        for _ in range(TICK_STEPS):
            if not self._running or self._get_mode() != "normal":
                return
            self._do_step(NORMAL_DELAY)
        # Pause between ticks
        self._release()
        self._sleep_check(0.3, "normal")

    def _do_good_reaction(self) -> None:
        """Gradually slow down over ~5 seconds, stop, wait 15s, resume normal."""
        # Phase 1: gradually slow down (15 ticks, getting slower)
        total_ticks = 15
        for t in range(total_ticks):
            if not self._running:
                return
            frac = t / total_ticks
            delay = NORMAL_DELAY + (GOOD_STOP_DELAY - NORMAL_DELAY) * frac
            for _ in range(TICK_STEPS):
                if not self._running:
                    return
                self._do_step(delay)
            self._release()
            # Pause grows longer as it slows (0.3s → 1.5s)
            self._sleep_check(0.3 + frac * 1.2, "good")

        # Phase 2: fully stopped for remaining time up to 15s total
        self._release()
        self._sleep_check(15.0, "good")

        # Return to normal
        self._set_mode("normal")

    def _do_bad_reaction(self) -> None:
        """Spin fast (4x speed) for 10 full revolutions, then normal for 15s."""
        # Phase 1: fast spin — 10 revolutions, no pauses, continuous
        total_steps = STEPS_PER_REV * 10
        for _ in range(total_steps):
            if not self._running:
                return
            self._do_step(FAST_DELAY)
        self._release()

        # Phase 2: back to normal ticking for 15 seconds
        self._set_mode("normal")

    def _sleep_check(self, duration: float, expected_mode: str) -> None:
        """Sleep that exits early if mode changes or shutdown."""
        end = time.time() + duration
        while time.time() < end:
            if not self._running or self._get_mode() != expected_mode:
                return
            time.sleep(0.05)

    def good_vibe(self) -> None:
        """Trigger good vibe reaction."""
        self._set_mode("good")

    def bad_vibe(self) -> None:
        """Trigger bad vibe reaction."""
        self._set_mode("bad")

    def stop(self) -> None:
        self._running = False
        self._thread.join(timeout=3.0)
        self._release()

    def __enter__(self) -> "StepperClock":
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()
