"""
Test stepper motor (28BYJ-48) with ULN2003 driver — clock-style rotation.

Wiring (ULN2003 → Pi 5):
    IN1  → GPIO 17 (Pin 11)
    IN2  → GPIO 27 (Pin 13)
    IN3  → GPIO 22 (Pin 15)
    IN4  → GPIO 5  (Pin 29)
    5V   → Pin 2 or 4
    GND  → Pin 14

The 28BYJ-48 has 2048 steps per revolution (full-step mode).
"""
import time
from gpiozero import OutputDevice
from gpiozero.pins.lgpio import LGPIOFactory
from src.display.oled import OLEDDisplay

factory = LGPIOFactory()

# 4 control pins in order
PINS = [17, 27, 22, 5]
coils = [OutputDevice(p, pin_factory=factory) for p in PINS]

# Full-step sequence (4 steps)
FULL_STEP_SEQ = [
    [1, 0, 0, 1],
    [1, 1, 0, 0],
    [0, 1, 1, 0],
    [0, 0, 1, 1],
]

# Half-step sequence (8 steps, smoother)
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


def step(seq, steps, delay):
    """Move the motor a given number of steps. Positive=CW, negative=CCW."""
    direction = 1 if steps > 0 else -1
    idx = 0
    for _ in range(abs(steps)):
        for i, coil in enumerate(coils):
            coil.value = seq[idx % len(seq)][i]
        idx += direction
        time.sleep(delay)
    # Release coils to prevent heating
    for c in coils:
        c.off()


d = OLEDDisplay()

print("Stepper Motor test — 28BYJ-48 + ULN2003")
print("=" * 45)

# 1. Quarter turn clockwise
d.show_status("Test 1/6", "Quarter turn CW")
print("\n1. Quarter turn clockwise (90°)")
step(HALF_STEP_SEQ, STEPS_PER_REV // 4, 0.001)
time.sleep(1)

# 2. Quarter turn counter-clockwise
d.show_status("Test 2/6", "Quarter turn CCW")
print("2. Quarter turn counter-clockwise (90°)")
step(HALF_STEP_SEQ, -(STEPS_PER_REV // 4), 0.001)
time.sleep(1)

# 3. Full 360° rotation
d.show_status("Test 3/6", "Full 360 rotation")
print("3. Full 360° rotation")
step(HALF_STEP_SEQ, STEPS_PER_REV, 0.001)
time.sleep(1)

# 4. Slow clock — good vibe
d.show_status("Test 4/6", "GOOD vibe (slow)")
print("\n4. Good vibe — slow clock rotation")
step(HALF_STEP_SEQ, STEPS_PER_REV, 0.003)
time.sleep(1)

# 5. Fast clock — bad vibe
d.show_status("Test 5/6", "BAD vibe (fast)")
print("5. Bad vibe — fast clock rotation")
step(HALF_STEP_SEQ, STEPS_PER_REV, 0.0008)
time.sleep(1)

# 6. Clock tick — step by step like a second hand
d.show_status("Test 6/6", "Clock tick mode")
print("6. Clock tick — 12 ticks around the dial")
tick_steps = STEPS_PER_REV // 12  # 30° per tick
for i in range(12):
    d.show_text(f"Tick {i + 1}/12")
    print(f"   Tick {i + 1}/12")
    step(HALF_STEP_SEQ, tick_steps, 0.001)
    time.sleep(0.5)

# Done
d.show_status("Done", "All tests complete")
print("\nDone.")
time.sleep(2)
d.clear()
