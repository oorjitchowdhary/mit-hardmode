"""Test servo movements interactively with OLED display."""
import time
from gpiozero import Servo
from gpiozero.pins.lgpio import LGPIOFactory
from src.display.oled import OLEDDisplay

factory = LGPIOFactory()
servo = Servo(12, pin_factory=factory)
d = OLEDDisplay()

print("Servo test — GPIO 12")
print("=" * 40)

# 1. Go to known positions
d.show_status("Test 1/7", "Fixed positions")
print("\n1. Fixed positions")
d.show_text("0 degrees (min)")
print("   0° (min)...")
servo.min()
time.sleep(1)
d.show_text("90 degrees (mid)")
print("   90° (mid)...")
servo.mid()
time.sleep(1)
d.show_text("180 degrees (max)")
print("   180° (max)...")
servo.max()
time.sleep(1)
d.show_text("Back to 90 degrees")
print("   Back to 90°...")
servo.mid()
time.sleep(1)

# 2. Slow sweep 0° → 180°
d.show_status("Test 2/7", "Slow sweep 0-180")
print("\n2. Slow sweep 0° → 180°")
for i in range(31):
    servo.value = -1.0 + (2.0 * i / 30)
    time.sleep(0.08)
time.sleep(0.5)

# 3. Fast sweep 0° → 180°
d.show_status("Test 3/7", "Fast sweep 0-180")
print("3. Fast sweep 0° → 180°")
servo.min()
time.sleep(0.3)
for i in range(31):
    servo.value = -1.0 + (2.0 * i / 30)
    time.sleep(0.02)
time.sleep(0.5)

# 4. Snap between extremes
d.show_status("Test 4/7", "Snap min <-> max")
print("4. Snap min <-> max (3x)")
for _ in range(3):
    servo.min()
    time.sleep(0.3)
    servo.max()
    time.sleep(0.3)
time.sleep(0.5)

# 5. Small oscillation around center
d.show_status("Test 5/7", "Small oscillation")
print("5. Small oscillation around 90°")
for _ in range(5):
    servo.value = -0.3
    time.sleep(0.2)
    servo.value = 0.3
    time.sleep(0.2)
time.sleep(0.5)

# 6. Continuous value ramp (smooth)
d.show_status("Test 6/7", "Smooth ramp 0-180-0")
print("6. Smooth ramp 0° → 180° → 0°")
for i in range(61):
    val = -1.0 + (2.0 * i / 30) if i <= 30 else 1.0 - (2.0 * (i - 30) / 30)
    servo.value = val
    time.sleep(0.04)
time.sleep(0.5)

# 7. Try different speed values (for continuous rotation servos)
d.show_status("Test 7/7", "Continuous rotation")
print("\n7. Continuous rotation test")
d.show_text("value=0.1 (slow)")
print("   value=0.1 (slow)...")
servo.value = 0.1
time.sleep(2)
d.show_text("value=0.5 (medium)")
print("   value=0.5 (medium)...")
servo.value = 0.5
time.sleep(2)
d.show_text("value=1.0 (fast)")
print("   value=1.0 (fast)...")
servo.value = 1.0
time.sleep(2)
d.show_text("value=0.0 (stop)")
print("   value=0.0 (stop)...")
servo.value = 0.0
time.sleep(1)

# 8. Full 0° → 360° sweep (beyond standard servo range)
d.show_status("Test 8/8", "Full 0-360 sweep")
print("\n8. Full 0° → 360° sweep")
print("   Going to -1.0 (0°)...")
servo.value = -1.0
time.sleep(0.5)
for i in range(61):
    # -1.0 → 1.0 maps to 0° → 180°, but we push beyond
    # Standard servos will clamp at 180°; continuous ones may keep going
    val = -1.0 + (2.0 * i / 60)
    servo.value = max(-1.0, min(1.0, val))
    time.sleep(0.06)
time.sleep(1)

# Done
d.show_status("Done", "All tests complete")
print("\nDone. Detaching servo.")
servo.mid()
time.sleep(0.3)
servo.detach()
time.sleep(2)
d.clear()
