"""Quick test — show text on the OLED display for 5 seconds."""
import time
from src.display.oled import OLEDDisplay

d = OLEDDisplay()
d.show_text("Hello Pi 5")
print("Text should be visible on the display.")
time.sleep(5)
d.clear()
print("Done.")
