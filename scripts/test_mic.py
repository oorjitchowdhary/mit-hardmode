"""Quick test — record 3 seconds from the microphone with OLED feedback."""
from src.audio.microphone import MicrophoneManager
from src.display.oled import OLEDDisplay

d = OLEDDisplay()
mic = MicrophoneManager()

print("Available audio devices:")
mic.list_devices()

d.show_status("Mic", "Recording 10s...")
print("\nRecording 10 seconds... speak now!")
audio = mic.record(seconds=10.0)
print(f"Recorded {len(audio)} samples.")

speech = mic.is_speech(audio)
print(f"Speech detected: {speech}")
d.show_status("Mic", "Speech: " + ("YES" if speech else "NO"))

mic.save_wav("/tmp/test_mic.wav", seconds=10.0)
print("Saved to /tmp/test_mic.wav — play with: aplay /tmp/test_mic.wav")

import time
time.sleep(3)
d.clear()
print("Done.")
