"""Central configuration — all hardware constants and API settings live here."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent.parent
MODELS_DIR = ROOT_DIR / "models"

# ── Anthropic / Claude ────────────────────────────────────────────────────────
ANTHROPIC_API_KEY: str = os.environ["ANTHROPIC_API_KEY"]
CLAUDE_MODEL: str = "claude-opus-4-6"

SYSTEM_PROMPT: str = (
    "You are a concise embedded AI assistant running on a Raspberry Pi 5. "
    "You have a camera, microphone, and a tiny 128×64 OLED display. "
    "Keep responses short — aim for 1-3 sentences that fit on the display. "
    "When describing images, be specific and brief."
)

# ── OLED display — Inland 1.3″ SH1106, 128×64, I2C ──────────────────────────
OLED_I2C_PORT: int = int(os.getenv("OLED_I2C_PORT", "1"))
# Default 0x3C; pull SA0 high on the module for 0x3D
OLED_I2C_ADDRESS: int = int(os.getenv("OLED_I2C_ADDRESS", "0x3C"), 16)
OLED_WIDTH: int = 128
OLED_HEIGHT: int = 64

# ── Raspberry Pi AI Camera (IMX500) ──────────────────────────────────────────
CAMERA_PREVIEW_SIZE: tuple[int, int] = (640, 480)
CAMERA_CAPTURE_SIZE: tuple[int, int] = (1920, 1080)
# .rpk network package for on-sensor inference (optional, leave empty to skip)
IMX500_MODEL_PATH: str = os.getenv(
    "IMX500_MODEL_PATH", str(MODELS_DIR / "imx500_mobilenet_ssd.rpk")
)

# ── Hailo AI HAT+ (Hailo-8L, 13 TOPS) ────────────────────────────────────────
# .hef = Hailo Executable Format compiled model
HAILO_HEF_PATH: str = os.getenv(
    "HAILO_HEF_PATH", str(MODELS_DIR / "yolov8s.hef")
)

# ── I2S MEMS Microphone — 28AWG, 4-pin ───────────────────────────────────────
# Wiring (Raspberry Pi 5):
#
#   Mic pin 1  VDD  ──→  3.3V      (physical pin 1 or 17)
#   Mic pin 2  GND  ──→  GND       (physical pin 6, 9, 14, …)
#   Mic pin 3  BCLK ──→  GPIO 18   (physical pin 12) — I2S Bit Clock
#   Mic pin 4  DOUT ──→  GPIO 20   (physical pin 38) — I2S Data Out
#
# If your mic has a 5th WS/LRCLK wire, connect it to GPIO 19 (pin 35).
# For 4-pin mics the WS line is usually tied to GND internally (left channel).
#
# Enable I2S in /boot/firmware/config.txt:
#   dtparam=i2s=on
#   dtoverlay=i2s-mmap
#
_audio_device_env = os.getenv("AUDIO_DEVICE", "")
AUDIO_DEVICE: int | str | None = (
    int(_audio_device_env) if _audio_device_env.isdigit()
    else (_audio_device_env or None)
)
AUDIO_SAMPLE_RATE: int = int(os.getenv("AUDIO_SAMPLE_RATE", "16000"))
AUDIO_CHANNELS: int = int(os.getenv("AUDIO_CHANNELS", "1"))
AUDIO_CHUNK_SIZE: int = 1024          # frames per callback block
AUDIO_RECORD_SECONDS: float = 5.0    # default capture duration
