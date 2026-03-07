#!/usr/bin/env bash
# scripts/setup.sh — first-boot setup for Raspberry Pi 5
# Run as the default 'pi' user (sudo will be invoked where needed).
set -euo pipefail

echo "╔══════════════════════════════════════════════════════╗"
echo "║  hardmode — Raspberry Pi 5 setup                    ║"
echo "╚══════════════════════════════════════════════════════╝"

# ── 0. Update system ──────────────────────────────────────────────────────────
echo ""
echo "▸ Updating system packages …"
sudo apt-get update -q
sudo apt-get upgrade -y -q

# ── 1. Enable hardware interfaces ─────────────────────────────────────────────
echo ""
echo "▸ Enabling I2C (OLED), SPI, I2S (microphone), camera …"

CONFIG=/boot/firmware/config.txt
sudo raspi-config nonint do_i2c 0          # enable I2C
sudo raspi-config nonint do_spi 0          # enable SPI (optional, for OLED over SPI)
sudo raspi-config nonint do_camera 0       # enable camera

# I2S for the microphone
grep -qxF 'dtparam=i2s=on'       "$CONFIG" || echo 'dtparam=i2s=on'       | sudo tee -a "$CONFIG"
grep -qxF 'dtoverlay=i2s-mmap'   "$CONFIG" || echo 'dtoverlay=i2s-mmap'   | sudo tee -a "$CONFIG"

# ── 2. System packages ────────────────────────────────────────────────────────
echo ""
echo "▸ Installing system packages …"
sudo apt-get install -y -q \
    python3-pip \
    python3-venv \
    python3-dev \
    python3-picamera2 \
    python3-libcamera \
    python3-kms++ \
    i2c-tools \
    portaudio19-dev \
    libasound2-dev \
    libopenblas-dev \
    git

# Verify I2C device (OLED should show at 0x3c)
echo ""
echo "▸ Scanning I2C bus 1 (OLED should appear at 0x3c) …"
i2cdetect -y 1 || true

# ── 3. Python virtual environment ─────────────────────────────────────────────
echo ""
echo "▸ Creating Python virtual environment …"
python3 -m venv .venv --system-site-packages
# shellcheck disable=SC1091
source .venv/bin/activate

pip install --upgrade pip wheel

# ── 4. Python dependencies ────────────────────────────────────────────────────
echo ""
echo "▸ Installing Python dependencies …"
pip install -e ".[dev]"

# ── 5. Hailo AI HAT+ ──────────────────────────────────────────────────────────
echo ""
echo "▸ Hailo AI HAT+ setup …"
echo ""
echo "  ┌─────────────────────────────────────────────────────────────────┐"
echo "  │  Hailo requires a manual download from the Developer Zone.      │"
echo "  │  1. Register/login at https://hailo.ai/developer-zone/          │"
echo "  │  2. Download 'HailoRT' for ARM64 (≥ v4.18 for Pi 5)            │"
echo "  │  3. Install:                                                     │"
echo "  │       sudo dpkg -i hailort_*_arm64.deb                          │"
echo "  │       pip install hailort                                        │"
echo "  │  4. Verify:  hailortcli fw-control identify                     │"
echo "  └─────────────────────────────────────────────────────────────────┘"

# Uncomment and adjust the line below once you have the deb package:
# sudo dpkg -i ~/Downloads/hailort_4.20.0_arm64.deb && pip install hailort

# ── 6. IMX500 model packages (optional) ──────────────────────────────────────
echo ""
echo "▸ IMX500 model packages (optional) …"
MODELS_DIR="$(pwd)/models"
mkdir -p "$MODELS_DIR"

# Install the Raspberry Pi imx500-models apt package which provides .rpk files
sudo apt-get install -y -q imx500-all 2>/dev/null || \
    echo "  (imx500-all not available — copy .rpk files to models/ manually)"

echo "  Pre-built .rpk files are installed to /usr/share/imx500-models/"
echo "  Copy the one you want: cp /usr/share/imx500-models/<model>.rpk models/"

# ── 7. Hailo .hef model download ─────────────────────────────────────────────
echo ""
echo "▸ Hailo model files …"
echo "  Download .hef files from https://github.com/hailo-ai/hailo_model_zoo"
echo "  and place them in $(pwd)/models/"
echo "  Example:  wget -P models/ https://...yolov8s.hef"

# ── 8. Environment file ───────────────────────────────────────────────────────
echo ""
echo "▸ Setting up .env …"
if [ ! -f .env ]; then
    cp .env.example .env
    echo "  Created .env from .env.example — add your ANTHROPIC_API_KEY!"
else
    echo "  .env already exists — skipping."
fi

# ── 9. Audio device check ─────────────────────────────────────────────────────
echo ""
echo "▸ Audio devices:"
python3 -c "import sounddevice; print(sounddevice.query_devices())" 2>/dev/null || \
    echo "  (sounddevice not yet installed — run after pip install)"

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  Setup complete!  Reboot to activate I2C / I2S.     ║"
echo "║                                                      ║"
echo "║  1. Add ANTHROPIC_API_KEY to .env                   ║"
echo "║  2. Install HailoRT (see step 5 above)              ║"
echo "║  3. Place .hef / .rpk models in models/             ║"
echo "║  4. sudo reboot                                     ║"
echo "║  5. source .venv/bin/activate && python -m src.main ║"
echo "╚══════════════════════════════════════════════════════╝"
