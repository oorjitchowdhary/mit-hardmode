# hardmode

A Raspberry Pi 5 AI device that listens to conversations in real time, judges the vibe, and physically reacts — a "doom clock" that freezes when you're being productive and spins faster when you're not.

Built for MIT's hardmode hackathon.

---

## What it does

**hardmode** monitors ambient audio through a USB microphone, streams it to Deepgram for live transcription with speaker diarization, then periodically sends the transcript to Claude for a vibe assessment. The result is shown on a 128×64 OLED display and drives a stepper motor that mimics a clock second hand:

- **Good vibe** (productive conversation, studying, coding): clock slows and freezes. The doom clock stops counting.
- **Bad vibe** (brain rot, doomscrolling, passive media consumption): clock spins fast and jumps forward 4 hours.
- **Brain rot detected by audio alone** (music playing with no speech — TikToks, reels, background YouTube): skips Claude entirely and immediately scores bad.

The display shows a running `HH:MM:SS` doom clock alongside the current vibe label and score.

---

## Hardware

| Component | Part | Interface | Notes |
|---|---|---|---|
| Computer | Raspberry Pi 5 | — | Requires Pi OS Bookworm 64-bit |
| Display | Inland 1.3″ OLED, SH1106, 128×64 | I2C (bus 1, 0x3C) | `luma.oled` driver |
| Camera | Raspberry Pi AI Camera (IMX500) | CSI / libcamera | `picamera2`; supports on-sensor inference |
| AI Accelerator | Raspberry Pi AI HAT+ (Hailo-8L, 13 TOPS) | PCIe M.2 | `hailo_platform` / `hailort` |
| Microphone | 28AWG MEMS, 4-pin I2S | GPIO 18 (BCLK), GPIO 20 (DOUT) | `sounddevice` + ALSA |
| Stepper motor | 28BYJ-48 + ULN2003 driver | GPIO 17, 27, 22, 5 | Half-step mode, 4096 steps/rev |
| Servo (optional) | Standard PWM servo | GPIO 12 | Alternate motor for vibe feedback |

### Wiring — Stepper (ULN2003 → Pi 5)

```
IN1 → GPIO 17  (Pin 11)
IN2 → GPIO 27  (Pin 13)
IN3 → GPIO 22  (Pin 15)
IN4 → GPIO 5   (Pin 29)
5V  → Pin 2 or 4
GND → Pin 14
```

### Wiring — OLED (SH1106, I2C)

```
VCC → 3.3V  (Pin 1 or 17)
GND → GND   (Pin 6)
SCL → GPIO 3 (Pin 5)
SDA → GPIO 2 (Pin 3)
```

### Wiring — I2S Microphone

```
VDD  → 3.3V     (Pin 1 or 17)
GND  → GND      (Pin 6)
BCLK → GPIO 18  (Pin 12)
DOUT → GPIO 20  (Pin 38)
WS   → GPIO 19  (Pin 35)  ← if your mic has a 5th wire
```

---

## Architecture

All hardware constants (I2C address, GPIO assignments, audio settings, model paths, API keys) live in **`config/settings.py`** and are loaded from `.env` via `python-dotenv`. Nothing else reads the environment directly.

Each peripheral is a self-contained manager class:

| Module | Class | Responsibility |
|---|---|---|
| `src/llm/client.py` | `ClaudeClient` | Streaming Claude API calls; multi-turn history; vision support |
| `src/display/oled.py` | `OLEDDisplay` | SH1106 OLED; text wrap, status bars, progress bar, raw canvas |
| `src/camera/ai_camera.py` | `AICameraManager` | picamera2 capture; optional IMX500 on-sensor inference |
| `src/audio/microphone.py` | `MicrophoneManager` | Blocking record, streaming chunks, WAV save, energy VAD |
| `src/audio/classifier.py` | `AudioClassifier` | Spectral music/speech/silence detection; brainrot flag |
| `src/inference/hailo.py` | `HailoInferenceEngine` | Hailo-8L `.hef` model inference; YOLO detection parser |
| `src/motor/stepper.py` | `StepperClock` | 28BYJ-48 doom clock; reacts to good/bad vibe commands |
| `src/servo/motor.py` | `ServoController` | Servo sweep at variable speed for vibe feedback |

All manager classes are context managers (`__enter__` / `__exit__`).

---

## Main demo — Vibe Check (`scripts/test_vibe.py`)

The primary application. Runs a continuous 4-phase loop:

```
Phase 1 — LISTEN (15s)
  • Mic audio → Deepgram WebSocket (nova-2, diarization on)
  • Raw audio buffered for local classification
  • OLED shows countdown

Phase 2 — ANALYZE
  • AudioClassifier runs spectral analysis on buffered audio
    - Detects music vs speech vs silence from FFT band energies
    - Flags "brainrot" if music is dominant with no speech
  • If brainrot → score BAD immediately (no Claude call)
  • If audio but no transcript → "Passive media consumption" → BAD
  • Otherwise → send transcript + audio metadata to Claude
    - Claude returns SCORE (0/1) + VIBE (2-4 words)

Phase 3 — RESULT (15s)
  • OLED displays doom clock + score label + vibe description
  • StepperClock reacts:
      GOOD → slows down then stops; doom clock freezes
      BAD  → spins at 3× speed; doom clock jumps +4 hours

Phase 4 — NEUTRAL (15s)
  • Motor returns to normal tick speed
  • Clock resumes counting
  • Transcript buffer cleared, ready for next cycle
```

The doom clock is displayed as `HH:MM:SS` and is capped at `28:00:00`.

Three background threads run concurrently:
- **Thread 1**: streams mic PCM to Deepgram via WebSocket
- **Thread 2**: receives and stores diarized transcript lines
- **Thread 3**: runs the 4-phase vibe analysis loop

---

## Audio classifier

`src/audio/classifier.py` uses pure NumPy spectral analysis — no ML framework needed:

- Computes FFT over the audio buffer
- Measures energy in three bands: bass (20–300 Hz), speech formants (300–3000 Hz), highs (3–8 kHz)
- Measures energy variation over 0.5s windows (music is steady; speech is variable)
- Computes spectral flatness (tonal vs noisy)
- Scores `music_score` and `speech_score` independently
- Sets `is_brainrot = True` when music dominates with little speech

This runs locally on every cycle and can short-circuit the Claude API call when the answer is obvious.

---

## Claude API usage

- **Model**: `claude-opus-4-6` with `thinking: {"type": "adaptive"}` on every call
- **Streaming**: all requests use `.stream()` + `text_stream`; a per-token callback updates the OLED live
- **Vision**: PIL Image → base64 JPEG in the `image` source block; RGBA auto-converted to RGB
- **Multi-turn**: `ClaudeClient.chat()` maintains history; `ClaudeClient.ask()` is stateless (used for vibe analysis)
- **System prompt**: tuned for brevity — responses target 1-3 sentences to fit the 128×64 display

---

## Setup

### 1. System dependencies (Pi only)

```bash
# I2C + I2S
sudo raspi-config  # → Interface Options → I2C → Enable

# Add to /boot/firmware/config.txt:
#   dtparam=i2s=on
#   dtoverlay=i2s-mmap

# Camera + audio
sudo apt install -y python3-picamera2 python3-libcamera portaudio19-dev

# Or run the setup script:
bash scripts/setup.sh
```

### 2. Hailo (Pi only)

```bash
# Download hailort_<version>_arm64.deb from https://hailo.ai/developer-zone/
sudo dpkg -i hailort_<version>_arm64.deb
pip install hailort
```

### 3. Python environment

```bash
# Use --system-site-packages on Pi so picamera2 is visible
python -m venv .venv --system-site-packages
source .venv/bin/activate
pip install -e ".[dev]"
```

### 4. Environment variables

Copy `.env.example` to `.env` and fill in:

```env
ANTHROPIC_API_KEY=sk-ant-...
DEEPGRAM_API_KEY=...
AUDIO_DEVICE=          # leave blank for default, or set to device index
AUDIO_SAMPLE_RATE=16000
AUDIO_CHANNELS=1
OLED_I2C_PORT=1
```

---

## Running

```bash
# Main vibe check demo
python3 scripts/test_vibe.py

# Full hardware loop (camera → Claude → OLED)
python -m src.main
# or after editable install:
hardmode
```

### Peripheral test scripts

| Script | Tests |
|---|---|
| `scripts/test_display.py` | Ask Claude a question, display response on OLED |
| `scripts/test_mic.py` | Record 10s audio, VAD check, save to WAV |
| `scripts/test_servo.py` | 8 servo movement patterns with OLED feedback |
| `scripts/test_dc_motor.py` | Stepper motor clock-tick mode (6 patterns) |

---

## Model files

Drop compiled model files into `models/` (git-ignored):

| Extension | Format | Use |
|---|---|---|
| `.hef` | Hailo Executable Format | Hailo AI HAT+ local inference (e.g. `yolov8s.hef`) |
| `.rpk` | IMX500 network package | On-sensor camera inference (e.g. `imx500_mobilenet_ssd.rpk`) |

Default paths are overridable via `HAILO_HEF_PATH` and `IMX500_MODEL_PATH` in `.env`.

---

## Tests

```bash
pytest                                              # all tests (no hardware required)
pytest tests/test_llm.py::test_pil_to_base64_returns_string  # single test
```

Pi-specific imports (`picamera2`, `hailo_platform`, `luma.oled`, `gpiozero`) are guarded so the test suite runs on any machine.

---

## Project structure

```
config/
  settings.py          — all constants loaded from .env
models/                — .hef and .rpk model files (git-ignored)
scripts/
  test_vibe.py         — main vibe check demo (doom clock)
  test_display.py      — OLED + Claude test
  test_mic.py          — microphone test
  test_servo.py        — servo test
  test_dc_motor.py     — stepper motor test
  setup.sh             — Pi first-boot setup
src/
  audio/
    microphone.py      — I2S mic recording and streaming
    classifier.py      — spectral audio classifier (music/speech/brainrot)
  camera/
    ai_camera.py       — picamera2 + IMX500 on-sensor inference
  display/
    oled.py            — SH1106 OLED driver
  inference/
    hailo.py           — Hailo-8L inference engine
  llm/
    client.py          — Claude streaming client
  motor/
    stepper.py         — 28BYJ-48 doom clock stepper
  servo/
    motor.py           — servo vibe feedback controller
  main.py              — one-shot hardware demo loop
tests/
  test_llm.py
  test_audio.py
  test_display.py
```
