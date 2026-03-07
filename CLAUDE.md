# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install (use --system-site-packages on Pi so picamera2 is visible)
python -m venv .venv --system-site-packages && source .venv/bin/activate
pip install -e ".[dev]"

# Run the application
python -m src.main
# or after editable install:
hardmode

# Tests (no hardware required — Pi-specific imports are guarded)
pytest
pytest tests/test_llm.py::test_pil_to_base64_returns_string   # single test

# Lint / format
ruff check src/ tests/
black src/ tests/
```

## Architecture

All hardware constants (I2C address, GPIO assignments, model file paths, audio rate) live in **`config/settings.py`** and are loaded from `.env` via `python-dotenv`. Nothing else reads the environment directly.

Each hardware peripheral is a self-contained manager class in its own module under `src/`:

| Module | Class | Notes |
|---|---|---|
| `src/llm/client.py` | `ClaudeClient` | Wraps `anthropic` SDK; streaming-first; stores conversation history as text-only (thinking blocks are stripped before appending to history) |
| `src/display/oled.py` | `OLEDDisplay` | `luma.oled` SH1106 driver; lazy `luma` import inside `__init__` so the module can be imported on non-Pi machines |
| `src/camera/ai_camera.py` | `AICameraManager` | `picamera2`; supports plain capture or IMX500 on-sensor inference via `.rpk` model packages |
| `src/audio/microphone.py` | `MicrophoneManager` | `sounddevice`/ALSA; blocking `record()`, streaming `iter_chunks()`, simple energy VAD |
| `src/inference/hailo.py` | `HailoInferenceEngine` | `hailo_platform`; `HAILO_AVAILABLE` bool guards all imports so the module is safe to import everywhere |

All manager classes are context managers (`__enter__`/`__exit__`).

## Pi-specific dependencies

Two packages **cannot** be installed via pip in the normal way:

- **`picamera2`** — `sudo apt install python3-picamera2 python3-libcamera` (system package only)
- **`hailort`** — download the `.deb` from [hailo.ai/developer-zone](https://hailo.ai/developer-zone/), then `pip install hailort`

`scripts/setup.sh` automates the rest of the first-boot setup (I2C enable, I2S enable, `portaudio19-dev`, etc.).

## Model files

Drop compiled model files into `models/` (git-ignored):

- **`.hef`** — Hailo Executable Format for the AI HAT+ (e.g. `yolov8s.hef`)
- **`.rpk`** — IMX500 on-sensor network package (e.g. `imx500_mobilenet_ssd.rpk`)

Default paths are overridable via `HAILO_HEF_PATH` and `IMX500_MODEL_PATH` in `.env`.

## Claude API usage

- Model: `claude-opus-4-6` with `thinking: {"type": "adaptive"}` on every call.
- All requests stream by default (`.stream()` + `text_stream`).
- Vision is passed as base64 JPEG in the `image` source block; RGBA images are converted to RGB before encoding.
- `ClaudeClient.chat()` maintains history; `ClaudeClient.ask()` is stateless.
