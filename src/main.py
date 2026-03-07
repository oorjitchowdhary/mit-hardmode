"""
hardmode — main entry point.

Demonstrates the full hardware loop:
  1. Start OLED, camera, microphone, and optionally Hailo inference.
  2. Show startup status on the display.
  3. Capture a camera frame.
  4. Ask Claude what it sees (vision + text, streaming to OLED).
  5. Record audio and run a simple VAD check.
  6. Optionally run Hailo object detection on the captured frame.

Run:
    python -m src.main
    # or after `pip install -e .`:
    hardmode
"""
from __future__ import annotations

import signal
import sys
import time

from config.settings import HAILO_HEF_PATH, IMX500_MODEL_PATH
from src.audio.microphone import MicrophoneManager
from src.camera.ai_camera import AICameraManager
from src.display.oled import OLEDDisplay
from src.inference.hailo import HAILO_AVAILABLE, HailoInferenceEngine
from src.llm.client import ClaudeClient


def main() -> None:
    print("[hardmode] starting up …")

    # ── display ───────────────────────────────────────────────────────────────
    display = OLEDDisplay()
    display.show_status("hardmode", "booting…")

    # ── LLM client ────────────────────────────────────────────────────────────
    claude = ClaudeClient()

    # ── camera ────────────────────────────────────────────────────────────────
    display.show_status("Camera", "initialising…")
    camera = AICameraManager()
    camera.start()

    # ── microphone ────────────────────────────────────────────────────────────
    mic = MicrophoneManager()

    # ── Hailo (optional) ──────────────────────────────────────────────────────
    hailo: HailoInferenceEngine | None = None
    if HAILO_AVAILABLE:
        try:
            hailo = HailoInferenceEngine(HAILO_HEF_PATH)
            hailo.start()
            display.show_status("Hailo", "model loaded")
            time.sleep(0.8)
        except Exception as exc:
            print(f"[hailo] skipping — {exc}")
            hailo = None
    else:
        print("[hailo] HailoRT not installed — skipping local inference")

    # ── graceful shutdown ─────────────────────────────────────────────────────
    def _shutdown(sig: int, frame: object) -> None:
        print("\n[hardmode] shutting down …")
        camera.stop()
        if hailo:
            hailo.stop()
        display.clear()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # ── main loop ─────────────────────────────────────────────────────────────
    display.show_status("Ready", "press Ctrl+C to quit")
    print("[hardmode] ready.  Press Ctrl+C to quit.\n")

    try:
        _demo_loop(display, camera, mic, claude, hailo)
    finally:
        camera.stop()
        if hailo:
            hailo.stop()
        display.clear()


def _demo_loop(
    display: OLEDDisplay,
    camera: AICameraManager,
    mic: MicrophoneManager,
    claude: ClaudeClient,
    hailo: HailoInferenceEngine | None,
) -> None:
    """Single demo iteration — capture → infer → describe → display."""

    # 1. Capture frame
    display.show_status("Camera", "capturing…")
    frame = camera.capture_frame()
    print(f"[camera] frame captured: {frame.size}")

    # 2. Local Hailo detection (fast, offline)
    if hailo is not None:
        display.show_status("Hailo", "inferring…")
        try:
            processed = hailo.preprocess(frame)
            outputs = hailo.infer(processed)
            detections = hailo.parse_detections(outputs)
            n = len(detections)
            print(f"[hailo] {n} detection(s): {detections[:3]}")
            display.show_status("Hailo", f"{n} object(s) found")
            time.sleep(1.0)
        except Exception as exc:
            print(f"[hailo] inference error: {exc}")

    # 3. Send frame to Claude for a natural-language description
    display.show_status("Claude", "thinking…")
    print("[claude] asking about the image …")

    # Stream the response and update the display one chunk at a time
    response_tokens: list[str] = []

    def _on_token(token: str) -> None:
        response_tokens.append(token)
        display.show_text("".join(response_tokens))

    response = claude.chat(
        "Briefly describe what you see in this image in one or two sentences.",
        image=frame,
        on_token=_on_token,
    )
    print(f"\n[claude] {response}\n")

    # Show the final response on the OLED for a few seconds
    display.show_text(response)
    time.sleep(4.0)

    # 4. Record a short audio clip and check for speech
    display.show_status("Mic", "listening 3s…")
    print("[mic] recording 3 seconds …")
    audio = mic.record(seconds=3.0)
    speech_detected = mic.is_speech(audio)
    print(f"[mic] speech detected: {speech_detected}")
    display.show_status("Mic", "speech: " + ("yes" if speech_detected else "no"))
    time.sleep(1.5)

    display.show_status("Done", "loop complete")
    print("[hardmode] demo loop complete.\n")
    time.sleep(2.0)


if __name__ == "__main__":
    main()
