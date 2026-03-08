"""
Real-time conversation vibe check.

Streams audio from the USB mic to Deepgram for transcription with speaker
diarization, then periodically asks Claude to assess the conversation vibe.
Results are shown on the OLED display and a stepper motor reacts physically.

Usage:
    python3 scripts/test_vibe.py
    # Press Ctrl+C to stop
"""
import json
import signal
import sys
import textwrap
import threading
import time

import numpy as np
import sounddevice as sd
import websockets.sync.client as ws_client
from PIL import Image, ImageDraw, ImageFont

from config.settings import (
    AUDIO_CHANNELS,
    AUDIO_DEVICE,
    AUDIO_SAMPLE_RATE,
    DEEPGRAM_API_KEY,
    OLED_HEIGHT,
    OLED_WIDTH,
)
from src.display.oled import OLEDDisplay
from src.llm.client import ClaudeClient
from src.motor.stepper import StepperClock

# How often (in seconds) to send transcript to Claude for vibe analysis
VIBE_CHECK_INTERVAL = 15.0

# Deepgram streaming endpoint with diarization enabled
DG_URL = (
    "wss://api.deepgram.com/v1/listen"
    "?model=nova-2"
    "&encoding=linear16"
    f"&sample_rate={AUDIO_SAMPLE_RATE}"
    f"&channels={AUDIO_CHANNELS}"
    "&punctuate=true"
    "&diarize=true"
    "&interim_results=false"
)


def main() -> None:
    if not DEEPGRAM_API_KEY:
        print("ERROR: Set DEEPGRAM_API_KEY in .env")
        sys.exit(1)

    display = OLEDDisplay()
    claude = ClaudeClient()
    motor = StepperClock()
    transcript_lines: list[str] = []
    state = {"vibe": "...", "score": -1}  # score: 1=good, 0=bad, -1=unknown
    lock = threading.Lock()
    running = threading.Event()
    running.set()
    font = ImageFont.load_default()

    def update_display():
        """Show vibe score and description on the full display."""
        with lock:
            vibe = state["vibe"]
            score = state["score"]

        img = Image.new("1", (OLED_WIDTH, OLED_HEIGHT), 0)
        draw = ImageDraw.Draw(img)

        # Title bar
        draw.rectangle((0, 0, OLED_WIDTH - 1, 13), fill=1)
        draw.text((1, 3), "VIBE", font=font, fill=0)

        # Score indicator
        if score == 1:
            label = "GOOD"
        elif score == 0:
            label = "BAD"
        else:
            label = "..."
        # Right-align the score label in the title bar
        bbox = font.getbbox(label)
        label_w = bbox[2] - bbox[0]
        label_x = OLED_WIDTH - label_w - 4
        draw.text((label_x, 3), label, font=font, fill=0)

        # Big circle indicator: filled = good, empty = bad
        cx, cy, r = 20, 38, 16
        if score == 1:
            draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=1)
        elif score == 0:
            draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=1)
        else:
            draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=1)
            draw.text((cx - 3, cy - 4), "?", font=font, fill=1)

        # Vibe text on the right side
        vibe_lines = textwrap.wrap(vibe, width=12) or [""]
        for i, line in enumerate(vibe_lines[:4]):
            draw.text((44, 18 + i * 12), line, font=font, fill=1)

        display._device.display(img)

    def shutdown(sig, frame):
        print("\nShutting down...")
        running.clear()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    display.show_status("Vibe Check", "Connecting...")
    print("[vibe] Connecting to Deepgram...")

    ws = ws_client.connect(
        DG_URL,
        additional_headers={"Authorization": f"Token {DEEPGRAM_API_KEY}"},
    )
    print("[vibe] Connected. Listening...")
    print("[motor] Clock ticking at normal speed...")
    display.show_status("Vibe Check", "Listening...")

    # --- Thread 1: stream mic audio to Deepgram ---
    def audio_sender():
        def callback(indata: np.ndarray, frames, time_info, status):
            if not running.is_set():
                return
            pcm = (indata * 32767).astype(np.int16).tobytes()
            try:
                ws.send(pcm)
            except Exception:
                pass

        with sd.InputStream(
            samplerate=AUDIO_SAMPLE_RATE,
            channels=AUDIO_CHANNELS,
            dtype="float32",
            device=AUDIO_DEVICE,
            blocksize=4096,
            callback=callback,
        ):
            while running.is_set():
                time.sleep(0.1)

    # --- Thread 2: receive transcripts from Deepgram ---
    def transcript_receiver():
        while running.is_set():
            try:
                raw = ws.recv(timeout=1.0)
            except TimeoutError:
                continue
            except Exception:
                break

            msg = json.loads(raw)
            channel = msg.get("channel", {})
            alt = (channel.get("alternatives") or [{}])[0]
            text = alt.get("transcript", "").strip()
            if not text:
                continue

            words = alt.get("words", [])
            speaker = None
            if words:
                speaker = words[0].get("speaker")

            line = f"[Speaker {speaker}]: {text}" if speaker is not None else text
            print(f"  {line}")

            with lock:
                transcript_lines.append(line)

    # --- Thread 3: periodic vibe analysis via Claude ---
    def vibe_analyzer():
        while running.is_set():
            # Phase 1: Neutral — collect transcript with countdown
            print("\n[vibe] Neutral — listening for 15s...")
            with lock:
                transcript_lines.clear()
            collect_end = time.time() + VIBE_CHECK_INTERVAL
            while time.time() < collect_end and running.is_set():
                remaining = int(collect_end - time.time())
                with lock:
                    state["vibe"] = f"Listening ({remaining}s)"
                    state["score"] = -1
                update_display()
                time.sleep(1.0)

            # Phase 2: Analyze
            with lock:
                if not transcript_lines:
                    continue
                recent = list(transcript_lines)
                transcript_lines.clear()

            full_transcript = "\n".join(recent)
            print("[vibe] Analyzing...")
            with lock:
                state["vibe"] = "Analyzing..."
                state["score"] = -1
            update_display()

            response = claude.ask(
                "Analyze this conversation snippet. Reply in EXACTLY this format:\n"
                "SCORE: 0 or 1 (0=bad/negative/tense/awkward, 1=good/positive/friendly/productive)\n"
                "VIBE: 2-4 words describing the vibe\n\n"
                "Nothing else. Example:\n"
                "SCORE: 1\n"
                "VIBE: Excited and hopeful\n\n"
                f"{full_transcript}"
            )

            # Parse response
            score = -1
            vibe = response.strip()
            for resp_line in response.strip().split("\n"):
                resp_line = resp_line.strip()
                if resp_line.startswith("SCORE:"):
                    try:
                        score = int(resp_line.split(":")[1].strip()[0])
                    except (ValueError, IndexError):
                        pass
                elif resp_line.startswith("VIBE:"):
                    vibe = resp_line.split(":", 1)[1].strip()

            # Phase 3: Show result immediately
            print(f"[vibe] score={score} vibe={vibe}")
            with lock:
                state["vibe"] = vibe
                state["score"] = score
            update_display()

            # Drive stepper motor based on score
            if score == 1:
                print("[motor] Good vibe — slowing down and stopping...")
                motor.good_vibe()
            elif score == 0:
                print("[motor] Bad vibe — speeding up...")
                motor.bad_vibe()

            # Hold the result on screen for 10s while motor reacts
            time.sleep(10.0)

            # Phase 4: Neutral — motor back to normal ticking, 15s countdown
            print("[vibe] Neutral — 15s cooldown...")
            motor.set_normal()
            with lock:
                transcript_lines.clear()
            neutral_end = time.time() + 15.0
            while time.time() < neutral_end and running.is_set():
                remaining = int(neutral_end - time.time())
                with lock:
                    state["vibe"] = f"Neutral ({remaining}s)"
                    state["score"] = -1
                update_display()
                time.sleep(1.0)

    t1 = threading.Thread(target=audio_sender, daemon=True)
    t2 = threading.Thread(target=transcript_receiver, daemon=True)
    t3 = threading.Thread(target=vibe_analyzer, daemon=True)
    t1.start()
    t2.start()
    t3.start()

    # Wait until Ctrl+C
    while running.is_set():
        time.sleep(0.5)

    # Cleanup
    try:
        ws.send(json.dumps({"type": "CloseStream"}))
        ws.close()
    except Exception:
        pass
    motor.stop()
    display.clear()
    print("[vibe] Done.")


if __name__ == "__main__":
    main()
