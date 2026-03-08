"""
Real-time conversation vibe check.

Streams audio from the USB mic to Deepgram for transcription with speaker
diarization, then periodically asks Claude to assess the conversation vibe.
Results are shown on the OLED display.

Usage:
    python3 scripts/test_vibe.py
    # Press Ctrl+C to stop
"""
import json
import signal
import sys
import threading
import time

import numpy as np
import sounddevice as sd
import websockets.sync.client as ws_client

from config.settings import (
    AUDIO_CHANNELS,
    AUDIO_DEVICE,
    AUDIO_SAMPLE_RATE,
    DEEPGRAM_API_KEY,
)
from PIL import Image, ImageDraw, ImageFont

from config.settings import OLED_HEIGHT, OLED_WIDTH
from src.display.oled import OLEDDisplay
from src.llm.client import ClaudeClient

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
    transcript_lines: list[str] = []
    state = {"vibe": "...", "transcript": ""}
    lock = threading.Lock()
    running = threading.Event()
    running.set()
    font = ImageFont.load_default()

    def update_display():
        """Draw split screen: transcript on top, vibe on bottom."""
        with lock:
            txt = state["transcript"]
            vibe = state["vibe"]

        img = Image.new("1", (OLED_WIDTH, OLED_HEIGHT), 0)
        draw = ImageDraw.Draw(img)

        # Top half: latest transcript (lines 0-2, ~30px)
        import textwrap
        lines = textwrap.wrap(txt, width=21) or [""]
        for i, line in enumerate(lines[:3]):
            draw.text((0, i * 10), line, font=font, fill=1)

        # Divider line
        draw.line((0, 31, OLED_WIDTH - 1, 31), fill=1)

        # Bottom half: vibe (inverted title bar + text)
        draw.rectangle((0, 33, OLED_WIDTH - 1, 43), fill=1)
        draw.text((1, 34), "VIBE", font=font, fill=0)
        vibe_lines = textwrap.wrap(vibe, width=21) or [""]
        for i, line in enumerate(vibe_lines[:2]):
            draw.text((0, 45 + i * 10), line, font=font, fill=1)

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

            # Extract speaker labels from words
            words = alt.get("words", [])
            speaker = None
            if words:
                speaker = words[0].get("speaker")

            line = f"[Speaker {speaker}]: {text}" if speaker is not None else text
            print(f"  {line}")

            with lock:
                transcript_lines.append(line)
                state["transcript"] = line
            update_display()

    # --- Thread 3: periodic vibe analysis via Claude ---
    def vibe_analyzer():
        last_check = time.time()
        while running.is_set():
            time.sleep(1.0)
            if time.time() - last_check < VIBE_CHECK_INTERVAL:
                continue
            last_check = time.time()

            with lock:
                if not transcript_lines:
                    continue
                full_transcript = "\n".join(transcript_lines)

            print("\n[vibe] Analyzing conversation vibe...")
            display.show_status("Vibe Check", "Analyzing...")

            vibe = claude.ask(
                "In exactly two words separated by a period, summarize the vibe of this conversation transcript. "
                "First word: one adjective describing the overall tone. "
                "Second part: two to three words describing the outcome or dynamic. "
                "Format strictly as: 'Adjective. Short phrase.' with nothing else.\n\n"
                "Transcript:\n"
                f"{full_transcript}"
            )
            print(f"[vibe] {vibe}\n")
            with lock:
                state["vibe"] = vibe
            update_display()

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
    display.clear()
    print("[vibe] Done.")


if __name__ == "__main__":
    main()
