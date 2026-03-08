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
from src.audio.classifier import AudioClassifier
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
    audio_clf = AudioClassifier(sample_rate=AUDIO_SAMPLE_RATE)
    audio_clf.start()
    print("[audio] Classifier ready (numpy spectral analysis).")
    transcript_lines: list[str] = []
    audio_buffer: list[np.ndarray] = []  # raw audio chunks for YAMNet
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
            # Buffer raw audio for YAMNet classification
            with lock:
                audio_buffer.append(indata.copy().flatten())
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
            # ── Phase 1: LISTENING (15s) ──
            print("\n[vibe] Phase: LISTENING")
            with lock:
                transcript_lines.clear()
            listen_end = time.time() + VIBE_CHECK_INTERVAL
            while time.time() < listen_end and running.is_set():
                remaining = int(listen_end - time.time())
                with lock:
                    state["vibe"] = f"Listening ({remaining}s)"
                    state["score"] = -1
                update_display()
                time.sleep(1.0)

            # ── Phase 2: ANALYZING ──
            print("[vibe] Phase: ANALYZING")
            with lock:
                state["vibe"] = "Analyzing..."
                state["score"] = -1
            update_display()

            # Run YAMNet audio classification
            with lock:
                raw_audio = np.concatenate(audio_buffer) if audio_buffer else np.array([], dtype=np.float32)
                audio_buffer.clear()

            audio_result = None
            if len(raw_audio) > AUDIO_SAMPLE_RATE:  # need at least 1s of audio
                audio_result = audio_clf.classify(raw_audio, sample_rate=AUDIO_SAMPLE_RATE)
                print(f"[audio] category={audio_result['category']} "
                      f"music={audio_result['music_score']:.2f} "
                      f"speech={audio_result['speech_score']:.2f} "
                      f"top={audio_result['top_class']} "
                      f"brainrot={audio_result['is_brainrot']}")

            # If YAMNet detects brain rot (music, no speech), skip Claude
            if audio_result and audio_result["is_brainrot"]:
                score = 0
                vibe = f"Brain rot ({audio_result['top_class']})"
                print(f"[vibe] YAMNet detected brain rot — skipping Claude")
            else:
                # Use transcript + Claude for conversation analysis
                with lock:
                    if not transcript_lines:
                        # No speech and no brain rot — silence, restart
                        print("[vibe] No speech or music detected, restarting...")
                        continue
                    recent = list(transcript_lines)
                    transcript_lines.clear()

                full_transcript = "\n".join(recent)

                # Include YAMNet context in the prompt
                audio_context = ""
                if audio_result:
                    audio_context = (
                        f"\nAudio analysis: category={audio_result['category']}, "
                        f"music_score={audio_result['music_score']:.2f}, "
                        f"top_sound={audio_result['top_class']}\n"
                    )

                response = claude.ask(
                    "Analyze this audio transcript and audio analysis. Reply in EXACTLY this format:\n"
                    "SCORE: 0 or 1\n"
                    "VIBE: 2-4 words\n\n"
                    "SCORE 0 (BAD) if ANY of these are detected:\n"
                    "- Brain rot content (TikTok, reels, shorts, memes, viral trends)\n"
                    "- Doomscrolling or mindless media consumption\n"
                    "- Background music with no productive conversation\n"
                    "- Negative, tense, awkward, or unproductive conversation\n"
                    "- Slang heavy brain rot language (skibidi, rizz, sigma, etc.)\n\n"
                    "SCORE 1 (GOOD) if:\n"
                    "- Genuine productive conversation, studying, learning\n"
                    "- Positive, friendly, focused, or meaningful interaction\n"
                    "- Working, coding, problem-solving discussion\n\n"
                    "Nothing else. Example:\n"
                    "SCORE: 0\n"
                    "VIBE: Brain rot scrolling\n\n"
                    f"{audio_context}"
                    f"Transcript:\n{full_transcript}"
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

            # ── Phase 3: RESULT (15s) — show score, motor reacts ──
            print(f"[vibe] Phase: RESULT — score={score} vibe={vibe}")
            with lock:
                state["vibe"] = vibe
                state["score"] = score
            update_display()

            if score == 1:
                print("[motor] Good vibe — stopping for 15s...")
                motor.good_vibe()
            elif score == 0:
                print("[motor] Bad vibe — fast for 15s...")
                motor.bad_vibe()

            # Hold result on screen for 15s
            result_end = time.time() + 15.0
            while time.time() < result_end and running.is_set():
                time.sleep(0.5)

            # ── Phase 4: NEUTRAL (15s) — motor normal, countdown ──
            print("[vibe] Phase: NEUTRAL")
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
    audio_clf.stop()
    display.clear()
    print("[vibe] Done.")


if __name__ == "__main__":
    main()
