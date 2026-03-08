"""
Audio classifier using YAMNet (TFLite) — detects music, speech, silence, etc.

YAMNet classifies audio into 521 AudioSet classes. We group them into
categories relevant to brain rot detection:
  - "music"   → background music, singing, trending audio
  - "speech"  → conversation, narration
  - "silence" → no significant audio
  - "other"   → everything else

Model setup (run once on the Pi):
    python -m src.audio.classifier --download
"""
from __future__ import annotations

import csv
import os
import urllib.request
from pathlib import Path

import numpy as np

from config.settings import MODELS_DIR

YAMNET_TFLITE_URL = "https://storage.googleapis.com/tfhub-lite-models/google/lite-model/yamnet/tflite/1.tflite"
YAMNET_CLASSES_URL = "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv"

YAMNET_MODEL_PATH = MODELS_DIR / "yamnet.tflite"
YAMNET_CLASSES_PATH = MODELS_DIR / "yamnet_class_map.csv"

# YAMNet expects 16kHz mono audio
YAMNET_SAMPLE_RATE = 16000

# Class indices that indicate "music" / brain rot media consumption
# These are looked up from yamnet_class_map.csv at load time
MUSIC_KEYWORDS = {
    "music", "singing", "song", "rapping", "hip hop", "pop music",
    "electronic music", "dance music", "techno", "beat", "drum machine",
    "bass", "soundtrack", "theme music", "jingle", "ringtone",
    "musical instrument", "guitar", "piano", "keyboard", "synthesizer",
    "drum", "percussion", "reggaeton", "funk", "disco", "rock music",
    "heavy metal", "punk rock", "grunge", "progressive rock",
    "rock and roll", "jazz", "blues", "soul music", "rhythm and blues",
    "country", "swing music", "bluegrass", "flamenco", "folk music",
}

SPEECH_KEYWORDS = {
    "speech", "conversation", "narration", "child speech",
    "female speech", "male speech",
}


class AudioClassifier:
    """Classify audio chunks as music, speech, silence, or other using YAMNet."""

    def __init__(self) -> None:
        self._interpreter = None
        self._class_names: list[str] = []
        self._music_indices: set[int] = set()
        self._speech_indices: set[int] = set()

    def start(self) -> None:
        """Load model and class map."""
        _ensure_downloaded()

        import tflite_runtime.interpreter as tflite

        self._interpreter = tflite.Interpreter(str(YAMNET_MODEL_PATH))
        self._interpreter.allocate_tensors()

        # Load class names
        self._class_names = []
        with open(YAMNET_CLASSES_PATH) as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for row in reader:
                self._class_names.append(row[2].strip().lower())

        # Build index sets
        for i, name in enumerate(self._class_names):
            if any(kw in name for kw in MUSIC_KEYWORDS):
                self._music_indices.add(i)
            if any(kw in name for kw in SPEECH_KEYWORDS):
                self._speech_indices.add(i)

    def classify(self, audio: np.ndarray, sample_rate: int = YAMNET_SAMPLE_RATE) -> dict:
        """
        Classify an audio chunk.

        Args:
            audio: float32 mono numpy array
            sample_rate: sample rate of the audio (will resample to 16kHz if needed)

        Returns:
            dict with keys:
                "category": "music" | "speech" | "silence" | "other"
                "music_score": float (0-1)
                "speech_score": float (0-1)
                "top_class": str (top predicted class name)
                "is_brainrot": bool
        """
        if self._interpreter is None:
            raise RuntimeError("Call start() first")

        # Resample to 16kHz if needed
        if sample_rate != YAMNET_SAMPLE_RATE:
            import resampy
            audio = resampy.resample(audio, sample_rate, YAMNET_SAMPLE_RATE)

        # Ensure float32 and normalize
        audio = audio.astype(np.float32)
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))

        # Run inference
        input_details = self._interpreter.get_input_details()
        output_details = self._interpreter.get_output_details()

        self._interpreter.resize_tensor_input(input_details[0]["index"], audio.shape)
        self._interpreter.allocate_tensors()
        self._interpreter.set_tensor(input_details[0]["index"], audio)
        self._interpreter.invoke()

        # Scores shape: (n_frames, 521)
        scores = self._interpreter.get_tensor(output_details[0]["index"])
        avg_scores = np.mean(scores, axis=0)

        # Aggregate music and speech scores
        music_score = float(np.max([avg_scores[i] for i in self._music_indices]) if self._music_indices else 0)
        speech_score = float(np.max([avg_scores[i] for i in self._speech_indices]) if self._speech_indices else 0)

        top_idx = int(np.argmax(avg_scores))
        top_class = self._class_names[top_idx] if top_idx < len(self._class_names) else "unknown"
        top_score = float(avg_scores[top_idx])

        # Determine category
        if top_score < 0.1:
            category = "silence"
        elif top_idx in self._music_indices:
            category = "music"
        elif top_idx in self._speech_indices:
            category = "speech"
        else:
            category = "other"

        # Brain rot = music playing without meaningful speech
        is_brainrot = music_score > 0.3 and speech_score < 0.2

        return {
            "category": category,
            "music_score": music_score,
            "speech_score": speech_score,
            "top_class": top_class,
            "is_brainrot": is_brainrot,
        }

    def stop(self) -> None:
        self._interpreter = None

    def __enter__(self) -> "AudioClassifier":
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()


def _ensure_downloaded() -> None:
    """Download YAMNet model and class map if not present."""
    os.makedirs(MODELS_DIR, exist_ok=True)

    if not YAMNET_MODEL_PATH.exists():
        print(f"[yamnet] Downloading model to {YAMNET_MODEL_PATH}...")
        urllib.request.urlretrieve(YAMNET_TFLITE_URL, YAMNET_MODEL_PATH)
        print("[yamnet] Model downloaded.")

    if not YAMNET_CLASSES_PATH.exists():
        print(f"[yamnet] Downloading class map to {YAMNET_CLASSES_PATH}...")
        urllib.request.urlretrieve(YAMNET_CLASSES_URL, YAMNET_CLASSES_PATH)
        print("[yamnet] Class map downloaded.")


if __name__ == "__main__":
    import sys
    if "--download" in sys.argv:
        _ensure_downloaded()
        print("YAMNet model files ready.")
    else:
        print("Usage: python -m src.audio.classifier --download")
