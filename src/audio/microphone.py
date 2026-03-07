"""
I2S MEMS microphone manager (28AWG, 4-pin).

Wiring to Raspberry Pi 5
────────────────────────
  Mic  │ Signal  │ Pi GPIO   │ Physical pin
  ─────┼─────────┼───────────┼─────────────
  1    │ VDD     │ 3.3V      │ pin 1 or 17
  2    │ GND     │ GND       │ pin 6, 9, 14, 20 …
  3    │ BCLK    │ GPIO 18   │ pin 12   (I2S bit clock)
  4    │ DOUT    │ GPIO 20   │ pin 38   (I2S data out / SD)

  If your module has a 5th wire (WS / LRCLK) connect it to GPIO 19 (pin 35).
  Many 4-pin MEMS mics tie WS to GND internally (left channel, mono).

Enable I2S in /boot/firmware/config.txt
    dtparam=i2s=on
    dtoverlay=i2s-mmap        ← exposes the I2S device to ALSA

Verify the device appears:
    arecord -l
    # should list something like "card 0: sndrpii2scard"

Run a test recording:
    arecord -D plughw:0,0 -c1 -r16000 -f S16_LE -d 3 test.wav && aplay test.wav
"""
from __future__ import annotations

import queue
import threading
from typing import Callable

import numpy as np
import sounddevice as sd

from config.settings import (
    AUDIO_CHANNELS,
    AUDIO_CHUNK_SIZE,
    AUDIO_DEVICE,
    AUDIO_RECORD_SECONDS,
    AUDIO_SAMPLE_RATE,
)


class MicrophoneManager:
    """
    Record audio from the I2S MEMS microphone via sounddevice (ALSA).

    Usage — blocking capture:
        mic = MicrophoneManager()
        audio = mic.record(seconds=3)   # float32 NumPy array

    Usage — real-time streaming:
        with mic.stream(callback=my_fn):
            time.sleep(5)

    Usage — blocking stream that queues chunks:
        for chunk in mic.iter_chunks(duration=5.0):
            process(chunk)
    """

    def __init__(
        self,
        device: int | str | None = AUDIO_DEVICE,
        sample_rate: int = AUDIO_SAMPLE_RATE,
        channels: int = AUDIO_CHANNELS,
        chunk_size: int = AUDIO_CHUNK_SIZE,
    ) -> None:
        self.device = device
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size

    # ── blocking capture ──────────────────────────────────────────────────────

    def record(self, seconds: float = AUDIO_RECORD_SECONDS) -> np.ndarray:
        """
        Record *seconds* of audio.
        Returns a float32 mono array of shape (N,).
        """
        n_samples = int(seconds * self.sample_rate)
        audio = sd.rec(
            n_samples,
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
            device=self.device,
        )
        sd.wait()
        return audio.flatten()

    def record_int16(self, seconds: float = AUDIO_RECORD_SECONDS) -> np.ndarray:
        """Record as int16 PCM — suitable for direct WAV encoding."""
        n_samples = int(seconds * self.sample_rate)
        audio = sd.rec(
            n_samples,
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="int16",
            device=self.device,
        )
        sd.wait()
        return audio.flatten()

    def save_wav(self, path: str, seconds: float = AUDIO_RECORD_SECONDS) -> None:
        """Record and save a WAV file to *path*."""
        from scipy.io.wavfile import write as wav_write

        audio = self.record_int16(seconds)
        wav_write(path, self.sample_rate, audio)

    # ── real-time streaming ───────────────────────────────────────────────────

    def stream(
        self,
        callback: Callable[[np.ndarray, int, object, object], None],
        blocksize: int | None = None,
    ) -> sd.InputStream:
        """
        Return a sounddevice.InputStream context manager.
        *callback* signature: (indata: ndarray, frames: int, time, status) → None

        Example:
            def cb(indata, frames, time, status):
                rms = np.sqrt(np.mean(indata ** 2))
                print(f"RMS: {rms:.4f}")

            with mic.stream(cb):
                time.sleep(5)
        """
        return sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
            device=self.device,
            blocksize=blocksize or self.chunk_size,
            callback=callback,
        )

    def iter_chunks(
        self, duration: float, chunk_seconds: float = 0.1
    ) -> "typing.Generator[np.ndarray, None, None]":
        """
        Generator that yields overlapping float32 chunks for *duration* seconds.
        Each chunk has shape (chunk_samples,).
        """
        import typing

        q: "queue.Queue[np.ndarray]" = queue.Queue()
        stop_event = threading.Event()

        def _cb(indata: np.ndarray, frames: int, time: object, status: object) -> None:
            if not stop_event.is_set():
                q.put(indata.copy().flatten())

        blocksize = int(chunk_seconds * self.sample_rate)
        elapsed = 0.0

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
            device=self.device,
            blocksize=blocksize,
            callback=_cb,
        ):
            while elapsed < duration:
                chunk = q.get(timeout=2.0)
                elapsed += chunk_seconds
                yield chunk

        stop_event.set()

    # ── utility ───────────────────────────────────────────────────────────────

    @staticmethod
    def is_speech(audio: np.ndarray, threshold: float = 0.01) -> bool:
        """
        Trivial energy-based VAD.
        Returns True when RMS amplitude exceeds *threshold*.
        """
        rms = float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))
        return rms > threshold

    @staticmethod
    def list_devices() -> None:
        """Print all available audio devices (helpful for finding the I2S mic)."""
        print(sd.query_devices())
