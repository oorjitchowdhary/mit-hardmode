"""Unit tests for audio helpers that don't require real hardware."""
import numpy as np

from src.audio.microphone import MicrophoneManager


def test_is_speech_silent():
    silence = np.zeros(1600, dtype=np.float32)
    assert MicrophoneManager.is_speech(silence) is False


def test_is_speech_loud():
    loud = np.ones(1600, dtype=np.float32)
    assert MicrophoneManager.is_speech(loud) is True


def test_is_speech_threshold():
    audio = np.full(1600, 0.02, dtype=np.float32)
    assert MicrophoneManager.is_speech(audio, threshold=0.01) is True
    assert MicrophoneManager.is_speech(audio, threshold=0.05) is False
