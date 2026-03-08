"""
Lightweight audio classifier — detects music, speech, and silence using
spectral analysis. No ML frameworks required, just numpy + scipy.

Music detection: consistent energy with strong spectral peaks (bass, rhythm)
Speech detection: variable energy, formant frequencies (300-3000 Hz)
Silence: low overall energy
"""
from __future__ import annotations

import numpy as np


class AudioClassifier:
    """Classify audio as music, speech, or silence using spectral features."""

    def __init__(self, sample_rate: int = 44100) -> None:
        self._sample_rate = sample_rate

    def start(self) -> None:
        """No-op — included for interface compatibility."""
        pass

    def stop(self) -> None:
        """No-op — included for interface compatibility."""
        pass

    def classify(self, audio: np.ndarray, sample_rate: int | None = None) -> dict:
        """
        Classify an audio chunk.

        Args:
            audio: float32 mono numpy array
            sample_rate: sample rate (defaults to constructor value)

        Returns:
            dict with keys:
                "category": "music" | "speech" | "silence"
                "music_score": float (0-1)
                "speech_score": float (0-1)
                "energy": float
                "is_brainrot": bool
        """
        sr = sample_rate or self._sample_rate
        audio = audio.astype(np.float32)

        # Overall energy (RMS)
        rms = float(np.sqrt(np.mean(audio ** 2)))

        if rms < 0.005:
            return {
                "category": "silence",
                "music_score": 0.0,
                "speech_score": 0.0,
                "energy": rms,
                "is_brainrot": False,
            }

        # Compute FFT over the full clip
        n = len(audio)
        fft = np.abs(np.fft.rfft(audio))
        freqs = np.fft.rfftfreq(n, d=1.0 / sr)

        # Frequency band energies
        def band_energy(low: float, high: float) -> float:
            mask = (freqs >= low) & (freqs < high)
            return float(np.mean(fft[mask] ** 2)) if np.any(mask) else 0.0

        bass = band_energy(20, 300)       # bass / beat
        mid = band_energy(300, 3000)      # speech formants
        high = band_energy(3000, 8000)    # sibilants, cymbals
        total = bass + mid + high + 1e-10

        bass_ratio = bass / total
        mid_ratio = mid / total
        high_ratio = high / total

        # Analyze energy variation over time (split into 0.5s chunks)
        chunk_size = max(1, int(sr * 0.5))
        chunks = [audio[i:i + chunk_size] for i in range(0, len(audio), chunk_size) if len(audio[i:i + chunk_size]) > chunk_size // 2]
        chunk_rms = [float(np.sqrt(np.mean(c ** 2))) for c in chunks]

        if len(chunk_rms) > 1:
            energy_std = float(np.std(chunk_rms))
            energy_mean = float(np.mean(chunk_rms))
            energy_variation = energy_std / (energy_mean + 1e-10)
        else:
            energy_variation = 0.0

        # Spectral flatness (how "noisy" vs "tonal" the signal is)
        # Music tends to be more tonal (lower flatness in specific bands)
        log_fft = np.log(fft + 1e-10)
        spectral_flatness = float(np.exp(np.mean(log_fft)) / (np.mean(fft) + 1e-10))

        # --- Scoring ---
        # Music: strong bass, consistent energy, tonal
        music_score = 0.0
        music_score += min(bass_ratio * 2.0, 0.4)         # bass presence
        music_score += max(0, 0.3 - energy_variation) * 1.0  # consistent energy
        music_score += max(0, 0.3 - spectral_flatness) * 0.5  # tonal content
        music_score += min(high_ratio * 1.5, 0.2)         # cymbals/hi-hats
        music_score = min(music_score, 1.0)

        # Speech: mid-range dominant, variable energy
        speech_score = 0.0
        speech_score += min(mid_ratio * 1.5, 0.4)         # formant range
        speech_score += min(energy_variation * 2.0, 0.3)   # natural variation
        speech_score += max(0, 0.5 - bass_ratio) * 0.3    # less bass than music
        speech_score = min(speech_score, 1.0)

        # Determine category
        if music_score > speech_score and music_score > 0.35:
            category = "music"
        elif speech_score > 0.25:
            category = "speech"
        else:
            category = "other"

        # Brain rot = music playing without much speech
        is_brainrot = category == "music" and music_score > 0.4 and speech_score < 0.3

        return {
            "category": category,
            "music_score": round(music_score, 3),
            "speech_score": round(speech_score, 3),
            "energy": round(rms, 4),
            "is_brainrot": is_brainrot,
        }

    def __enter__(self) -> "AudioClassifier":
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()
