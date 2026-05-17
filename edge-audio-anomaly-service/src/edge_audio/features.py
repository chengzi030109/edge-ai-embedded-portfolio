from __future__ import annotations

"""Audio loading and feature extraction."""

import wave
from pathlib import Path

import numpy as np

FEATURE_NAMES = ["rms", "zcr", "spectral_centroid_hz", "low_band", "mid_band", "high_band"]


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Read a mono WAV file as float32 samples in [-1, 1]."""

    with wave.open(str(path), "rb") as fh:
        sample_rate = fh.getframerate()
        channels = fh.getnchannels()
        width = fh.getsampwidth()
        data = fh.readframes(fh.getnframes())
    if width != 2:
        raise ValueError("demo loader expects 16-bit PCM WAV")
    pcm = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        pcm = pcm.reshape(-1, channels).mean(axis=1)
    return pcm, sample_rate


def extract_features(samples: np.ndarray, sample_rate_hz: int) -> np.ndarray:
    """Extract lightweight features suitable for an embedded Linux service."""

    x = np.asarray(samples, dtype=np.float32)
    rms = float(np.sqrt(np.mean(x * x)))
    zcr = float(np.mean(np.signbit(x[1:]) != np.signbit(x[:-1]))) if len(x) > 1 else 0.0
    spectrum = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    freqs = np.fft.rfftfreq(len(x), d=1.0 / sample_rate_hz)
    total = float(np.sum(spectrum) + 1e-12)
    centroid = float(np.sum(freqs * spectrum) / total)
    low = float(np.sum(spectrum[(freqs >= 20) & (freqs < 400)]) / total)
    mid = float(np.sum(spectrum[(freqs >= 400) & (freqs < 2000)]) / total)
    high = float(np.sum(spectrum[freqs >= 2000]) / total)
    return np.asarray([rms, zcr, centroid, low, mid, high], dtype=np.float32)


def load_feature_rows(root: str | Path) -> list[dict]:
    """Load all WAV files under normal/anomaly folders."""

    rows: list[dict] = []
    for label in ("normal", "anomaly"):
        for path in sorted((Path(root) / label).glob("*.wav")):
            samples, sr = read_wav(path)
            rows.append({"path": str(path), "label": label, "features": extract_features(samples, sr)})
    return rows

