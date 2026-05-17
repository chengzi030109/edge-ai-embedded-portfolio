from __future__ import annotations

"""Audio loading and feature extraction."""

import wave
from pathlib import Path

import numpy as np

FEATURE_NAMES = [
    "rms",
    "peak",
    "crest_factor",
    "zcr",
    "spectral_centroid_hz",
    "spectral_flatness",
    "low_band",
    "mid_band",
    "high_band",
]


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
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    crest = float(peak / (rms + 1e-9))
    zcr = float(np.mean(np.signbit(x[1:]) != np.signbit(x[:-1]))) if len(x) > 1 else 0.0
    spectrum = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    freqs = np.fft.rfftfreq(len(x), d=1.0 / sample_rate_hz)
    total = float(np.sum(spectrum) + 1e-12)
    centroid = float(np.sum(freqs * spectrum) / total)
    flatness = float(np.exp(np.mean(np.log(spectrum + 1e-12))) / (np.mean(spectrum) + 1e-12))
    low = float(np.sum(spectrum[(freqs >= 20) & (freqs < 400)]) / total)
    mid = float(np.sum(spectrum[(freqs >= 400) & (freqs < 2000)]) / total)
    high = float(np.sum(spectrum[freqs >= 2000]) / total)
    return np.asarray([rms, peak, crest, zcr, centroid, flatness, low, mid, high], dtype=np.float32)


def iter_windows(samples: np.ndarray, sample_rate_hz: int, window_seconds: float, hop_seconds: float):
    """Yield sliding audio windows with start/end timestamps.

    This models the application-layer worker that would consume chunks from
    ALSA, PulseAudio, or a UDP stream on an embedded Linux board.
    """

    window = max(1, int(sample_rate_hz * window_seconds))
    hop = max(1, int(sample_rate_hz * hop_seconds))
    x = np.asarray(samples, dtype=np.float32)
    for start in range(0, max(0, len(x) - window + 1), hop):
        end = start + window
        yield {
            "start_sample": start,
            "end_sample": end,
            "start_s": start / sample_rate_hz,
            "end_s": end / sample_rate_hz,
            "samples": x[start:end],
        }


def load_feature_rows(root: str | Path) -> list[dict]:
    """Load all WAV files under normal/anomaly folders."""

    rows: list[dict] = []
    for label in ("normal", "anomaly"):
        for path in sorted((Path(root) / label).glob("*.wav")):
            samples, sr = read_wav(path)
            rows.append({"path": str(path), "label": label, "features": extract_features(samples, sr), "sample_rate_hz": sr})
    return rows
