from __future__ import annotations

"""Synthetic WAV generation for hardware-free audio demos."""

import math
import wave
from pathlib import Path

import numpy as np


def write_wav(path: str | Path, samples: np.ndarray, sample_rate_hz: int) -> None:
    """Write mono int16 WAV data."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * 32767).astype("<i2")
    with wave.open(str(out), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(sample_rate_hz)
        fh.writeframes(pcm.tobytes())


def generate_demo_wavs(root: str | Path, sample_rate_hz: int = 16000, clip_seconds: float = 1.0) -> list[Path]:
    """Generate normal and anomalous machine-sound clips.

    Normal clips are stable fan/motor tones. Anomaly clips add impulses and a
    high-frequency rubbing component, which is enough for a lightweight
    detector to separate them without external datasets.
    """

    root = Path(root)
    rng = np.random.default_rng(2026)
    n = int(sample_rate_hz * clip_seconds)
    t = np.arange(n) / sample_rate_hz
    paths: list[Path] = []
    for idx in range(8):
        signal = 0.35 * np.sin(2 * math.pi * 180 * t) + 0.08 * np.sin(2 * math.pi * 360 * t)
        signal += rng.normal(0.0, 0.015, size=n)
        path = root / "normal" / f"normal_{idx:02d}.wav"
        write_wav(path, signal, sample_rate_hz)
        paths.append(path)
    for idx in range(5):
        signal = 0.35 * np.sin(2 * math.pi * 180 * t) + 0.2 * np.sin(2 * math.pi * 1900 * t)
        signal += rng.normal(0.0, 0.04, size=n)
        impulse_positions = rng.integers(0, n, size=12)
        signal[impulse_positions] += rng.choice([-0.8, 0.8], size=12)
        path = root / "anomaly" / f"anomaly_{idx:02d}.wav"
        write_wav(path, signal, sample_rate_hz)
        paths.append(path)
    return paths

