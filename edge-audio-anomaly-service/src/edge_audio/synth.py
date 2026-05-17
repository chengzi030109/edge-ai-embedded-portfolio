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

    The generator intentionally avoids a perfectly clean classroom split. Real
    motors and fans vary with load, mounting, and background noise, so normal
    clips include small frequency/amplitude shifts, weak high-frequency
    components, and occasional benign clicks. Anomaly clips are still synthetic,
    but they mix subtle and obvious rubbing/impact patterns so the report is
    credible as an offline replay demo rather than a memorized toy dataset.
    """

    root = Path(root)
    rng = np.random.default_rng(2026)
    n = int(sample_rate_hz * clip_seconds)
    t = np.arange(n) / sample_rate_hz
    paths: list[Path] = []
    for idx in range(8):
        base_freq = 175.0 + rng.normal(0.0, 4.0)
        amp = rng.uniform(0.28, 0.38)
        harmonic = rng.uniform(0.04, 0.10)
        noise = rng.uniform(0.018, 0.035)
        signal = amp * np.sin(2 * math.pi * base_freq * t)
        signal += harmonic * np.sin(2 * math.pi * (2.0 * base_freq + rng.normal(0.0, 3.0)) * t)
        signal += 0.012 * np.sin(2 * math.pi * rng.uniform(1100.0, 1700.0) * t)
        signal += rng.normal(0.0, noise, size=n)
        if idx % 3 == 0:
            click_positions = rng.integers(0, n, size=2)
            signal[click_positions] += rng.choice([-0.18, 0.18], size=2)
        path = root / "normal" / f"normal_{idx:02d}.wav"
        write_wav(path, signal, sample_rate_hz)
        paths.append(path)
    for idx in range(5):
        base_freq = 175.0 + rng.normal(0.0, 5.0)
        rub_freq = rng.uniform(1450.0, 2300.0)
        rub_amp = rng.uniform(0.055, 0.13)
        signal = rng.uniform(0.30, 0.38) * np.sin(2 * math.pi * base_freq * t)
        signal += rng.uniform(0.04, 0.09) * np.sin(2 * math.pi * 2.0 * base_freq * t)
        signal += rub_amp * np.sin(2 * math.pi * rub_freq * t)
        signal += rng.normal(0.0, rng.uniform(0.028, 0.050), size=n)
        impulse_positions = rng.integers(0, n, size=rng.integers(5, 11))
        signal[impulse_positions] += rng.choice([-1.0, 1.0], size=len(impulse_positions)) * rng.uniform(0.25, 0.55)
        path = root / "anomaly" / f"anomaly_{idx:02d}.wav"
        write_wav(path, signal, sample_rate_hz)
        paths.append(path)
    return paths


def generate_public_audio_sample(root: str | Path, sample_rate_hz: int = 16000, clip_seconds: float = 1.0) -> list[Path]:
    """Generate a MIMII-shaped public-dataset fixture.

    The folder layout is intentionally closer to real industrial audio datasets
    than the simple demo folder:

    ``fan/id_00/train/normal/*.wav``
    ``fan/id_00/test/normal/*.wav``
    ``fan/id_00/test/abnormal/*.wav``

    Signals are also less separated than the basic demo. Test normals include
    operating-condition drift, and some abnormal clips are subtle. This keeps the
    evaluation pipeline honest: the sample proves the adapter works offline, but
    README/docs still tell the user to use real MIMII/ToyADMOS downloads for
    final claims.
    """

    root = Path(root)
    rng = np.random.default_rng(2027)
    n = int(sample_rate_hz * clip_seconds)
    t = np.arange(n) / sample_rate_hz
    paths: list[Path] = []

    def machine_tone(base_freq: float, amp: float, noise: float, rub_amp: float = 0.0, impulses: int = 0) -> np.ndarray:
        signal = amp * np.sin(2 * math.pi * base_freq * t)
        signal += 0.06 * np.sin(2 * math.pi * (2.0 * base_freq + rng.normal(0.0, 2.0)) * t)
        signal += 0.018 * np.sin(2 * math.pi * rng.uniform(900.0, 1500.0) * t)
        if rub_amp:
            signal += rub_amp * np.sin(2 * math.pi * rng.uniform(1300.0, 2400.0) * t)
        signal += rng.normal(0.0, noise, size=n)
        if impulses:
            positions = rng.integers(0, n, size=impulses)
            signal[positions] += rng.choice([-1.0, 1.0], size=impulses) * rng.uniform(0.12, 0.38)
        return signal

    for idx in range(10):
        signal = machine_tone(180.0 + rng.normal(0.0, 2.5), rng.uniform(0.30, 0.36), rng.uniform(0.018, 0.028))
        path = root / "fan" / "id_00" / "train" / "normal" / f"normal_train_{idx:02d}.wav"
        write_wav(path, signal, sample_rate_hz)
        paths.append(path)

    for idx in range(5):
        signal = machine_tone(180.0 + rng.normal(0.0, 7.0), rng.uniform(0.28, 0.38), rng.uniform(0.020, 0.038), impulses=1)
        path = root / "fan" / "id_00" / "test" / "normal" / f"normal_test_{idx:02d}.wav"
        write_wav(path, signal, sample_rate_hz)
        paths.append(path)

    for idx in range(6):
        signal = machine_tone(
            180.0 + rng.normal(0.0, 7.0),
            rng.uniform(0.28, 0.38),
            rng.uniform(0.028, 0.048),
            rub_amp=rng.uniform(0.025, 0.090),
            impulses=int(rng.integers(1, 5)),
        )
        path = root / "fan" / "id_00" / "test" / "abnormal" / f"abnormal_test_{idx:02d}.wav"
        write_wav(path, signal, sample_rate_hz)
        paths.append(path)

    return paths
