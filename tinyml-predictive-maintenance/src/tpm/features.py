"""Feature extraction for the simulated TinyML maintenance node.

This module represents the kind of signal-processing code that would normally
run on the MCU side before inference. The goal is to reduce a raw vibration
window, such as 256 acceleration samples, into a small fixed-length vector.

Why this matters for embedded AI:
- raw time-series windows are expensive to store and process on a small MCU
- compact features make the model smaller and easier to port to C
- FFT band features are explainable during interviews and debugging
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass(frozen=True)
class FeatureConfig:
    """Configuration shared by time-domain and frequency-domain features.

    The defaults intentionally look like a low-cost vibration node:
    - 1600 Hz sampling rate is high enough to see basic motor harmonics
    - frequency bands split low/mid/high energy for simple fault separation

    Keeping this as a dataclass makes later hardware migration easier: a real
    accelerometer driver can pass the actual sample rate without changing the
    feature extraction code.
    """

    sample_rate_hz: int = 1600
    low_band_hz: tuple[float, float] = (0.0, 80.0)
    mid_band_hz: tuple[float, float] = (80.0, 300.0)
    high_band_hz: tuple[float, float] = (300.0, 800.0)


def extract_features(samples: np.ndarray, config: FeatureConfig) -> Dict[str, float]:
    """Extract compact vibration features suitable for MCU-side inference.

    Input:
        samples: one sliding vibration window from the simulated sensor.
        config: sample rate and frequency-band definitions.

    Output:
        A dictionary with named scalar features. The dictionary is useful for
        telemetry/debugging, while ``vectorize`` below converts the same values
        into the fixed-order model input vector.
    """

    # Force float32 to mimic the numeric precision we would likely use on an
    # MCU or an embedded-Linux edge process. It also keeps serialized telemetry
    # and benchmark behavior stable across machines.
    x = np.asarray(samples, dtype=np.float32)
    if x.ndim != 1 or x.size < 8:
        raise ValueError("samples must be a 1-D array with at least 8 values")

    # Remove the DC component before RMS/FFT. In vibration monitoring, the
    # sensor offset is usually less interesting than the changing acceleration.
    centered = x - float(np.mean(x))

    # Time-domain features. These are cheap enough for an MCU and often catch
    # obvious faults: higher RMS means stronger vibration, high crest factor can
    # indicate impulse-like rubbing, and kurtosis/skewness describe shape.
    rms = float(np.sqrt(np.mean(centered * centered)))
    std = float(np.std(centered))
    abs_peak = float(np.max(np.abs(centered)))
    eps = 1e-8

    # Frequency-domain features. rfft is used because the signal is real-valued,
    # so the negative-frequency half of the FFT would be redundant.
    spectrum = np.abs(np.fft.rfft(centered))
    freqs = np.fft.rfftfreq(centered.size, d=1.0 / config.sample_rate_hz)
    power = spectrum * spectrum
    total_power = float(np.sum(power) + eps)

    # The dominant frequency is a very interpretable feature: imbalance often
    # shifts energy around the running frequency, while bearing faults tend to
    # inject higher-frequency energy.
    dominant_idx = int(np.argmax(spectrum[1:]) + 1) if spectrum.size > 1 else 0
    spectral_centroid = float(np.sum(freqs * power) / total_power)

    def band_power(band: tuple[float, float]) -> float:
        """Return normalized power in a frequency band.

        Normalization by total power makes this feature less sensitive to the
        absolute sensor scale. That is helpful when moving from simulated data
        to a real accelerometer with different gain/noise characteristics.
        """

        lo, hi = band
        mask = (freqs >= lo) & (freqs < hi)
        return float(np.sum(power[mask]) / total_power)

    # Standardized fourth moment (kurtosis). The small epsilon prevents division
    # by zero if a test passes an almost-constant window.
    fourth = float(np.mean((centered / (std + eps)) ** 4))

    return {
        "rms": rms,
        "std": std,
        "peak_to_peak": float(np.ptp(x)),
        "crest_factor": abs_peak / (rms + eps),
        "kurtosis": fourth,
        "dominant_freq_hz": float(freqs[dominant_idx]),
        "spectral_centroid_hz": spectral_centroid,
        "low_band_power": band_power(config.low_band_hz),
        "mid_band_power": band_power(config.mid_band_hz),
        "high_band_power": band_power(config.high_band_hz),
    }


FEATURE_NAMES = [
    # This fixed order is part of the model contract. If a future agent adds a
    # feature, it must retrain the model and update benchmark input size.
    "rms",
    "std",
    "peak_to_peak",
    "crest_factor",
    "kurtosis",
    "dominant_freq_hz",
    "spectral_centroid_hz",
    "low_band_power",
    "mid_band_power",
    "high_band_power",
]


def vectorize(feature_map: Dict[str, float]) -> np.ndarray:
    """Convert named features into the fixed-order model input vector."""

    return np.asarray([feature_map[name] for name in FEATURE_NAMES], dtype=np.float32)
