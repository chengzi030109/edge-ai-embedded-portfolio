from __future__ import annotations

"""Synthetic motor vibration source for hardware-free development.

Until real hardware is available, this simulator stands in for an accelerometer
attached to a motor, fan, or pump. It produces repeatable windows for several
states so the rest of the project can be built as if an MCU sensor driver
already existed.
"""

from dataclasses import dataclass
from typing import Iterator

import numpy as np


@dataclass(frozen=True)
class SignalConfig:
    """Parameters that describe the simulated sensor and motor."""

    sample_rate_hz: int = 1600
    base_freq_hz: float = 60.0
    noise_std: float = 0.04
    seed: int = 7


class MotorSignalSimulator:
    """Synthetic vibration source for normal and faulty motor states.

    The simulator keeps a running ``sample_index`` so consecutive calls produce
    a continuous signal. That is closer to real sensor sampling than generating
    each window from time zero.
    """

    def __init__(self, config: SignalConfig):
        self.config = config
        # A fixed seed keeps demos, tests, and benchmark reports reproducible.
        self.rng = np.random.default_rng(config.seed)
        self.sample_index = 0

    def read(self, n_samples: int, state: str = "normal") -> np.ndarray:
        """Return the next vibration window for the requested motor state."""

        # Convert sample indices to timestamps. A real firmware driver would
        # obtain these samples from ADC/I2C/SPI instead of synthesizing them.
        t = (np.arange(n_samples) + self.sample_index) / self.config.sample_rate_hz
        self.sample_index += n_samples

        # Normal operation is modeled as a running-frequency sine wave plus a
        # small second harmonic and sensor noise.
        base = 0.6 * np.sin(2 * np.pi * self.config.base_freq_hz * t)
        harmonic = 0.08 * np.sin(2 * np.pi * 2 * self.config.base_freq_hz * t)
        noise = self.rng.normal(0.0, self.config.noise_std, size=n_samples)
        signal = base + harmonic + noise

        if state == "normal":
            return signal.astype(np.float32)
        if state == "imbalance":
            # Imbalance increases energy near the running frequency. The 1.2x
            # multiplier makes it separable without being a totally different
            # waveform.
            return (signal + 0.35 * np.sin(2 * np.pi * 1.2 * self.config.base_freq_hz * t)).astype(np.float32)
        if state == "rubbing":
            # Rubbing or impacts create sparse impulse-like peaks. This should
            # raise crest factor and kurtosis in feature extraction.
            impulses = (self.rng.random(n_samples) < 0.015).astype(np.float32)
            return (signal + impulses * self.rng.normal(1.2, 0.2, size=n_samples)).astype(np.float32)
        if state == "bearing":
            # A bearing-like fault injects higher-frequency modulated vibration,
            # which should increase high-band power and spectral centroid.
            bearing = 0.22 * np.sin(2 * np.pi * 310.0 * t) * (1.0 + 0.4 * np.sin(2 * np.pi * 9.0 * t))
            return (signal + bearing).astype(np.float32)
        raise ValueError(f"unknown motor state: {state}")


def state_schedule(duration_s: float, window_s: float) -> Iterator[str]:
    """Yield a deterministic state sequence for a full demo run.

    The first part is normal so the console shows quiet operation, then the
    schedule moves through several fault types so alarms are easy to verify.
    """

    total_windows = int(duration_s / window_s)
    for idx in range(total_windows):
        phase = idx / max(total_windows, 1)
        if phase < 0.45:
            yield "normal"
        elif phase < 0.65:
            yield "imbalance"
        elif phase < 0.82:
            yield "rubbing"
        else:
            yield "bearing"
