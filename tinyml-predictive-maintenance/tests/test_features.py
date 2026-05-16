"""Unit tests for MCU-style feature extraction.

The test uses a clean 60 Hz sine wave so the expected dominant frequency is
obvious. This protects the feature contract used by training, runtime inference,
and edgebench input-size assumptions.
"""

import numpy as np

from tpm.features import FeatureConfig, extract_features, vectorize


def test_extract_features_returns_stable_vector():
    # Generate one deterministic vibration window at 60 Hz.
    sample_rate = 1600
    t = np.arange(256) / sample_rate
    samples = np.sin(2 * np.pi * 60 * t).astype(np.float32)

    features = extract_features(samples, FeatureConfig(sample_rate_hz=sample_rate))
    vector = vectorize(features)

    # The vector size is part of the model/benchmark contract.
    assert vector.shape[0] == 10

    # A unit sine wave has RMS near 0.707; the loose bound leaves room for the
    # finite window not containing an exact integer number of periods.
    assert features["rms"] > 0.6

    # FFT bin resolution is finite, so check a frequency range rather than a
    # single exact value.
    assert 50 <= features["dominant_freq_hz"] <= 70
