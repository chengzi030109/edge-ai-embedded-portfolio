"""Tests for the 1D-CNN autoencoder anomaly detector."""

import numpy as np
import pytest

# Skip the whole module when torch is unavailable or unusable. On the user's
# Windows machine, importing torch can transitively import asyncio and hit a
# local WinSock/_overlapped problem; that is an environment issue, not a project
# failure, so local lite tests should skip cleanly.
try:
    import torch  # noqa: F401
except Exception as exc:  # pragma: no cover - depends on local Python install
    pytest.skip(f"torch is not usable in this environment: {exc}", allow_module_level=True)

from tpm.autoencoder import AutoencoderDetector


def test_autoencoder_separates_far_points():
    """Far points should be flagged as anomalous after training on a tight cluster."""

    rng = np.random.default_rng(42)
    X_train = rng.normal(0.0, 0.1, size=(128, 10)).astype(np.float32)
    far = rng.normal(6.0, 0.1, size=(4, 10)).astype(np.float32)

    det = AutoencoderDetector(epochs=80, seed=42)
    det.fit(X_train)

    result = det.predict(far[0])
    assert result["is_anomaly"] is True
    assert result["score"] > result["threshold"]


def test_autoencoder_normal_below_threshold():
    """Training-distribution points should stay below threshold."""

    rng = np.random.default_rng(7)
    X_train = rng.normal(0.0, 0.1, size=(128, 10)).astype(np.float32)

    det = AutoencoderDetector(epochs=80, seed=7)
    det.fit(X_train)

    normal_scores = [det.score(X_train[i]) for i in range(20)]
    below = sum(1 for s in normal_scores if s < det.threshold)
    assert below >= 18


def test_autoencoder_predict_contract():
    """predict() must return score, threshold, and is_anomaly keys."""

    rng = np.random.default_rng(0)
    X_train = rng.normal(0.0, 0.1, size=(64, 5)).astype(np.float32)

    det = AutoencoderDetector(input_dim=5, epochs=30)
    det.fit(X_train)

    result = det.predict(X_train[0])
    assert "score" in result
    assert "threshold" in result
    assert "is_anomaly" in result
