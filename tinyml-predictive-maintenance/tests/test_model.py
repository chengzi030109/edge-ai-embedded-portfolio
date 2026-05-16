"""Tests for the tiny centroid anomaly detector.

These tests protect the model contract used by training, runtime inference, and
EdgeBench. If a future agent changes serialization or score behavior, these
tests should catch accidental breakage.
"""

import numpy as np

from tpm.model import CentroidAnomalyDetector


def test_model_save_load_roundtrip(tmp_path):
    """A saved model should load back with equivalent predictions."""

    vectors = [
        np.asarray([0.0, 0.1, -0.1], dtype=np.float32),
        np.asarray([0.1, 0.0, -0.1], dtype=np.float32),
        np.asarray([-0.1, 0.1, 0.0], dtype=np.float32),
    ]
    model = CentroidAnomalyDetector.train(vectors, ["a", "b", "c"], quantile=0.9)
    path = tmp_path / "model.json"

    model.save(path)
    loaded = CentroidAnomalyDetector.load(path)

    sample = np.asarray([0.05, 0.05, -0.05], dtype=np.float32)
    assert loaded.feature_names == ["a", "b", "c"]
    assert loaded.predict(sample)["score"] == model.predict(sample)["score"]


def test_far_vector_is_anomaly():
    """A vector far from normal training data should cross the threshold."""

    vectors = [np.zeros(3, dtype=np.float32) for _ in range(8)]
    model = CentroidAnomalyDetector.train(vectors, ["a", "b", "c"], quantile=0.9)

    result = model.predict(np.asarray([10.0, 10.0, 10.0], dtype=np.float32))

    assert result["is_anomaly"] is True
