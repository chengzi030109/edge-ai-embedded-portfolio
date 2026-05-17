"""Tiny anomaly detector used by the predictive-maintenance prototype.

The model is deliberately simple: it learns the centroid and standard deviation
of normal vibration features, then scores new windows by normalized distance.

This is not meant to beat deep-learning baselines. It is chosen because it is
easy to explain, tiny to serialize, and close to something that can be ported to
embedded C when hardware becomes available.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np


@dataclass
class CentroidAnomalyDetector:
    """Normal-only anomaly detector with a small JSON footprint.

    Fields:
        feature_names: feature order expected by the model.
        mean: centroid of normal feature vectors.
        scale: per-feature standard deviation used for normalization.
        threshold: score above which a frame is treated as anomalous.

    The model file is intentionally JSON instead of pickle. JSON is portable,
    inspectable in GitHub, and safer for a project that may later export the
    same parameters into C arrays for MCU firmware.
    """

    feature_names: List[str]
    mean: np.ndarray
    scale: np.ndarray
    threshold: float

    @classmethod
    def train(
        cls,
        vectors: Iterable[np.ndarray],
        feature_names: List[str],
        quantile: float = 0.995,
        margin: float = 1.15,
    ) -> "CentroidAnomalyDetector":
        """Fit the detector using only normal-condition feature vectors.

        ``quantile`` estimates the high end of normal scores. ``margin`` gives a
        little extra room so random normal windows are less likely to trigger an
        alarm. This mirrors practical condition-monitoring systems where false
        positives are costly.
        """

        # Stack the training windows into a 2-D matrix:
        # rows = windows, columns = features.
        matrix = np.vstack([np.asarray(v, dtype=np.float32) for v in vectors])

        # Per-feature normalization prevents high-scale features like frequency
        # in Hz from dominating low-scale features like normalized band power.
        mean = np.mean(matrix, axis=0)
        scale = np.std(matrix, axis=0) + 1e-6

        # Euclidean distance in z-score space is the anomaly score.
        scores = np.linalg.norm((matrix - mean) / scale, axis=1)
        threshold = float(np.quantile(scores, quantile) * margin)
        return cls(feature_names=feature_names, mean=mean, scale=scale, threshold=threshold)

    def score(self, vector: np.ndarray) -> float:
        """Return normalized distance from the learned normal centroid."""

        z = (np.asarray(vector, dtype=np.float32) - self.mean) / self.scale
        return float(np.linalg.norm(z))

    def predict(self, vector: np.ndarray) -> dict:
        """Return the full inference result used by telemetry and dashboards."""

        score = self.score(vector)
        return {
            "score": score,
            "threshold": self.threshold,
            "is_anomaly": bool(score > self.threshold),
        }

    def save(self, path: str | Path) -> None:
        """Save model parameters as portable JSON."""

        payload = {
            "type": "centroid_anomaly_detector",
            "feature_names": self.feature_names,
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "threshold": self.threshold,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CentroidAnomalyDetector":
        """Load a JSON model saved by ``save``."""

        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            feature_names=list(payload["feature_names"]),
            mean=np.asarray(payload["mean"], dtype=np.float32),
            scale=np.asarray(payload["scale"], dtype=np.float32),
            threshold=float(payload["threshold"]),
        )
