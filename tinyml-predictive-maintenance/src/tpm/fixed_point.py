"""Fixed-point simulation for the centroid anomaly detector.

This module does not claim to be a final MCU implementation. Its job is to
answer the embedded-design question: "If we store centroid parameters in a
fixed-point Q format, how much drift do we introduce compared with float?"

The simulation quantizes model parameters to signed int32 Q24.8 values and
dequantizes them during scoring. Q24.8 has enough integer range for reciprocal
scales that can become large when a feature is nearly constant.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import CentroidAnomalyDetector


Q_FRACTIONAL_BITS = 8
Q_STEP = 1.0 / (1 << Q_FRACTIONAL_BITS)


@dataclass(frozen=True)
class FixedPointCentroid:
    """Q24.8-parameter approximation of ``CentroidAnomalyDetector``."""

    mean_q: np.ndarray
    inv_scale_q: np.ndarray
    threshold_q: int
    q_step: float = Q_STEP

    @classmethod
    def from_float_model(cls, model: CentroidAnomalyDetector) -> "FixedPointCentroid":
        """Quantize centroid parameters from a trained float model."""

        inv_scale = 1.0 / model.scale
        mean_q = np.round(model.mean / Q_STEP).astype(np.int32)
        inv_scale_q = np.round(inv_scale / Q_STEP).astype(np.int32)
        threshold_q = int(round(model.threshold / Q_STEP))
        return cls(
            mean_q=mean_q,
            inv_scale_q=inv_scale_q,
            threshold_q=threshold_q,
        )

    @property
    def mean(self) -> np.ndarray:
        """Dequantized mean vector."""

        return self.mean_q.astype(np.float32) * self.q_step

    @property
    def inv_scale(self) -> np.ndarray:
        """Dequantized reciprocal scale vector."""

        return self.inv_scale_q.astype(np.float32) * self.q_step

    @property
    def threshold(self) -> float:
        """Dequantized anomaly threshold."""

        return self.threshold_q * self.q_step

    @property
    def parameter_bytes(self) -> int:
        """Approximate flash footprint for Q parameters."""

        vector_bytes = int(self.mean_q.nbytes + self.inv_scale_q.nbytes)
        return vector_bytes + 4

    def score(self, vector: np.ndarray) -> float:
        """Score one feature vector using dequantized fixed-point parameters."""

        x = np.asarray(vector, dtype=np.float32)
        z = (x - self.mean) * self.inv_scale
        return float(np.linalg.norm(z))

    def predict(self, vector: np.ndarray) -> dict:
        """Return the same prediction contract as the float centroid model."""

        score = self.score(vector)
        return {
            "score": score,
            "threshold": self.threshold,
            "is_anomaly": bool(score > self.threshold),
        }


def compare_fixed_point(model: CentroidAnomalyDetector, vectors: np.ndarray) -> dict:
    """Compare float and fixed-point centroid decisions on a vector batch."""

    fixed = FixedPointCentroid.from_float_model(model)
    float_scores: list[float] = []
    fixed_scores: list[float] = []
    mismatches = 0
    for vector in vectors:
        f = model.predict(vector)
        q = fixed.predict(vector)
        float_scores.append(float(f["score"]))
        fixed_scores.append(float(q["score"]))
        if bool(f["is_anomaly"]) != bool(q["is_anomaly"]):
            mismatches += 1

    diffs = np.abs(np.asarray(float_scores) - np.asarray(fixed_scores))
    return {
        "float_model_bytes": int(model.mean.nbytes + model.scale.nbytes + 4),
        "fixed_point_bytes": fixed.parameter_bytes,
        "n_vectors": int(len(vectors)),
        "decision_mismatches": int(mismatches),
        "score_error": {
            "mean_abs": float(np.mean(diffs)),
            "max_abs": float(np.max(diffs)),
        },
        "quantization": {
            "q_fractional_bits": Q_FRACTIONAL_BITS,
            "q_step": fixed.q_step,
            "threshold_q": fixed.threshold_q,
        },
    }
