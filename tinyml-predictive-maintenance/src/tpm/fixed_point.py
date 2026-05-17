"""Fixed-point simulation for the centroid anomaly detector.

This module does not claim to be a final MCU implementation. Its job is to
answer the embedded-design question: "If we store centroid parameters in a
fixed-point Q format, how much drift do we introduce compared with float?"

The simulation quantizes model parameters to signed int32 Q24.8 values. It has
two scoring paths:

* ``score`` dequantizes parameters and is useful for human-readable drift
  reports.
* ``score_q`` keeps the MCU-style integer pipeline, including quantized input
  features and integer square root. That path is the Python reference for
  ``firmware/inference_fixed.c`` parity tests.

Q24.8 has enough integer range for reciprocal scales that can become large when
a feature is nearly constant.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .model import CentroidAnomalyDetector


Q_FRACTIONAL_BITS = 8
Q_STEP = 1.0 / (1 << Q_FRACTIONAL_BITS)
Q_SCALE = 1 << Q_FRACTIONAL_BITS


def quantize_q24_8(value: float | np.ndarray) -> int | np.ndarray:
    """Quantize a scalar or numpy array to signed Q24.8 ``int32`` values.

    The helper mirrors the firmware conversion rule: round to nearest after
    multiplying by 2^8. Values are clipped to int32 range so accidental extreme
    features fail predictably instead of wrapping around.
    """

    raw = np.asarray(value, dtype=np.float64) * Q_SCALE
    scaled = np.where(raw >= 0.0, np.floor(raw + 0.5), np.ceil(raw - 0.5))
    clipped = np.clip(scaled, np.iinfo(np.int32).min, np.iinfo(np.int32).max).astype(np.int32)
    if np.ndim(value) == 0:
        return int(clipped)
    return clipped


def dequantize_q24_8(value: int | np.ndarray) -> float | np.ndarray:
    """Convert Q24.8 integer values back to float for reporting/debugging."""

    out = np.asarray(value, dtype=np.float32) / Q_SCALE
    if np.ndim(value) == 0:
        return float(out)
    return out


def _isqrt_u64(value: int) -> int:
    """Integer square root used to match the C fixed-point score path."""

    if value < 0:
        raise ValueError("integer square root input must be non-negative")
    return math.isqrt(value)


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

    def score_q(self, vector: np.ndarray) -> int:
        """Score one vector with the same integer math as the C fixed path.

        Feature values are first quantized to Q24.8, which is what a firmware
        task would normally pass between feature extraction and inference. Each
        normalized component remains Q24.8:

            z_q = ((x_q - mean_q) * inv_scale_q) >> 8

        Squaring Q24.8 values produces Q48.16 terms. The integer square root of
        their sum returns a Q24.8 score, directly comparable with threshold_q.
        """

        x_q = quantize_q24_8(np.asarray(vector, dtype=np.float32)).astype(np.int64)
        mean_q = self.mean_q.astype(np.int64)
        inv_scale_q = self.inv_scale_q.astype(np.int64)
        z_q = ((x_q - mean_q) * inv_scale_q) >> Q_FRACTIONAL_BITS
        sum_squares_q16 = int(np.sum(z_q * z_q, dtype=np.int64))
        return _isqrt_u64(sum_squares_q16)

    def score_integer(self, vector: np.ndarray) -> float:
        """Return the integer-path score as a dequantized float."""

        return float(dequantize_q24_8(self.score_q(vector)))

    def predict(self, vector: np.ndarray) -> dict:
        """Return the same prediction contract as the float centroid model."""

        score = self.score(vector)
        return {
            "score": score,
            "threshold": self.threshold,
            "is_anomaly": bool(score > self.threshold),
        }

    def predict_integer(self, vector: np.ndarray) -> dict:
        """Predict with the integer pipeline used by the firmware fixed path."""

        score_q = self.score_q(vector)
        return {
            "score_q": score_q,
            "score": float(dequantize_q24_8(score_q)),
            "threshold_q": self.threshold_q,
            "threshold": self.threshold,
            "is_anomaly": bool(score_q > self.threshold_q),
        }


def compare_fixed_point(model: CentroidAnomalyDetector, vectors: np.ndarray) -> dict:
    """Compare float and fixed-point centroid decisions on a vector batch."""

    fixed = FixedPointCentroid.from_float_model(model)
    float_scores: list[float] = []
    fixed_scores: list[float] = []
    integer_scores: list[float] = []
    mismatches = 0
    integer_mismatches = 0
    for vector in vectors:
        f = model.predict(vector)
        q = fixed.predict(vector)
        qi = fixed.predict_integer(vector)
        float_scores.append(float(f["score"]))
        fixed_scores.append(float(q["score"]))
        integer_scores.append(float(qi["score"]))
        if bool(f["is_anomaly"]) != bool(q["is_anomaly"]):
            mismatches += 1
        if bool(f["is_anomaly"]) != bool(qi["is_anomaly"]):
            integer_mismatches += 1

    diffs = np.abs(np.asarray(float_scores) - np.asarray(fixed_scores))
    integer_diffs = np.abs(np.asarray(float_scores) - np.asarray(integer_scores))
    return {
        "float_model_bytes": int(model.mean.nbytes + model.scale.nbytes + 4),
        "fixed_point_bytes": fixed.parameter_bytes,
        "n_vectors": int(len(vectors)),
        "decision_mismatches": int(mismatches),
        "integer_path_decision_mismatches": int(integer_mismatches),
        "score_error": {
            "mean_abs": float(np.mean(diffs)),
            "max_abs": float(np.max(diffs)),
        },
        "integer_path_score_error": {
            "mean_abs": float(np.mean(integer_diffs)),
            "max_abs": float(np.max(integer_diffs)),
        },
        "quantization": {
            "q_fractional_bits": Q_FRACTIONAL_BITS,
            "q_step": fixed.q_step,
            "threshold_q": fixed.threshold_q,
        },
    }
