from __future__ import annotations

"""Tiny centroid anomaly model for audio features."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .features import FEATURE_NAMES


@dataclass
class AudioCentroidModel:
    mean: np.ndarray
    scale: np.ndarray
    threshold: float
    feature_names: list[str]
    backend_name: str = "centroid"
    model_path: str = ""

    @classmethod
    def train(cls, vectors: list[np.ndarray]) -> "AudioCentroidModel":
        matrix = np.vstack(vectors).astype(np.float32)
        mean = np.mean(matrix, axis=0)
        scale = np.std(matrix, axis=0) + 1e-6
        scores = np.linalg.norm((matrix - mean) / scale, axis=1)
        return cls(
            mean=mean,
            scale=scale,
            threshold=float(np.quantile(scores, 0.98) * 1.4),
            feature_names=list(FEATURE_NAMES),
        )

    def predict(self, vector: np.ndarray) -> dict:
        score = float(np.linalg.norm((np.asarray(vector, dtype=np.float32) - self.mean) / self.scale))
        return {"score": score, "threshold": self.threshold, "is_anomaly": bool(score > self.threshold)}

    def save(self, path: str | Path) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "feature_names": self.feature_names,
                    "mean": self.mean.tolist(),
                    "scale": self.scale.tolist(),
                    "threshold": self.threshold,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "AudioCentroidModel":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            mean=np.asarray(payload["mean"], dtype=np.float32),
            scale=np.asarray(payload["scale"], dtype=np.float32),
            threshold=float(payload["threshold"]),
            feature_names=list(payload.get("feature_names", FEATURE_NAMES)),
        )
