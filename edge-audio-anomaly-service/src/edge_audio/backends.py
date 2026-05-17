from __future__ import annotations

"""Inference backend abstraction for edge audio deployment.

The service started with a tiny centroid model because it is easy to inspect and
cheap enough for any embedded Linux board. This module adds the deployment seam:
the streaming, API, storage, and report layers depend on ``predict(features)``
instead of a concrete model class. That lets the project demonstrate a realistic
path from prototype scoring to ONNX Runtime without changing the rest of the
application pipeline.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from .model import AudioCentroidModel


class AudioModelBackend(Protocol):
    """Common interface implemented by every audio anomaly inference backend."""

    feature_names: list[str]
    backend_name: str
    model_path: str

    def predict(self, vector: np.ndarray) -> dict:
        """Return ``score``, ``threshold``, and ``is_anomaly`` for one feature vector."""


def load_backend(
    backend: str,
    *,
    centroid_model_path: str | Path,
    onnx_model_path: str | Path | None = None,
) -> AudioModelBackend:
    """Load an inference backend by name.

    ``centroid`` is always available and keeps the local demo lightweight.
    ``onnx`` is optional: it requires ``onnxruntime`` and a previously exported
    ONNX model. The error message is intentionally explicit because this is a
    portfolio project where a reviewer may run the default flow without heavy
    deployment packages installed.
    """

    normalized = backend.strip().lower()
    if normalized == "centroid":
        model = AudioCentroidModel.load(centroid_model_path)
        model.backend_name = "centroid"
        model.model_path = str(Path(centroid_model_path))
        return model
    if normalized == "onnx":
        if onnx_model_path is None:
            raise ValueError("ONNX backend requires onnx_model_path")
        return OnnxAudioModel.load(onnx_model_path, centroid_model_path)
    raise ValueError(f"unknown audio model backend: {backend!r}; expected 'centroid' or 'onnx'")


@dataclass
class OnnxAudioModel:
    """ONNX Runtime wrapper for the exported centroid scorer.

    The ONNX graph computes the anomaly score only. Thresholding remains in
    Python so the output contract matches ``AudioCentroidModel.predict`` and the
    alarm/storage layers do not care which backend produced the score.
    """

    session: object
    input_name: str
    threshold: float
    feature_names: list[str]
    model_path: str
    backend_name: str = "onnx"

    @classmethod
    def load(cls, onnx_model_path: str | Path, centroid_model_path: str | Path) -> "OnnxAudioModel":
        try:
            import onnxruntime as ort
        except Exception as exc:  # pragma: no cover - depends on optional package
            raise RuntimeError(
                "ONNX backend requested, but onnxruntime is not installed. "
                "Install optional dependencies with: pip install -e .[onnx]"
            ) from exc

        onnx_path = Path(onnx_model_path)
        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}. Run scripts/export_audio_onnx.py first.")

        companion = AudioCentroidModel.load(centroid_model_path)
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        return cls(
            session=session,
            input_name=input_name,
            threshold=companion.threshold,
            feature_names=companion.feature_names,
            model_path=str(onnx_path),
        )

    def predict(self, vector: np.ndarray) -> dict:
        features = np.asarray(vector, dtype=np.float32)
        outputs = self.session.run(None, {self.input_name: features})
        score = float(np.asarray(outputs[0]).reshape(-1)[0])
        return {"score": score, "threshold": self.threshold, "is_anomaly": bool(score > self.threshold)}
