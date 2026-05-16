from __future__ import annotations

"""Inference backend abstraction for EdgeBench.

EdgeBench is meant to grow from a simple benchmark script into a small
deployment tool. To make that possible, the runner talks to a generic backend
interface instead of directly knowing about model formats such as JSON, ONNX,
TFLite, TensorRT, or OpenVINO.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


class Backend(Protocol):
    """Minimal contract required by the benchmark runner.

    Any future backend only needs a display name and an ``infer`` method that
    accepts one NumPy input vector and returns a NumPy output vector.
    """

    name: str

    def infer(self, x: np.ndarray) -> np.ndarray:
        ...


@dataclass
class BuiltinCentroidBackend:
    """Tiny built-in backend used for smoke tests and demos.

    This avoids requiring a model file when someone only wants to verify that
    the CLI and benchmark loop work on a new machine.
    """

    input_size: int
    name: str = "builtin-centroid"

    def infer(self, x: np.ndarray) -> np.ndarray:
        """Compute distance from a zero centroid."""

        center = np.zeros(self.input_size, dtype=np.float32)
        score = np.linalg.norm(x.astype(np.float32) - center)
        return np.asarray([score], dtype=np.float32)


@dataclass
class JsonCentroidBackend:
    """Backend for the predictive-maintenance project's JSON model.

    This makes the two projects connect cleanly: the maintenance project exports
    a tiny centroid model, and EdgeBench measures its latency and footprint.
    """

    model_path: Path
    mean: np.ndarray
    scale: np.ndarray
    threshold: float
    name: str = "json-centroid"

    @classmethod
    def load(cls, model_path: str | Path) -> "JsonCentroidBackend":
        """Load and validate a supported JSON model file."""

        path = Path(model_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("type") != "centroid_anomaly_detector":
            raise ValueError("Only centroid_anomaly_detector JSON models are supported by this backend.")
        return cls(
            model_path=path,
            mean=np.asarray(payload["mean"], dtype=np.float32),
            scale=np.asarray(payload["scale"], dtype=np.float32),
            threshold=float(payload["threshold"]),
        )

    def infer(self, x: np.ndarray) -> np.ndarray:
        """Run one centroid-anomaly inference.

        The output contains both the continuous score and a binary anomaly flag
        so reports can later include accuracy-like checks if labeled data is
        added.
        """

        z = (x.astype(np.float32) - self.mean) / self.scale
        score = np.linalg.norm(z)
        return np.asarray([score, float(score > self.threshold)], dtype=np.float32)


def load_backend(model: str | None, builtin: str | None, input_size: int) -> Backend:
    """Choose the backend requested by CLI arguments."""

    if model:
        return JsonCentroidBackend.load(model)
    if builtin == "centroid":
        return BuiltinCentroidBackend(input_size=input_size)
    raise ValueError("Provide --model or --builtin centroid.")
