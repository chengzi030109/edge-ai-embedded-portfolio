"""1D-CNN autoencoder for anomaly detection.

This module adds a neural-network baseline to the comparison table. The
architecture is a small 1D convolutional autoencoder trained to reconstruct
normal vibration feature vectors. At inference time, reconstruction error
(MSE) is the anomaly score: high error means the input looks unlike the
training distribution.

Why an autoencoder instead of a classifier:
- Same normal-only training paradigm as the centroid detector.
- Reconstruction error is interpretable: "how different is this from normal?"
- The model is small enough to quantize to INT8 and run on a Cortex-M.

The module exposes the same ``fit/predict`` contract as the baselines so it
plugs into ``compare_models.py`` without special-casing.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class _Encoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _Decoder(nn.Module):
    def __init__(self, latent_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Autoencoder1D(nn.Module):
    """Tiny fully-connected autoencoder operating on feature vectors.

    Input dimension equals the number of features (10 by default). The
    bottleneck (latent_dim) forces the network to learn a compressed
    representation of normal vibration patterns.
    """

    def __init__(self, input_dim: int = 10, latent_dim: int = 4):
        super().__init__()
        self.encoder = _Encoder(input_dim, latent_dim)
        self.decoder = _Decoder(latent_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


@dataclass
class AutoencoderDetector:
    """Anomaly detector wrapping the 1D autoencoder.

    Exposes the same interface as the sklearn baselines and the centroid
    detector so it can be used interchangeably in the comparison script.
    """

    input_dim: int = 10
    latent_dim: int = 4
    epochs: int = 100
    lr: float = 1e-3
    batch_size: int = 32
    quantile: float = 0.995
    margin: float = 1.15
    seed: int = 42
    threshold: float = 0.0
    name: str = "Autoencoder1D"

    _model: Autoencoder1D = field(default=None, repr=False, init=False)
    _mean: np.ndarray = field(default=None, repr=False, init=False)
    _std: np.ndarray = field(default=None, repr=False, init=False)

    def fit(self, X: np.ndarray) -> "AutoencoderDetector":
        """Train the autoencoder on normal feature vectors."""

        torch.manual_seed(self.seed)
        X = np.asarray(X, dtype=np.float32)
        self.input_dim = X.shape[1]

        # Standardize inputs so the network trains stably regardless of
        # feature scale differences (Hz vs normalized power).
        self._mean = X.mean(axis=0)
        self._std = X.std(axis=0) + 1e-8
        X_norm = (X - self._mean) / self._std

        self._model = Autoencoder1D(input_dim=self.input_dim, latent_dim=self.latent_dim)
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        dataset = torch.from_numpy(X_norm)
        self._model.train()
        for _ in range(self.epochs):
            perm = torch.randperm(len(dataset))
            for start in range(0, len(dataset), self.batch_size):
                batch = dataset[perm[start : start + self.batch_size]]
                recon = self._model(batch)
                loss = criterion(recon, batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Calibrate threshold from training reconstruction errors.
        self._model.eval()
        with torch.no_grad():
            recon = self._model(dataset)
            errors = ((recon - dataset) ** 2).mean(dim=1).numpy()
        q = float(np.quantile(errors, self.quantile))
        span = float(np.max(errors) - np.min(errors))
        self.threshold = q + (self.margin - 1.0) * max(abs(q), span)
        return self

    def score(self, vector: np.ndarray) -> float:
        """Return reconstruction MSE as anomaly score."""

        x = (np.asarray(vector, dtype=np.float32) - self._mean) / self._std
        t = torch.from_numpy(x.reshape(1, -1))
        self._model.eval()
        with torch.no_grad():
            recon = self._model(t)
        return float(((recon - t) ** 2).mean().item())

    def predict(self, vector: np.ndarray) -> dict:
        s = self.score(vector)
        return {"score": s, "threshold": self.threshold, "is_anomaly": bool(s > self.threshold)}

    def export_onnx(self, path: str | Path) -> int:
        """Export the trained model to ONNX format. Returns file size in bytes."""

        self._model.eval()
        dummy = torch.randn(1, self.input_dim)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.onnx.export(
            self._model,
            dummy,
            str(path),
            input_names=["features"],
            output_names=["reconstruction"],
            dynamo=False,
        )
        return path.stat().st_size

    def export_onnx_int8(self, path: str | Path, X_cal: np.ndarray | None = None) -> int:
        """Export a dynamic-quantized INT8 ONNX model. Returns file size.

        Dynamic quantization is applied to Linear layers, which is the
        standard approach for small models where per-tensor calibration data
        is limited. This mirrors what you would do with TFLite INT8 on MCU.
        """

        from onnxruntime.quantization import QuantType, quantize_dynamic

        out_path = Path(path)
        tmp_fp32 = out_path.parent / (out_path.stem + "_tmp_fp32.onnx")
        self.export_onnx(tmp_fp32)

        quantize_dynamic(
            str(tmp_fp32),
            str(out_path),
            weight_type=QuantType.QInt8,
        )
        tmp_fp32.unlink()
        return out_path.stat().st_size

    def export_onnx_static_int8(self, path: str | Path, X_cal: np.ndarray) -> int:
        """Export a static-quantized INT8 ONNX model. Returns file size.

        Static quantization observes activations on calibration data to pick
        per-tensor scales and zero points. The result is smaller than dynamic
        quantization because activations are quantized too, not just weights.
        For tiny models this is the difference between "INT8 is bigger because
        of quant metadata" and "INT8 is meaningfully smaller".

        ``X_cal`` should be representative normal feature vectors. The same
        standardization applied during training is applied here.
        """

        from onnxruntime.quantization import (
            CalibrationDataReader,
            QuantFormat,
            QuantType,
            quantize_static,
        )

        out_path = Path(path)
        tmp_fp32 = out_path.parent / (out_path.stem + "_tmp_fp32.onnx")
        self.export_onnx(tmp_fp32)

        X_norm = (np.asarray(X_cal, dtype=np.float32) - self._mean) / self._std

        class _Reader(CalibrationDataReader):
            def __init__(self, data: np.ndarray):
                self._iter = iter(data.reshape(-1, 1, data.shape[-1]))

            def get_next(self):
                try:
                    batch = next(self._iter)
                except StopIteration:
                    return None
                return {"features": batch.astype(np.float32)}

        quantize_static(
            model_input=str(tmp_fp32),
            model_output=str(out_path),
            calibration_data_reader=_Reader(X_norm),
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            per_channel=False,
        )
        tmp_fp32.unlink()
        return out_path.stat().st_size

    def onnx_inference(self, vector: np.ndarray, onnx_path: str | Path) -> float:
        """Run inference through ONNX Runtime. Returns reconstruction MSE.

        Used by the FP32 vs INT8 precision comparison: the same input vector
        through the same standardization, but with parameters loaded from the
        exported ONNX file. Drift between this and ``score()`` is the
        quantization error.
        """

        import onnxruntime as ort

        x = (np.asarray(vector, dtype=np.float32) - self._mean) / self._std
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        recon = session.run(None, {"features": x.reshape(1, -1).astype(np.float32)})[0]
        return float(np.mean((recon - x.reshape(1, -1)) ** 2))

    def model_size_bytes(self) -> int:
        """Return the in-memory parameter size (what would go into flash)."""

        buf = io.BytesIO()
        torch.save(self._model.state_dict(), buf)
        return buf.tell()
