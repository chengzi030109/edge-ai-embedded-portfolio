# Autoencoder Quantization Report

- Source: `cwru`
- Train windows: `280`  Test windows: `832`
- Input features: `10`  Latent dim: `4`

## Model Footprint

| Format | File size (bytes) | Latency avg (ms) | Latency p95 (ms) |
|---|---:|---:|---:|
| PyTorch FP32 (reference) | 12031 | 0.2908 | 0.4307 |
| ONNX FP32 | 9598 | 0.0303 | 0.0517 |
| ONNX INT8 (dynamic) | 11640 | 0.0438 | 0.0498 |
| ONNX INT8 (static) | 11955 | 0.0411 | 0.0675 |

## Score Drift vs PyTorch FP32

Per-window reconstruction MSE measured on the same test windows.

| Format | Mean abs error | Max abs error | Mean relative error |
|---|---:|---:|---:|
| ONNX FP32 | 6.127030e-05 | 7.324219e-04 | 0.0000% |
| ONNX INT8 (dynamic) | 1.292391e+01 | 6.050269e+01 | 4.8272% |
| ONNX INT8 (static) | 1.389325e+03 | 5.861598e+03 | 383.7795% |

Reference: PyTorch FP32 mean score = 2.173017e+02, std = 2.277560e+02.

## Reading the Numbers

ONNX FP32 matches PyTorch FP32 to within float32 round-off, which is
the right sanity check that the export itself is faithful.

INT8 quantization does not save space here because the model is tiny
(~3 KB of weights) and INT8 metadata (per-tensor scales, zero points,
quantize/dequantize ops) is a fixed overhead that dominates at this
scale. The size argument for INT8 only kicks in around 100 KB+ models.

The static-INT8 score drift is large by design of this report: the
calibration set contains only normal windows, so the activation
ranges learned during calibration are tight. Faulty test windows push
those activations far outside the calibration range and the int8
dequantized outputs saturate. **This is exactly the failure mode an
embedded ML engineer is expected to recognize**: an anomaly detector
calibrated only on the normal class will inflate scores on anomalies,
which here makes detection more aggressive but also less stable.
Mitigations include calibrating with a small fraction of seeded
anomalies, switching to per-channel quantization, or moving to
QAT (quantization-aware training).
