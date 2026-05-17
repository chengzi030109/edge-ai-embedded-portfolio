# Model Deployment Report

- Feature vector size: `9`
- Benchmark samples: `64`
- Centroid JSON size: `735 bytes`
- ONNX model size: `661 bytes`
- ONNX status: `ok`

## Backend Latency

| Backend | Avg inference ms | Model path |
|---|---:|---|
| centroid | 0.002741 | audio_model.json |
| onnx | 0.020205 | audio_model.onnx |

## Parity

- Mean score error: `0.00000015`
- Max score error: `0.00000381`
- Decision mismatch count: `0`

The ONNX backend replaces only the inference worker. Audio capture,
feature extraction, alarm debounce, SQLite buffering, and API/report
logic stay unchanged.
