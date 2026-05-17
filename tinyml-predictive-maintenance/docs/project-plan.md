# Project Plan

## MVP

- Train a normal-only vibration anomaly detector.
- Run an RTOS-style simulated node.
- Write telemetry to JSONL.
- Display score and alarms in the gateway dashboard.
- Generate evaluation reports with accuracy, precision, recall, F1, false
  positive rate, per-state detection rate, and latency.
- Provide a portable C feature-extraction prototype for MCU migration.

## Advanced Milestones

- Add MQTT transport.
- Add ONNX or TFLite export.
- Add INT8/fixed-point quantization simulation.
- Add packet loss and network reconnect tests.
- Extend the C implementation with FFT band-power features through CMSIS-DSP.
- Add hardware migration guide for ESP32-S3 or STM32.
- Add screenshots from the Streamlit dashboard.
- Add a comparison table against a small 1D-CNN or autoencoder baseline.
- Add PHM2008/C-MAPSS degradation detection as a harder non-ceiling dataset.

## Interview Talking Points

- Window size and sample-rate tradeoffs.
- Why anomaly detection fits maintenance better than pure classification.
- How feature extraction reduces model memory and compute.
- What changes when moving from PC simulation to MCU firmware.
- How gateway logging supports debugging and fleet monitoring.
- Why false positive rate matters for maintenance alarms.
- Why JSON model serialization is used before moving parameters into C arrays.
