# Interview Notes

This project is an embedded Linux audio anomaly service. It uses WAV replay as
the hardware-free stand-in for microphone or UDP audio input, extracts
lightweight spectral features, and runs a tiny anomaly detector.

Key talking points:

- WAV replay, microphone, and UDP stream can share the same feature interface.
- The default model is intentionally small and inspectable.
- ONNX Runtime can be plugged in later without changing input/reports.
- systemd deployment shows how it would run on a Linux board.

