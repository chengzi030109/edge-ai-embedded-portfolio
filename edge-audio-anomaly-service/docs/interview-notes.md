# Interview Notes

This project is an embedded Linux audio anomaly service. It uses WAV replay as
the hardware-free stand-in for microphone or UDP audio input, processes audio as
sliding windows, extracts lightweight spectral features, and runs a tiny
anomaly detector.

Key talking points:

- WAV replay, microphone, and UDP stream can share the same feature interface.
- The data adapter supports MIMII/ToyADMOS-style public folders, so the same
  project can move from generated WAV fixtures to real industrial audio.
- Training and inference both use the same sliding-window shape, which avoids a
  common prototype mistake where full-clip training looks good but online
  windows drift.
- The model output is not treated as the final equipment alarm. A small
  debouncer requires consecutive anomalous windows to enter alarm and
  consecutive normal windows to recover.
- The default model is intentionally small and inspectable.
- The inference backend is abstracted: `centroid` is the default, and ONNX
  Runtime can be selected later without changing the stream/storage/API layers.
- SQLite stores window-level events as an edge buffer.
- Anomaly clips are saved for later inspection.
- ONNX Runtime can be plugged in later without changing input/reports.
- systemd deployment shows how it would run on a Linux board.

Good interview framing:

> The first version is not trying to be a huge deep audio model. It is an
> embedded Linux service skeleton with deterministic replay, stream windows,
> local buffering, explainable features, and deployment hooks.

If the interviewer asks why not start with a large model, answer that the
application-layer architecture is the point of this project: input replay,
window timing, local persistence, API contract, and deployment can be validated
before replacing the centroid scorer with ONNX Runtime.

If asked about false alarms, point to the separation between `is_anomaly_raw`
and `is_alarm`: the raw score is useful for analysis, while `is_alarm` is the
debounced state that a real device would expose to operators.

If asked about deployment depth, explain the upgrade path as:
`centroid prototype -> ONNX export -> ONNX Runtime backend -> same alarm and
SQLite edge service`. The important engineering point is that the deployment
runtime is isolated behind a stable `predict(features)` contract.

If asked about dataset credibility, be explicit: the built-in sample is a
hardware-free fixture that validates the loader and report path; final accuracy
claims should come from real MIMII or ToyADMOS downloads.
