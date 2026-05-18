# Interview Cheatsheet

## 30-Second Summary

This repository is a hardware-free AI + embedded portfolio. It covers a TinyML
predictive-maintenance node and several embedded Linux application-layer
services. The strongest demo is the audio anomaly service: it handles WAV replay
or upload, extracts lightweight features, runs anomaly inference, debounces
alarms, buffers events in SQLite, and exposes API/systemd deployment hooks.

## Best Project To Present First

**Edge Audio Anomaly Service**

- Problem: detect abnormal industrial machine sound on an edge Linux device.
- Input: synthetic WAV replay, uploaded WAV files, or MIMII/ToyADMOS-style folders.
- Model: centroid anomaly scorer by default, optional ONNX Runtime backend.
- System behavior: raw anomaly windows are debounced into stable alarm states.
- Persistence: SQLite stores events with `uploaded` and `ack` fields for offline operation.
- Ops: `/healthz`, `/metrics`, upload API, event query API, systemd service.

Good phrasing:

> I separated model output from equipment alarm state. The model can flicker on
> one window, but the service only enters alarm after consecutive anomalous
> windows. That is closer to how an industrial edge service should behave.

## Questions You May Get

**Why not use a large deep audio model first?**

Because the project focuses on embedded Linux service architecture. The feature
and backend contract is stable, so a heavier ONNX model can replace the centroid
backend later without rewriting upload, debounce, SQLite, metrics, or systemd.

**How do you handle network loss?**

Events are written to SQLite first. The `uploaded` and `ack` fields simulate a
store-and-forward upload path: rows are marked only after the remote side
acknowledges them.

**How do you avoid false alarms?**

The raw model decision is `is_anomaly_raw`; the operator-facing state is
`is_alarm`. An `AlarmDebouncer` requires consecutive bad windows to enter alarm
and consecutive good windows to recover.

**What makes this embedded Linux rather than just Python ML?**

The service has local buffering, health checks, metrics, upload endpoints,
systemd deployment files, logrotate notes, and explicit input boundaries for
ALSA/UDP/file replay.

**What should be improved with real data?**

Use actual MIMII or ToyADMOS downloads outside Git, run the public dataset
evaluation script, and report metrics separately from generated fixtures.

## Resume Bullets

- Built a hardware-free TinyML predictive-maintenance pipeline with telemetry,
  fixed-point analysis, C parameter export, and static reports.
- Built an embedded Linux audio anomaly service with WAV replay/upload,
  lightweight spectral features, ONNX-compatible inference, alarm debounce,
  SQLite buffering, Prometheus-style metrics, and systemd deployment.
- Added public industrial audio dataset support for MIMII/ToyADMOS-style folder
  layouts with window-level ROC-AUC, F1, and confusion-matrix reports.
- Implemented local edge gateway and visual inspection demos to show API,
  persistence, reporting, and batch replay patterns.
