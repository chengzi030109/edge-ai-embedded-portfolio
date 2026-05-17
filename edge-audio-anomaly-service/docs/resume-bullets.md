# Resume Bullets

- Built an embedded Linux audio anomaly service using WAV replay, sliding-window
  stream analysis, spectral feature extraction, lightweight anomaly scoring, and
  local report generation.
- Added a public industrial audio dataset adapter for MIMII/ToyADMOS-style
  folders with window-level ROC-AUC, F1, and confusion-matrix reports.
- Aligned training and inference on the same window contract, reducing
  prototype/train-serving mismatch and making stream metrics more credible.
- Implemented alarm debounce with separate raw anomaly and stable alarm states
  to reduce one-window false alarm behavior in stream monitoring.
- Added a pluggable inference backend with optional ONNX Runtime export and
  latency/parity reporting for edge deployment.
- Designed the service so microphone/UDP input and ONNX Runtime inference can
  be added without changing the feature/report contract.
- Added SQLite event buffering, anomaly clip extraction, synthetic
  industrial-sound fixtures, API route contracts, systemd deployment notes, and
  pytest coverage.
