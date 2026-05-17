# AI + Embedded Internship Portfolio

This workspace contains hardware-free projects designed for AI + embedded
internship applications.

## Projects

### 1. TinyML Predictive Maintenance System

Path: `tinyml-predictive-maintenance`

Simulates an MCU/RTOS vibration-monitoring node and an embedded Linux gateway.
It covers real-time sampling, FFT/statistical feature extraction, tiny anomaly
detection, telemetry, and dashboard visualization.

### 2. EdgeBench

Path: `edgebench`

A lightweight benchmark CLI for edge AI inference. It measures latency,
throughput, model footprint, and generates reproducible reports.

### 3. Edge AI Maintenance Gateway

Path: `edge-ai-maintenance-gateway`

An embedded Linux application-layer gateway. It replays TinyML telemetry,
buffers it in SQLite, exposes API route contracts, and generates a local
dashboard/report.

### 4. Edge Audio Anomaly Service

Path: `edge-audio-anomaly-service`

An industrial audio anomaly service. It generates synthetic WAV clips, extracts
audio features, runs lightweight anomaly detection, and writes reports.

### 5. Edge Vision Inspection

Path: `edge-vision-inspection`

An edge visual inspection app. It generates synthetic defect images, extracts
lightweight image features, detects defects, and saves annotated outputs.

## Suggested Resume Positioning

Use the first project as the MCU/TinyML anchor, the three edge projects as
embedded Linux application-layer demos, and EdgeBench as a supporting
measurement tool:

```text
TinyML Predictive Maintenance System
├── simulated RTOS node
├── float + Q24.8 C inference parity
├── Edge AI Maintenance Gateway
├── Edge Audio Anomaly Service
├── Edge Vision Inspection
└── EdgeBench latency/model-footprint reports
```

## Development Order

1. Run `tinyml-predictive-maintenance` end to end.
2. Use `edgebench` to benchmark its exported JSON model.
3. Add reports and screenshots to each project README.
4. Later, replace the simulated sensor with a real I2C/SPI accelerometer.

## Required Tools

- Python 3.10+
- pip
- Optional: ripgrep (`rg`) for fast code search

Install project dependencies:

```powershell
cd E:\linux\tinyml-predictive-maintenance
.\setup.ps1

cd E:\linux\edgebench
.\setup.ps1
```

If `pip install` reports `getaddrinfo failed`, the shell cannot resolve DNS.
Fix the network/DNS issue first, then rerun the setup scripts.

Optional `rg` install on Windows:

```powershell
winget install --id BurntSushi.ripgrep.MSVC -e
```

After dependencies are installed, verify the full portfolio:

```powershell
cd E:\linux
.\smoke-test.ps1
```
