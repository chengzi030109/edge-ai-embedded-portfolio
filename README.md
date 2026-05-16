# AI + Embedded Internship Portfolio

This workspace contains two hardware-free projects designed for AI + embedded
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

## Suggested Resume Positioning

Use the first project as the main application project and the second as a
supporting engineering tool:

```text
TinyML Predictive Maintenance System
├── simulated RTOS node
├── quantized-style anomaly inference
├── embedded Linux gateway dashboard
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
