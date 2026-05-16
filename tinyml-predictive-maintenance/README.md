# TinyML Predictive Maintenance System

Hardware-free prototype of an embedded AI predictive-maintenance node.

This project simulates the full path of an MCU/RTOS vibration-monitoring device
plus an embedded-Linux style gateway:

```text
synthetic motor vibration stream
-> fixed-size sliding windows
-> FFT/statistical feature extraction
-> tiny anomaly detector
-> JSONL telemetry
-> evaluation reports and dashboard
```

The current code runs on a laptop today. Later, the simulated sensor can be
replaced by an I2C/SPI accelerometer on ESP32-S3, STM32, or another MCU.

## Why This Is Internship-Oriented

This is not just "train a model and print accuracy". It demonstrates embedded
AI engineering habits:

- real-time windowed signal processing
- RTOS-style stage separation: sensor, feature, inference, communication
- feature extraction under memory/compute constraints
- normal-only anomaly detection for maintenance scenarios
- portable JSON model serialization
- telemetry, logging, and gateway visualization
- evaluation metrics, latency metrics, and model-footprint reporting
- a C feature-extraction prototype for MCU migration

## Current Demo Results

Generated with:

```powershell
python scripts/train_model.py --out artifacts/model.json --windows 120
python scripts/evaluate_model.py --model artifacts/model.json --windows-per-state 40
```

| Metric | Value |
|---|---:|
| Accuracy | 0.9750 |
| Precision | 1.0000 |
| Recall | 0.9667 |
| F1 | 0.9831 |
| False positive rate | 0.0000 |
| Model file size | 856 bytes |
| Feature extraction avg latency | 0.105816 ms |
| Inference avg latency | 0.005232 ms |

Per-state detection:

| State | Windows | Detected as anomaly | Detection rate |
|---|---:|---:|---:|
| normal | 40 | 0 | 0.0000 |
| imbalance | 40 | 40 | 1.0000 |
| rubbing | 40 | 36 | 0.9000 |
| bearing | 40 | 40 | 1.0000 |

Full reports are written to:

```text
reports/evaluation.json
reports/evaluation.md
```

## Quick Start

```powershell
cd E:\linux\tinyml-predictive-maintenance
.\setup.ps1

.\.venv\Scripts\python.exe scripts\train_model.py
.\.venv\Scripts\python.exe scripts\run_simulated_node.py --duration 4
.\.venv\Scripts\python.exe scripts\evaluate_model.py --windows-per-state 40
```

Telemetry is written to:

```text
runs/telemetry.jsonl
```

Optional dashboard:

```powershell
.\.venv\Scripts\streamlit.exe run gateway\app.py
```

## Configuration

Default project settings live in:

```text
configs/default.json
```

Scripts load this config by default and allow command-line overrides. Example:

```powershell
.\.venv\Scripts\python.exe scripts\train_model.py --config configs/default.json --windows 800
```

Important fields:

- `sample_rate_hz`: simulated sensor sampling rate
- `window_size`: samples per inference window
- `train_windows`: normal windows used for training
- `states`: states used by evaluation
- `model_path`: JSON model output
- `telemetry_path`: JSONL node output

## Project Layout

```text
configs/
  default.json              reproducible project settings
src/tpm/
  config.py                 typed config loader
  signal_sim.py             synthetic motor vibration generator
  features.py               MCU-friendly feature extraction
  model.py                  tiny centroid anomaly detector
  evaluation.py             metrics and report rendering
  rtos_sim.py               synchronous RTOS-style node pipeline
  telemetry.py              JSONL telemetry sink
scripts/
  train_model.py            train normal-only detector
  run_simulated_node.py     run sensor -> feature -> inference -> telemetry
  evaluate_model.py         generate JSON/Markdown evaluation reports
gateway/
  app.py                    Streamlit gateway dashboard
firmware/
  feature_extract.c/.h      portable C time-domain feature prototype
tests/
  test_*.py                 unit and smoke tests
```

## Testing

On this Windows environment, disabling third-party pytest plugin autoload avoids
unrelated network/asyncio plugins from interfering with simple unit tests:

```powershell
cd E:\linux\tinyml-predictive-maintenance
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
.\.venv\Scripts\python.exe -m pytest
```

Expected result:

```text
8 passed
```

## Hardware Migration Story

When hardware becomes available, replace only the synthetic sensor source:

```text
signal_sim.py -> I2C/SPI accelerometer driver
```

Most of the application should stay stable:

- feature vector contract
- model thresholding
- telemetry schema
- evaluation scripts
- gateway/dashboard
- EdgeBench latency/model-footprint reports

The `firmware/` folder contains the first C migration step for time-domain
features. FFT band-power features can later be implemented with CMSIS-DSP or a
target-specific DSP library.

