# TinyML Predictive Maintenance System

<!--
Once the repo is pushed to GitHub, replace OWNER with the actual GitHub user/org
name in the badge URL below. The workflow file is already in place at
.github/workflows/tinyml-predictive-maintenance.yml.
-->
[![CI](https://github.com/OWNER/linux/actions/workflows/tinyml-predictive-maintenance.yml/badge.svg)](https://github.com/OWNER/linux/actions/workflows/tinyml-predictive-maintenance.yml)

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

Static figures are generated for README and portfolio use:

```powershell
python scripts/generate_figures.py
```

![Synthetic score curve](reports/figures/synthetic_score_curve.png)

![Alarm debounce timeline](reports/figures/alarm_debounce_timeline.png)

## Real-World Validation: CWRU Bearing Dataset

Validating only on simulator output is not credible. The project also runs on
the [CWRU bearing dataset](https://engineering.case.edu/bearingdatacenter), the
de-facto reference for condition monitoring research, and compares the
project's centroid detector against three sklearn baselines on the same
features and the same windows.

Get the data (pick one):

```powershell
# Option A: download all 8 canonical files in one shot. Each file gets a
# 60s timeout and 2 retries. Already-downloaded files are skipped, so
# re-running after a network blip just resumes the failed ones.
python scripts/prepare_cwru.py download-all

# Option B: print the file list with URLs and download manually in a browser,
# then drop them into data/cwru/<label>/.
python scripts/prepare_cwru.py manifest

# Option C: generate CWRU-shape synthetic .mat files locally so the pipeline
# is reproducible without waiting on a download. Reports will be flagged
# "synthetic-cwru-shape" so they cannot be confused with real CWRU numbers.
python scripts/prepare_cwru.py synthetic
```

Run the comparison:

```powershell
python scripts/compare_models.py
```

Output (real CWRU data, 1024-sample windows @ 12 kHz, 280 normal training
windows, 832 test windows = 120 normal + 712 faulty across inner/outer/ball
race faults):

| Model | Accuracy | F1 | Avg latency (ms) | Size (bytes) |
|---|---:|---:|---:|---:|
| CentroidAnomalyDetector | 1.0000 | 1.0000 | 0.003 | 641 |
| IsolationForest | 0.9712 | 0.9829 | 5.005 | 1,227,444 |
| OneClassSVM | 1.0000 | 1.0000 | 0.062 | 2,600 |
| LocalOutlierFactor | 0.9988 | 0.9993 | 0.317 | 64,762 |
| Autoencoder1D (PyTorch) | 1.0000 | 1.0000 | 0.145 | 12,031 |

The autoencoder is also exported to ONNX FP32 (~9.6 KB) and ONNX INT8
(~11.6 KB, dynamic quantization). These artifacts live in ``artifacts/`` and
represent what would be deployed to an MCU via ONNX Runtime Micro or converted
to TFLite for TFLite Micro.

A separate `scripts/quantization_report.py` runs FP32 vs dynamic-INT8 vs
static-INT8 on the same test set and reports size, latency, and score drift.
The findings (in `reports/quantization_report.md`) are useful for interview
discussion: at this model scale (~3 KB of weights), INT8 metadata is a fixed
overhead that dominates the savings, and static quantization calibrated only
on normal data inflates scores on anomalies — a failure mode worth recognizing
before shipping any quantized anomaly detector.

Honest reading of these numbers: CWRU faults are seeded and severe, so RMS,
kurtosis, and frequency-band energy already separate normal from faulty
windows by a wide margin. The expected behavior on this dataset is that any
reasonable detector lands near the ceiling. The interesting comparison is
therefore deployment cost, not detection accuracy alone.

### Why the centroid detector for embedded deployment

On these features the five detectors are essentially tied on accuracy. The
relevant axis is cost:

- **Size.** The centroid detector is ~640 bytes of mean/scale/threshold. The
  autoencoder is ~12 KB (FP32 state dict) or ~9.6 KB as ONNX. The
  IsolationForest pickle is ~1.2 MB of tree splits, well over the flash budget
  of typical TinyML targets.
- **Latency.** Centroid scoring is two subtractions, two multiplies, and one
  square root per feature. Per-window inference here is ~3 µs versus 0.15 ms
  for the autoencoder and 5 ms for IsolationForest. On a 160 MHz Cortex-M4
  the centroid fits comfortably inside a single RTOS tick.
- **Portability.** The centroid model is a 4-field JSON file. Porting it to C
  is a 30-line job. The autoencoder needs an inference runtime (ONNX Runtime
  Micro or TFLite Micro), which adds ~50-100 KB of flash for the runtime
  itself. Porting an IsolationForest or RBF SVM to bare-metal C is a project
  of its own.
- **When to upgrade.** If early-stage faults with subtle spectral shifts
  become the target, the autoencoder's learned representation will likely
  outperform the centroid's fixed distance metric. The ONNX INT8 export is
  ready for that transition.

For a normal-only detector on this feature set, the centroid model is the
right tradeoff. The numbers above are the evidence, not just the claim.

Reports are written to:

```text
reports/cwru_comparison.json
reports/cwru_comparison.md
```

![CWRU model comparison](reports/figures/cwru_model_comparison.png)

## CSV Replay and Alarm Debounce

The node can replay real or exported sensor samples from CSV. The minimum
schema is `signal`; optional columns are `timestamp` and `label`.

```powershell
python scripts/run_simulated_node.py `
  --source csv `
  --input data/examples/vibration_demo.csv `
  --telemetry runs/csv_telemetry.jsonl
```

Telemetry contains both raw model decisions and debounced device alarms:

- `is_anomaly_raw`: single-window model output
- `is_alarm`: debounced alarm state
- `alarm_state`: `normal`, `pending`, `alarm`, or `recovering`

Default debounce policy is 3 consecutive anomaly windows to enter alarm and 5
consecutive normal windows to recover:

```powershell
python scripts/run_simulated_node.py --alarm-on-count 3 --alarm-off-count 5
```

## Fixed-Point / TinyML Simulation

The current C inference path uses float centroid parameters. For MCU planning,
the project also simulates storing centroid parameters in Q24.8 fixed-point
format and reports the decision drift:

```powershell
python scripts/fixed_point_report.py --model artifacts/model.json
```

Output:

```text
reports/fixed_point_report.json
reports/fixed_point_report.md
```

The report is intentionally separate from ONNX quantization. It answers a lower
level TinyML question: "Can this centroid detector become a fixed-point C
routine without changing decisions?"

## Harder Dataset: PHM 2008 / C-MAPSS

PHM 2008 is not a bearing vibration dataset. It is a NASA C-MAPSS aircraft
engine degradation dataset with multiple operating settings and sensor channels.
This project treats it as a normal-vs-degradation detection task rather than a
full RUL prediction task.

Official references:

- [data.gov PHM 2008 Challenge](https://catalog.data.gov/dataset/phm-2008-challenge-d1f2b)
- [NASA DASHlink C-MAPSS](https://c3.ndc.nasa.gov/dashlink/resources/139/)

Prepare a small PHM-shape sample for local reproducibility:

```powershell
python scripts/prepare_phm2008.py synthetic --out data/phm2008_sample/train_FD001.txt
python scripts/compare_phm2008.py --data-root data/phm2008_sample
```

The synthetic PHM-shape sample is deliberately more overlapping than CWRU. A
recent local run produced:

```text
CentroidAnomalyDetector accuracy=0.9362 f1=0.9600
```

Reports:

```text
reports/phm2008_comparison.json
reports/phm2008_comparison.md
```

## Quick Start

```powershell
cd E:\linux\tinyml-predictive-maintenance
.\setup.ps1

.\.venv\Scripts\python.exe scripts\train_model.py
.\.venv\Scripts\python.exe scripts\run_simulated_node.py --duration 4
.\.venv\Scripts\python.exe scripts\run_simulated_node.py --source csv --input data\examples\vibration_demo.csv --window-size 8
.\.venv\Scripts\python.exe scripts\evaluate_model.py --windows-per-state 40
.\.venv\Scripts\python.exe scripts\fixed_point_report.py
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
  baselines.py              sklearn baselines with a shared predict() contract
  autoencoder.py            1D autoencoder (PyTorch) with ONNX INT8 export
  alarm.py                  raw anomaly -> debounced alarm state
  fixed_point.py            Q24.8 centroid parameter simulation
  evaluation.py             metrics and report rendering
  rtos_sim.py               synchronous RTOS-style node pipeline
  telemetry.py              JSONL telemetry sink
  datasets/
    cwru.py                 CWRU bearing dataset loader
    csv_replay.py           generic timestamp/signal/label CSV replay
    phm2008.py              PHM08/C-MAPSS multivariate degradation loader
scripts/
  train_model.py            train normal-only detector
  run_simulated_node.py     run sensor -> feature -> inference -> telemetry
  evaluate_model.py         generate JSON/Markdown evaluation reports
  prepare_cwru.py           download/synthesize CWRU files into data/cwru/
  compare_models.py         centroid vs sklearn vs autoencoder on CWRU features
  quantization_report.py    FP32 / INT8-dynamic / INT8-static comparison
  fixed_point_report.py     float centroid vs Q-format simulation
  generate_figures.py       static PNG plots for README/reports
  prepare_phm2008.py        manifest/synthetic PHM08/C-MAPSS sample generator
  compare_phm2008.py        centroid/baseline comparison on PHM08 windows
  export_model_to_c.py      generate firmware/model_params.h from JSON model
gateway/
  app.py                    Streamlit gateway dashboard
firmware/
  feature_extract.c/.h      portable C time-domain feature prototype
  inference.c/.h            portable C centroid inference for MCU deployment
  test_inference.c          host-side harness for Python-C parity testing
  model_params.h            generated trained-model constants (auto-built)
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
20 passed, 3 skipped
```

Skipped tests are optional-stack checks: local Windows environments without a C
compiler, or with broken `torch`/`sklearn` imports due to system
`asyncio/_overlapped` issues, skip those modules cleanly. Linux CI exercises the
full path.

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

The `firmware/` folder now contains both halves of the inference pipeline in
portable C:

1. **`feature_extract.c`** — time-domain features (mean, RMS, std, peak-to-peak,
   crest factor) ready for an RTOS task or bare-metal loop.
2. **`inference.c`** — centroid anomaly inference (per-feature z-score, L2
   distance, threshold). Numerical parity with the Python implementation is
   verified on every CI run by `tests/test_c_inference_parity.py`.

Trained model parameters are exported from JSON to a compile-time C header by
`scripts/export_model_to_c.py`. The whole model on the MCU side fits in
~84 bytes of flash for the 10-feature centroid detector.

FFT band-power features can be added later with CMSIS-DSP on Cortex-M, or with
a target-specific DSP library. The autoencoder path uses ONNX Runtime Micro or
TFLite Micro and is ready to deploy via the artifacts in `artifacts/`.
