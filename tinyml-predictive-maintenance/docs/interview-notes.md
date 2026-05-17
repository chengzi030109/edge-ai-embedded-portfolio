# Interview Notes

This document is a speaking guide for the TinyML predictive-maintenance
project. It is not meant to be read word-for-word in an interview; it gives the
main storyline, the demo order, and the answers to likely technical follow-up
questions.

## 30-Second Project Pitch

I built a hardware-free AI + embedded predictive-maintenance prototype. It
simulates an MCU/RTOS vibration monitoring node on a laptop, supports CSV replay
for sensor data, extracts lightweight signal features, runs a tiny anomaly
detector, debounces alarms, and writes telemetry/reports.

The embedded part is not just a story: the model can be exported to C, there is
float C inference parity, Q24.8 fixed-point C inference parity, and an MCU
resource-budget report for model bytes, feature buffers, sample buffers, and
telemetry size.

## 3-Minute Demo Flow

Run the one-command demo first:

```powershell
cd E:\linux\tinyml-predictive-maintenance
.\.venv\Scripts\python.exe scripts\run_portfolio_demo.py --quick
```

What to point out while it runs:

1. It trains a normal-only detector on synthetic vibration windows.
2. It runs the RTOS-style simulated node and writes JSONL telemetry.
3. It replays a CSV sensor sample, proving the input path is not hardcoded to
   the simulator.
4. It evaluates synthetic detection metrics.
5. It regenerates fixed-point drift and MCU resource reports.
6. It runs the PHM2008/C-MAPSS-shaped degradation comparison.
7. It regenerates portfolio figures and `reports/portfolio_summary.md`.

Then show:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
.\.venv\Scripts\python.exe -m pytest -q
```

Expected local result after installing the E-drive C toolchain:

```text
23 passed, 2 skipped
```

The important talking point is that C parity tests now run locally. The two
skips are optional Python ML-stack checks affected by the local Windows
`asyncio/_overlapped` issue.

## Architecture Explanation

The project is intentionally split like embedded firmware:

```text
sensor/replay
-> sample window
-> feature extraction
-> inference
-> alarm debounce
-> telemetry/gateway
```

On the laptop, the sensor is a synthetic signal generator or CSV replay. On an
MCU, only that left edge changes: it becomes ADC/I2C/SPI sensor input. The
feature contract, detector contract, alarm contract, and telemetry schema stay
stable.

## Why This Is Not Just A Toy ML Project

The project includes engineering surfaces that are easy to miss in a simple ML
demo:

- real-time style windowing instead of offline whole-file classification
- normal-only anomaly detection, which matches many maintenance settings where
  fault data is scarce
- alarm debounce, so a single noisy window does not become a device alarm
- C inference parity, so the model is not trapped in Python
- Q24.8 fixed-point C inference, so the MCU path is concrete
- resource-budget reporting, so memory/telemetry tradeoffs are visible
- multiple datasets, including CWRU and a harder PHM2008/C-MAPSS-shaped sample

## Likely Interview Questions

### Why use a centroid detector instead of a neural network?

For the first MCU target, the centroid detector is the right baseline because it
is tiny, explainable, and easy to port. It stores only mean, scale, and
threshold. Inference is subtract, multiply/divide, distance, and threshold.

The project still includes an autoencoder/ONNX path to show when I would upgrade
the model. If early faults are too subtle for hand-designed features and a
centroid distance, the learned representation becomes more attractive. But for a
resource-constrained MCU, I want the simple detector to be the first thing I can
prove end-to-end.

### Why are CWRU results near 1.0?

CWRU is a useful bearing benchmark, but many seeded faults are strongly
separated from normal windows by RMS, kurtosis, and spectral energy. So near-1.0
metrics on CWRU are not surprising and should not be oversold.

I use CWRU as a real-data sanity check, not as the only proof. That is why the
project also includes PHM2008/C-MAPSS-style gradual degradation, where the false
positive behavior is noisier and closer to a realistic maintenance discussion.

### What is Q24.8 fixed-point?

Q24.8 means a signed 32-bit integer stores a real value scaled by `2^8 = 256`.
For example, `1.5` is stored as `384`.

The fixed-point model stores:

- `mean_q`
- `inv_scale_q`
- `threshold_q`

Using reciprocal scale avoids division in inference:

```text
z_q = ((feature_q - mean_q) * inv_scale_q) >> 8
score_q = isqrt(sum(z_q * z_q))
is_alarm_raw = score_q > threshold_q
```

The Python integer reference and C implementation are tested against each other.

### What would change on real hardware?

The simulator/CSV replay would be replaced by a sensor driver. For example:

- ADC or I2C/SPI accelerometer sampling
- DMA or interrupt-backed sample buffer
- CMSIS-DSP FFT for spectral features
- FreeRTOS queues between sensor, feature, inference, alarm, and comm tasks
- UART or gateway telemetry instead of laptop JSONL files

The model interface should stay stable: fixed window in, ordered feature vector,
score/threshold out.

### How would you split FreeRTOS tasks?

| Task | Responsibility |
|---|---|
| Sensor task | Fill a ring buffer from ADC/I2C/SPI samples |
| Feature task | Convert one sample window into the ordered feature vector |
| Inference task | Run float or Q24.8 centroid inference |
| Alarm task | Apply on/off debounce and maintain alarm state |
| Comm task | Send compact telemetry to UART/gateway/MQTT path |

Sensor timing should not depend on network or dashboard output. Communication is
kept out of the timing-critical path.

### What are the current limitations?

- There is no real hardware driver yet.
- The PHM2008 sample is an offline synthetic PHM-shape fixture unless the real
  dataset is manually downloaded.
- The local Windows environment has optional `torch`/`sklearn` import issues, so
  some optional-stack tests skip locally.
- The fixed-point C path currently assumes feature values arrive as Q24.8; the
  next firmware step is direct Q-format feature extraction.

Being clear about these limits is a strength: it separates implemented evidence
from future work.

## Best Files To Open In An Interview

- `README.md`: project overview and quick start
- `reports/portfolio_summary.md`: current metrics, figures, and resume bullets
- `docs/mcu-migration.md`: MCU/RTOS migration story
- `reports/mcu_resource_budget.md`: byte budget and inference work estimate
- `firmware/inference_fixed.c`: Q24.8 C inference implementation
- `tests/test_c_fixed_inference_parity.py`: proof that Python and C fixed-point
  paths match
