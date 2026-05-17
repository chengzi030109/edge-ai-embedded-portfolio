# MCU Migration Notes

This document explains how the current laptop prototype maps to a future
MCU/RTOS implementation. It is written for portfolio review: the goal is to be
honest about what is implemented today, what has been simulated, and what work
remains before flashing firmware onto real hardware.

## Current Laptop Prototype

The repository currently runs the full predictive-maintenance logic on a PC:

```text
synthetic signal or CSV replay
-> fixed-size sample window
-> FFT/statistical feature extraction
-> centroid anomaly score
-> alarm debounce
-> JSONL telemetry
-> reports and static figures
```

Implemented pieces:

- `scripts/run_simulated_node.py` runs the staged node loop and writes telemetry
  with `is_anomaly_raw`, `is_alarm`, and `alarm_state`.
- `src/tpm/features.py` extracts the same feature vector used by training,
  evaluation, replay, and C parity tests.
- `src/tpm/model.py` stores a tiny centroid detector as JSON. The model is easy
  to inspect and portable to C because it only needs `mean`, `scale`, and
  `threshold`.
- `firmware/inference.c` contains the float C inference path. The parity test
  checks that the C implementation and Python model make matching decisions on
  representative feature vectors.
- `firmware/inference_fixed.c` contains the Q24.8 integer inference path. It is
  validated against the Python integer reference in `src/tpm/fixed_point.py`.
- `scripts/fixed_point_report.py` and `scripts/mcu_resource_report.py` report
  fixed-point score drift, decision mismatches, and MCU-facing byte budgets.

Current status in one sentence: the PC prototype, float C inference, and Q24.8
integer C inference are implemented and tested; the next production step is to
make firmware feature extraction produce Q-format features directly.

## MCU Porting Boundary

The clean boundary is the sample-window input. On the laptop, windows come from
the simulator or CSV replay. On an MCU, those sources are replaced by a real
sensor driver and buffering layer.

What changes on MCU:

- Replace `MotorSignalSimulator.read(...)` or CSV replay with an ADC/I2C/SPI
  accelerometer driver.
- Replace file telemetry with UART, BLE, CAN, or a gateway-facing serial/MQTT
  protocol.
- Replace Python feature extraction with C feature extraction using a fixed
  sample buffer.

What should stay stable:

- Window contract: one inference consumes a fixed number of samples.
- Feature contract: the detector receives the same ordered feature vector.
- Alarm contract: raw anomaly decisions are debounced before becoming device
  alarms.
- Report contract: telemetry still contains sequence, timestamp, score,
  threshold, raw anomaly, debounced alarm, and timing fields.

This boundary matters because it lets the project be developed without buying
hardware first. The AI and embedded architecture can be validated on a laptop,
then the left edge of the pipeline can be replaced with a physical sensor.

## Resource Budget

The current centroid detector is intentionally small. It is a good first MCU
target because its memory and compute footprint are predictable.

| Item | Current Prototype Budget | MCU Interpretation |
|---|---:|---|
| Sample window | 256 float samples in the default synthetic path | Could become `int16_t` ADC samples or scaled accelerometer counts |
| Feature vector | 10 float features in the default vibration model | 40 bytes as float32, 20 bytes as Q15/int16 if quantized |
| Float model parameters | Mean + scale + threshold, about 84 bytes in the fixed-point report | Small enough for flash/const memory |
| Fixed-point parameters | Q24.8 report currently uses int32 arrays | Same size here because int32 mirrors float32; Q15/int16 can reduce it later |
| Telemetry payload | JSONL on laptop for readability | Binary or compact text frame on MCU; JSON can live on Linux gateway |
| Inference compute | Distance over feature vector plus threshold compare | Fits comfortably inside a periodic task after feature extraction |

Expected RTOS task split:

| Task | Period / Trigger | Responsibility | Notes |
|---|---|---|---|
| Sensor task | Sample timer or DMA half/full callback | Fill a ring buffer from ADC/I2C/SPI samples | Owns hardware timing and overrun counters |
| Feature task | Window-ready event | Convert one sample window into the ordered feature vector | Future implementation should use CMSIS-DSP FFT |
| Inference task | Feature-ready event | Run centroid score and threshold comparison | Float C and Q24.8 C paths both exist |
| Alarm task | Inference result event | Apply on/off debounce and maintain device alarm state | Prevents one noisy window from triggering a user-visible alarm |
| Comm task | Periodic or event-driven | Send telemetry to UART/MQTT/gateway | Should never block sensor acquisition |

The task design also gives a clean interview explanation: real-time acquisition
is isolated from model logic, and communication is isolated from the timing
critical path.

## Next Firmware Steps

Recommended implementation order:

1. Port feature extraction to production C.
   Use `firmware/feature_extract.c` as the starting point, then replace any
   simple spectral code with CMSIS-DSP FFT routines on ARM Cortex-M targets.

2. Tighten Q-format feature production.
   Q24.8 C inference now exists. The next firmware step is to make the feature
   task emit quantized feature vectors directly instead of converting float
   features during host-side tests.

3. Split the simulated loop into FreeRTOS tasks.
   Map the current staged node into sensor, feature, inference, alarm, and comm
   tasks. Use queues or stream buffers between stages so slow telemetry cannot
   block sampling.

4. Add a hardware abstraction layer.
   Keep the detector and feature interface independent from a specific board.
   Board-specific files should own ADC/I2C/SPI setup, timer setup, and serial
   output.

5. Decide the gateway path.
   The current repository does not implement MQTT as the main path. A practical
   next step is MCU UART telemetry into an embedded-Linux gateway, then MQTT or
   a dashboard from the gateway where memory is less constrained.

ONNX/TFLite note: the autoencoder export path is useful if the centroid model is
not expressive enough for subtle early faults. For MCU deployment, that path
would become PyTorch autoencoder -> ONNX or TFLite -> ONNX Runtime Micro or
TFLite Micro. It is documented as a migration path, not required by the current
centroid demo.
