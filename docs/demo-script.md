# Demo Script

Use this when showing the repository to an interviewer or when recording a short
project walkthrough.

## 1. Open The GitHub Homepage

Start at:

https://github.com/chengzi030109/ai-embedded-linux-portfolio

Point out:

- CI badges are visible at the top.
- The project table shows MCU/TinyML, embedded Linux audio, gateway, vision, and benchmarking.
- Edge Audio Anomaly Service is the strongest Linux application-layer project.

Suggested wording:

> This repository is a hardware-free AI + embedded portfolio. I built one
> TinyML-style predictive-maintenance project and several embedded Linux
> application-layer services around replay, inference, SQLite buffering, APIs,
> reports, and deployment files.

## 2. Run The Edge Audio Demo

```powershell
cd E:\linux\edge-audio-anomaly-service
..\tinyml-predictive-maintenance\.venv\Scripts\python.exe scripts\run_audio_demo.py
```

Expected output:

```text
windows analyzed: 91
raw anomaly windows: 35
alarm windows: 33
backend: centroid
```

Explain:

- The service reads WAV clips as if they were stream chunks.
- It extracts lightweight spectral/statistical features.
- It separates raw model anomaly from debounced device alarm.

## 3. Show The Reports

Open:

- `edge-audio-anomaly-service/reports/audio_anomaly_report.md`
- `edge-audio-anomaly-service/reports/public_audio_evaluation.md`
- `edge-audio-anomaly-service/reports/model_deployment_report.md`

Point out:

- Audio score curve shows normal, pending anomaly, and alarm windows.
- Public-audio evaluation gives F1 and ROC-AUC on a MIMII/ToyADMOS-shaped fixture.
- Model deployment report compares centroid and ONNX Runtime parity.

Suggested wording:

> I do not claim the generated fixture is a real benchmark. It validates the
> adapter and evaluation path. For final claims I would download MIMII or
> ToyADMOS and run the same script against the real folder.

## 4. Show CI Green Status

Open the Actions tab or point to the badge in the root README.

Explain:

- CI runs edge-audio tests.
- CI runs the audio demo.
- CI runs the public-audio fixture evaluation.
- CI runs the model deployment benchmark.

## 5. Explain Linux Service Features

Open:

- `edge-audio-anomaly-service/src/edge_audio/api.py`
- `edge-audio-anomaly-service/src/edge_audio/storage.py`
- `edge-audio-anomaly-service/systemd/edge-audio-anomaly-service.service`

Talk track:

- `/healthz` proves process health.
- `/metrics` exposes Prometheus-style counters.
- `/api/v1/audio/upload` accepts WAV uploads and analyzes them.
- `/api/v1/audio/events/ack` simulates cloud acknowledgement.
- SQLite stores events before upload so network loss does not drop data.
- systemd/logrotate files show deployment shape on an embedded Linux board.

## 6. Mention ONNX Deployment

```powershell
..\tinyml-predictive-maintenance\.venv\Scripts\python.exe scripts\benchmark_model_backends.py
```

Explain:

- The default centroid backend is lightweight and inspectable.
- The ONNX backend replaces only inference.
- Audio capture, features, alarm debounce, SQLite, API, and reports remain unchanged.

## 7. Finish With TinyML

Open:

- `tinyml-predictive-maintenance/README.md`
- `tinyml-predictive-maintenance/docs/mcu-migration.md`

Suggested wording:

> The audio service is my embedded Linux application-layer showcase. The TinyML
> project is my MCU/TinyML anchor: it includes fixed-point analysis, C parameter
> export, telemetry replay, and a migration story toward FreeRTOS/CMSIS-DSP.

## 8. Closing Sentence

> The main point of this portfolio is not just model accuracy. It shows that I
> can build the engineering around edge AI: replayable inputs, lightweight
> features, deployment backends, local persistence, health/metrics, CI, and
> documentation that another engineer can run.
