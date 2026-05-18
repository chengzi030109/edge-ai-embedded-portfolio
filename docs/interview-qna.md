# Interview Q&A

This document is a short defense sheet for the portfolio. Use it before an
internship interview to keep answers concrete and tied to files in the repo.

## 1. Why Is This Embedded Linux, Not Just Python ML?

Short answer:

> I did not stop at model inference. The audio project includes service
> endpoints, local SQLite buffering, health checks, Prometheus-style metrics,
> upload/ack behavior, systemd deployment, and Docker Compose for reproducible
> service startup.

Where to point:

- `edge-audio-anomaly-service/src/edge_audio/api.py`
- `edge-audio-anomaly-service/src/edge_audio/storage.py`
- `edge-audio-anomaly-service/systemd/edge-audio-anomaly-service.service`
- `docker-compose.yml`

## 2. Why Use A Lightweight Centroid Model First?

Short answer:

> The goal is to prove the embedded service boundary first. The centroid backend
> is small, deterministic, explainable, and runs without GPU or large packages.
> The code also has an ONNX backend path, so the inference worker can be swapped
> later without rewriting upload, features, alarm logic, SQLite, API, or systemd.

Where to point:

- `edge-audio-anomaly-service/src/edge_audio/backends.py`
- `edge-audio-anomaly-service/scripts/export_audio_onnx.py`
- `edge-audio-anomaly-service/reports/model_deployment_report.md`

## 3. How Does The ONNX Upgrade Work?

Short answer:

> The service has a backend contract: feature vector in, score/threshold/anomaly
> decision out. ONNX Runtime only replaces the inference backend. Capture,
> feature extraction, alarm debounce, event storage, API routes, and metrics stay
> unchanged.

Good phrase:

> This is the difference between a model script and a deployable edge service:
> the model is a replaceable component, not the whole system.

## 4. How Do You Reduce False Alarms?

Short answer:

> The raw model output is stored as `is_anomaly_raw`, but the operator-facing
> alarm is `is_alarm`. The `AlarmDebouncer` requires consecutive anomalous
> windows before entering alarm, and consecutive normal windows before recovery.

Where to point:

- `edge-audio-anomaly-service/src/edge_audio/alarm.py`
- `edge-audio-anomaly-service/src/edge_audio/streaming.py`

## 5. How Do You Handle Network Loss?

Short answer:

> Events are written to SQLite first. The `uploaded` and `ack` fields simulate a
> store-and-forward path: the edge service can keep collecting events locally and
> mark them only after a later cloud upload is acknowledged.

Where to point:

- `edge-audio-anomaly-service/src/edge_audio/storage.py`
- `POST /api/v1/audio/events/ack`
- `GET /metrics`

## 6. What Would Change On A Real Board?

Short answer:

> The input source changes first. WAV replay becomes ALSA microphone capture,
> UDP audio packets, or a file-drop directory. The downstream pipeline can stay
> the same: windowing, features, backend inference, alarm debounce, SQLite, and
> API/dashboard.

Where to point:

- `edge-audio-anomaly-service/docs/linux-deployment.md`
- `edge-audio-anomaly-service/systemd/edge-audio-anomaly-service.service`

## 7. Why Add Docker If This Is Embedded Linux?

Short answer:

> Docker is mainly for reproducible review. It lets a recruiter or interviewer
> run the dashboard with one command. For a real board image, the systemd unit is
> the deployment shape; Docker does not replace that story.

Command:

```bash
docker compose up --build edge-audio
```

Then open:

```text
http://localhost:8080/dashboard
http://localhost:8080/healthz
http://localhost:8080/metrics
```

## 8. What Are The Current Limitations?

Short answer:

> The default demo uses generated WAVs and a small public-audio-shaped fixture,
> so I do not claim it is a production benchmark. The service path is real, but
> stronger accuracy claims need real MIMII or ToyADMOS data and a documented
> train/test split.

Where to point:

- `edge-audio-anomaly-service/docs/real-datasets.md`
- `edge-audio-anomaly-service/scripts/evaluate_public_audio_dataset.py`

## 9. What Is The Best Three-Minute Demo?

1. Open the GitHub README and show the monorepo project table.
2. Run `edge-audio-anomaly-service/scripts/run_audio_demo.py`.
3. Open `reports/audio_score_curve.png` and the model deployment report.
4. Start the dashboard with Docker Compose or uvicorn.
5. Show `/dashboard`, `/healthz`, `/metrics`, SQLite events, and systemd file.
6. Close by mentioning TinyML fixed-point and MCU migration docs.

## 10. What Should Be Improved Next?

- Replace WAV replay with an ALSA reader adapter.
- Run real MIMII/ToyADMOS data and write a verified result note.
- Add an upload worker that batches pending SQLite rows to a remote endpoint.
- Add retention policy for anomaly clips and old SQLite events.
- Add a small dashboard screenshot or GIF to the root README.
