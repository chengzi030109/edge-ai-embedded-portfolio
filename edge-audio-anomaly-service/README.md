# Edge Audio Anomaly Service

Embedded Linux audio anomaly detection service for industrial machine sounds.
The first version runs without hardware by generating synthetic WAV files.

## Quick Start

```powershell
cd E:\linux\edge-audio-anomaly-service
..\tinyml-predictive-maintenance\.venv\Scripts\python.exe scripts\run_audio_demo.py
..\tinyml-predictive-maintenance\.venv\Scripts\python.exe -m pytest -q
```

Outputs:

- `data/wav/`
- `artifacts/audio_model.json`
- `reports/audio_anomaly_report.md`
- `reports/audio_score_curve.png`

## API Contract

- `POST /api/v1/audio/analyze`
- `GET /api/v1/audio/events`
- `GET /api/v1/audio/summary`

FastAPI is optional for the first local demo.

