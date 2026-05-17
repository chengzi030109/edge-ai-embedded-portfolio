from __future__ import annotations

"""Optional FastAPI routes for audio analysis."""

from pathlib import Path

from .features import extract_features, read_wav
from .model import AudioCentroidModel

ROUTES = ["POST /api/v1/audio/analyze", "GET /api/v1/audio/events", "GET /api/v1/audio/summary"]


def create_app(model_path: str | Path = "artifacts/audio_model.json"):
    try:
        from fastapi import FastAPI
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("FastAPI is optional; install requirements.txt to run the API server") from exc

    app = FastAPI(title="Edge Audio Anomaly Service")
    events: list[dict] = []

    @app.post("/api/v1/audio/analyze")
    def analyze(payload: dict):
        model = AudioCentroidModel.load(model_path)
        samples, sr = read_wav(payload["path"])
        result = model.predict(extract_features(samples, sr))
        event = {"path": payload["path"], **result}
        events.append(event)
        return event

    @app.get("/api/v1/audio/events")
    def get_events():
        return events

    @app.get("/api/v1/audio/summary")
    def get_summary():
        return {"events": len(events), "anomalies": sum(1 for e in events if e["is_anomaly"])}

    return app

