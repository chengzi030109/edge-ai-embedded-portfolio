from __future__ import annotations

"""Optional FastAPI routes for audio analysis."""

import time
from pathlib import Path

from .alarm import AlarmDebouncer
from .backends import load_backend
from .features import extract_features, read_wav
from .storage import connect, init_db, insert_events, list_events, summary
from .streaming import analyze_wav_windows

ROUTES = [
    "POST /api/v1/audio/analyze",
    "POST /api/v1/audio/analyze-windowed",
    "GET /api/v1/audio/events",
    "GET /api/v1/audio/summary",
]


def create_app(
    model_path: str | Path = "artifacts/audio_model.json",
    database_path: str | Path = "data/audio_events.db",
    backend: str = "centroid",
    onnx_model_path: str | Path = "artifacts/audio_model.onnx",
):
    """Create the optional FastAPI application.

    FastAPI is deliberately optional because the core portfolio demo should run
    on a plain Python environment. When FastAPI is installed, this factory turns
    the same feature/model code into a Linux service endpoint. Events are stored
    in SQLite instead of only an in-memory list so the application behaves like
    an edge gateway that can keep local evidence during network outages.
    """

    try:
        from fastapi import FastAPI
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("FastAPI is optional; install requirements.txt to run the API server") from exc

    app = FastAPI(title="Edge Audio Anomaly Service")
    conn = connect(database_path)
    init_db(conn)

    @app.post("/api/v1/audio/analyze")
    def analyze(payload: dict):
        """Analyze a complete WAV file as one clip-level request.

        The windowed endpoint is closer to the embedded runtime, but this
        simpler endpoint is useful for API smoke tests and one-off uploads from
        a factory operator or a batch script.
        """

        model = load_backend(backend, centroid_model_path=model_path, onnx_model_path=onnx_model_path)
        start = time.perf_counter()
        samples, sr = read_wav(payload["path"])
        features = extract_features(samples, sr)
        feature_ms = (time.perf_counter() - start) * 1000.0
        infer_start = time.perf_counter()
        result = model.predict(features)
        inference_ms = (time.perf_counter() - infer_start) * 1000.0
        event = {
            "source": payload["path"],
            "label": str(payload.get("label", "unknown")),
            "window_index": 0,
            "start_s": 0.0,
            "end_s": float(len(samples) / sr),
            "score": result["score"],
            "threshold": result["threshold"],
            "is_anomaly_raw": result["is_anomaly"],
            "is_anomaly": result["is_anomaly"],
            "is_alarm": result["is_anomaly"],
            "alarm_state": "alarm" if result["is_anomaly"] else "normal",
            "alarm_bad_streak": 1 if result["is_anomaly"] else 0,
            "alarm_good_streak": 0 if result["is_anomaly"] else 1,
            "feature_ms": feature_ms,
            "inference_ms": inference_ms,
            "clip_path": "",
            "features": {name: float(value) for name, value in zip(model.feature_names, features, strict=False)},
        }
        insert_events(conn, [event])
        return event

    @app.post("/api/v1/audio/analyze-windowed")
    def analyze_windowed(payload: dict):
        """Analyze a WAV file as stream windows and persist every event."""

        model = load_backend(backend, centroid_model_path=model_path, onnx_model_path=onnx_model_path)
        rows = analyze_wav_windows(
            payload["path"],
            str(payload.get("label", "unknown")),
            model,
            float(payload.get("window_seconds", 0.25)),
            float(payload.get("hop_seconds", 0.125)),
            debouncer=AlarmDebouncer(
                on_count=int(payload.get("alarm_on_count", 3)),
                off_count=int(payload.get("alarm_off_count", 5)),
            ),
        )
        insert_events(conn, rows)
        return {"events": rows, "count": len(rows)}

    @app.get("/api/v1/audio/events")
    def get_events(limit: int = 50):
        return list_events(conn, limit=limit)

    @app.get("/api/v1/audio/summary")
    def get_summary():
        return summary(conn)

    return app
