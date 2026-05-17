from __future__ import annotations

"""Optional FastAPI routes for audio analysis."""

import time
from pathlib import Path

from .alarm import AlarmDebouncer
from .backends import load_backend
from .features import extract_features, read_wav
from .storage import connect, init_db, insert_events, list_events, mark_events_uploaded, summary
from .streaming import analyze_wav_windows

ROUTES = [
    "POST /api/v1/audio/analyze",
    "POST /api/v1/audio/analyze-windowed",
    "POST /api/v1/audio/upload",
    "POST /api/v1/audio/events/ack",
    "GET /api/v1/audio/events",
    "GET /api/v1/audio/summary",
    "GET /healthz",
    "GET /metrics",
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
        from fastapi import FastAPI, File, UploadFile
        from fastapi.responses import PlainTextResponse
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("FastAPI is optional; install requirements.txt or pip install -e .[api] to run the API server") from exc

    app = FastAPI(title="Edge Audio Anomaly Service")
    conn = connect(database_path)
    init_db(conn)
    upload_dir = Path(database_path).resolve().parent / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    def current_model():
        """Load the configured backend on demand so model files can be replaced."""

        return load_backend(backend, centroid_model_path=model_path, onnx_model_path=onnx_model_path)

    @app.post("/api/v1/audio/analyze")
    def analyze(payload: dict):
        """Analyze a complete WAV file as one clip-level request.

        The windowed endpoint is closer to the embedded runtime, but this
        simpler endpoint is useful for API smoke tests and one-off uploads from
        a factory operator or a batch script.
        """

        model = current_model()
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

        model = current_model()
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

    @app.post("/api/v1/audio/upload")
    async def upload_audio(
        file: UploadFile = File(...),
        label: str = "unknown",
        window_seconds: float = 0.25,
        hop_seconds: float = 0.125,
        alarm_on_count: int = 3,
        alarm_off_count: int = 5,
    ):
        """Upload a WAV file, persist it locally, analyze windows, and buffer events.

        On a board this mimics a file-drop or remote operator upload workflow.
        The uploaded bytes are saved first so later debugging can replay the
        exact audio that produced each SQLite event.
        """

        safe_name = Path(file.filename or "upload.wav").name
        saved_path = upload_dir / f"{int(time.time() * 1000)}_{safe_name}"
        saved_path.write_bytes(await file.read())
        model = current_model()
        rows = analyze_wav_windows(
            saved_path,
            str(label),
            model,
            float(window_seconds),
            float(hop_seconds),
            debouncer=AlarmDebouncer(on_count=int(alarm_on_count), off_count=int(alarm_off_count)),
        )
        insert_events(conn, rows)
        return {
            "path": str(saved_path),
            "count": len(rows),
            "raw_anomalies": sum(1 for row in rows if row["is_anomaly_raw"]),
            "alarms": sum(1 for row in rows if row["is_alarm"]),
            "events": rows,
        }

    @app.post("/api/v1/audio/events/ack")
    def ack_events(payload: dict):
        """Mark buffered events as uploaded and optionally acknowledged."""

        event_ids = [int(event_id) for event_id in payload.get("event_ids", [])]
        changed = mark_events_uploaded(conn, event_ids, ack=bool(payload.get("ack", True)))
        return {"updated": changed, "ack": bool(payload.get("ack", True))}

    @app.get("/api/v1/audio/events")
    def get_events(limit: int = 50):
        return list_events(conn, limit=limit)

    @app.get("/api/v1/audio/summary")
    def get_summary():
        return summary(conn)

    @app.get("/healthz")
    def healthz():
        return {
            "status": "ok",
            "backend": backend,
            "model_exists": Path(model_path).exists() if backend == "centroid" else Path(onnx_model_path).exists(),
            "database": str(database_path),
        }

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics():
        stats = summary(conn)
        lines = [
            "# HELP edge_audio_events_total Total audio window events buffered locally.",
            "# TYPE edge_audio_events_total counter",
            f"edge_audio_events_total {stats['event_count']}",
            "# HELP edge_audio_raw_anomalies_total Raw anomaly windows emitted by the model.",
            "# TYPE edge_audio_raw_anomalies_total counter",
            f"edge_audio_raw_anomalies_total {stats['raw_anomaly_count']}",
            "# HELP edge_audio_alarm_windows_total Debounced alarm windows.",
            "# TYPE edge_audio_alarm_windows_total counter",
            f"edge_audio_alarm_windows_total {stats['alarm_count']}",
            "# HELP edge_audio_pending_upload_events Locally buffered events not marked uploaded.",
            "# TYPE edge_audio_pending_upload_events gauge",
            f"edge_audio_pending_upload_events {stats['pending_upload_count']}",
            "# HELP edge_audio_inference_ms_avg Average model inference latency in milliseconds.",
            "# TYPE edge_audio_inference_ms_avg gauge",
            f"edge_audio_inference_ms_avg {stats['inference_ms_avg']:.6f}",
            "",
        ]
        return "\n".join(lines)

    return app
