from __future__ import annotations

"""Optional FastAPI surface for the maintenance gateway."""

from pathlib import Path

from .storage import connect, init_db, ingest_telemetry, list_alarms, list_devices, list_telemetry, summary

ROUTES = [
    "POST /api/v1/telemetry",
    "GET /api/v1/devices",
    "GET /api/v1/telemetry",
    "GET /api/v1/alarms",
    "GET /api/v1/summary",
]


def create_app(database_path: str | Path = "data/gateway.db", default_device_id: str = "sim-node-001"):
    """Create a FastAPI app if FastAPI is installed.

    The core project does not require FastAPI so demos can run in a restricted
    environment. Installing `requirements.txt` enables this API server.
    """

    try:
        from fastapi import FastAPI
    except Exception as exc:  # pragma: no cover - depends on optional package
        raise RuntimeError("FastAPI is optional; install requirements.txt to run the API server") from exc

    app = FastAPI(title="Edge AI Maintenance Gateway")
    conn = connect(database_path)
    init_db(conn)

    @app.post("/api/v1/telemetry")
    def post_telemetry(payload: dict):
        ingest_telemetry(conn, payload, device_id=str(payload.get("device_id", default_device_id)))
        return {"ok": True}

    @app.get("/api/v1/devices")
    def get_devices():
        return list_devices(conn)

    @app.get("/api/v1/telemetry")
    def get_telemetry(device_id: str | None = None, limit: int = 50):
        return list_telemetry(conn, device_id=device_id, limit=limit)

    @app.get("/api/v1/alarms")
    def get_alarms(limit: int = 50):
        return list_alarms(conn, limit=limit)

    @app.get("/api/v1/summary")
    def get_summary():
        return summary(conn)

    return app

