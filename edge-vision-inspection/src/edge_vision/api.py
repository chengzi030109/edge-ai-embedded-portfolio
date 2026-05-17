from __future__ import annotations

"""Optional FastAPI routes for image inspection."""

from pathlib import Path

from .features import extract_features
from .model import VisionCentroidModel

ROUTES = ["POST /api/v1/images/analyze", "GET /api/v1/images/results", "GET /api/v1/images/summary"]


def create_app(model_path: str | Path = "artifacts/vision_model.json"):
    try:
        from fastapi import FastAPI
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("FastAPI is optional; install requirements.txt to run the API server") from exc

    app = FastAPI(title="Edge Vision Inspection")
    results: list[dict] = []

    @app.post("/api/v1/images/analyze")
    def analyze(payload: dict):
        model = VisionCentroidModel.load(model_path)
        result = model.predict(extract_features(payload["path"]))
        event = {"path": payload["path"], **result}
        results.append(event)
        return event

    @app.get("/api/v1/images/results")
    def get_results():
        return results

    @app.get("/api/v1/images/summary")
    def get_summary():
        return {"results": len(results), "defects": sum(1 for r in results if r["is_defect"])}

    return app

