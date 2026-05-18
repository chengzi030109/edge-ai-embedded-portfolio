from __future__ import annotations

"""Optional FastAPI routes for image inspection.

The API accepts image paths because the local demo replays files from disk
instead of using a real USB camera or RTSP stream. That is convenient for a
hardware-free portfolio, but a service must not blindly open any path supplied
over HTTP. ``create_app`` therefore constrains path-based analysis to a small
set of safe roots, mirroring the audio service.
"""

from pathlib import Path

from .features import extract_features
from .model import VisionCentroidModel

ROUTES = ["POST /api/v1/images/analyze", "GET /api/v1/images/results", "GET /api/v1/images/summary"]


def create_app(model_path: str | Path = "artifacts/vision_model.json", safe_roots: list[str | Path] | None = None):
    """Create the optional FastAPI application.

    ``safe_roots`` extends the default allowed image directories. The default
    covers ``data/`` for local demos and the model directory for small tests
    that keep generated fixtures beside the model artifact.
    """

    try:
        from fastapi import FastAPI, HTTPException
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("FastAPI is optional; install pip install -e .[api] to run the API server") from exc

    app = FastAPI(title="Edge Vision Inspection")
    results: list[dict] = []
    model_file = Path(model_path).resolve()
    default_roots = [Path("data").resolve(), model_file.parent]
    allowed_roots = [root.resolve() for root in default_roots + [Path(p).resolve() for p in (safe_roots or [])]]

    def resolve_safe_path(raw: str | Path | None) -> Path:
        """Return a real image path only when it is inside an allowed root."""

        if not raw:
            raise HTTPException(status_code=400, detail="path is required")
        candidate = Path(str(raw)).resolve()
        for root in allowed_roots:
            try:
                candidate.relative_to(root)
            except ValueError:
                continue
            if not candidate.is_file():
                raise HTTPException(status_code=404, detail="image file not found")
            return candidate
        raise HTTPException(status_code=400, detail="path is outside the allowed image roots")

    @app.post("/api/v1/images/analyze")
    def analyze(payload: dict):
        model = VisionCentroidModel.load(model_path)
        safe_path = resolve_safe_path(payload.get("path"))
        result = model.predict(extract_features(safe_path))
        event = {"path": str(safe_path), **result}
        results.append(event)
        return event

    @app.get("/api/v1/images/results")
    def get_results():
        return results

    @app.get("/api/v1/images/summary")
    def get_summary():
        return {"results": len(results), "defects": sum(1 for r in results if r["is_defect"])}

    return app
