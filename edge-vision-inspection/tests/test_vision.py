from __future__ import annotations

from pathlib import Path

import pytest

from edge_vision.api import ROUTES, create_app
from edge_vision.config import load_config
from edge_vision.features import load_feature_rows
from edge_vision.model import VisionCentroidModel
from edge_vision.report import render_markdown
from edge_vision.synth import generate_demo_images


def test_config_loads():
    cfg = load_config("configs/default.toml")
    assert cfg.image_size == 160


def test_synthetic_images_and_features(tmp_path):
    generate_demo_images(tmp_path, image_size=64)
    rows = load_feature_rows(tmp_path)
    assert len(rows) == 16
    assert rows[0]["features"].shape[0] == 6


def test_model_detects_defects(tmp_path):
    generate_demo_images(tmp_path, image_size=64)
    rows = load_feature_rows(tmp_path)
    model = VisionCentroidModel.train([r["features"] for r in rows if r["label"] == "normal"])
    results = [{"path": r["path"], "label": r["label"], **model.predict(r["features"])} for r in rows]
    assert any(r["is_defect"] for r in results if r["label"] == "defect")
    assert "Vision Inspection" in render_markdown(results)


def test_api_route_contract():
    assert "POST /api/v1/images/analyze" in ROUTES


def test_api_blocks_paths_outside_safe_roots(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    image_root = tmp_path / "images"
    generate_demo_images(image_root, image_size=64)
    rows = load_feature_rows(image_root)
    model_path = tmp_path / "vision_model.json"
    model = VisionCentroidModel.train([r["features"] for r in rows if r["label"] == "normal"])
    model.save(model_path)

    app = create_app(model_path=model_path)
    client = TestClient(app)

    allowed = client.post("/api/v1/images/analyze", json={"path": rows[0]["path"]})
    assert allowed.status_code == 200
    assert allowed.json()["path"] == str(Path(rows[0]["path"]).resolve())

    outside_file = tmp_path.parent / f"{tmp_path.name}_outside.png"
    outside_file.write_bytes(Path(rows[0]["path"]).read_bytes())
    try:
        blocked = client.post("/api/v1/images/analyze", json={"path": str(outside_file)})
        assert blocked.status_code == 400
        assert "outside the allowed image roots" in blocked.json()["detail"]
    finally:
        outside_file.unlink(missing_ok=True)
