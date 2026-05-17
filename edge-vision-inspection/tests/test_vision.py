from __future__ import annotations

from edge_vision.api import ROUTES
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

