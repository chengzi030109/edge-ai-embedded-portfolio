from __future__ import annotations

from edge_audio.api import ROUTES
from edge_audio.config import load_config
from edge_audio.features import load_feature_rows, read_wav
from edge_audio.model import AudioCentroidModel
from edge_audio.report import render_markdown
from edge_audio.synth import generate_demo_wavs


def test_config_loads():
    cfg = load_config("configs/default.toml")
    assert cfg.sample_rate_hz == 16000


def test_synthetic_wavs_and_features(tmp_path):
    generate_demo_wavs(tmp_path, sample_rate_hz=8000, clip_seconds=0.2)
    rows = load_feature_rows(tmp_path)
    samples, sr = read_wav(rows[0]["path"])
    assert len(rows) == 13
    assert sr == 8000
    assert samples.ndim == 1
    assert rows[0]["features"].shape[0] == 6


def test_model_detects_synthetic_anomalies(tmp_path):
    generate_demo_wavs(tmp_path, sample_rate_hz=8000, clip_seconds=0.2)
    rows = load_feature_rows(tmp_path)
    model = AudioCentroidModel.train([r["features"] for r in rows if r["label"] == "normal"])
    results = [{"path": r["path"], "label": r["label"], **model.predict(r["features"])} for r in rows]
    assert any(r["is_anomaly"] for r in results if r["label"] == "anomaly")
    assert "Audio Anomaly" in render_markdown(results, tmp_path / "curve.png")


def test_api_route_contract():
    assert "POST /api/v1/audio/analyze" in ROUTES

