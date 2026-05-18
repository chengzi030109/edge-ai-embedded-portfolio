from __future__ import annotations

from pathlib import Path

import pytest

from edge_audio.alarm import AlarmDebouncer
from edge_audio.api import ROUTES, create_app
from edge_audio.backends import load_backend
from edge_audio.config import load_config
from edge_audio.datasets import load_public_audio_rows, split_train_eval_rows, summarize_rows
from edge_audio.deployment import compare_backends, render_deployment_markdown
from edge_audio.evaluation import collect_labeled_window_vectors, evaluate_predictions, render_public_dataset_markdown, roc_auc
from edge_audio.features import load_feature_rows, read_wav
from edge_audio.model import AudioCentroidModel
from edge_audio.report import render_markdown
from edge_audio.storage import connect, init_db, insert_events, list_events, mark_events_uploaded, summary
from edge_audio.streaming import analyze_dataset_windows, analyze_wav_windows, collect_window_feature_vectors
from edge_audio.synth import generate_demo_wavs, generate_public_audio_sample


def test_config_loads():
    cfg = load_config("configs/default.toml")
    assert cfg.sample_rate_hz == 16000
    assert cfg.alarm_on_count == 3
    assert cfg.alarm_off_count == 5
    assert cfg.model_backend == "centroid"
    assert cfg.onnx_model_path.name == "audio_model.onnx"


def test_alarm_debouncer_requires_consecutive_windows():
    debouncer = AlarmDebouncer(on_count=3, off_count=2)

    # Two bad windows are only a pending condition; the third bad window turns
    # into a device alarm. This protects the service from one-window spikes.
    assert debouncer.update(True)["alarm_state"] == "pending"
    assert debouncer.update(True)["is_alarm"] is False
    entered = debouncer.update(True)
    assert entered["is_alarm"] is True
    assert entered["alarm_state"] == "alarm"

    # A single good window is not enough to clear the alarm. The second
    # consecutive good window returns the state machine to normal.
    assert debouncer.update(False)["alarm_state"] == "recovering"
    cleared = debouncer.update(False)
    assert cleared["is_alarm"] is False
    assert cleared["alarm_state"] == "normal"


def test_synthetic_wavs_and_features(tmp_path):
    generate_demo_wavs(tmp_path, sample_rate_hz=8000, clip_seconds=0.2)
    rows = load_feature_rows(tmp_path)
    samples, sr = read_wav(rows[0]["path"])
    assert len(rows) == 13
    assert sr == 8000
    assert samples.ndim == 1
    assert rows[0]["features"].shape[0] == 9


def test_public_audio_dataset_loader_supports_mimii_shape(tmp_path):
    generate_public_audio_sample(tmp_path, sample_rate_hz=8000, clip_seconds=0.2)
    rows = load_public_audio_rows(tmp_path)
    train_rows, eval_rows = split_train_eval_rows(rows)
    summary = summarize_rows(rows)

    assert summary["labels"]["normal"] > 0
    assert summary["labels"]["anomaly"] > 0
    assert all(row["label"] == "normal" for row in train_rows)
    assert any(row["label"] == "anomaly" for row in eval_rows)
    assert {row["split"] for row in rows} == {"train", "test"}


def test_public_audio_evaluation_report(tmp_path):
    generate_public_audio_sample(tmp_path, sample_rate_hz=8000, clip_seconds=0.3)
    rows = load_public_audio_rows(tmp_path)
    train_rows, eval_rows = split_train_eval_rows(rows)
    train_windows = collect_labeled_window_vectors(train_rows, 0.1, 0.05)
    eval_windows = collect_labeled_window_vectors(eval_rows, 0.1, 0.05)
    model = AudioCentroidModel.train([row["features"] for row in train_windows])
    result = evaluate_predictions(eval_windows, model)
    result.update(
        {
            "dataset_root": str(tmp_path),
            "dataset_summary": summarize_rows(rows),
            "train_clip_count": len(train_rows),
            "eval_clip_count": len(eval_rows),
            "train_window_count": len(train_windows),
            "eval_window_count": len(eval_windows),
        }
    )
    markdown = render_public_dataset_markdown(result)

    assert result["metrics"]["roc_auc"] is not None
    assert result["metrics"]["tp"] + result["metrics"]["tn"] + result["metrics"]["fp"] + result["metrics"]["fn"] == len(eval_windows)
    assert "Public Audio Dataset Evaluation" in markdown


def test_roc_auc_rank_statistic_handles_ties_and_single_class():
    assert roc_auc([0, 1], [0.2, 0.9]) == 1.0
    assert roc_auc([0, 1], [0.5, 0.5]) == 0.5
    assert roc_auc([0, 0], [0.1, 0.2]) is None


def test_model_detects_synthetic_anomalies(tmp_path):
    generate_demo_wavs(tmp_path, sample_rate_hz=8000, clip_seconds=0.2)
    rows = load_feature_rows(tmp_path)
    model = AudioCentroidModel.train([r["features"] for r in rows if r["label"] == "normal"])
    results = [{"path": r["path"], "label": r["label"], **model.predict(r["features"])} for r in rows]
    assert any(r["is_anomaly"] for r in results if r["label"] == "anomaly")
    assert "Audio Anomaly" in render_markdown(results, tmp_path / "curve.png")


def test_backend_loader_uses_centroid_contract(tmp_path):
    generate_demo_wavs(tmp_path, sample_rate_hz=8000, clip_seconds=0.2)
    rows = load_feature_rows(tmp_path)
    model_path = tmp_path / "audio_model.json"
    model = AudioCentroidModel.train([r["features"] for r in rows if r["label"] == "normal"])
    model.save(model_path)

    backend = load_backend("centroid", centroid_model_path=model_path, onnx_model_path=tmp_path / "missing.onnx")
    result = backend.predict(rows[0]["features"])
    assert backend.backend_name == "centroid"
    assert backend.model_path.endswith("audio_model.json")
    assert {"score", "threshold", "is_anomaly"}.issubset(result)


def test_onnx_backend_missing_dependency_or_file_is_clear(tmp_path):
    model_path = tmp_path / "audio_model.json"
    AudioCentroidModel.train([read_wav(generate_demo_wavs(tmp_path, 8000, 0.2)[0])[0][:9]]).save(model_path)
    try:
        load_backend("onnx", centroid_model_path=model_path, onnx_model_path=tmp_path / "missing.onnx")
    except (RuntimeError, FileNotFoundError) as exc:
        message = str(exc)
        assert "onnxruntime" in message or "ONNX model not found" in message
    else:  # pragma: no cover - only possible if a stale file unexpectedly exists
        raise AssertionError("missing ONNX backend should not load successfully")


def test_deployment_report_handles_missing_onnx(tmp_path):
    generate_demo_wavs(tmp_path / "wav", sample_rate_hz=8000, clip_seconds=0.2)
    rows = load_feature_rows(tmp_path / "wav")
    vectors = [r["features"] for r in rows[:3]]
    model_path = tmp_path / "audio_model.json"
    AudioCentroidModel.train([r["features"] for r in rows if r["label"] == "normal"]).save(model_path)
    report = compare_backends(vectors, centroid_model_path=model_path, onnx_model_path=tmp_path / "missing.onnx")
    markdown = render_deployment_markdown(report)
    assert report["centroid"]["backend"] == "centroid"
    assert report["onnx_status"].startswith("skipped")
    assert "Model Deployment Report" in markdown


def test_windowed_analysis_and_storage(tmp_path):
    generate_demo_wavs(tmp_path, sample_rate_hz=8000, clip_seconds=0.5)
    rows = load_feature_rows(tmp_path)
    # Train and infer on the same window size so the test exercises the
    # production contract used by the demo service.
    model = AudioCentroidModel.train(collect_window_feature_vectors(rows, 0.1, 0.05))
    events = analyze_wav_windows(rows[0]["path"], rows[0]["label"], model, 0.1, 0.05)
    assert len(events) > 1
    assert {"feature_ms", "inference_ms", "start_s", "end_s", "is_anomaly_raw", "is_alarm", "alarm_state"}.issubset(events[0])

    conn = connect(tmp_path / "audio.db")
    init_db(conn)
    insert_events(conn, events)
    assert summary(conn)["event_count"] == len(events)
    assert "alarm_count" in summary(conn)
    stored = list_events(conn, limit=1)[0]
    assert isinstance(stored["is_alarm"], bool)
    assert isinstance(stored["features"], dict)
    assert stored["uploaded"] is False
    assert stored["ack"] is False
    assert mark_events_uploaded(conn, [stored["id"]], ack=True) == 1
    updated = list_events(conn, limit=1)[0]
    assert updated["uploaded"] is True
    assert updated["ack"] is True


def test_dataset_window_analysis_saves_anomaly_clips(tmp_path):
    generate_demo_wavs(tmp_path / "wav", sample_rate_hz=8000, clip_seconds=0.5)
    rows = load_feature_rows(tmp_path / "wav")
    model = AudioCentroidModel.train(collect_window_feature_vectors(rows, 0.1, 0.05))
    events = analyze_dataset_windows(
        rows,
        model,
        0.1,
        0.05,
        clips_dir=tmp_path / "clips",
        save_anomaly_clips=True,
    )
    assert any(e["is_anomaly"] for e in events)
    assert any(e["clip_path"] for e in events if e["is_anomaly"])


def test_api_route_contract():
    assert "GET /" in ROUTES
    assert "GET /dashboard" in ROUTES
    assert "POST /api/v1/audio/analyze" in ROUTES
    assert "POST /api/v1/audio/analyze-windowed" in ROUTES
    assert "POST /api/v1/audio/upload" in ROUTES
    assert "POST /api/v1/audio/events/ack" in ROUTES
    assert "GET /healthz" in ROUTES
    assert "GET /metrics" in ROUTES


def test_api_health_metrics_and_upload(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    wav_root = tmp_path / "wav"
    generate_demo_wavs(wav_root, sample_rate_hz=8000, clip_seconds=0.5)
    rows = load_feature_rows(wav_root)
    model_path = tmp_path / "audio_model.json"
    model = AudioCentroidModel.train(collect_window_feature_vectors(rows, 0.1, 0.05))
    model.save(model_path)
    app = create_app(model_path=model_path, database_path=tmp_path / "audio.db")
    client = TestClient(app)

    health = client.get("/healthz")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    analyze = client.post("/api/v1/audio/analyze", json={"path": rows[0]["path"], "label": rows[0]["label"]})
    assert analyze.status_code == 200
    assert analyze.json()["source"] == str(Path(rows[0]["path"]).resolve())

    outside_file = tmp_path.parent / f"{tmp_path.name}_outside.wav"
    outside_file.write_bytes(Path(rows[0]["path"]).read_bytes())
    try:
        blocked = client.post("/api/v1/audio/analyze-windowed", json={"path": str(outside_file)})
        assert blocked.status_code == 400
        assert "outside the allowed audio roots" in blocked.json()["detail"]
    finally:
        outside_file.unlink(missing_ok=True)

    dashboard = client.get("/dashboard")
    assert dashboard.status_code == 200
    assert "Edge Audio Anomaly Dashboard" in dashboard.text
    assert "/api/v1/audio/upload" in dashboard.text
    assert "/metrics" in dashboard.text

    wav_path = rows[0]["path"]
    with open(wav_path, "rb") as fh:
        uploaded = client.post(
            "/api/v1/audio/upload",
            files={"file": ("sample.wav", fh, "audio/wav")},
            data={"label": "normal", "window_seconds": "0.1", "hop_seconds": "0.05"},
        )
    assert uploaded.status_code == 200
    payload = uploaded.json()
    assert payload["count"] > 0

    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    assert "edge_audio_events_total" in metrics.text

    event_id = client.get("/api/v1/audio/events?limit=1").json()[0]["id"]
    ack = client.post("/api/v1/audio/events/ack", json={"event_ids": [event_id], "ack": True})
    assert ack.status_code == 200
    assert ack.json()["updated"] == 1
