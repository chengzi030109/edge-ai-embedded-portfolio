"""Tests for configuration and evaluation reporting."""

import json

from tpm.config import load_config
from tpm.evaluation import evaluate_model, render_evaluation_markdown
from tpm.features import FEATURE_NAMES, FeatureConfig, extract_features, vectorize
from tpm.model import CentroidAnomalyDetector
from tpm.signal_sim import MotorSignalSimulator, SignalConfig


def test_load_config_from_json(tmp_path):
    """Config values should load from JSON and preserve state ordering."""

    path = tmp_path / "config.json"
    path.write_text(
        json.dumps({"sample_rate_hz": 800, "window_size": 128, "states": ["normal", "bearing"]}),
        encoding="utf-8",
    )

    cfg = load_config(path)

    assert cfg.sample_rate_hz == 800
    assert cfg.window_size == 128
    assert cfg.states == ("normal", "bearing")


def test_evaluation_report_contains_metrics():
    """Evaluation should produce JSON-ready metrics and Markdown text."""

    simulator = MotorSignalSimulator(SignalConfig(sample_rate_hz=1600))
    feature_cfg = FeatureConfig(sample_rate_hz=1600)
    vectors = []
    for _ in range(40):
        vectors.append(vectorize(extract_features(simulator.read(128, "normal"), feature_cfg)))
    model = CentroidAnomalyDetector.train(vectors, FEATURE_NAMES)

    report = evaluate_model(
        model=model,
        states=("normal", "imbalance"),
        windows_per_state=8,
        sample_rate_hz=1600,
        window_size=128,
    )
    markdown = render_evaluation_markdown(report)

    assert "accuracy" in report["metrics"]
    assert "normal" in report["per_state"]
    assert "Evaluation Report" in markdown
