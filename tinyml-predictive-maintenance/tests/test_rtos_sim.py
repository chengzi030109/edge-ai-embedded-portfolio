"""Tests for the RTOS-style node loop."""

import json

from tpm.features import FEATURE_NAMES, FeatureConfig, extract_features, vectorize
from tpm.model import CentroidAnomalyDetector
from tpm.rtos_sim import NodeConfig, run_node
from tpm.signal_sim import MotorSignalSimulator, SignalConfig


def test_run_node_writes_telemetry(tmp_path):
    """A short node run should create JSONL telemetry with inference fields."""

    simulator = MotorSignalSimulator(SignalConfig(sample_rate_hz=1600))
    feature_cfg = FeatureConfig(sample_rate_hz=1600)
    vectors = []
    for _ in range(20):
        vectors.append(vectorize(extract_features(simulator.read(128, "normal"), feature_cfg)))

    model_path = tmp_path / "model.json"
    telemetry_path = tmp_path / "telemetry.jsonl"
    CentroidAnomalyDetector.train(vectors, FEATURE_NAMES).save(model_path)

    run_node(
        model_path=model_path,
        telemetry_path=telemetry_path,
        config=NodeConfig(sample_rate_hz=1600, window_size=128, duration_s=0.6),
    )

    rows = [json.loads(line) for line in telemetry_path.read_text(encoding="utf-8").splitlines()]
    assert rows
    assert {"seq", "score", "threshold", "is_anomaly", "features"}.issubset(rows[0])
