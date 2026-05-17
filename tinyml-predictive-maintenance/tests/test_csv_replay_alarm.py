"""Tests for CSV replay and alarm debounce behavior."""

import csv
import json

import numpy as np
import pytest

from tpm.alarm import AlarmDebouncer
from tpm.datasets.csv_replay import load_csv_signal
from tpm.features import FEATURE_NAMES, FeatureConfig, extract_features, vectorize
from tpm.model import CentroidAnomalyDetector
from tpm.rtos_sim import NodeConfig, run_node
from tpm.signal_sim import MotorSignalSimulator, SignalConfig


def _write_csv(path, rows, fieldnames=("timestamp", "signal", "label")):
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_csv_loader_with_optional_label_and_timestamp(tmp_path):
    path = tmp_path / "signal.csv"
    _write_csv(
        path,
        [
            {"timestamp": 0.0, "signal": 0.1, "label": "normal"},
            {"timestamp": 0.1, "signal": 0.2, "label": "normal"},
            {"timestamp": 0.2, "signal": 0.9, "label": "fault"},
            {"timestamp": 0.3, "signal": 1.0, "label": "fault"},
        ],
    )

    signal = load_csv_signal(path)
    windows = list(signal.windows(window_size=2))

    assert len(windows) == 2
    assert windows[0].label == "normal"
    assert windows[1].label == "fault"
    assert windows[0].timestamp == pytest.approx(0.0)


def test_csv_loader_without_label_defaults_to_unknown(tmp_path):
    path = tmp_path / "signal.csv"
    _write_csv(path, [{"signal": 0.1}, {"signal": 0.2}], fieldnames=("signal",))

    signal = load_csv_signal(path)
    window = next(signal.windows(window_size=2))

    assert window.label == "unknown"


def test_csv_loader_rejects_missing_signal(tmp_path):
    path = tmp_path / "bad.csv"
    _write_csv(path, [{"value": 0.1}], fieldnames=("value",))

    with pytest.raises(ValueError, match="signal"):
        load_csv_signal(path)


def test_csv_short_sequence_yields_no_windows(tmp_path):
    path = tmp_path / "short.csv"
    _write_csv(path, [{"signal": 0.1}, {"signal": 0.2}], fieldnames=("signal",))

    signal = load_csv_signal(path)

    assert list(signal.windows(window_size=4)) == []


def test_alarm_debouncer_requires_consecutive_windows():
    debouncer = AlarmDebouncer(on_count=3, off_count=2)

    outputs = [debouncer.update(x).is_alarm for x in [True, True, False, True, True, True]]
    assert outputs == [False, False, False, False, False, True]

    recovery = [debouncer.update(x).is_alarm for x in [False, True, False, False]]
    assert recovery == [True, True, True, False]


def test_run_node_csv_source_writes_debounced_telemetry(tmp_path):
    feature_cfg = FeatureConfig(sample_rate_hz=1600)
    sim = MotorSignalSimulator(SignalConfig(sample_rate_hz=1600))
    vectors = [vectorize(extract_features(sim.read(8, "normal"), feature_cfg)) for _ in range(20)]
    model_path = tmp_path / "model.json"
    CentroidAnomalyDetector.train(vectors, FEATURE_NAMES).save(model_path)

    csv_path = tmp_path / "replay.csv"
    rows = []
    for i, value in enumerate(np.linspace(-0.2, 0.2, 24)):
        rows.append({"timestamp": i * 0.01, "signal": float(value), "label": "normal"})
    _write_csv(csv_path, rows)

    telemetry_path = tmp_path / "telemetry.jsonl"
    run_node(
        model_path=model_path,
        telemetry_path=telemetry_path,
        config=NodeConfig(
            sample_rate_hz=1600,
            window_size=8,
            source="csv",
            input_path=str(csv_path),
            alarm_on_count=2,
            alarm_off_count=2,
        ),
    )

    first = json.loads(telemetry_path.read_text(encoding="utf-8").splitlines()[0])
    assert first["true_state"] == "normal"
    assert {"is_anomaly_raw", "is_alarm", "alarm_state"}.issubset(first)
