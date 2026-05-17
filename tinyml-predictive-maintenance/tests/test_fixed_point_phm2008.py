"""Tests for fixed-point simulation and PHM2008 windowing."""

import numpy as np

from tpm.datasets.phm2008 import load_phm_windows, make_windows
from tpm.fixed_point import compare_fixed_point
from tpm.model import CentroidAnomalyDetector


def test_fixed_point_report_counts_mismatches():
    X = np.vstack(
        [
            np.zeros((8, 3), dtype=np.float32),
            np.ones((4, 3), dtype=np.float32) * 4.0,
        ]
    )
    model = CentroidAnomalyDetector.train([x for x in X[:8]], ["a", "b", "c"], quantile=0.9)

    report = compare_fixed_point(model, X)

    assert report["n_vectors"] == 12
    assert report["fixed_point_bytes"] < report["float_model_bytes"] + 32
    assert "decision_mismatches" in report


def test_phm2008_window_shapes_from_table():
    rows = []
    for unit in (1, 2):
        for cycle in range(1, 41):
            rows.append([unit, cycle, 0.0, 0.0, 100.0, cycle * 0.01, unit + cycle * 0.02])
    table = np.asarray(rows, dtype=np.float32)

    windows = make_windows(table, window_size=5, hop=5, healthy_fraction=0.3, degraded_fraction=0.7)

    assert windows.features.shape[1] == 12  # 2 sensors * 5 stats + 2 cycle features
    assert windows.labels.any()
    assert (~windows.labels).any()
    assert windows.feature_names[0] == "sensor_1_mean"


def test_phm2008_load_from_text(tmp_path):
    path = tmp_path / "train_FD001.txt"
    rows = []
    for cycle in range(1, 50):
        rows.append([1, cycle, 0.0, 0.0, 100.0, 1.0 + cycle * 0.01])
    np.savetxt(path, np.asarray(rows, dtype=np.float32), fmt="%.6f")

    windows = load_phm_windows(path, window_size=6, hop=6)

    assert windows.features.shape[0] > 0
    assert windows.features.shape[1] == 7
