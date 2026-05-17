"""Tests for sklearn baseline wrappers.

The baselines must satisfy the same minimal predict() contract as the centroid
detector so the comparison script does not need per-detector branches.
"""

import numpy as np
import pytest

# sklearn imports joblib, which imports asyncio on recent versions. The user's
# local Windows Python has a broken asyncio/_overlapped stack, so skip these
# tests locally when sklearn cannot be imported cleanly. CI Linux still runs
# them and catches real baseline regressions.
try:
    import sklearn  # noqa: F401
except Exception as exc:  # pragma: no cover - depends on local Python install
    pytest.skip(f"sklearn is not usable in this environment: {exc}", allow_module_level=True)

from tpm.baselines import (
    IsolationForestDetector,
    LocalOutlierFactorDetector,
    OneClassSvmDetector,
    build_baselines,
)


def _toy_dataset(rng_seed: int = 0):
    rng = np.random.default_rng(rng_seed)
    # Tight cluster around the origin = "normal". 256 points is enough for
    # IsolationForest / LOF to estimate a stable threshold; smaller sizes give
    # noisy training-quantile thresholds that can mask obvious anomalies.
    X_train = rng.normal(0.0, 0.1, size=(256, 4)).astype(np.float32)
    far = rng.normal(8.0, 0.1, size=(8, 4)).astype(np.float32)
    return X_train, far


def test_each_baseline_separates_far_points():
    """Each baseline should call far points anomalous after fitting on tight cluster."""

    X_train, far = _toy_dataset()

    for detector in [
        IsolationForestDetector(),
        OneClassSvmDetector(),
        LocalOutlierFactorDetector(n_neighbors=10),
    ]:
        detector.fit(X_train)
        result = detector.predict(far[0])
        assert "score" in result
        assert "threshold" in result
        assert "is_anomaly" in result
        assert result["is_anomaly"] is True


def test_build_baselines_returns_three():
    """The default baseline set is the one the comparison report assumes."""

    baselines = build_baselines()
    names = {b.name for b in baselines}
    assert names == {"IsolationForest", "OneClassSVM", "LocalOutlierFactor"}
