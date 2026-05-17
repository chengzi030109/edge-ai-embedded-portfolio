"""Sklearn baselines for the anomaly detector comparison.

The project's own ``CentroidAnomalyDetector`` is interpretable and tiny but it
is also the simplest possible normal-only detector. Putting it next to common
sklearn baselines is the honest way to answer the interview question
"why did you pick this model?".

All baselines expose the same minimal interface as the centroid detector:

    detector.fit(X_normal)
    detector.predict(vector) -> {"score", "threshold", "is_anomaly"}

That keeps ``scripts/compare_models.py`` independent of which detector it is
evaluating. Each wrapper picks a threshold from the high quantile of training
scores (with a small safety margin) so results are comparable to the centroid
detector, which uses the same rule.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


class AnomalyDetector(Protocol):
    """Common minimal contract for the comparison script."""

    threshold: float

    def fit(self, X: np.ndarray) -> "AnomalyDetector": ...
    def score(self, vector: np.ndarray) -> float: ...
    def predict(self, vector: np.ndarray) -> dict: ...


def _calibrate_threshold(scores: np.ndarray, quantile: float, margin: float) -> float:
    """Match the centroid detector's threshold rule for fair comparison.

    The centroid detector uses ``quantile(scores) * margin`` because its scores
    are non-negative L2 distances. Some sklearn detectors (notably LOF in
    novelty mode) can produce negative training scores, which would make a
    multiplicative margin tighten the threshold instead of loosening it. We
    add ``margin - 1`` times the score range as the slack instead, which
    behaves the same as ``* margin`` for the non-negative case but stays
    monotone everywhere.
    """

    q = float(np.quantile(scores, quantile))
    span = float(np.max(scores) - np.min(scores))
    return q + (margin - 1.0) * max(abs(q), span)


@dataclass
class IsolationForestDetector:
    """Wrapper around ``sklearn.ensemble.IsolationForest``.

    Uses the negated anomaly score so larger numbers mean more anomalous,
    matching the convention used by the centroid detector and reports.
    """

    n_estimators: int = 100
    quantile: float = 0.995
    margin: float = 1.15
    random_state: int = 0
    threshold: float = 0.0
    _model: object = field(default=None, repr=False)

    name: str = "IsolationForest"

    def fit(self, X: np.ndarray) -> "IsolationForestDetector":
        from sklearn.ensemble import IsolationForest

        # ``contamination='auto'`` is fine since we calibrate our own threshold.
        self._model = IsolationForest(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
        ).fit(X)
        train_scores = -self._model.score_samples(X)
        self.threshold = _calibrate_threshold(train_scores, self.quantile, self.margin)
        return self

    def score(self, vector: np.ndarray) -> float:
        v = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        return float(-self._model.score_samples(v)[0])

    def predict(self, vector: np.ndarray) -> dict:
        s = self.score(vector)
        return {"score": s, "threshold": self.threshold, "is_anomaly": bool(s > self.threshold)}


@dataclass
class OneClassSvmDetector:
    """Wrapper around ``sklearn.svm.OneClassSVM`` with an RBF kernel."""

    nu: float = 0.05
    gamma: str | float = "scale"
    quantile: float = 0.995
    margin: float = 1.15
    threshold: float = 0.0
    _model: object = field(default=None, repr=False)

    name: str = "OneClassSVM"

    def fit(self, X: np.ndarray) -> "OneClassSvmDetector":
        from sklearn.svm import OneClassSVM

        self._model = OneClassSVM(nu=self.nu, gamma=self.gamma).fit(X)
        # decision_function: positive inside the boundary. Negate so anomalies
        # have larger scores like the other detectors.
        train_scores = -self._model.decision_function(X)
        self.threshold = _calibrate_threshold(train_scores, self.quantile, self.margin)
        return self

    def score(self, vector: np.ndarray) -> float:
        v = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        return float(-self._model.decision_function(v)[0])

    def predict(self, vector: np.ndarray) -> dict:
        s = self.score(vector)
        return {"score": s, "threshold": self.threshold, "is_anomaly": bool(s > self.threshold)}


@dataclass
class LocalOutlierFactorDetector:
    """Wrapper around ``sklearn.neighbors.LocalOutlierFactor``.

    LOF in novelty mode does not refit on every score call, which is what we
    need for streaming-style evaluation.
    """

    n_neighbors: int = 20
    quantile: float = 0.995
    margin: float = 1.15
    threshold: float = 0.0
    _model: object = field(default=None, repr=False)

    name: str = "LocalOutlierFactor"

    def fit(self, X: np.ndarray) -> "LocalOutlierFactorDetector":
        from sklearn.neighbors import LocalOutlierFactor

        # ``novelty=True`` is the right setting for "fit on normal, score new".
        self._model = LocalOutlierFactor(
            n_neighbors=min(self.n_neighbors, max(2, len(X) - 1)),
            novelty=True,
        ).fit(X)
        train_scores = -self._model.decision_function(X)
        self.threshold = _calibrate_threshold(train_scores, self.quantile, self.margin)
        return self

    def score(self, vector: np.ndarray) -> float:
        v = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        return float(-self._model.decision_function(v)[0])

    def predict(self, vector: np.ndarray) -> dict:
        s = self.score(vector)
        return {"score": s, "threshold": self.threshold, "is_anomaly": bool(s > self.threshold)}


def build_baselines() -> list[AnomalyDetector]:
    """Default baseline set used by ``scripts/compare_models.py``.

    Adding/removing detectors here is the only place the comparison script
    needs to learn about new algorithms.
    """

    return [
        IsolationForestDetector(),
        OneClassSvmDetector(),
        LocalOutlierFactorDetector(),
    ]
