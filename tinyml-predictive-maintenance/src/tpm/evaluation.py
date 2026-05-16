from __future__ import annotations

"""Evaluation utilities for the predictive-maintenance model.

The runtime node answers the question "can the system run end to end?" This file
answers the interview question "how well does the detector work?" It generates
balanced synthetic test windows for each state, runs the model, and computes
binary anomaly-detection metrics without requiring scikit-learn.
"""

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

from .features import FeatureConfig, extract_features, vectorize
from .model import CentroidAnomalyDetector
from .signal_sim import MotorSignalSimulator, SignalConfig


def confusion_counts(y_true: Iterable[bool], y_pred: Iterable[bool]) -> dict[str, int]:
    """Compute binary confusion-matrix counts.

    ``True`` means anomaly. ``False`` means normal. Returning named counts keeps
    the JSON report readable and avoids depending on external ML metric libs.
    """

    counts = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    for truth, pred in zip(y_true, y_pred):
        if truth and pred:
            counts["tp"] += 1
        elif not truth and not pred:
            counts["tn"] += 1
        elif not truth and pred:
            counts["fp"] += 1
        else:
            counts["fn"] += 1
    return counts


def safe_div(numerator: float, denominator: float) -> float:
    """Divide while avoiding ZeroDivisionError in tiny test runs."""

    return float(numerator / denominator) if denominator else 0.0


def metrics_from_counts(counts: dict[str, int]) -> dict[str, float]:
    """Convert confusion counts to common detection metrics."""

    tp = counts["tp"]
    tn = counts["tn"]
    fp = counts["fp"]
    fn = counts["fn"]
    total = tp + tn + fp + fn
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    return {
        "accuracy": safe_div(tp + tn, total),
        "precision": precision,
        "recall": recall,
        "f1": safe_div(2 * precision * recall, precision + recall),
        "false_positive_rate": safe_div(fp, fp + tn),
        "false_negative_rate": safe_div(fn, fn + tp),
    }


def evaluate_model(
    model: CentroidAnomalyDetector,
    states: Iterable[str],
    windows_per_state: int,
    sample_rate_hz: int,
    window_size: int,
) -> dict:
    """Evaluate the detector on synthetic windows for each requested state.

    The simulator is reset for each state so every class gets a deterministic
    and balanced evaluation stream. This makes reports stable across runs.
    """

    feature_cfg = FeatureConfig(sample_rate_hz=sample_rate_hz)
    y_true: list[bool] = []
    y_pred: list[bool] = []
    scores: list[float] = []
    feature_ms: list[float] = []
    inference_ms: list[float] = []
    per_state: dict[str, dict[str, int | float]] = defaultdict(lambda: {"windows": 0, "detected": 0})

    for state_index, state in enumerate(states):
        # Use a different seed per state to avoid identical noise sequences while
        # remaining deterministic for reports and tests.
        simulator = MotorSignalSimulator(
            SignalConfig(sample_rate_hz=sample_rate_hz, seed=100 + state_index)
        )
        is_anomaly = state != "normal"

        for _ in range(windows_per_state):
            samples = simulator.read(window_size, state)

            feature_start = time.perf_counter()
            feature_map = extract_features(samples, feature_cfg)
            feature_ms.append((time.perf_counter() - feature_start) * 1000)

            inference_start = time.perf_counter()
            result = model.predict(vectorize(feature_map))
            inference_ms.append((time.perf_counter() - inference_start) * 1000)

            predicted_anomaly = bool(result["is_anomaly"])
            y_true.append(is_anomaly)
            y_pred.append(predicted_anomaly)
            scores.append(float(result["score"]))

            per_state[state]["windows"] = int(per_state[state]["windows"]) + 1
            if predicted_anomaly:
                per_state[state]["detected"] = int(per_state[state]["detected"]) + 1

    counts = confusion_counts(y_true, y_pred)
    summary = metrics_from_counts(counts)

    for state, state_stats in per_state.items():
        windows = int(state_stats["windows"])
        detected = int(state_stats["detected"])
        state_stats["detection_rate"] = safe_div(detected, windows)

    return {
        "sample_rate_hz": sample_rate_hz,
        "window_size": window_size,
        "windows_per_state": windows_per_state,
        "threshold": model.threshold,
        "confusion": counts,
        "metrics": summary,
        "per_state": dict(per_state),
        "score": {
            "mean": float(np.mean(scores)),
            "max": float(np.max(scores)),
            "min": float(np.min(scores)),
        },
        "latency_ms": {
            "feature_avg": float(np.mean(feature_ms)),
            "inference_avg": float(np.mean(inference_ms)),
            "feature_p95": percentile(feature_ms, 0.95),
            "inference_p95": percentile(inference_ms, 0.95),
        },
    }


def percentile(values: list[float], pct: float) -> float:
    """Return a percentile with linear interpolation."""

    if not values:
        return 0.0
    ordered = sorted(values)
    rank = (len(ordered) - 1) * pct
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    weight = rank - lo
    return float(ordered[lo] * (1 - weight) + ordered[hi] * weight)


def save_evaluation_json(report: dict, path: str | Path) -> None:
    """Save the machine-readable evaluation report."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")


def render_evaluation_markdown(report: dict) -> str:
    """Render evaluation metrics for README snippets or portfolio review."""

    metrics = report["metrics"]
    latency = report["latency_ms"]
    lines = [
        "# Evaluation Report",
        "",
        f"- Sample rate: `{report['sample_rate_hz']}` Hz",
        f"- Window size: `{report['window_size']}` samples",
        f"- Windows per state: `{report['windows_per_state']}`",
        f"- Threshold: `{report['threshold']:.4f}`",
        "",
        "## Binary Anomaly Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Accuracy | {metrics['accuracy']:.4f} |",
        f"| Precision | {metrics['precision']:.4f} |",
        f"| Recall | {metrics['recall']:.4f} |",
        f"| F1 | {metrics['f1']:.4f} |",
        f"| False positive rate | {metrics['false_positive_rate']:.4f} |",
        f"| False negative rate | {metrics['false_negative_rate']:.4f} |",
        "",
        "## Per-State Detection",
        "",
        "| State | Windows | Detected as anomaly | Detection rate |",
        "|---|---:|---:|---:|",
    ]

    for state, state_stats in report["per_state"].items():
        lines.append(
            f"| {state} | {state_stats['windows']} | {state_stats['detected']} | "
            f"{state_stats['detection_rate']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Latency",
            "",
            "| Stage | Avg ms | P95 ms |",
            "|---|---:|---:|",
            f"| Feature extraction | {latency['feature_avg']:.6f} | {latency['feature_p95']:.6f} |",
            f"| Inference | {latency['inference_avg']:.6f} | {latency['inference_p95']:.6f} |",
            "",
        ]
    )
    return "\n".join(lines)


def save_evaluation_markdown(report: dict, path: str | Path) -> None:
    """Save the human-readable evaluation report."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_evaluation_markdown(report), encoding="utf-8")
