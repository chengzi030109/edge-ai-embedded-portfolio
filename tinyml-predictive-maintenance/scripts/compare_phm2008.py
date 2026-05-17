from __future__ import annotations

"""Compare anomaly detectors on PHM2008/C-MAPSS degradation windows."""

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from tpm.datasets.phm2008 import load_phm_windows
from tpm.evaluation import metrics_from_counts, percentile
from tpm.model import CentroidAnomalyDetector


def _evaluate(detector, X: np.ndarray, y: np.ndarray) -> dict:
    """Run one-window-at-a-time predictions and compute metrics."""

    pred = []
    scores = []
    latencies = []
    for vector in X:
        start = time.perf_counter()
        result = detector.predict(vector)
        latencies.append((time.perf_counter() - start) * 1000)
        pred.append(bool(result["is_anomaly"]))
        scores.append(float(result["score"]))
    pred_arr = np.asarray(pred, dtype=bool)
    counts = {
        "tp": int(np.sum(y & pred_arr)),
        "tn": int(np.sum(~y & ~pred_arr)),
        "fp": int(np.sum(~y & pred_arr)),
        "fn": int(np.sum(y & ~pred_arr)),
    }
    return {
        "metrics": metrics_from_counts(counts),
        "confusion": counts,
        "latency_ms": {"avg": float(np.mean(latencies)), "p95": percentile(latencies, 0.95)},
        "score": {"mean": float(np.mean(scores)), "min": float(np.min(scores)), "max": float(np.max(scores))},
        "threshold": float(detector.threshold),
    }


def _size_bytes(detector) -> int:
    """Approximate model footprint for report tables."""

    if isinstance(detector, CentroidAnomalyDetector):
        return int(detector.mean.nbytes + detector.scale.nbytes + 4)
    return len(pickle.dumps(detector))


def _render_markdown(report: dict) -> str:
    rows = [
        "# PHM2008 / C-MAPSS Degradation Comparison",
        "",
        f"- Source file: `{report['source_file']}`",
        f"- Window size: `{report['window_size']}` cycles",
        f"- Hop: `{report['hop']}` cycles",
        f"- Features: `{report['n_features']}`",
        f"- Train windows: `{report['n_train']}`  Test windows: `{report['n_test']}`",
        "",
        "PHM2008 is harder than CWRU for this project because the signal is",
        "multivariate, multi-unit, and gradually degrading rather than a seeded",
        "bearing fault with a large spectral separation.",
        "",
        "| Model | Accuracy | Precision | Recall | F1 | FPR | Avg latency (ms) | Size (bytes) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for entry in report["models"]:
        m = entry["metrics"]
        rows.append(
            f"| {entry['name']} | {m['accuracy']:.4f} | {m['precision']:.4f} | "
            f"{m['recall']:.4f} | {m['f1']:.4f} | {m['false_positive_rate']:.4f} | "
            f"{entry['latency_ms']['avg']:.4f} | {entry['size_bytes']} |"
        )
    rows.append("")
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare detectors on PHM2008/C-MAPSS windows.")
    parser.add_argument("--data-root", default="data/phm2008_sample")
    parser.add_argument("--file", default="train_FD001.txt")
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--hop", type=int, default=10)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-out", default="reports/phm2008_comparison.json")
    parser.add_argument("--md-out", default="reports/phm2008_comparison.md")
    args = parser.parse_args()

    path = Path(args.data_root) / args.file
    windows = load_phm_windows(path, window_size=args.window_size, hop=args.hop)
    X = windows.features
    y = windows.labels
    rng = np.random.default_rng(args.seed)
    normal_idx = np.where(~y)[0]
    anomaly_idx = np.where(y)[0]
    normal_perm = rng.permutation(normal_idx)
    train_cut = max(1, int(len(normal_perm) * args.train_ratio))
    train_idx = normal_perm[:train_cut]
    normal_test_idx = normal_perm[train_cut:]
    test_idx = np.concatenate([normal_test_idx, anomaly_idx])
    X_train = X[train_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]

    centroid = CentroidAnomalyDetector.train(list(X_train), list(windows.feature_names), quantile=0.95, margin=1.05)
    detectors: list[tuple[str, object]] = [("CentroidAnomalyDetector", centroid)]
    try:
        from tpm.baselines import build_baselines

        for baseline in build_baselines():
            baseline.fit(X_train)
            detectors.append((baseline.name, baseline))
    except Exception as exc:
        print(f"warning: skipping sklearn baselines because they are unavailable: {exc}")

    models = []
    for name, detector in detectors:
        result = _evaluate(detector, X_test, y_test)
        size = _size_bytes(detector)
        print(
            f"{name:<22} acc={result['metrics']['accuracy']:.4f} "
            f"f1={result['metrics']['f1']:.4f} size={size}B"
        )
        models.append({"name": name, "size_bytes": size, **result})

    report = {
        "source": "phm2008-cmapss",
        "source_file": str(path),
        "window_size": args.window_size,
        "hop": args.hop,
        "n_features": int(X.shape[1]),
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "n_anomaly": int(np.sum(y_test)),
        "models": models,
    }
    Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json_out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    Path(args.md_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.md_out).write_text(_render_markdown(report), encoding="utf-8")
    print(f"saved JSON: {args.json_out}")
    print(f"saved Markdown: {args.md_out}")


if __name__ == "__main__":
    main()
