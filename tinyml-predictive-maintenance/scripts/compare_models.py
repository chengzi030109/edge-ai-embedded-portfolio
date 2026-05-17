"""Compare anomaly detectors on the CWRU bearing dataset.

This is the script behind the "model selection" story in the README:

    Why centroid? Show its numbers next to IsolationForest, OneClassSVM, and
    LocalOutlierFactor on the same features and the same data, then talk
    about the tradeoffs.

The script does not download anything by itself. ``scripts/prepare_cwru.py``
populates ``data/cwru/`` first; this script consumes whatever it finds.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from tpm.datasets.cwru import CWRU_SAMPLE_RATE_HZ, load_all
from tpm.evaluation import metrics_from_counts, percentile
from tpm.features import FEATURE_NAMES, FeatureConfig, extract_features, vectorize
from tpm.model import CentroidAnomalyDetector


def _features_matrix(windows: np.ndarray, feature_cfg: FeatureConfig) -> np.ndarray:
    """Apply the project's feature extractor to every window."""

    out = np.empty((windows.shape[0], len(FEATURE_NAMES)), dtype=np.float32)
    for i, window in enumerate(windows):
        out[i] = vectorize(extract_features(window, feature_cfg))
    return out


def _evaluate_detector(detector, X_test: np.ndarray, y_true: np.ndarray) -> dict:
    """Run streaming-style predictions and gather metrics + latency."""

    y_pred = np.empty(len(X_test), dtype=bool)
    scores = np.empty(len(X_test), dtype=np.float32)
    latencies_ms: list[float] = []
    for i, vector in enumerate(X_test):
        # Per-sample timing reflects how this detector would behave at the edge,
        # one window at a time, instead of in a vectorized batch.
        start = time.perf_counter()
        result = detector.predict(vector)
        latencies_ms.append((time.perf_counter() - start) * 1000)
        y_pred[i] = result["is_anomaly"]
        scores[i] = result["score"]

    counts = {
        "tp": int(np.sum(y_true & y_pred)),
        "tn": int(np.sum(~y_true & ~y_pred)),
        "fp": int(np.sum(~y_true & y_pred)),
        "fn": int(np.sum(y_true & ~y_pred)),
    }
    return {
        "metrics": metrics_from_counts(counts),
        "confusion": counts,
        "threshold": float(detector.threshold),
        "latency_ms": {
            "avg": float(np.mean(latencies_ms)),
            "p95": percentile(latencies_ms, 0.95),
        },
        "score": {
            "mean": float(np.mean(scores)),
            "max": float(np.max(scores)),
            "min": float(np.min(scores)),
        },
    }


def _model_size_bytes(detector) -> int:
    """Approximate footprint for the comparison table.

    The centroid detector serializes to JSON. Sklearn detectors do not have a
    portable JSON form, so we use pickle byte length as a comparable proxy.
    The autoencoder reports its state_dict size (what would go into flash).
    """

    if isinstance(detector, CentroidAnomalyDetector):
        payload = {
            "feature_names": detector.feature_names,
            "mean": detector.mean.tolist(),
            "scale": detector.scale.tolist(),
            "threshold": detector.threshold,
        }
        return len(json.dumps(payload).encode("utf-8"))
    if detector.__class__.__name__ == "AutoencoderDetector":
        return detector.model_size_bytes()
    return len(pickle.dumps(detector))


def _split_train_test(
    X_normal: np.ndarray,
    train_ratio: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Random non-overlapping split for normal windows."""

    n = X_normal.shape[0]
    indices = rng.permutation(n)
    cut = int(n * train_ratio)
    return X_normal[indices[:cut]], X_normal[indices[cut:]]


def _render_markdown(report: dict) -> str:
    rows = [
        "# Model Comparison on CWRU",
        "",
        f"- Source: `{report['source']}`",
        f"- Window size: `{report['window_size']}` samples @ {report['sample_rate_hz']} Hz",
        f"- Train (normal) windows: `{report['n_train']}`",
        f"- Test windows: `{report['n_test']}` (normal={report['n_test_normal']}, faulty={report['n_test_fault']})",
        "",
        "## Detection Metrics",
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
    rows.append("## Per-Label Detection Rate")
    rows.append("")
    labels = report["labels"]
    header = "| Model | " + " | ".join(labels) + " |"
    align = "|---|" + "---:|" * len(labels)
    rows.extend([header, align])
    for entry in report["models"]:
        cells = [f"{entry['per_label'][label]:.4f}" for label in labels]
        rows.append(f"| {entry['name']} | " + " | ".join(cells) + " |")
    rows.append("")
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare anomaly detectors on CWRU windows.")
    parser.add_argument("--data-root", default="data/cwru")
    parser.add_argument("--window-size", type=int, default=1024)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-out", default="reports/cwru_comparison.json")
    parser.add_argument("--md-out", default="reports/cwru_comparison.md")
    parser.add_argument(
        "--max-windows-per-label",
        type=int,
        default=400,
        help="Cap windows per label to keep evaluation fast.",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    rng = np.random.default_rng(args.seed)

    print(f"loading CWRU windows from {data_root}/...")
    sets = load_all(data_root, window_size=args.window_size)
    feature_cfg = FeatureConfig(sample_rate_hz=CWRU_SAMPLE_RATE_HZ)

    # Cap each label so the comparison runs in seconds, even when CWRU files
    # are large. The cap is applied at the window level after loading, which
    # keeps the loader simple.
    capped: dict[str, np.ndarray] = {}
    for label, ws in sets.items():
        if ws.windows.shape[0] > args.max_windows_per_label:
            idx = rng.choice(ws.windows.shape[0], args.max_windows_per_label, replace=False)
            capped[label] = ws.windows[idx]
        else:
            capped[label] = ws.windows
        print(f"  {label}: {capped[label].shape[0]} windows ({', '.join(ws.source_files)})")

    # Feature matrix per label. The detectors only see features, never raw
    # windows, which keeps the comparison fair.
    print("extracting features...")
    feats = {label: _features_matrix(w, feature_cfg) for label, w in capped.items()}

    X_train, X_normal_test = _split_train_test(feats["normal"], args.train_ratio, rng)

    fault_labels = [label for label in feats if label != "normal"]
    X_fault_test = np.concatenate([feats[label] for label in fault_labels], axis=0)
    fault_owner = np.concatenate(
        [np.full(feats[label].shape[0], label) for label in fault_labels]
    )

    X_test = np.concatenate([X_normal_test, X_fault_test], axis=0)
    y_test = np.concatenate(
        [np.zeros(X_normal_test.shape[0], dtype=bool), np.ones(X_fault_test.shape[0], dtype=bool)]
    )
    test_owner = np.concatenate([np.full(X_normal_test.shape[0], "normal"), fault_owner])

    print(f"training detectors on {X_train.shape[0]} normal windows...")
    centroid = CentroidAnomalyDetector.train(list(X_train), FEATURE_NAMES)
    detectors: list[tuple[str, object]] = [("CentroidAnomalyDetector", centroid)]
    try:
        from tpm.baselines import build_baselines

        for baseline in build_baselines():
            baseline.fit(X_train)
            detectors.append((baseline.name, baseline))
    except Exception as exc:
        print(f"warning: skipping sklearn baselines because they are unavailable: {exc}")

    try:
        from tpm.autoencoder import AutoencoderDetector

        autoenc = AutoencoderDetector(input_dim=X_train.shape[1], epochs=150)
        autoenc.fit(X_train)
        detectors.append((autoenc.name, autoenc))
    except Exception as exc:
        print(f"warning: skipping autoencoder because torch/onnx stack is unavailable: {exc}")

    labels_in_order = ["normal"] + fault_labels
    report_models = []
    for name, detector in detectors:
        eval_result = _evaluate_detector(detector, X_test, y_test)
        # Per-label detection rate: for normal it is the false-positive rate;
        # for fault labels it is recall on that label alone.
        per_label: dict[str, float] = {}
        for label in labels_in_order:
            mask = test_owner == label
            if not mask.any():
                per_label[label] = 0.0
                continue
            y_pred_label = np.array(
                [detector.predict(X_test[i])["is_anomaly"] for i in np.where(mask)[0]]
            )
            per_label[label] = float(np.mean(y_pred_label))
        size_bytes = _model_size_bytes(detector)
        print(
            f"  {name:<22} acc={eval_result['metrics']['accuracy']:.4f} "
            f"f1={eval_result['metrics']['f1']:.4f} "
            f"latency={eval_result['latency_ms']['avg']:.3f}ms size={size_bytes}B"
        )
        report_models.append(
            {
                "name": name,
                "size_bytes": size_bytes,
                "per_label": per_label,
                **eval_result,
            }
        )

    is_synthetic = (data_root / "SYNTHETIC.txt").exists()
    report = {
        "source": "synthetic-cwru-shape" if is_synthetic else "cwru",
        "data_root": str(data_root),
        "sample_rate_hz": CWRU_SAMPLE_RATE_HZ,
        "window_size": args.window_size,
        "labels": labels_in_order,
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "n_test_normal": int(X_normal_test.shape[0]),
        "n_test_fault": int(X_fault_test.shape[0]),
        "models": report_models,
    }

    json_path = Path(args.json_out)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    md_path = Path(args.md_out)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(_render_markdown(report), encoding="utf-8")

    print(f"\nsaved JSON: {json_path}")
    print(f"saved Markdown: {md_path}")
    if is_synthetic:
        print("note: data is synthetic CWRU-shape — replace with real .mat files for credible numbers.")


if __name__ == "__main__":
    main()
