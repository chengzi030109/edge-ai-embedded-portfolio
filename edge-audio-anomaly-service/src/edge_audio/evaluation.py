from __future__ import annotations

"""Evaluation utilities for real or real-shaped industrial audio datasets."""

import json
from pathlib import Path

from .features import extract_features, iter_windows, read_wav


def collect_labeled_window_vectors(rows: list[dict], window_seconds: float, hop_seconds: float) -> list[dict]:
    """Expand clip rows into labeled window feature rows.

    Model training and deployment both operate on windows, so evaluation should
    do the same. Each output row keeps the original clip label and path while
    adding a ``window_index`` and feature vector. This prevents an overly rosy
    clip-level score from hiding unstable per-window behavior.
    """

    windows: list[dict] = []
    for row in rows:
        samples, sample_rate_hz = read_wav(row["path"])
        for index, window in enumerate(iter_windows(samples, sample_rate_hz, window_seconds, hop_seconds)):
            windows.append(
                {
                    "path": row["path"],
                    "relative_path": row.get("relative_path", Path(row["path"]).name),
                    "label": row["label"],
                    "split": row.get("split", "all"),
                    "machine_id": row.get("machine_id", "default"),
                    "window_index": index,
                    "features": extract_features(window["samples"], sample_rate_hz),
                }
            )
    return windows


def evaluate_predictions(window_rows: list[dict], model) -> dict:
    """Score labeled windows and compute threshold metrics."""

    scored: list[dict] = []
    for row in window_rows:
        pred = model.predict(row["features"])
        scored.append(
            {
                "path": row["path"],
                "relative_path": row["relative_path"],
                "label": row["label"],
                "window_index": row["window_index"],
                "score": float(pred["score"]),
                "threshold": float(pred["threshold"]),
                "is_anomaly": bool(pred["is_anomaly"]),
            }
        )
    metrics = binary_metrics(scored)
    metrics["roc_auc"] = roc_auc(
        [1 if row["label"] == "anomaly" else 0 for row in scored],
        [row["score"] for row in scored],
    )
    return {"metrics": metrics, "scored_rows": scored}


def binary_metrics(scored_rows: list[dict]) -> dict:
    """Compute confusion matrix and F1 from the model threshold."""

    tp = sum(1 for row in scored_rows if row["label"] == "anomaly" and row["is_anomaly"])
    tn = sum(1 for row in scored_rows if row["label"] == "normal" and not row["is_anomaly"])
    fp = sum(1 for row in scored_rows if row["label"] == "normal" and row["is_anomaly"])
    fn = sum(1 for row in scored_rows if row["label"] == "anomaly" and not row["is_anomaly"])
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    accuracy = (tp + tn) / max(1, tp + tn + fp + fn)
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


def roc_auc(labels: list[int], scores: list[float]) -> float | None:
    """Compute ROC-AUC without sklearn using rank statistics.

    AUC is the probability that a random anomaly has a higher score than a
    random normal window. Ties receive half credit. ``None`` means the dataset
    contains only one class, which can happen when a user points at a training
    folder instead of a train/test root.
    """

    positives = [(score, idx) for idx, (label, score) in enumerate(zip(labels, scores, strict=False)) if label == 1]
    negatives = [(score, idx) for idx, (label, score) in enumerate(zip(labels, scores, strict=False)) if label == 0]
    if not positives or not negatives:
        return None

    wins = 0.0
    for pos_score, _ in positives:
        for neg_score, _ in negatives:
            if pos_score > neg_score:
                wins += 1.0
            elif pos_score == neg_score:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


def write_public_dataset_report(result: dict, json_path: str | Path, md_path: str | Path) -> None:
    """Write JSON and Markdown reports for public audio dataset evaluation."""

    json_out = Path(json_path)
    md_out = Path(md_path)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    portable = _portable_result(result, json_out)
    json_out.write_text(json.dumps(portable, indent=2), encoding="utf-8")
    md_out.write_text(render_public_dataset_markdown(portable), encoding="utf-8")


def render_public_dataset_markdown(result: dict) -> str:
    """Render a compact report that is suitable for README linking."""

    metrics = result["metrics"]
    auc = "not available" if metrics["roc_auc"] is None else f"{metrics['roc_auc']:.3f}"
    lines = [
        "# Public Audio Dataset Evaluation",
        "",
        f"- Dataset root: `{result['dataset_root']}`",
        f"- Dataset rows: `{result['dataset_summary']['count']}`",
        f"- Training normal clips: `{result['train_clip_count']}`",
        f"- Evaluation clips: `{result['eval_clip_count']}`",
        f"- Evaluation windows: `{result['eval_window_count']}`",
        f"- Precision / recall / F1: `{metrics['precision']:.3f}` / `{metrics['recall']:.3f}` / `{metrics['f1']:.3f}`",
        f"- ROC-AUC: `{auc}`",
        "",
        "## Confusion Matrix",
        "",
        "| TP | TN | FP | FN |",
        "|---:|---:|---:|---:|",
        f"| {metrics['tp']} | {metrics['tn']} | {metrics['fp']} | {metrics['fn']} |",
        "",
        "## Notes",
        "",
        "This report uses the same window feature contract as the streaming service.",
        "For MIMII/ToyADMOS-style data, train on normal clips and evaluate on both",
        "normal and abnormal clips. The included generated sample validates the",
        "adapter offline; real downloaded data should be used for final claims.",
        "",
        "## Recent Scored Windows",
        "",
        "| File | Window | Label | Score | Threshold | Anomaly |",
        "|---|---:|---|---:|---:|---|",
    ]
    for row in result["scored_rows"][:60]:
        lines.append(
            f"| {Path(row['relative_path']).name} | {row['window_index']} | {row['label']} | "
            f"{row['score']:.3f} | {row['threshold']:.3f} | {row['is_anomaly']} |"
        )
    return "\n".join(lines)


def _portable_result(result: dict, json_out: Path) -> dict:
    """Convert local project paths in report payloads to relative strings."""

    project_root = json_out.resolve().parents[1]

    def portable_path(value: str) -> str:
        try:
            return str(Path(value).resolve().relative_to(project_root))
        except ValueError:
            return value

    portable = dict(result)
    portable["dataset_root"] = portable_path(str(portable["dataset_root"]))
    portable_rows = []
    for row in result["scored_rows"]:
        item = dict(row)
        item["path"] = portable_path(str(item["path"]))
        portable_rows.append(item)
    portable["scored_rows"] = portable_rows
    return portable
