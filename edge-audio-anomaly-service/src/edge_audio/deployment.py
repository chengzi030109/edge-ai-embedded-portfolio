from __future__ import annotations

"""Model deployment benchmarking and report helpers."""

import json
import time
from pathlib import Path

import numpy as np

from .backends import AudioModelBackend, load_backend


def benchmark_backend(model: AudioModelBackend, vectors: list[np.ndarray]) -> dict:
    """Measure average inference latency and collect scores for one backend."""

    started = time.perf_counter()
    predictions = [model.predict(vector) for vector in vectors]
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    scores = [float(pred["score"]) for pred in predictions]
    decisions = [bool(pred["is_anomaly"]) for pred in predictions]
    return {
        "backend": model.backend_name,
        "model_path": model.model_path,
        "latency_ms_avg": elapsed_ms / max(1, len(vectors)),
        "scores": scores,
        "decisions": decisions,
    }


def compare_backends(
    vectors: list[np.ndarray],
    *,
    centroid_model_path: str | Path,
    onnx_model_path: str | Path,
) -> dict:
    """Benchmark centroid and, when available, ONNX Runtime on the same vectors."""

    centroid = load_backend("centroid", centroid_model_path=centroid_model_path, onnx_model_path=onnx_model_path)
    centroid_result = benchmark_backend(centroid, vectors)
    report = {
        "feature_vector_size": int(len(vectors[0])) if vectors else 0,
        "sample_count": len(vectors),
        "centroid_json_size_bytes": _size_or_zero(centroid_model_path),
        "onnx_model_size_bytes": _size_or_zero(onnx_model_path),
        "centroid": _public_backend_result(centroid_result),
        "onnx": None,
        "score_error_mean": None,
        "score_error_max": None,
        "decision_mismatch_count": None,
        "onnx_status": "not_run",
    }

    try:
        onnx_backend = load_backend("onnx", centroid_model_path=centroid_model_path, onnx_model_path=onnx_model_path)
        onnx_result = benchmark_backend(onnx_backend, vectors)
    except Exception as exc:
        report["onnx_status"] = f"skipped: {_portable_message(exc, centroid_model_path)}"
        return report

    errors = [
        abs(float(a) - float(b))
        for a, b in zip(centroid_result["scores"], onnx_result["scores"], strict=False)
    ]
    mismatches = sum(
        1
        for a, b in zip(centroid_result["decisions"], onnx_result["decisions"], strict=False)
        if bool(a) != bool(b)
    )
    report["onnx"] = _public_backend_result(onnx_result)
    report["score_error_mean"] = float(np.mean(errors)) if errors else 0.0
    report["score_error_max"] = float(np.max(errors)) if errors else 0.0
    report["decision_mismatch_count"] = int(mismatches)
    report["onnx_status"] = "ok"
    return report


def write_deployment_report(report: dict, json_path: str | Path, md_path: str | Path) -> None:
    """Write JSON and Markdown deployment reports."""

    json_out = Path(json_path)
    md_out = Path(md_path)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_out.write_text(render_deployment_markdown(report), encoding="utf-8")


def render_deployment_markdown(report: dict) -> str:
    """Render a compact model deployment report."""

    onnx = report.get("onnx") or {}
    lines = [
        "# Model Deployment Report",
        "",
        f"- Feature vector size: `{report['feature_vector_size']}`",
        f"- Benchmark samples: `{report['sample_count']}`",
        f"- Centroid JSON size: `{report['centroid_json_size_bytes']} bytes`",
        f"- ONNX model size: `{report['onnx_model_size_bytes']} bytes`",
        f"- ONNX status: `{report['onnx_status']}`",
        "",
        "## Backend Latency",
        "",
        "| Backend | Avg inference ms | Model path |",
        "|---|---:|---|",
        f"| centroid | {report['centroid']['latency_ms_avg']:.6f} | {report['centroid']['model_path']} |",
    ]
    if onnx:
        lines.append(f"| onnx | {onnx['latency_ms_avg']:.6f} | {onnx['model_path']} |")
    lines.extend(
        [
            "",
            "## Parity",
            "",
            f"- Mean score error: `{_fmt_optional(report.get('score_error_mean'))}`",
            f"- Max score error: `{_fmt_optional(report.get('score_error_max'))}`",
            f"- Decision mismatch count: `{_fmt_optional(report.get('decision_mismatch_count'))}`",
            "",
            "The ONNX backend replaces only the inference worker. Audio capture,",
            "feature extraction, alarm debounce, SQLite buffering, and API/report",
            "logic stay unchanged.",
            "",
        ]
    )
    return "\n".join(lines)


def _public_backend_result(result: dict) -> dict:
    return {
        "backend": result["backend"],
        "model_path": Path(result["model_path"]).name,
        "latency_ms_avg": result["latency_ms_avg"],
    }


def _size_or_zero(path: str | Path) -> int:
    p = Path(path)
    return p.stat().st_size if p.exists() else 0


def _fmt_optional(value: object) -> str:
    if value is None:
        return "not available"
    if isinstance(value, float):
        return f"{value:.8f}"
    return str(value)


def _portable_message(exc: Exception, anchor_path: str | Path) -> str:
    """Remove machine-specific absolute project paths from report messages."""

    message = str(exc)
    try:
        project_root = Path(anchor_path).resolve().parents[1]
    except IndexError:
        return message
    return message.replace(str(project_root), ".")
