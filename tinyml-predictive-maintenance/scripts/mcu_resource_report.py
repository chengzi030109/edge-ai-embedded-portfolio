from __future__ import annotations

"""Generate an MCU resource-budget report for the TinyML detector.

This report is intentionally engineering-facing rather than ML-facing. It
answers the questions an embedded interviewer is likely to ask:

* how many bytes does the model need?
* how large are the sample and feature buffers?
* what is still float, and what is already fixed-point?
* how much work happens per inference window?
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tpm.config import load_config
from tpm.fixed_point import Q_FRACTIONAL_BITS, FixedPointCentroid
from tpm.model import CentroidAnomalyDetector


def build_report(model_path: str | Path, config_path: str | Path, fixed_report_path: str | Path | None) -> dict:
    """Collect model/config values and estimate MCU-facing resource use."""

    cfg = load_config(config_path)
    model = CentroidAnomalyDetector.load(model_path)
    fixed = FixedPointCentroid.from_float_model(model)
    n_features = len(model.feature_names)

    fixed_report = None
    if fixed_report_path is not None and Path(fixed_report_path).exists():
        fixed_report = json.loads(Path(fixed_report_path).read_text(encoding="utf-8"))

    # The sample buffer estimate assumes a common accelerometer/ADC path where
    # samples are stored as signed 16-bit integers before conversion/scaling.
    sample_buffer_i16_bytes = cfg.window_size * 2

    # A compact binary telemetry frame is not implemented yet, but this estimate
    # is useful for RTOS queue sizing: sequence, timestamp, score, threshold,
    # flags, and a small checksum/status byte. JSONL stays on the laptop/gateway.
    compact_telemetry_bytes = 4 + 4 + 4 + 4 + 1 + 1 + 2

    # Centroid inference does one subtract, one multiply, one shift, and one
    # square per feature, then one integer sqrt and one threshold comparison.
    ops_per_feature = {
        "subtract": 1,
        "multiply": 2,  # normalization multiply plus square
        "shift": 1,
        "accumulate": 1,
    }

    return {
        "model_path": str(model_path),
        "sample_rate_hz": cfg.sample_rate_hz,
        "window_size": cfg.window_size,
        "n_features": n_features,
        "feature_names": model.feature_names,
        "buffers": {
            "sample_buffer_i16_bytes": sample_buffer_i16_bytes,
            "feature_vector_float_bytes": n_features * 4,
            "feature_vector_q24_8_bytes": n_features * 4,
            "compact_telemetry_estimate_bytes": compact_telemetry_bytes,
        },
        "model_parameters": {
            "float_bytes": int(model.mean.nbytes + model.scale.nbytes + 4),
            "fixed_q24_8_bytes": fixed.parameter_bytes,
            "q_fractional_bits": Q_FRACTIONAL_BITS,
            "threshold_q": fixed.threshold_q,
        },
        "inference_work": {
            "ops_per_feature": ops_per_feature,
            "integer_sqrt_per_window": 1,
            "threshold_compare_per_window": 1,
            "notes": "FFT/statistical feature extraction dominates compute; centroid scoring is tiny.",
        },
        "fixed_point_validation": {
            "decision_mismatches": None if fixed_report is None else fixed_report.get("decision_mismatches"),
            "integer_path_decision_mismatches": None
            if fixed_report is None
            else fixed_report.get("integer_path_decision_mismatches"),
            "mean_abs_score_error": None
            if fixed_report is None
            else fixed_report.get("score_error", {}).get("mean_abs"),
            "integer_path_mean_abs_score_error": None
            if fixed_report is None
            else fixed_report.get("integer_path_score_error", {}).get("mean_abs"),
        },
    }


def render_markdown(report: dict) -> str:
    """Render the resource report as portfolio-friendly Markdown."""

    buffers = report["buffers"]
    params = report["model_parameters"]
    validation = report["fixed_point_validation"]
    return "\n".join(
        [
            "# MCU Resource Budget",
            "",
            "This report estimates the resource footprint of the centroid detector",
            "when moved from the laptop prototype into MCU firmware.",
            "",
            "## Model And Buffer Sizes",
            "",
            "| Item | Bytes | Notes |",
            "|---|---:|---|",
            f"| Raw sample window (`int16_t`) | {buffers['sample_buffer_i16_bytes']} | {report['window_size']} samples |",
            f"| Feature vector (`float`) | {buffers['feature_vector_float_bytes']} | Current float C path |",
            f"| Feature vector (`Q24.8 int32_t`) | {buffers['feature_vector_q24_8_bytes']} | Fixed-point inference path |",
            f"| Float centroid parameters | {params['float_bytes']} | mean + scale + threshold |",
            f"| Q24.8 centroid parameters | {params['fixed_q24_8_bytes']} | mean_q + inv_scale_q + threshold_q |",
            f"| Compact telemetry estimate | {buffers['compact_telemetry_estimate_bytes']} | Binary MCU-to-gateway frame estimate |",
            "",
            "## Fixed-Point Validation",
            "",
            f"- Q format: `Q24.{params['q_fractional_bits']}`",
            f"- threshold_q: `{params['threshold_q']}`",
            f"- parameter-only decision mismatches: `{validation['decision_mismatches']}`",
            f"- integer-path decision mismatches: `{validation['integer_path_decision_mismatches']}`",
            f"- parameter-only mean absolute score error: `{validation['mean_abs_score_error']}`",
            f"- integer-path mean absolute score error: `{validation['integer_path_mean_abs_score_error']}`",
            "",
            "## Per-Window Inference Work",
            "",
            "For each feature, fixed-point centroid inference performs one subtract,",
            "one normalization multiply, one shift, one square multiply, and one",
            "accumulate. Each window then performs one integer square root and one",
            "threshold comparison.",
            "",
            "Feature extraction, especially FFT band power, is expected to dominate",
            "runtime. The centroid scoring stage is intentionally small enough to fit",
            "comfortably inside a periodic RTOS inference task.",
            "",
            "## Interview Reading",
            "",
            "The important engineering point is that the project now has both a float C",
            "path for clarity and a Q-format C path for MCU realism. The next true",
            "firmware step is to make feature extraction produce Q-format values",
            "directly, then replace the laptop replay source with a sensor driver.",
            "",
        ]
    )


def main() -> None:
    """CLI entry point for report generation."""

    parser = argparse.ArgumentParser(description="Generate MCU resource-budget reports.")
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--model", default="artifacts/model.json")
    parser.add_argument("--fixed-report", default="reports/fixed_point_report.json")
    parser.add_argument("--json-out", default="reports/mcu_resource_budget.json")
    parser.add_argument("--md-out", default="reports/mcu_resource_budget.md")
    args = parser.parse_args()

    report = build_report(args.model, args.config, args.fixed_report)
    Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json_out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    Path(args.md_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.md_out).write_text(render_markdown(report), encoding="utf-8")
    print(f"saved JSON: {args.json_out}")
    print(f"saved Markdown: {args.md_out}")


if __name__ == "__main__":
    main()
