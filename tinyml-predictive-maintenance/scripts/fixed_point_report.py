from __future__ import annotations

"""Generate a float-vs-fixed-point centroid report."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from tpm.config import load_config
from tpm.features import FeatureConfig, extract_features, vectorize
from tpm.fixed_point import compare_fixed_point
from tpm.model import CentroidAnomalyDetector
from tpm.signal_sim import MotorSignalSimulator, SignalConfig


def _sample_vectors(sample_rate_hz: int, window_size: int, windows_per_state: int) -> np.ndarray:
    """Build a deterministic mixed normal/fault feature matrix."""

    feature_cfg = FeatureConfig(sample_rate_hz=sample_rate_hz)
    vectors = []
    for idx, state in enumerate(("normal", "imbalance", "rubbing", "bearing")):
        sim = MotorSignalSimulator(SignalConfig(sample_rate_hz=sample_rate_hz, seed=700 + idx))
        for _ in range(windows_per_state):
            vectors.append(vectorize(extract_features(sim.read(window_size, state), feature_cfg)))
    return np.vstack(vectors).astype(np.float32)


def _render_markdown(report: dict) -> str:
    """Render a concise portfolio-ready fixed-point report."""

    q = report["quantization"]
    err = report["score_error"]
    int_err = report.get("integer_path_score_error", {})
    return "\n".join(
        [
            "# Fixed-Point Centroid Report",
            "",
            f"- Vectors evaluated: `{report['n_vectors']}`",
            f"- Float parameter footprint: `{report['float_model_bytes']}` bytes",
            f"- Fixed-point parameter footprint: `{report['fixed_point_bytes']}` bytes",
            f"- Decision mismatches: `{report['decision_mismatches']}`",
            f"- Integer-path decision mismatches: `{report.get('integer_path_decision_mismatches')}`",
            f"- Mean absolute score error: `{err['mean_abs']:.6e}`",
            f"- Max absolute score error: `{err['max_abs']:.6e}`",
            f"- Integer-path mean absolute score error: `{int_err.get('mean_abs', 0.0):.6e}`",
            f"- Integer-path max absolute score error: `{int_err.get('max_abs', 0.0):.6e}`",
            "",
            "## Quantization Parameters",
            "",
            "| Parameter | Value |",
            "|---|---:|",
            f"| Q fractional bits | {q['q_fractional_bits']} |",
            f"| Q step | {q['q_step']:.6e} |",
            f"| threshold_q | {q['threshold_q']} |",
            "",
            "This report simulates storing centroid parameters in Q24.8 int32 format.",
            "The parameter-only path measures quantized model drift while keeping",
            "float feature inputs. The integer path also quantizes feature inputs",
            "and mirrors the fixed-point C implementation in `firmware/`.",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate fixed-point centroid drift report.")
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--model", help="Model JSON path. Overrides config model_path.")
    parser.add_argument("--windows-per-state", type=int, default=80)
    parser.add_argument("--json-out", default="reports/fixed_point_report.json")
    parser.add_argument("--md-out", default="reports/fixed_point_report.md")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model = CentroidAnomalyDetector.load(args.model or cfg.model_path)
    vectors = _sample_vectors(cfg.sample_rate_hz, cfg.window_size, args.windows_per_state)
    report = compare_fixed_point(model, vectors)

    Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json_out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    Path(args.md_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.md_out).write_text(_render_markdown(report), encoding="utf-8")

    print(f"saved JSON: {args.json_out}")
    print(f"saved Markdown: {args.md_out}")
    print(
        f"mismatches={report['decision_mismatches']} "
        f"mean_abs_error={report['score_error']['mean_abs']:.3e}"
    )


if __name__ == "__main__":
    main()
