from __future__ import annotations

"""Evaluate the anomaly detector and write portfolio-ready reports.

This is the script to run before updating a README, making a demo video, or
talking to an interviewer. It quantifies detection quality instead of relying on
console output from the simulated node.
"""

import argparse
import sys
from pathlib import Path

# Keep direct script execution simple:
#   python scripts/evaluate_model.py
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tpm.config import load_config
from tpm.evaluation import evaluate_model, save_evaluation_json, save_evaluation_markdown
from tpm.model import CentroidAnomalyDetector


def main() -> None:
    """Parse CLI options, run evaluation, and save JSON/Markdown reports."""

    parser = argparse.ArgumentParser(description="Evaluate the TinyML anomaly detector.")
    parser.add_argument("--config", default="configs/default.json", help="JSON config file with project defaults.")
    parser.add_argument("--model", help="Model JSON path. Overrides config model_path.")
    parser.add_argument("--windows-per-state", type=int, default=120)
    parser.add_argument("--json-out", help="Evaluation JSON path. Overrides config evaluation_json.")
    parser.add_argument("--md-out", help="Evaluation Markdown path. Overrides config evaluation_md.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_path = args.model if args.model is not None else cfg.model_path
    json_out = args.json_out if args.json_out is not None else cfg.evaluation_json
    md_out = args.md_out if args.md_out is not None else cfg.evaluation_md

    model = CentroidAnomalyDetector.load(model_path)
    report = evaluate_model(
        model=model,
        states=cfg.states,
        windows_per_state=args.windows_per_state,
        sample_rate_hz=cfg.sample_rate_hz,
        window_size=cfg.window_size,
    )

    save_evaluation_json(report, json_out)
    save_evaluation_markdown(report, md_out)

    metrics = report["metrics"]
    print(f"saved evaluation JSON: {json_out}")
    print(f"saved evaluation Markdown: {md_out}")
    print(
        "accuracy={accuracy:.4f} precision={precision:.4f} "
        "recall={recall:.4f} f1={f1:.4f}".format(**metrics)
    )


if __name__ == "__main__":
    main()
