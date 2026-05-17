from __future__ import annotations

"""Evaluate MIMII/ToyADMOS-style industrial audio folders.

The default mode generates a small local fixture with a MIMII-like directory
shape, so the command works without downloading data. For a real experiment,
pass ``--data-root`` pointing at a directory containing labeled WAV files such
as ``fan/id_00/train/normal`` and ``fan/id_00/test/abnormal``.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from edge_audio.config import load_config
from edge_audio.datasets import load_public_audio_rows, split_train_eval_rows, summarize_rows
from edge_audio.evaluation import collect_labeled_window_vectors, evaluate_predictions, write_public_dataset_report
from edge_audio.model import AudioCentroidModel
from edge_audio.synth import generate_public_audio_sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a public industrial audio dataset folder")
    parser.add_argument("--config", default=str(ROOT / "configs/default.toml"))
    parser.add_argument("--data-root", default=str(ROOT / "data/public_audio_sample"))
    parser.add_argument(
        "--no-generate-sample",
        action="store_true",
        help="Do not generate the built-in MIMII-shaped sample when data-root is missing or empty.",
    )
    parser.add_argument("--json-out", default=str(ROOT / "reports/public_audio_evaluation.json"))
    parser.add_argument("--md-out", default=str(ROOT / "reports/public_audio_evaluation.md"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    data_root = Path(args.data_root)
    if not args.no_generate_sample and not any(data_root.rglob("*.wav")):
        generate_public_audio_sample(data_root, cfg.sample_rate_hz, cfg.clip_seconds)

    rows = load_public_audio_rows(data_root)
    if not rows:
        raise SystemExit(
            "No labeled WAV files found. Expected path segments like normal, abnormal, or anomaly "
            "under a MIMII/ToyADMOS-style folder."
        )

    train_rows, eval_rows = split_train_eval_rows(rows)
    if not train_rows or not eval_rows:
        raise SystemExit("Need at least one normal training clip and one evaluation clip.")

    train_windows = collect_labeled_window_vectors(train_rows, cfg.window_seconds, cfg.hop_seconds)
    eval_windows = collect_labeled_window_vectors(eval_rows, cfg.window_seconds, cfg.hop_seconds)
    model = AudioCentroidModel.train([row["features"] for row in train_windows])
    result = evaluate_predictions(eval_windows, model)
    result.update(
        {
            "dataset_root": str(data_root),
            "dataset_summary": summarize_rows(rows),
            "train_clip_count": len(train_rows),
            "eval_clip_count": len(eval_rows),
            "train_window_count": len(train_windows),
            "eval_window_count": len(eval_windows),
            "model_backend": "centroid",
        }
    )
    write_public_dataset_report(result, args.json_out, args.md_out)
    metrics = result["metrics"]
    auc = "n/a" if metrics["roc_auc"] is None else f"{metrics['roc_auc']:.3f}"
    print(f"dataset rows: {len(rows)}")
    print(f"eval windows: {len(eval_windows)}")
    print(f"precision/recall/f1: {metrics['precision']:.3f}/{metrics['recall']:.3f}/{metrics['f1']:.3f}")
    print(f"roc_auc: {auc}")
    print(f"report: {args.md_out}")


if __name__ == "__main__":
    main()
