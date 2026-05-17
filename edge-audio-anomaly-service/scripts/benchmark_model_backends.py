from __future__ import annotations

"""Benchmark centroid and optional ONNX inference backends."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from edge_audio.config import load_config
from edge_audio.deployment import compare_backends, write_deployment_report
from edge_audio.features import load_feature_rows
from edge_audio.model import AudioCentroidModel
from edge_audio.streaming import collect_window_feature_vectors
from edge_audio.synth import generate_demo_wavs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark audio inference deployment backends")
    parser.add_argument("--config", default=str(ROOT / "configs/default.toml"))
    parser.add_argument("--samples", type=int, default=64, help="Maximum window feature vectors to benchmark")
    return parser.parse_args()


def ensure_demo_vectors(cfg, limit: int):
    """Return deterministic window features and ensure the centroid artifact exists."""

    generate_demo_wavs(cfg.data_dir, cfg.sample_rate_hz, cfg.clip_seconds)
    rows = load_feature_rows(cfg.data_dir)
    vectors = collect_window_feature_vectors(rows, cfg.window_seconds, cfg.hop_seconds, label_filter=None)
    normal_vectors = collect_window_feature_vectors(rows, cfg.window_seconds, cfg.hop_seconds)
    model = AudioCentroidModel.train(normal_vectors)
    model.save(cfg.model_path)
    return vectors[:limit]


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    vectors = ensure_demo_vectors(cfg, args.samples)
    report = compare_backends(
        vectors,
        centroid_model_path=cfg.model_path,
        onnx_model_path=cfg.onnx_model_path,
    )
    json_path = ROOT / "reports/model_deployment_report.json"
    md_path = ROOT / "reports/model_deployment_report.md"
    write_deployment_report(report, json_path, md_path)
    print(f"deployment report: {md_path}")
    print(f"onnx status: {report['onnx_status']}")


if __name__ == "__main__":
    main()
