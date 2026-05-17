from __future__ import annotations

"""Export the demo centroid audio scorer to ONNX."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from edge_audio.config import load_config
from edge_audio.features import load_feature_rows
from edge_audio.model import AudioCentroidModel
from edge_audio.onnx_export import export_centroid_to_onnx
from edge_audio.streaming import collect_window_feature_vectors
from edge_audio.synth import generate_demo_wavs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export centroid audio anomaly scorer to ONNX")
    parser.add_argument("--config", default=str(ROOT / "configs/default.toml"))
    parser.add_argument("--output", default=None, help="Override ONNX output path")
    return parser.parse_args()


def ensure_centroid_model(cfg) -> AudioCentroidModel:
    """Train the demo centroid model if the JSON artifact does not exist yet."""

    if cfg.model_path.exists():
        return AudioCentroidModel.load(cfg.model_path)

    generate_demo_wavs(cfg.data_dir, cfg.sample_rate_hz, cfg.clip_seconds)
    rows = load_feature_rows(cfg.data_dir)
    vectors = collect_window_feature_vectors(rows, cfg.window_seconds, cfg.hop_seconds)
    model = AudioCentroidModel.train(vectors)
    model.save(cfg.model_path)
    return model


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    output = Path(args.output) if args.output else cfg.onnx_model_path
    model = ensure_centroid_model(cfg)
    try:
        out = export_centroid_to_onnx(model, output)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    print(f"exported ONNX model: {out}")


if __name__ == "__main__":
    main()
