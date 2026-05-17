from __future__ import annotations

"""Run the embedded-Linux audio anomaly demo."""

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from edge_audio.backends import load_backend
from edge_audio.config import load_config
from edge_audio.features import load_feature_rows
from edge_audio.model import AudioCentroidModel
from edge_audio.report import write_report
from edge_audio.storage import connect, init_db, insert_events, summary
from edge_audio.streaming import analyze_dataset_windows, collect_window_feature_vectors
from edge_audio.synth import generate_demo_wavs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the edge audio anomaly portfolio demo")
    parser.add_argument(
        "--backend",
        choices=["centroid", "onnx"],
        default=None,
        help="Inference backend to use. Defaults to configs/default.toml.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(ROOT / "configs/default.toml")
    backend_name = args.backend or cfg.model_backend

    # The synthetic generator gives us deterministic, hardware-free input data:
    # normal clips are stable motor/fan tones, while anomaly clips contain
    # rubbing-like high-frequency content and impulses. In a real deployment,
    # this directory would be replaced by ALSA, PulseAudio, UDP, or file-drop
    # input while the downstream feature/model/report contract stays the same.
    generate_demo_wavs(cfg.data_dir, cfg.sample_rate_hz, cfg.clip_seconds)
    rows = load_feature_rows(cfg.data_dir)

    # Train on the same sliding-window shape used at inference time. This is
    # important for credibility: if the online worker scores 0.25 s windows,
    # the one-class centroid should learn the distribution of 0.25 s healthy
    # windows instead of smoother full-clip summaries.
    normal_vectors = collect_window_feature_vectors(rows, cfg.window_seconds, cfg.hop_seconds)
    centroid_model = AudioCentroidModel.train(normal_vectors)
    centroid_model.save(cfg.model_path)
    model = load_backend(
        backend_name,
        centroid_model_path=cfg.model_path,
        onnx_model_path=cfg.onnx_model_path,
    )

    # Keep demo output deterministic. SQLite and anomaly clips are generated
    # artifacts, so removing old files avoids stale events from previous runs.
    cfg.database_path.unlink(missing_ok=True)
    if cfg.clips_dir.exists():
        shutil.rmtree(cfg.clips_dir)

    results = analyze_dataset_windows(
        rows,
        model,
        cfg.window_seconds,
        cfg.hop_seconds,
        clips_dir=cfg.clips_dir,
        save_anomaly_clips=cfg.save_anomaly_clips,
        alarm_on_count=cfg.alarm_on_count,
        alarm_off_count=cfg.alarm_off_count,
    )
    conn = connect(cfg.database_path)
    init_db(conn)
    insert_events(conn, results)
    write_report(
        results,
        cfg.report_path,
        cfg.figure_path,
        cfg.events_json,
        model_backend=model.backend_name,
        model_path=model.model_path,
        feature_names=model.feature_names,
    )
    stats = summary(conn)
    print(f"windows analyzed: {len(results)}")
    print(f"raw anomaly windows: {stats['raw_anomaly_count']}")
    print(f"alarm windows: {stats['alarm_count']}")
    print(f"backend: {model.backend_name}")
    print(f"database: {cfg.database_path}")
    print(f"model: {model.model_path}")
    print(f"report: {cfg.report_path}")


if __name__ == "__main__":
    main()
