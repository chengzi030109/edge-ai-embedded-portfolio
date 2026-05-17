from __future__ import annotations

"""Run the embedded-Linux audio anomaly demo."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from edge_audio.config import load_config
from edge_audio.features import load_feature_rows
from edge_audio.model import AudioCentroidModel
from edge_audio.report import write_report
from edge_audio.synth import generate_demo_wavs


def main() -> None:
    cfg = load_config(ROOT / "configs/default.toml")
    generate_demo_wavs(cfg.data_dir, cfg.sample_rate_hz, cfg.clip_seconds)
    rows = load_feature_rows(cfg.data_dir)
    normal_vectors = [row["features"] for row in rows if row["label"] == "normal"]
    model = AudioCentroidModel.train(normal_vectors)
    model.save(cfg.model_path)
    results = []
    for row in rows:
        pred = model.predict(row["features"])
        results.append({"path": row["path"], "label": row["label"], **pred})
    write_report(results, cfg.report_path, cfg.figure_path)
    print(f"clips analyzed: {len(results)}")
    print(f"model: {cfg.model_path}")
    print(f"report: {cfg.report_path}")


if __name__ == "__main__":
    main()

