from __future__ import annotations

"""Run the edge vision inspection demo."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from edge_vision.config import load_config
from edge_vision.features import load_feature_rows
from edge_vision.model import VisionCentroidModel
from edge_vision.report import write_report
from edge_vision.synth import generate_demo_images


def main() -> None:
    cfg = load_config(ROOT / "configs/default.toml")
    generate_demo_images(cfg.data_dir, cfg.image_size)
    rows = load_feature_rows(cfg.data_dir)
    model = VisionCentroidModel.train([r["features"] for r in rows if r["label"] == "normal"])
    model.save(cfg.model_path)
    results = []
    for row in rows:
        pred = model.predict(row["features"])
        results.append({"path": row["path"], "label": row["label"], **pred})
    write_report(results, cfg.report_path, cfg.annotated_dir, cfg.figure_path)
    print(f"images analyzed: {len(results)}")
    print(f"model: {cfg.model_path}")
    print(f"report: {cfg.report_path}")


if __name__ == "__main__":
    main()

