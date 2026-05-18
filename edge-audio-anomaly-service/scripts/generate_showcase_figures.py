from __future__ import annotations

"""Generate small README-friendly showcase figures.

The project already writes detailed Markdown/JSON reports. These figures are
thin visual summaries for GitHub readers who skim the repository before opening
the full reports.
"""

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    report_dir = ROOT / "reports"
    out_dir = report_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    public_eval = json.loads((report_dir / "public_audio_evaluation.json").read_text(encoding="utf-8"))
    deployment = json.loads((report_dir / "model_deployment_report.json").read_text(encoding="utf-8"))
    draw_public_audio_card(public_eval, out_dir / "public_audio_eval_summary.png")
    draw_deployment_card(deployment, out_dir / "model_deployment_summary.png")
    print(f"showcase figures: {out_dir}")


def draw_public_audio_card(report: dict, path: Path) -> None:
    metrics = report["metrics"]
    rows = [
        ("Precision", metrics["precision"], "#2f7d32"),
        ("Recall", metrics["recall"], "#1565c0"),
        ("F1", metrics["f1"], "#6a1b9a"),
        ("ROC-AUC", metrics["roc_auc"], "#ef6c00"),
    ]
    image, draw = new_card("Public Audio Evaluation")
    draw.text((40, 88), "MIMII/ToyADMOS-style fixture, window-level scoring", fill="#555555", font=font(20))
    y = 145
    for label, value, color in rows:
        draw.text((48, y), label, fill="#222222", font=font(22))
        draw.text((760, y), f"{value:.3f}", fill="#222222", font=font(22), anchor="ra")
        draw_bar(draw, 220, y + 5, 500, 18, float(value), color)
        y += 56
    cm = metrics
    draw.text((48, 395), f"TP {cm['tp']}   TN {cm['tn']}   FP {cm['fp']}   FN {cm['fn']}", fill="#333333", font=font(24))
    image.save(path)


def draw_deployment_card(report: dict, path: Path) -> None:
    image, draw = new_card("Model Deployment Summary")
    draw.text((40, 88), "Centroid scorer exported to ONNX Runtime backend", fill="#555555", font=font(20))
    lines = [
        ("Feature vector", f"{report['feature_vector_size']} float32 values"),
        ("Centroid JSON", f"{report['centroid_json_size_bytes']} bytes"),
        ("ONNX model", f"{report['onnx_model_size_bytes']} bytes"),
        ("Centroid latency", f"{report['centroid']['latency_ms_avg']:.4f} ms"),
        ("ONNX latency", f"{report['onnx']['latency_ms_avg']:.4f} ms" if report.get("onnx") else "not available"),
        ("Mismatch count", str(report["decision_mismatch_count"])),
    ]
    y = 145
    for key, value in lines:
        draw.text((48, y), key, fill="#222222", font=font(22))
        draw.text((760, y), value, fill="#1b4f72", font=font(22), anchor="ra")
        y += 45
    draw.text((48, 430), f"ONNX status: {report['onnx_status']}", fill="#2f7d32", font=font(22))
    image.save(path)


def new_card(title: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (900, 520), "#f7f9fb")
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((24, 24, 876, 496), radius=18, fill="white", outline="#d8dee6", width=2)
    draw.text((40, 42), title, fill="#111827", font=font(32))
    return image, draw


def draw_bar(draw: ImageDraw.ImageDraw, x: int, y: int, width: int, height: int, value: float, color: str) -> None:
    draw.rounded_rectangle((x, y, x + width, y + height), radius=height // 2, fill="#e7ecf2")
    draw.rounded_rectangle((x, y, x + int(width * max(0.0, min(1.0, value))), y + height), radius=height // 2, fill=color)


def font(size: int):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


if __name__ == "__main__":
    main()
