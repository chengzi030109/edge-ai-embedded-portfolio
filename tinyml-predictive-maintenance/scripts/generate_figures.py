from __future__ import annotations

"""Generate static PNG figures for README/report use.

The project treats figures as lightweight build artifacts, not as hand-edited
screenshots. Every plot can be regenerated from JSON/JSONL reports, and each
matplotlib plot has a Pillow fallback so the Windows laptop demo still works
when optional plotting dependencies are missing.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - depends on optional plotting dependency
    matplotlib = None
    plt = None


def _load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _save(fig, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"wrote {out}")


def _fallback_line_plot(series: dict[str, list[float]], title: str, out: Path) -> None:
    """Draw a simple line chart with Pillow when matplotlib is unavailable."""

    from PIL import Image, ImageDraw

    width, height = 960, 420
    margin = 60
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw.text((margin, 20), title, fill="black")
    draw.rectangle((margin, margin, width - margin, height - margin), outline="black")

    all_values = [v for values in series.values() for v in values]
    vmin = min(all_values)
    vmax = max(all_values)
    if vmax == vmin:
        vmax = vmin + 1.0
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for idx, (name, values) in enumerate(series.items()):
        pts = []
        for i, value in enumerate(values):
            x = margin + i * (width - 2 * margin) / max(1, len(values) - 1)
            y = height - margin - (value - vmin) * (height - 2 * margin) / (vmax - vmin)
            pts.append((x, y))
        if len(pts) > 1:
            draw.line(pts, fill=colors[idx % len(colors)], width=3)
        draw.text((margin + 140 * idx, height - margin + 15), name, fill=colors[idx % len(colors)])
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    print(f"wrote {out} (Pillow fallback)")


def _fallback_bar_line(names: list[str], bars: list[float], line: list[float], title: str, out: Path) -> None:
    """Draw a compact bar+line chart with Pillow."""

    from PIL import Image, ImageDraw

    width, height = 1000, 460
    margin = 70
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw.text((margin, 20), title, fill="black")
    draw.rectangle((margin, margin, width - margin, height - margin), outline="black")
    max_bar = max(max(bars), 1e-9)
    max_line = max(max(line), 1e-9)
    step = (width - 2 * margin) / max(1, len(names))
    line_pts = []
    for i, (name, bar, line_value) in enumerate(zip(names, bars, line, strict=False)):
        x0 = margin + i * step + step * 0.2
        x1 = margin + (i + 1) * step - step * 0.2
        y = height - margin - bar / max_bar * (height - 2 * margin)
        draw.rectangle((x0, y, x1, height - margin), fill="#4c78a8")
        lx = (x0 + x1) / 2
        ly = height - margin - line_value / max_line * (height - 2 * margin)
        line_pts.append((lx, ly))
        draw.text((x0, height - margin + 10), name[:12], fill="black")
    if len(line_pts) > 1:
        draw.line(line_pts, fill="#e45756", width=3)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    print(f"wrote {out} (Pillow fallback)")


def plot_portfolio_pipeline(out: Path) -> None:
    """Draw the end-to-end demo pipeline as a generated PNG.

    The pipeline is not a measured chart, so using Pillow directly is simpler
    than forcing it through matplotlib. It gives the README a first-screen
    visual that explains the embedded AI story before readers dive into code.
    """

    from PIL import Image, ImageDraw

    width, height = 1300, 360
    margin_x = 45
    box_w = 185
    box_h = 92
    y = 130
    stages = [
        ("Sensor", "simulator / CSV"),
        ("Window", "256 samples"),
        ("Features", "FFT + stats"),
        ("Detector", "centroid score"),
        ("Alarm", "debounced"),
        ("Telemetry", "JSONL report"),
    ]
    colors = ["#2f6f73", "#6a7f2a", "#9b6b1f", "#7c4d8a", "#a34545", "#345f9c"]
    img = Image.new("RGB", (width, height), "#f7f7f4")
    draw = ImageDraw.Draw(img)
    draw.text((margin_x, 35), "TinyML predictive-maintenance demo pipeline", fill="#111111")
    draw.text((margin_x, 62), "Laptop prototype today; sensor driver and RTOS tasks replace the left edge on MCU.", fill="#333333")

    for idx, ((title, subtitle), color) in enumerate(zip(stages, colors, strict=False)):
        x = margin_x + idx * (box_w + 26)
        draw.rounded_rectangle((x, y, x + box_w, y + box_h), radius=8, fill="white", outline=color, width=3)
        draw.text((x + 18, y + 24), title, fill=color)
        draw.text((x + 18, y + 52), subtitle, fill="#222222")
        if idx < len(stages) - 1:
            ax0 = x + box_w + 4
            ax1 = x + box_w + 22
            ay = y + box_h / 2
            draw.line((ax0, ay, ax1, ay), fill="#333333", width=3)
            draw.polygon([(ax1, ay), (ax1 - 8, ay - 6), (ax1 - 8, ay + 6)], fill="#333333")

    draw.text((margin_x, 285), "Firmware path: CMSIS-DSP features -> float C parity now -> Q-format C inference next.", fill="#111111")
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    print(f"wrote {out} (Pillow pipeline)")


def plot_synthetic_score_curve(telemetry: str | Path, out: Path) -> None:
    """Plot model score and threshold from node telemetry."""

    rows = _load_jsonl(telemetry)
    seq = [r["seq"] for r in rows]
    score = [r["score"] for r in rows]
    threshold = [r["threshold"] for r in rows]
    if plt is None:
        _fallback_line_plot({"score": score, "threshold": threshold}, "Synthetic node score curve", out)
        return
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(seq, score, label="score", linewidth=2)
    ax.plot(seq, threshold, label="threshold", linestyle="--")
    ax.set_xlabel("window")
    ax.set_ylabel("anomaly score")
    ax.set_title("Synthetic node score curve")
    ax.legend()
    _save(fig, out)


def plot_alarm_timeline(telemetry: str | Path, out: Path) -> None:
    """Plot raw anomaly decisions versus debounced alarm state."""

    rows = _load_jsonl(telemetry)
    seq = [r["seq"] for r in rows]
    raw = [int(r.get("is_anomaly_raw", r.get("is_anomaly", False))) for r in rows]
    alarm = [int(r.get("is_alarm", r.get("is_anomaly", False))) for r in rows]
    if plt is None:
        _fallback_line_plot({"raw anomaly": raw, "debounced alarm": alarm}, "Alarm debounce timeline", out)
        return
    fig, ax = plt.subplots(figsize=(8, 2.5))
    ax.step(seq, raw, where="post", label="raw anomaly", linewidth=2)
    ax.step(seq, alarm, where="post", label="debounced alarm", linewidth=2)
    ax.set_xlabel("window")
    ax.set_yticks([0, 1])
    ax.set_title("Alarm debounce timeline")
    ax.legend(loc="upper left")
    _save(fig, out)


def plot_cwru_comparison(report_path: str | Path, out: Path) -> None:
    """Plot CWRU model F1 and latency side by side."""

    report = _load_json(report_path)
    names = [m["name"] for m in report["models"]]
    f1 = [m["metrics"]["f1"] for m in report["models"]]
    latency = [m["latency_ms"]["avg"] for m in report["models"]]
    if plt is None:
        _fallback_bar_line(names, f1, latency, "CWRU model comparison", out)
        return
    fig, ax1 = plt.subplots(figsize=(9, 3.5))
    ax1.bar(names, f1, color="#4c78a8", label="F1")
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("F1")
    ax1.tick_params(axis="x", rotation=20)
    ax2 = ax1.twinx()
    ax2.plot(names, latency, color="#f58518", marker="o", label="latency")
    ax2.set_ylabel("avg latency (ms)")
    ax1.set_title("CWRU model comparison")
    _save(fig, out)


def plot_quantization(report_path: str | Path, out: Path) -> None:
    """Plot quantized autoencoder size and latency."""

    report = _load_json(report_path)
    names = [entry["name"].replace(" ", "\n") for entry in report["formats"]]
    sizes = [entry["size_bytes"] / 1024.0 for entry in report["formats"]]
    latency = [entry["latency_ms_avg"] for entry in report["formats"]]
    if plt is None:
        _fallback_bar_line(names, sizes, latency, "Autoencoder export size vs latency", out)
        return
    fig, ax1 = plt.subplots(figsize=(8, 3.5))
    ax1.bar(names, sizes, color="#54a24b")
    ax1.set_ylabel("size (KiB)")
    ax2 = ax1.twinx()
    ax2.plot(names, latency, color="#e45756", marker="o")
    ax2.set_ylabel("avg latency (ms)")
    ax1.set_title("Autoencoder export size vs latency")
    _save(fig, out)


def plot_phm2008_comparison(report_path: str | Path, out: Path) -> None:
    """Plot PHM2008/C-MAPSS model F1 with false-positive rate annotations."""

    report = _load_json(report_path)
    models = report.get("models", [])
    if not models:
        print(f"warning: {report_path} has no model rows; skipping PHM2008 figure")
        return
    names = [entry["name"] for entry in models]
    f1 = [entry["metrics"]["f1"] for entry in models]
    fpr = [entry["metrics"]["false_positive_rate"] for entry in models]
    if plt is None:
        _fallback_bar_line(names, f1, fpr, "PHM2008/C-MAPSS harder degradation comparison", out)
        return
    fig, ax1 = plt.subplots(figsize=(8, 3.4))
    ax1.bar(names, f1, color="#4c78a8", label="F1")
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("F1")
    ax1.tick_params(axis="x", rotation=15)
    ax2 = ax1.twinx()
    ax2.plot(names, fpr, color="#e45756", marker="o", label="FPR")
    ax2.set_ylim(0, max(0.35, max(fpr) * 1.2))
    ax2.set_ylabel("false positive rate")
    ax1.set_title("PHM2008/C-MAPSS harder degradation comparison")
    _save(fig, out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate README/report PNG figures.")
    parser.add_argument("--telemetry", default="runs/telemetry.jsonl")
    parser.add_argument("--cwru-report", default="reports/cwru_comparison.json")
    parser.add_argument("--quant-report", default="reports/quantization_report.json")
    parser.add_argument("--phm-report", default="reports/phm2008_comparison.json")
    parser.add_argument("--out-dir", default="reports/figures")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    plot_portfolio_pipeline(out_dir / "portfolio_pipeline.png")
    if Path(args.telemetry).exists():
        plot_synthetic_score_curve(args.telemetry, out_dir / "synthetic_score_curve.png")
        plot_alarm_timeline(args.telemetry, out_dir / "alarm_debounce_timeline.png")
    else:
        print(f"warning: telemetry file {args.telemetry} not found; skipping telemetry figures")
    if Path(args.cwru_report).exists():
        plot_cwru_comparison(args.cwru_report, out_dir / "cwru_model_comparison.png")
    if Path(args.quant_report).exists():
        plot_quantization(args.quant_report, out_dir / "quantization_size_latency.png")
    if Path(args.phm_report).exists():
        plot_phm2008_comparison(args.phm_report, out_dir / "phm2008_comparison.png")


if __name__ == "__main__":
    main()
