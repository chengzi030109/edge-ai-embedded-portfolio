from __future__ import annotations

"""Report and figure generation for audio anomaly results."""

from pathlib import Path

from PIL import Image, ImageDraw


def write_score_figure(results: list[dict], path: str | Path) -> None:
    """Draw a compact score timeline with Pillow."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    width, height, margin = 900, 360, 60
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw.text((margin, 20), "Audio anomaly score curve", fill="black")
    draw.rectangle((margin, margin, width - margin, height - margin), outline="black")
    max_score = max([r["score"] for r in results] + [1.0])
    points = []
    for idx, row in enumerate(results):
        x = margin + idx * (width - 2 * margin) / max(1, len(results) - 1)
        y = height - margin - row["score"] / max_score * (height - 2 * margin)
        points.append((x, y))
        color = "#d62728" if row["is_anomaly"] else "#1f77b4"
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color)
    if len(points) > 1:
        draw.line(points, fill="#444444", width=2)
    img.save(out)


def render_markdown(results: list[dict], figure_path: str | Path) -> str:
    """Render an audio anomaly report."""

    anomalies = sum(1 for row in results if row["is_anomaly"])
    lines = [
        "# Edge Audio Anomaly Report",
        "",
        f"- Clips analyzed: `{len(results)}`",
        f"- Anomaly events: `{anomalies}`",
        f"- Figure: `{figure_path}`",
        "",
        "![Audio score curve](audio_score_curve.png)",
        "",
        "| File | Label | Score | Threshold | Anomaly |",
        "|---|---|---:|---:|---|",
    ]
    for row in results:
        lines.append(
            f"| {Path(row['path']).name} | {row['label']} | {row['score']:.3f} | "
            f"{row['threshold']:.3f} | {row['is_anomaly']} |"
        )
    lines.extend(
        [
            "",
            "## Linux Application Notes",
            "",
            "WAV replay stands in for microphone or UDP audio input. The feature and",
            "model boundary can stay stable when the input source becomes ALSA, PulseAudio,",
            "or an embedded recorder process.",
            "",
        ]
    )
    return "\n".join(lines)


def write_report(results: list[dict], report_path: str | Path, figure_path: str | Path) -> None:
    write_score_figure(results, figure_path)
    report = Path(report_path)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(render_markdown(results, figure_path), encoding="utf-8")

