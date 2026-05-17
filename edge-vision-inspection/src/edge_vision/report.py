from __future__ import annotations

"""Report, score figure, and annotated image generation."""

from pathlib import Path

from PIL import Image, ImageDraw


def annotate_images(results: list[dict], out_dir: str | Path) -> None:
    """Save annotated copies of analyzed images."""

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for row in results:
        img = Image.open(row["path"]).convert("RGB")
        draw = ImageDraw.Draw(img)
        color = "red" if row["is_defect"] else "green"
        draw.rectangle((4, 4, img.width - 5, img.height - 5), outline=color, width=4)
        draw.text((10, 10), f"score={row['score']:.2f}", fill=color)
        img.save(out / Path(row["path"]).name)


def write_score_distribution(results: list[dict], path: str | Path) -> None:
    """Draw a small bar chart of inspection scores."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    width, height, margin = 900, 360, 60
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw.text((margin, 20), "Vision inspection score distribution", fill="black")
    max_score = max([r["score"] for r in results] + [1.0])
    step = (width - 2 * margin) / max(1, len(results))
    for idx, row in enumerate(results):
        x0 = margin + idx * step + 2
        x1 = margin + (idx + 1) * step - 2
        y = height - margin - row["score"] / max_score * (height - 2 * margin)
        draw.rectangle((x0, y, x1, height - margin), fill="#d62728" if row["is_defect"] else "#1f77b4")
    img.save(out)


def render_markdown(results: list[dict]) -> str:
    defects = sum(1 for r in results if r["is_defect"])
    lines = [
        "# Edge Vision Inspection Report",
        "",
        f"- Images analyzed: `{len(results)}`",
        f"- Defects detected: `{defects}`",
        "",
        "![Vision score distribution](vision_score_distribution.png)",
        "",
        "| File | Label | Score | Threshold | Defect |",
        "|---|---|---:|---:|---|",
    ]
    for row in results:
        lines.append(
            f"| {Path(row['path']).name} | {row['label']} | {row['score']:.3f} | "
            f"{row['threshold']:.3f} | {row['is_defect']} |"
        )
    lines.extend(
        [
            "",
            "## Linux Application Notes",
            "",
            "Folder replay stands in for USB camera or RTSP input. ONNX detectors can",
            "replace the centroid model later while keeping the batch/report pipeline.",
            "",
        ]
    )
    return "\n".join(lines)


def write_report(results: list[dict], report_path: str | Path, annotated_dir: str | Path, figure_path: str | Path) -> None:
    annotate_images(results, annotated_dir)
    write_score_distribution(results, figure_path)
    report = Path(report_path)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(render_markdown(results), encoding="utf-8")

