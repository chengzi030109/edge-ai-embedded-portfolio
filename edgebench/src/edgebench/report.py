from __future__ import annotations

"""Human-readable report rendering for EdgeBench."""

import json
from pathlib import Path


def render_markdown(report: dict) -> str:
    """Render a benchmark report as Markdown.

    JSON is best for machines, but Markdown is better for README snippets,
    internship portfolios, and interview screenshots.
    """

    latency = report["latency_ms"]
    config = report["config"]

    # Build lines explicitly instead of using a template engine; this keeps the
    # project dependency-light and easy to run on embedded Linux boards.
    lines = [
        "# EdgeBench Report",
        "",
        f"- Backend: `{report['backend']}`",
        f"- Input size: `{config['input_size']}`",
        f"- Warmup: `{config['warmup']}`",
        f"- Repeat: `{config['repeat']}`",
        f"- Model size: `{report['model_size_bytes']}` bytes",
        f"- Throughput: `{report['throughput_fps']:.2f}` inferences/s",
        "",
        "## Latency",
        "",
        "| Metric | ms |",
        "|---|---:|",
        f"| avg | {latency['avg']:.6f} |",
        f"| p50 | {latency['p50']:.6f} |",
        f"| p95 | {latency['p95']:.6f} |",
        f"| p99 | {latency['p99']:.6f} |",
        f"| min | {latency['min']:.6f} |",
        f"| max | {latency['max']:.6f} |",
        "",
        "## System",
        "",
    ]
    for key, value in report["system"].items():
        # Include system metadata so benchmark numbers are not floating around
        # without hardware/software context.
        lines.append(f"- {key}: `{value}`")
    lines.append("")
    return "\n".join(lines)


def load_json_report(path: str | Path) -> dict:
    """Load the JSON report produced by ``save_json_report``."""

    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_markdown_report(report: dict, path: str | Path) -> None:
    """Render and save a Markdown report."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_markdown(report), encoding="utf-8")
