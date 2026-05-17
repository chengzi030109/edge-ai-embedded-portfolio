from __future__ import annotations

"""Markdown and static HTML dashboard rendering."""

from html import escape
from pathlib import Path

from .storage import list_alarms, list_devices, list_telemetry, summary


def render_markdown(conn) -> str:
    """Render a portfolio-friendly gateway report."""

    stats = summary(conn)
    devices = list_devices(conn)
    alarms = list_alarms(conn, limit=10)
    lines = [
        "# Edge AI Maintenance Gateway Report",
        "",
        f"- Telemetry rows: `{stats['telemetry_count']}`",
        f"- Alarm events: `{stats['alarm_event_count']}`",
        f"- Avg feature+inference latency: `{stats['avg_pipeline_ms']:.4f} ms`",
        "",
        "## Devices",
        "",
        "| Device | Last seen | Last alarm state |",
        "|---|---:|---|",
    ]
    for device in devices:
        lines.append(f"| {device['device_id']} | {device['last_seen']:.3f} | {device['last_alarm_state']} |")
    lines.extend(["", "## Recent Alarms", "", "| Seq | State | Score | Threshold |", "|---:|---|---:|---:|"])
    for alarm in alarms:
        lines.append(f"| {alarm['seq']} | {alarm['alarm_state']} | {alarm['score']:.3f} | {alarm['threshold']:.3f} |")
    lines.extend(
        [
            "",
            "## Linux Application Notes",
            "",
            "This gateway uses SQLite for local buffering and exposes the same data through",
            "API handlers. MQTT/UART ingestion can reuse `ingest_telemetry()` without",
            "changing dashboard or report logic.",
            "",
        ]
    )
    return "\n".join(lines)


def render_dashboard_html(conn) -> str:
    """Render a static dashboard for environments without Streamlit/FastAPI."""

    rows = list(reversed(list_telemetry(conn, limit=40)))
    alarms = list_alarms(conn, limit=10)
    max_score = max([float(r["score"]) for r in rows] + [1.0])
    points = []
    for idx, row in enumerate(rows):
        x = 20 + idx * (560 / max(1, len(rows) - 1))
        y = 180 - (float(row["score"]) / max_score) * 140
        points.append(f"{x:.1f},{y:.1f}")
    table_rows = "\n".join(
        f"<tr><td>{r['seq']}</td><td>{escape(r['true_state'])}</td><td>{r['score']:.2f}</td>"
        f"<td>{escape(r['alarm_state'])}</td></tr>"
        for r in rows[-12:]
    )
    alarm_rows = "\n".join(
        f"<li>seq {a['seq']}: {escape(a['alarm_state'])}, score {a['score']:.2f}</li>" for a in alarms
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Edge AI Maintenance Gateway</title>
  <style>
    body {{ font-family: Segoe UI, Arial, sans-serif; margin: 32px; color: #1f2933; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border-bottom: 1px solid #ddd; padding: 6px 8px; text-align: left; }}
    svg {{ background: #f8fafc; border: 1px solid #d9e2ec; }}
  </style>
</head>
<body>
  <h1>Edge AI Maintenance Gateway</h1>
  <p>Local embedded-Linux style dashboard generated from SQLite telemetry.</p>
  <div class="grid">
    <section>
      <h2>Score Timeline</h2>
      <svg width="620" height="220">
        <polyline points="{' '.join(points)}" fill="none" stroke="#2563eb" stroke-width="3" />
      </svg>
    </section>
    <section>
      <h2>Recent Alarm Events</h2>
      <ul>{alarm_rows}</ul>
    </section>
  </div>
  <h2>Recent Telemetry</h2>
  <table><tr><th>Seq</th><th>State</th><th>Score</th><th>Alarm</th></tr>{table_rows}</table>
</body>
</html>
"""


def write_reports(conn, report_path: str | Path, dashboard_path: str | Path) -> None:
    """Write Markdown and HTML reports to disk."""

    report = Path(report_path)
    dashboard = Path(dashboard_path)
    report.parent.mkdir(parents=True, exist_ok=True)
    dashboard.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(render_markdown(conn), encoding="utf-8")
    dashboard.write_text(render_dashboard_html(conn), encoding="utf-8")

