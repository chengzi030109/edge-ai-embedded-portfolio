from __future__ import annotations

"""Report and figure generation for audio anomaly results."""

import json
from pathlib import Path

from PIL import Image, ImageDraw


def write_score_figure(results: list[dict], path: str | Path) -> None:
    """Draw a compact score timeline with raw and debounced alarm states."""

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
        if row.get("is_alarm", False):
            color = "#d62728"
        elif row.get("is_anomaly_raw", row["is_anomaly"]):
            color = "#ff7f0e"
        else:
            color = "#1f77b4"
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color)
    if len(points) > 1:
        draw.line(points, fill="#444444", width=2)
    img.save(out)


def _metrics(results: list[dict]) -> dict:
    """Compute binary anomaly metrics from raw model decisions.

    Alarm debouncing intentionally delays alarm activation and clearing, so the
    model metrics use ``is_anomaly_raw``. The report separately lists debounced
    alarm windows as a systems behavior metric.
    """

    tp = sum(1 for r in results if r["label"] == "anomaly" and r.get("is_anomaly_raw", r["is_anomaly"]))
    tn = sum(1 for r in results if r["label"] == "normal" and not r.get("is_anomaly_raw", r["is_anomaly"]))
    fp = sum(1 for r in results if r["label"] == "normal" and r.get("is_anomaly_raw", r["is_anomaly"]))
    fn = sum(1 for r in results if r["label"] == "anomaly" and not r.get("is_anomaly_raw", r["is_anomaly"]))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def render_markdown(
    results: list[dict],
    figure_path: str | Path,
    *,
    model_backend: str = "centroid",
    model_path: str | Path = "artifacts/audio_model.json",
    feature_names: list[str] | None = None,
) -> str:
    """Render an audio anomaly report."""

    anomalies = sum(1 for row in results if row.get("is_anomaly_raw", row["is_anomaly"]))
    alarms = sum(1 for row in results if row.get("is_alarm", row["is_anomaly"]))
    metrics = _metrics(results)
    feature_ms = sum(float(r.get("feature_ms", 0.0)) for r in results) / max(1, len(results))
    inference_ms = sum(float(r.get("inference_ms", 0.0)) for r in results) / max(1, len(results))
    clips = sum(1 for row in results if row.get("clip_path"))
    figure_name = Path(figure_path).name
    lines = [
        "# Edge Audio Anomaly Report",
        "",
        f"- Windows analyzed: `{len(results)}`",
        f"- Raw anomaly windows: `{anomalies}`",
        f"- Debounced alarm windows: `{alarms}`",
        f"- Saved anomaly clips: `{clips}`",
        f"- Raw model precision / recall / F1: `{metrics['precision']:.3f}` / `{metrics['recall']:.3f}` / `{metrics['f1']:.3f}`",
        f"- Avg feature latency: `{feature_ms:.4f} ms`",
        f"- Avg inference latency: `{inference_ms:.4f} ms`",
        f"- Model backend: `{model_backend}`",
        f"- Model path: `{Path(model_path).name}`",
        f"- Figure: `{figure_name}`",
        "",
        "![Audio score curve](audio_score_curve.png)",
        "",
        "## Confusion Matrix",
        "",
        "| TP | TN | FP | FN |",
        "|---:|---:|---:|---:|",
        f"| {metrics['tp']} | {metrics['tn']} | {metrics['fp']} | {metrics['fn']} |",
        "",
        "The demo data is synthetic, so these metrics validate the replay,",
        "windowing, feature, scoring, alarm debounce, storage, and report pipeline. Field accuracy",
        "should be measured again with real machine recordings from the target",
        "factory environment.",
        "",
        "## Recent Window Events",
        "",
        "| File | Window | Label | Score | Threshold | Raw | Alarm | State | Clip |",
        "|---|---:|---|---:|---:|---|---|---|---|",
    ]
    for row in results[:60]:
        lines.append(
            f"| {Path(row.get('source', row.get('path', 'unknown'))).name} | {row.get('window_index', 0)} | "
            f"{row['label']} | {row['score']:.3f} | {row['threshold']:.3f} | "
            f"{row.get('is_anomaly_raw', row['is_anomaly'])} | {row.get('is_alarm', row['is_anomaly'])} | "
            f"{row.get('alarm_state', '')} | {Path(row.get('clip_path', '')).name if row.get('clip_path') else ''} |"
        )
    lines.extend(
        [
            "",
            "## Resource Budget",
            "",
            "- Feature vector: 9 float32 values, about 36 bytes.",
            f"- Feature names: `{', '.join(feature_names or [])}`",
            "- Default stream window: 0.25 s at 16 kHz, about 8 KB as int16 PCM.",
            "- Model parameters: mean + scale + threshold, about 76 bytes as float32.",
            "- Alarm debounce state: two counters and one boolean, suitable for a small service task.",
            "- SQLite buffers events locally so the service can tolerate network loss.",
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


def write_events_json(results: list[dict], path: str | Path) -> None:
    """Write portable JSON events for API/debug replay.

    Runtime events use absolute paths because that is easiest for a local
    service process to reopen files. A report artifact is different: it should
    be readable after the project is moved or pushed to Git. Before writing the
    JSON file, convert paths under the project root to relative strings while
    leaving external paths untouched.
    """

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    project_root = out.resolve().parents[1]

    def portable_path(value: str) -> str:
        if not value:
            return ""
        try:
            return str(Path(value).resolve().relative_to(project_root))
        except ValueError:
            return value

    portable_results: list[dict] = []
    for row in results:
        item = dict(row)
        item["source"] = portable_path(str(item.get("source", "")))
        item["clip_path"] = portable_path(str(item.get("clip_path", "")))
        portable_results.append(item)

    out.write_text(json.dumps(portable_results, indent=2), encoding="utf-8")


def write_report(
    results: list[dict],
    report_path: str | Path,
    figure_path: str | Path,
    events_json: str | Path | None = None,
    *,
    model_backend: str = "centroid",
    model_path: str | Path = "artifacts/audio_model.json",
    feature_names: list[str] | None = None,
) -> None:
    write_score_figure(results, figure_path)
    if events_json is not None:
        write_events_json(results, events_json)
    report = Path(report_path)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        render_markdown(
            results,
            figure_path,
            model_backend=model_backend,
            model_path=model_path,
            feature_names=feature_names,
        ),
        encoding="utf-8",
    )
