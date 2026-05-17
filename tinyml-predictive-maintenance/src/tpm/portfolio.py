from __future__ import annotations

"""Portfolio report rendering utilities.

The command-line demo script is allowed to orchestrate training/evaluation, but
the Markdown summary should live in a normal importable module. Keeping the
rendering logic here makes it easy to unit test the "what do we show an
interviewer?" layer without running a full model-training pipeline.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PortfolioPaths:
    """Input and output locations used by the portfolio summary renderer.

    All paths are deliberately explicit instead of being hidden in globals.
    That makes tests simple: they can point the renderer at a temporary reports
    directory with small fake JSON files.
    """

    reports_dir: Path = Path("reports")
    evaluation_json: Path = Path("reports/evaluation.json")
    fixed_point_json: Path = Path("reports/fixed_point_report.json")
    phm2008_json: Path = Path("reports/phm2008_comparison.json")
    cwru_json: Path = Path("reports/cwru_comparison.json")
    quantization_json: Path = Path("reports/quantization_report.json")
    mcu_resource_json: Path = Path("reports/mcu_resource_budget.json")
    output_md: Path = Path("reports/portfolio_summary.md")


DEFAULT_FIGURES: tuple[tuple[str, str], ...] = (
    ("Portfolio pipeline", "figures/portfolio_pipeline.png"),
    ("Synthetic score curve", "figures/synthetic_score_curve.png"),
    ("Alarm debounce timeline", "figures/alarm_debounce_timeline.png"),
    ("CWRU model comparison", "figures/cwru_model_comparison.png"),
    ("Quantization size and latency", "figures/quantization_size_latency.png"),
    ("PHM2008 comparison", "figures/phm2008_comparison.png"),
)


def _load_optional_json(path: Path) -> dict[str, Any] | None:
    """Load a JSON report if it exists, otherwise return ``None``.

    Portfolio generation should be friendly during partial demos. For example,
    a local Windows machine may not be able to run the optional Torch/ONNX path,
    but the synthetic and fixed-point sections should still render.
    """

    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _metric(report: dict[str, Any] | None, name: str) -> float | None:
    """Read a metric from a standard report dictionary."""

    if report is None:
        return None
    value = report.get("metrics", {}).get(name)
    return float(value) if value is not None else None


def _best_model(report: dict[str, Any] | None, preferred: str = "CentroidAnomalyDetector") -> dict[str, Any] | None:
    """Pick the model row to summarize from a comparison report.

    The centroid detector is the embedded-friendly baseline and is therefore
    the default row used in the portfolio narrative. If a report does not
    contain that exact name, the first model is still useful for a partial
    summary.
    """

    if report is None:
        return None
    models = report.get("models") or []
    for model in models:
        if model.get("name") == preferred:
            return model
    return models[0] if models else None


def _fmt_float(value: float | None, digits: int = 4, missing: str = "not generated") -> str:
    """Format optional numeric values consistently for Markdown tables."""

    if value is None:
        return missing
    return f"{value:.{digits}f}"


def _figure_markdown(reports_dir: Path, figures: tuple[tuple[str, str], ...]) -> list[str]:
    """Render image links only for figures that exist on disk.

    ``portfolio_summary.md`` lives inside ``reports/``, so figure paths are
    relative to that directory. Missing images are listed as commands to run
    rather than causing a hard failure.
    """

    rows: list[str] = []
    missing: list[str] = []
    for title, rel_path in figures:
        figure_path = reports_dir / rel_path
        if figure_path.exists():
            rows.append(f"![{title}]({rel_path})")
            rows.append("")
        else:
            missing.append(rel_path)
    if missing:
        rows.append("Missing figures can be regenerated with `python scripts/generate_figures.py`:")
        rows.extend(f"- `{path}`" for path in missing)
        rows.append("")
    return rows


def render_portfolio_summary(
    evaluation: dict[str, Any] | None,
    fixed_point: dict[str, Any] | None,
    phm2008: dict[str, Any] | None,
    cwru: dict[str, Any] | None,
    quantization: dict[str, Any] | None,
    mcu_resource: dict[str, Any] | None,
    *,
    reports_dir: Path = Path("reports"),
    figures: tuple[tuple[str, str], ...] = DEFAULT_FIGURES,
) -> str:
    """Build a single Markdown summary for portfolio review.

    The tone is intentionally practical: a recruiter or interviewer can read
    the first screen for the result, then scroll down for embedded/TinyML depth.
    """

    synthetic_f1 = _metric(evaluation, "f1")
    synthetic_acc = _metric(evaluation, "accuracy")
    fixed_mismatches = None if fixed_point is None else fixed_point.get("decision_mismatches")
    fixed_integer_mismatches = None if fixed_point is None else fixed_point.get("integer_path_decision_mismatches")
    phm_model = _best_model(phm2008)
    cwru_model = _best_model(cwru)
    quant_formats = [] if quantization is None else quantization.get("formats", [])

    lines = [
        "# TinyML Predictive Maintenance Portfolio Summary",
        "",
        "This is the short, interview-facing index for the project. It connects",
        "the laptop prototype, the repeatable demo commands, and the MCU/TinyML",
        "migration story in one place.",
        "",
        "## Result Snapshot",
        "",
        "| Area | What to Look At | Current Result |",
        "|---|---|---:|",
        f"| Synthetic simulator | Normal-only anomaly detection | F1 `{_fmt_float(synthetic_f1)}`, accuracy `{_fmt_float(synthetic_acc)}` |",
        f"| Fixed-point simulation | Q-format centroid drift | `{fixed_mismatches if fixed_mismatches is not None else 'not generated'}` parameter mismatches, `{fixed_integer_mismatches if fixed_integer_mismatches is not None else 'not generated'}` integer-path mismatches |",
    ]

    if phm_model is not None:
        phm_metrics = phm_model.get("metrics", {})
        lines.append(
            "| PHM2008/C-MAPSS sample | Harder gradual degradation task | "
            f"F1 `{_fmt_float(phm_metrics.get('f1'))}`, FPR `{_fmt_float(phm_metrics.get('false_positive_rate'))}` |"
        )
    else:
        lines.append("| PHM2008/C-MAPSS sample | Harder gradual degradation task | not generated |")

    if cwru_model is not None:
        cwru_metrics = cwru_model.get("metrics", {})
        lines.append(
            "| CWRU bearing data | Real-data sanity check, ceiling warning | "
            f"F1 `{_fmt_float(cwru_metrics.get('f1'))}` |"
        )
    else:
        lines.append("| CWRU bearing data | Real-data sanity check, ceiling warning | optional report not generated |")

    lines.extend(
        [
            "| C inference parity | Firmware-facing float centroid path | covered by `tests/test_c_inference_parity.py` |",
            "| Fixed C inference parity | Q24.8 MCU-style path | covered by `tests/test_c_fixed_inference_parity.py` |",
            "",
            "## Demo Command",
            "",
            "```powershell",
            ".\\.venv\\Scripts\\python.exe scripts\\run_portfolio_demo.py --quick",
            "```",
            "",
            "The demo trains a fresh synthetic model, runs both synthetic and CSV replay",
            "telemetry, evaluates the model, refreshes fixed-point and PHM2008 reports,",
            "regenerates figures, and rewrites this summary.",
            "",
            "## Figures",
            "",
        ]
    )
    lines.extend(_figure_markdown(reports_dir, figures))

    lines.extend(
        [
            "## Engineering Read",
            "",
            "- The synthetic result proves the whole edge pipeline is wired correctly:",
            "  simulator/replay, windowing, FFT/statistical features, anomaly score,",
            "  debounce, telemetry, and reports.",
            "- CWRU is useful but close to a ceiling because seeded bearing faults are",
            "  strongly separated in RMS, kurtosis, and spectral-band features. In an",
            "  interview, treat CWRU as a validation dataset, not as the only claim.",
            "- PHM2008/C-MAPSS is the better difficulty story: multivariate engine",
            "  cycles, gradual degradation, and noisier false-positive behavior.",
            "- Fixed-point Q24.8 is now measured in two ways: parameter-only drift",
            "  and an integer C-shaped path with a matching C parity test.",
            "",
            "## Deployment Notes",
            "",
            f"- Quantization formats generated: `{len(quant_formats)}`"
            if quant_formats
            else "- Quantization formats generated: optional ONNX report not generated on this machine.",
            f"- MCU resource report: `{mcu_resource['model_parameters']['fixed_q24_8_bytes']}` bytes of Q24.8 parameters."
            if mcu_resource is not None
            else "- MCU resource report: not generated yet.",
            "- MCU boundary: replace simulator/CSV replay with sensor drivers, keep the",
            "  feature vector and detector contract stable.",
            "- Next firmware depth: CMSIS-DSP FFT, direct Q-format feature extraction,",
            "  FreeRTOS task split, and UART/MQTT telemetry on an embedded-Linux gateway.",
            "",
            "## Resume Bullets",
            "",
            "- Built a hardware-free TinyML predictive-maintenance demo with RTOS-style",
            "  sensor, feature, inference, alarm, and telemetry stages.",
            "- Implemented CSV replay, alarm debounce, fixed-point drift analysis, C",
            "  inference parity tests, Q24.8 integer inference, and MCU resource reports.",
            "- Evaluated the same embedded-friendly anomaly detector on synthetic",
            "  vibration, CWRU bearing data, and PHM2008/C-MAPSS-style degradation",
            "  windows, with honest discussion of dataset difficulty.",
            "",
            "## Follow-Up Reports",
            "",
            "- [`evaluation.md`](evaluation.md)",
            "- [`fixed_point_report.md`](fixed_point_report.md)",
            "- [`phm2008_comparison.md`](phm2008_comparison.md)",
            "- [`cwru_comparison.md`](cwru_comparison.md)",
            "- [`quantization_report.md`](quantization_report.md)",
            "- [`mcu_resource_budget.md`](mcu_resource_budget.md)",
            "- [`../docs/mcu-migration.md`](../docs/mcu-migration.md)",
            "- [`../docs/interview-notes.md`](../docs/interview-notes.md)",
            "- [`../docs/resume-bullets.md`](../docs/resume-bullets.md)",
            "",
        ]
    )
    return "\n".join(lines)


def write_portfolio_summary(paths: PortfolioPaths = PortfolioPaths()) -> str:
    """Load available reports, render the summary, and write it to disk."""

    evaluation = _load_optional_json(paths.evaluation_json)
    fixed_point = _load_optional_json(paths.fixed_point_json)
    phm2008 = _load_optional_json(paths.phm2008_json)
    cwru = _load_optional_json(paths.cwru_json)
    quantization = _load_optional_json(paths.quantization_json)
    mcu_resource = _load_optional_json(paths.mcu_resource_json)
    markdown = render_portfolio_summary(
        evaluation,
        fixed_point,
        phm2008,
        cwru,
        quantization,
        mcu_resource,
        reports_dir=paths.reports_dir,
    )
    paths.output_md.parent.mkdir(parents=True, exist_ok=True)
    paths.output_md.write_text(markdown, encoding="utf-8")
    return markdown
