"""Tests for the portfolio Markdown summary renderer.

The portfolio demo itself runs several scripts and can take longer than a unit
test should. These tests exercise the reusable summary logic with tiny JSON
fixtures so regressions in the interview-facing report are caught quickly.
"""

import json

from tpm.portfolio import PortfolioPaths, write_portfolio_summary


def _write_json(path, payload):
    """Write a compact JSON fixture used by the summary renderer."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_portfolio_summary_collects_metrics_and_figures(tmp_path):
    reports = tmp_path / "reports"
    figures = reports / "figures"
    figures.mkdir(parents=True)
    (figures / "portfolio_pipeline.png").write_bytes(b"fake png")
    (figures / "phm2008_comparison.png").write_bytes(b"fake png")

    _write_json(
        reports / "evaluation.json",
        {"metrics": {"accuracy": 0.975, "f1": 0.9831}},
    )
    _write_json(
        reports / "fixed_point_report.json",
        {"decision_mismatches": 0, "score_error": {"mean_abs": 0.1}},
    )
    _write_json(
        reports / "phm2008_comparison.json",
        {
            "models": [
                {
                    "name": "CentroidAnomalyDetector",
                    "metrics": {"f1": 0.96, "false_positive_rate": 0.27},
                }
            ]
        },
    )
    _write_json(
        reports / "mcu_resource_budget.json",
        {"model_parameters": {"fixed_q24_8_bytes": 84}},
    )

    markdown = write_portfolio_summary(
        PortfolioPaths(
            reports_dir=reports,
            evaluation_json=reports / "evaluation.json",
            fixed_point_json=reports / "fixed_point_report.json",
            phm2008_json=reports / "phm2008_comparison.json",
            cwru_json=reports / "cwru_comparison.json",
            quantization_json=reports / "quantization_report.json",
            mcu_resource_json=reports / "mcu_resource_budget.json",
            output_md=reports / "portfolio_summary.md",
        )
    )

    assert "Synthetic" in markdown
    assert "PHM2008" in markdown
    assert "Fixed-point" in markdown
    assert "Fixed C inference parity" in markdown
    assert "Resume Bullets" in markdown
    assert "figures/portfolio_pipeline.png" in markdown
    assert "figures/phm2008_comparison.png" in markdown
    assert (reports / "portfolio_summary.md").exists()
