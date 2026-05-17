from __future__ import annotations

"""Run the portfolio-oriented TinyML predictive-maintenance demo.

This script is intentionally different from ``smoke-test.ps1``:

* ``smoke-test.ps1`` answers "does the engineering pipeline still work?"
* this script answers "can an interviewer run the project and see the story?"

The commands are still ordinary project scripts, not hidden APIs. That means a
future maintainer can copy any printed command and run/debug it independently.
"""

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tpm.portfolio import PortfolioPaths, write_portfolio_summary


@dataclass(frozen=True)
class DemoCommand:
    """A command plus the human reason it exists in the portfolio demo."""

    name: str
    args: list[str]
    reason: str


def _run(command: DemoCommand) -> None:
    """Execute one demo command and stream output to the console.

    ``check=True`` is deliberate: the demo should stop at the first broken
    stage so the user does not accidentally present stale reports.
    """

    pretty = " ".join(command.args)
    print(f"\n== {command.name} ==", flush=True)
    print(command.reason, flush=True)
    print(f"$ {pretty}", flush=True)
    subprocess.run(command.args, cwd=ROOT, check=True)


def _commands(quick: bool) -> list[DemoCommand]:
    """Build the ordered command list for either quick or full demo mode."""

    python = sys.executable
    train_windows = "80" if quick else "160"
    duration = "2" if quick else "4"
    # Evaluation is cheap, so even ``--quick`` keeps 40 windows per state. That
    # matches the README/test-plan numbers and avoids committing two subtly
    # different report shapes.
    eval_windows = "40"
    fixed_windows = "20" if quick else "80"

    return [
        DemoCommand(
            "Train synthetic centroid model",
            [python, "scripts/train_model.py", "--out", "artifacts/model.json", "--windows", train_windows],
            "Learns the normal vibration profile used by the simulated node and reports model footprint.",
        ),
        DemoCommand(
            "Run synthetic node telemetry",
            [
                python,
                "scripts/run_simulated_node.py",
                "--duration",
                duration,
                "--telemetry",
                "runs/telemetry.jsonl",
            ],
            "Produces the score/alarm timeline used in the README and portfolio figures.",
        ),
        DemoCommand(
            "Run CSV replay telemetry",
            [
                python,
                "scripts/run_simulated_node.py",
                "--source",
                "csv",
                "--input",
                "data/examples/vibration_demo.csv",
                "--telemetry",
                "runs/csv_telemetry.jsonl",
            ],
            "Shows that the node can replay exported sensor samples instead of only synthetic data.",
        ),
        DemoCommand(
            "Evaluate synthetic detector",
            [
                python,
                "scripts/evaluate_model.py",
                "--model",
                "artifacts/model.json",
                "--windows-per-state",
                eval_windows,
            ],
            "Refreshes reports/evaluation.json and reports/evaluation.md for the result snapshot.",
        ),
        DemoCommand(
            "Generate fixed-point drift report",
            [
                python,
                "scripts/fixed_point_report.py",
                "--model",
                "artifacts/model.json",
                "--windows-per-state",
                fixed_windows,
            ],
            "Compares float centroid scores against Q-format simulated parameters for MCU planning.",
        ),
        DemoCommand(
            "Export float and fixed C parameters",
            [
                python,
                "scripts/export_model_to_c.py",
                "--model",
                "artifacts/model.json",
                "--out",
                "firmware/model_params.h",
                "--fixed-out",
                "firmware/model_params_fixed.h",
            ],
            "Produces both float and Q24.8 C headers used by firmware parity tests.",
        ),
        DemoCommand(
            "Generate MCU resource budget",
            [
                python,
                "scripts/mcu_resource_report.py",
                "--model",
                "artifacts/model.json",
            ],
            "Summarizes model bytes, buffer bytes, telemetry estimate, and fixed-point validation.",
        ),
        DemoCommand(
            "Prepare PHM2008-shape sample",
            [
                python,
                "scripts/prepare_phm2008.py",
                "synthetic",
                "--out",
                "data/phm2008_sample/train_FD001.txt",
                "--units",
                "12",
                "--cycles",
                "180",
                "--sensors",
                "6",
                "--seed",
                "2027",
            ],
            "Creates a small offline C-MAPSS-style file so the harder-dataset demo is reproducible.",
        ),
        DemoCommand(
            "Compare PHM2008 degradation windows",
            [python, "scripts/compare_phm2008.py", "--data-root", "data/phm2008_sample"],
            "Runs the multivariate gradual-degradation comparison without downloading large data.",
        ),
        DemoCommand(
            "Generate static figures",
            [python, "scripts/generate_figures.py"],
            "Refreshes README/portfolio PNGs, including the pipeline and PHM2008 comparison figure.",
        ),
    ]


def main() -> None:
    """Parse CLI flags, run demo stages, and write the portfolio summary."""

    parser = argparse.ArgumentParser(description="Run the portfolio demo and produce reports/portfolio_summary.md.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use shorter synthetic runs for a fast laptop demo. No external network is required.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only regenerate reports/portfolio_summary.md from existing JSON/figures.",
    )
    args = parser.parse_args()

    # The generated directories are ignored by Git where appropriate, but they
    # must exist before subprocesses start writing model files and telemetry.
    (ROOT / "artifacts").mkdir(exist_ok=True)
    (ROOT / "runs").mkdir(exist_ok=True)
    (ROOT / "reports").mkdir(exist_ok=True)

    if not args.summary_only:
        for command in _commands(quick=args.quick):
            _run(command)

    write_portfolio_summary(
        PortfolioPaths(
            reports_dir=ROOT / "reports",
            evaluation_json=ROOT / "reports/evaluation.json",
            fixed_point_json=ROOT / "reports/fixed_point_report.json",
            phm2008_json=ROOT / "reports/phm2008_comparison.json",
            cwru_json=ROOT / "reports/cwru_comparison.json",
            quantization_json=ROOT / "reports/quantization_report.json",
            mcu_resource_json=ROOT / "reports/mcu_resource_budget.json",
            output_md=ROOT / "reports/portfolio_summary.md",
        )
    )
    print("\nwrote reports/portfolio_summary.md")
    print("Open README.md first, then reports/portfolio_summary.md for the detailed demo index.")


if __name__ == "__main__":
    main()
