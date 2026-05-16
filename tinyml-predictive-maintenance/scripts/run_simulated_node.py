from __future__ import annotations

"""Run the simulated RTOS-style TinyML node from the command line."""

import argparse
from pathlib import Path
import sys

# Add ``src`` to the import path so users can run this script directly from the
# repository without installing the package in editable mode.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tpm.config import load_config
from tpm.rtos_sim import NodeConfig, run_node


def main() -> None:
    """Collect CLI options, build a NodeConfig, and execute the node loop."""

    parser = argparse.ArgumentParser(description="Run the RTOS-style simulated TinyML node.")
    parser.add_argument("--config", default="configs/default.json", help="JSON config file with project defaults.")
    parser.add_argument("--model", help="Model JSON path. Overrides config model_path.")
    parser.add_argument("--telemetry", help="Telemetry JSONL path. Overrides config telemetry_path.")
    parser.add_argument("--duration", type=float, help="Run duration in seconds. Overrides config duration_s.")
    parser.add_argument("--window-size", type=int, help="Samples per window. Overrides config window_size.")
    parser.add_argument("--sample-rate", type=int, help="Sample rate in Hz. Overrides config sample_rate_hz.")
    parser.add_argument("--append", action="store_true", help="Append to existing telemetry instead of overwriting it.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_path = args.model or cfg.model_path
    telemetry_path = args.telemetry or cfg.telemetry_path

    # Default to a clean telemetry file so repeated demos and smoke tests do not
    # mix old and new runs. ``--append`` is available for long-running log
    # collection experiments.
    if not args.append:
        Path(telemetry_path).unlink(missing_ok=True)

    config = NodeConfig(
        sample_rate_hz=args.sample_rate or cfg.sample_rate_hz,
        window_size=args.window_size or cfg.window_size,
        duration_s=args.duration or cfg.duration_s,
    )

    # ``run_node`` contains the staged sensor -> feature -> inference ->
    # telemetry pipeline. The script layer only handles user input/output.
    run_node(model_path, telemetry_path, config)
    print(f"telemetry written to: {telemetry_path}")


if __name__ == "__main__":
    # This guard prevents accidental execution if another tool imports the file
    # to inspect CLI defaults.
    main()
