from __future__ import annotations

"""Command-line interface for EdgeBench."""

import argparse

from .backends import load_backend
from .report import load_json_report, save_markdown_report
from .runner import BenchmarkConfig, run_benchmark, save_json_report


def run_command(args: argparse.Namespace) -> None:
    """Handle ``edgebench run``.

    This command selects a backend, executes the benchmark, and writes a JSON
    report. It prints only the key numbers so terminal output stays readable.
    """

    backend = load_backend(args.model, args.builtin, args.input_size)
    report = run_benchmark(
        backend,
        BenchmarkConfig(
            input_size=args.input_size,
            warmup=args.warmup,
            repeat=args.repeat,
            seed=args.seed,
        ),
        model_path=args.model,
    )
    save_json_report(report, args.out)
    latency = report["latency_ms"]
    print(f"saved JSON report: {args.out}")
    print(f"backend={report['backend']} avg={latency['avg']:.6f}ms p95={latency['p95']:.6f}ms")


def report_command(args: argparse.Namespace) -> None:
    """Handle ``edgebench report`` by converting JSON to Markdown."""

    report = load_json_report(args.input)
    save_markdown_report(report, args.out)
    print(f"saved Markdown report: {args.out}")


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse command tree.

    Keeping parser construction in a separate function makes it easier for a
    future agent to add subcommands such as ``compare`` or ``batch``.
    """

    parser = argparse.ArgumentParser(prog="edgebench", description="Benchmark edge AI inference workloads.")
    sub = parser.add_subparsers(dest="command", required=True)

    # ``run`` is the main data-producing command.
    run = sub.add_parser("run", help="Run an inference benchmark.")
    run.add_argument("--model", help="Path to a supported model file.")
    run.add_argument("--builtin", choices=["centroid"], help="Use a built-in toy backend.")
    run.add_argument("--input-size", type=int, default=10)
    run.add_argument("--warmup", type=int, default=20)
    run.add_argument("--repeat", type=int, default=200)
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--out", default="runs/benchmark.json")
    run.set_defaults(func=run_command)

    # ``report`` is intentionally separate so reports can be regenerated after
    # changing formatting without rerunning a benchmark.
    rep = sub.add_parser("report", help="Render a Markdown report from JSON.")
    rep.add_argument("--input", required=True)
    rep.add_argument("--out", default="runs/benchmark.md")
    rep.set_defaults(func=report_command)
    return parser


def main() -> None:
    """CLI entry point used by ``python -m edgebench`` and console scripts."""

    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
