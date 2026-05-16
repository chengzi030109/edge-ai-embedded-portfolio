"""Smoke test for the benchmark runner.

The goal is not to verify NumPy math exhaustively. The test protects the CLI
core contract: a backend can be benchmarked and the report includes latency and
throughput fields.
"""

from edgebench.backends import BuiltinCentroidBackend
from edgebench.runner import BenchmarkConfig, run_benchmark


def test_run_benchmark_returns_latency_metrics():
    # Use a tiny repeat count so the test stays fast in CI or on a low-power
    # embedded Linux board.
    report = run_benchmark(
        BuiltinCentroidBackend(input_size=4),
        BenchmarkConfig(input_size=4, warmup=2, repeat=5, seed=1),
    )

    # These assertions check the report shape consumed by Markdown rendering and
    # future portfolio automation.
    assert report["backend"] == "builtin-centroid"
    assert report["latency_ms"]["avg"] >= 0
    assert report["throughput_fps"] > 0
