from __future__ import annotations

"""Benchmark execution logic.

This module owns the timing methodology. CLI parsing and report rendering live
elsewhere so this code can be reused by tests, future batch runners, or a web UI.
"""

import json
import platform
import statistics
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from .backends import Backend


@dataclass
class BenchmarkConfig:
    """Configuration for one benchmark run."""

    input_size: int
    warmup: int
    repeat: int
    seed: int


def percentile(values: list[float], pct: float) -> float:
    """Return a percentile using linear interpolation.

    Percentile latency matters for edge devices because users often feel tail
    latency more than average latency. p95/p99 are also common metrics in real
    deployment reports.
    """

    if not values:
        return 0.0
    ordered = sorted(values)
    rank = (len(ordered) - 1) * pct
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    weight = rank - lo
    return ordered[lo] * (1 - weight) + ordered[hi] * weight


def run_benchmark(backend: Backend, config: BenchmarkConfig, model_path: str | None = None) -> dict:
    """Run warmup + measured inference iterations and return a JSON-ready report."""

    # Deterministic random inputs make repeated benchmark runs comparable. For
    # real model support, this can later be extended to load sample .npy files.
    rng = np.random.default_rng(config.seed)
    samples = rng.normal(0.0, 1.0, size=(config.repeat + config.warmup, config.input_size)).astype(np.float32)

    # Warmup is excluded from metrics because the first few inferences may pay
    # one-time costs such as lazy imports, memory allocation, or backend setup.
    for i in range(config.warmup):
        backend.infer(samples[i])

    latencies_ms: list[float] = []
    for i in range(config.warmup, config.warmup + config.repeat):
        # perf_counter_ns gives high-resolution monotonic timing. Convert to ms
        # because model latency is usually discussed in milliseconds.
        start = time.perf_counter_ns()
        backend.infer(samples[i])
        elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000
        latencies_ms.append(elapsed_ms)

    total_ms = sum(latencies_ms)
    model_size_bytes = Path(model_path).stat().st_size if model_path else 0

    # The report is plain JSON-compatible data so it can be committed, diffed,
    # rendered as Markdown, or consumed by another tool.
    return {
        "backend": getattr(backend, "name", backend.__class__.__name__),
        "config": asdict(config),
        "latency_ms": {
            "avg": statistics.fmean(latencies_ms),
            "min": min(latencies_ms),
            "max": max(latencies_ms),
            "p50": percentile(latencies_ms, 0.50),
            "p95": percentile(latencies_ms, 0.95),
            "p99": percentile(latencies_ms, 0.99),
        },
        "throughput_fps": config.repeat / (total_ms / 1000.0) if total_ms > 0 else 0.0,
        "model_size_bytes": model_size_bytes,
        "system": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
    }


def save_json_report(report: dict, path: str | Path) -> None:
    """Write the benchmark report to disk."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
