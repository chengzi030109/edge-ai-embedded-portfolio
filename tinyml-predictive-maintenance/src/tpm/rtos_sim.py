"""RTOS-style node loop for the hardware-free maintenance prototype.

The code is synchronous on purpose. Earlier versions used asyncio, but this
project must run reliably on the user's Windows environment and still preserve
the embedded architecture story. The loop is written as sequential stages that
map directly to RTOS tasks:

1. sensor stage: read a fixed-size sample window
2. feature stage: extract MCU-friendly signal features
3. inference stage: run tiny anomaly detection
4. communication stage: publish telemetry

When porting to FreeRTOS/Zephyr, these stages can become separate tasks with
queues between them.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .alarm import AlarmDebouncer
from .datasets.csv_replay import load_csv_signal
from .features import FeatureConfig, extract_features, vectorize
from .model import CentroidAnomalyDetector
from .signal_sim import MotorSignalSimulator, SignalConfig, state_schedule
from .telemetry import JsonlTelemetrySink


@dataclass(frozen=True)
class NodeConfig:
    """Runtime settings for the simulated edge node."""

    sample_rate_hz: int = 1600
    window_size: int = 256
    duration_s: float = 20.0
    source: str = "synthetic"
    input_path: str | None = None
    alarm_on_count: int = 3
    alarm_off_count: int = 5

    @property
    def window_s(self) -> float:
        """Duration represented by one sample window."""

        return self.window_size / self.sample_rate_hz


def _synthetic_frames(config: NodeConfig) -> Iterator[dict]:
    """Yield fixed-size windows from the synthetic motor simulator."""

    simulator = MotorSignalSimulator(SignalConfig(sample_rate_hz=config.sample_rate_hz))
    for seq, state in enumerate(state_schedule(config.duration_s, config.window_s)):
        yield {
            "seq": seq,
            "state": state,
            "samples": simulator.read(config.window_size, state),
            "ts": time.time(),
        }


def _csv_frames(config: NodeConfig) -> Iterator[dict]:
    """Yield fixed-size windows from a CSV replay file."""

    if config.input_path is None:
        raise ValueError("--input is required when --source csv")
    signal = load_csv_signal(config.input_path)
    for seq, window in enumerate(signal.windows(config.window_size)):
        yield {
            "seq": seq,
            "state": window.label,
            "samples": window.samples,
            "ts": window.timestamp if window.timestamp is not None else time.time(),
        }


def _frame_source(config: NodeConfig) -> Iterator[dict]:
    """Select the sensor source used by the node loop."""

    if config.source == "synthetic":
        return _synthetic_frames(config)
    if config.source == "csv":
        return _csv_frames(config)
    raise ValueError(f"unsupported node source: {config.source}")


def run_node(model_path: str | Path, telemetry_path: str | Path, config: NodeConfig) -> None:
    """Run the complete simulated embedded node.

    This function is the main application loop used by the CLI script. It loads
    a model, reads simulated sensor windows, extracts features, performs
    inference, and publishes telemetry to JSONL.
    """

    # These objects correspond to the resources an embedded application would
    # initialize during boot: model parameters, sensor driver, feature settings,
    # and communication/logging sink.
    model = CentroidAnomalyDetector.load(model_path)
    features_cfg = FeatureConfig(sample_rate_hz=config.sample_rate_hz)
    sink = JsonlTelemetrySink(telemetry_path)
    debouncer = AlarmDebouncer(on_count=config.alarm_on_count, off_count=config.alarm_off_count)

    for frame in _frame_source(config):
        # Sensor task: collect exactly one sliding/fixed window. In a real MCU
        # this would likely block on a DMA buffer or sensor interrupt.
        seq = int(frame["seq"])
        state = str(frame["state"])

        # Feature task: measure feature extraction time separately because this
        # is often one of the largest MCU CPU costs.
        feature_start = time.perf_counter()
        feature_map = extract_features(frame["samples"], features_cfg)
        feature_ms = (time.perf_counter() - feature_start) * 1000

        # Inference task: vectorize the named features and run the tiny model.
        # Keeping inference time visible makes this project useful for
        # edge-deployment and optimization discussions.
        inference_start = time.perf_counter()
        result = model.predict(vectorize(feature_map))
        inference_ms = (time.perf_counter() - inference_start) * 1000
        alarm = debouncer.update(bool(result["is_anomaly"]))

        # Communication task: JSONL is used as a local stand-in for MQTT. Each
        # line is a complete telemetry message, so it can be tailed, replayed,
        # plotted, or converted to MQTT later.
        payload = {
            "seq": seq,
            "timestamp": frame["ts"],
            "true_state": state,
            "score": result["score"],
            "threshold": result["threshold"],
            "is_anomaly_raw": result["is_anomaly"],
            "is_alarm": alarm.is_alarm,
            "alarm_state": alarm.alarm_state,
            "anomaly_streak": alarm.anomaly_streak,
            "normal_streak": alarm.normal_streak,
            # Backward-compatible alias for older scripts that still read
            # ``is_anomaly``. New code should prefer ``is_anomaly_raw``.
            "is_anomaly": result["is_anomaly"],
            "feature_ms": feature_ms,
            "inference_ms": inference_ms,
            "features": feature_map,
        }
        sink.publish(payload)
        label = "ALARM" if payload["is_alarm"] else str(payload["alarm_state"]).upper()
        print(
            f"{payload['seq']:04d} {label:<9} state={payload['true_state']:<9} "
            f"score={payload['score']:.2f} raw={payload['is_anomaly_raw']} "
            f"infer={payload['inference_ms']:.3f}ms"
        )

        # The sleep is scaled down so demos finish quickly. It still preserves
        # the idea that windows arrive periodically instead of all at once.
        if config.source == "synthetic":
            time.sleep(config.window_s * 0.1)
