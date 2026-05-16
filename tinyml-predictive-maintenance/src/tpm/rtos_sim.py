from __future__ import annotations

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

import time
from dataclasses import dataclass
from pathlib import Path

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

    @property
    def window_s(self) -> float:
        """Duration represented by one sample window."""

        return self.window_size / self.sample_rate_hz


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
    simulator = MotorSignalSimulator(SignalConfig(sample_rate_hz=config.sample_rate_hz))
    features_cfg = FeatureConfig(sample_rate_hz=config.sample_rate_hz)
    sink = JsonlTelemetrySink(telemetry_path)

    for seq, state in enumerate(state_schedule(config.duration_s, config.window_s)):
        # Sensor task: collect exactly one sliding/fixed window. In a real MCU
        # this would likely block on a DMA buffer or sensor interrupt.
        frame = {
            "seq": seq,
            "state": state,
            "samples": simulator.read(config.window_size, state),
            "ts": time.time(),
        }

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

        # Communication task: JSONL is used as a local stand-in for MQTT. Each
        # line is a complete telemetry message, so it can be tailed, replayed,
        # plotted, or converted to MQTT later.
        payload = {
            "seq": seq,
            "timestamp": frame["ts"],
            "true_state": state,
            "score": result["score"],
            "threshold": result["threshold"],
            "is_anomaly": result["is_anomaly"],
            "feature_ms": feature_ms,
            "inference_ms": inference_ms,
            "features": feature_map,
        }
        sink.publish(payload)
        label = "ALARM" if payload["is_anomaly"] else "OK"
        print(
            f"{payload['seq']:04d} {label:<5} state={payload['true_state']:<9} "
            f"score={payload['score']:.2f} infer={payload['inference_ms']:.3f}ms"
        )

        # The sleep is scaled down so demos finish quickly. It still preserves
        # the idea that windows arrive periodically instead of all at once.
        time.sleep(config.window_s * 0.1)
