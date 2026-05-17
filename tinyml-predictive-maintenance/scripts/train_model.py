from __future__ import annotations

"""Train the hardware-free anomaly detector.

This script is intentionally small and explicit because it is the first command
a reviewer or future agent will run. It generates normal motor windows from the
simulator, extracts the same features used by the runtime node, and saves a tiny
JSON model that can be benchmarked by the sibling ``edgebench`` project.
"""

import argparse
import sys
from pathlib import Path

# The project is not installed as a package during quick demos, so add ``src``
# to the import path. This keeps the command runnable with plain:
#   python scripts/train_model.py
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tpm.config import load_config
from tpm.features import FEATURE_NAMES, FeatureConfig, extract_features, vectorize
from tpm.model import CentroidAnomalyDetector
from tpm.signal_sim import MotorSignalSimulator, SignalConfig


def main() -> None:
    """Parse CLI arguments, train on normal windows, and save the model."""

    parser = argparse.ArgumentParser(description="Train a tiny anomaly detector on normal motor vibration.")
    parser.add_argument("--config", default="configs/default.json", help="JSON config file with project defaults.")
    parser.add_argument("--out", help="Output model JSON path. Overrides config model_path.")
    parser.add_argument("--windows", type=int, help="Normal training windows. Overrides config train_windows.")
    parser.add_argument("--window-size", type=int, help="Samples per window. Overrides config window_size.")
    parser.add_argument("--sample-rate", type=int, help="Sample rate in Hz. Overrides config sample_rate_hz.")
    args = parser.parse_args()

    # Load project defaults first, then let explicit CLI arguments override
    # them. This makes scripted demos reproducible while keeping ad-hoc
    # experiments convenient.
    cfg = load_config(args.config)
    out_path = args.out if args.out is not None else cfg.model_path
    train_windows = args.windows if args.windows is not None else cfg.train_windows
    window_size = args.window_size if args.window_size is not None else cfg.window_size
    sample_rate = args.sample_rate if args.sample_rate is not None else cfg.sample_rate_hz

    simulator = MotorSignalSimulator(SignalConfig(sample_rate_hz=sample_rate))
    feature_cfg = FeatureConfig(sample_rate_hz=sample_rate)
    vectors = []

    # Train only on normal data. At runtime, the detector flags windows that are
    # too far from this learned normal profile. That is a common strategy when
    # fault data is rare or unavailable.
    for _ in range(train_windows):
        samples = simulator.read(window_size, "normal")
        vectors.append(vectorize(extract_features(samples, feature_cfg)))

    model = CentroidAnomalyDetector.train(vectors, FEATURE_NAMES)
    model.save(out_path)

    # Print the footprint because model size is an important embedded metric,
    # even for this JSON prototype.
    footprint = Path(out_path).stat().st_size
    print(f"saved model: {out_path}")
    print(f"features: {len(FEATURE_NAMES)}")
    print(f"threshold: {model.threshold:.3f}")
    print(f"training windows: {train_windows}")
    print(f"sample rate/window: {sample_rate} Hz / {window_size} samples")
    print(f"serialized footprint: {footprint} bytes")


if __name__ == "__main__":
    # Keep all work inside main() so future tests can import this file without
    # accidentally training a model.
    main()
