from __future__ import annotations

"""Prepare or synthesize PHM 2008 / C-MAPSS style data.

The script intentionally does not auto-download large archives. It prints the
official references and can generate a tiny PHM-shape sample so tests and demos
run without external data.
"""

import argparse
from pathlib import Path

import numpy as np

OFFICIAL_LINKS = [
    "https://catalog.data.gov/dataset/phm-2008-challenge-d1f2b",
    "https://c3.ndc.nasa.gov/dashlink/resources/139/",
]


def cmd_manifest() -> None:
    """Print dataset references and expected local layout."""

    print("PHM 2008 / NASA C-MAPSS references:")
    for url in OFFICIAL_LINKS:
        print(f"  {url}")
    print()
    print("Expected local file for this project:")
    print("  data/phm2008_sample/train_FD001.txt")
    print()
    print("Columns: unit cycle setting_1 setting_2 setting_3 sensor_1 ... sensor_N")


def cmd_synthetic(out: Path, units: int, cycles: int, sensors: int, seed: int) -> None:
    """Generate a small C-MAPSS-shape degradation text file."""

    rng = np.random.default_rng(seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for unit in range(1, units + 1):
        unit_noise = rng.normal(0.0, 0.02, size=sensors)
        for cycle in range(1, cycles + 1):
            life = cycle / cycles
            settings = [
                rng.normal(0.0, 0.01),
                rng.normal(0.0, 0.01),
                rng.normal(100.0, 0.5),
            ]
            sensor_values = []
            for sensor_idx in range(sensors):
                base = 1.0 + 0.1 * sensor_idx + unit_noise[sensor_idx]
                # Keep the synthetic degradation intentionally subtle. The goal
                # is not to create another ceiling dataset like CWRU, but a
                # PHM-shape sample where gradual drift overlaps with normal
                # operating variation.
                degradation = max(0.0, life - 0.45) ** 1.4 * (0.18 + 0.015 * sensor_idx)
                noise = rng.normal(0.0, 0.06)
                sensor_values.append(base + degradation + noise)
            rows.append([unit, cycle, *settings, *sensor_values])
    np.savetxt(out, np.asarray(rows, dtype=np.float32), fmt="%.6f")
    print(f"wrote {out} ({len(rows)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare PHM2008/C-MAPSS data.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("manifest", help="Print official links and expected local layout.")
    syn = sub.add_parser("synthetic", help="Generate a small PHM-shape sample file.")
    syn.add_argument("--out", default="data/phm2008_sample/train_FD001.txt")
    syn.add_argument("--units", type=int, default=8)
    syn.add_argument("--cycles", type=int, default=160)
    syn.add_argument("--sensors", type=int, default=6)
    syn.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()

    if args.cmd == "manifest":
        cmd_manifest()
    elif args.cmd == "synthetic":
        cmd_synthetic(Path(args.out), args.units, args.cycles, args.sensors, args.seed)


if __name__ == "__main__":
    main()
