from __future__ import annotations

"""Inspect the local SQLite event buffer from the command line.

Embedded Linux applications need boring but useful operator tools. This script
answers the practical questions an engineer asks on a board over SSH:

- Is the service writing events?
- How many raw anomalies and debounced alarms were observed?
- What were the latest high-score windows?

It deliberately uses only the standard library plus this project's storage
module, so it can run on a minimal target image without pulling in FastAPI.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from edge_audio.config import load_config
from edge_audio.storage import connect, list_events, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query edge audio anomaly events")
    parser.add_argument("--config", default=str(ROOT / "configs/default.toml"), help="Path to default.toml")
    parser.add_argument("--limit", type=int, default=10, help="Number of newest events to print")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON instead of a text table")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    conn = connect(cfg.database_path)
    stats = summary(conn)
    events = list_events(conn, limit=args.limit)

    if args.json:
        print(json.dumps({"summary": stats, "events": events}, indent=2))
        return

    print("Edge audio event buffer")
    print(f"database: {cfg.database_path}")
    print(
        "summary: "
        f"events={stats['event_count']} "
        f"raw_anomalies={stats['raw_anomaly_count']} "
        f"alarms={stats['alarm_count']} "
        f"feature_ms_avg={stats['feature_ms_avg']:.4f} "
        f"inference_ms_avg={stats['inference_ms_avg']:.4f}"
    )
    print()
    print("latest events:")
    print("id  state       raw  alarm  score     file")
    for event in events:
        source = Path(event["source"]).name
        print(
            f"{event['id']:<3} {event['alarm_state']:<10} "
            f"{str(event['is_anomaly_raw']):<4} {str(event['is_alarm']):<5} "
            f"{event['score']:<8.3f} {source}"
        )


if __name__ == "__main__":
    main()
