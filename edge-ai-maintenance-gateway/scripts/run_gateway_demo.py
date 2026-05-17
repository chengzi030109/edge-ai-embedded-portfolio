from __future__ import annotations

"""Run the hardware-free maintenance gateway demo."""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from edge_gateway.config import load_config
from edge_gateway.replay import replay_jsonl
from edge_gateway.report import write_reports
from edge_gateway.storage import connect, init_db, summary


def _fallback_telemetry(path: Path) -> None:
    """Create a tiny telemetry file if the TinyML project has not run yet."""

    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    now = time.time()
    for seq, score in enumerate([1.2, 2.1, 7.8, 8.5, 3.0]):
        rows.append(
            {
                "seq": seq,
                "timestamp": now + seq,
                "true_state": "bearing" if score > 6 else "normal",
                "score": score,
                "threshold": 6.0,
                "is_anomaly_raw": score > 6,
                "is_alarm": seq >= 3,
                "alarm_state": "alarm" if seq >= 3 else "normal",
                "feature_ms": 0.2,
                "inference_ms": 0.02,
                "features": {"rms": score / 10.0},
            }
        )
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def main() -> None:
    cfg = load_config(ROOT / "configs/default.toml")
    if not cfg.telemetry_path.exists():
        _fallback_telemetry(cfg.telemetry_path)
    # Keep the portfolio demo deterministic. A deployed gateway would keep its
    # SQLite buffer, but the local demo should not double-count telemetry when
    # smoke-test.ps1 is run repeatedly.
    cfg.database_path.unlink(missing_ok=True)
    conn = connect(cfg.database_path)
    init_db(conn)
    imported = replay_jsonl(conn, cfg.telemetry_path, cfg.device_id)
    write_reports(conn, cfg.report_path, cfg.dashboard_path)
    stats = summary(conn)
    print(f"imported telemetry rows: {imported}")
    print(f"database: {cfg.database_path}")
    print(f"report: {cfg.report_path}")
    print(f"dashboard: {cfg.dashboard_path}")
    print(f"alarms: {stats['alarm_event_count']}")


if __name__ == "__main__":
    main()
