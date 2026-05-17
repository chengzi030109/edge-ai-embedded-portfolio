from __future__ import annotations

"""SQLite persistence layer for edge telemetry and alarms."""

import json
import sqlite3
from pathlib import Path
from typing import Any


SCHEMA = """
CREATE TABLE IF NOT EXISTS devices (
  device_id TEXT PRIMARY KEY,
  last_seen REAL NOT NULL,
  last_alarm_state TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS telemetry (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  device_id TEXT NOT NULL,
  seq INTEGER NOT NULL,
  timestamp REAL NOT NULL,
  true_state TEXT NOT NULL,
  score REAL NOT NULL,
  threshold REAL NOT NULL,
  is_anomaly_raw INTEGER NOT NULL,
  is_alarm INTEGER NOT NULL,
  alarm_state TEXT NOT NULL,
  feature_ms REAL NOT NULL,
  inference_ms REAL NOT NULL,
  features_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS alarm_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  device_id TEXT NOT NULL,
  seq INTEGER NOT NULL,
  timestamp REAL NOT NULL,
  alarm_state TEXT NOT NULL,
  score REAL NOT NULL,
  threshold REAL NOT NULL
);
"""


def connect(path: str | Path) -> sqlite3.Connection:
    """Open a SQLite connection and ensure parent directories exist."""

    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    """Create all tables used by the gateway."""

    conn.executescript(SCHEMA)
    conn.commit()


def ingest_telemetry(conn: sqlite3.Connection, payload: dict[str, Any], device_id: str) -> None:
    """Persist one telemetry message and create alarm events when needed."""

    alarm_state = str(payload.get("alarm_state", "unknown"))
    timestamp = float(payload.get("timestamp", 0.0))
    seq = int(payload.get("seq", 0))
    score = float(payload.get("score", 0.0))
    threshold = float(payload.get("threshold", 0.0))
    is_alarm = bool(payload.get("is_alarm", payload.get("is_anomaly", False)))
    is_raw = bool(payload.get("is_anomaly_raw", payload.get("is_anomaly", False)))

    conn.execute(
        """
        INSERT OR REPLACE INTO devices(device_id, last_seen, last_alarm_state)
        VALUES (?, ?, ?)
        """,
        (device_id, timestamp, alarm_state),
    )
    conn.execute(
        """
        INSERT INTO telemetry(
          device_id, seq, timestamp, true_state, score, threshold,
          is_anomaly_raw, is_alarm, alarm_state, feature_ms, inference_ms,
          features_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            device_id,
            seq,
            timestamp,
            str(payload.get("true_state", "unknown")),
            score,
            threshold,
            int(is_raw),
            int(is_alarm),
            alarm_state,
            float(payload.get("feature_ms", 0.0)),
            float(payload.get("inference_ms", 0.0)),
            json.dumps(payload.get("features", {}), ensure_ascii=True),
        ),
    )
    if is_alarm or alarm_state in {"alarm", "pending", "recovering"}:
        conn.execute(
            """
            INSERT INTO alarm_events(device_id, seq, timestamp, alarm_state, score, threshold)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (device_id, seq, timestamp, alarm_state, score, threshold),
        )
    conn.commit()


def list_devices(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Return all known devices sorted by id."""

    rows = conn.execute("SELECT * FROM devices ORDER BY device_id").fetchall()
    return [dict(row) for row in rows]


def list_telemetry(conn: sqlite3.Connection, device_id: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
    """Return recent telemetry rows."""

    if device_id:
        rows = conn.execute(
            "SELECT * FROM telemetry WHERE device_id = ? ORDER BY id DESC LIMIT ?",
            (device_id, limit),
        ).fetchall()
    else:
        rows = conn.execute("SELECT * FROM telemetry ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(row) for row in rows]


def list_alarms(conn: sqlite3.Connection, limit: int = 50) -> list[dict[str, Any]]:
    """Return recent alarm events."""

    rows = conn.execute("SELECT * FROM alarm_events ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(row) for row in rows]


def summary(conn: sqlite3.Connection) -> dict[str, Any]:
    """Build a compact gateway status summary."""

    total = conn.execute("SELECT COUNT(*) AS n FROM telemetry").fetchone()["n"]
    alarms = conn.execute("SELECT COUNT(*) AS n FROM alarm_events").fetchone()["n"]
    latest = conn.execute("SELECT * FROM telemetry ORDER BY id DESC LIMIT 1").fetchone()
    avg_latency = conn.execute("SELECT AVG(feature_ms + inference_ms) AS ms FROM telemetry").fetchone()["ms"]
    return {
        "telemetry_count": int(total),
        "alarm_event_count": int(alarms),
        "avg_pipeline_ms": float(avg_latency or 0.0),
        "latest": dict(latest) if latest else None,
    }

