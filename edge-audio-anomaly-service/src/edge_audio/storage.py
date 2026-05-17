from __future__ import annotations

"""SQLite event storage for the audio anomaly service."""

import json
import sqlite3
from pathlib import Path


SCHEMA = """
CREATE TABLE IF NOT EXISTS audio_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  source TEXT NOT NULL,
  label TEXT NOT NULL,
  window_index INTEGER NOT NULL,
  start_s REAL NOT NULL,
  end_s REAL NOT NULL,
  score REAL NOT NULL,
  threshold REAL NOT NULL,
  is_anomaly_raw INTEGER NOT NULL,
  is_anomaly INTEGER NOT NULL,
  is_alarm INTEGER NOT NULL,
  alarm_state TEXT NOT NULL,
  alarm_bad_streak INTEGER NOT NULL,
  alarm_good_streak INTEGER NOT NULL,
  feature_ms REAL NOT NULL,
  inference_ms REAL NOT NULL,
  clip_path TEXT NOT NULL,
  features_json TEXT NOT NULL
);
"""


def connect(path: str | Path) -> sqlite3.Connection:
    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


def insert_events(conn: sqlite3.Connection, events: list[dict]) -> None:
    """Insert analyzed window events into SQLite."""

    conn.executemany(
        """
        INSERT INTO audio_events(
          source, label, window_index, start_s, end_s, score, threshold,
          is_anomaly_raw, is_anomaly, is_alarm, alarm_state,
          alarm_bad_streak, alarm_good_streak, feature_ms, inference_ms,
          clip_path, features_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                event["source"],
                event["label"],
                int(event["window_index"]),
                float(event["start_s"]),
                float(event["end_s"]),
                float(event["score"]),
                float(event["threshold"]),
                int(event.get("is_anomaly_raw", event["is_anomaly"])),
                int(event["is_anomaly"]),
                int(event.get("is_alarm", event["is_anomaly"])),
                str(event.get("alarm_state", "alarm" if event["is_anomaly"] else "normal")),
                int(event.get("alarm_bad_streak", 0)),
                int(event.get("alarm_good_streak", 0)),
                float(event["feature_ms"]),
                float(event["inference_ms"]),
                event.get("clip_path", ""),
                json.dumps(event.get("features", {}), ensure_ascii=True),
            )
            for event in events
        ],
    )
    conn.commit()


def _row_to_event(row: sqlite3.Row) -> dict:
    """Convert one SQLite row back to the public event dictionary shape."""

    event = dict(row)
    event["is_anomaly_raw"] = bool(event["is_anomaly_raw"])
    event["is_anomaly"] = bool(event["is_anomaly"])
    event["is_alarm"] = bool(event["is_alarm"])
    event["features"] = json.loads(event.pop("features_json") or "{}")
    return event


def list_events(conn: sqlite3.Connection, limit: int = 50) -> list[dict]:
    rows = conn.execute("SELECT * FROM audio_events ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [_row_to_event(row) for row in rows]


def summary(conn: sqlite3.Connection) -> dict:
    row = conn.execute(
        """
        SELECT COUNT(*) AS n,
               SUM(is_anomaly_raw) AS raw_anomalies,
               SUM(is_alarm) AS alarms,
               AVG(feature_ms) AS feature_ms,
               AVG(inference_ms) AS inference_ms
        FROM audio_events
        """
    ).fetchone()
    return {
        "event_count": int(row["n"] or 0),
        "raw_anomaly_count": int(row["raw_anomalies"] or 0),
        "alarm_count": int(row["alarms"] or 0),
        "anomaly_count": int(row["raw_anomalies"] or 0),
        "feature_ms_avg": float(row["feature_ms"] or 0.0),
        "inference_ms_avg": float(row["inference_ms"] or 0.0),
    }
