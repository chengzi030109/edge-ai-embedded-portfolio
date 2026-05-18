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
  features_json TEXT NOT NULL,
  uploaded INTEGER NOT NULL DEFAULT 0,
  ack INTEGER NOT NULL DEFAULT 0
);
"""


def connect(path: str | Path) -> sqlite3.Connection:
    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    # FastAPI runs normal def endpoints in a thread pool. The API factory keeps
    # one SQLite connection for this tiny edge service, so the connection must
    # be usable from those worker threads. The API layer serializes access with
    # a lock; scripts still behave the same as before.
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    _ensure_column(conn, "uploaded", "INTEGER NOT NULL DEFAULT 0")
    _ensure_column(conn, "ack", "INTEGER NOT NULL DEFAULT 0")
    conn.commit()


def _ensure_column(conn: sqlite3.Connection, name: str, ddl: str) -> None:
    """Add a column when an older SQLite database was created before it existed."""

    columns = {row["name"] for row in conn.execute("PRAGMA table_info(audio_events)").fetchall()}
    if name not in columns:
        conn.execute(f"ALTER TABLE audio_events ADD COLUMN {name} {ddl}")


def insert_events(conn: sqlite3.Connection, events: list[dict]) -> None:
    """Insert analyzed window events into SQLite."""

    conn.executemany(
        """
        INSERT INTO audio_events(
          source, label, window_index, start_s, end_s, score, threshold,
          is_anomaly_raw, is_anomaly, is_alarm, alarm_state,
          alarm_bad_streak, alarm_good_streak, feature_ms, inference_ms,
          clip_path, features_json, uploaded, ack
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                int(event.get("uploaded", False)),
                int(event.get("ack", False)),
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
    event["uploaded"] = bool(event.get("uploaded", 0))
    event["ack"] = bool(event.get("ack", 0))
    event["features"] = json.loads(event.pop("features_json") or "{}")
    return event


def list_events(conn: sqlite3.Connection, limit: int = 50) -> list[dict]:
    rows = conn.execute("SELECT * FROM audio_events ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [_row_to_event(row) for row in rows]


def mark_events_uploaded(conn: sqlite3.Connection, event_ids: list[int], *, ack: bool = False) -> int:
    """Mark selected events as uploaded, and optionally cloud-acknowledged."""

    if not event_ids:
        return 0
    placeholders = ",".join("?" for _ in event_ids)
    cursor = conn.execute(
        f"UPDATE audio_events SET uploaded = 1, ack = ? WHERE id IN ({placeholders})",
        [int(ack), *[int(event_id) for event_id in event_ids]],
    )
    conn.commit()
    return int(cursor.rowcount)


def summary(conn: sqlite3.Connection) -> dict:
    row = conn.execute(
        """
        SELECT COUNT(*) AS n,
               SUM(is_anomaly_raw) AS raw_anomalies,
               SUM(is_alarm) AS alarms,
               SUM(uploaded) AS uploaded,
               SUM(ack) AS acked,
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
        "uploaded_count": int(row["uploaded"] or 0),
        "ack_count": int(row["acked"] or 0),
        "pending_upload_count": int((row["n"] or 0) - (row["uploaded"] or 0)),
        "feature_ms_avg": float(row["feature_ms"] or 0.0),
        "inference_ms_avg": float(row["inference_ms"] or 0.0),
    }
