from __future__ import annotations

"""Telemetry replay helpers.

The replay path is the hardware-free stand-in for MQTT/UART input. It consumes
the JSONL emitted by the TinyML maintenance node and pushes each row through the
same storage function that the HTTP API uses.
"""

import json
from pathlib import Path

from .storage import ingest_telemetry


def read_jsonl(path: str | Path) -> list[dict]:
    """Read telemetry JSON Lines from disk."""

    rows: list[dict] = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def replay_jsonl(conn, path: str | Path, device_id: str) -> int:
    """Replay a telemetry file into SQLite and return imported row count."""

    count = 0
    for payload in read_jsonl(path):
        ingest_telemetry(conn, payload, device_id=device_id)
        count += 1
    return count

