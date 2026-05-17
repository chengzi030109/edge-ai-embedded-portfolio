"""Telemetry output helpers.

The first implementation writes JSON Lines instead of using a real MQTT broker.
That keeps the project hardware-free and easy to run on any laptop, while still
preserving a message-oriented architecture. A future MQTT sink can implement the
same ``publish`` method and be swapped into the node loop.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping


class JsonlTelemetrySink:
    """Append each telemetry message as one JSON object per line."""

    def __init__(self, path: str | Path):
        # Create the output folder eagerly so the node loop can publish without
        # handling filesystem setup on every message.
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def publish(self, payload: Mapping) -> None:
        """Persist one telemetry payload.

        ``ensure_ascii=True`` keeps the output ASCII-only and friendly to simple
        embedded/Linux log-processing tools.
        """

        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=True) + "\n")
