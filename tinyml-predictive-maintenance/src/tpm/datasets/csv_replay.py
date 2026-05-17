"""CSV replay loader for laptop-only sensor pipeline testing.

The embedded node normally consumes windows from the synthetic simulator. This
module lets the same node replay signal samples from a CSV file, which is the
lowest-friction bridge to real data captured from a phone, DAQ, microcontroller,
or public dataset export.

Minimum CSV schema:
    signal

Optional columns:
    timestamp, label

Example:
    timestamp,signal,label
    0.0000,0.12,normal
    0.0006,0.15,normal
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np


@dataclass(frozen=True)
class CsvReplayWindow:
    """One fixed-size replay window from a CSV signal stream."""

    samples: np.ndarray
    label: str
    timestamp: float | None


@dataclass(frozen=True)
class CsvSignal:
    """Loaded CSV signal columns.

    ``labels`` and ``timestamps`` may be ``None`` when the CSV omits the
    optional columns. The node will then publish ``true_state='unknown'`` and
    use wall-clock time for telemetry timestamps.
    """

    samples: np.ndarray
    labels: tuple[str, ...] | None
    timestamps: np.ndarray | None

    def windows(self, window_size: int, hop: int | None = None) -> Iterator[CsvReplayWindow]:
        """Yield non-overlapping or hopped windows.

        The label for a window is the majority label inside that window. This
        keeps replay robust if a CSV contains transition regions between normal
        and faulty operation.
        """

        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        hop = window_size if hop is None else hop
        if hop < 1:
            raise ValueError("hop must be >= 1")

        if self.samples.size < window_size:
            return

        for start in range(0, self.samples.size - window_size + 1, hop):
            end = start + window_size
            label = "unknown"
            if self.labels is not None:
                label = _majority_label(self.labels[start:end])
            timestamp = None
            if self.timestamps is not None:
                timestamp = float(self.timestamps[start])
            yield CsvReplayWindow(
                samples=self.samples[start:end].astype(np.float32, copy=False),
                label=label,
                timestamp=timestamp,
            )


def _majority_label(labels: tuple[str, ...]) -> str:
    """Return the most common label, preserving first-seen order on ties."""

    counts: dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return max(counts, key=counts.get)


def load_csv_signal(path: str | Path) -> CsvSignal:
    """Load a CSV file with required ``signal`` and optional metadata columns."""

    path = Path(path)
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"{path} is empty or missing a header row")
        fields = {name.strip() for name in reader.fieldnames}
        if "signal" not in fields:
            raise ValueError("CSV replay requires a 'signal' column")

        samples: list[float] = []
        labels: list[str] = []
        timestamps: list[float] = []
        saw_label = "label" in fields
        saw_timestamp = "timestamp" in fields

        for row_index, row in enumerate(reader, start=2):
            try:
                samples.append(float(row["signal"]))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid signal value on CSV line {row_index}") from exc

            if saw_label:
                label = (row.get("label") or "unknown").strip() or "unknown"
                labels.append(label)
            if saw_timestamp:
                try:
                    timestamps.append(float(row["timestamp"]))
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"invalid timestamp value on CSV line {row_index}") from exc

    if not samples:
        raise ValueError(f"{path} contains no samples")

    return CsvSignal(
        samples=np.asarray(samples, dtype=np.float32),
        labels=tuple(labels) if saw_label else None,
        timestamps=np.asarray(timestamps, dtype=np.float64) if saw_timestamp else None,
    )
