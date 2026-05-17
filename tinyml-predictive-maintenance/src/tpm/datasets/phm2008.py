"""PHM 2008 / NASA C-MAPSS multivariate degradation loader.

PHM 2008 is not a bearing vibration dataset. It is an aircraft engine
degradation dataset derived from NASA C-MAPSS, with multiple operating settings
and sensor channels over engine life cycles. This loader treats it as a harder
normal-vs-degradation detection problem rather than a full RUL prediction task.

Expected file format follows the common C-MAPSS text layout:
    unit cycle setting_1 setting_2 setting_3 sensor_1 ... sensor_21

Rows are whitespace-separated. Small synthetic fixtures with fewer sensors are
also supported for tests and demos.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class PhmWindowSet:
    """Windowed PHM/C-MAPSS data ready for anomaly detection."""

    features: np.ndarray
    labels: np.ndarray
    units: np.ndarray
    feature_names: tuple[str, ...]


def load_phm_table(path: str | Path) -> np.ndarray:
    """Load a PHM08/C-MAPSS whitespace-separated table."""

    table = np.loadtxt(str(path), dtype=np.float32)
    if table.ndim == 1:
        table = table.reshape(1, -1)
    if table.shape[1] < 6:
        raise ValueError("PHM2008 table must contain unit, cycle, settings, and at least one sensor")
    return table


def _window_sensor_features(window: np.ndarray) -> np.ndarray:
    """Compute mean/std/min/max/slope for each sensor in a window."""

    cycles = window[:, 1]
    sensors = window[:, 5:]
    values: list[float] = []
    x = cycles - cycles.mean()
    denom = float(np.sum(x * x) + 1e-8)
    for col in range(sensors.shape[1]):
        y = sensors[:, col]
        slope = float(np.sum(x * (y - y.mean())) / denom)
        values.extend(
            [
                float(np.mean(y)),
                float(np.std(y)),
                float(np.min(y)),
                float(np.max(y)),
                slope,
            ]
        )
    # Add two window-level features that help downstream scripts explain the
    # degradation timeline independent of a particular sensor index.
    values.append(float(cycles[0]))
    values.append(float(cycles[-1] - cycles[0] + 1))
    return np.asarray(values, dtype=np.float32)


def feature_names(n_sensors: int) -> tuple[str, ...]:
    """Return deterministic names for PHM window statistics."""

    names: list[str] = []
    for idx in range(1, n_sensors + 1):
        for stat in ("mean", "std", "min", "max", "slope"):
            names.append(f"sensor_{idx}_{stat}")
    names.extend(["cycle_start", "cycle_span"])
    return tuple(names)


def make_windows(
    table: np.ndarray,
    window_size: int = 30,
    hop: int = 10,
    healthy_fraction: float = 0.30,
    degraded_fraction: float = 0.70,
) -> PhmWindowSet:
    """Slice each engine trajectory into statistical feature windows.

    Windows ending in the first ``healthy_fraction`` of an engine life are
    labeled normal. Windows starting after ``degraded_fraction`` are labeled
    degraded/anomalous. Middle-life windows are skipped so the binary task is
    not contaminated by ambiguous transition data.
    """

    if not 0.0 < healthy_fraction < degraded_fraction < 1.0:
        raise ValueError("fractions must satisfy 0 < healthy < degraded < 1")
    if window_size < 2 or hop < 1:
        raise ValueError("window_size must be >= 2 and hop must be >= 1")

    rows: list[np.ndarray] = []
    labels: list[bool] = []
    units: list[int] = []
    for unit in np.unique(table[:, 0].astype(np.int32)):
        unit_rows = table[table[:, 0] == unit]
        unit_rows = unit_rows[np.argsort(unit_rows[:, 1])]
        n = unit_rows.shape[0]
        if n < window_size:
            continue
        healthy_cut = healthy_fraction * n
        degraded_cut = degraded_fraction * n
        for start in range(0, n - window_size + 1, hop):
            end = start + window_size
            if end <= healthy_cut:
                label = False
            elif start >= degraded_cut:
                label = True
            else:
                continue
            rows.append(_window_sensor_features(unit_rows[start:end]))
            labels.append(label)
            units.append(int(unit))

    if not rows:
        raise ValueError("no PHM windows produced; reduce window_size or check input file")

    n_sensors = table.shape[1] - 5
    return PhmWindowSet(
        features=np.vstack(rows).astype(np.float32),
        labels=np.asarray(labels, dtype=bool),
        units=np.asarray(units, dtype=np.int32),
        feature_names=feature_names(n_sensors),
    )


def load_phm_windows(path: str | Path, window_size: int = 30, hop: int = 10) -> PhmWindowSet:
    """Convenience wrapper: load table and slice windows."""

    return make_windows(load_phm_table(path), window_size=window_size, hop=hop)
