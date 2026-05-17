"""Alarm debounce logic for streaming anomaly decisions.

The model produces a raw decision for every window. Real embedded devices
rarely turn a single bad window into a user-visible alarm because vibration
signals are noisy and transient spikes are common. This module converts raw
per-window anomaly decisions into a debounced device alarm state.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AlarmUpdate:
    """One debouncer output sample.

    Fields:
        is_alarm: stable device-level alarm bit after debounce.
        alarm_state: human-readable state for telemetry/dashboard timelines.
        anomaly_streak: consecutive raw anomaly windows seen so far.
        normal_streak: consecutive raw normal windows seen so far.
    """

    is_alarm: bool
    alarm_state: str
    anomaly_streak: int
    normal_streak: int


class AlarmDebouncer:
    """Debounce raw anomaly decisions into a stable alarm signal.

    Defaults are intentionally conservative for demos:
    - enter alarm after 3 consecutive anomaly windows
    - leave alarm after 5 consecutive normal windows

    This is the same idea as button debounce or fault confirmation in embedded
    systems: make the system slower to flip state than the raw sensor signal.
    """

    def __init__(self, on_count: int = 3, off_count: int = 5):
        if on_count < 1:
            raise ValueError("on_count must be >= 1")
        if off_count < 1:
            raise ValueError("off_count must be >= 1")
        self.on_count = on_count
        self.off_count = off_count
        self._is_alarm = False
        self._anomaly_streak = 0
        self._normal_streak = 0

    def update(self, is_anomaly_raw: bool) -> AlarmUpdate:
        """Feed one raw model decision and return the debounced alarm state."""

        if is_anomaly_raw:
            self._anomaly_streak += 1
            self._normal_streak = 0
        else:
            self._normal_streak += 1
            self._anomaly_streak = 0

        if not self._is_alarm and self._anomaly_streak >= self.on_count:
            self._is_alarm = True
        elif self._is_alarm and self._normal_streak >= self.off_count:
            self._is_alarm = False

        if self._is_alarm:
            state = "alarm" if is_anomaly_raw else "recovering"
        else:
            state = "pending" if is_anomaly_raw else "normal"

        return AlarmUpdate(
            is_alarm=self._is_alarm,
            alarm_state=state,
            anomaly_streak=self._anomaly_streak,
            normal_streak=self._normal_streak,
        )
