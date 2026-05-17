from __future__ import annotations

"""Alarm debounce state machine for noisy audio anomaly windows.

An anomaly model emits a decision for every short window. On a 0.125 second hop
that means eight decisions per second, and individual decisions can flicker
because of harmless clicks, packet loss, or background noise. Industrial alarm
logic usually adds hysteresis: several consecutive bad windows are required to
enter alarm, and several consecutive good windows are required to clear it.
"""

from dataclasses import dataclass


@dataclass
class AlarmDebouncer:
    """Convert raw model decisions into a stable device alarm state.

    ``on_count`` is the number of consecutive raw anomaly windows required
    before the device enters alarm. ``off_count`` is the number of consecutive
    normal windows required before the alarm clears. The intermediate states are
    kept explicit because they are useful in a dashboard and easy to explain in
    an embedded interview:

    - ``normal``: no active alarm and no pending streak.
    - ``pending``: anomaly streak is building, but not long enough to alarm.
    - ``alarm``: alarm is active and the latest window is still anomalous.
    - ``recovering``: alarm is active, but normal windows are accumulating.
    """

    on_count: int = 3
    off_count: int = 5
    _bad_streak: int = 0
    _good_streak: int = 0
    _alarm: bool = False

    def update(self, is_anomaly_raw: bool) -> dict:
        """Advance the debouncer by one model decision."""

        if is_anomaly_raw:
            self._bad_streak += 1
            self._good_streak = 0
        else:
            self._good_streak += 1
            self._bad_streak = 0

        if not self._alarm and self._bad_streak >= self.on_count:
            self._alarm = True
        elif self._alarm and self._good_streak >= self.off_count:
            self._alarm = False

        if self._alarm:
            state = "alarm" if is_anomaly_raw else "recovering"
        else:
            state = "pending" if self._bad_streak > 0 else "normal"

        return {
            "is_alarm": self._alarm,
            "alarm_state": state,
            "alarm_bad_streak": self._bad_streak,
            "alarm_good_streak": self._good_streak,
        }
