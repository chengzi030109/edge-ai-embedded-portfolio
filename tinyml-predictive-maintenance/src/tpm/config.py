"""Project configuration loading.

The project can run entirely from command-line defaults, but a JSON config file
makes the system easier to reproduce and easier for future agents to modify. The
fields here intentionally mirror the parameters a real embedded deployment would
care about: sample rate, window size, model path, telemetry path, and report
locations.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ProjectConfig:
    """Typed representation of ``configs/default.json``.

    Keeping this small and explicit is helpful for interviews: it shows which
    parameters are part of the embedded signal-processing contract.
    """

    sample_rate_hz: int = 1600
    window_size: int = 256
    train_windows: int = 600
    duration_s: float = 20.0
    states: tuple[str, ...] = ("normal", "imbalance", "rubbing", "bearing")
    model_path: str = "artifacts/model.json"
    telemetry_path: str = "runs/telemetry.jsonl"
    evaluation_json: str = "reports/evaluation.json"
    evaluation_md: str = "reports/evaluation.md"

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "ProjectConfig":
        """Create a config object from a raw JSON dictionary."""

        return cls(
            sample_rate_hz=int(payload.get("sample_rate_hz", cls.sample_rate_hz)),
            window_size=int(payload.get("window_size", cls.window_size)),
            train_windows=int(payload.get("train_windows", cls.train_windows)),
            duration_s=float(payload.get("duration_s", cls.duration_s)),
            states=tuple(payload.get("states", cls.states)),
            model_path=str(payload.get("model_path", cls.model_path)),
            telemetry_path=str(payload.get("telemetry_path", cls.telemetry_path)),
            evaluation_json=str(payload.get("evaluation_json", cls.evaluation_json)),
            evaluation_md=str(payload.get("evaluation_md", cls.evaluation_md)),
        )


def load_config(path: str | Path | None) -> ProjectConfig:
    """Load a JSON config file, or return defaults if ``path`` is missing.

    A ``None`` path keeps existing scripts convenient for quick demos. Passing an
    explicit config path makes results easier to reproduce in a README or report.
    """

    if path is None:
        return ProjectConfig()
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return ProjectConfig.from_mapping(payload)
