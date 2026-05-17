from __future__ import annotations

"""Configuration loading for the maintenance gateway.

The project uses TOML because it maps cleanly to Linux deployment files and is
available through ``tomllib`` in Python 3.11+. For Python 3.10, install
``tomli`` or keep using the default config path through the demo venv.
"""

from dataclasses import dataclass
from pathlib import Path
import sys

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - only used on Python 3.10 without stdlib tomllib
    import tomli as tomllib


@dataclass(frozen=True)
class GatewayConfig:
    """All file paths and defaults needed by the local gateway demo."""

    device_id: str
    database_path: Path
    telemetry_path: Path
    report_path: Path
    dashboard_path: Path


def load_config(path: str | Path = "configs/default.toml") -> GatewayConfig:
    """Load a gateway config file and resolve paths relative to project root."""

    config_path = Path(path)
    if not config_path.exists() and not config_path.is_absolute():
        config_path = Path(__file__).resolve().parents[2] / config_path
    root = config_path.resolve().parents[1]
    payload = tomllib.loads(config_path.read_text(encoding="utf-8"))
    section = payload["gateway"]
    return GatewayConfig(
        device_id=str(section["device_id"]),
        database_path=(root / section["database_path"]).resolve(),
        telemetry_path=(root / section["telemetry_path"]).resolve(),
        report_path=(root / section["report_path"]).resolve(),
        dashboard_path=(root / section["dashboard_path"]).resolve(),
    )
