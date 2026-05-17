from __future__ import annotations

"""TOML configuration for the audio anomaly service."""

from dataclasses import dataclass
from pathlib import Path
import sys

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib


@dataclass(frozen=True)
class AudioConfig:
    sample_rate_hz: int
    clip_seconds: float
    data_dir: Path
    model_path: Path
    report_path: Path
    figure_path: Path


def load_config(path: str | Path = "configs/default.toml") -> AudioConfig:
    config_path = Path(path)
    if not config_path.exists() and not config_path.is_absolute():
        config_path = Path(__file__).resolve().parents[2] / config_path
    root = config_path.resolve().parents[1]
    section = tomllib.loads(config_path.read_text(encoding="utf-8"))["audio"]
    return AudioConfig(
        sample_rate_hz=int(section["sample_rate_hz"]),
        clip_seconds=float(section["clip_seconds"]),
        data_dir=(root / section["data_dir"]).resolve(),
        model_path=(root / section["model_path"]).resolve(),
        report_path=(root / section["report_path"]).resolve(),
        figure_path=(root / section["figure_path"]).resolve(),
    )
