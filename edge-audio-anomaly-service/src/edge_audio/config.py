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
    model_backend: str
    model_path: Path
    onnx_model_path: Path
    report_path: Path
    figure_path: Path
    database_path: Path
    events_json: Path
    clips_dir: Path
    window_seconds: float
    hop_seconds: float
    save_anomaly_clips: bool
    alarm_on_count: int
    alarm_off_count: int


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
        model_backend=str(section.get("model_backend", "centroid")),
        model_path=(root / section["model_path"]).resolve(),
        onnx_model_path=(root / section.get("onnx_model_path", "artifacts/audio_model.onnx")).resolve(),
        report_path=(root / section["report_path"]).resolve(),
        figure_path=(root / section["figure_path"]).resolve(),
        database_path=(root / section["database_path"]).resolve(),
        events_json=(root / section["events_json"]).resolve(),
        clips_dir=(root / section["clips_dir"]).resolve(),
        window_seconds=float(section["window_seconds"]),
        hop_seconds=float(section["hop_seconds"]),
        save_anomaly_clips=bool(section["save_anomaly_clips"]),
        alarm_on_count=int(section["alarm_on_count"]),
        alarm_off_count=int(section["alarm_off_count"]),
    )
