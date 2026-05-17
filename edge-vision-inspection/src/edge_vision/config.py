from __future__ import annotations

"""Configuration loading for vision inspection."""

from dataclasses import dataclass
from pathlib import Path
import sys

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib


@dataclass(frozen=True)
class VisionConfig:
    image_size: int
    data_dir: Path
    model_path: Path
    report_path: Path
    annotated_dir: Path
    figure_path: Path


def load_config(path: str | Path = "configs/default.toml") -> VisionConfig:
    config_path = Path(path)
    if not config_path.exists() and not config_path.is_absolute():
        config_path = Path(__file__).resolve().parents[2] / config_path
    root = config_path.resolve().parents[1]
    section = tomllib.loads(config_path.read_text(encoding="utf-8"))["vision"]
    return VisionConfig(
        image_size=int(section["image_size"]),
        data_dir=(root / section["data_dir"]).resolve(),
        model_path=(root / section["model_path"]).resolve(),
        report_path=(root / section["report_path"]).resolve(),
        annotated_dir=(root / section["annotated_dir"]).resolve(),
        figure_path=(root / section["figure_path"]).resolve(),
    )
