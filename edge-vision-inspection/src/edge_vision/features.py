from __future__ import annotations

"""Lightweight visual feature extraction."""

from pathlib import Path

from PIL import Image, ImageFilter
import numpy as np

FEATURE_NAMES = ["mean", "std", "dark_ratio", "bright_ratio", "edge_mean", "edge_p95"]


def extract_features(path: str | Path) -> np.ndarray:
    """Extract low-cost image statistics for edge inspection."""

    img = Image.open(path).convert("L")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    edges = np.asarray(img.filter(ImageFilter.FIND_EDGES), dtype=np.float32) / 255.0
    return np.asarray(
        [
            float(np.mean(arr)),
            float(np.std(arr)),
            float(np.mean(arr < 0.35)),
            float(np.mean(arr > 0.85)),
            float(np.mean(edges)),
            float(np.quantile(edges, 0.95)),
        ],
        dtype=np.float32,
    )


def load_feature_rows(root: str | Path) -> list[dict]:
    rows: list[dict] = []
    for label in ("normal", "defect"):
        for path in sorted((Path(root) / label).glob("*.png")):
            rows.append({"path": str(path), "label": label, "features": extract_features(path)})
    return rows

