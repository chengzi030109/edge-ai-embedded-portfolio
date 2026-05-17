from __future__ import annotations

"""Synthetic inspection image generation."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter
import numpy as np


def generate_demo_images(root: str | Path, image_size: int = 160) -> list[Path]:
    """Generate simple normal and defective product-surface images."""

    root = Path(root)
    rng = np.random.default_rng(2026)
    paths: list[Path] = []
    for label in ("normal", "defect"):
        (root / label).mkdir(parents=True, exist_ok=True)
    for idx in range(10):
        base = Image.new("L", (image_size, image_size), 170)
        noise = rng.normal(0, 5, size=(image_size, image_size)).astype(np.int16)
        arr = np.clip(np.asarray(base, dtype=np.int16) + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr, "L").filter(ImageFilter.GaussianBlur(radius=0.4)).convert("RGB")
        path = root / "normal" / f"normal_{idx:02d}.png"
        img.save(path)
        paths.append(path)
    for idx in range(6):
        base = Image.new("L", (image_size, image_size), 170)
        arr = np.asarray(base, dtype=np.uint8)
        img = Image.fromarray(arr, "L").convert("RGB")
        draw = ImageDraw.Draw(img)
        defect_w = max(10, image_size // 5)
        defect_h = max(6, image_size // 12)
        margin = max(6, image_size // 8)
        x = int(rng.integers(margin, max(margin + 1, image_size - defect_w - margin)))
        y = int(rng.integers(margin, max(margin + 1, image_size - defect_w - margin)))
        if idx % 2 == 0:
            draw.rectangle((x, y, x + defect_w, y + defect_h), fill=(45, 45, 45))
        else:
            draw.line((x, y, x + defect_w, y + defect_w // 2), fill=(30, 30, 30), width=max(3, image_size // 32))
        path = root / "defect" / f"defect_{idx:02d}.png"
        img.save(path)
        paths.append(path)
    return paths
