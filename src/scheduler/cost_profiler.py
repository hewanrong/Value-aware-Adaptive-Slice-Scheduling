from __future__ import annotations

import io
from pathlib import Path

from PIL import Image


def profile_slice_costs(
    image_path: str | Path,
    bbox: list[float],
    jpeg_quality: dict[str, int],
    latency_per_mb_ms: float = 25.0,
) -> dict[str, float]:
    with Image.open(image_path) as img:
        return profile_slice_costs_from_image(img.convert("RGB"), bbox, jpeg_quality, latency_per_mb_ms)


def profile_slice_costs_from_image(
    image: Image.Image,
    bbox: list[float],
    jpeg_quality: dict[str, int],
    latency_per_mb_ms: float = 25.0,
) -> dict[str, float]:
    x, y, w, h = [int(round(v)) for v in bbox]
    crop = image.crop((x, y, max(x + 1, x + w), max(y + 1, y + h))).convert("RGB")
    raw_bytes = max(1, crop.width * crop.height * 3)
    costs: dict[str, float] = {}
    for quality, q in jpeg_quality.items():
        buf = io.BytesIO()
        crop.save(buf, format="JPEG", quality=int(q), optimize=False)
        byte_count = float(len(buf.getvalue()))
        costs[f"bytes_{quality}"] = byte_count
        costs[f"compression_ratio_{quality}"] = byte_count / raw_bytes
        costs[f"estimated_latency_{quality}"] = byte_count / (1024 * 1024) * latency_per_mb_ms
    return costs
