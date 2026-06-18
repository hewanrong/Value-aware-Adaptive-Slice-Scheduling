from __future__ import annotations

from pathlib import Path

from src.utils.images import image_size


def profile_slice_costs(
    image_path: str | Path,
    bbox: list[float],
    jpeg_quality: dict[str, int],
    latency_per_mb_ms: float = 25.0,
) -> dict[str, float]:
    costs: dict[str, float] = {}
    image_path = Path(image_path)
    width, height = image_size(image_path)
    _, _, w, h = bbox
    area_ratio = max(0.0, min(1.0, float(w * h) / max(1.0, width * height)))
    full_equiv = image_path.stat().st_size * area_ratio
    for quality, q in jpeg_quality.items():
        quality_factor = max(0.05, min(1.0, float(q) / 95.0))
        byte_count = float(full_equiv * quality_factor)
        costs[f"bytes_{quality}"] = byte_count
        costs[f"estimated_latency_{quality}"] = byte_count / (1024 * 1024) * latency_per_mb_ms
    return costs
