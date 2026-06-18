from __future__ import annotations

import math
from pathlib import Path

from src.utils.images import image_size


def entropy(values: bytes) -> float:
    if not values:
        return 0.0
    counts = [0] * 256
    for b in values:
        counts[b] += 1
    total = len(values)
    return float(-sum((c / total) * math.log2(c / total) for c in counts if c))


def extract_visual_features(image_path: str | Path, bbox: list[float]) -> dict[str, float]:
    image_path = Path(image_path)
    width, height = image_size(image_path)
    _, _, w, h = bbox
    area_ratio = max(0.0, min(1.0, float(w * h) / max(1.0, width * height)))
    data = image_path.read_bytes()
    sample = data[: min(len(data), 1_000_000)]
    byte_entropy = entropy(sample)
    file_mb = image_path.stat().st_size / (1024 * 1024)
    return {
        "entropy": byte_entropy,
        "edge_density": area_ratio,
        "laplacian_variance": byte_entropy * area_ratio,
        "gradient_mean": byte_entropy / 8.0,
        "local_contrast": min(255.0, file_mb * area_ratio * 10.0),
        "motion_intensity": 0.0,
        "optical_flow_magnitude": 0.0,
    }
