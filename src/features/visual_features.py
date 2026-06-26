from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


def entropy(values: bytes) -> float:
    if not values:
        return 0.0
    counts = [0] * 256
    for b in values:
        counts[b] += 1
    total = len(values)
    return float(-sum((c / total) * math.log2(c / total) for c in counts if c))


def _gray_entropy(gray: np.ndarray) -> float:
    hist, _ = np.histogram(gray, bins=256, range=(0, 255), density=False)
    total = int(hist.sum())
    if total == 0:
        return 0.0
    probs = hist[hist > 0] / total
    return float(-(probs * np.log2(probs)).sum())


def _colorfulness(rgb: np.ndarray) -> float:
    arr = rgb.astype(np.float32)
    rg = arr[..., 0] - arr[..., 1]
    yb = 0.5 * (arr[..., 0] + arr[..., 1]) - arr[..., 2]
    return float(np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))


def extract_visual_features_from_image(
    image: Image.Image,
    bbox: list[float],
    previous_image: Image.Image | None = None,
) -> dict[str, float]:
    x, y, w, h = [int(round(v)) for v in bbox]
    x2, y2 = max(x + 1, x + w), max(y + 1, y + h)
    crop = image.crop((x, y, x2, y2)).convert("RGB")
    rgb = np.asarray(crop, dtype=np.uint8)
    gray = np.asarray(crop.convert("L"), dtype=np.float32)
    gy, gx = np.gradient(gray)
    gradient = np.hypot(gx, gy)
    edge_threshold = max(10.0, float(gradient.mean() + gradient.std()))
    lap = np.asarray(crop.convert("L").filter(ImageFilter.FIND_EDGES), dtype=np.float32)
    brightness_mean = float(gray.mean())
    brightness_std = float(gray.std())
    if previous_image is None:
        # No previous frame exists for the first frame in a sequence; use zero-valued
        # temporal features so downstream CSV consumers do not need NaN handling.
        frame_difference_mean = 0.0
        frame_difference_ratio = 0.0
    else:
        prev = previous_image.crop((x, y, x2, y2)).convert("L")
        prev_gray = np.asarray(prev, dtype=np.float32)
        diff = np.abs(gray - prev_gray)
        frame_difference_mean = float(diff.mean())
        frame_difference_ratio = float((diff > 15.0).mean())
    return {
        "entropy": _gray_entropy(gray),
        "edge_density": float((gradient > edge_threshold).mean()),
        "laplacian_variance": float(lap.var()),
        "gradient_mean": float(gradient.mean()),
        "gradient_std": float(gradient.std()),
        "local_contrast": brightness_std,
        "brightness_mean": brightness_mean,
        "brightness_std": brightness_std,
        "colorfulness": _colorfulness(rgb),
        "frame_difference_mean": frame_difference_mean,
        "frame_difference_ratio": frame_difference_ratio,
    }


def extract_visual_features(image_path: str | Path, bbox: list[float]) -> dict[str, float]:
    with Image.open(image_path) as img:
        return extract_visual_features_from_image(img.convert("RGB"), bbox)
