from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

DetectionSource = Literal["edge", "cloud"]
DetectionQuality = Literal["low", "mid", "high", "full"]


@dataclass(frozen=True)
class DetectionResult:
    frame_id: str
    slice_id: str | None
    bbox: list[float]
    score: float
    category_id: int
    source: DetectionSource
    quality: DetectionQuality

    def to_dict(self) -> dict:
        return asdict(self)


def validate_detection(row: dict) -> dict:
    required = {"frame_id", "bbox", "score", "category_id", "source", "quality"}
    missing = required - set(row)
    if missing:
        raise ValueError(f"Detection row missing keys: {sorted(missing)}")
    if row["source"] not in {"edge", "cloud"}:
        raise ValueError(f"Invalid source: {row['source']}")
    if row["quality"] not in {"low", "mid", "high", "full"}:
        raise ValueError(f"Invalid quality: {row['quality']}")
    if len(row["bbox"]) != 4:
        raise ValueError("bbox must be [x, y, w, h]")
    return row
