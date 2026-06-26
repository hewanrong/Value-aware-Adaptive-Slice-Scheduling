from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

DetectionSource = Literal["edge", "cloud"]
DetectionQuality = Literal["none", "low", "mid", "high", "full"]


@dataclass(frozen=True)
class DetectionResult:
    frame_id: str
    slice_id: str | None
    bbox_xyxy: list[float]
    class_id: int
    model_name: str
    backend: str
    input_width: int
    input_height: int
    inference_time_ms: float
    bbox_xyxy_local: list[float]
    bbox_xyxy_frame: list[float]
    score: float
    category_id: int
    source: DetectionSource
    quality: DetectionQuality
    slice_x1: float
    slice_y1: float
    slice_x2: float
    slice_y2: float

    def to_dict(self) -> dict:
        return asdict(self)


def validate_detection(row: dict) -> dict:
    # New canonical fields are slice-local and model-agnostic. Legacy fields are
    # still accepted below so existing mock cache consumers keep working.
    canonical_required = {
        "frame_id",
        "slice_id",
        "bbox_xyxy",
        "class_id",
        "score",
        "model_name",
        "backend",
        "input_width",
        "input_height",
        "inference_time_ms",
    }
    canonical_missing = canonical_required - set(row)
    if not canonical_missing:
        if len(row["bbox_xyxy"]) != 4:
            raise ValueError("bbox_xyxy must be slice-local [x1, y1, x2, y2]")
        return row

    required = {
        "frame_id",
        "slice_id",
        "bbox_xyxy_local",
        "bbox_xyxy_frame",
        "score",
        "category_id",
        "source",
        "quality",
        "slice_x1",
        "slice_y1",
        "slice_x2",
        "slice_y2",
    }
    missing = required - set(row)
    if missing:
        raise ValueError(f"Detection row missing keys: {sorted(missing)}")
    if row["source"] not in {"edge", "cloud"}:
        raise ValueError(f"Invalid source: {row['source']}")
    if row["quality"] not in {"none", "low", "mid", "high", "full"}:
        raise ValueError(f"Invalid quality: {row['quality']}")
    if len(row["bbox_xyxy_local"]) != 4 or len(row["bbox_xyxy_frame"]) != 4:
        raise ValueError("bbox fields must be [x1, y1, x2, y2]")
    return row


def canonical_detection_record(
    *,
    frame_id: str,
    slice_id: str | None,
    bbox_xyxy: list[float],
    class_id: int,
    score: float,
    model_name: str,
    backend: str,
    input_width: int,
    input_height: int,
    inference_time_ms: float = 0.0,
    **extra,
) -> dict:
    row = {
        "frame_id": frame_id,
        "slice_id": slice_id,
        "bbox_xyxy": bbox_xyxy,
        "class_id": class_id,
        "score": score,
        "model_name": model_name,
        "backend": backend,
        "input_width": input_width,
        "input_height": input_height,
        "inference_time_ms": inference_time_ms,
    }
    row.update(extra)
    return row
