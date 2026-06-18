from __future__ import annotations

import json
from pathlib import Path

from src.utils.images import image_size


def _axis_starts(length: int, slice_size: int, overlap_ratio: float) -> list[int]:
    if length <= slice_size:
        return [0]
    step = max(1, int(slice_size * (1.0 - overlap_ratio)))
    starts = list(range(0, max(1, length - slice_size + 1), step))
    last = length - slice_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def generate_slices_for_image(
    frame_id: str,
    image_path: str | Path,
    slice_size: int = 1024,
    overlap_ratio: float = 0.25,
    width: int | None = None,
    height: int | None = None,
    original_width: int | None = None,
    original_height: int | None = None,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
) -> dict:
    image_path = Path(image_path)
    if width is None or height is None:
        width, height = image_size(image_path)
    original_width = original_width or width
    original_height = original_height or height
    slices = []
    for y in _axis_starts(height, slice_size, overlap_ratio):
        for x in _axis_starts(width, slice_size, overlap_ratio):
            w = min(slice_size, width - x)
            h = min(slice_size, height - y)
            slice_id = f"{frame_id}::x{x}_y{y}_w{w}_h{h}"
            slices.append({"slice_id": slice_id, "bbox": [x, y, w, h]})
    return {
        "frame_id": frame_id,
        "image_path": str(image_path),
        "width": width,
        "height": height,
        "original_width": original_width,
        "original_height": original_height,
        "scale_x": scale_x,
        "scale_y": scale_y,
        "slice_size": slice_size,
        "overlap_ratio": overlap_ratio,
        "slices": slices,
    }


def build_slice_index(records, slice_size: int = 1024, overlap_ratio: float = 0.25) -> dict:
    frames = [
        generate_slices_for_image(
            r.frame_id,
            r.image_path,
            slice_size,
            overlap_ratio,
            r.width,
            r.height,
            r.original_width,
            r.original_height,
            r.scale_x,
            r.scale_y,
        )
        for r in records
    ]
    return {
        "schema_version": 1,
        "slice_size": slice_size,
        "overlap_ratio": overlap_ratio,
        "num_frames": len(frames),
        "num_slices": sum(len(f["slices"]) for f in frames),
        "frames": frames,
    }


def save_slice_index(index: dict, output_path: str | Path) -> None:
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)
