from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from src.utils.images import image_size

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass(frozen=True)
class FrameRecord:
    frame_id: str
    image_path: Path
    width: int | None = None
    height: int | None = None
    original_width: int | None = None
    original_height: int | None = None
    scale_x: float = 1.0
    scale_y: float = 1.0
    annotations: tuple[dict, ...] = ()


def _load_json(path: str | Path | None) -> dict:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _panda_object_to_detection(
    obj: dict,
    width: int,
    height: int,
    category_id: int,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    gt_id: str | None = None,
) -> dict:
    rect = obj.get("rect", {})
    tl = rect.get("tl", {})
    br = rect.get("br", {})
    x1 = float(tl.get("x", 0.0)) * width
    y1 = float(tl.get("y", 0.0)) * height
    x2 = float(br.get("x", 0.0)) * width
    y2 = float(br.get("y", 0.0)) * height
    bbox = [
        x1 * scale_x,
        y1 * scale_y,
        max(0.0, x2 - x1) * scale_x,
        max(0.0, y2 - y1) * scale_y,
    ]
    return {
        "gt_id": gt_id,
        "bbox": bbox,
        "original_bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
        "category": obj.get("category", "object"),
        "category_id": category_id,
    }


class PandaFrameDataset:
    """Reads PANDA extracted image frames plus optional PANDA bbox JSON files."""

    def __init__(
        self,
        images_dir: str | Path,
        person_gt_json: str | Path | None = None,
        vehicle_gt_json: str | Path | None = None,
        target_long_side: int | None = None,
        keep_aspect_ratio: bool = True,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.person_gt = _load_json(person_gt_json)
        self.vehicle_gt = _load_json(vehicle_gt_json)
        self.target_long_side = target_long_side
        self.keep_aspect_ratio = keep_aspect_ratio

    def iter_images(self) -> Iterable[Path]:
        for path in sorted(self.images_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
                yield path

    def _relative_id(self, image_path: Path) -> str:
        return image_path.relative_to(self.images_dir).as_posix()

    def _scaled_size(self, width: int, height: int) -> tuple[int, int, float, float]:
        if not self.target_long_side:
            return width, height, 1.0, 1.0
        long_side = max(width, height)
        if long_side <= 0:
            return width, height, 1.0, 1.0
        if self.keep_aspect_ratio:
            scale = self.target_long_side / long_side
            scaled_width = max(1, int(round(width * scale)))
            scaled_height = max(1, int(round(height * scale)))
            return scaled_width, scaled_height, scale, scale
        scale_x = self.target_long_side / width
        scale_y = self.target_long_side / height
        return self.target_long_side, self.target_long_side, scale_x, scale_y

    def _anno_for(
        self,
        frame_id: str,
        original_width: int,
        original_height: int,
        scale_x: float,
        scale_y: float,
    ) -> tuple[tuple[dict, ...], int, int]:
        annotations: list[dict] = []
        person_count = 0
        vehicle_count = 0
        for gt, category_id in ((self.person_gt, 1), (self.vehicle_gt, 2)):
            item = gt.get(frame_id, {})
            size = item.get("image size", {})
            anno_width = int(size.get("width") or original_width)
            anno_height = int(size.get("height") or original_height)
            objects = item.get("objects list", [])
            if category_id == 1:
                person_count += len(objects)
            else:
                vehicle_count += len(objects)
            for obj_index, obj in enumerate(objects):
                annotations.append(
                    _panda_object_to_detection(
                        obj,
                        anno_width,
                        anno_height,
                        category_id,
                        scale_x,
                        scale_y,
                        f"{frame_id}::c{category_id}::{obj_index}",
                    )
                )
        return tuple(annotations), person_count, vehicle_count

    def records(self, max_frames: int | None = None) -> list[FrameRecord]:
        rows: list[FrameRecord] = []
        for image_path in self.iter_images():
            frame_id = self._relative_id(image_path)
            original_width, original_height = image_size(image_path)
            width, height, scale_x, scale_y = self._scaled_size(original_width, original_height)
            annotations, _, _ = self._anno_for(frame_id, original_width, original_height, scale_x, scale_y)
            rows.append(
                FrameRecord(
                    frame_id,
                    image_path,
                    width,
                    height,
                    original_width,
                    original_height,
                    scale_x,
                    scale_y,
                    annotations,
                )
            )
            if max_frames is not None and len(rows) >= max_frames:
                break
        return rows
