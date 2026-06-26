from __future__ import annotations

from collections import defaultdict


def local_to_frame_bbox(local_xyxy: list[float], slice_xyxy: list[float]) -> list[float]:
    sx1, sy1, _, _ = slice_xyxy
    x1, y1, x2, y2 = local_xyxy
    return [x1 + sx1, y1 + sy1, x2 + sx1, y2 + sy1]


def frame_to_local_bbox(frame_xyxy: list[float], slice_xyxy: list[float]) -> list[float]:
    sx1, sy1, _, _ = slice_xyxy
    x1, y1, x2, y2 = frame_xyxy
    return [x1 - sx1, y1 - sy1, x2 - sx1, y2 - sy1]


def clip_bbox_to_frame(bbox_xyxy: list[float], width: int | float, height: int | float) -> list[float]:
    x1, y1, x2, y2 = bbox_xyxy
    x1 = min(max(0.0, float(x1)), float(width))
    y1 = min(max(0.0, float(y1)), float(height))
    x2 = min(max(0.0, float(x2)), float(width))
    y2 = min(max(0.0, float(y2)), float(height))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def bbox_iou_xyxy(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def global_classwise_nms(detections: list[dict], iou_threshold: float = 0.5) -> list[dict]:
    by_class: dict[int, list[dict]] = defaultdict(list)
    for det in detections:
        by_class[int(det["category_id"])].append(det)
    kept: list[dict] = []
    for _, rows in by_class.items():
        candidates = sorted(rows, key=lambda r: float(r.get("score", 0.0)), reverse=True)
        class_kept: list[dict] = []
        for det in candidates:
            if all(bbox_iou_xyxy(det["bbox_xyxy_frame"], prev["bbox_xyxy_frame"]) < iou_threshold for prev in class_kept):
                class_kept.append(det)
        kept.extend(class_kept)
    return sorted(kept, key=lambda r: float(r.get("score", 0.0)), reverse=True)
