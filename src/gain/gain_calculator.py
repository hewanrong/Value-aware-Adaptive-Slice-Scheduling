from __future__ import annotations

from collections import defaultdict


def iou_xywh(a: list[float], b: list[float]) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def count_correct(detections: list[dict], gt: list[dict], iou_threshold: float = 0.5) -> int:
    matched: set[int] = set()
    correct = 0
    for det in sorted(detections, key=lambda d: d.get("score", 0), reverse=True):
        for idx, label in enumerate(gt):
            if idx in matched or det.get("category_id") != label.get("category_id"):
                continue
            if iou_xywh(det["bbox"], label["bbox"]) >= iou_threshold:
                matched.add(idx)
                correct += 1
                break
    return correct


def compute_slice_gain(
    edge_results: list[dict],
    cloud_results: list[dict],
    gt_by_slice: dict[str, list[dict]],
    iou_threshold: float = 0.5,
) -> list[dict]:
    edge_by_slice: dict[str, list[dict]] = defaultdict(list)
    cloud_by_slice: dict[str, list[dict]] = defaultdict(list)
    for row in edge_results:
        edge_by_slice[row.get("slice_id")].append(row)
    for row in cloud_results:
        cloud_by_slice[row.get("slice_id")].append(row)

    rows = []
    for slice_id, labels in gt_by_slice.items():
        edge_correct = count_correct(edge_by_slice.get(slice_id, []), labels, iou_threshold)
        cloud_correct = count_correct(cloud_by_slice.get(slice_id, []), labels, iou_threshold)
        rows.append(
            {
                "slice_id": slice_id,
                "num_gt": len(labels),
                "edge_correct": edge_correct,
                "cloud_correct": cloud_correct,
                "gain": max(0, cloud_correct - edge_correct),
            }
        )
    return rows
