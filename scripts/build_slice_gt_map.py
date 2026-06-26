from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from src.dataset.panda_loader import PandaFrameDataset
from src.slicing.sahi_slicer import generate_slices_for_image
from src.utils.config import add_config_arg, load_config


def _center(bbox: list[float]) -> tuple[float, float]:
    x, y, w, h = bbox
    return x + w / 2, y + h / 2


def _contains_point(slice_bbox: list[float], x: float, y: float) -> bool:
    sx, sy, sw, sh = slice_bbox
    return sx <= x <= sx + sw and sy <= y <= sy + sh


def _contains_bbox(slice_bbox: list[float], bbox: list[float]) -> bool:
    sx, sy, sw, sh = slice_bbox
    x, y, w, h = bbox
    return sx <= x and sy <= y and x + w <= sx + sw and y + h <= sy + sh


def _area_bucket(area: float) -> str:
    if area < 32 * 32:
        return "small"
    if area < 96 * 96:
        return "medium"
    return "large"


def _slice_rows(records, slice_size: int, overlap_ratio: float) -> list[dict]:
    rows = []
    for record in records:
        index = generate_slices_for_image(
            record.frame_id,
            record.image_path,
            slice_size,
            overlap_ratio,
            record.width,
            record.height,
            record.original_width,
            record.original_height,
            record.scale_x,
            record.scale_y,
        )
        for s in index["slices"]:
            sx, sy, sw, sh = s["bbox"]
            assigned = []
            for gt in record.annotations:
                cx, cy = _center(gt["bbox"])
                if _contains_point(s["bbox"], cx, cy):
                    assigned.append(gt)
            areas = [gt["bbox"][2] * gt["bbox"][3] for gt in assigned]
            boundary_cut = [gt for gt in assigned if not _contains_bbox(s["bbox"], gt["bbox"])]
            buckets = {"small": 0, "medium": 0, "large": 0}
            for area in areas:
                buckets[_area_bucket(area)] += 1
            rows.append(
                {
                    "frame_id": record.frame_id,
                    "slice_id": s["slice_id"],
                    "slice_x1": sx,
                    "slice_y1": sy,
                    "slice_x2": sx + sw,
                    "slice_y2": sy + sh,
                    "person_gt_count": sum(1 for gt in assigned if gt["category_id"] == 1),
                    "vehicle_gt_count": sum(1 for gt in assigned if gt["category_id"] == 2),
                    "total_gt_count": len(assigned),
                    "small_gt_count": buckets["small"],
                    "medium_gt_count": buckets["medium"],
                    "large_gt_count": buckets["large"],
                    "gt_density": len(assigned) / max(1.0, sw * sh),
                    "max_gt_area": max(areas) if areas else 0.0,
                    "mean_gt_area": sum(areas) / len(areas) if areas else 0.0,
                    "min_gt_area": min(areas) if areas else 0.0,
                    "boundary_cut_gt_count": len(boundary_cut),
                    "boundary_cut_gt_ratio": len(boundary_cut) / len(assigned) if assigned else 0.0,
                    "gt_ids_json": json.dumps([gt["gt_id"] for gt in assigned], ensure_ascii=False),
                }
            )
    return rows


def _write_summary(rows: list[dict], records_count: int, output_path: Path) -> None:
    total_slices = len(rows)
    non_empty = [r for r in rows if int(r["total_gt_count"]) > 0]
    total_gt = sum(int(r["total_gt_count"]) for r in rows)
    boundary_cut = sum(int(r["boundary_cut_gt_count"]) for r in rows)
    summary = {
        "total_frames": records_count,
        "total_slices": total_slices,
        "mean_slices_per_frame": total_slices / records_count if records_count else 0.0,
        "empty_slice_ratio": sum(1 for r in rows if int(r["total_gt_count"]) == 0) / total_slices if total_slices else 0.0,
        "mean_gt_per_non_empty_slice": sum(int(r["total_gt_count"]) for r in non_empty) / len(non_empty) if non_empty else 0.0,
        "person_total": sum(int(r["person_gt_count"]) for r in rows),
        "vehicle_total": sum(int(r["vehicle_gt_count"]) for r in rows),
        "boundary_cut_gt_ratio": boundary_cut / total_gt if total_gt else 0.0,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)
    print(
        "summary "
        f"frames={summary['total_frames']} slices={summary['total_slices']} "
        f"empty_slice_ratio={summary['empty_slice_ratio']:.6f} "
        f"boundary_cut_gt_ratio={summary['boundary_cut_gt_ratio']:.6f}"
    )


def main() -> None:
    parser = add_config_arg(argparse.ArgumentParser())
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--output", default="cache/slice_gt_map.csv")
    args = parser.parse_args()
    cfg = load_config(args.config)
    preprocess = cfg.get("preprocess", {})
    dataset = PandaFrameDataset(
        cfg["paths"]["images_dir"],
        cfg["paths"].get("gt_json"),
        cfg["paths"].get("vehicle_gt_json"),
        preprocess.get("target_long_side"),
        preprocess.get("keep_aspect_ratio", True),
    )
    records = dataset.records(max_frames=args.max_frames)
    rows = _slice_rows(records, cfg["slicing"]["slice_size"], cfg["slicing"]["overlap_ratio"])
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "frame_id",
        "slice_id",
        "slice_x1",
        "slice_y1",
        "slice_x2",
        "slice_y2",
        "person_gt_count",
        "vehicle_gt_count",
        "total_gt_count",
        "small_gt_count",
        "medium_gt_count",
        "large_gt_count",
        "gt_density",
        "max_gt_area",
        "mean_gt_area",
        "min_gt_area",
        "boundary_cut_gt_count",
        "boundary_cut_gt_ratio",
        "gt_ids_json",
    ]
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} slice GT rows to {out}")
    _write_summary(rows, len(records), Path("results/phase1/slice_gt_map_summary.csv"))


if __name__ == "__main__":
    main()
