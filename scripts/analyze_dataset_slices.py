from __future__ import annotations

import argparse
import csv
import html
from pathlib import Path

import _bootstrap  # noqa: F401
from src.dataset.panda_loader import PandaFrameDataset, FrameRecord
from src.slicing.sahi_slicer import generate_slices_for_image
from src.utils.config import add_config_arg, load_config


def _bbox_center(bbox: list[float]) -> tuple[float, float]:
    x, y, w, h = bbox
    return x + w / 2, y + h / 2


def _contains_point(container: list[float], x: float, y: float) -> bool:
    sx, sy, sw, sh = container
    return sx <= x <= sx + sw and sy <= y <= sy + sh


def _contains_bbox(container: list[float], bbox: list[float]) -> bool:
    sx, sy, sw, sh = container
    x, y, w, h = bbox
    return sx <= x and sy <= y and x + w <= sx + sw and y + h <= sy + sh


def _row_for_record(record: FrameRecord, slice_size: int, overlap_ratio: float) -> dict:
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
    slices = index["slices"]
    per_slice_counts = [0 for _ in slices]
    for gt in record.annotations:
        cx, cy = _bbox_center(gt["bbox"])
        for idx, s in enumerate(slices):
            if _contains_point(s["bbox"], cx, cy):
                per_slice_counts[idx] += 1
    total_gt = len(record.annotations)
    boundary_cut = sum(
        1 for gt in record.annotations if not any(_contains_bbox(s["bbox"], gt["bbox"]) for s in slices)
    )
    return {
        "image_id": record.frame_id,
        "width": record.width,
        "height": record.height,
        "slice_count": len(slices),
        "person_gt_count": sum(1 for gt in record.annotations if gt["category_id"] == 1),
        "vehicle_gt_count": sum(1 for gt in record.annotations if gt["category_id"] == 2),
        "gt_per_slice_mean": sum(per_slice_counts) / len(slices) if slices else 0.0,
        "empty_slice_ratio": sum(1 for c in per_slice_counts if c == 0) / len(slices) if slices else 0.0,
        "max_gt_per_slice": max(per_slice_counts) if per_slice_counts else 0,
        "boundary_cut_gt_count": boundary_cut,
        "boundary_cut_gt_ratio": boundary_cut / total_gt if total_gt else 0.0,
    }


def _svg_header(width: int, height: int, image_path: Path) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" ',
        f'viewBox="0 0 {width} {height}">\n',
        f'<image href="{html.escape(image_path.resolve().as_uri())}" x="0" y="0" width="{width}" height="{height}" preserveAspectRatio="none"/>\n',
    ]


def _gt_svg(record: FrameRecord) -> list[str]:
    lines = []
    for gt in record.annotations:
        x, y, w, h = gt["bbox"]
        color = "#00c853" if gt["category_id"] == 1 else "#2979ff"
        lines.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" '
            f'fill="none" stroke="{color}" stroke-width="4"/>\n'
        )
    return lines


def _grid_svg(slices: list[dict]) -> list[str]:
    return [
        f'<rect x="{s["bbox"][0]:.2f}" y="{s["bbox"][1]:.2f}" width="{s["bbox"][2]:.2f}" height="{s["bbox"][3]:.2f}" '
        'fill="none" stroke="#ff6d00" stroke-width="3" stroke-opacity="0.45"/>\n'
        for s in slices
    ]


def _write_visual_samples(record: FrameRecord, slice_size: int, overlap_ratio: float, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
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
    width = int(record.width or 0)
    height = int(record.height or 0)
    variants = {
        "sample_gt.svg": _gt_svg(record),
        "sample_slice_grid.svg": _grid_svg(index["slices"]),
        "sample_gt_slice_grid.svg": _grid_svg(index["slices"]) + _gt_svg(record),
    }
    for name, overlay in variants.items():
        lines = _svg_header(width, height, record.image_path)
        lines.extend(overlay)
        lines.append("</svg>\n")
        (out_dir / name).write_text("".join(lines), encoding="utf-8")


def _print_summary(rows: list[dict], target_long_side: int | None) -> None:
    widths = [int(r["width"]) for r in rows]
    heights = [int(r["height"]) for r in rows]
    slices = [int(r["slice_count"]) for r in rows]
    cut = [float(r["boundary_cut_gt_ratio"]) for r in rows]
    print(f"frames={len(rows)} target_long_side={target_long_side}")
    print(f"width_range={min(widths)}..{max(widths)} height_range={min(heights)}..{max(heights)}")
    print(f"slice_count_range={min(slices)}..{max(slices)} slice_count_mean={sum(slices)/len(slices):.2f}")
    print(f"boundary_cut_ratio_mean={sum(cut)/len(cut):.6f}")


def main() -> None:
    parser = add_config_arg(argparse.ArgumentParser())
    parser.add_argument("--target-long-side", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--visualize-samples", action="store_true")
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    preprocess = cfg.get("preprocess", {})
    target_long_side = args.target_long_side
    if target_long_side is None:
        target_long_side = preprocess.get("target_long_side")
    dataset = PandaFrameDataset(
        cfg["paths"]["images_dir"],
        cfg["paths"].get("gt_json"),
        cfg["paths"].get("vehicle_gt_json"),
        target_long_side,
        preprocess.get("keep_aspect_ratio", True),
    )
    records = dataset.records(max_frames=args.max_frames)
    rows = [
        _row_for_record(r, cfg["slicing"]["slice_size"], cfg["slicing"]["overlap_ratio"])
        for r in records
    ]
    out = Path(args.output or "results/sanity/slice_stats.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image_id",
        "width",
        "height",
        "slice_count",
        "person_gt_count",
        "vehicle_gt_count",
        "gt_per_slice_mean",
        "empty_slice_ratio",
        "max_gt_per_slice",
        "boundary_cut_gt_count",
        "boundary_cut_gt_ratio",
    ]
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    if args.visualize_samples and records:
        _write_visual_samples(records[0], cfg["slicing"]["slice_size"], cfg["slicing"]["overlap_ratio"], out.parent)
    _print_summary(rows, target_long_side)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
