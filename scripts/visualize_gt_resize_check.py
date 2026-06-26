from __future__ import annotations

import argparse
import csv
from pathlib import Path

import _bootstrap  # noqa: F401
from src.dataset.panda_loader import PandaFrameDataset
from src.slicing.sahi_slicer import generate_slices_for_image
from src.utils.config import add_config_arg, load_config
from src.utils.image_ops import draw_boxes, draw_slice_grid, open_display_image, open_resized_image


def _safe_name(frame_id: str) -> str:
    return frame_id.replace("/", "__").replace("\\", "__").replace(".", "_")


def main() -> None:
    parser = add_config_arg(argparse.ArgumentParser())
    parser.add_argument("--raw-config", default="configs/panda_raw.yaml")
    parser.add_argument("--max-frames", type=int, default=3)
    parser.add_argument("--output-dir", default="results/phase1/gt_resize_check")
    args = parser.parse_args()

    raw_cfg = load_config(args.raw_config)
    resized_cfg = load_config(args.config)
    raw_pre = raw_cfg.get("preprocess", {})
    resized_pre = resized_cfg.get("preprocess", {})
    raw_dataset = PandaFrameDataset(
        raw_cfg["paths"]["images_dir"],
        raw_cfg["paths"].get("gt_json"),
        raw_cfg["paths"].get("vehicle_gt_json"),
        raw_pre.get("target_long_side"),
        raw_pre.get("keep_aspect_ratio", True),
    )
    resized_dataset = PandaFrameDataset(
        resized_cfg["paths"]["images_dir"],
        resized_cfg["paths"].get("gt_json"),
        resized_cfg["paths"].get("vehicle_gt_json"),
        resized_pre.get("target_long_side"),
        resized_pre.get("keep_aspect_ratio", True),
    )
    raw_records = raw_dataset.records(max_frames=args.max_frames)
    resized_records = resized_dataset.records(max_frames=args.max_frames)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for raw, resized in zip(raw_records, resized_records):
        stem = _safe_name(raw.frame_id)
        raw_img, vis_sx, vis_sy = open_display_image(raw.image_path, long_side=1600)
        raw_vis = draw_boxes(raw_img, list(raw.annotations), vis_sx, vis_sy)
        raw_vis.save(out_dir / f"{stem}__raw_gt.png")

        resized_img = open_resized_image(resized.image_path, int(resized.width), int(resized.height))
        resized_gt = draw_boxes(resized_img, list(resized.annotations))
        resized_gt.save(out_dir / f"{stem}__resized_gt.png")

        slice_index = generate_slices_for_image(
            resized.frame_id,
            resized.image_path,
            resized_cfg["slicing"]["slice_size"],
            resized_cfg["slicing"]["overlap_ratio"],
            resized.width,
            resized.height,
            resized.original_width,
            resized.original_height,
            resized.scale_x,
            resized.scale_y,
        )
        resized_grid = draw_slice_grid(resized_gt, slice_index["slices"])
        resized_grid.save(out_dir / f"{stem}__resized_gt_slice_grid.png")

        rows.append(
            {
                "frame_id": raw.frame_id,
                "raw_width": raw.width,
                "raw_height": raw.height,
                "resized_width": resized.width,
                "resized_height": resized.height,
                "scale_x": resized.scale_x,
                "scale_y": resized.scale_y,
                "person_count": sum(1 for gt in resized.annotations if gt["category_id"] == 1),
                "vehicle_count": sum(1 for gt in resized.annotations if gt["category_id"] == 2),
            }
        )
    summary = out_dir / "resize_check_summary.csv"
    with summary.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "frame_id",
                "raw_width",
                "raw_height",
                "resized_width",
                "resized_height",
                "scale_x",
                "scale_y",
                "person_count",
                "vehicle_count",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote resize check for {len(rows)} frames to {out_dir}")


if __name__ == "__main__":
    main()
