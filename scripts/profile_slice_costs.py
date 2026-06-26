from __future__ import annotations

import argparse
import csv
from pathlib import Path

import _bootstrap  # noqa: F401
from src.dataset.panda_loader import PandaFrameDataset
from src.scheduler.cost_profiler import profile_slice_costs_from_image
from src.slicing.sahi_slicer import generate_slices_for_image
from src.utils.config import add_config_arg, load_config
from src.utils.image_ops import open_resized_image


def main() -> None:
    parser = add_config_arg(argparse.ArgumentParser())
    parser.add_argument("--max-slices", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
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
    rows = []
    for record in records:
        image = open_resized_image(record.image_path, int(record.width), int(record.height))
        frame = generate_slices_for_image(
            record.frame_id,
            record.image_path,
            cfg["slicing"]["slice_size"],
            cfg["slicing"]["overlap_ratio"],
            record.width,
            record.height,
            record.original_width,
            record.original_height,
            record.scale_x,
            record.scale_y,
        )
        for s in frame["slices"]:
            row = {"frame_id": record.frame_id, "slice_id": s["slice_id"]}
            row.update(profile_slice_costs_from_image(image, s["bbox"], cfg["cost_profiler"]["jpeg_quality"]))
            rows.append(row)
            if args.max_slices and len(rows) >= args.max_slices:
                break
        if args.max_slices and len(rows) >= args.max_slices:
            break
    out = Path(cfg["paths"]["slice_costs_csv"])
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["frame_id", "slice_id"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} cost rows for {len(records)} frames to {out}")


if __name__ == "__main__":
    main()
