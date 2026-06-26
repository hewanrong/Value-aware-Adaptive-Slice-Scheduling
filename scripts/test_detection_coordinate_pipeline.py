from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from PIL import ImageDraw

from src.dataset.panda_loader import PandaFrameDataset
from src.detection.coordinate_utils import clip_bbox_to_frame, global_classwise_nms, local_to_frame_bbox
from src.slicing.sahi_slicer import generate_slices_for_image
from src.utils.config import add_config_arg, load_config
from src.utils.image_ops import draw_slice_grid, open_resized_image


def _mock_local_boxes(w: float, h: float) -> list[tuple[list[float], float]]:
    base = [0.22 * w, 0.22 * h, 0.48 * w, 0.48 * h]
    duplicate = [0.24 * w, 0.24 * h, 0.50 * w, 0.50 * h]
    return [(base, 0.90), (duplicate, 0.65)]


def main() -> None:
    parser = add_config_arg(argparse.ArgumentParser())
    parser.add_argument("--max-frames", type=int, default=1)
    parser.add_argument("--output", default="results/phase1_5/coordinate_pipeline_check.png")
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
    if not records:
        raise SystemExit("No frames found.")
    record = records[0]
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
    detections = []
    for s in frame["slices"]:
        sx, sy, sw, sh = s["bbox"]
        slice_xyxy = [sx, sy, sx + sw, sy + sh]
        for local, score in _mock_local_boxes(sw, sh):
            frame_box = clip_bbox_to_frame(local_to_frame_bbox(local, slice_xyxy), frame["width"], frame["height"])
            detections.append(
                {
                    "frame_id": frame["frame_id"],
                    "slice_id": s["slice_id"],
                    "bbox_xyxy_local": local,
                    "bbox_xyxy_frame": frame_box,
                    "score": score,
                    "category_id": 1,
                    "source": "edge",
                    "quality": "full",
                    "slice_x1": sx,
                    "slice_y1": sy,
                    "slice_x2": sx + sw,
                    "slice_y2": sy + sh,
                }
            )
    nms_detections = global_classwise_nms(detections, iou_threshold=0.5)
    for det in nms_detections:
        x1, y1, x2, y2 = det["bbox_xyxy_frame"]
        assert 0 <= x1 <= x2 <= frame["width"], det
        assert 0 <= y1 <= y2 <= frame["height"], det

    image = open_resized_image(record.image_path, int(record.width), int(record.height))
    image = draw_slice_grid(image, frame["slices"])
    draw = ImageDraw.Draw(image)
    for det in detections:
        x1, y1, x2, y2 = det["bbox_xyxy_frame"]
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)
    for det in nms_detections:
        x1, y1, x2, y2 = det["bbox_xyxy_frame"]
        draw.rectangle([x1, y1, x2, y2], outline=(0, 220, 80), width=4)
    draw.rectangle([0, 0, 900, 92], fill=(255, 255, 255))
    draw.text((20, 14), f"frame={record.frame_id}", fill=(0, 0, 0))
    draw.text((20, 38), f"slices={len(frame['slices'])} boxes_before_nms={len(detections)} boxes_after_nms={len(nms_detections)}", fill=(0, 0, 0))
    draw.text((20, 62), "orange=slice grid, red=pre-NMS mapped boxes, green=post-NMS boxes", fill=(0, 0, 0))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    image.save(out)
    print(f"coordinate pipeline passed: before_nms={len(detections)} after_nms={len(nms_detections)} output={out}")


if __name__ == "__main__":
    main()
