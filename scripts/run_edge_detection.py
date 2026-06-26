from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from src.dataset.panda_loader import PandaFrameDataset
from src.detection.coordinate_utils import clip_bbox_to_frame, local_to_frame_bbox
from src.detection.adapters import MockDetectorAdapter
from src.detection.vitdet_runner_stub import missing_vitdet_fields
from src.slicing.sahi_slicer import build_slice_index
from src.utils.config import add_config_arg, load_config


def mock_detections(slices: dict, source: str = "edge", quality: str = "full") -> list[dict]:
    rows = []
    adapter = MockDetectorAdapter(model_name="mock_edge_detector", backend="mock")
    for frame in slices["frames"]:
        for s in frame["slices"]:
            x, y, w, h = s["bbox"]
            pred = adapter.normalize_predictions(
                adapter.predict_slice(frame["frame_id"], s["slice_id"], None, s["bbox"]),
                frame_id=frame["frame_id"],
                slice_id=s["slice_id"],
                input_width=int(w),
                input_height=int(h),
            )[0]
            local = pred["bbox_xyxy"]
            slice_xyxy = [x, y, x + w, y + h]
            frame_box = clip_bbox_to_frame(local_to_frame_bbox(local, slice_xyxy), frame["width"], frame["height"])
            pred.update(
                {
                    "bbox": [frame_box[0], frame_box[1], frame_box[2] - frame_box[0], frame_box[3] - frame_box[1]],
                    "bbox_xyxy_local": local,
                    "bbox_xyxy_frame": frame_box,
                    "category_id": pred["class_id"],
                    "source": source,
                    "quality": quality,
                    "slice_x1": x,
                    "slice_y1": y,
                    "slice_x2": x + w,
                    "slice_y2": y + h,
                }
            )
            rows.append(pred)
    return rows


def _slice_index_from_config(cfg: dict, max_frames: int | None) -> dict:
    preprocess = cfg.get("preprocess", {})
    dataset = PandaFrameDataset(
        cfg["paths"]["images_dir"],
        cfg["paths"].get("gt_json"),
        cfg["paths"].get("vehicle_gt_json"),
        preprocess.get("target_long_side"),
        preprocess.get("keep_aspect_ratio", True),
    )
    records = dataset.records(max_frames=max_frames)
    return build_slice_index(records, cfg["slicing"]["slice_size"], cfg["slicing"]["overlap_ratio"])


def _diagnose_vitdet(cfg: dict) -> None:
    vitdet = cfg.get("vitdet", cfg.get("detection", {}).get("vitdet", {}))
    normalized = {
        "wsl_distro": vitdet.get("wsl_distro"),
        "python_executable": vitdet.get("python_executable"),
        "vitdet_repo_root": vitdet.get("vitdet_repo_root"),
        "config_path": vitdet.get("edge_config") or vitdet.get("config_path"),
        "checkpoint_path": vitdet.get("edge_checkpoint") or vitdet.get("checkpoint_path"),
    }
    missing = missing_vitdet_fields(normalized)
    if missing:
        print("ViTDet backend requested, but required configuration is missing:")
        for key in missing:
            if key == "config_path":
                print("- vitdet.edge_config")
            elif key == "checkpoint_path":
                print("- vitdet.edge_checkpoint")
            else:
                print(f"- vitdet.{key}")
        print("No fallback to mock was performed.")
        print("Run scripts/check_vitdet_runtime.py --config configs/vitdet_example.yaml for runtime diagnostics.")
        raise SystemExit(2)
    print("ViTDet backend requested and configuration fields are present.")
    print("Minimal inference execution is not enabled until a specific config/checkpoint is confirmed.")
    raise SystemExit(2)


def main() -> None:
    parser = add_config_arg(argparse.ArgumentParser())
    parser.add_argument("--backend", choices=["mock", "vitdet"], default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--max-slices", type=int, default=None)
    parser.add_argument("--output", default="cache/edge_results.json")
    args = parser.parse_args()
    cfg = load_config(args.config)
    backend = args.backend or cfg["detection"]["edge"].get("mode", "mock")
    if backend == "vitdet":
        _diagnose_vitdet(cfg)
    existing = cfg["detection"]["edge"].get("existing_results_json")
    if existing:
        rows = json.loads(Path(existing).read_text(encoding="utf-8"))
    else:
        slices = _slice_index_from_config(cfg, args.max_frames)
        rows = mock_detections(slices, "edge", "full")
    if args.max_slices:
        rows = rows[: args.max_slices]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} edge detections to {out}")


if __name__ == "__main__":
    main()
