from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
from src.dataset.panda_loader import PandaFrameDataset
from src.slicing.sahi_slicer import build_slice_index, save_slice_index
from src.utils.config import add_config_arg, load_config


def main() -> None:
    parser = add_config_arg(argparse.ArgumentParser())
    parser.add_argument("--images-dir", default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = cfg["paths"]
    preprocess = cfg.get("preprocess", {})
    max_frames = args.max_frames if args.max_frames is not None else cfg.get("experiment", {}).get("max_frames")
    dataset = PandaFrameDataset(
        args.images_dir or paths["images_dir"],
        paths.get("gt_json"),
        paths.get("vehicle_gt_json"),
        preprocess.get("target_long_side"),
        preprocess.get("keep_aspect_ratio", True),
    )
    records = dataset.records(max_frames=max_frames)
    index = build_slice_index(records, cfg["slicing"]["slice_size"], cfg["slicing"]["overlap_ratio"])
    save_slice_index(index, paths["slices_json"])
    print(f"Wrote {index['num_slices']} slices for {index['num_frames']} frames to {paths['slices_json']}")


if __name__ == "__main__":
    main()
