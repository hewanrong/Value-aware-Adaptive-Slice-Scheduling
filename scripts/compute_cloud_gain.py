from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from src.dataset.panda_loader import PandaFrameDataset
from src.gain.gain_calculator import compute_slice_gain
from src.utils.config import add_config_arg, load_config


def _gt_by_slice(cfg: dict, slices: dict) -> dict[str, list[dict]]:
    preprocess = cfg.get("preprocess", {})
    dataset = PandaFrameDataset(
        cfg["paths"]["images_dir"],
        cfg["paths"].get("gt_json"),
        cfg["paths"].get("vehicle_gt_json"),
        preprocess.get("target_long_side"),
        preprocess.get("keep_aspect_ratio", True),
    )
    by_frame = {r.frame_id: list(r.annotations) for r in dataset.records()}
    labels = {}
    for frame in slices["frames"]:
        frame_labels = by_frame.get(frame["frame_id"], [])
        for s in frame["slices"]:
            sx, sy, sw, sh = s["bbox"]
            rows = []
            for gt in frame_labels:
                x, y, w, h = gt["bbox"]
                cx, cy = x + w / 2, y + h / 2
                if sx <= cx <= sx + sw and sy <= cy <= sy + sh:
                    rows.append(gt)
            labels[s["slice_id"]] = rows
    return labels


def main() -> None:
    parser = add_config_arg(argparse.ArgumentParser())
    parser.add_argument("--edge-results", default="cache/edge_results.json")
    parser.add_argument("--cloud-results", default="cache/cloud_results_full.json")
    args = parser.parse_args()
    cfg = load_config(args.config)
    slices = json.loads(Path(cfg["paths"]["slices_json"]).read_text(encoding="utf-8"))
    edge = json.loads(Path(args.edge_results).read_text(encoding="utf-8"))
    cloud = json.loads(Path(args.cloud_results).read_text(encoding="utf-8"))
    rows = compute_slice_gain(edge, cloud, _gt_by_slice(cfg, slices), cfg["gain"]["iou_threshold"])
    out = Path(cfg["paths"]["slice_gain_csv"])
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["slice_id", "num_gt", "edge_correct", "cloud_correct", "gain"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} gain rows to {out}")


if __name__ == "__main__":
    main()
