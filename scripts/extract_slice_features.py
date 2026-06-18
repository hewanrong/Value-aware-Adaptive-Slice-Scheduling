from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from src.features.tracking_features import extract_tracking_features
from src.features.visual_features import extract_visual_features
from src.utils.config import add_config_arg, load_config


def main() -> None:
    parser = add_config_arg(argparse.ArgumentParser())
    parser.add_argument("--max-slices", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    index = json.loads(Path(cfg["paths"]["slices_json"]).read_text(encoding="utf-8"))
    rows = []
    for frame in index["frames"]:
        for s in frame["slices"]:
            row = {"frame_id": frame["frame_id"], "slice_id": s["slice_id"]}
            row.update(extract_visual_features(frame["image_path"], s["bbox"]))
            row.update(extract_tracking_features([], s["bbox"]))
            rows.append(row)
            if args.max_slices and len(rows) >= args.max_slices:
                break
        if args.max_slices and len(rows) >= args.max_slices:
            break
    out = Path(cfg["paths"]["slice_features_csv"])
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["frame_id", "slice_id"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} feature rows to {out}")


if __name__ == "__main__":
    main()
