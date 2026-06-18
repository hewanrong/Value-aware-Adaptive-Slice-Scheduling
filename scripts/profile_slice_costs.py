from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from src.scheduler.cost_profiler import profile_slice_costs
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
            row.update(profile_slice_costs(frame["image_path"], s["bbox"], cfg["cost_profiler"]["jpeg_quality"]))
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
    print(f"Wrote {len(rows)} cost rows to {out}")


if __name__ == "__main__":
    main()
