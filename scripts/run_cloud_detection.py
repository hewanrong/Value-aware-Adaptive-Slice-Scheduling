from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from src.utils.config import add_config_arg, load_config


QUALITY_SCORE = {"low": 0.45, "mid": 0.60, "high": 0.75, "full": 0.90}
QUALITY_SCALE = {"low": 0.12, "mid": 0.14, "high": 0.16, "full": 0.18}


def mock_cloud_detections(slices: dict, quality: str) -> list[dict]:
    rows = []
    for frame in slices["frames"]:
        for s in frame["slices"]:
            x, y, w, h = s["bbox"]
            size = QUALITY_SCALE[quality]
            rows.append(
                {
                    "frame_id": frame["frame_id"],
                    "slice_id": s["slice_id"],
                    "bbox": [x + 0.33 * w, y + 0.33 * h, size * w, size * h],
                    "score": QUALITY_SCORE[quality],
                    "category_id": 1,
                    "source": "cloud",
                    "quality": quality,
                }
            )
    return rows


def main() -> None:
    parser = add_config_arg(argparse.ArgumentParser())
    parser.add_argument("--quality", choices=["low", "mid", "high", "full"], default="full")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    existing = cfg["detection"]["cloud"].get("existing_results_json")
    if existing:
        rows = json.loads(Path(existing).read_text(encoding="utf-8"))
    else:
        slices = json.loads(Path(cfg["paths"]["slices_json"]).read_text(encoding="utf-8"))
        rows = mock_cloud_detections(slices, args.quality)
    out = Path(args.output or f"cache/cloud_results_{args.quality}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} cloud detections to {out}")


if __name__ == "__main__":
    main()
