from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from src.utils.config import add_config_arg, load_config


def mock_detections(slices: dict, source: str = "edge", quality: str = "full") -> list[dict]:
    rows = []
    for frame in slices["frames"]:
        for s in frame["slices"]:
            x, y, w, h = s["bbox"]
            rows.append(
                {
                    "frame_id": frame["frame_id"],
                    "slice_id": s["slice_id"],
                    "bbox": [x + 0.35 * w, y + 0.35 * h, 0.15 * w, 0.15 * h],
                    "score": 0.30,
                    "category_id": 1,
                    "source": source,
                    "quality": quality,
                }
            )
    return rows


def main() -> None:
    parser = add_config_arg(argparse.ArgumentParser())
    parser.add_argument("--output", default="cache/edge_results.json")
    args = parser.parse_args()
    cfg = load_config(args.config)
    existing = cfg["detection"]["edge"].get("existing_results_json")
    if existing:
        rows = json.loads(Path(existing).read_text(encoding="utf-8"))
    else:
        slices = json.loads(Path(cfg["paths"]["slices_json"]).read_text(encoding="utf-8"))
        rows = mock_detections(slices, "edge", "full")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} edge detections to {out}")


if __name__ == "__main__":
    main()
