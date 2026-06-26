from __future__ import annotations

import argparse
import csv
from pathlib import Path

import _bootstrap  # noqa: F401
from PIL import Image, ImageDraw

from src.utils.config import add_config_arg, load_config


def _read_rows(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    return list(csv.DictReader(p.open("r", encoding="utf-8")))


def _save_bar(values: list[tuple[str, float]], out: Path, title: str) -> None:
    img = Image.new("RGB", (900, 520), "white")
    draw = ImageDraw.Draw(img)
    draw.text((30, 20), title, fill="black")
    max_v = max([v for _, v in values] or [1])
    x = 80
    for label, value in values:
        h = int(360 * value / max_v) if max_v else 0
        draw.rectangle([x, 440 - h, x + 120, 440], fill=(41, 121, 255))
        draw.text((x, 450), label, fill="black")
        draw.text((x, 420 - h), f"{value:.2f}", fill="black")
        x += 180
    img.save(out)


def _save_hist(values: list[float], out: Path, title: str, bins: int = 20) -> None:
    img = Image.new("RGB", (900, 520), "white")
    draw = ImageDraw.Draw(img)
    draw.text((30, 20), title, fill="black")
    if not values:
        img.save(out)
        return
    lo, hi = min(values), max(values)
    if lo == hi:
        hi = lo + 1.0
    counts = [0] * bins
    for v in values:
        idx = min(bins - 1, int((v - lo) / (hi - lo) * bins))
        counts[idx] += 1
    max_c = max(counts) or 1
    bar_w = 760 / bins
    for i, c in enumerate(counts):
        x1 = 70 + i * bar_w
        h = int(360 * c / max_c)
        draw.rectangle([x1, 440 - h, x1 + bar_w - 2, 440], fill=(0, 200, 83))
    draw.text((70, 455), f"{lo:.3g}", fill="black")
    draw.text((780, 455), f"{hi:.3g}", fill="black")
    img.save(out)


def _save_scatter(rows: list[dict], x_key: str, y_key: str, out: Path, title: str) -> None:
    img = Image.new("RGB", (900, 520), "white")
    draw = ImageDraw.Draw(img)
    draw.text((30, 20), title, fill="black")
    pts = []
    for r in rows:
        try:
            pts.append((float(r[x_key]), float(r[y_key])))
        except (KeyError, ValueError):
            pass
    if not pts:
        img.save(out)
        return
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    if min_x == max_x:
        max_x = min_x + 1.0
    if min_y == max_y:
        max_y = min_y + 1.0
    draw.rectangle([70, 60, 840, 440], outline="black")
    for x, y in pts:
        px = 70 + (x - min_x) / (max_x - min_x) * 770
        py = 440 - (y - min_y) / (max_y - min_y) * 380
        draw.ellipse([px - 2, py - 2, px + 2, py + 2], fill=(255, 109, 0))
    draw.text((70, 455), x_key, fill="black")
    draw.text((10, 240), y_key, fill="black")
    img.save(out)


def _save_density_heatmap(rows: list[dict], out: Path) -> None:
    first_frame = rows[0]["frame_id"] if rows else ""
    frame_rows = [r for r in rows if r.get("frame_id") == first_frame]
    img = Image.new("RGB", (900, 520), "white")
    draw = ImageDraw.Draw(img)
    draw.text((30, 20), f"slice_gt_density_heatmap: {first_frame}", fill="black")
    if not frame_rows:
        img.save(out)
        return
    max_x = max(float(r["slice_x2"]) for r in frame_rows)
    max_y = max(float(r["slice_y2"]) for r in frame_rows)
    max_gt = max(float(r["total_gt_count"]) for r in frame_rows) or 1.0
    scale = min(800 / max_x, 420 / max_y)
    ox, oy = 50, 70
    for r in frame_rows:
        x1, y1 = float(r["slice_x1"]), float(r["slice_y1"])
        x2, y2 = float(r["slice_x2"]), float(r["slice_y2"])
        intensity = int(255 * float(r["total_gt_count"]) / max_gt)
        color = (255, 255 - intensity, 255 - intensity)
        draw.rectangle([ox + x1 * scale, oy + y1 * scale, ox + x2 * scale, oy + y2 * scale], fill=color, outline=(180, 180, 180))
    img.save(out)


def main() -> None:
    parser = add_config_arg(argparse.ArgumentParser())
    parser.add_argument("--output-dir", default="results/phase1/summary")
    args = parser.parse_args()
    cfg = load_config(args.config)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gt_rows = _read_rows("cache/slice_gt_map.csv")
    feature_rows = _read_rows(cfg["paths"]["slice_features_csv"])
    cost_rows = _read_rows(cfg["paths"]["slice_costs_csv"])
    _save_density_heatmap(gt_rows, out_dir / "slice_gt_density_heatmap.png")
    empty = sum(1 for r in gt_rows if int(r.get("total_gt_count", 0)) == 0)
    non_empty = max(0, len(gt_rows) - empty)
    _save_bar([("empty", empty), ("non-empty", non_empty)], out_dir / "empty_non_empty_distribution.png", "empty / non-empty slices")
    _save_hist([float(r.get("mean_gt_area", 0)) for r in gt_rows if float(r.get("mean_gt_area", 0)) > 0], out_dir / "bbox_size_distribution.png", "bbox size distribution")
    _save_hist([float(r.get("bytes_full", 0)) for r in cost_rows], out_dir / "jpeg_cost_distribution.png", "JPEG full-quality byte distribution")
    _save_scatter(feature_rows, "entropy", "gt_density", out_dir / "entropy_vs_gt_density.png", "entropy vs gt_density")
    _save_scatter(feature_rows, "edge_density", "gt_density", out_dir / "edge_density_vs_gt_density.png", "edge_density vs gt_density")
    print(f"wrote Phase 1 summary visualizations to {out_dir}")


if __name__ == "__main__":
    main()
