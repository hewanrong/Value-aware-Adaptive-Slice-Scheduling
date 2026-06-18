from __future__ import annotations

import argparse
import csv
from pathlib import Path

import _bootstrap  # noqa: F401
from src.utils.config import add_config_arg, load_config


def main() -> None:
    parser = add_config_arg(argparse.ArgumentParser())
    args = parser.parse_args()
    cfg = load_config(args.config)
    out_dir = Path(cfg["visualization"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = out_dir / "summary.txt"
    lines = ["Visualization scaffold\n", "Expected figures: gain heatmap, gain CDF, schedule trace, detection panels.\n"]
    metrics = Path(cfg["paths"]["metrics_csv"])
    if metrics.exists():
        rows = list(csv.DictReader(metrics.open("r", encoding="utf-8")))
        lines.append(f"Metrics rows: {rows}\n")
    summary.write_text("".join(lines), encoding="utf-8")
    print(f"Wrote visualization scaffold summary to {summary}")


if __name__ == "__main__":
    main()
