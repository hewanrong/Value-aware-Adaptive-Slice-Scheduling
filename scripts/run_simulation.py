from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from src.simulation.offline_simulator import run_offline_simulation
from src.utils.config import add_config_arg, load_config


def main() -> None:
    parser = add_config_arg(argparse.ArgumentParser())
    parser.add_argument("--edge-results", default="cache/edge_results.json")
    parser.add_argument("--cloud-results", default="cache/cloud_results_full.json")
    args = parser.parse_args()
    cfg = load_config(args.config)
    edge = json.loads(Path(args.edge_results).read_text(encoding="utf-8"))
    cloud = json.loads(Path(args.cloud_results).read_text(encoding="utf-8"))
    decisions = list(csv.DictReader(Path(cfg["paths"]["schedule_decisions_csv"]).open("r", encoding="utf-8")))
    result = run_offline_simulation(edge, cloud, decisions)
    metrics_path = Path(cfg["paths"]["metrics_csv"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(result["metrics"].keys()))
        writer.writeheader()
        writer.writerow(result["metrics"])
    frame_path = Path(cfg["paths"]["frame_results_json"])
    frame_path.write_text(json.dumps(result["frame_results"], indent=2), encoding="utf-8")
    print(f"Wrote metrics to {metrics_path} and frame results to {frame_path}")


if __name__ == "__main__":
    main()
