from __future__ import annotations

import argparse
import csv
from pathlib import Path

import _bootstrap  # noqa: F401
from src.scheduler.learned_scheduler import schedule_learned
from src.scheduler.oracle_scheduler import schedule_oracle
from src.utils.config import add_config_arg, load_config


def _read_csv(path: str) -> dict[str, dict]:
    return {r["slice_id"]: r for r in csv.DictReader(Path(path).open("r", encoding="utf-8"))}


def main() -> None:
    parser = add_config_arg(argparse.ArgumentParser())
    parser.add_argument("--mode", choices=["learned", "oracle"], default=None)
    parser.add_argument("--budget", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    mode = args.mode or cfg["scheduler"]["mode"]
    budget = args.budget or cfg["scheduler"].get("fixed_bandwidth_bytes_per_frame") or 5_000_000
    gains = _read_csv(cfg["paths"]["predicted_gain_csv"])
    costs = _read_csv(cfg["paths"]["slice_costs_csv"])
    items = []
    for slice_id, gain in gains.items():
        if slice_id not in costs:
            continue
        item = {"slice_id": slice_id, "frame_id": gain["frame_id"]}
        item.update({k: float(v) for k, v in gain.items() if k.startswith("V_")})
        item.update({k: float(v) for k, v in costs[slice_id].items() if k.startswith("bytes_")})
        items.append(item)
    qualities = cfg["scheduler"]["qualities"]
    decisions = schedule_oracle(items, budget, qualities) if mode == "oracle" else schedule_learned(items, budget, qualities)
    out = Path(cfg["paths"]["schedule_decisions_csv"])
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["frame_id", "slice_id", "action"])
        writer.writeheader()
        writer.writerows(decisions)
    print(f"Wrote {len(decisions)} schedule decisions to {out}")


if __name__ == "__main__":
    main()
