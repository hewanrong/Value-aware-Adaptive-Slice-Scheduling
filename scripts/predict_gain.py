from __future__ import annotations

import argparse
import csv
from pathlib import Path

import _bootstrap  # noqa: F401
from src.gain.gain_predictor import HeuristicGainPredictor
from src.utils.config import add_config_arg, load_config


def main() -> None:
    parser = add_config_arg(argparse.ArgumentParser())
    args = parser.parse_args()
    cfg = load_config(args.config)
    src = Path(cfg["paths"]["slice_features_csv"])
    out = Path(cfg["paths"]["predicted_gain_csv"])
    rows = list(csv.DictReader(src.open("r", encoding="utf-8")))
    predictor = HeuristicGainPredictor()
    pred_rows = []
    for row in rows:
        pred_rows.append({"frame_id": row["frame_id"], "slice_id": row["slice_id"], **predictor.predict_row(row)})
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["frame_id", "slice_id", "V_low", "V_mid", "V_high", "V_full"])
        writer.writeheader()
        writer.writerows(pred_rows)
    print(f"Wrote {len(pred_rows)} predicted gain rows to {out}")


if __name__ == "__main__":
    main()
