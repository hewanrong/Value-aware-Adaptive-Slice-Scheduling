from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import _bootstrap  # noqa: F401
from src.utils.config import add_config_arg, load_config


def main() -> None:
    parser = add_config_arg(argparse.ArgumentParser())
    args = parser.parse_args()
    cfg = load_config(args.config)
    model_path = Path("results/models/gain_predictor.pkl")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model = {
        "type": "heuristic-baseline",
        "note": "Replace with LightGBM or sklearn GradientBoostingRegressor when enough labels exist.",
        "feature_csv": cfg["paths"]["slice_features_csv"],
        "label_csv": cfg["paths"]["slice_gain_csv"],
    }
    with model_path.open("wb") as f:
        pickle.dump(model, f)
    print(f"Wrote baseline predictor metadata to {model_path}")


if __name__ == "__main__":
    main()
