from __future__ import annotations


class HeuristicGainPredictor:
    """Dependency-free baseline used until LightGBM/sklearn training is enabled."""

    def predict_row(self, features: dict) -> dict[str, float]:
        density = float(features.get("edge_density", 0.0))
        contrast = float(features.get("local_contrast", 0.0))
        base = max(0.0, density * 0.5 + min(contrast / 255.0, 1.0) * 0.5)
        return {
            "V_low": base * 0.4,
            "V_mid": base * 0.65,
            "V_high": base * 0.85,
            "V_full": base,
        }
