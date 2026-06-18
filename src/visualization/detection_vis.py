from __future__ import annotations


def detection_panel_spec() -> dict:
    return {
        "panels": ["Original / GT", "Edge Only", "Baseline", "Ours"],
        "colors": {
            "correct_detection": "green",
            "missed_gt": "red",
            "cloud_gain_detection": "blue",
            "tracking_reused_detection": "yellow",
        },
    }
