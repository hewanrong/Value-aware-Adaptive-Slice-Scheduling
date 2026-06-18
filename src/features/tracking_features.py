from __future__ import annotations


def extract_tracking_features(tracks: list[dict] | None, slice_bbox: list[float]) -> dict[str, float]:
    tracks = tracks or []
    _, _, sw, sh = slice_bbox
    slice_area = max(1.0, float(sw * sh))
    areas = [float(t.get("bbox", [0, 0, 0, 0])[2] * t.get("bbox", [0, 0, 0, 0])[3]) for t in tracks]
    confidences = [float(t.get("confidence", 0.0)) for t in tracks]
    ages = [float(t.get("age", 0.0)) for t in tracks]
    velocities = [float(t.get("velocity", 0.0)) for t in tracks]
    return {
        "tracked_object_count": float(len(tracks)),
        "mean_track_box_area": _mean(areas),
        "min_track_box_area": min(areas) if areas else 0.0,
        "small_track_ratio": float(sum(a / slice_area < 0.01 for a in areas) / len(areas)) if areas else 0.0,
        "track_density": float(len(tracks) / slice_area),
        "track_confidence_mean": _mean(confidences),
        "track_lost_count": float(sum(bool(t.get("lost", False)) for t in tracks)),
        "track_age_mean": _mean(ages),
        "track_velocity_mean": _mean(velocities),
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
