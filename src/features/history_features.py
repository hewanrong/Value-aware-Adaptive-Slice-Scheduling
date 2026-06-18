from __future__ import annotations

from collections import deque


class HistoryFeatureState:
    def __init__(self, window: int = 10, ewma_alpha: float = 0.3) -> None:
        self.window = window
        self.ewma_alpha = ewma_alpha
        self._gains: deque[float] = deque(maxlen=window)
        self._cloud_checks: deque[int] = deque(maxlen=window)
        self._last_check_frame: int | None = None
        self._ewma = 0.0

    def features(self, frame_index: int) -> dict[str, float]:
        gains = list(self._gains)
        return {
            "last_observed_gain": gains[-1] if gains else 0.0,
            "ewma_observed_gain": self._ewma,
            "observed_gain_variance": float(_variance(gains)),
            "cloud_edge_disagreement_rate": float(sum(g > 0 for g in gains) / len(gains)) if gains else 0.0,
            "time_since_last_cloud_check": float(frame_index - self._last_check_frame) if self._last_check_frame is not None else -1.0,
            "num_recent_cloud_checks": float(sum(self._cloud_checks)),
        }

    def observe(self, frame_index: int, observed_gain: float | None, cloud_checked: bool) -> None:
        self._cloud_checks.append(1 if cloud_checked else 0)
        if cloud_checked:
            self._last_check_frame = frame_index
        if observed_gain is not None:
            self._gains.append(float(observed_gain))
            self._ewma = self.ewma_alpha * observed_gain + (1.0 - self.ewma_alpha) * self._ewma


def _variance(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return sum((v - mean) ** 2 for v in values) / len(values)
