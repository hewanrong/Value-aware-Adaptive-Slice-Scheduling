from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class DetectorAdapter(ABC):
    """Minimal model adapter interface for slice-local detector outputs."""

    def __init__(self, config: dict | None = None, model_name: str = "unknown", backend: str = "unknown") -> None:
        self.config = config or {}
        self.model_name = model_name
        self.backend = backend

    @abstractmethod
    def validate_config(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict_slice(self, frame_id: str, slice_id: str, image: Any, slice_bbox: list[float]) -> list[dict]:
        raise NotImplementedError

    @abstractmethod
    def normalize_predictions(
        self,
        predictions: list[dict],
        *,
        frame_id: str,
        slice_id: str,
        input_width: int,
        input_height: int,
    ) -> list[dict]:
        raise NotImplementedError

    def model_metadata(self) -> dict:
        return {"model_name": self.model_name, "backend": self.backend}
