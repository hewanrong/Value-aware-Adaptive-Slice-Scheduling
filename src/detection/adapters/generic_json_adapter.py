from __future__ import annotations

from src.detection.result_schema import validate_detection

from .base import DetectorAdapter


class GenericJsonAdapter(DetectorAdapter):
    """Adapter for future external detectors that export the unified JSON schema."""

    def __init__(self, config: dict | None = None, model_name: str = "external_detector", backend: str = "generic_json") -> None:
        super().__init__(config, model_name, backend)

    def validate_config(self) -> None:
        return None

    def predict_slice(self, frame_id: str, slice_id: str, image, slice_bbox: list[float]) -> list[dict]:
        raise NotImplementedError("GenericJsonAdapter consumes exported JSON; it does not run external frameworks.")

    def normalize_predictions(
        self,
        predictions: list[dict],
        *,
        frame_id: str,
        slice_id: str,
        input_width: int,
        input_height: int,
    ) -> list[dict]:
        return [validate_detection(row) for row in predictions]
