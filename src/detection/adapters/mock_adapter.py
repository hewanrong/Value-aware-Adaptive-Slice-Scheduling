from __future__ import annotations

from src.detection.result_schema import canonical_detection_record

from .base import DetectorAdapter


class MockDetectorAdapter(DetectorAdapter):
    def __init__(self, config: dict | None = None, model_name: str = "mock_detector", backend: str = "mock") -> None:
        super().__init__(config, model_name, backend)

    def validate_config(self) -> None:
        return None

    def predict_slice(self, frame_id: str, slice_id: str, image, slice_bbox: list[float]) -> list[dict]:
        _, _, w, h = slice_bbox
        return [{"bbox_xyxy": [0.35 * w, 0.35 * h, 0.50 * w, 0.50 * h], "class_id": 1, "score": 0.30}]

    def normalize_predictions(
        self,
        predictions: list[dict],
        *,
        frame_id: str,
        slice_id: str,
        input_width: int,
        input_height: int,
    ) -> list[dict]:
        rows = []
        for pred in predictions:
            rows.append(
                canonical_detection_record(
                    frame_id=frame_id,
                    slice_id=slice_id,
                    bbox_xyxy=list(pred["bbox_xyxy"]),
                    class_id=int(pred.get("class_id", pred.get("category_id", 1))),
                    score=float(pred["score"]),
                    model_name=self.model_name,
                    backend=self.backend,
                    input_width=input_width,
                    input_height=input_height,
                    inference_time_ms=float(pred.get("inference_time_ms", 0.0)),
                )
            )
        return rows
