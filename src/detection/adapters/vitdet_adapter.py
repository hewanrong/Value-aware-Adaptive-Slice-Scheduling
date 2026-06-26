from __future__ import annotations

from .base import DetectorAdapter


class VitDetAdapter(DetectorAdapter):
    def __init__(self, config: dict | None = None, role: str = "cloud") -> None:
        vitdet = (config or {}).get("vitdet", (config or {}).get("detection", {}).get("vitdet", {}))
        model_name = vitdet.get(f"{role}_model_name") or ("vitdet_b" if role == "cloud" else "unresolved")
        super().__init__(vitdet, model_name=model_name, backend="vitdet")
        self.role = role

    def validate_config(self) -> None:
        missing = []
        for key in ["wsl_distro", "python_executable", "vitdet_repo_root", f"{self.role}_config", f"{self.role}_checkpoint"]:
            if not self.config.get(key):
                missing.append(f"vitdet.{key}")
        if missing:
            raise ValueError(
                "ViTDet adapter is not ready. Missing required fields: "
                + ", ".join(missing)
                + ". No fallback to mock should be performed."
            )

    def predict_slice(self, frame_id: str, slice_id: str, image, slice_bbox: list[float]) -> list[dict]:
        self.validate_config()
        raise NotImplementedError("Real ViTDet inference is not enabled until config and checkpoint are confirmed.")

    def normalize_predictions(
        self,
        predictions: list[dict],
        *,
        frame_id: str,
        slice_id: str,
        input_width: int,
        input_height: int,
    ) -> list[dict]:
        raise NotImplementedError("ViTDet output normalization will be enabled with real inference.")
