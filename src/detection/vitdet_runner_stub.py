from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VitDetCommandTemplate:
    wsl_command: str | None
    config_path: str | None
    checkpoint_path: str | None
    input_path: str | None
    output_path: str | None

    def render(self) -> str:
        # TODO: Fill ViTDet config path once WSL Detectron2 environment is ready.
        # TODO: Fill checkpoint path once model weights are available.
        # TODO: Fill image path or slice directory path from config.
        # TODO: Fill output path for unified JSON conversion.
        if not self.wsl_command:
            return "WSL ViTDet command is not configured."
        return (
            f"{self.wsl_command} --config-file {self.config_path} "
            f"--input {self.input_path} --output {self.output_path} "
            f"MODEL.WEIGHTS {self.checkpoint_path}"
        )
