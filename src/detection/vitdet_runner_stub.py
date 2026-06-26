from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath


@dataclass(frozen=True)
class VitDetCommandTemplate:
    wsl_distro: str | None
    python_executable: str | None
    vitdet_repo_root: str | None
    config_path: str | None
    checkpoint_path: str | None
    input_path: str | None
    output_path: str | None
    device: str | None = None

    def render(self) -> str:
        if not self.wsl_distro:
            return "WSL distro is not configured."
        py = self.python_executable or "python"
        repo = self.vitdet_repo_root or "$VITDET_REPO_ROOT"
        device = self.device or "cuda"
        # This is a command template only. It intentionally does not run inference
        # until paths and the WSL Detectron2 environment are verified.
        body = (
            f"cd {repo} && {py} demo/demo.py "
            f"--config-file {self.config_path} "
            f"--input {self.input_path} --output {self.output_path} "
            f"--opts MODEL.WEIGHTS {self.checkpoint_path} MODEL.DEVICE {device}"
        )
        return f"wsl -d {self.wsl_distro} -- bash -lc {body!r}"


def missing_vitdet_fields(config: dict) -> list[str]:
    required = ["wsl_distro", "python_executable", "vitdet_repo_root", "config_path", "checkpoint_path"]
    return [key for key in required if not config.get(key)]


def windows_to_wsl_path(path: str) -> str:
    if len(path) >= 3 and path[1] == ":" and path[2] in {"\\", "/"}:
        drive = path[0].lower()
        rest = path[3:].replace("\\", "/")
        return str(PurePosixPath("/mnt") / drive / rest)
    return path
