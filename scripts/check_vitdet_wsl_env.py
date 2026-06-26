from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

import _bootstrap  # noqa: F401
from src.utils.config import add_config_arg, load_config


def _safe_print(text: str) -> None:
    print(text.encode("gbk", errors="replace").decode("gbk"))


def _cfg_value(cfg: dict, key: str, env_key: str | None = None) -> str | None:
    value = cfg.get("vitdet", {}).get(key)
    if value is None and env_key:
        value = os.environ.get(env_key)
    return value


def _run_wsl(distro: str | None, command: str, timeout: int = 20) -> tuple[bool, str]:
    base = ["wsl"]
    if distro:
        base.extend(["-d", distro])
    base.extend(["--", "bash", "-lc", command])
    try:
        proc = subprocess.run(base, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
    except FileNotFoundError:
        return False, "wsl.exe not found"
    except subprocess.TimeoutExpired:
        return False, "timed out"
    output = ((proc.stdout or "") + (proc.stderr or "")).strip()
    return proc.returncode == 0, output


def main() -> None:
    parser = add_config_arg(argparse.ArgumentParser())
    args = parser.parse_args()
    cfg = load_config(args.config)
    distro = _cfg_value(cfg, "wsl_distro", "VITDET_WSL_DISTRO")
    py = _cfg_value(cfg, "python_executable", "VITDET_PYTHON") or "python"
    repo = _cfg_value(cfg, "vitdet_repo_root", "VITDET_REPO_ROOT")
    edge_config = _cfg_value(cfg, "edge_config", "VITDET_EDGE_CONFIG")
    edge_ckpt = _cfg_value(cfg, "edge_checkpoint", "VITDET_EDGE_CHECKPOINT")
    cloud_config = _cfg_value(cfg, "cloud_config", "VITDET_CLOUD_CONFIG")
    cloud_ckpt = _cfg_value(cfg, "cloud_checkpoint", "VITDET_CLOUD_CHECKPOINT")

    _safe_print(f"WSL distro: {distro or 'default/unset'}")
    ok, out = _run_wsl(distro, "printf ok")
    _safe_print(f"WSL available: {ok} ({out})")
    if not ok:
        _safe_print("Python version: unavailable")
        _safe_print("PyTorch version: unavailable")
        _safe_print("CUDA available: unavailable")
        _safe_print("GPU name: unavailable")
        _safe_print("Detectron2 import: unavailable")
    else:
        for label, command in [
            ("Python version", f"{py} -c \"import sys; print(sys.version.split()[0])\""),
            ("PyTorch version", f"{py} -c \"import torch; print(torch.__version__)\""),
            ("CUDA available", f"{py} -c \"import torch; print(torch.cuda.is_available())\""),
            ("GPU name", f"{py} -c \"import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')\""),
            ("Detectron2 import", f"{py} -c \"import detectron2; print('ok')\""),
        ]:
            cmd_ok, cmd_out = _run_wsl(distro, command)
            _safe_print(f"{label}: {'ok' if cmd_ok else 'missing'} ({cmd_out})")

    for label, path in [
        ("ViTDet repo root", repo),
        ("Edge config", edge_config),
        ("Edge checkpoint", edge_ckpt),
        ("Cloud config", cloud_config),
        ("Cloud checkpoint", cloud_ckpt),
    ]:
        if not path:
            _safe_print(f"{label}: unset")
        elif ok:
            exists_ok, exists_out = _run_wsl(distro, f"test -e {path!r} && printf exists || printf missing")
            _safe_print(f"{label}: {exists_out}")
        else:
            _safe_print(f"{label}: configured but not checked ({path})")


if __name__ == "__main__":
    main()
