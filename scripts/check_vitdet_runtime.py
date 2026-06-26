from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import _bootstrap  # noqa: F401
from src.utils.config import add_config_arg, load_config


RUNTIME_CODE = r"""
import importlib.util
import json
import os
from pathlib import Path
import sys

result = {
    "python_executable": sys.executable,
    "python_version": sys.version.split()[0],
    "torch": None,
    "torchvision": None,
    "cuda_available": None,
    "cuda_version": None,
    "gpu_name": None,
    "detectron2": None,
    "detectron2_file": None,
    "vitdet_config_dirs": [],
    "vitdet_s_candidates": [],
    "vitdet_b_candidates": [],
}

try:
    import torch
    result["torch"] = torch.__version__
    result["cuda_available"] = bool(torch.cuda.is_available())
    result["cuda_version"] = getattr(torch.version, "cuda", None)
    result["gpu_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
except Exception as exc:
    result["torch_error"] = repr(exc)

try:
    import torchvision
    result["torchvision"] = torchvision.__version__
except Exception as exc:
    result["torchvision_error"] = repr(exc)

try:
    import detectron2
    result["detectron2"] = getattr(detectron2, "__version__", "installed")
    result["detectron2_file"] = getattr(detectron2, "__file__", None)
except Exception as exc:
    result["detectron2_error"] = repr(exc)

roots = []
if result.get("detectron2_file"):
    p = Path(result["detectron2_file"]).resolve()
    roots.extend([p.parent, *p.parents[:5]])
roots.extend([Path("/home/asus"), Path("/mnt/c/Users/ASUS/Documents/Value-aware Adaptive Slice Scheduling")])

seen_dirs = set()
configs = []
for root in roots:
    if not root.exists():
        continue
    for dirpath, dirnames, filenames in os.walk(root):
        path = Path(dirpath)
        if any(part in {".cache", "__pycache__", ".git", "site-packages"} for part in path.parts) and "projects" not in path.parts:
            dirnames[:] = []
            continue
        if path.parts[-3:] == ("projects", "ViTDet", "configs"):
            s = str(path)
            if s not in seen_dirs:
                seen_dirs.add(s)
                result["vitdet_config_dirs"].append(s)
        for name in filenames:
            lower = name.lower()
            if "vitdet" in lower and lower.endswith((".yaml", ".yml")):
                configs.append(str(path / name))
        if len(configs) > 200:
            break

for cfg in sorted(set(configs)):
    lower = Path(cfg).name.lower()
    if any(token in lower for token in ["vitdet_s", "vitdet-s", "_s_", "s_"]):
        result["vitdet_s_candidates"].append(cfg)
    if any(token in lower for token in ["vitdet_b", "vitdet-b", "_b_", "b_"]):
        result["vitdet_b_candidates"].append(cfg)

print(json.dumps(result, indent=2))
bad = (not result.get("cuda_available")) or (result.get("detectron2") is None)
raise SystemExit(2 if bad else 0)
"""


def main() -> None:
    parser = add_config_arg(argparse.ArgumentParser())
    args = parser.parse_args()
    cfg = load_config(args.config)
    vitdet = cfg.get("vitdet", cfg.get("detection", {}).get("vitdet", {}))
    distro = vitdet.get("wsl_distro") or "Ubuntu"
    python_exe = vitdet.get("python_executable") or "/home/asus/miniforge3/envs/vitdet/bin/python"
    cmd = ["wsl", "-d", distro, "--", python_exe, "-c", RUNTIME_CODE]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    output = ((proc.stdout or "") + (proc.stderr or "")).strip()
    print(output.encode("gbk", errors="replace").decode("gbk"))
    raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
