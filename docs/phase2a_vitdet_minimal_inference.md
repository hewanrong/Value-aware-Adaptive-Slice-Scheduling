# Phase 2A Minimal ViTDet Inference Prep

Phase 2A prepares the minimal real ViTDet inference interface. It does not batch-run PANDA, train models, modify scheduler/simulator/Gain Predictor logic, or download checkpoints.

Current status: Phase 2A is paused at model selection. Runtime validation and config discovery are complete, but checkpoint download and real inference have not started.

## Runtime Status

Checked through WSL Ubuntu using:

```text
/home/asus/miniforge3/envs/vitdet/bin/python
```

Observed runtime:

- Python: 3.11.15
- torch: 2.5.1+cu121
- torchvision: 0.20.1+cu121
- CUDA available: true
- CUDA version: 12.1
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU
- detectron2: 0.6
- detectron2 import path: `/home/asus/miniforge3/envs/vitdet/lib/python3.11/site-packages/detectron2/__init__.py`

## ViTDet Config Discovery

The runtime check searched for:

- `projects/ViTDet/configs`
- YAML files with `vitdet` in their names
- ViTDet-S style candidates
- ViTDet-B style candidates

Current result:

- `projects/ViTDet/configs`: not found
- ViTDet-S config candidates: none found
- ViTDet-B config candidates: none found

This means Detectron2 is installed, but the ViTDet project config files are not currently available in the searched WSL paths.

## Current Config

`configs/vitdet_example.yaml` is configured for the runtime but leaves model-specific fields unset:

```yaml
vitdet:
  wsl_distro: Ubuntu
  python_executable: /home/asus/miniforge3/envs/vitdet/bin/python
  vitdet_repo_root: /home/asus/repos/detectron2
  cloud_model_name: vitdet_b
  edge_model_name: unresolved
  edge_model_status: pending_hyperion_or_replacement
  cloud_model_status: pending_checkpoint
  edge_config: null
  edge_checkpoint: null
  cloud_config: null
  cloud_checkpoint: null
  device: cuda
```

## Required Before Real Inference

Before running `--backend vitdet`, provide:

- ViTDet source/repo root containing `projects/ViTDet/configs`
- Edge config path for the final edge model. This is unresolved and must not be assumed to be ViTDet-B.
- Cloud config path, provisionally ViTDet-B once a reproducible config/checkpoint is selected.
- Edge checkpoint path
- Cloud checkpoint path

Do not download checkpoints until candidate URLs and file sizes have been listed and confirmed.

## Minimal Interface Behavior

`scripts/run_edge_detection.py` and `scripts/run_cloud_detection.py` support:

```powershell
python scripts/run_edge_detection.py --config configs/vitdet_example.yaml --backend vitdet --max-frames 1 --max-slices 2
python scripts/run_cloud_detection.py --config configs/vitdet_example.yaml --backend vitdet --quality full --max-frames 1 --max-slices 2
```

If config/checkpoint fields are unset, these commands print the missing fields and exit without falling back to mock.
