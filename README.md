# Value-aware Adaptive Slice Scheduling

Offline experimental framework scaffold for **Value-aware Adaptive Slice Scheduling for UHD Video Analytics**.

This repository is currently prepared for the first milestone: reproducible structure, configurable paths, cache locations, and module boundaries. The first implementation round should run without PANDA ground truth and without real ViTDet/Detectron2 inference.

## Quick Start

The default config points to `C:/expr/PANDA/image_train` and the PANDA train annotation JSON files. Override those paths if the dataset moves.

For future Codex tasks and normal experiments, use `configs/panda_3840.yaml`. `configs/default.yaml` is intentionally equivalent to the 3840-long-side PANDA configuration. Use `configs/panda_raw.yaml` only when checking original-resolution UHD scale.

## PANDA Slice Sanity Check

Before running expensive experiments, inspect the PANDA resolution and slice count distribution:

```powershell
python scripts/analyze_dataset_slices.py --config configs/panda_raw.yaml --output results/sanity/slice_stats_original.csv --visualize-samples
python scripts/analyze_dataset_slices.py --config configs/panda_3840.yaml --output results/sanity/slice_stats_3840.csv
```

The script writes per-image statistics including image size, slice count, GT density per slice, empty slice ratio, and boundary-cut GT ratio. It also saves three SVG samples under `results/sanity/`: GT overlay, slice grid, and GT plus slice grid.

The PANDA configs include:

```yaml
preprocess:
  target_long_side: null
  keep_aspect_ratio: true
```

`target_long_side: null` in `configs/panda_raw.yaml` preserves the original PANDA resolution. `target_long_side: 3840` in `configs/panda_3840.yaml` maps each image and GT box into a 3840-long-side coordinate system before slicing, while preserving aspect ratio. This does not modify the source images; it controls the experiment coordinate system and future resized-cache generation.

For early experiments, use the 3840-long-side version first. PANDA `image_train` contains very large UHD images, and original-resolution slicing can create hundreds to more than one thousand 1024px slices per image. The 3840 setting keeps slice counts small enough for sanity checks, debugging, scheduler experiments, and visualization without losing the core wide-area analytics behavior.

Smoke-test the pipeline on one UHD frame:

```powershell
python scripts/prepare_slices.py --config configs/default.yaml --max-frames 1
python scripts/run_edge_detection.py --config configs/default.yaml
python scripts/run_cloud_detection.py --config configs/default.yaml --quality full
python scripts/extract_slice_features.py --config configs/default.yaml --max-slices 20
python scripts/profile_slice_costs.py --config configs/default.yaml --max-slices 20
python scripts/predict_gain.py --config configs/default.yaml
python scripts/run_scheduler.py --config configs/default.yaml --budget 5000000
python scripts/run_simulation.py --config configs/default.yaml
python scripts/visualize_results.py --config configs/default.yaml
```

Compute GT-derived cloud gain when edge/cloud detections and PANDA labels are available:

```powershell
python scripts/compute_cloud_gain.py --config configs/default.yaml
```

Detailed command examples will be added as each stage is implemented.

## ViTDet / WSL Note

Real ViTDet integration is intentionally out of scope for the first scaffold. The future runner should expose a command template and keep all WSL, config, checkpoint, image, and output paths configurable.
