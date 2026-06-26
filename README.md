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

## Phase 1 Pipeline

Phase 1 uses `configs/panda_3840.yaml` and still does not connect ViTDet or train any model.

```powershell
python scripts/visualize_gt_resize_check.py --config configs/panda_3840.yaml --max-frames 3
python scripts/build_slice_gt_map.py --config configs/panda_3840.yaml --max-frames 10
python scripts/extract_slice_features.py --config configs/panda_3840.yaml --max-frames 10
python scripts/profile_slice_costs.py --config configs/panda_3840.yaml --max-frames 10
python scripts/visualize_phase1_summary.py --config configs/panda_3840.yaml
```

Generated Phase 1 caches:

- `cache/slice_gt_map.csv`
- `cache/slice_features.csv`
- `cache/slice_costs.csv`

Generated Phase 1 sanity outputs:

- `results/phase1/gt_resize_check/`
- `results/phase1/slice_gt_map_summary.csv`
- `results/phase1/summary/`

Install the optional experiment dependencies from `requirements.txt` for a normal Python environment. The current implementation uses Pillow and NumPy for real image features and JPEG byte profiling; it does not require Detectron2 or ViTDet.

## Phase 1.5 Detection Protocol Checks

Phase 1.5 freezes the real detection cache protocol and validates coordinate mapping without running batch inference.

```powershell
python scripts/test_detection_coordinate_pipeline.py --config configs/panda_3840.yaml --max-frames 1
python scripts/check_vitdet_wsl_env.py --config configs/vitdet_example.yaml
```

Detection rows use both slice-local and resized full-frame boxes:

- `bbox_xyxy_local`
- `bbox_xyxy_frame`
- `slice_x1`, `slice_y1`, `slice_x2`, `slice_y2`

All slice detections must be mapped to full-frame coordinates before class-aware global NMS and final AP50/Recall evaluation. Do not average slice-level AP as frame-level AP.

Edge/cloud scripts now support:

```powershell
python scripts/run_edge_detection.py --config configs/panda_3840.yaml --backend mock --max-frames 1 --max-slices 2
python scripts/run_cloud_detection.py --config configs/panda_3840.yaml --backend mock --quality full --max-frames 1 --max-slices 2
```

`--backend vitdet` is diagnostic-only at this stage: if WSL, Detectron2, config, or checkpoint information is missing, the scripts print the missing fields and exit instead of silently falling back to mock.

## Phase 2A Minimal ViTDet Prep

The WSL runtime can be checked without running inference:

```powershell
python scripts/check_vitdet_runtime.py --config configs/vitdet_example.yaml
```

Current confirmed runtime is Python 3.11, torch `2.5.1+cu121`, torchvision `0.20.1+cu121`, CUDA available on RTX 4060, and Detectron2 importable. ViTDet config files were not found in the current searched paths, so real inference still requires `vitdet_repo_root`, edge/cloud config paths, and edge/cloud checkpoints.

Do not run batch PANDA inference or download checkpoints until config/checkpoint candidates are explicitly confirmed.

## Detector Model Abstraction

Detection backends now use a model-agnostic slice-local record:

- `frame_id`
- `slice_id`
- `bbox_xyxy`
- `class_id`
- `score`
- `model_name`
- `backend`
- `input_width`
- `input_height`
- `inference_time_ms`

`bbox_xyxy` is always slice-local. Full-frame mapping and class-aware NMS happen after detector output normalization. Legacy mock fields such as `bbox_xyxy_local`, `bbox_xyxy_frame`, `category_id`, `source`, and `quality` are still emitted for backward compatibility.

Adapters live under `src/detection/adapters/`:

- `MockDetectorAdapter`
- `VitDetAdapter`
- `GenericJsonAdapter`

The edge model is currently unresolved and must not be assumed to be ViTDet-B. `configs/model_pair_example.yaml` records the provisional model-pair metadata.

## Edge Detector Selection

The current decision document is `docs/edge_detector_selection.md`.

Decision rule:

- Prefer Hyperion-compatible ViTDet-Small if the author provides reproducible config/checkpoint within the waiting period.
- Do not invent a ViTDet-S configuration if those files are unavailable.
- Any replacement edge detector must be clearly lighter than cloud ViTDet-B, have a public reproducible checkpoint, output bbox/class/score, map to person/vehicle, and connect through the unified JSON schema.
- Installing new frameworks or downloading weights requires explicit user confirmation.
