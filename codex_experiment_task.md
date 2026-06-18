# Codex Task: Value-aware Adaptive Slice Scheduling Experiment Framework

## 0. Project Goal

Build a reproducible offline experimental framework for a paper prototype: **Value-aware Adaptive Slice Scheduling for UHD Video Analytics**.

The first milestone is **not** to run full ViTDet inference. The first milestone is to build the repository structure, data/cache schemas, slice generation, mock detection interfaces, cloud-gain calculation interfaces, cost profiler, scheduler, simulator, and visualization scaffolding.

The later goal is to validate:

1. Whether Cloud Gain is concentrated in a small subset of slices.
2. Whether Oracle scheduling can approach full-cloud accuracy under lower bandwidth.
3. Whether a learned Gain Predictor + Scheduler can approach Oracle.
4. Whether dynamic bandwidth adaptation outperforms Edge-only, Full-cloud, Random, Texture-based, and Hyperion-style scheduling.
5. Whether tracking reduces repeated upload and cloud inference frequency.

## 1. Environment

- Host OS: Windows.
- ViTDet / Detectron2 can only run inside WSL Ubuntu.
- Hardware: two GPU laptops and one old CPU-only desktop.
- First stage: offline framework only. Do not assume real multi-machine deployment.
- Dataset: PANDA video dataset extracted frames. GT annotation will be added later.
- All paths must be configurable in `configs/default.yaml`.
- Do not hard-code local paths.
- The code must run a mock pipeline even without GT and without ViTDet.

## 2. Core Concepts

- **Slice** is the unit of transmission, encoding, scheduling, and cloud detection.
- **ROI / track** is only for temporal state maintenance and object tracking. ROI is not the direct upload unit.
- **Gain Predictor** predicts detection value under different upload qualities.
- **Cost Profiler** estimates byte cost / latency under different encoding qualities.
- **Scheduler** jointly decides which slices to upload and what quality to use.
- Do not use `bytes_low`, `bytes_mid`, or `bytes_high` as Gain Predictor inputs. They belong to Cost Profiler and Scheduler.
- Historical gain must come from real observed feedback or offline GT-derived labels, not recursively from predicted gain.

## 3. Required Repository Structure

Create:

```text
value_aware_slice_scheduling/
  README.md
  configs/
    default.yaml
  data/
  cache/
  results/
  scripts/
  src/
    dataset/
    slicing/
    detection/
    gain/
    features/
    scheduler/
    tracking/
    simulation/
    visualization/
    utils/
```

## 4. Stage 1: Dataset and Slice Framework

Implement:

### `src/dataset/panda_loader.py`

Requirements:

- Read an image directory.
- Support future COCO-format GT JSON.
- If no GT exists, still allow no-GT detection visualization flow.

### `src/slicing/sahi_slicer.py`

Defaults:

- `slice_size = 1024`
- `overlap_ratio = 0.25`
- Output slice bboxes for each image.
- Save slice index mapping.

### `scripts/prepare_slices.py`

Inputs:

- `images_dir`

Outputs:

- `cache/slices.json`
- optional slice visualization images under `results/slice_vis/`

## 5. Stage 2: Detection Result Interface

Do not run real ViTDet yet. First implement unified schema and stubs.

### `src/detection/result_schema.py`

Unified detection format:

```text
frame_id
slice_id
bbox
score
category_id
source=edge/cloud
quality=low/mid/high/full
```

### `src/detection/vitdet_runner_stub.py`

Requirements:

- Provide interface and command template only.
- Note that ViTDet must run in WSL.
- Do not hard-code WSL paths.
- Add TODO comments for ViTDet config, checkpoint, image path, and output path.

### `scripts/run_edge_detection.py`

- Calls edge detector.
- For now, allow mock result or existing JSON result.

### `scripts/run_cloud_detection.py`

- Calls cloud detector.
- Supports `quality in {low, mid, high, full}`.
- For now, allow mock result or existing JSON result.

## 6. Stage 3: Cloud Gain Calculation

### `src/gain/gain_calculator.py`

If GT exists:

- Compute edge AP / cloud AP / gain for each slice.
- Support simplified object-level label:
  - `Cloud Gain Object = Cloud correct && Edge wrong`
- Default IoU threshold: `0.5`.

### `scripts/compute_cloud_gain.py`

Inputs:

- `edge_results.json`
- `cloud_results.json`
- `gt.json`
- `slices.json`

Output:

- `cache/slice_gain.csv`

## 7. Stage 4: Feature Extraction

### `src/features/visual_features.py`

Implement:

- entropy
- edge_density
- laplacian_variance
- gradient_mean
- local_contrast
- motion_intensity
- optical_flow_magnitude

### `src/features/history_features.py`

Implement:

- last_observed_gain
- ewma_observed_gain
- observed_gain_variance
- cloud_edge_disagreement_rate
- time_since_last_cloud_check
- num_recent_cloud_checks

Important: these must come from observed cloud feedback / offline GT labels, not predicted gain recursion.

### `src/features/tracking_features.py`

Implement:

- tracked_object_count
- mean_track_box_area
- min_track_box_area
- small_track_ratio
- track_density
- track_confidence_mean
- track_lost_count
- track_age_mean
- track_velocity_mean

### `scripts/extract_slice_features.py`

Output:

- `cache/slice_features.csv`

## 8. Stage 5: Cost Profiler

### `src/scheduler/cost_profiler.py`

Input:

- slice image

Output:

- bytes_low
- bytes_mid
- bytes_high
- bytes_full
- estimated_latency_low
- estimated_latency_mid
- estimated_latency_high
- estimated_latency_full

Default JPEG simulation:

- low = 30
- mid = 55
- high = 80
- full = 95

Important: cost features are for Scheduler, not Gain Predictor.

### `scripts/profile_slice_costs.py`

Output:

- `cache/slice_costs.csv`

## 9. Stage 6: Gain Predictor

### `src/gain/gain_predictor.py`

Use:

- LightGBM if available, otherwise sklearn GradientBoostingRegressor.

Inputs:

- `slice_features.csv`

Labels:

- from `slice_gain.csv`

Outputs:

- V_low
- V_mid
- V_high
- V_full

### `scripts/train_gain_predictor.py`

Requirements:

- Train model.
- Save to `results/models/gain_predictor.pkl`.
- Output feature importance plot.

### `scripts/predict_gain.py`

Output:

- `cache/predicted_gain.csv`

## 10. Stage 7: Scheduler

### `src/scheduler/oracle_scheduler.py`

Inputs:

- true Gain@quality
- Cost@quality

Use:

- multiple-choice knapsack.

Output:

- per-frame, per-slice action:
  - none / low / mid / high / full

### `src/scheduler/learned_scheduler.py`

Inputs:

- predicted Gain@quality
- Cost@quality

Use the same multiple-choice knapsack formulation.

### `scripts/run_scheduler.py`

Requirements:

- Support fixed bandwidth budget.
- Support dynamic bandwidth trace.
- Output `cache/schedule_decisions.csv`.

## 11. Stage 8: Offline Simulator

### `src/simulation/offline_simulator.py`

Inputs:

- edge_results
- cloud_results_by_quality
- schedule_decisions
- GT if available

For each frame:

- Uploaded slices use cloud results.
- Non-uploaded slices use edge results.
- If tracking is enabled, allow tracking result reuse.
- Merge slice-level detections into frame-level detections.

### `scripts/run_simulation.py`

Outputs:

- `results/metrics.csv`
- `results/frame_results.json`

## 12. Stage 9: Visualization

### `src/visualization/detection_vis.py`

Generate four-panel visualization:

1. Original / GT
2. Edge Only
3. Baseline
4. Ours

Color convention:

- green = correct detection
- red = missed GT
- blue = cloud gain detection
- yellow = tracking reused detection

### `src/visualization/gain_vis.py`

Generate:

- Cloud Gain heatmap
- Gain CDF
- Gain vs object size
- Gain vs density

### `src/visualization/schedule_vis.py`

Generate dynamic trace plots:

- bandwidth
- uploaded slices / uploaded MB
- AP50 / recall

### `scripts/visualize_results.py`

Generate all figures.

## 13. README Requirements

Write `README.md` explaining:

1. How to prepare data.
2. How to generate slices.
3. How to connect ViTDet in WSL.
4. How to calculate Cloud Gain.
5. How to train Gain Predictor.
6. How to run Oracle Scheduler.
7. How to run Learned Scheduler.
8. How to run dynamic trace simulator.
9. How to generate visualizations.

## 14. First-Round Constraints

- Do not assume my data path.
- All paths must be configured in `configs/default.yaml`.
- All scripts must support `--config`.
- Do not run real ViTDet in the first round.
- First round focuses on engineering structure, interfaces, cache format, and mock pipeline.
- Code must run without GT and without ViTDet.
- Later I will provide PANDA GT and ViTDet paths.

## 15. What to Do First

First scan the current repository structure. Then provide an implementation plan. Ask for confirmation before making large code changes.

