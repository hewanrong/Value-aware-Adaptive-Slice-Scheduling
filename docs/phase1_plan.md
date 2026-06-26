# Phase 1 Plan and Completion Notes

Phase 1 prepares the 3840-long-side PANDA experiment pipeline while still avoiding ViTDet integration and model training.

## Goals

- Verify that GT coordinate scaling is correct under `target_long_side=3840`. Completed with `scripts/visualize_gt_resize_check.py`.
- Generate explicit slice-level GT mapping files for person and vehicle objects. Completed with `scripts/build_slice_gt_map.py`.
- Implement real visual features using an imaging backend when dependencies are available. Completed with Pillow/NumPy in `src/features/visual_features.py`.
- Implement a real JPEG cost profiler based on encoded slice bytes. Completed with Pillow JPEG encoding in `src/scheduler/cost_profiler.py`.

## Non-Goals

- Do not connect ViTDet.
- Do not train gain prediction models.
- Do not modify scheduler or simulator main logic.
- Do not remove or overwrite existing smoke-test commands.

## Proposed Outputs

- `results/phase1/gt_resize_check/resize_check_summary.csv`
- `cache/slice_gt_map.csv`
- `results/phase1/slice_gt_map_summary.csv`
- `cache/slice_features.csv`
- `cache/slice_costs.csv`
- `results/phase1/summary/*.png`

## Acceptance Checks

- `configs/panda_3840.yaml` remains the default experiment configuration.
- A small `--max-frames` run completes on 10 frames.
- GT coordinate scaling is visualized for raw and 3840 resized coordinates.
- Boundary-cut statistics are reported before any downstream scheduling experiment.

## Verified Minimal Run

The Phase 1 minimal run processed 10 frames and 150 slices under `configs/panda_3840.yaml`.

- `empty_slice_ratio`: 0.226667
- `boundary_cut_gt_ratio`: 0.055358
- `person_total`: 1156
- `vehicle_total`: 253

ViTDet remains disconnected and no gain predictor model has been trained.
