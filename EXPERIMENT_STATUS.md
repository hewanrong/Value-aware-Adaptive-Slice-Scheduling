# Experiment Status

## 2026-06-26 Edge Detector Selection Decision Draft

- Added `docs/edge_detector_selection.md`.
- Compared Hyperion-compatible ViTDet-Small, Swin-T detector variants, and generic lightweight Transformer detector options.
- Decision rule: prefer reproducible Hyperion ViT-Small if provided; otherwise do not fabricate ViTDet-S.
- Replacement edge model must be lighter than ViTDet-B, publicly reproducible, emit bbox/class/score, map to person/vehicle, and export the unified JSON schema.
- Added `tests/test_result_schema.py` for slice-local bbox and frame-offset mapping.
- Unit test passed.
- Edge/cloud mock smoke tests passed.
- No checkpoint was downloaded.
- No real detection was run.
- No new framework was installed.
- Scheduler, simulator, and Gain Predictor logic remain unchanged.

## 2026-06-26 Detector Abstraction for Replaceable Edge Model

- Unified detection output now includes model-agnostic slice-local fields: `bbox_xyxy`, `class_id`, `model_name`, `backend`, `input_width`, `input_height`, and `inference_time_ms`.
- Existing mock outputs remain backward compatible with legacy fields.
- Added detector adapter abstraction under `src/detection/adapters/`.
- Added `MockDetectorAdapter`, `VitDetAdapter` skeleton, and `GenericJsonAdapter`.
- Added `configs/model_pair_example.yaml`.
- Updated `configs/vitdet_example.yaml` with `cloud_model_name: vitdet_b` and `edge_model_name: unresolved`.
- Added `docs/model_abstraction_and_replacement_plan.md`.
- Edge model remains unresolved pending Hyperion author response or replacement decision.
- ViTDet-B is not treated as the edge proxy.
- No checkpoint was downloaded.
- No real detection was run.
- No new framework was installed.
- Scheduler, simulator, and Gain Predictor logic remain unchanged.

## 2026-06-22 Phase 2A Minimal ViTDet Prep

- WSL runtime check succeeded through `Ubuntu`.
- Conda Python: `/home/asus/miniforge3/envs/vitdet/bin/python`.
- Python: 3.11.15.
- torch: 2.5.1+cu121.
- torchvision: 0.20.1+cu121.
- CUDA available: true.
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU.
- detectron2 import: success, version 0.6.
- ViTDet config directories: none found.
- ViTDet-S/B config candidates: none found.
- `configs/vitdet_example.yaml` now points to the WSL runtime but keeps repo/config/checkpoint fields unset.
- Edge/cloud scripts retain mock backend and now report missing vitdet config/checkpoint fields without falling back to mock.
- No real detection was run.
- No checkpoint was downloaded.
- Scheduler, simulator, and Gain Predictor logic remain unchanged.

## 2026-06-20 Phase 1.5 Detection Protocol Check Complete

- Phase 1.5 completed as a protocol/environment validation pass.
- Batch PANDA inference was not run.
- Gain Predictor training remains not run.
- Oracle Scheduler and Learned Scheduler main logic remain unchanged.
- Added `docs/phase1_5_detection_protocol.md`.
- Added unified detection schema with slice-local and full-frame bbox fields.
- Added coordinate mapping utilities and class-aware global NMS.
- Added coordinate pipeline check output at `results/phase1_5/coordinate_pipeline_check.png`.
- Added WSL/ViTDet environment audit script and placeholder config.

## Phase 1.5 Verification Summary

- Coordinate pipeline test: passed.
- Global NMS boxes before/after: 30 -> 15.
- WSL status: default WSL distro unavailable on this machine.
- PyTorch/CUDA/GPU/Detectron2 status: unavailable because WSL is not available/configured.
- ViTDet config/checkpoint status: unset in `configs/vitdet_example.yaml`.
- Required manual inputs before real inference: WSL distro, ViTDet repo root, Python executable, edge/cloud configs, edge/cloud checkpoints, and device choice.

## 2026-06-18 Phase 1 Minimal Pipeline Complete

- Phase 1 completed for a controlled `configs/panda_3840.yaml` minimal run.
- ViTDet integration remains not connected.
- Gain Predictor training remains not run.
- Scheduler and simulator main logic remain unchanged in this pass.
- Added real GT resize visual checks under `results/phase1/gt_resize_check/`.
- Added slice-level GT map at `cache/slice_gt_map.csv`.
- Added real visual feature extraction at `cache/slice_features.csv`.
- Added real JPEG cost profiling at `cache/slice_costs.csv`.
- Added Phase 1 sanity visualizations under `results/phase1/summary/`.

## Phase 1 Minimal Run Summary

- Frames processed for GT map/features/costs: 10.
- Slices generated: 150.
- `empty_slice_ratio`: 0.226667.
- `boundary_cut_gt_ratio`: 0.055358.
- `cache/slice_gt_map.csv`: exists.
- `cache/slice_features.csv`: exists.
- `cache/slice_costs.csv`: exists.
- `results/phase1/`: visualization outputs generated.

## 2026-06-18 Phase 0 Frozen

- Phase 0 sanity check is frozen.
- `configs/panda_raw.yaml` preserves original PANDA resolution with `preprocess.target_long_side: null`.
- `configs/panda_3840.yaml` sets `preprocess.target_long_side: 3840` and is the default for future controlled experiments.
- `configs/default.yaml` is now equivalent to the 3840-long-side PANDA configuration.
- Future sanity, debug, scheduler, and ViTDet experiments should use `configs/panda_3840.yaml` unless raw UHD scale is explicitly required.
- ViTDet integration remains not connected.
- Model training remains not run.
- Scheduler and simulator main logic remain unchanged in this pass.

## Phase 0 Documentation

- Added `docs/phase0_sanity.md`.
- Added `docs/experiment_protocol.md`.
- Added `docs/phase1_plan.md`.
- Phase 1 is planned only; no Phase 1 implementation was started.

## 2026-06-18

- Current focus: PANDA dataset and slice sanity checks only.
- ViTDet integration: not connected.
- Model training: not run.
- Scheduler/simulator main logic: unchanged in this pass.
- Default config keeps original behavior with `preprocess.target_long_side: null`.
- Added optional long-side preprocessing so future slice generation can use a scaled coordinate system while mapping PANDA GT boxes correctly.

## PANDA `image_train` Resolution Scan

- Frames scanned: 390.
- Original width range: 24,853 to 35,503.
- Original height range: 13,983 to 26,627.
- Original 1024px slice count range: 594 to 1,610 per image.
- Original mean slice count: 767.81 per image.
- `target_long_side=3840` width range: 3,840 to 3,840.
- `target_long_side=3840` height range: 2,160 to 2,880.
- `target_long_side=3840` slice count range: 15 to 20 per image.
- `target_long_side=3840` mean slice count: 15.77 per image.

## Generated Artifacts

- `results/sanity/slice_stats_original.csv`
- `results/sanity/slice_stats_3840.csv`
- `results/sanity/sample_gt.svg`
- `results/sanity/sample_slice_grid.svg`
- `results/sanity/sample_gt_slice_grid.svg`
