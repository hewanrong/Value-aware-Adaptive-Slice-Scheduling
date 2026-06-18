# Experiment Status

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
