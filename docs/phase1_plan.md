# Phase 1 Plan

Phase 1 prepares the 3840-long-side PANDA experiment pipeline while still avoiding ViTDet integration and model training.

## Goals

- Verify that GT coordinate scaling is correct under `target_long_side=3840`.
- Generate explicit slice-level GT mapping files for person and vehicle objects.
- Implement real visual features using an imaging backend when dependencies are available.
- Implement a real JPEG cost profiler based on encoded slice bytes.

## Non-Goals

- Do not connect ViTDet.
- Do not train gain prediction models.
- Do not modify scheduler or simulator main logic.
- Do not remove or overwrite existing smoke-test commands.

## Proposed Outputs

- Slice-level GT cache with frame id, slice id, category id, scaled bbox, original bbox, and boundary-cut flags.
- Visual feature CSV computed from actual resized or cropped image content.
- JPEG cost CSV computed from actual encoded slice crops at configured quality levels.
- Validation notes comparing a small set of GT overlays against slice grids.

## Acceptance Checks

- `configs/panda_3840.yaml` remains the default experiment configuration.
- A small `--max-frames` run completes quickly.
- GT counts are conserved after scaling and slice assignment.
- Boundary-cut statistics are reported before any downstream scheduling experiment.
