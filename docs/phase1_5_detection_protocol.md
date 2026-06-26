# Phase 1.5 Detection Protocol

Phase 1.5 freezes the detection-cache protocol and checks the WSL inference environment. It does not run batch PANDA inference, train the Gain Predictor, or modify Oracle/Learned Scheduler logic.

## Experiment Coordinate System

- Main experiments use `target_long_side=3840`.
- `slice_size=1024`.
- `overlap_ratio=0.25`.
- Slice bboxes are represented in the resized full-frame coordinate system.
- Detector outputs from slice inference must first be represented as `bbox_xyxy_local` in slice-local coordinates.
- Every slice-local detection must be mapped into resized full-frame coordinates as `bbox_xyxy_frame` before evaluation, merging, or scheduling feedback.

## Edge and Cloud Detection Roles

- Edge detector runs on all slices selected for the frame-level edge cache.
- Cloud detector runs only on slices selected by the scheduler.
- Cloud quality must be recorded as one of `low`, `mid`, `high`, or `full`.
- Edge detections should use `quality=full` for now unless a separate edge-quality experiment is explicitly introduced.

## Unified Detection Row

Each detection row must include at least:

- `frame_id`
- `slice_id`
- `bbox_xyxy_local`
- `bbox_xyxy_frame`
- `score`
- `category_id`
- `source`: `edge` or `cloud`
- `quality`: `none`, `low`, `mid`, `high`, or `full`
- `slice_x1`, `slice_y1`, `slice_x2`, `slice_y2`

## Merging and NMS

- All edge/cloud detections for a frame must be merged in resized full-frame coordinates.
- Overlap-induced duplicate detections must be removed by class-aware global NMS.
- NMS operates on `bbox_xyxy_frame`, never on slice-local coordinates.
- NMS must not suppress boxes across different `category_id` values.

## Evaluation

- Final AP50, Recall, and small-object Recall must be computed on full-frame detections against full-frame GT.
- It is invalid to directly average or sum slice-level AP as frame-level performance.
- Slice-level statistics may be used for diagnostics, feature analysis, and scheduler labels, but not as the final detection metric.

## Out of Scope

- No ViTDet batch inference in this phase.
- No Gain Predictor training in this phase.
- No scheduler or simulator logic changes in this phase.
