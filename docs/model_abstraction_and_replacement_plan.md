# Model Abstraction and Replacement Plan

## Current Model Status

- Cloud model is provisionally identified as `vitdet_b`.
- Edge model is `unresolved`.
- Hyperion's edge ViT-Small config/checkpoint has not been located yet.
- The author response or a reproducible public replacement decision is still required before final edge experiments.

## Why ViTDet-B Is Not the Edge Proxy

ViTDet-B should not be used as the formal edge detector proxy because it would collapse the intended edge/cloud asymmetry. The experiment needs a lightweight edge detector and a stronger cloud detector so Cloud Gain, scheduling value, and upload decisions have meaningful interpretation.

Using ViTDet-B for both sides would be acceptable only as an explicit debugging smoke test, not as a reported experiment setting.

## Adapter Boundary

All detectors should emit the same slice-local record:

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

`bbox_xyxy` is slice-local. Full-frame mapping and class-aware NMS happen after detector output normalization.

The adapter boundary is:

- `DetectorAdapter`
- `MockDetectorAdapter`
- `VitDetAdapter`
- `GenericJsonAdapter`

Future Hyperion ViT-Small, RT-DETR, Swin, or other replacement models can enter through `GenericJsonAdapter` if they export the unified JSON schema. No external framework logic is required inside this repo for that path.

## Caches to Rerun When Edge Model Changes

Rerun:

- edge detection cache
- cloud gain labels that depend on edge/cloud disagreement
- history features derived from observed edge/cloud feedback
- Gain Predictor training data and predictions
- scheduler decisions
- simulation metrics and visualizations

Do not rerun solely because the edge detector changes:

- PANDA image sanity CSVs
- slice index / slice grid under the same `configs/panda_3840.yaml`
- slice-level GT map
- static visual features
- JPEG cost profiler outputs

## Next Decision

Before real inference, choose either:

- Hyperion edge ViT-Small config/checkpoint, if the author response makes it reproducible.
- A documented public lightweight Transformer detector replacement that can export the unified JSON schema.

Do not install RT-DETR, MMDetection, Swin, or any other framework until that decision is explicit.
