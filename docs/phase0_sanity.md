# Phase 0 Sanity Check

Phase 0 freezes the PANDA dataset and slice sanity-check baseline. No ViTDet integration or model training is part of this phase.

## Dataset

- Dataset split: PANDA `image_train`.
- Frames scanned: 390.
- Source image root: `C:/expr/PANDA/image_train`.
- Person GT: `C:/expr/PANDA/image_annos/person_bbox_train.json`.
- Vehicle GT: `C:/expr/PANDA/image_annos/vehicle_bbox_train.json`.

## Original Resolution Results

- Width range: 24,853 to 35,503.
- Height range: 13,983 to 26,627.
- Slice size: 1024.
- Overlap ratio: 0.25.
- Slice count range: 594 to 1,610 per image.
- Mean slice count: 767.81 per image.

## 3840 Long-Side Results

- `target_long_side`: 3840.
- `keep_aspect_ratio`: true.
- Width range after scaling: 3,840 to 3,840.
- Height range after scaling: 2,160 to 2,880.
- Slice count range: 15 to 20 per image.
- Mean slice count: 15.77 per image.

## Conclusion

Future sanity, debug, scheduler, and ViTDet experiments should default to `target_long_side=3840` through `configs/panda_3840.yaml` or `configs/default.yaml`.

The original-resolution configuration remains available as `configs/panda_raw.yaml`, but it is only for documenting UHD data scale and verifying assumptions. It should not be the default for iterative experiments because it produces hundreds to more than one thousand slices per image.
