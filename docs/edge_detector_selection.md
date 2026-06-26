# Edge Detector Selection

This document freezes the decision criteria for choosing an edge detector while keeping the cloud detector provisionally tied to ViTDet-B. No checkpoints are downloaded and no new frameworks are installed at this stage.

## Current Baseline

- Cloud candidate: ViTDet-B, pending config/checkpoint confirmation.
- Edge candidate: unresolved.
- Hyperion edge ViT-Small config/checkpoint: not yet available.
- Current detector integration point: unified JSON through `DetectorAdapter`.

Detectron2's ViTDet project publicly lists ViTDet-B/L/H models and configs, including ViTDet-B model entries, but the current local WSL search did not find a reproducible ViTDet-S config/checkpoint. The Swin Transformer object detection repository publicly lists Swin-T Mask R-CNN and Cascade Mask R-CNN configs/checkpoints, but it is based on MMDetection, which is not installed in this project.

References:

- Detectron2 ViTDet project: https://github.com/facebookresearch/detectron2/tree/main/projects/ViTDet
- Swin Transformer Object Detection: https://github.com/SwinTransformer/Swin-Transformer-Object-Detection

## Option 1: Hyperion-Compatible ViTDet-Small + ViTDet-Base

| Criterion | Assessment |
| --- | --- |
| Edge backbone / detector type | Plain ViT-style detector if the Hyperion ViT-Small configuration is obtained. |
| ViT-based scheduling narrative | Strongest match. Edge and cloud both remain in the ViTDet family, preserving the intended Hyperion-style comparison. |
| Public official config/checkpoint | Not currently available in this repo or local WSL search. Requires author-provided or otherwise reproducible files. |
| Needs new framework | No, if it is a Detectron2/ViTDet config compatible with the current environment. |
| Pipeline compatibility | Compatible with slice-local bbox -> frame bbox -> global NMS as long as detections export bbox/class/score. |
| Formal paper suitability | Best option if reproducible config/checkpoint arrives within the waiting period. |
| Temporary pipeline bring-up | Good only after config is available; otherwise cannot be fabricated. |
| Cloud gain / small-object recall / tracking impact | Cleanest interpretation: cloud gain reflects stronger ViTDet-B over a smaller ViT edge model; small-object recall and tracking feedback remain aligned with the intended paper story. |
| Main risks | Author may not provide files; config may depend on private training details; checkpoint may be unavailable; cannot invent a ViTDet-S and call it Hyperion-compatible. |

## Option 2: Swin-T Mask R-CNN or Cascade Mask R-CNN + ViTDet-B

| Criterion | Assessment |
| --- | --- |
| Edge backbone / detector type | Hierarchical shifted-window Transformer detector, usually Mask R-CNN or Cascade Mask R-CNN with Swin-T backbone. |
| ViT-based scheduling narrative | Partially preserved as a Transformer edge/cloud story, but no longer plain-ViT/ViTDet-on-both-sides. |
| Public official config/checkpoint | The official Swin object detection repository lists Swin-T configs and model links. |
| Needs new framework | Yes. The official implementation is based on MMDetection, which is currently not installed and should not be added without user confirmation. |
| Pipeline compatibility | Compatible in principle if outputs are normalized to bbox/class/score JSON. |
| Formal paper suitability | Plausible fallback if the paper is reframed as lightweight Transformer edge vs stronger ViTDet cloud, but it is not strict Hyperion reproduction. |
| Temporary pipeline bring-up | Not suitable under current constraints because it needs a new framework. Could be revisited after explicit approval. |
| Cloud gain / small-object recall / tracking impact | May alter gain distribution because Swin-T inductive bias and training recipe differ from ViTDet; small-object recall could be stronger/weaker depending config; tracking feedback remains compatible after JSON normalization. |
| Main risks | Requires MMDetection stack; framework differences complicate attribution; Cascade variants may be too heavy for edge; not a Hyperion ViT-Small replacement unless explicitly justified. |

## Option 3: Lightweight Transformer Detector + ViTDet-B

| Criterion | Assessment |
| --- | --- |
| Edge backbone / detector type | Any public lightweight Transformer detector that exports bbox/class/score, such as a DETR-like, deformable-attention, or hybrid Transformer detector. |
| ViT-based scheduling narrative | Weakest unless the chosen model has a clear Transformer backbone and the paper text is careful. |
| Public official config/checkpoint | Must be verified case by case before adoption. |
| Needs new framework | Likely yes unless it can run through existing Detectron2 or export JSON externally. No new framework should be installed before user confirmation. |
| Pipeline compatibility | Compatible if converted through `GenericJsonAdapter` into the unified slice-local JSON schema. |
| Formal paper suitability | Suitable only as a documented reproducible replacement, not as Hyperion reproduction. |
| Temporary pipeline bring-up | Good if it can produce JSON externally without changing this repo; otherwise blocked by dependency constraints. |
| Cloud gain / small-object recall / tracking impact | May substantially change edge/cloud disagreement and therefore cloud-gain labels; all downstream learned gain/scheduler results must be regenerated after selection. |
| Main risks | Reproducibility, class mapping to person/vehicle, dependency burden, speed mismatch, and narrative drift away from Hyperion. |

## Decision Rules

1. If the Hyperion author provides a reproducible ViT-Small config/checkpoint within the waiting period, use that option first.
2. If the files cannot be obtained, do not fabricate or relabel an arbitrary model as ViTDet-S.
3. Any replacement edge model must be clearly lighter than cloud ViTDet-B, have a public reproducible checkpoint, output bbox/class/score, map cleanly to person/vehicle, and connect through the unified JSON schema.
4. Final model choice requires user confirmation before installing any new framework or downloading any weights.

## Cache Impact After Final Selection

Rerun after edge model changes:

- edge detection cache
- cloud gain labels
- history features based on observed edge/cloud disagreement
- gain predictor training data and predictions
- scheduler decisions
- simulator metrics and result visualizations

No need to rerun solely due to edge model selection:

- PANDA image sanity checks
- 3840 slice index
- slice-level GT map
- static visual features
- JPEG cost profiler
