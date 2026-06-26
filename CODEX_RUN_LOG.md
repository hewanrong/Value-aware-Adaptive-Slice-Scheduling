# Codex Run Log

## 2026-06-26 Edge Detector Selection and Schema Offset Test

Commands run:

```powershell
python -m unittest tests.test_result_schema
python scripts/run_edge_detection.py --config configs/panda_3840.yaml --backend mock --max-frames 1 --max-slices 2 --output cache/edge_results_selection_smoke.json
python scripts/run_cloud_detection.py --config configs/panda_3840.yaml --backend mock --quality full --max-frames 1 --max-slices 2 --output cache/cloud_results_selection_smoke.json
```

Output summary:

- Unit test passed: slice-local `bbox_xyxy` remains unchanged while `bbox_xyxy_frame` adds slice offsets.
- Edge mock smoke wrote 2 detections.
- Cloud mock smoke wrote 2 detections.
- Added edge detector selection document with Hyperion ViT-Small, Swin-T, and generic lightweight Transformer detector comparison.

Notes:

- No WSL real model was called.
- No checkpoint was downloaded.
- No new framework was installed.
- Scheduler, simulator, and Gain Predictor logic were not modified.

## 2026-06-26 Detector Abstraction and Mock Schema Check

Commands run:

```powershell
python -m compileall src\detection scripts\run_edge_detection.py scripts\run_cloud_detection.py
python -c "from src.detection.adapters import DetectorAdapter, MockDetectorAdapter, VitDetAdapter, GenericJsonAdapter; from src.detection.result_schema import validate_detection, canonical_detection_record; row=canonical_detection_record(frame_id='f', slice_id='s', bbox_xyxy=[0,0,1,1], class_id=1, score=0.5, model_name='m', backend='mock', input_width=1, input_height=1); validate_detection(row); print('adapter/schema import ok')"
python scripts/run_edge_detection.py --config configs/panda_3840.yaml --backend mock --max-frames 1 --max-slices 2 --output cache/edge_results_model_abstraction_smoke.json
python scripts/run_cloud_detection.py --config configs/panda_3840.yaml --backend mock --quality full --max-frames 1 --max-slices 2 --output cache/cloud_results_model_abstraction_smoke.json
```

Output summary:

- Static compile passed for detection modules and edge/cloud scripts.
- Adapter/schema import test passed.
- Edge mock backend wrote 2 detections.
- Cloud mock backend wrote 2 detections.
- Mock records include the new canonical fields and legacy compatibility fields.

Notes:

- WSL real model was not called.
- No checkpoint was downloaded.
- No new framework was installed.
- Scheduler, simulator, and Gain Predictor logic were not modified.

## 2026-06-22 Phase 2A Minimal ViTDet Runtime Check

Commands run:

```powershell
python scripts/check_vitdet_runtime.py --config configs/vitdet_example.yaml
```

Output summary:

- Python executable: `/home/asus/miniforge3/envs/vitdet/bin/python`.
- Python version: `3.11.15`.
- torch: `2.5.1+cu121`.
- torchvision: `0.20.1+cu121`.
- CUDA available: `true`.
- CUDA version: `12.1`.
- GPU: `NVIDIA GeForce RTX 4060 Laptop GPU`.
- detectron2: `0.6`.
- detectron2 file: `/home/asus/miniforge3/envs/vitdet/lib/python3.11/site-packages/detectron2/__init__.py`.
- `projects/ViTDet/configs`: not found.
- ViTDet-S candidates: none found.
- ViTDet-B candidates: none found.

Notes:

- No real detection was run.
- No checkpoint was downloaded.
- No model was trained.
- Scheduler, simulator, and Gain Predictor logic were not modified.

## 2026-06-20 Phase 1.5 Detection Protocol and WSL Audit

Commands run:

```powershell
C:\Users\ASUS\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe -m compileall src scripts
C:\Users\ASUS\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe scripts/test_detection_coordinate_pipeline.py --config configs/panda_3840.yaml --max-frames 1
python scripts/check_vitdet_wsl_env.py --config configs/vitdet_example.yaml
python scripts/run_edge_detection.py --config configs/panda_3840.yaml --backend mock --max-frames 1 --max-slices 2 --output cache/edge_results_phase1_5_smoke.json
python scripts/run_cloud_detection.py --config configs/panda_3840.yaml --backend mock --quality full --max-frames 1 --max-slices 2 --output cache/cloud_results_phase1_5_smoke.json
```

Output summary:

- Compile check passed for `src` and `scripts` after fixing a stale return block in `vitdet_runner_stub.py`.
- Coordinate pipeline passed.
- NMS boxes before/after: `before_nms=30`, `after_nms=15`.
- Coordinate visualization written to `results/phase1_5/coordinate_pipeline_check.png`.
- WSL available: `False`.
- Python/PyTorch/CUDA/GPU/Detectron2 inside WSL: unavailable because WSL is not available/configured.
- ViTDet repo/config/checkpoint fields: unset in `configs/vitdet_example.yaml`.
- Edge mock backend smoke: wrote 2 detections with local/frame bbox fields.
- Cloud mock backend smoke: wrote 2 detections with local/frame bbox fields.

Notes:

- No batch PANDA inference was run.
- ViTDet was not connected.
- No Gain Predictor model was trained.
- Oracle Scheduler and Learned Scheduler main logic were not modified.

## 2026-06-18 Phase 1 Minimal Pipeline

Commands run:

```powershell
C:\Users\ASUS\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe -m compileall src scripts
C:\Users\ASUS\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe scripts/visualize_gt_resize_check.py --config configs/panda_3840.yaml --max-frames 3
C:\Users\ASUS\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe scripts/build_slice_gt_map.py --config configs/panda_3840.yaml --max-frames 10
C:\Users\ASUS\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe scripts/extract_slice_features.py --config configs/panda_3840.yaml --max-frames 10
C:\Users\ASUS\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe scripts/profile_slice_costs.py --config configs/panda_3840.yaml --max-frames 10
C:\Users\ASUS\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe scripts/visualize_phase1_summary.py --config configs/panda_3840.yaml
```

Output summary:

- Compile check passed for `src` and `scripts`.
- GT resize check: wrote 3 frames to `results/phase1/gt_resize_check`.
- Slice GT map: wrote 150 rows to `cache/slice_gt_map.csv`.
- Slice GT summary: `frames=10`, `slices=150`, `empty_slice_ratio=0.226667`, `boundary_cut_gt_ratio=0.055358`.
- Visual features: wrote 150 rows for 10 frames to `cache/slice_features.csv`.
- JPEG costs: wrote 150 rows for 10 frames to `cache/slice_costs.csv`.
- Summary visualizations: wrote PNGs to `results/phase1/summary`.

Notes:

- ViTDet was not connected.
- No Gain Predictor model was trained.
- Scheduler and simulator main logic were not modified.

## 2026-06-18 Phase 0 Freeze and Config Split

Commands run:

```powershell
python scripts/analyze_dataset_slices.py --config configs/panda_raw.yaml --max-frames 3
python scripts/analyze_dataset_slices.py --config configs/panda_3840.yaml --max-frames 3
```

Output summary:

- Raw config sample: `frames=3`, `target_long_side=None`, `width_range=26753..26753`, `height_range=15052..15052`, `slice_count_range=700..700`, `slice_count_mean=700.00`, `boundary_cut_ratio_mean=0.091091`.
- 3840 config sample: `frames=3`, `target_long_side=3840`, `width_range=3840..3840`, `height_range=2160..2160`, `slice_count_range=15..15`, `slice_count_mean=15.00`, `boundary_cut_ratio_mean=0.000000`.

Notes:

- Added `configs/panda_raw.yaml` for raw-resolution sanity checks.
- Added `configs/panda_3840.yaml` for the main controlled PANDA experiment configuration.
- Updated `configs/default.yaml` to match the 3840-long-side configuration.
- Added Phase 0, experiment protocol, and Phase 1 planning docs.
- ViTDet was not connected.
- No model training was run.
- Scheduler and simulator main logic were not modified.

## 2026-06-18 PANDA Slice Sanity Pass

Commands run:

```powershell
python -m compileall src scripts
python scripts/analyze_dataset_slices.py --config configs/default.yaml --output results/sanity/slice_stats_original.csv --visualize-samples
python scripts/analyze_dataset_slices.py --config configs/default.yaml --target-long-side 3840 --output results/sanity/slice_stats_3840.csv
```

Output summary:

- Compile check passed for `src` and `scripts`.
- Original resolution scan: `frames=390`, `width_range=24853..35503`, `height_range=13983..26627`, `slice_count_range=594..1610`, `slice_count_mean=767.81`, `boundary_cut_ratio_mean=0.080252`.
- 3840-long-side scan: `frames=390`, `width_range=3840..3840`, `height_range=2160..2880`, `slice_count_range=15..20`, `slice_count_mean=15.77`, `boundary_cut_ratio_mean=0.002216`.

Notes:

- ViTDet was not connected.
- No model training was run.
- Scheduler and simulator main logic were not modified.
- `boundary_cut_gt_count` is defined as GT boxes that are not fully contained by any generated slice in the selected coordinate system.
