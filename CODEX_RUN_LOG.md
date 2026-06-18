# Codex Run Log

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
