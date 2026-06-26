当前已完成 Phase 1。现在进入 Phase 1.5：冻结真实检测协议并验证 WSL 推理环境。

本阶段不要批量跑 PANDA，不要训练 Gain Predictor，不要修改 Oracle Scheduler / Learned Scheduler 主逻辑。

目标是确保后续真实 Edge/Cloud 检测缓存的坐标、融合和运行环境正确。

# 任务 1：检测协议文档

新增 docs/phase1_5_detection_protocol.md，明确以下规范：

- 主实验图像使用 target_long_side=3840。
- slice_size=1024，overlap_ratio=0.25。
- Edge detector 对全部 slice 运行。
- Cloud detector 对 scheduler 选择的 slice 运行。
- 每个 slice 检测框先从 slice-local 坐标映射到 resized full-frame 坐标。
- 对整帧所有 edge/cloud detection 进行 class-aware global NMS。
- 最终 AP50、Recall、small-object Recall 必须在整帧 full-frame detection 与 full-frame GT 上计算。
- 不允许直接平均或累加 slice-level AP 作为帧级性能。
- overlap 造成的重复检测必须由 global NMS 消除。

# 任务 2：统一结果格式与坐标映射

检查并完善 src/detection/result_schema.py。

每个 detection 至少保存：

- frame_id
- slice_id
- bbox_xyxy_local
- bbox_xyxy_frame
- score
- category_id
- source: edge or cloud
- quality: none/low/mid/high/full
- slice_x1
- slice_y1
- slice_x2
- slice_y2

新增 src/detection/coordinate_utils.py，实现并测试：

- local_to_frame_bbox()
- frame_to_local_bbox()
- clip_bbox_to_frame()
- global_classwise_nms()

新增 scripts/test_detection_coordinate_pipeline.py：

- 使用一张 3840 配置 PANDA 图像。
- 对所有 slice 生成若干确定性 mock boxes。
- 映射回整帧后执行 global NMS。
- 输出 results/phase1_5/coordinate_pipeline_check.png。
- 可视化 slice grid、局部框映射结果、NMS 前后框数。
- 断言所有最终 bbox 均在 3840 图像边界内。

# 任务 3：WSL / ViTDet 环境审计

完善 src/detection/vitdet_runner_stub.py 和 README。

新增 scripts/check_vitdet_wsl_env.py，仅做环境检查，不跑推理。

该脚本或文档应检查并报告：

- WSL distro 名称
- Python 版本
- PyTorch 版本
- CUDA 是否可用
- GPU 名称
- Detectron2 是否可 import
- ViTDet config 是否存在
- ViTDet checkpoint 是否存在

不要硬编码我的路径；所有路径通过 configs/panda_3840.yaml 或环境变量配置。

新增 configs/vitdet_example.yaml，仅提供占位字段：

- wsl_distro
- python_executable
- vitdet_repo_root
- edge_config
- edge_checkpoint
- cloud_config
- cloud_checkpoint
- device

# 任务 4：最小真实推理接口

不要求现在一定安装成功 Detectron2。

为以下脚本补齐可执行接口和错误提示：

- scripts/run_edge_detection.py
- scripts/run_cloud_detection.py

要求：

- 支持 --backend mock 或 --backend vitdet。
- mock 保持现有行为。
- vitdet 模式下如果 WSL / Detectron2 / config / checkpoint 缺失，必须打印清晰诊断，不要静默回退到 mock。
- 支持 --max-frames 1 和 --max-slices 2。
- Cloud 脚本支持 --quality full。

# 任务 5：测试与文档

运行：

python scripts/test_detection_coordinate_pipeline.py --config configs/panda_3840.yaml --max-frames 1
python scripts/check_vitdet_wsl_env.py --config configs/vitdet_example.yaml

更新：

- README.md
- EXPERIMENT_STATUS.md
- CODEX_RUN_LOG.md

完成后报告：

1. 坐标映射测试是否通过；
2. global NMS 前后检测框数量；
3. WSL、PyTorch、CUDA、Detectron2、ViTDet config/checkpoint 的当前状态；
4. 哪些信息需要我手动补充；
5. 不要开始批量推理，不要进入 Phase 2。
