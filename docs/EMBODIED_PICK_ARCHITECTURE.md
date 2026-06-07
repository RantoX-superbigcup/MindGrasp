# 一念拾取具身抓取框架

## 模块关系

`HybridBCI/P300/SSVEP -> IPCBridge -> BCICommandRouter -> QwenVisionClient -> GraspNetAdapter -> GraspPlanner/MoveIt2Planner -> SerialArmController`

## 每层职责

- `BCICommandRouter`：把平台输出的 A-H 或 1-8 指令映射成上一个、下一个、确认、执行、急停等高层动作。
- `QwenVisionClient`：调用 Qwen-VL API 做目标理解、遮挡判断和澄清问题生成。
- `GraspNetAdapter`：调用现有 `graspnet_baseline`，从 RGB-D 数据生成抓取候选。
- `GraspPlanner`：当前先选最高分抓取并生成 pregrasp/grasp/lift/place 步骤。
- `MoveIt2Planner`：预留 ROS 2 MoveIt 接口，后续把抓取位姿转成真正避障轨迹。
- `SerialArmController`：把规划结果转换为机械臂串口命令，目前兼容 `<beta1;L4>` 协议。

## 当前可运行模式

默认 `configs/default_config.json` 中 `dry_run=true`，不需要 Qwen API、GraspNet checkpoint、真实机械臂，也能跑通状态机。

运行：

```powershell
python .\run_embodied_demo.py --config .\configs\default_config.json --command E --pretty
```

## 接入真实 Qwen API

1. 设置环境变量 `QWEN_API_KEY`。
2. 在 `configs/default_config.json` 中确认 `qwen_base_url` 和 `qwen_model`。
3. 把 `dry_run` 改为 `false`。

## 接入真实 GraspNet

1. GraspNet baseline 源码随本仓库放在 `third_party/graspnet-baseline/`。
2. 将官方预训练权重放到 `weights/graspnet/checkpoint-rs.tar`。
3. 将 `configs/default_config.json` 中的 `dry_run` 改成 `false`。
4. 确认 RGB-D 输入目录包含 `color.png`、`depth.png`、`workspace_mask.png`、`meta.mat`。

权重文件不直接提交到 Git，避免仓库过大。需要共享时建议使用 GitHub Releases、网盘或 Git LFS。
## 接入机械臂

当前 Arduino 代码接收 `<beta1;L4>`，因此框架里的 `SerialArmController` 会把抓取候选位置转换成 `beta` 和 `L4` 后通过串口发送。

更稳定的后续方案是改 Arduino 协议，直接接收关节角或规划后的 waypoint，例如 `<J;alpha1;alpha2>` 或 `<P;x;y;z;gripper>`。

## 和现有 PyQt demo 的连接点

现有 `app/main_window.py` 在 `on_server_data` 中已经解析 `ipc_algorithm_test`：

```python
result = ipc_json_data['result_args']['data']
```

后续只需要在这里把 `ipc_json_data` 交给 `IPCBridge.handle_platform_json(...)`，就能让脑机指令进入完整 pipeline。