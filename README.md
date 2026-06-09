# IntentGrasp

脑机接口引导的具身机械臂抓取框架。当前版本采用分层架构：Qwen-VL 负责场景理解和 A/B/C/D 选项生成，脑机接口负责选择，GraspNet 负责抓取候选，规划器负责轨迹，机械臂控制层负责串口执行。

## 当前目标

第一阶段先跑通 dry-run 状态机，确认完整链路可执行。第二阶段接入 Qwen API 和 RGB-D 相机，让模型根据画面生成 A/B/C/D 选项。第三阶段接入 GraspNet 和自制机械臂串口控制。第四阶段再加入障碍物、澄清和失败回退。

## 快速验证

默认 `configs/default_config.json` 是全 dry-run，不会调用真实模型和机械臂：

```powershell
cd "E:\IntentGrasp"
python -B .\run_embodied_demo.py --config .\configs\default_config.json --command E --pretty
```

看到下面结果说明状态机跑通：

```json
{
  "status": "done"
}
```

## 只测试真实 GraspNet

如果只想真实调用 GraspNet，但 Qwen 和机械臂仍然保持模拟，使用：

```powershell
cd "E:\IntentGrasp"

$env:CUDA_HOME="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"
$env:CUDA_PATH=$env:CUDA_HOME
$env:Path="$env:CUDA_HOME\bin;$env:CUDA_HOME\libnvvp;$env:Path"
$torchLib = python -c "import os, torch; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"
$env:Path="$torchLib;$env:Path"

python -B .\run_embodied_demo.py --config .\configs\graspnet_only_config.json --command E --pretty
```

该配置含义是：

```json
{
  "qwen_dry_run": true,
  "graspnet_dry_run": false,
  "planner_dry_run": true,
  "robot_dry_run": true
}
```



## GraspNet 依赖目录

官方 `graspnet-baseline` 源码放在：

```text
third_party/graspnet-baseline/
```

默认配置使用仓库内相对路径：

```json
{
  "graspnet_root": "third_party\\graspnet-baseline",
  "graspnet_checkpoint": "weights\\graspnet\\checkpoint-rs.tar"
}
```

真实推理前，把官方预训练权重放到：

```text
weights/graspnet/checkpoint-rs.tar
```

权重文件不建议直接提交到 Git。团队共享时建议使用 GitHub Releases、网盘或 Git LFS。

## Windows 编译备注

建议把项目放在纯英文路径，例如：

```text
E:\IntentGrasp
```

中文路径下 `nvcc + ninja + PyTorch C++ extension` 容易出现乱码路径和 `.obj` 生成失败。

本项目已对 GraspNet Windows 编译做过几个兼容补丁：

- `pointnet2/setup.py` 和 `knn/setup.py` 添加 `-allow-unsupported-compiler`。
- `pointnet2/setup.py` 和 `knn/setup.py` 添加 `_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH`。
- `dataset/graspnet_dataset.py` 将 `torch._six` 替换为 `collections.abc`，兼容 PyTorch 2.x。
- `knn` 中将索引类型从 Windows 不稳定的 `long` 改为 `int64_t`。
- `demo.py` 在没有 `grasp_nms` 时允许跳过 NMS。

RTX 2070 推荐：

```powershell
$env:TORCH_CUDA_ARCH_LIST="7.5"
$env:MAX_JOBS="1"
```

## Qwen 的作用

Qwen 不直接控制机械臂。它负责看 RGB 图像并生成可供脑机接口选择的选项，例如：

```json
{
  "question": "请选择你想抓取的物体",
  "options": [
    {"key": "A", "label": "红色杯子", "target_id": "red_cup"},
    {"key": "B", "label": "药瓶", "target_id": "medicine_bottle"},
    {"key": "C", "label": "勺子", "target_id": "spoon"},
    {"key": "D", "label": "纸巾", "target_id": "tissue"}
  ]
}
```

脑机接口输出 A/B/C/D 后，系统再把选项映射成具体目标，交给目标定位、GraspNet 和规划器。

API key 不要写进代码，建议用环境变量：

```powershell
$env:QWEN_API_KEY="你的 API Key"
```

如需永久保存：

```powershell
setx QWEN_API_KEY "你的 API Key"
```

`setx` 后需要重新打开 PowerShell。


## 目标选择 + GraspNet 过滤 Demo

如果只想验证“先识别可选物体，再只显示/输出目标物体的抓取姿态”，不要启用 BCI 和机械臂，使用独立脚本：

```powershell
cd "E:\IntentGrasp"

$env:QWEN_API_KEY="你的 API Key"
$env:CUDA_HOME="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"
$env:CUDA_PATH=$env:CUDA_HOME
$env:Path="$env:CUDA_HOME\bin;$env:CUDA_HOME\libnvvp;$env:Path"
$torchLib = python -c "import os, torch; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"
$env:Path="$torchLib;$env:Path"

python -B .\run_target_grasp_demo.py --max-options 8
```

第一次不加 `--choice` 时，脚本只会调用 Qwen-VL，根据 `third_party/graspnet-baseline/doc/example_data/color.png` 生成 A/B/C/D 选项和 bbox。看到选项后，例如想选 A，再运行：

```powershell
python -B .\run_target_grasp_demo.py --choice A
```

如果不想第二次重新调用 Qwen，可以直接复用第一次保存的选项文件：

```powershell
python -B .\run_target_grasp_demo.py --options-json .\outputs\target_grasp\qwen_options.json --choice A
```

脚本会执行：

```text
Qwen-VL 看 RGB 图像生成 A/B/C/D 和 bbox
-> 根据选择的 bbox 运行目标过滤
-> GraspNet 对 RGB-D 场景生成抓取候选
-> 将抓取中心投影回图像，只保留目标 bbox 附近的候选
-> Open3D 只显示目标区域点云和该目标的抓取姿态
-> 终端输出 top grasp 的 translation / quaternion / rotation_matrix / score / width
```

如果暂时不想调用 Qwen，也可以手工给 bbox 测试完整 GraspNet 过滤链路：

```powershell
python -B .\run_target_grasp_demo.py --bbox 300,200,700,650 --choice A --no-vis --top-k 3
```

注意：Qwen 只需要 RGB 图像就能生成选项；GraspNet 真实抓取姿态必须要 RGB-D 数据，即同一目录下包含：

```text
color.png
depth.png
workspace_mask.png
meta.mat
```




### 正式自动链路：Qwen 只做选项，GroundingDINO/SAM 做定位

病人使用场景不能手工框选，也不能直接相信 Qwen 的 bbox。Qwen-VL 在这里只负责识别桌面上有哪些可抓取物体，并生成 A/B/C/D/E/F/G/H 选项；真正进入 GraspNet 前，必须由自动定位/分割模块得到目标区域。

推荐正式流程：

```text
RGB 图像
-> Qwen-VL 生成可选物体列表
-> 用户/BCI 选择 A/B/C...
-> GroundingDINO 根据选中物体名称自动定位 bbox
-> SAM 根据 bbox 自动分割目标 mask
-> GraspNet 只筛选 mask 内的抓取候选
-> 输出抓取姿态
```

运行方式：

```powershell
python -B .\run_target_grasp_demo.py --max-options 8
python -B .\run_target_grasp_demo.py --options-json .\outputs\target_grasp\qwen_options.json --choice A --localizer groundingdino --segmenter sam --vis-mode compare --top-k 8
```

如果没有安装 GroundingDINO/SAM 或没有放权重，脚本应该报错并提示缺什么，而不是退回 Qwen 粗框继续抓。`--bbox-ui` 和 `--allow-qwen-bbox-grasp` 只用于开发调试，不属于病人可用方案。

### 开发调试兜底：人工框选 bbox

`--bbox-ui` 只用于开发阶段判断“如果目标区域完全正确，GraspNet 后半段能不能工作”。它不属于病人端或正式自动方案，因为病人无法手工框选。

只有在排查问题时才使用：

```powershell
python -B .\run_target_grasp_demo.py --options-json .\outputs\target_grasp\qwen_options.json --choice A --bbox-ui --segmenter grabcut --vis-mode compare --top-k 8
```

如果这条命令能抓对，而自动链路抓不对，说明问题在自动定位/分割；如果这条也抓不对，才继续排查 GraspNet 候选筛选和排序。

### 当前图片的关键问题

这张桌面图里不止 4 个物体，至少有香蕉、灰色玩具、白色瓶子、蓝色碗、白红瓶、红色盒子、电动螺丝刀等多个可抓取目标。因此不能只生成 A/B/C/D 四个粗框；脚本默认已经改为 `--max-options 8`。

如果选择 A 香蕉后发现 bbox 把白色瓶子、碗或盒子也框进去了，说明“目标定位/分割”还没过关，不应该继续相信 GraspNet 的抓取排序。此时先用二次 Qwen 精定位并打开左右对比视图：

```powershell
python -B .\run_target_grasp_demo.py --options-json .\outputs\target_grasp\qwen_options.json --choice A --localizer groundingdino --segmenter sam --vis-mode compare --top-k 8
```

对比窗口里：左边是完整原始点云，红色点是当前选中的 mask；右边是单独拉出来的目标点云和抓取姿态。只有当红色点/右侧点云确实是香蕉时，后面的 GraspNet 姿态才有意义。

### 可选：GroundingDINO / SAM 精定位

当前独立脚本已经预留了 GroundingDINO 和 SAM 的入口，默认先不依赖它们，直接使用 Qwen 给出的 bbox 跑通目标筛选：

```text
Qwen-VL 先生成 A/B/C/D 选项
-> 选择 A/B/C/D
-> qwen_bbox：直接使用 Qwen 给出的 bbox
-> groundingdino：用选中的 label/target_id 重新定位 bbox
-> grabcut：不装 SAM 时，用 OpenCV 把 bbox 粗分割成前景 mask
-> sam：装好 SAM 后，把 bbox 细化成更准确的 mask
-> GraspNet 全场景预测抓取候选
-> 只保留投影落在目标 mask/bbox 内的抓取姿态
```

默认跑法：

```powershell
python -B .\run_target_grasp_demo.py --choice A --localizer groundingdino --segmenter sam
```

注意：`--segmenter none` 只使用矩形 bbox，香蕉这类细长/弯曲物体很容易把旁边物体一起框进去；优先用 `grabcut`，最终版建议用 `sam`。

后续如果你放好 GroundingDINO 和 SAM 权重，可以切换为：

```powershell
python -B .\run_target_grasp_demo.py --choice A --localizer groundingdino --segmenter sam
```

默认权重路径：

```text
weights/groundingdino/GroundingDINO_SwinT_OGC.py
weights/groundingdino/groundingdino_swint_ogc.pth
weights/sam/sam_vit_b_01ec64.pth
```

脚本会生成这些调试文件，用来检查模型到底选中了哪里：

```text
outputs/target_grasp/qwen_options.json
outputs/target_grasp/qwen_options_overlay.png
outputs/target_grasp/target_mask.png
outputs/target_grasp/target_overlay.png
outputs/target_grasp/target_grasps.json
```


## GraspNet 的定位：抓取候选，不是目标选择

GraspNet 本身不能理解“我要抓哪个物品”。它的能力是：输入一组 RGB-D 场景数据，输出许多可行的 6D 抓取候选位姿。它回答的是“哪里可以抓、怎么抓更稳”，而不是“用户想抓杯子还是药瓶”。

因此，想实现“想要哪个物品就抓哪个”，需要在 GraspNet 前后增加目标理解和筛选步骤：

```text
RGB/RGB-D 场景输入
-> Qwen-VL 识别场景并生成 A/B/C/D 物品选项
-> 脑机接口选择目标，例如 B 药瓶
-> Qwen / GroundingDINO / SAM 得到目标 bbox 或 mask
-> GraspNet 生成全场景抓取候选
-> 根据目标 bbox/mask 过滤抓取候选
-> 执行属于目标物体的最佳抓取位姿
```

第一阶段可以先做简化版：Qwen 给出目标的大致位置或 bbox，GraspNet 输出全场景候选，然后选择离目标 bbox 中心最近的抓取点。后续再升级为 SAM/GroundingDINO 的精确 mask 过滤。

项目中应把 GraspNet 理解为“抓取候选生成器”，而不是语言理解模型、目标选择模型或端到端机械臂控制模型。

## 自制机械臂串口协议

当前机械臂采用 1-5 号电机角度控制，角度单位为度，范围为 0-180。根据最新确认的编号，电机含义固定为：

| 编号 | 位置 | 作用 |
|---|---|---|
| 1号 | 腕部 | 控制夹爪整体姿态/旋转 |
| 2号 | 夹爪 | 控制夹爪张开/闭合 |
| 3号 | 小臂 | 控制小臂伸展/俯仰 |
| 4号 | 大臂 | 控制大臂抬起/放下 |
| 5号 | 底座 | 控制整条机械臂水平旋转 |

当前建议把坏电机位置放到 1 号腕部。这样 2 号夹爪仍然可用，抓取闭合动作可以保留；1 号腕部先固定在 home 角度即可。

推荐协议：

```text
<J;motor1;motor2;motor3;motor4;motor5>
```

示例：

```text
<J;90;45;80;110;90>
```

含义：

```text
motor1 = 90   # 腕部，占位/固定；当前坏位默认放这里
motor2 = 45   # 夹爪开合
motor3 = 80   # 小臂
motor4 = 110  # 大臂
motor5 = 90   # 底座水平旋转
```

紧急停止协议：

```text
<STOP>
```

相关代码在 `embodied_pick/robot.py`：

- `JointAngles`：1-5 号电机角度结构体。
- `SerialArmController.send_joint_angles()`：直接发送指定角度。
- `SerialArmController._format_joint_angles()`：生成串口字符串。
- `SerialArmController._plan_to_joint_angles()`：临时把抓取计划转换成关节角，后续应替换为真实逆运动学。

当前配置：

```json
{
  "robot_protocol": "joint_angles",
  "robot_disabled_joints": [1],
  "robot_home_angles": [90, 90, 90, 90, 90]
}
```


## RealSense 一键抓取姿态流程

现在推荐使用总控脚本完成完整闭环：

```text
RealSense 采集 RGB-D
-> Qwen-VL 生成可读目标选项
-> 输入 A/B/C... 选择目标
-> GroundingDINO 自动定位目标 bbox
-> SAM 自动分割目标 mask
-> GraspNet 只筛选目标区域内的抓取候选
-> 输出后续机械臂可读取的 grasp_pose.json
```

运行：

```powershell
cd "E:\IntentGrasp"
conda activate uno
$env:QWEN_API_KEY="你的 API Key"
python -B .\run_realsense_grasp_workflow.py
```



默认采集不是只保存最后一帧深度，而是在按下 `s` 后连续采集 9 帧 depth，并用中值融合生成 `depth.png`。这能减少 RealSense 的碎点、空洞和漂浮点，但它仍然只是单视角点云，不能恢复物体背面。
如果想调节融合帧数：

```powershell
python -B .\run_realsense_grasp_workflow.py --depth-frames 15 --depth-fusion median
```

如果要退回原来的单帧行为：

```powershell
python -B .\run_realsense_grasp_workflow.py --depth-frames 1 --depth-fusion last
```

脚本会打开 RealSense 预览窗口，按 `s` 保存当前帧，然后终端会显示类似：

```text
A. 香蕉 [banana] - 画面中部偏左的黄色弯曲香蕉
B. 红色饼干盒 [cheezit_box] - 右侧竖放的红色饼干盒
```

输入目标字母后，脚本会自动输出抓取姿态。结果保存在：

```text
outputs/realsense_workflow/run_时间戳/qwen_options.json
outputs/realsense_workflow/run_时间戳/qwen_options_overlay.png
outputs/realsense_workflow/run_时间戳/target_overlay.png
outputs/realsense_workflow/run_时间戳/target_mask.png
outputs/realsense_workflow/run_时间戳/target_grasps.json
outputs/realsense_workflow/run_时间戳/grasp_pose.json
outputs/realsense_workflow/run_时间戳/workflow_result.json
```

`grasp_pose.json` 是后续接机械臂最重要的文件，字段包括：

```json
{
  "status": "pose_ready_camera_frame",
  "grasp_pose": {
    "frame_id": "camera_color_optical_frame",
    "position_m": [0.0, 0.0, 0.0],
    "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0]
  },
  "gripper": {
    "width_m": 0.05
  }
}
```

注意：这个姿态目前是相机坐标系下的抓取位姿。真正驱动机械臂前，还需要手眼标定矩阵 `T_base_camera`，把相机坐标转成机械臂基座坐标，再做 IK/轨迹规划。

调试阶段默认会打开 Open3D 的 `compare` 可视化：左边用于检查原始点云/目标 mask，右边只显示选中目标区域和抓取姿态。如果只想后台验证输出文件，可以加 `--no-vis`。

如果 GroundingDINO 对某个 Qwen 选项名找不到目标，例如 `durian_case` 被转成 `durian case` 后无法检测，脚本会自动尝试多组英文定位短语，如 `green spiky toy`、`toy case`、`case`，并用 Qwen 的粗 bbox 只做候选框排序参考。Qwen 新生成的选项也会包含 `grounding_prompts` 字段，后续扩展新物体时优先检查这个字段是否足够像“检测器能听懂的英文物体短语”。



如果只想用已有采集目录重跑，不重新拍摄：

```powershell
python -B .\run_realsense_grasp_workflow.py --frame-dir .\captures\realsense_20260609_190015_filtered
```

如果想直接指定选择项，例如自动选择 A：

```powershell
python -B .\run_realsense_grasp_workflow.py --choice A
```

也可以用快捷脚本：

```powershell
.\scripts\run_realsense_grasp_workflow.ps1
.\scripts\run_realsense_grasp_workflow.ps1 -Choice A
```

## 后续接入顺序

1. 保持 `dry_run=true`，先确认平台脑机指令能触发状态机。
2. 接入 Qwen API，让它根据相机图像生成 A/B/C/D 选项。
3. 接入 Arduino 串口，先只测试 `<J;90;90;90;90;90>` 和 `<STOP>`。
4. 接入 RGB-D 相机和 GraspNet，获得抓取候选位姿。
5. 加入目标 bbox/mask 过滤，实现“选哪个物品就抓哪个”。
6. 标定相机坐标系、机械臂坐标系和关节零位。
7. 用逆运动学替换 `_plan_to_joint_angles()`，再做真实抓取。
