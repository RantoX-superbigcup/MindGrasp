# MindGrasp 打包说明

## 两种包

1. 本机平台联调包：运行 `package.bat`，输出 `dist\python_demo\python_demo.exe`。这个包只保证当前开发机可用，真实 GraspNet/GroundingDINO/SAM 会调用本机 Python 环境。
2. 便携发布包：运行 `package_portable.bat`，输出完整目录 `dist\python_demo`，其中包含 `runtime\python\python.exe`。给别人电脑时必须复制整个 `dist\python_demo` 文件夹，不能只复制单个 exe。

## 运行时查找顺序

打包后的 GUI 会按下面顺序寻找真实抓取流程使用的 Python：

1. 环境变量 `MINDGRASP_PYTHON` 指定的目标电脑 Python 环境
2. `python_demo.exe` 同级目录下的 `runtime\python\python.exe`
3. `python_demo.exe` 同级目录下的 `runtime\python.exe`
4. `python_demo.exe` 同级目录下的 `python\python.exe`
5. 系统 `python`

因此，发给其他电脑时推荐使用便携发布包结构：

```text
dist/python_demo/
  python_demo.exe
  _internal/
  runtime/python/python.exe
  runtime/python/Lib/
  runtime/python/Scripts/
```

第一次在新电脑或新路径运行时，程序会尝试自动执行 `runtime\python\Scripts\conda-unpack.exe`，修正便携 conda 环境路径。

## 目标电脑仍然需要的外部条件

便携包不能替代硬件和驱动。目标电脑仍然需要：

- Windows x64。
- NVIDIA 显卡驱动能支持当前 PyTorch/CUDA 版本。
- 如果使用 RealSense 真机采集，需要安装 Intel RealSense 驱动/运行库，并接入相机。
- 如果调用 Qwen，需要设置 `QWEN_API_KEY`。
- 如果控制机械臂，需要目标电脑能访问对应串口，并确认串口号/波特率配置。

## 构建机路径

`package.bat` 和 `package_portable.bat` 默认使用当前开发机的：

```text
E:\XWJ\anaconda\envs\uno\python.exe
E:\XWJ\anaconda\envs\uno
E:\XWJ\anaconda\Scripts\conda-pack.exe
```

如果换构建机，先设置下面变量再运行脚本：

```bat
set MINDGRASP_BUILD_PYTHON=D:\path\to\env\python.exe
set MINDGRASP_CONDA_ENV=D:\path\to\env
set MINDGRASP_CONDA_PACK=D:\path\to\conda-pack.exe
package_portable.bat
```
