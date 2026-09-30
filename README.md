# Gaze-DMS

基于计算机视觉的人眼视线追踪与驾驶员状态监测原型。项目组合视线/头部姿态估计、人脸特征分析和目标检测，尝试识别疲劳、分心及使用手机等状态，并通过桌面界面展示结果。

> 本项目用于研究和演示，不是经过认证的驾驶安全设备。请勿将其作为实际驾驶安全判断的唯一依据。

## 功能概览

- 从摄像头或视频流读取画面并进行实时处理。
- 通过人脸特征点估计闭眼、张嘴和头部偏转等行为。
- 融合行为、视线与目标检测信号，生成驾驶员状态提示。
- 使用 Tkinter 展示视频帧、检测信息和状态。

## 项目结构

| 路径 | 用途 |
| --- | --- |
| `main.py`、`video_processor.py` | 桌面应用入口与视频处理线程 |
| `my_detector.py`、`tracker.py`、`retinaface.py` | 目标（YOLO）、人脸/特征点（ONNX）与视线推理 |
| `behavior_detector.py`、`EAR.py`、`MAR.py` | 疲劳（闭眼/哈欠）与分心行为分析 |
| `gaze_estimation/` | MPIIGaze / MPIIFaceGaze 视线估计模型与数据处理 |
| `train.py`、`evaluate.py`、`convert_to_onnx.py` | 视线估计模型的训练、评估与 ONNX 导出（用法见 [`TRAINING.md`](TRAINING.md)） |
| `models/`、`utils/` | 模型结构及检测、训练辅助代码 |
| `ui/` | Tkinter 图形界面 |
| `configs/`、`data/calib/` | 训练/演示配置与相机标定参数 |
| `tools/`、`scripts/` | 数据预处理、视频采集和辅助脚本 |

## 环境安装

推荐使用 Python 3.10。创建虚拟环境后安装依赖：

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`torch` / `torchvision` 的安装方式可能因操作系统、CUDA 版本和硬件而异；如默认安装包不适用，请按 [PyTorch 安装说明](https://pytorch.org/get-started/locally/)选择对应版本。Windows 上安装 `dlib` 也可能需要兼容的预编译包或 C++ 构建工具。

## 模型与数据

仓库已自带运行所需的模型权重（来源与许可见 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)）：

| 文件 | 用途 | 随仓库分发 |
| --- | --- | --- |
| `outputs/exp5/weights/best.pt` | YOLO 行为检测（类别：`face`、`phone`） | ✅ 已包含 |
| `models/*.onnx`、`models/priorbox_640x640.json` | RetinaFace 人脸检测、68 点特征点、头部姿态/视线（OpenSeeFace，BSD-2） | ✅ 已包含 |
| `data/models/mpiifacegaze/resnet_simple_14/checkpoint_0015.pth` | MPIIFaceGaze 视线估计 | ✅ 已包含 |
| `data/calib/*.yaml` | 相机与归一化相机参数 | ✅ 已包含 |

需要**自行下载**的内容（因许可限制不随仓库分发）：

```bash
# dlib 68 点人脸特征模型（训练自 iBUG 300-W，仅限非商业研究用途）
bash scripts/download_dlib_model.sh
# 下载后确认文件位于 data/dlib/shape_predictor_68_face_landmarks.dat
```

数据集（MPIIGaze / MPIIFaceGaze）以及训练产生的 `datasets/`、`experiments/` 同样不随仓库分发，获取方式见 [`TRAINING.md`](TRAINING.md)。模型路径可在 `config.py` 中按实际位置调整。

仓库还包含一段演示样例视频 `test.mp4`（10.87 秒 / 1920×1080 / 30fps），可在没有摄像头时用于离线演示：把 `config.py` 的 `gaze_config.demo.use_camera` 设为 `False`、`video_path` 指向 `test.mp4` 即可（默认配置使用摄像头，不受影响）。

## 训练

视线估计模型的完整训练流程（数据集下载与预处理、配置文件字段说明、四个模型的训练命令、训练产物、评估与 ONNX 导出、硬件需求、常见报错）见 [`TRAINING.md`](TRAINING.md)。行为检测（YOLO）部分本仓库只提供推理代码。

## 启动

准备好上述依赖和模型文件后，在项目根目录运行：

```powershell
python main.py
```

应用默认尝试使用摄像头。摄像头权限、可用设备、模型兼容性和算力会影响运行结果。


## 许可

本项目自有代码以 **GNU GPL-3.0** 发布，全文见 [`LICENSE`](LICENSE)。

仓库中还包含第三方组件（许可分别为 MIT、BSD-2-Clause、GPL-3.0），其版权声明与许可证原文保存在 [`LICENSES/`](LICENSES) 目录，对应关系见 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。再分发或修改本项目时，请一并遵守这些条款。
