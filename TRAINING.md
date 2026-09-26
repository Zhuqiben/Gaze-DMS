# Gaze-DMS 训练教程（视线估计）

本教程基于仓库源码整理，覆盖**视线估计模型（Gaze Estimation）**的完整训练流程：
数据准备 → 预处理 → 配置 → 训练 → 评估 → ONNX 导出。

> 说明：仓库当前的训练入口 `train.py` 只训练视线估计模型（MPIIGaze / MPIIFaceGaze）；
> 数据准备与训练/评估代码沿用 [Zhuqiben/pytorch_mpiigaze](https://github.com/Zhuqiben/pytorch_mpiigaze)（上游为 [hysts/pytorch_mpiigaze](https://github.com/hysts/pytorch_mpiigaze)）；
> 行为检测（YOLO）部分在本仓库只有推理代码，没有训练入口，详见 [第 12 章](#第-12-章行为检测yolo部分的说明)。
>
> 与上游的差异（已逐文件比对）：`train.py`、`evaluate.py`、`convert_to_onnx.py`、`tools/`、`gaze_estimation/` 内容一致；
> 仅 `configs/mpiifacegaze/*.yaml` 的 `num_workers` 由 `4` 改为 `0`，demo 相关配置被本地化，`TRAINING.md` 为本仓库新增。

---

## 目录

1. [训练入口](#第-1-章训练入口)（问题 1）
2. [支持的模型](#第-2-章支持的模型)（问题 2）
3. [数据集获取与目录布局](#第-3-章数据集获取与目录布局)（问题 3）
4. [数据预处理](#第-4-章数据预处理)（问题 4）
5. [配置文件字段详解](#第-5-章配置文件字段详解)（问题 5）
6. [训练命令](#第-6-章训练命令)（问题 6）
7. [训练输出](#第-7-章训练输出)（问题 7）
8. [评估与 ONNX 导出](#第-8-章评估与-onnx-导出)（问题 8）
9. [硬件需求与耗时](#第-9-章硬件需求与耗时)（问题 9）
10. [常见报错与解决](#第-10-章常见报错与解决)（问题 10）
11. [命令速查表](#第-11-章命令速查表)
12. [行为检测（YOLO）部分的说明](#第-12-章行为检测yolo部分的说明)

---

## 第 0 章：整体流程速览

**目的**：先建立全局认知，知道每条命令在流程中的位置。

```
          datasets/MPIIGaze(.h5)                        datasets/MPIIFaceGaze.h5
                   │                                            │
   tools/preprocess_mpiigaze.py                  tools/preprocess_mpiifacegaze.py
                   │                                            │
                   ├──────────────► configs/*/*_train.yaml ◄─────┤
                   │                       │
                   │                  train.py  ──► experiments/<dataset>/<arch>/exp<ID>/<test_id>/
                   │                       │                ├── config.yaml
                   │                       │                ├── log.txt / log_plain.txt
                   │                       │                ├── events.out.tfevents.*
                   │                       │                └── checkpoint_XXXX.pth
                   │                       ▼
                   │                 evaluate.py ──► <test.output_dir>/<checkpoint 名>/error.txt
                   │
                   └────────────────► convert_to_onnx.py ──► *.onnx
```

**参数说明**：本项目所有训练/评估超参数都通过 YAML 配置文件 + 命令行 `key value` 覆盖，没有单独的 `--batch-size` 之类的开关。

**注意事项**：仓库自带 `scripts/*.sh` 是 Linux（bash）脚本；在 Windows 上需要 Git Bash / WSL，或按本教程给出的 PowerShell 等价命令执行。

---

## 第 1 章：训练入口

**目的**：明确“用哪个文件训练、怎么传参”。

### 1.1 入口文件

| 文件 | 作用 |
| --- | --- |
| [train.py](./train.py) | 训练入口，含 `train()`、`validate()`、`main()` |
| [evaluate.py](./evaluate.py) | 评估入口，计算平均角度误差并保存预测结果 |
| [convert_to_onnx.py](./convert_to_onnx.py) | 把 checkpoint 导出为 ONNX |
| [gaze_estimation/utils.py](./gaze_estimation/utils.py) | `load_config()` 参数解析、随机种子、cuDNN、输出目录创建 |

`train.py` 的 `main()` 流程（源码顺序）：

1. `load_config()` 解析参数并合并配置；
2. `set_seeds(config.train.seed)` / `setup_cudnn(config)`；
3. `create_train_output_dir(config)` 创建输出目录（**已存在则直接报错**）；
4. `save_config()` 写出 `config.yaml`，`create_logger()` 写出 `log.txt`；
5. `create_dataloader()` → `create_model()` → `create_loss()` → `create_optimizer()` → `create_scheduler()` → `Checkpointer` → TensorBoard `SummaryWriter`；
6. 若 `train.val_first: True`，先在 epoch 0 跑一次验证（`validate(0, ...)`）；
7. 循环 `for epoch in range(1, config.scheduler.epochs + 1)`：`train()` → `scheduler.step()` → 每 `val_period` 个 epoch 验证 → 每 `checkpoint_period` 个 epoch 保存 checkpoint（最后一个 epoch 必存）。

### 1.2 调用方式

```bash
python train.py --config <配置文件.yaml> [key value ...]
```

**参数说明**：

- `--config`：可选。不传时使用 [gaze_estimation/config/defaults.py](./gaze_estimation/config/defaults.py) 里的默认值（默认是 `MPIIGaze` + `lenet`）。
- `key value ...`：`argparse.REMAINDER` 收集的剩余参数，成对给出，由 `config.merge_from_list()` 合并。例如 `train.test_id 3`、`train.output_dir experiments/mpiigaze/lenet/exp01`。
- **不是** `key=value`（那是别的框架 OmegaConf 的写法），本仓库必须用空格分隔。
- 覆盖项必须写在 `--config` **之后**（REMAINDER 会把之后的所有内容都当成覆盖项）。

**预期输出**：终端打印带颜色的日志，形如：

```
[2025-02-15 13:25:27] __main__ INFO: <完整配置>
[2025-02-15 13:25:27] __main__ INFO: Val 0
[2025-02-15 13:25:27] __main__ INFO: Epoch 0 loss 0.1626 angle error 14.46
[2025-02-15 13:25:27] __main__ INFO: Train 1
[2025-02-15 13:25:30] __main__ INFO: Epoch 1 Step 0/1181 lr 0.100000 loss 0.1652 (0.1652) angle error 14.62 (14.62)
...
[2025-02-16 03:47:49] fvcore.common.checkpoint INFO: Saving checkpoint to experiments/mpiifacegaze/resnet_simple_14/exp00/00\checkpoint_0015.pth
```

**注意事项**：

- 训练入口**没有** `--gpu`、`--epochs`、`--lr` 等参数，全部通过配置文件/覆盖项控制。
- `config.freeze()` 会在 `load_config()` 末尾调用，配置在训练期间不可再修改（代码内部也不会改）。
- `create_train_output_dir()` 生成的子目录名是 `f'{train.test_id:02}'`，即 `test_id=0 → 00`，`test_id=-1 → all`。

---

## 第 2 章：支持的模型

**目的**：搞清楚 `config.mode` + `config.model.name` 如何映射到一个模型类。

### 2.1 模型解析机制

```python
# gaze_estimation/models/__init__.py
dataset_name = config.mode.lower()                      # 'MPIIGaze' -> 'mpiigaze'
module = importlib.import_module(
    f'gaze_estimation.models.{dataset_name}.{config.model.name}')
model = module.Model(config)
```

即：`config.mode` 决定子包目录，`config.model.name` 决定模块文件，模块内必须有一个 `class Model(config)`。
**因此新增模型只需在对应目录下新增同名 `.py` 文件，并在 yaml 中写 `model.name` 即可，无需改动 `train.py`。**

### 2.2 四个可用模型

| mode | model.name | 实现文件 | 输入张量 | 是否使用头部姿态 | 关键结构 |
| --- | --- | --- | --- | --- | --- |
| `MPIIGaze` | `lenet` | [lenet.py](./gaze_estimation/models/mpiigaze/lenet.py) | `1×36×60`（灰度） | ✅ 需要 `pose` | 2 层卷积 + 2 层全连接；`fc2 = Linear(502, 2)`，其中 500 维来自图像特征，2 维来自 pose |
| `MPIIGaze` | `resnet_preact` | [resnet_preact.py](./gaze_estimation/models/mpiigaze/resnet_preact.py) | `1×36×60`（灰度） | ✅ 需要 `pose` | ResNet-preact-**8**（`depth = 8`，`base_channels = 16`），3 个 stage 各 1 个 BasicBlock（pre-activation），`fc = Linear(feature + 2, 2)` |
| `MPIIFaceGaze` | `alexnet` | [alexnet.py](./gaze_estimation/models/mpiifacegaze/alexnet.py) | `3×448×448`（BGR） | ❌ 不需要 | torchvision AlexNet `features`（ImageNet 预训练，通道序改为 BGR）+ 3 个 1×1 卷积生成注意力 mask `x * mask`；`fc1 = Linear(256*13², 4096)` → **只能接收 448×448 输入** |
| `MPIIFaceGaze` | `resnet_simple` | [resnet_simple.py](./gaze_estimation/models/mpiifacegaze/resnet_simple.py) + [backbones/resnet_simple.py](./gaze_estimation/models/mpiifacegaze/backbones/resnet_simple.py) | `3×224×224`（BGR） | ❌ 不需要 | torchvision `ResNet`，只保留 `conv1/bn1/relu/maxpool/layer1/layer2/layer3`，删掉 `layer4/avgpool/fc`；`resnet_layers: [2,2,2]` + 隐含的 1 个 stage → 共 14 层，即 **ResNet-Simple-14**；`fc = Linear(n_channels*14², 2)` |

**重要命名约定**：

- 配置文件叫 `resnet_simple_14_train.yaml`，但里面的 `model.name` 是 **`resnet_simple`**（`14` 只是实验命名，不是模块名）。
- `transform.mpiifacegaze_face_size` 必须与模型匹配：`resnet_simple → 224`，`alexnet → 448`。填错会在 `Linear` 层报维度不匹配。

### 2.3 关键实现细节（会直接影响训练行为）

- **BGR 通道序**：AlexNet 和 ResNet backbone 都把第一层卷积权重按 `[:, [2, 1, 0]]` 重新排列，配合 `transforms.py` 中 `Normalize(mean=[0.406,0.456,0.485], std=[0.225,0.224,0.229])`（ImageNet 统计量的 BGR 版本）。不要自行改成 RGB，否则预训练权重失效。
- **预训练权重是训练/评估时联网下载的**：`resnet_simple` 走 `torch.hub.load_state_dict_from_url(torchvision.models.resnet.model_urls['resnet18'])`，`alexnet` 走 `torchvision.models.alexnet(pretrained=True)`。离线环境必须先准备好缓存。
- **AlexNet 的梯度缩放 hook**：`conv3` 注册了 `register_backward_hook`，把回传到该层的梯度除以通道数（论文实现细节），属于源码行为，不要删。
- **`no_weight_decay_on_bn`**：为 `True` 时，只有 `conv.weight` 使用 weight decay，其余参数为 0。
- **两个分支的 `forward` 签名不同**：`MPIIGaze` 是 `forward(images, poses)`，`MPIIFaceGaze` 是 `forward(images)`；`train.py` / `evaluate.py` / `convert_to_onnx.py` 都用 `config.mode` 分支处理，导出 ONNX 时的输入个数也随之不同。

**注意事项**：`config.mode` 是**大小写敏感**的字符串，只能是 `MPIIGaze` 或 `MPIIFaceGaze`；写成别的值会在 `train.py` 的 `raise ValueError` 处直接失败。

### 2.4 模型自检（可选）

**目的**：正式训练前确认「配置 → 模型类」映射正确、参数量符合预期。

**命令**（把 `--config` 换成另外三个配置即可自检其余模型）：

```bash
python -c "import sys; sys.argv = ['x', '--config', 'configs/mpiigaze/lenet_train.yaml']; from gaze_estimation.utils import load_config; from gaze_estimation import create_model; print(create_model(load_config()))"
```

**参数说明**：通过覆盖 `sys.argv` 复用 `load_config()` 的参数解析逻辑；`create_model()` 会按 `config.mode` + `config.model.name` 动态导入模块并构造模型。

**预期输出**：模型的 `repr`。LeNet 会看到 `conv1 / conv2 / fc1 / fc2`；`resnet_simple` 会看到 `feature_extractor(...)` 与 `conv / fc`；`alexnet` 会看到 `feature_extractor(...)` 和 3 个 1×1 `conv`。

**注意事项**：`model.name` 写错会直接抛 `ModuleNotFoundError: No module named 'gaze_estimation.models.<mode>.<name>'`；`resnet_simple` / `alexnet` 会触发预训练权重下载，离线环境这一步会失败（见第 10 章第 8 条）。

---

## 第 3 章：数据集获取与目录布局

**目的**：把原始数据集放到正确位置，满足 `config.dataset.dataset_dir` 的路径要求。

### 3.1 数据集与下载脚本

> 数据准备流程沿用上游实现 [Zhuqiben/pytorch_mpiigaze](https://github.com/Zhuqiben/pytorch_mpiigaze)（其上游为 MIT 许可的 [hysts/pytorch_mpiigaze](https://github.com/hysts/pytorch_mpiigaze)）。本仓库的 `scripts/download_*.sh`、`tools/preprocess_*.py`、`gaze_estimation/datasets/` 与上游逐文件比对**完全一致**，因此上游 README 中「Download the dataset and preprocess it」一节可直接套用。

| 数据集 | 下载脚本 | 下载内容 | 解压后目录 |
| --- | --- | --- | --- |
| MPIIGaze | [scripts/download_mpiigaze_dataset.sh](./scripts/download_mpiigaze_dataset.sh) | `MPIIGaze.tar.gz`（压缩包约 2.8 GB） | `datasets/MPIIGaze/` |
| MPIIFaceGaze | [scripts/download_mpiifacegaze_dataset.sh](./scripts/download_mpiifacegaze_dataset.sh) | `MPIIFaceGaze_normalized.zip` | `datasets/MPIIFaceGaze_normalized/` |

两个下载脚本的原始下载地址（与上游脚本中一致）：

```
http://datasets.d2.mpi-inf.mpg.de/MPIIGaze/MPIIGaze.tar.gz
http://datasets.d2.mpi-inf.mpg.de/MPIIGaze/MPIIFaceGaze_normalized.zip
```

> 这两个地址会 301 重定向到 HTTPS 版本；`wget`、`curl -L`、`Invoke-WebRequest` 默认都会跟随重定向，无需额外处理。

命令（Linux / Git Bash）：

```bash
bash scripts/download_mpiigaze_dataset.sh
bash scripts/download_mpiifacegaze_dataset.sh
```

Windows PowerShell 等价写法：

```powershell
New-Item -ItemType Directory -Force -Path datasets | Out-Null
Invoke-WebRequest -Uri "http://datasets.d2.mpi-inf.mpg.de/MPIIGaze/MPIIGaze.tar.gz" -OutFile "datasets\MPIIGaze.tar.gz"
tar -xzvf datasets\MPIIGaze.tar.gz -C datasets

Invoke-WebRequest -Uri "http://datasets.d2.mpi-inf.mpg.de/MPIIGaze/MPIIFaceGaze_normalized.zip" -OutFile "datasets\MPIIFaceGaze_normalized.zip"
Expand-Archive -Path datasets\MPIIFaceGaze_normalized.zip -DestinationPath datasets
# 注意：zip 内的目录名可能是 MPIIFaceGaze_normalizad（上游拼写问题），请重命名为 MPIIFaceGaze_normalized
```

下载脚本要求原始目录结构：

```
datasets/MPIIGaze/
├── Data/
│   ├── Original/          # 原始裁剪裁剪图（本仓库不使用）
│   └── Normalized/        # p00/ ... p14/ 每人若干 *.mat（每日一个 mat）
└── Evaluation Subset/
    └── sample list for eye image/
        ├── p00.txt        # 每行: <day>/<filename> <left|right>
        └── ...
```

```
datasets/MPIIFaceGaze_normalized/
├── p00.mat   # Data/data: (3000,3,224,224) uint8, Data/label: (3000,4)  # label 前 2 列为 gaze，后 2 列为 pose
├── p01.mat
└── ... p14.mat
```

**参数说明**：

- `--dataset`：原始数据集根目录（预处理脚本的输入）。
- `-o` / `--output-dir`：**HDF5 输出目录**（不是文件路径），脚本会在其中生成固定的文件名。

**预期输出**：预处理完成后得到

```
datasets/MPIIGaze.h5          # ~0.1 GB
datasets/MPIIFaceGaze.h5      # ~6.8 GB（未压缩 uint8 存储）
```

**注意事项**：

- 数据集**不随仓库分发**，`.gitignore` 已排除 `/datasets/`、`/data/models/`、`/experiments/` 等目录。请遵守各数据集的许可条款，仅用于研究。
- `tools/preprocess_mpiigaze.py` 依赖 `Evaluation Subset/sample list for eye image/p00.txt`~`p14.txt`，**该目录已包含在 `MPIIGaze.tar.gz` 中**，正常下载解压后无需任何额外步骤；若解压后找不到它，说明压缩包不完整/解压中断，请重新下载。
- `data/calib/` 下的相机参数（`sample_params.yaml`、`normalized_camera_params_*.yaml`）**只在 demo/推理阶段使用，训练不需要**。
- `create_dataset()` 里有 `assert dataset_dir.exists()`，路径写错会在这里报 `AssertionError`。

---

## 第 4 章：数据预处理

**目的**：把原始数据集转换成训练/评估直接读取的 HDF5 文件。

### 4.1 MPIIGaze

```bash
python tools/preprocess_mpiigaze.py --dataset datasets/MPIIGaze -o datasets/
```

**处理逻辑**（[tools/preprocess_mpiigaze.py](./tools/preprocess_mpiigaze.py)）：

- 遍历 `p00`~`p14`；
- 读取 `Data/Normalized/<person>/<day>.mat` 中的 `left/right` 的 `image/pose/gaze`；
- 按 `Evaluation Subset/sample list for eye image/<person>.txt` 的记录逐条生成样本；
- 左眼：`pose = convert_pose(vec) = [pitch, yaw]`，`gaze = convert_gaze(vec) = [pitch, yaw]`（弧度）；
- 右眼：图像水平翻转 `image[:, ::-1]`，`pose/gaze` 乘以 `[1, -1]`（yaw 取反）。

**输出结构**（每人 3000 条，由 `Evaluation Subset` 样本列表逐行生成，每行的 `side` 决定取左眼还是右眼）：

```
MPIIGaze.h5
└── p00/image   uint8   (3000, 36, 60)
    p00/pose    float32 (3000, 2)    # [pitch, yaw]
    p00/gaze    float32 (3000, 2)    # [pitch, yaw]
    ... p14
```

### 4.2 MPIIFaceGaze

```bash
python tools/preprocess_mpiifacegaze.py --dataset datasets/MPIIFaceGaze_normalized -o datasets/
```

**处理逻辑**（[tools/preprocess_mpiifacegaze.py](./tools/preprocess_mpiifacegaze.py)）：

- 读取每个 `pXX.mat`：`Data/data`（图像）、`Data/label`（`label[:, :2] = gaze`，`label[:, 2:] = pose`）；
- 图像由 `(N, 3, 224, 224)` 转置为 `(N, 224, 224, 3)` 并存为 `uint8`；
- `assert len(images) == len(labels) == 3000`。

**输出结构**（每人 3000 条，**每条样本一个 HDF5 dataset**）：

```
MPIIFaceGaze.h5
└── p00/image/0000 ... p00/image/2999   uint8 (224,224,3)
    p00/pose/0000  ...                   float32 (2,)
    p00/gaze/0000  ...                   float32 (2,)
    ... p14
```

### 4.3 数据集读取方式（影响训练速度）

| 数据集 | Dataset 类 | 读取策略 |
| --- | --- | --- |
| MPIIGaze | [datasets/mpiigaze.py](./gaze_estimation/datasets/mpiigaze.py) | `__init__` 中**一次性把每个人的 image/pose/gaze 全部载入内存**（每人约 6.5 MB，15 人约 97 MB），`__len__ = 3000` |
| MPIIFaceGaze | [datasets/mpiifacegaze.py](./gaze_estimation/datasets/mpiifacegaze.py) | `__getitem__` 中**每条样本都重新打开 HDF5 文件**读取，I/O 较重，`__len__ = 3000` |

**预期输出**：终端 `tqdm` 进度条，完成后无异常退出。

**注意事项**：

- 两个脚本在输出文件已存在时都会 `raise ValueError(f'{output_path} already exists.')` —— 重新预处理前先删除旧文件。
- MPIIFaceGaze 预处理会创建 135000 个 HDF5 dataset（15×3000×3），**耗时较长（数分钟到数十分钟，取决于磁盘）**，生成的文件约 6.8 GB（未压缩）。
- 需要依赖：`h5py`、`scipy`、`pandas`、`opencv-python`、`numpy`、`tqdm`（都在 [requirements.txt](./requirements.txt) 中）。
- 预处理**不做归一化/缩放**：224×224 的缩放、通道序、`Normalize` 都在训练时的 `gaze_estimation/transforms.py` 里完成。

---

## 第 5 章：配置文件字段详解

**目的**：能独立读懂并修改 [configs/](./configs) 下的 yaml。

### 5.1 默认值来源

所有字段的默认值定义在 [gaze_estimation/config/defaults.py](./gaze_estimation/config/defaults.py)（`get_default_config()` 返回其 `clone()`）。yaml 只需写要覆盖的字段，其余取默认值。

### 5.2 字段总表

| 字段 | 默认值 | 含义 |
| --- | --- | --- |
| `mode` | `MPIIGaze` | 数据集/模型族：`MPIIGaze` 或 `MPIIFaceGaze`（大小写敏感） |
| `device` | `cuda` | `cuda` 或 `cpu`。**若 `torch.cuda.is_available()` 为 False，`load_config()` 会强制改为 `cpu` 并关闭所有 `pin_memory`** |
| `dataset.dataset_dir` | `datasets/MPIIGaze.h5` | 预处理后的 HDF5 文件路径（必须存在） |
| `transform.mpiifacegaze_face_size` | `224` | MPIIFaceGaze 输入边长；`448` 表示不缩放（AlexNet 必须 448） |
| `transform.mpiifacegaze_gray` | `False` | 是否转灰度（会做直方图均衡后再转回 3 通道） |
| `model.name` | `lenet` | 模型名，对应 `gaze_estimation/models/<mode 小写>/<name>.py` |
| `model.backbone.name` | `resnet_simple` | backbone 名（仅 `resnet_simple` 用到） |
| `model.backbone.pretrained` | `resnet18` | torchvision 预训练权重名，可设为空字符串 `''` 表示不加载预训练 |
| `model.backbone.resnet_block` | `basic` | `basic` 或 `bottleneck` |
| `model.backbone.resnet_layers` | `[2, 2, 2]` | 三个 stage 的 block 数（`layer4` 会被删除） |
| `train.batch_size` | `64` | 训练/验证 batch size（验证集复用同一值） |
| `train.optimizer` | `sgd` | `sgd` / `adam` / `amsgrad` |
| `train.base_lr` | `0.01` | 初始学习率 |
| `train.momentum` | `0.9` | SGD 动量 |
| `train.nesterov` | `True` | SGD Nesterov |
| `train.weight_decay` | `1e-4` | 权重衰减 |
| `train.no_weight_decay_on_bn` | `False` | 为 True 时只有 `conv.weight` 施加 weight decay |
| `train.loss` | `L2` | `L1` / `L2`（MSE）/ `SmoothL1`，作用于 `[pitch, yaw]` |
| `train.seed` | `0` | 随机种子（Python/NumPy/Torch/CUDA 全部设置） |
| `train.val_first` | `True` | 训练前先在验证集跑一次（epoch 0），可当作数据连通性自检 |
| `train.val_period` | `1` | 每多少个 epoch 验证一次 |
| `train.test_id` | `0` | **留一法（leave-one-person-out）的被试编号**：`0`~`14` 表示用该人做测试、其余 14 人训练（42000 条）；`-1` 表示用全部 15 人训练（45000 条，配合 `val_ratio: 0.0`） |
| `train.val_ratio` | `0.1` | 从训练集中划分出的验证集比例（必须 `< 1`） |
| `train.output_dir` | `experiments/mpiigaze/exp00` | 输出根目录，实际目录为 `<output_dir>/<test_id:02>` |
| `train.log_period` | `100` | 每多少个 step 打印一次训练日志 |
| `train.checkpoint_period` | `10` | 每多少个 epoch 保存一次 checkpoint（最后一个 epoch 一定保存） |
| `train.use_tensorboard` | `True` | 是否写 TensorBoard（False 时用 `DummyWriter` 空实现） |
| `train.train_dataloader.num_workers` | `2` | 训练 DataLoader 进程数 |
| `train.train_dataloader.drop_last` | `True` | 丢弃不完整 batch |
| `train.train_dataloader.pin_memory` | `False` | 是否锁页内存（CPU 环境自动关闭） |
| `train.val_dataloader.num_workers` / `pin_memory` | `1` / `False` | 验证 DataLoader 配置 |
| `tensorboard.train_images` | `False` | 是否把训练图片网格写入 TensorBoard |
| `tensorboard.val_images` | `False` | 是否把验证图片写入 TensorBoard（仅在 epoch 0 的第一个 batch） |
| `tensorboard.model_params` | `False` | 是否记录参数直方图 |
| `optim.adam.betas` | `(0.9, 0.999)` | Adam 系列优化器的 betas |
| `scheduler.epochs` | `40` | 总训练 epoch 数 |
| `scheduler.type` | `multistep` | `multistep`（`MultiStepLR`）或 `cosine`（`CosineAnnealingLR`） |
| `scheduler.milestones` | `[20, 30]` | multistep 的降 lr 节点 |
| `scheduler.lr_decay` | `0.1` | 每次降 lr 的 gamma |
| `scheduler.lr_min_factor` | `0.001` | cosine 的 `eta_min` |
| `test.test_id` | `0` | 评估时的测试人编号（`0`~`14`） |
| `test.checkpoint` | `''` | 要加载的 checkpoint 路径 |
| `test.output_dir` | `''` | 评估结果根目录（脚本会在其下再建一层 `<checkpoint 文件名>`） |
| `test.batch_size` | `256` | 评估 batch size |
| `test.dataloader.num_workers` / `pin_memory` | `2` / `False` | 评估 DataLoader 配置 |
| `cudnn.benchmark` | `True` | cuDNN 自动选最优算法 |
| `cudnn.deterministic` | `False` | 是否强制确定性（为 True 会变慢） |
| `face_detector.*` / `gaze_estimator.*` / `demo.*` | 见 defaults | **仅供 demo/推理使用，训练与评估都用不到** |

### 5.3 数据划分逻辑（`gaze_estimation/datasets/__init__.py`）

- `person_ids = [p00 ... p14]`；
- `test_id in 0..14` → 训练集 = 其余 14 人（42000 条），测试集 = 该人（3000 条）；
- `test_id == -1` → 训练集 = 全部 15 人（45000 条）；
- 训练集再按 `val_ratio` 用 `random_split` 切出验证集（例如 42000 → 37800 训练 / 4200 验证，`drop_last` 后为 1181 step/epoch）；
- **训练时也会执行 `assert config.test.test_id in range(15)`**，所以 train 配置里必须保留合法的 `test.test_id`（默认 0 即可）。

### 5.4 四个 train 配置对比（仓库内置）

| 字段 | [lenet_train.yaml](./configs/mpiigaze/lenet_train.yaml) | [resnet_preact_train.yaml](./configs/mpiigaze/resnet_preact_train.yaml) | [alexnet_train.yaml](./configs/mpiifacegaze/alexnet_train.yaml) | [resnet_simple_14_train.yaml](./configs/mpiifacegaze/resnet_simple_14_train.yaml) |
| --- | --- | --- | --- | --- |
| `mode` | MPIIGaze | MPIIGaze | MPIIFaceGaze | MPIIFaceGaze |
| `model.name` | lenet | resnet_preact | alexnet | resnet_simple |
| `transform.mpiifacegaze_face_size` | —（不适用） | —（不适用） | **448** | **224** |
| `dataset.dataset_dir` | datasets/MPIIGaze.h5 | datasets/MPIIGaze.h5 | datasets/MPIIFaceGaze.h5 | datasets/MPIIFaceGaze.h5 |
| `train.batch_size` | 32 | 32 | 32 | 32 |
| `train.base_lr` | 0.01 | **0.1** | 0.01 | **0.1** |
| `train.loss` | L2 | L2 | **L1** | **L1** |
| `scheduler.epochs` | 10 | **40** | **15** | **15** |
| `scheduler.milestones` | [8, 9] | [30, 35] | [10, 13] | [10, 13] |
| `train.checkpoint_period` | 10 | 10 | 5 | 5 |
| `train.test_id` | 0 | 0 | 0 | 0 |
| `train.output_dir` | experiments/mpiigaze/lenet/exp00 | experiments/mpiigaze/resnet_preact/exp00 | experiments/mpiifacegaze/alexnet/exp00 | experiments/mpiifacegaze/resnet_simple_14/exp00 |
| DataLoader `num_workers` | 4 | 4 | 0 | 0 |

**注意事项**：

- 验证集**不是最终测试集**：`validate()` 用的是训练集切出的 10%（同一个人群），真正的泛化指标要看 `evaluate.py` 在留出被试上的结果。仓库现有日志中 `resnet_simple_14` 的验证误差约 1.56°，而同一模型在 p01 上的测试误差为 **2.3170°**。
- 上表中 mpiifacegaze 两个配置的 `num_workers` 是 `0`，这是本仓库相对上游（上游均为 `4`）的唯一配置改动，用于提升 Windows 环境下的稳定性；在 Linux/服务器上可以自行改回 `4`~`8` 以加快数据加载。
- 若 `val_ratio: 0`（`*_using_all_data.yaml`），验证集为空，`validate()` 的 loss/角度误差会打印 0，这是正常现象。
- `config.device` 在 yaml 里写 `cuda` 但机器无 GPU 时会被静默改成 `cpu`；实际使用的配置会写进输出目录的 `config.yaml`（例如仓库现有实验的 `config.yaml` 里就是 `device: cpu`），排查问题时要看这份文件。

### 5.5 配置自检（可选）

**目的**：确认合并后的配置（含命令行覆盖、`device` 自动回退）与预期一致，避免训练跑了一半才发现参数写错。

**命令**：

```bash
python -c "import sys; sys.argv = ['x', '--config', 'configs/mpiifacegaze/alexnet_train.yaml', 'train.batch_size', '8']; from gaze_estimation.utils import load_config; print(load_config())"
```

**参数说明**：`sys.argv` 里的覆盖项写法与真实命令行完全一致（`key value` 成对，放在 `--config` 之后）。

**预期输出**：打印完整配置树；`train.batch_size` 应显示 `8`；无 GPU 时 `device` 显示 `cpu` 且三处 `pin_memory` 均为 `False`。

**注意事项**：自检不会创建任何目录；训练时这份最终配置会原样写入 `<output_dir>/config.yaml`，是排查配置类问题的第一手依据。

---

## 第 6 章：训练命令

**目的**：给出每个模型可直接复制粘贴的训练命令。

### 6.1 前置条件

```bash
python -m pip install -r requirements.txt
# 已按第 3、4 章准备好 datasets/MPIIGaze.h5 或 datasets/MPIIFaceGaze.h5
```

### 6.2 每个模型一条命令

```bash
# ① LeNet（MPIIGaze，1×36×60 灰度 + head pose）
python train.py --config configs/mpiigaze/lenet_train.yaml

# ② ResNet-preact-8（MPIIGaze）
python train.py --config configs/mpiigaze/resnet_preact_train.yaml

# ③ AlexNet（MPIIFaceGaze，3×448×448，必须配套 448 的 transform）
python train.py --config configs/mpiifacegaze/alexnet_train.yaml

# ④ ResNet-Simple-14（MPIIFaceGaze，3×224×224）
python train.py --config configs/mpiifacegaze/resnet_simple_14_train.yaml
```

### 6.3 常用覆盖示例

```bash
# 指定第 3 号被试作为测试人，输出到 exp01
python train.py --config configs/mpiigaze/lenet_train.yaml \
    train.test_id 3 \
    train.output_dir experiments/mpiigaze/lenet/exp01

# 用全部 15 人训练（无验证集），用于最终部署模型
python train.py --config configs/mpiigaze/resnet_preact_train_using_all_data.yaml

# 临时改小 batch 和 lr（显存不足时）
python train.py --config configs/mpiifacegaze/alexnet_train.yaml train.batch_size 8 train.base_lr 0.0025

# 指定使用的 GPU（Linux / Git Bash）
CUDA_VISIBLE_DEVICES=0 python train.py --config configs/mpiigaze/lenet_train.yaml
```

Windows PowerShell 指定 GPU（CMD/PowerShell 没有 `VAR=x cmd` 语法）：

```powershell
$env:CUDA_VISIBLE_DEVICES = "0"
python train.py --config configs/mpiigaze/lenet_train.yaml
```

### 6.4 15 折留一法（脚本）

```bash
bash scripts/run_all_mpiigaze_lenet.sh 00 0              # 参数1: exp_id 后缀, 参数2: GPU 编号
bash scripts/run_all_mpiigaze_resnet_preact.sh 00 0
bash scripts/run_all_mpiifacegaze_alexnet.sh 00 0
bash scripts/run_all_mpiifacegaze_resnet_simple_14.sh 00 0
```

脚本内部对 `test_id = 0..14` 循环，训练后立刻评估，目录结构为
`experiments/<dataset>/<arch>/exp<exp_id>/<test_id 两位>/`，评估结果在 `.../<test_id>/eval/<checkpoint 名>/`。
脚本使用的 checkpoint 名与各配置的 epoch 数对应：lenet `checkpoint_0010.pth`、resnet_preact `checkpoint_0040.pth`、mpiifacegaze `checkpoint_0015.pth`。

**参数说明**：

| 参数 | 含义 |
| --- | --- |
| `--config` | 训练/评估配置，必须放在覆盖项之前 |
| `train.test_id` | 留出被试编号：`0`~`14`；`-1` 表示使用全部数据 |
| `train.output_dir` | 输出根目录（可复用，不同 `test_id` 会落到不同子目录） |
| `train.batch_size` | 视显存调整 |
| `train.base_lr` | 视 batch size 调整（线性缩放：batch 减半 → lr 减半） |
| `CUDA_VISIBLE_DEVICES` | 环境变量，选择物理 GPU（本仓库无多卡/分布式训练支持） |

**预期输出**：见 [第 7 章](#第-7-章训练输出) 的目录结构与日志样例。

**注意事项**：

- `train.output_dir/<test_id>` **已存在会直接 `RuntimeError`**，重复实验请换 `output_dir`/`exp_id` 或先删除旧目录。
- 覆盖项必须成对且写在 `--config` 之后；写错（例如只给键不给值）会在 `merge_from_list` 处报 `ValueError: Length of input list must be even`。
- 训练**不支持多卡/DDP**，也没有 AMP 混合精度、没有 early stopping；学习率调度只有 `multistep` 和 `cosine`。
- Windows 上若 `num_workers: 0`（mpiifacegaze 配置默认如此）训练慢但稳定；MPIIGaze 配置是 4，遇到 `DataLoader` 进程卡死可临时改成 0 验证是否为多进程问题。

---

## 第 7 章：训练输出

**目的**：知道训练产物的位置、内容和用途。

### 7.1 目录结构

以 `python train.py --config configs/mpiifacegaze/resnet_simple_14_train.yaml`（`test_id: 0`）为例：

```
experiments/mpiifacegaze/resnet_simple_14/exp00/
└── 00/                     # train.test_id 的两位编号；test_id=-1 时为 all/
    ├── config.yaml                        # 实际生效的完整配置（含 device 自动回退结果）
    ├── log.txt                            # 带 ANSI 颜色的日志
    ├── log_plain.txt                      # 纯文本日志（推荐 grep/分享）
    ├── events.out.tfevents.<时间戳>.<主机名>  # TensorBoard 事件文件
    ├── checkpoint_0005.pth                # 每 checkpoint_period 个 epoch 一个
    ├── checkpoint_0010.pth
    ├── checkpoint_0015.pth                # 最后一个 epoch 必定保存
    └── last_checkpoint                    # fvcore 记录的"最近一次 checkpoint"信息
```

文件体积参考（仓库内实测）：

| 模型 | 单个 checkpoint 大小 |
| --- | --- |
| ResNet-Simple-14 | **22.1 MB** |
| AlexNet | **约 1500 MB**（`fc1` 有 1.77 亿参数，且优化器状态一并保存） |

### 7.2 checkpoint 内容

由 `fvcore.common.checkpoint.Checkpointer` 保存，包括：

```python
{
  'model': <model.state_dict()>,        # 评估/推理时加载的就是这一项
  'optimizer': <optimizer.state_dict()>,
  'scheduler': <scheduler.state_dict()>,
  'epoch': <int>,
  'config': <dict>,                     # 训练时的配置快照
}
```

### 7.3 日志样例（仓库现有真实运行，CPU 训练）

```
[2025-02-15 13:25:27] __main__ INFO: Epoch 0 loss 0.1626 angle error 14.46     # val_first
[2025-02-15 13:25:27] __main__ INFO: Train 1
[2025-02-15 13:25:30] __main__ INFO: Epoch 1 Step 0/1181 lr 0.100000 loss 0.1652 (0.1652) angle error 14.62 (14.62)
...
[2025-02-16 02:50:30] __main__ INFO: Epoch 14 loss 0.0175 angle error 1.57    # 验证集
[2025-02-16 03:44:17] __main__ INFO: Elapsed 3226.28                          # 一个 epoch 的训练耗时（秒）
[2025-02-16 03:47:49] fvcore.common.checkpoint INFO: Saving checkpoint to .../checkpoint_0015.pth
```

### 7.4 查看训练曲线

```bash
tensorboard --logdir experiments/mpiifacegaze/resnet_simple_14/exp00
```

记录的标量：`Train/Loss`、`Train/lr`、`Train/AngleError`、`Train/Time`、`Val/Loss`、`Val/AngleError`、`Val/Time`。

### 7.5 把训练结果接入 demo

把 checkpoint 放到 demo 配置引用的位置（[config.py](./config.py) 与 [configs/demo_mpiifacegaze_resnet_simple_14.yaml](./configs/demo_mpiifacegaze_resnet_simple_14.yaml)）：

```powershell
New-Item -ItemType Directory -Force -Path data\models\mpiifacegaze\resnet_simple_14 | Out-Null
Copy-Item experiments\mpiifacegaze\resnet_simple_14\exp00\00\checkpoint_0015.pth data\models\mpiifacegaze\resnet_simple_14\
python main.py
```

**注意事项**：

- 输出目录、`*.pth`、`logs/` 等均被 `.gitignore` 排除，训练产物不会进版本库。
- `log.txt` 含 ANSI 转义码，做统计请用 `log_plain.txt`。
- 推送到远端/分享时注意不要包含可识别驾驶员身份的数据。

---

## 第 8 章：评估与 ONNX 导出

**目的**：用留出被试衡量泛化误差，并把模型导出为 ONNX。

### 8.1 评估

```bash
python evaluate.py --config configs/mpiigaze/lenet_eval.yaml

# 或显式指定 checkpoint / 被试 / 输出目录
python evaluate.py --config configs/mpiifacegaze/resnet_simple_14_eval.yaml \
    test.test_id 1 \
    test.checkpoint experiments/mpiifacegaze/resnet_simple_14/exp00/00/checkpoint_0015.pth \
    test.output_dir experiments/mpiifacegaze/resnet_simple_14/exp00/00/eval
```

**参数说明**：

- `test.test_id`：被评估的被试（`0`~`14`），必须是**训练时留出的那一位**，否则指标无意义；
- `test.checkpoint`：checkpoint 路径，内部读取 `checkpoint['model']`；
- `test.output_dir`：结果根目录；脚本会再拼一层 checkpoint 文件名（`Path(...).stem`）；
- 评估配置里的 `transform.mpiifacegaze_face_size` 必须与训练/模型一致（resnet_simple→224，alexnet→448）。

**预期输出**：

```
The mean angle error (deg): 2.32
```

产物：

```
<test.output_dir>/<checkpoint 名>/
├── config.yaml
├── predictions.npy     # (N, 2) 预测的 [pitch, yaw]（弧度）
├── gts.npy             # (N, 2) 真值
└── error.txt           # 平均角度误差（度），单行文本
```

仓库已有结果示例：`experiments/mpiifacegaze/resnet_simple_14/exp00/00/eval/checkpoint_0015/error.txt` = **2.3170552253723145**（p01 被试）。
上游论文实现的参考指标（GTX 1080 Ti，全部被试平均角度误差）：

| 模型 | 平均测试角度误差 | 训练耗时参考 |
| --- | --- | --- |
| LeNet | 6.52° | 3.5 s/epoch |
| ResNet-preact-8 | 5.73° | 7 s/epoch |
| AlexNet | 5.06° | 135 s/epoch |
| ResNet-Simple-14 | 4.83° | 62 s/epoch |

**注意事项**：

- 评估会重新 `create_model(config)`，如果配置里带了 `pretrained`，**仍会尝试联网下载预训练权重**再被 `load_state_dict` 覆盖；离线环境需要缓存（见第 10 章）。
- `test.output_dir` 的父级不存在也能创建（脚本用了 `parents=True`），但同一次评估重复运行会覆盖同名文件（`exist_ok=True`）。
- 评估不做多折平均，需要 15 折平均就自己循环（即 `scripts/run_all_*.sh` 的做法）。

### 8.2 导出 ONNX

```bash
python convert_to_onnx.py --config configs/mpiigaze/lenet_eval.yaml \
    --weight experiments/mpiigaze/lenet/exp00/00/checkpoint_0010.pth \
    -o models/lenet.onnx
```

**参数说明**：

| 参数 | 必填 | 说明 |
| --- | --- | --- |
| `--config` | ✅ | 结构配置（只用来建模型，不读 checkpoint 路径） |
| `--weight` | ❌ | checkpoint 路径；不给则导出**随机初始化**的模型结构 |
| `-o` / `--output-path` | ✅ | 输出 `.onnx` 路径 |

**说明与预期输出**：

- 脚本用虚拟输入 `torch.zeros(...)` 做 tracing：`MPIIGaze → ((1,1,36,60), (1,2))` 两个输入；`MPIIFaceGaze → (1,3,224,224)` 单个输入；
- 因此导出的 ONNX 输入是**固定 batch=1**；
- 成功时静默生成 `*.onnx` 文件，无标准输出。

**注意事项**：

- `convert_to_onnx.py` 使用 `get_default_config().merge_from_file(...)`，**不支持** `train.xxx value` 形式的命令行覆盖（与 `train.py` 不同）；要改配置就改 yaml 或新写一个 yaml。
- **AlexNet 无法用本脚本直接导出**：脚本对 `MPIIFaceGaze` 硬编码了 `(1, 3, 224, 224)`，而 AlexNet 的 `fc1 = Linear(256*13², 4096)` 需要 448×448 输入，导出时会因维度不匹配报错。做法：把脚本里的 dummy 输入改成 `(1, 3, 448, 448)` 后再导出（记得 `configs/mpiifacegaze/alexnet_eval.yaml` 也要用 448）。
- 需要安装 `onnx`（`requirements.txt` 已包含）；导出 MPIIGaze 模型时 ONNX 会有两个输入，接入推理端时别只喂图像。
- 导出的模型**不做融合/量化/图优化**，如需简化可用 `onnx-simplifier`、如需 NMS 类后处理可参考 [utils/add_nms.py](./utils/add_nms.py)（那是给 YOLO 用的）。
- 注意 `models/` 目录下已有的 `*.onnx`（`retinaface_640x640_opt.onnx`、`mnv3_gaze32_split_opt.onnx` 等）是 demo 用的第三方模型，**不是**本脚本产物。

---

## 第 9 章：硬件需求与耗时

**目的**：预估训练成本，判断是否需要降配。

### 9.1 硬件建议

| 项目 | 建议 |
| --- | --- |
| GPU | 必须（CUDA）。**单卡**即可，代码没有 DDP/多卡支持；`CUDA_VISIBLE_DEVICES` 选择卡 |
| 显存 | 一般建议 ≥ 8 GB；**AlexNet（448×448, batch 32）建议 ≥ 12 GB**（经验值，随驱动/框架版本浮动） |
| 内存 | ≥ 16 GB（MPIIGaze 全量载入内存约 100 MB，压力主要在 DataLoader 与 HDF5 缓存） |
| 磁盘 | 数据集约 7 GB + 每个 checkpoint（ResNet-14 ≈ 22 MB / AlexNet ≈ 1.5 GB） |

CPU 也能跑通（`load_config()` 会自动降级到 `device: cpu`），但只适合小模型和调试。

### 9.2 实测/参考耗时

**CPU（仓库现有真实日志，ResNet-Simple-14，224×224，batch 32，42000 条训练数据 = 1181 step/epoch）**：

| 阶段 | 耗时 |
| --- | --- |
| 训练 1 epoch | ≈ 3200 s（约 53 分钟） |
| 验证 1 次 | ≈ 213 s（约 3.5 分钟） |
| 15 epochs 合计 | ≈ 14.5 小时（2025-02-15 13:20 → 2025-02-16 03:47） |

**GPU（GTX 1080 Ti 参考，来自上游实现）**：

| 模型 | 每 epoch | 单折（epoch 数 × 每 epoch） | 15 折合计（约） |
| --- | --- | --- | --- |
| LeNet | 3.5 s | 10 × 3.5 s ≈ 35 s | ≈ 9 分钟 |
| ResNet-preact-8 | 7 s | 40 × 7 s ≈ 4.7 分钟 | ≈ 70 分钟 |
| ResNet-Simple-14 | 62 s | 15 × 62 s ≈ 15.5 分钟 | ≈ 3.9 小时 |
| AlexNet | 135 s | 15 × 135 s ≈ 34 分钟 | ≈ 8.4 小时 |

**注意事项**：

- AlexNet 的瓶颈是 `fc1`（43264→4096，1.77 亿参数），显存和磁盘占用都明显高于其他三个模型；实际项目中 ResNet-Simple-14 是性价比最优的默认选择。
- MPIIFaceGaze 的 Dataset 每条样本都重新打开 HDF5，I/O 是主要瓶颈之一；把 `train_dataloader.num_workers` 从 0 提到 4~8 通常有收益（仓库配置为 0，主要是为了 Windows 稳定性与 CPU 环境）。
- 想要更快的迭代：先跑 `lenet_train.yaml`（MPIIGaze，10 epoch）验证整条链路，再跑大模型。

### 9.3 环境自检

**目的**：确认硬件与安装的 PyTorch 匹配，避免「以为在用 GPU，其实在跑 CPU」。

**命令**：

```bash
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())"
```

**参数说明**：第一条查看驱动版本、当前显存占用和占用进程；第二条确认 torch 能否识别 CUDA。

**预期输出**：

```
1.13.1 11.7 True 1
```

`torch.cuda.is_available()` 为 `True` 时才会使用 GPU；为 `False` 时 `load_config()` 会自动把 `device` 改成 `cpu`（速度差异见 9.2）。

**注意事项**：`nvidia-smi` 里的 `CUDA Version` 是驱动支持的上限，不代表本机 torch 的 CUDA 运行时版本；多卡机器用 `CUDA_VISIBLE_DEVICES` 指定单卡，本仓库不支持多卡并行。

---

## 第 10 章：常见报错与解决

**目的**：快速定位并修复训练/评估过程中最常见的问题。

### 10.1 报错速查表

| # | 报错 / 现象 | 原因 | 解决 |
| --- | --- | --- | --- |
| 1 | `RuntimeError: Output directory 'experiments/.../00' already exists.` | `create_train_output_dir()` 要求输出目录唯一 | 换 `train.output_dir`（如 `exp01`）或换 `train.test_id`，或删除旧目录 |
| 2 | `AssertionError`（`assert dataset_dir.exists()`） | `dataset.dataset_dir` 指向的 `.h5` 不存在或路径写错 | 先跑第 4 章预处理；检查 yaml 中路径（相对路径以项目根目录为基准） |
| 3 | `ValueError: datasets/MPIIGaze.h5 already exists.` | 预处理脚本拒绝覆盖输出 | 删除旧 `.h5` 后重跑预处理 |
| 4 | `AssertionError`（`assert len(train_dataset) == 42000` / `45000`） | 数据集不完整（预处理被中断、被试文件缺失） | 重新预处理，确认 `p00`~`p14` 全部生成、每人 3000 条 |
| 5 | `FileNotFoundError` / 预处理报 `Evaluation Subset` 相关错误 | `MPIIGaze.tar.gz` 未完整下载/解压 | 确认 `datasets/MPIIGaze/Evaluation Subset/sample list for eye image/p00.txt`~`p14.txt` 存在（该目录随压缩包提供）；缺失则重新下载解压 |
| 6 | `ModuleNotFoundError: No module named 'torch' / 'h5py' / 'tensorboardX' / 'fvcore'` | 依赖未安装 | `python -m pip install -r requirements.txt`；`torch/torchvision` 按 CUDA 版本从 PyTorch 官网选择安装命令 |
| 7 | Windows 安装 `dlib` 失败（仅 demo 需要） | 无预编译 wheel、缺 CMake/VS 构建工具 | 使用对应 Python 版本的 `dlib` wheel，或安装 CMake + Visual C++ Build Tools 后再 pip 安装 |
| 8 | `URLError` / `ConnectionError`，堆栈在 `load_state_dict_from_url` 或 `alexnet(pretrained=True)` | 训练时会联网下载预训练权重（resnet18 / alexnet） | 预先把权重放到 torch hub 缓存目录（Windows: `%USERPROFILE%\.cache\torch\hub\checkpoints\`），或把 `model.backbone.pretrained` 设为 `''`（会损失精度） |
| 9 | `RuntimeError: CUDA out of memory` | batch 太大（AlexNet 448 尤其明显） | 降低 `train.batch_size`（如 32 → 16 → 8），并按比例降低 `train.base_lr`；确认没有其他进程占用显存 |
| 10 | `RuntimeError: mat1 and mat2 shapes cannot be multiplied`（`fc` 相关） | `transform.mpiifacegaze_face_size` 与模型不匹配（alexnet 必须 448，resnet_simple 必须 224） | 改正配置或改用对应模型 |
| 11 | `AssertionError: Override list has odd length ... it must be a list of pairs` | 命令行覆盖项没成对出现 | 用 `key value` 成对书写，且放在 `--config` 之后 |
| 12 | 命令行覆盖项不生效 | `--config` 写在覆盖项之后，或被当成位置参数吃掉 | 把 `--config xxx.yaml` 放最前 |
| 13 | 训练极慢、GPU 利用率低 | DataLoader `num_workers: 0`（mpiifacegaze 配置默认值）或 HDF5 逐样本读取 | 提高 `train_dataloader.num_workers`（4~8）；确认 `device` 确实是 `cuda`（看输出目录 `config.yaml` 里的 `device`） |
| 14 | `device` 明明是 cuda 却在用 CPU | `torch.cuda.is_available()` 为 False（驱动/CUDA 版本与 torch 不匹配） | 用 `python -c "import torch;print(torch.cuda.is_available(), torch.version.cuda)"` 排查，重装匹配的 torch |
| 15 | `OSError: Unable to open file (truncated file: eof = ...)` | HDF5 文件生成中断/损坏 | 删除 `.h5` 重新预处理 |
| 16 | `UnpicklingError: Weights only load failed`（加载 checkpoint 时） | torch ≥ 2.6 的 `torch.load` 默认 `weights_only=True` | 在 `evaluate.py` / `gaze_estimation/gaze_estimator/gaze_estimator.py` 中显式传 `weights_only=False`，或使用 `torch<2.6` |
| 17 | 验证集 loss/角度误差恒为 0 | 使用了 `val_ratio: 0.0` 的 `*_using_all_data.yaml` | 属预期行为；要看验证指标就用 `val_ratio > 0` 的配置 |
| 18 | `bash: python: command not found` / 无法执行 `scripts/*.sh` | 脚本是 bash 脚本，Windows 默认没有 bash | 用 Git Bash / WSL，或按第 6 章手写 PowerShell 等价命令 |
| 19 | 训练中断后想续训 | **本仓库没有 resume 逻辑**（`train.py` 不读 `last_checkpoint`） | 只能从头训练；如需续训需自行实现加载 checkpoint 的逻辑 |
| 20 | 图表/日志里的角度误差是弧度还是度？ | `compute_angle_error()` 内部 `* 180 / np.pi`，日志和 `error.txt` 都是**度** | 预测/真值本身是弧度（`predictions.npy` / `gts.npy`） |

**注意事项**：

- 排查问题的第一步永远是看输出目录里的 `config.yaml`（实际生效配置）和 `log_plain.txt`（完整日志）。
- 报错发生在 `create_dataset()` 之后的阶段，通常说明数据侧已通过，问题在模型/显存/算子。

### 10.2 排查命令清单

**命令**：

```powershell
# ① 实际生效的配置（含 device 回退、命令行覆盖结果）
Get-Content experiments\mpiifacegaze\resnet_simple_14\exp00\00\config.yaml

# ② 训练日志尾部（Windows 下 log.txt 带 ANSI 颜色，看 log_plain.txt 更清晰）
Get-Content experiments\mpiifacegaze\resnet_simple_14\exp00\00\log_plain.txt -Tail 50

# ③ GPU 状态与显存占用
nvidia-smi

# ④ torch / CUDA 是否可用
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# ⑤ 数据集是否完整（应为 15 个被试目录）
python -c "import h5py; f=h5py.File('datasets/MPIIFaceGaze.h5','r'); print(sorted(f.keys()))"
```

**参数说明**：① 判断配置是否被正确覆盖；② 判断中断位置与最后的 loss/误差；③④ 判断是算力问题还是代码问题；⑤ 判断数据是否完整。

**预期输出**：① 完整的 yaml 配置树；② 最后若干行日志（如 `Epoch 15 Step 1100/1181 lr 0.001000 ...`）；③ GPU 使用率/显存表；④ 例如 `1.13.1 True`；⑤ `['p00', 'p01', ..., 'p14']`（MPIIFaceGaze 的每个 `pXX` 下应有 `image/pose/gaze` 三组）。

**注意事项**：如果 ⑤ 输出的被试数量少于 15，说明预处理中断，需要删除 `.h5` 重新预处理（对应第 4 条报错）。

---

## 第 11 章：命令速查表

**目的**：把本教程涉及的可执行命令汇总到一处，方便复制粘贴（详细说明见对应章节）。

**命令**：

```bash
# ---------- 环境 ----------
python -m pip install -r requirements.txt

# ---------- 数据 ----------
bash scripts/download_mpiigaze_dataset.sh
python tools/preprocess_mpiigaze.py --dataset datasets/MPIIGaze -o datasets/

bash scripts/download_mpiifacegaze_dataset.sh
python tools/preprocess_mpiifacegaze.py --dataset datasets/MPIIFaceGaze_normalized -o datasets/

# ---------- 训练（4 个模型各一条） ----------
python train.py --config configs/mpiigaze/lenet_train.yaml
python train.py --config configs/mpiigaze/resnet_preact_train.yaml
python train.py --config configs/mpiifacegaze/alexnet_train.yaml
python train.py --config configs/mpiifacegaze/resnet_simple_14_train.yaml

# ---------- 训练（全部数据，无验证集） ----------
python train.py --config configs/mpiigaze/lenet_train_using_all_data.yaml
python train.py --config configs/mpiigaze/resnet_preact_train_using_all_data.yaml

# ---------- 15 折留一法 ----------
bash scripts/run_all_mpiigaze_lenet.sh 00 0
bash scripts/run_all_mpiigaze_resnet_preact.sh 00 0
bash scripts/run_all_mpiifacegaze_alexnet.sh 00 0
bash scripts/run_all_mpiifacegaze_resnet_simple_14.sh 00 0

# ---------- 评估 ----------
python evaluate.py --config configs/mpiigaze/lenet_eval.yaml
python evaluate.py --config configs/mpiifacegaze/resnet_simple_14_eval.yaml \
    test.test_id 1 test.checkpoint <ckpt.pth> test.output_dir <out_dir>

# ---------- ONNX ----------
python convert_to_onnx.py --config configs/mpiigaze/lenet_eval.yaml \
    --weight <ckpt.pth> -o models/lenet.onnx

# ---------- 曲线 ----------
tensorboard --logdir experiments/mpiifacegaze/resnet_simple_14/exp00
```

---

## 第 12 章：行为检测（YOLO）部分的说明

**目的**：说明“行为检测/打电话”这类 YOLO 模型在本仓库中的现状，避免误用本教程的训练入口。

**事实（基于源码）**：

- 根目录 [train.py](./train.py) **只负责视线估计**（MPIIGaze / MPIIFaceGaze），不训练 YOLO。
- 仓库里的 YOLO 相关代码是**推理侧**的：`models/`（`yolo.py`、`common.py`、`experimental.py`）与 `utils/`（`datasets.py`、`loss.py`、`general.py` 等，源自 YOLOR/YOLOv5 系代码，见 [THIRD_PARTY_NOTICES.md](./THIRD_PARTY_NOTICES.md)）。
- 行为/手机检测的调用点在 [my_detector.py](./my_detector.py)：

  ```python
  from models.experimental import attempt_load
  self.model = attempt_load(self.config.weights, map_location=self.device)
  ```

  权重路径来自 [config.py](./config.py) 的 `AppConfig.weights = 'outputs/exp5/weights/best.pt'`（连同 `img_size=640`、`conf_thres=0.25`、`iou_thres=0.45`、`classes=None`）。
- [detection/behavior_detection.py](./detection/behavior_detection.py) 与 [detection/phone_use_detection.py](./detection/phone_use_detection.py) 目前是**占位实现**（返回写死的 `[{'xyxy': [...], 'label': 'Distracted Driving'}]` / `'Phone'`），并不是真实模型推理。
- 仓库的 `.gitignore` 排除了 `/outputs/`，`outputs/exp*/weights/*.pt` 中的权重是本地训练/外部获取的产物，**源码仓库不包含训练脚本**。

**如果要训练自己的行为检测模型**：

1. 使用本仓库 `models/` + `utils/` 同源的 YOLO 训练工程（或直接使用上游 YOLOv5/YOLOR 工程），数据采用 YOLO 的 `images/ + labels/*.txt` + `data.yaml`（类别名）格式；
2. 训练完成后把权重放到 `outputs/exp5/weights/best.pt`，或修改 [config.py](./config.py) 中的 `AppConfig.weights` 指向新权重；
3. 启动 `python main.py` 验证检测类别与阈值（`conf_thres` / `iou_thres` / `classes`）是否符合预期。

**验证当前使用的权重**：

```powershell
python -c "from config import AppConfig; print(AppConfig().weights)"
```

**预期输出**：`outputs/exp5/weights/best.pt`（若你已改了配置，则输出新的路径）。

**注意事项**：`models/`、`utils/` 中的 YOLOR 衍生代码受 GPL-3.0 约束，训练与再分发前请阅读 [LICENSES/GPL-3.0-YOLOR.txt](./LICENSES/GPL-3.0-YOLOR.txt)。行为检测模型的性能与训练数据强相关，请确保训练数据来源合法、不含未经授权的驾驶员影像。

---

## 参考资料

| 内容 | 来源 |
| --- | --- |
| 数据集准备 + 训练/评估代码（本教程第 3~8 章的依据） | [Zhuqiben/pytorch_mpiigaze](https://github.com/Zhuqiben/pytorch_mpiigaze) |
| 上述仓库的上游原始实现（MIT 许可） | [hysts/pytorch_mpiigaze](https://github.com/hysts/pytorch_mpiigaze)、[hysts/pytorch_mpiigaze_demo](https://github.com/hysts/pytorch_mpiigaze_demo) |
| MPIIGaze 数据集 | http://datasets.d2.mpi-inf.mpg.de/MPIIGaze/MPIIGaze.tar.gz |
| MPIIFaceGaze 数据集（normalized） | http://datasets.d2.mpi-inf.mpg.de/MPIIGaze/MPIIFaceGaze_normalized.zip |
| 数据集主页（含转换脚本与 3D↔2D 角度换算说明） | https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/gaze-based-human-computer-interaction/appearance-based-gaze-estimation-in-the-wild |

引用：

- Zhang, Xucong, Yusuke Sugano, Mario Fritz, and Andreas Bulling. "Appearance-based Gaze Estimation in the Wild." CVPR 2015.
- Zhang, Xucong, Yusuke Sugano, Mario Fritz, and Andreas Bulling. "It's Written All Over Your Face: Full-Face Appearance-Based Gaze Estimation." CVPRW 2017.
- Zhang, Xucong, Yusuke Sugano, Mario Fritz, and Andreas Bulling. "MPIIGaze: Real-World Dataset and Deep Appearance-Based Gaze Estimation." TPAMI 2019.
