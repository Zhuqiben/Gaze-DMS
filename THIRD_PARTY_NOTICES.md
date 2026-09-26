# Third-party notices

本项目**自有代码**以 **GNU GPL-3.0** 发布，全文见根目录 [`LICENSE`](LICENSE)。
此外，仓库中包含若干第三方组件，各自遵循其原始许可证；再分发或修改时必须一并遵守。

## 组件一览

| 组件 | 许可证 | 涉及内容 | 许可证原文 |
| --- | --- | --- | --- |
| [hysts/pytorch_mpiigaze](https://github.com/hysts/pytorch_mpiigaze) | MIT | `gaze_estimation/`、`train.py`、`evaluate.py`、`convert_to_onnx.py`、`configs/`、`scripts/`、`tools/`、`figures/` | [`LICENSES/MIT-pytorch_mpiigaze.txt`](LICENSES/MIT-pytorch_mpiigaze.txt) |
| [WongKinYiu/YOLOR](https://github.com/WongKinYiu/yolor) | GPL-3.0 | `models/common.py`、`models/experimental.py`、`models/yolo.py`、`utils/*.py` | [`LICENSES/GPL-3.0-YOLOR.txt`](LICENSES/GPL-3.0-YOLOR.txt) |
| [emilianavt/OpenSeeFace](https://github.com/emilianavt/OpenSeeFace) | BSD-2-Clause | `tracker.py`、`retinaface.py`、`remedian.py`、`models/*.onnx`、`models/priorbox_640x640.json` | [`LICENSES/BSD-2-Clause-OpenSeeFace.txt`](LICENSES/BSD-2-Clause-OpenSeeFace.txt) |
| [biubug6/Pytorch_Retinaface](https://github.com/biubug6/Pytorch_Retinaface)（经 OpenSeeFace 分发） | MIT | `retinaface.py`、`models/retinaface_640x640_opt.onnx`、`models/priorbox_640x640.json` | [`LICENSES/MIT-Pytorch_Retinaface.txt`](LICENSES/MIT-Pytorch_Retinaface.txt) |
| Tim Menzies, Remedian | MIT | `remedian.py` | [`LICENSES/MIT-Remedian.txt`](LICENSES/MIT-Remedian.txt) |

> OpenSeeFace 要求分发时一并提供其 `Licenses/` 目录中的第三方声明，上表后三项即为对应文件。
> 其 ONNX 推理依赖 [onnxruntime](https://github.com/microsoft/onnxruntime)（MIT）。

## 权重与模型文件

| 文件 / 内容 | 来源 | 说明 |
| --- | --- | --- |
| `models/*.onnx`、`models/priorbox_640x640.json` | OpenSeeFace 模型包（BSD-2-Clause，含 MIT 的 Pytorch_Retinaface 组件） | 可随仓库再分发，保留声明即可 |
| `outputs/exp5/weights/best.pt` | 本项目自行训练（2 类检测：`face` / `phone`） | 与源码同为 GPL-3.0 |
| `data/models/mpiifacegaze/resnet_simple_14/checkpoint_0015.pth` | 本仓库 `train.py` 在 MPIIFaceGaze 上训练得到 | 与源码同为 GPL-3.0；因训练数据仅限研究用途，请仅用于研究/演示 |
| `data/dlib/shape_predictor_68_face_landmarks.dat` | [dlib.net](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)，训练自 iBUG 300-W 数据集 | **不随仓库分发**，请用 [`scripts/download_dlib_model.sh`](scripts/download_dlib_model.sh) 获取；该模型限非商业研究用途 |
| MPIIGaze / MPIIFaceGaze 数据集 | 数据集官方 | **不随仓库分发**，获取与使用须遵守其许可（仅限非商业研究） |

## 其它

`EAR.py` 与 `MAR.py` 中的 Eye/Mouth Aspect Ratio 为本项目自行实现的标准公式。
本项目为研究与演示用途，不是经认证的驾驶安全产品。
