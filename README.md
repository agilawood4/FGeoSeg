# FGeoSeg: Fisheye Sky Segmentation Pipeline

FGeoSeg 是一个基于 Teacher-Student 联合架构的鱼眼天空分割管道。该项目利用教师模型（DarSwin-UNet）和学生模型（SegFormer-B0）结合 RDC（Receptive Field Distillation Connection）模块，实现高效的天空分割。最终模型可以导出为 ONNX 格式，便于部署。

## 特性

- **教师-学生架构**：教师模型提供高质量伪标签，学生模型通过知识蒸馏学习。
- **RDC 模块**：增强学生模型的感受野，提升分割精度。
- **ONNX 导出**：支持模型导出为 ONNX 格式，便于边缘设备部署。
- **鱼眼镜头支持**：专为鱼眼图像设计，处理畸变天空分割。

## 安装

1. 创建 Conda 环境：
   ```bash
   conda create -n fgeoseg python=3.10 -y
   conda activate fgeoseg
   ```

2. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

3. （可选）安装 DCNv2 以支持 RDC 模块，否则将回退到 Conv2d：
   ```bash
   pip install -r cuda_requirements.txt
   ```

## 数据集准备

1. 准备图像和标注文件。图像为 JPG 格式，标注为 JSON 格式，包含天空多边形。

2. 创建训练和验证列表文件：
   - `data/train_list.txt`
   - `data/val_list.txt`

   格式示例：
   ```
   /abs/path/img_0001.jpg\t/abs/path/img_0001.json
   /abs/path/img_0002.jpg\t/abs/path/img_0002.json
   ```

   参考 `data/fisheye_json_dataset.py` 查看 JSON 格式要求。

## 训练指南

### 步骤 1: 训练教师模型

教师模型使用 DarSwin-UNet 架构。

```bash
python scripts/train_teacher.py
```

训练完成后，模型权重保存在 `checkpoints/teacher_darswin_unet_best.pth`。

### 步骤 2: 生成伪标签

使用教师模型生成伪标签，用于学生模型训练。

```bash
python scripts/gen_pseudo_labels.py
```

伪标签保存在 `pseudo_labels/` 目录下。

### 步骤 3: 训练学生模型（CPS）

学生模型使用 SegFormer-B0，结合 CPS（Cross Pseudo Supervision）。

```bash
python scripts/train_cps_student.py
```

训练完成后，模型权重保存在 `checkpoints/student_segformer_b0_best.pth`。

## 导出 ONNX

训练完成后，导出学生模型为 ONNX 格式：

```bash
python scripts/export_onnx.py
```

导出的模型保存在 `checkpoints/segformer_b0_student.onnx`。

## 推理

使用导出的 ONNX 模型进行推理。参考 `tools/` 目录下的脚本。

## 项目结构

```
├── checkpoints/          # 模型权重和 ONNX 文件
├── configs/              # 配置文件
├── data/                 # 数据集和列表文件
├── log/                  # 日志
├── logs/                 # 额外日志
├── models/               # 模型定义
├── pseudo_labels/        # 生成的伪标签
├── runs_debug/           # 调试运行
├── runs_vis/             # 可视化运行
├── scripts/              # 训练和导出脚本
├── tools/                # 工具脚本
└── utils/                # 工具函数
```

## 配置

主要配置在 `configs/config.yaml` 中，包括模型参数、训练设置等。

## 贡献

欢迎提交 Issue 和 Pull Request！

## 联系

如有问题，请联系项目维护者。
