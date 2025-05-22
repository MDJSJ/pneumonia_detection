# 肺炎图像识别项目说明文档

## 1. 项目简介

本项目利用迁移学习技术，基于 Inception 预训练模型，实现对肺炎 X 光图像的自动分类识别。通过冻结预训练模型的大部分参数，仅重新训练最后的全连接层，实现了高效、准确的肺炎图像分类系统。

## 2. 系统要求

- Python 3.6+
- PyTorch
- PIL
- NumPy
- 推荐使用 GPU 加速训练过程
- 约 2GB 存储空间（用于保存模型和数据）

## 3. 安装方法

```bash
# 克隆仓库
git clone https://github.com/yourusername/pneumonia-detection.git
cd pneumonia-detection

# 安装依赖
pip install torch torchvision pillow numpy
```

## 4. 数据准备

本项目使用的数据集需要按照以下结构组织：

```
chest_xray_img/
  ├── train/
  │   ├── NORMAL/
  │   │   ├── image1.jpeg
  │   │   ├── image2.jpeg
  │   │   └── ...
  │   └── PNEUMONIA/
  │       ├── image1.jpeg
  │       ├── image2.jpeg
  │       └── ...
  ├── val/
  │   ├── NORMAL/
  │   │   └── ...
  │   └── PNEUMONIA/
  │       └── ...
  └── test/
      ├── NORMAL/
      │   └── ...
      └── PNEUMONIA/
          └── ...
```

每个类别文件夹中的图像将被用于训练、验证和测试。

## 5. 使用方法

### 5.1 模型训练

```bash
python pneumonia_detection_pytorch.py
```

主要参数说明：

- 无需额外参数，代码中已设置默认值

### 5.2 参数调优

可调整的主要参数包括：

- `num_epochs`: 训练轮数，默认 10
- `batch_size`: 批次大小，默认 32
- `learning_rate`: 学习率，默认 0.001

## 6. 项目结构

```
pneumonia_detection/
  ├── pneumonia_detection_pytorch.py  # PyTorch实现的主训练脚本 (基于Inception迁移学习)
  ├── evaluate_model.py             # 模型评估脚本
  ├── occlusion.py                  # 遮挡实验脚本
  ├── retrain.py                    # 模型重新训练脚本
  ├── chest_xray_img/                 # 图像数据集 (用于主模型和PneumoniaNet)
  │   ├── train/
  │   ├── val/
  │   └── test/
  ├── net/                            # 包含主模型训练脚本(train.py)和工具函数(utils.py)
  ├── pneumonia_net_implementation/   # 基于 "PneumoniaNet" 论文的独立模型实现 (三分类)
  │   ├── main.py                     # PneumoniaNet 模型的主脚本 (或 pneumonia_model_from_paper.py)
  │   ├── README.md                   # PneumoniaNet 实现的详细说明
  │   └── PneumoniaNet_Paper.pdf      # 相关论文
  ├── runs/                           # TensorBoard 或其他运行日志
  │   └── pneumonia_detection_experiment/
  ├── .gitignore                      # Git 忽略文件
  └── README.md                       # 项目说明文档 (本文档)
```

## 7. 模型描述

本项目采用迁移学习技术，基于 PyTorch 和预训练的 Inception 模型进行肺炎图像分类。主要步骤如下：

1.  **预训练模型**：加载在 ImageNet 数据集上预训练的 Inception 模型。
2.  **特征提取**：冻结预训练模型的大部分卷积层参数，使其作为固定的特征提取器。
3.  **分类器调整**：替换或重新初始化 Inception 模型的最后全连接层（分类器），以适应本项目中的二分类任务（NORMAL 和 PNEUMONIA）。新的全连接层将输出 2 个类别。
4.  **模型微调**：仅训练调整后的全连接层参数，或者对预训练模型的少数顶层进行微调（fine-tuning）。

输入图像为 3 通道 RGB 图像（X 光图像可能需要预处理以匹配此格式）。

模型训练过程包括数据预处理（如图像增强、归一化）、模型初始化、损失函数（如交叉熵损失）和优化器（如 Adam）设置、训练循环、验证和测试。

## 8. 性能评估

模型评估指标包括：

- 准确率（Accuracy）：测试集上的分类准确率
- 损失（Loss）：训练、验证和测试集上的损失值

在完整测试集上，模型的性能将在训练过程中输出。

## 9. 注意事项

- 训练过程中会生成模型文件，需要足够的磁盘空间
- 使用 GPU 加速训练时，请确保已正确配置 PyTorch 的 CUDA 支持
- 数据集路径应与代码中指定的路径一致
