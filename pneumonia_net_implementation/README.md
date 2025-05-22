# PneumoniaNet 模型实现 (基于论文复现)

本目录中包含了一个尝试根据论文 "PneumoniaNet: Automated Detection and Classification of Pediatric Pneumonia Using Chest X-ray Images and CNN Approach" 核心思想复现的肺炎检测与分类模型。

## 模型概述

该模型使用 PyTorch 实现，旨在对儿童胸部 X 光片进行三分类 (正常、细菌性肺炎、病毒性肺炎)。其核心特点包括：

- **X-Block 结构**: 模型采用了论文中描述的 X-Block 作为特征提取的核心单元，这是一种包含多尺度卷积和残差连接的模块。
- **深度架构**: 模型包含约 50 个主要层，包括卷积层、批量归一化层、激活函数 (ReLU)、池化层和全连接层。
- **数据增强**: 训练过程中应用了多种数据增强技术，如随机翻转、旋转、剪切以及添加高斯噪声，以提高模型的泛化能力。
- **权重初始化**: 采用 Xavier (Glorot) 均匀初始化方法初始化网络权重。
- **优化器与损失函数**: 使用 Adam 优化器和分类交叉熵损失函数进行模型训练。
- **评估方法**: 采用五折交叉验证 (5-fold cross-validation) 对模型的性能进行评估。
- **可解释性**: 集成了 Grad-CAM (Gradient-weighted Class Activation Mapping) 方法，用于可视化模型在做决策时关注的图像区域，增强模型的可解释性。

详细的模型结构、X-Block 实现、训练流程和评估细节可以在 `pneumonia_model_from_paper.py` 脚本中找到。相关的论文原文和我们的分析报告也位于本目录中：

- `PneumoniaNet_Paper.pdf` (原始论文)
- `PneumoniaNet_Analysis_Report_zh.md` (对论文的技术分析报告)

## 数据要求

PneumoniaNet 模型的训练和评估数据需要存放在项目根目录下的 `chest_xray_img` 文件夹中 (即 `../chest_xray_img/`)，并遵循以下结构，其中包含 `train`, `val`, 和 `test` 三个子目录，每个子目录下又包含三个类别的文件夹：

```
../chest_xray_img/
  ├── train/
  │   ├── NORMAL/
  │   │   ├── image1.jpeg
  │   │   └── ...
  │   ├── BACTERIAL/
  │   │   ├── image1.jpeg
  │   │   └── ...
  │   └── VIRAL/
  │       ├── image1.jpeg
  │       └── ...
  ├── val/
  │   ├── NORMAL/
  │   │   └── ...
  │   ├── BACTERIAL/
  │   │   └── ...
  │   └── VIRAL/
  │       └── ...
  └── test/
      ├── NORMAL/
      │   └── ...
      ├── BACTERIAL/
      │   └── ...
      └── VIRAL/
          └── ...
```

在进行五折交叉验证时，脚本会自动从项目根目录的 `chest_xray_img` 下的这三个子目录 (`train`, `val`, `test`) 中收集所有图像及其对应的标签 (NORMAL, BACTERIAL, VIRAL) 来构建完整的数据集，然后进行划分。

## 依赖项

运行 `pneumonia_model_from_paper.py` 脚本所需的 Python 依赖项已列在 `requirements.txt` 文件中。您可以使用以下命令在本目录 (`PneumoniaNet_Implementation`) 下安装它们：

```bash
pip install -r requirements.txt
```

主要依赖包括：

- `torch` 和 `torchvision` (PyTorch 核心库)
- `Pillow` (图像处理)
- `numpy` (数值计算)
- `matplotlib` 和 `seaborn` (绘图和可视化)
- `scikit-learn` (机器学习工具，用于评估指标和交叉验证)
- `opencv-python` (OpenCV，用于 CAM 可视化中的图像处理)
- `tensorboard` (训练过程监控和可视化)

## 运行方法

要训练和评估 PneumoniaNet 模型，请在项目**根目录**下运行以下命令 (请注意，不是在本 `PneumoniaNet_Implementation` 目录下运行，因为脚本内部的相对路径是基于项目根目录设置的，例如用于保存 `results` 和 `runs` 等)：

```bash
python PneumoniaNet_Implementation/pneumonia_model_from_paper.py
```

脚本将执行以下操作：

1. 加载并预处理位于项目根目录 `chest_xray_img/` 目录下的完整数据集。
2. 执行五折交叉验证：
   - 在每一折中，重新初始化模型，并在训练集上训练指定的轮数 (默认为 100 轮)。
   - 监控训练和验证过程中的损失和准确率，并将结果记录到 TensorBoard (日志保存在项目根目录的 `runs/pneumonia_paper_model_cv_final_experiment/fold_X` 目录)。
   - 保存每折在验证集上表现最佳的模型到项目根目录的 `results/` 目录下。
   - 在每折结束后，绘制并保存该折的损失/准确率曲线、混淆矩阵、ROC 曲线和 PR 曲线到项目根目录的 `results/` 目录。
   - 为每折验证集中的部分样本生成并保存 CAM 图像到项目根目录的 `results/fold_X_cam_images/` 目录。
3. 交叉验证完成后，输出平均验证准确率和损失，并将各折的详细性能指标汇总保存到项目根目录的 `results/cv_folds_validation_summary.csv` 文件。

**主要可配置参数** (在 `pneumonia_model_from_paper.py` 脚本的 `main()` 函数开头)：

- `dataDir`: 数据集根目录 (脚本内默认为 `'chest_xray_img'`，相对于项目根目录)。
- `batchSize`: 批处理大小 (默认为 32)。
- `numEpochs`: 每个交叉验证折叠的训练轮数 (默认为 100)。
- `learningRate`: 初始学习率 (默认为 0.001)。
- `k_folds`: 交叉验证折叠数 (默认为 5)。

运行脚本前，请确保已按“数据要求”一节所述准备好数据，并按“依赖项”一节安装了所有必要的依赖项。推荐在具有 GPU 的环境上运行以加速训练过程。
