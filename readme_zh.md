# 肺炎图像识别项目说明文档

## 1. 项目简介

本项目利用迁移学习技术，基于Inception预训练模型，实现对肺炎X光图像的自动分类识别。通过冻结预训练模型的大部分参数，仅重新训练最后的全连接层，实现了高效、准确的肺炎图像分类系统。

## 2. 系统要求

- Python 3.6+
- PyTorch
- PIL
- NumPy
- 推荐使用GPU加速训练过程
- 约2GB存储空间（用于保存模型和数据）

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
/your_image_folder/
  ├── NORMAL/
  │   ├── image1.jpeg
  │   ├── image2.jpeg
  │   └── ...
  └── PNEUMONIA/
      ├── image1.jpeg
      ├── image2.jpeg
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
- `num_epochs`: 训练轮数，默认10
- `batch_size`: 批次大小，默认32
- `learning_rate`: 学习率，默认0.001

## 6. 项目结构

```
project/
  ├── pneumonia_detection_pytorch.py  # PyTorch实现的主训练脚本
  ├── chest_xray_img/                 # 图像数据集
  │   ├── train/
  │   ├── val/
  │   └── test/
  ├── model/                          # 保存训练好的模型
  │   └── pneumonia_model.pth         # 训练好的模型文件
  └── readme_zh.md                    # 项目说明文档
```

## 7. 模型描述

本项目采用PyTorch实现了一个简单的卷积神经网络，用于肺炎图像分类。模型结构如下：

1. 输入层：3通道RGB图像
2. 卷积层1：32个3x3卷积核，ReLU激活，MaxPooling
3. 卷积层2：64个3x3卷积核，ReLU激活，MaxPooling
4. 卷积层3：128个3x3卷积核，ReLU激活，MaxPooling
5. 全连接层1：512个神经元，ReLU激活
6. 全连接层2：2个神经元（NORMAL和PNEUMONIA）

模型训练过程包括数据预处理、模型初始化、损失函数和优化器设置、训练循环、验证和测试。

## 8. 性能评估

模型评估指标包括：
- 准确率（Accuracy）：测试集上的分类准确率
- 损失（Loss）：训练、验证和测试集上的损失值

在完整测试集上，模型的性能将在训练过程中输出。

## 9. 注意事项

- 训练过程中会生成模型文件，需要足够的磁盘空间
- 使用GPU加速训练时，请确保已正确配置PyTorch的CUDA支持
- 数据集路径应与代码中指定的路径一致

## 10. 参考文献

- Kermany, D. S., et al. (2018). Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning. Cell, 172(5), 1122-1131.e9.
- Szegedy, C., et al. (2015). Going deeper with convolutions. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition.

## 11. 许可证

本项目基于MIT许可证开源。

---

*注：本项目代码由Daniel Kermany和Zhang Lab团队于2017年开发并改编，现已重构为PyTorch实现。*
