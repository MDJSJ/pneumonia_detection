import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import logging
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve, average_precision_score
import seaborn as sns
import pandas as pd
from matplotlib.ticker import MaxNLocator
from sklearn.preprocessing import label_binarize
from itertools import cycle
from sklearn.model_selection import StratifiedKFold # For K-fold cross-validation
import cv2 # For CAM visualization
import random # For CAM sample selection

# 设置日志记录
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建结果目录
os.makedirs('results', exist_ok=True)

# 自定义高斯噪声变换
class AddGaussianNoise(object):
    """自定义 torchvision 变换：向图像张量添加高斯噪声。"""
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        """向输入张量添加高斯噪声。"""
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + f'(mean={self.mean}, std={self.std})'

class ChestXRayDataset(Dataset):
    """
    自定义数据集类，用于加载胸部 X 光图像。
    支持三分类：NORMAL (正常), BACTERIAL (细菌性肺炎), VIRAL (病毒性肺炎)。
    """
    def __init__(self, dataDir, transform=None):
        """
        初始化数据集。
        参数:
            dataDir (str): 数据集根目录，应包含 'NORMAL', 'BACTERIAL', 'VIRAL' 子目录。
            transform (callable, optional): 应用于图像样本的预处理和增强操作。
        """
        self.dataDir = dataDir
        self.transform = transform
        self.images = []
        self.labels = []
        # 定义类别名称和对应的整数标签
        self.classes = ['NORMAL', 'BACTERIAL', 'VIRAL']
        self.class_to_label = {cls_name: i for i, cls_name in enumerate(self.classes)}

        try:
            for labelName in self.classes:
                labelDir = os.path.join(self.dataDir, labelName)
                if not os.path.isdir(labelDir):
                    logger.warning(f"目录 {labelDir} 不存在，跳过。")
                    continue
                
                # 获取有效图像文件名 (png, jpg, jpeg)
                imageNames = [imgName for imgName in os.listdir(labelDir) if imgName.lower().endswith(('.png', '.jpg', '.jpeg'))]
                self.images.extend([os.path.join(labelDir, imgName) for imgName in imageNames])
                self.labels.extend([self.class_to_label[labelName] for _ in imageNames])
        except OSError as e:
            logger.error(f"访问数据集目录 {self.dataDir} 时发生错误: {e}")

    def __len__(self):
        """返回数据集中样本的总数。"""
        return len(self.images)

    def __getitem__(self, idx):
        """
        获取数据集中指定索引的样本 (图像和标签)。
        参数:
            idx (int): 样本的索引。
        返回:
            tuple: (图像张量, 标签) 或在图像加载失败时 (空张量, -1)。
        """
        imgPath = self.images[idx]
        try:
            # 以 RGB 格式打开图像，即使原始图像是灰度图 (复制通道以适应常用 CNN 输入)
            image = Image.open(imgPath).convert('RGB')
        except FileNotFoundError:
            logger.error(f"图像文件未找到: {imgPath}")
            return torch.empty(0), -1 
        except IOError:
            logger.error(f"打开图像文件时发生 IO 错误: {imgPath}")
            return torch.empty(0), -1
        
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

class XBlock(nn.Module):
    """
    X-block 模块的基本实现，特点是包含残差连接 (element-wise addition)。
    此模块是构成 PneumoniaNetPaper 核心特征提取器的基本单元。
    """
    def __init__(self, in_channels, out_channels_list, kernel_sizes_list, stride=1, downsample=None):
        """
        初始化 X-block。
        参数:
            in_channels (int): 输入特征图的通道数。
            out_channels_list (list of int): 主路径上每个卷积层输出的通道数列表。
            kernel_sizes_list (list of int): 主路径上每个卷积层的卷积核大小列表。
            stride (int, optional): 主路径上第一个卷积层的步长，默认为 1。
            downsample (nn.Module, optional): 用于旁路连接的下采样模块 (通常是 1x1 卷积)，
                                            在输入和输出通道数或尺寸不匹配时使用。默认为 None。
        """
        super(XBlock, self).__init__()
        self.layers = nn.ModuleList()
        current_channels = in_channels
        
        # 构建主路径上的卷积层序列
        for i, (out_channels, kernel_size) in enumerate(zip(out_channels_list, kernel_sizes_list)):
            padding = (kernel_size - 1) // 2 # 确保 'same' 填充效果 (当 stride=1)
            self.layers.append(
                nn.Conv2d(current_channels, out_channels, kernel_size=kernel_size, 
                          stride=stride if i == 0 else 1, # 仅第一个卷积层应用指定的 stride
                          padding=padding, bias=False) # 通常 BN 层前不使用偏置
            )
            self.layers.append(nn.BatchNorm2d(out_channels)) # 批量归一化
            # 除最后一个卷积层外，其余卷积层后接 ReLU 激活函数
            if i < len(out_channels_list) - 1:
                 self.layers.append(nn.ReLU(inplace=True)) # ReLU 激活
            current_channels = out_channels
            
        self.relu_final = nn.ReLU(inplace=True) # 主路径输出和残差连接相加后的最终激活
        self.downsample = downsample

    def forward(self, x):
        """定义 X-block 的前向传播。"""
        identity = x # 旁路连接 (或称 shortcut)
        
        out = x
        for layer in self.layers: # 通过主路径
            out = layer(out)
            
        if self.downsample is not None: # 如果定义了下采样，则对旁路进行处理
            identity = self.downsample(x)
            
        # 确保 identity 和 out 的尺寸匹配以便进行元素加法
        # 如果维度不匹配 (通常是通道数)，应由 downsample 模块或 X-block 的设计保证
        if identity.size() != out.size():
            logger.warning(f"XBlock 警告: 旁路连接尺寸 {identity.size()} 与主路径输出尺寸 {out.size()} 不匹配。"
                           f"请检查 downsample 逻辑或模块设计。当前跳过残差连接以避免错误。")
            # 在严格的 ResNet 实现中，这里通常会报错或有明确的投影策略
            # 为了运行，暂时允许跳过，但这偏离了标准残差块的行为
        else:
            out += identity # 元素加法，实现残差连接
            
        out = self.relu_final(out) # 最终激活
        return out

class PneumoniaNetPaper(nn.Module):
    """
    PneumoniaNet 模型实现，尝试复现论文中的核心架构思想，特别是 X-block 结构。
    设计用于处理 256x256x3 的输入图像，并进行三分类任务。
    包含约 50 个主要层 (卷积, BN, 激活, 池化, 全连接)。
    """
    def __init__(self, numClasses=3): # 默认为三分类 (正常, 细菌性肺炎, 病毒性肺炎)
        super(PneumoniaNetPaper, self).__init__()
        
        # 初始卷积块：提取浅层特征
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=48, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(48)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2) # 输出尺寸: 48x128x128

        # X-block 序列：核心特征提取模块，堆叠多个 XBlock
        # Block 2: 输入 48 通道, 输出 32 通道 (通过 X-block 内部的 1x1 和 3x3 卷积序列实现)
        downsample_block2 = nn.Sequential(
            nn.Conv2d(48, 32, kernel_size=1, stride=1, bias=False), # 1x1 卷积用于调整旁路连接的通道数
            nn.BatchNorm2d(32),
        ) if 48 != 32 else None # 如果输入输出通道不一致，则需要下采样旁路
        self.xblock2 = XBlock(48, [128, 64, 32, 32], [1, 1, 1, 3], stride=1, downsample=downsample_block2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2) # 输出尺寸: 32x64x64

        # Block 3: 输入 32 通道, 输出 16 通道
        downsample_block3 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(16),
        ) if 32 != 16 else None
        self.xblock3 = XBlock(32, [64, 32, 16, 16], [1, 1, 1, 3], stride=1, downsample=downsample_block3)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2) # 输出尺寸: 16x32x32
        
        # Block 4: 输入 16 通道, 输出 32 通道
        downsample_block4 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(32),
        ) if 16 != 32 else None
        self.xblock4 = XBlock(16, [128, 64, 32, 32, 32], [1, 1, 1, 3, 3], stride=1, downsample=downsample_block4)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2) # 输出尺寸: 32x16x16, 作为 CAM 的目标特征图来源
        
        # 用于 CAM 的特征提取部分，聚合到最后一个卷积/池化层 (self.maxpool4)
        self.features_for_cam = nn.Sequential(
            self.conv1, self.bn1, self.relu, self.maxpool1,
            self.xblock2, self.maxpool2,
            self.xblock3, self.maxpool3,
            self.xblock4, self.maxpool4 
        )

        # 分类器模块
        # 经过特征提取后，特征图尺寸为 32 (通道数) x 16 (高) x 16 (宽)
        self.classifier_cam_amenable = nn.Sequential(
            nn.Flatten(), # 展平特征图
            nn.Linear(32 * 16 * 16, 512), # 全连接层，神经元数量 512
            nn.ReLU(inplace=True),
            nn.Dropout(0.5) # Dropout 层，防止过拟合
        )
        self.fc_final = nn.Linear(512, numClasses) # 最终输出层，对应类别数
        
        # 应用 Xavier (Glorot) 权重初始化方法
        self._initialize_weights()

    def _initialize_weights(self):
        """对模型中的卷积层和全连接层应用 Xavier 均匀初始化。"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d): # BN 层权重初始化为 1, 偏置为 0
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """定义模型的前向传播路径。"""
        features = self.features_for_cam(x)       # 获取最后一个卷积块的输出特征图
        out = self.classifier_cam_amenable(features) # 通过分类器的中间层
        out = self.fc_final(out)                   # 通过最终全连接层得到类别分数
        return out

def trainModel(model, trainLoader, criterion, optimizer, device, epoch, writer):
    """
    训练模型一个完整的轮次 (epoch)。
    """
    model.train() # 设置模型为训练模式
    runningLoss = 0.0
    correctPredictions = 0
    totalSamples = 0
    
    for batchIdx, (images, labels) in enumerate(trainLoader):
        # 跳过无效数据批次 (例如图像文件未找到导致的空数据)
        if images.numel() == 0 or (labels == -1).any():
            logger.warning(f"批次 {batchIdx} 包含无效数据，跳过。")
            continue

        images, labels = images.to(device), labels.to(device) # 数据移至指定设备
        
        optimizer.zero_grad() # 清零先前计算的梯度
        outputs = model(images) # 前向传播，获取模型输出
        loss = criterion(outputs, labels) # 计算损失，使用分类交叉熵损失函数
        loss.backward() # 反向传播，计算梯度
        optimizer.step() # 更新模型参数，使用 Adam 优化器
        
        runningLoss += loss.item()
        
        # 计算当前批次的训练准确率
        _, predicted = torch.max(outputs.data, 1)
        totalSamples += labels.size(0)
        correctPredictions += (predicted == labels).sum().item()
        
        # 每 100 个批次记录一次详细日志 (第一个批次除外)
        if batchIdx > 0 and batchIdx % 100 == 0:
             logger.info(f'训练轮次: {epoch} [{batchIdx * len(images)}/{len(trainLoader.dataset)} ({100. * batchIdx / len(trainLoader):.0f}%)]\t损失: {loss.item():.6f}')

    avgTrainLoss = runningLoss / len(trainLoader) if len(trainLoader) > 0 else 0
    trainAccuracy = correctPredictions / totalSamples if totalSamples > 0 else 0
    
    # 将平均训练损失和准确率写入 TensorBoard
    writer.add_scalar('Loss/train_epoch', avgTrainLoss, epoch)
    writer.add_scalar('Accuracy/train_epoch', trainAccuracy, epoch)
    
    return avgTrainLoss, trainAccuracy

def evaluateModel(model, valLoader, criterion, device, epoch, writer, num_classes=3, phase='Validation'):
    """
    在验证集或测试集上评估模型性能。
    """
    model.eval() # 设置模型为评估模式
    runningLoss = 0.0
    correctPredictions = 0
    totalSamples = 0
    
    allProbabilities_softmax_list = [] # 存储所有样本的 softmax 输出概率
    allPredictions = [] # 存储所有样本的预测标签
    allLabels = [] # 存储所有样本的真实标签
    
    with torch.no_grad(): # 评估时不需要计算梯度
        for images, labels in valLoader:
            if images.numel() == 0 or (labels == -1).any(): # 跳过无效数据
                continue
                
            images, labels = images.to(device), labels.to(device)
            outputs = model(images) # 前向传播
            loss = criterion(outputs, labels) # 计算损失
            runningLoss += loss.item()
            
            probabilities = torch.nn.functional.softmax(outputs, dim=1) # 计算 softmax 概率
            _, predicted = torch.max(outputs.data, 1) # 获取预测类别
            
            totalSamples += labels.size(0)
            correctPredictions += (predicted == labels).sum().item()
            
            allProbabilities_softmax_list.extend(probabilities.cpu().numpy())
            allPredictions.extend(predicted.cpu().numpy())
            allLabels.extend(labels.cpu().numpy())
            
    avgLoss = runningLoss / len(valLoader) if len(valLoader) > 0 else 0
    accuracy = correctPredictions / totalSamples if totalSamples > 0 else 0
    
    # 将损失和准确率写入 TensorBoard
    writer.add_scalar(f'Loss/{phase.lower()}_epoch', avgLoss, epoch)
    writer.add_scalar(f'Accuracy/{phase.lower()}_epoch', accuracy, epoch)
    
    logger.info(f'{phase} 集: 平均损失: {avgLoss:.4f}, 准确率: {correctPredictions}/{totalSamples} ({100. * accuracy:.2f}%)')
    
    # 返回评估结果，包括用于后续详细分析 (如 ROC/PR 曲线) 的原始概率和标签
    return avgLoss, accuracy, np.array(allProbabilities_softmax_list), allPredictions, allLabels

def plotConfusionMatrix(confMatrix, classNames, title='混淆矩阵', savePath='results/confusion_matrix.png'):
    """绘制并保存混淆矩阵图像。"""
    plt.figure(figsize=(10, 8))
    df_cm = pd.DataFrame(confMatrix, index=classNames, columns=classNames)
    try:
        sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')
    except ValueError: 
        sns.heatmap(df_cm, annot=False, fmt='d', cmap='Blues') # 如果矩阵全零等情况，不显示数字
        logger.warning("混淆矩阵热力图跳过数字标注 (可能由于矩阵全零)。")

    plt.title(title, fontsize=15)
    plt.ylabel('真实标签', fontsize=12)
    plt.xlabel('预测标签', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(savePath, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"混淆矩阵已保存至: {savePath}")

def plotROCCurve(allLabels, allProbabilities_softmax, num_classes, classNames, savePath='results/roc_curve.png'):
    """
    绘制并保存多分类任务的 ROC 曲线 (采用 one-vs-rest 策略)。
    参数:
        allProbabilities_softmax (np.array): Softmax 输出概率，形状为 (样本数, 类别数)。
    """
    y_true_binarized = label_binarize(allLabels, classes=list(range(num_classes)))
    # 处理 scikit-learn 在二分类时 binarize 输出为一维数组的情况
    if num_classes == 2 and y_true_binarized.ndim == 1:
         y_true_binarized = np.hstack(((1 - y_true_binarized).reshape(-1,1), y_true_binarized.reshape(-1,1)))

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    plt.figure(figsize=(12, 10))
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red']) # 为不同类别准备颜色

    for i, color in zip(range(num_classes), colors):
        if y_true_binarized.shape[1] > i :
            fpr[i], tpr[i], _ = roc_curve(y_true_binarized[:, i], allProbabilities_softmax[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                     label=f'类别 {classNames[i]} ROC 曲线 (AUC = {roc_auc[i]:.3f})')
        else:
            logger.warning(f"跳过类别 {i} 的 ROC 曲线绘制，因为标签二值化后的形状不匹配。")

    plt.plot([0, 1], [0, 1], 'k--', lw=2) # 对角线参考线
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假正例率 (False Positive Rate)', fontsize=12)
    plt.ylabel('真正例率 (True Positive Rate)', fontsize=12)
    plt.title('多分类受试者工作特征曲线 (ROC)', fontsize=15)
    plt.legend(loc="lower right", fontsize=10)
    
    plt.tight_layout()
    plt.savefig(savePath, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"多分类 ROC 曲线已保存至: {savePath}")

def plotPrecisionRecallCurve(allLabels, allProbabilities_softmax, num_classes, classNames, savePath='results/precision_recall_curve.png'):
    """
    绘制并保存多分类任务的精确率-召回率曲线 (采用 one-vs-rest 策略)。
    参数:
        allProbabilities_softmax (np.array): Softmax 输出概率，形状为 (样本数, 类别数)。
    """
    y_true_binarized = label_binarize(allLabels, classes=list(range(num_classes)))
    if num_classes == 2 and y_true_binarized.ndim == 1: # 同 ROC 曲线中的处理
         y_true_binarized = np.hstack(((1 - y_true_binarized).reshape(-1,1), y_true_binarized.reshape(-1,1)))

    precision = dict()
    recall = dict()
    average_precision = dict()

    plt.figure(figsize=(12, 10))
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red'])

    for i, color in zip(range(num_classes), colors):
        if y_true_binarized.shape[1] > i:
            precision[i], recall[i], _ = precision_recall_curve(y_true_binarized[:, i], allProbabilities_softmax[:, i])
            average_precision[i] = average_precision_score(y_true_binarized[:, i], allProbabilities_softmax[:, i])
            plt.plot(recall[i], precision[i], color=color, lw=2,
                     label=f'类别 {classNames[i]} PR 曲线 (AP = {average_precision[i]:.3f})')
        else:
            logger.warning(f"跳过类别 {i} 的 PR 曲线绘制，因为标签二值化后的形状不匹配。")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('召回率 (Recall)', fontsize=12)
    plt.ylabel('精确率 (Precision)', fontsize=12)
    plt.title('多分类精确率-召回率曲线', fontsize=15)
    plt.legend(loc="best", fontsize=10)
    
    plt.tight_layout()
    plt.savefig(savePath, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"多分类精确率-召回率曲线已保存至: {savePath}")

def plotTrainingHistory(trainLosses, valLosses, trainAccs, valAccs, epochs_list, saveLossPath='results/loss_curve.png', saveAccPath='results/accuracy_curve.png'):
    """绘制并保存训练过程中的损失和准确率历史曲线。"""
    epochRange = epochs_list # epochs_list 应为实际的轮次编号列表

    plt.figure(figsize=(12, 8))
    plt.plot(epochRange, trainLosses, 'b-', label='训练损失')
    plt.plot(epochRange, valLosses, 'r-', label='验证损失')
    plt.title('训练和验证损失', fontsize=15)
    plt.xlabel('轮次 (Epoch)', fontsize=12)
    plt.ylabel('损失 (Loss)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True)) # 确保 X 轴刻度为整数
    plt.tight_layout()
    plt.savefig(saveLossPath, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"损失曲线已保存至: {saveLossPath}")
    
    plt.figure(figsize=(12, 8))
    plt.plot(epochRange, trainAccs, 'b-', label='训练准确率')
    plt.plot(epochRange, valAccs, 'r-', label='验证准确率')
    plt.title('训练和验证准确率', fontsize=15)
    plt.xlabel('轮次 (Epoch)', fontsize=12)
    plt.ylabel('准确率 (Accuracy)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True)) # 确保 X 轴刻度为整数
    plt.tight_layout()
    plt.savefig(saveAccPath, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"准确率曲线已保存至: {saveAccPath}")

def generate_and_save_cam(model, image_tensor, original_image_path, target_class_idx, class_names, save_dir, device):
    """
    使用 Grad-CAM 方法为指定图像和目标类别生成并保存类激活图 (CAM)。
    参数:
        image_tensor (torch.Tensor): 经过预处理的单个图像张量，形状为 (1, C, H, W)。
        original_image_path (str): 原始图像文件的路径，用于最终可视化叠加。
        target_class_idx (int): 要为其生成 CAM 的目标类别的索引。
        class_names (list): 类别名称列表，用于生成文件名。
        save_dir (str): CAM 图像的保存目录。
        device (torch.device): 模型和数据所在的设备 (CPU 或 GPU)。
    """
    model.eval() # 确保模型处于评估模式
    image_tensor = image_tensor.to(device)

    # 存储特征图和梯度的变量
    features = None
    gradients = None

    # 定义钩子函数
    def hook_feature(module, input, output):
        nonlocal features
        features = output.detach() # (1, num_channels_feat, H_feat, W_feat)

    def hook_gradient(module, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0].detach() # (1, num_channels_feat, H_feat, W_feat)

    # 注册钩子到模型的特征提取部分的最后一个模块 (features_for_cam)
    # features_for_cam 模块的输出即为我们需要的特征图
    feature_extractor = model.features_for_cam 
    forward_hook_handle = feature_extractor.register_forward_hook(hook_feature)
    backward_hook_handle = feature_extractor.register_backward_hook(hook_gradient)
    
    # 执行一次完整的前向和后向传播以捕获特征和梯度
    model.zero_grad()
    output = model(image_tensor) # (1, num_classes)
    
    # 如果未指定目标类别，或者模型预测错误时想看预测类别的 CAM，可以从 output 中选择
    # 这里假设 target_class_idx 是明确给定的 (例如真实标签或特定想观察的类别)
    class_score = output[:, target_class_idx]
    class_score.backward() # 计算目标类别得分相对于网络参数的梯度
    
    # 移除钩子，避免影响后续操作
    forward_hook_handle.remove()
    backward_hook_handle.remove()

    if features is None or gradients is None:
        logger.error("Grad-CAM: 未能成功提取特征图或梯度。")
        return

    # 计算 alpha_k^c (论文中的梯度全局平均池化)
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3]) # (num_channels_feat)
    
    # 特征图加权求和
    # features: (1, num_channels_feat, H_feat, W_feat)
    for i in range(features.shape[1]): # 遍历每个通道
        features[:, i, :, :] *= pooled_gradients[i] # 通道加权
        
    # 计算热力图 (L_CAM^c)，即对加权后的特征图在通道维度上求均值
    heatmap = torch.mean(features, dim=1).squeeze() # (H_feat, W_feat)
    heatmap = nn.ReLU()(heatmap) # 应用 ReLU，去除负值贡献
    
    # 归一化热力图到 [0, 1]
    if heatmap.max() > 0:
        heatmap /= torch.max(heatmap)
    heatmap = heatmap.cpu().numpy()

    # 将热力图调整大小并与原始图像叠加
    try:
        img = cv2.imread(original_image_path)
        if img is None:
            logger.error(f"CAM: 无法读取原始图像: {original_image_path}")
            return
        img_height, img_width = img.shape[:2]
        
        heatmap_resized = cv2.resize(heatmap, (img_width, img_height)) # 调整至原图大小
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET) # 应用伪彩色
        
        superimposed_img = heatmap_colored * 0.4 + img * 0.6 # 热力图与原图叠加
        superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        original_filename = os.path.basename(original_image_path)
        fname_no_ext = os.path.splitext(original_filename)[0]
        target_class_name = class_names[target_class_idx] if target_class_idx < len(class_names) else f"class_{target_class_idx}"
        cam_save_path = os.path.join(save_dir, f"{fname_no_ext}_cam_for_{target_class_name}.png")
        cv2.imwrite(cam_save_path, superimposed_img)
        logger.info(f"CAM 图像已保存至: {cam_save_path} (目标类别: {target_class_name})")
    except Exception as e:
        logger.error(f"CAM: 图像处理或保存过程中发生错误: {e}")

def main():
    """主函数，执行模型训练和评估流程，采用 K 折交叉验证。"""
    # --- 基本参数配置 ---
    dataDir = 'chest_xray_img' # 数据集根目录
    batchSize = 32             # 批处理大小
    numEpochs = 100            # 每个交叉验证折叠的训练轮数 (论文中为 100)
    learningRate = 0.001       # 初始学习率 (论文中为 0.001)
    num_classes = 3            # 类别数量 (NORMAL, BACTERIAL, VIRAL)
    class_names = ['NORMAL', 'BACTERIAL', 'VIRAL'] # 类别名称列表
    k_folds = 5                # 交叉验证折叠数 (论文中为 5)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    tensorboardLogDir_base = 'runs/pneumonia_paper_model_cv_final_experiment' # TensorBoard 主日志目录
    os.makedirs(tensorboardLogDir_base, exist_ok=True)

    # --- 数据预处理与增强转换 ---
    # 训练集转换：包含尺寸调整、数据增强 (翻转、旋转、剪切、高斯噪声) 和归一化
    trainTransform = transforms.Compose([
        transforms.Resize((256, 256)), # 图像尺寸统一调整为 256x256
        transforms.RandomHorizontalFlip(), # 随机水平翻转
        transforms.RandomVerticalFlip(),   # 随机垂直翻转
        transforms.RandomRotation(15),     # 随机旋转 (15 度以内)
        transforms.RandomAffine(degrees=0, shear=(-10, 10, -10, 10)), # 随机剪切 (水平和垂直方向)
        transforms.ToTensor(),             # 转换为张量
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # 标准归一化
        AddGaussianNoise(mean=0., std=0.05) # 添加少量高斯噪声 (std=0.05, 可调优)
    ])
    
    # 验证/测试集转换：仅包含尺寸调整和归一化
    valTestTransform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # --- K 折交叉验证准备 ---
    # 加载整个数据集 (合并 train, val, test 目录) 用于交叉验证
    logger.info("开始加载完整数据集用于交叉验证...")
    full_image_paths = []
    full_labels = []
    
    # 遍历 train, val, test 子目录以收集所有图像路径和标签
    for sub_dir in ['train', 'val', 'test']:
        current_data_dir = os.path.join(dataDir, sub_dir)
        if not os.path.exists(current_data_dir):
            logger.warning(f"交叉验证: 目录 {current_data_dir} 不存在，跳过。")
            continue
        try:
            for class_idx, class_name_iter in enumerate(class_names):
                class_path = os.path.join(current_data_dir, class_name_iter)
                if os.path.isdir(class_path):
                    image_names_in_class = [imgName for imgName in os.listdir(class_path) if imgName.lower().endswith(('.png', '.jpg', '.jpeg'))]
                    full_image_paths.extend([os.path.join(class_path, img_name) for img_name in image_names_in_class])
                    full_labels.extend([class_idx] * len(image_names_in_class))
                else:
                     logger.warning(f"交叉验证: 类别目录 {class_path} 在 {current_data_dir} 中不存在，跳过。")
        except Exception as e:
            logger.error(f"交叉验证: 加载来自 {current_data_dir} 的数据时出错: {e}")

    if not full_image_paths or not full_labels:
        logger.error("未能从指定目录加载任何图像进行交叉验证。请确保 'NORMAL', 'BACTERIAL', 'VIRAL' 子目录存在且包含图像。")
        return
    logger.info(f"完整数据集加载完毕，总样本数: {len(full_image_paths)}.")
    
    full_labels_np = np.array(full_labels)
    full_image_paths_np = np.array(full_image_paths)

    # 使用分层 K 折交叉验证，确保每折中类别比例大致相同
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42) # random_state 保证可重复划分

    fold_results_val = [] # 存储每折的验证结果

    # 自定义 collate_fn，用于跳过 DataLoader 中因图像文件损坏等原因导致加载失败的样本
    def collate_fn_skip_error(batch):
        batch = list(filter(lambda x: x[0].numel() > 0 and x[1] != -1, batch)) # 过滤无效样本
        if not batch: # 如果整个批次都无效
            return torch.empty(0), torch.empty(0) 
        return torch.utils.data.dataloader.default_collate(batch)
    
    # --- K 折交叉验证主循环 ---
    for fold, (train_idx, val_idx) in enumerate(skf.split(full_image_paths_np, full_labels_np)):
        logger.info(f"--- 开始第 {fold+1}/{k_folds} 折交叉验证 ---")
        tensorboardLogDir_fold = os.path.join(tensorboardLogDir_base, f"fold_{fold+1}")
        writer = SummaryWriter(tensorboardLogDir_fold)

        # 获取当前折的训练集和验证集路径及标签
        train_paths, val_paths = full_image_paths_np[train_idx], full_image_paths_np[val_idx]
        train_labels_fold, val_labels_fold = full_labels_np[train_idx], full_labels_np[val_idx]

        # 为当前折创建 Dataset 实例 (FoldDataset 在 main 函数内部定义，以便访问外部变量如 transform)
        class FoldDataset(Dataset):
            def __init__(self, image_paths_fold, labels_fold, transform_fold=None):
                self.image_paths = image_paths_fold
                self.labels = labels_fold
                self.transform = transform_fold
            def __len__(self): return len(self.image_paths)
            def __getitem__(self, idx_fold):
                imgPath = self.image_paths[idx_fold]
                label = self.labels[idx_fold]
                try: image = Image.open(imgPath).convert('RGB')
                except: return torch.empty(0), -1 # 简化错误处理
                if self.transform: image = self.transform(image)
                return image, label

        trainDataset_fold = FoldDataset(train_paths, train_labels_fold, transform=trainTransform)
        valDataset_fold = FoldDataset(val_paths, val_labels_fold, transform=valTestTransform)

        if len(trainDataset_fold) == 0 or len(valDataset_fold) == 0:
            logger.error(f"第 {fold+1} 折: 训练集 ({len(trainDataset_fold)}) 或验证集 ({len(valDataset_fold)}) 为空，跳过此折。")
            writer.close()
            continue

        trainLoader_fold = DataLoader(trainDataset_fold, batch_size=batchSize, shuffle=True, num_workers=4, pin_memory=True, collate_fn=collate_fn_skip_error)
        valLoader_fold = DataLoader(valDataset_fold, batch_size=batchSize, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_fn_skip_error)

        # 每折重新初始化模型、优化器和学习率调度器
        model = PneumoniaNetPaper(numClasses=num_classes).to(device)
        criterion = nn.CrossEntropyLoss() # 分类交叉熵损失
        optimizer = optim.Adam(model.parameters(), lr=learningRate) # Adam 优化器
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.1, verbose=True) # 学习率调度

        trainLosses, valLosses, trainAccs, valAccs = [], [], [], []
        bestValAccuracy_fold = 0.0
        modelSavePath_fold = os.path.join('results', f'pneumonia_paper_model_v_cv_fold_{fold+1}_best.pth')

        # --- 单一折叠内的训练周期循环 ---
        for epoch in range(1, numEpochs + 1):
            logger.info(f"Fold {fold+1}, Epoch {epoch}/{numEpochs}")
            
            trainLoss, trainAcc = trainModel(model, trainLoader_fold, criterion, optimizer, device, epoch, writer)
            if not trainLoader_fold or len(trainLoader_fold) == 0: # 再次检查以防 collate_fn 返回空loader
                logger.warning(f"Fold {fold+1}, Epoch {epoch}: 训练加载器在迭代中变为空，跳过训练。")
                continue
            trainLosses.append(trainLoss)
            trainAccs.append(trainAcc)

            if not valLoader_fold or len(valLoader_fold) == 0:
                logger.warning(f"Fold {fold+1}, Epoch {epoch}: 验证加载器在迭代中变为空，跳过验证。")
                continue
            valLoss, valAcc, valProbs_softmax, valPreds, valLabels_epoch = evaluateModel(
                model, valLoader_fold, criterion, device, epoch, writer, num_classes=num_classes, phase='Validation'
            )
            valLosses.append(valLoss)
            valAccs.append(valAcc)
            
            logger.info(f'Fold {fold+1}, Epoch {epoch}: 训练损失: {trainLoss:.4f}, 准确率: {trainAcc:.4f} | 验证损失: {valLoss:.4f}, 准确率: {valAcc:.4f}')
            scheduler.step(valLoss) # 根据验证损失调整学习率

            # 保存当前折叠中验证集上表现最佳的模型
            if valAcc > bestValAccuracy_fold:
                bestValAccuracy_fold = valAcc
                try:
                    torch.save(model.state_dict(), modelSavePath_fold)
                    logger.info(f"Fold {fold+1}: 验证准确率提升至 {bestValAccuracy_fold:.4f}。模型已保存至 {modelSavePath_fold}")
                except IOError as e:
                    logger.error(f"保存模型时发生 IO 错误: {e}")
        
        # --- 当前折叠训练完成后的评估和记录 ---
        logger.info(f"--- 第 {fold+1} 折交叉验证训练结束。加载此折最佳模型进行最终验证... ---")
        if os.path.exists(modelSavePath_fold):
             model.load_state_dict(torch.load(modelSavePath_fold)) # 加载此折最佳模型
        else:
            logger.warning(f"Fold {fold+1}: 未找到保存的最佳模型 {modelSavePath_fold}，将使用最后训练的模型进行评估。")

        # 使用此折的最佳（或最后）模型在验证集上进行最终评估
        finalValLoss_fold, finalValAcc_fold, finalValProbs_softmax, finalValPreds, finalValLabels = evaluateModel(
             model, valLoader_fold, criterion, device, numEpochs, writer, num_classes=num_classes, phase='Fold_Validation_Final'
        )
        fold_results_val.append({
            'fold': fold + 1, 'loss': finalValLoss_fold, 'accuracy': finalValAcc_fold,
            'predictions': finalValPreds, 'labels': finalValLabels, 'probabilities': finalValProbs_softmax
        })
        logger.info(f"--- 第 {fold+1} 折最终验证准确率: {finalValAcc_fold:.4f} ---") 
        
        # 绘制并保存此折的训练历史曲线和性能图表
        epochs_completed_list = list(range(1, len(trainLosses) + 1))
        if trainLosses: 
            plotTrainingHistory(
                trainLosses, valLosses, trainAccs, valAccs, epochs_completed_list,
                saveLossPath=f'results/fold_{fold+1}_loss_curve.png', 
                saveAccPath=f'results/fold_{fold+1}_accuracy_curve.png'
            )
        if finalValLabels is not None and len(finalValLabels) > 0 and finalValProbs_softmax is not None and finalValProbs_softmax.size > 0 :
            cm = confusion_matrix(finalValLabels, finalValPreds, labels=list(range(num_classes)))
            plotConfusionMatrix(cm, class_names, title=f'Fold {fold+1} 验证集混淆矩阵', savePath=f'results/fold_{fold+1}_confusion_matrix_val.png')
            plotROCCurve(finalValLabels, finalValProbs_softmax, num_classes, class_names, savePath=f'results/fold_{fold+1}_roc_curve_val.png')
            plotPrecisionRecallCurve(finalValLabels, finalValProbs_softmax, num_classes, class_names, savePath=f'results/fold_{fold+1}_pr_curve_val.png')
            report_str_fold = classification_report(finalValLabels, finalValPreds, target_names=class_names, labels=list(range(num_classes)), zero_division=0)
            logger.info(f"Fold {fold+1} 分类报告:\n{report_str_fold}")
            with open(f'results/fold_{fold+1}_classification_report.txt', 'w', encoding='utf-8') as f: f.write(report_str_fold)

            # --- 为当前折叠的几个验证样本生成 CAM ---
            if len(val_paths) > 0: # val_paths 来自 StratifiedKFold 当前折的划分
                cam_save_directory_fold = os.path.join('results', f'fold_{fold+1}_cam_images')
                os.makedirs(cam_save_directory_fold, exist_ok=True)
                num_cam_samples_per_class = 1 # 每个类别生成 1 个 CAM 样本
                
                for class_idx_cam in range(num_classes):
                    # 从当前验证集标签中找到属于该类别的样本索引
                    class_specific_indices_in_val_labels = [i for i, label in enumerate(finalValLabels) if label == class_idx_cam]
                    if class_specific_indices_in_val_labels:
                        # 随机选择样本
                        selected_indices_for_cam = random.sample(
                            class_specific_indices_in_val_labels, 
                            min(num_cam_samples_per_class, len(class_specific_indices_in_val_labels))
                        )
                        for sample_original_idx_in_fold_val_set in selected_indices_for_cam:
                            # 获取原始图像路径 (val_paths[sample_original_idx_in_fold_val_set] 是该样本在当前验证集划分中的路径)
                            original_image_cv_path = val_paths[sample_original_idx_in_fold_val_set] 
                                                    
                            try:
                                img_pil = Image.open(original_image_cv_path).convert('RGB')
                                # 使用验证集的变换 valTestTransform 来准备图像张量
                                img_tensor_for_cam = valTestTransform(img_pil).unsqueeze(0) 
                                
                                true_label_for_cam = finalValLabels[sample_original_idx_in_fold_val_set]
                                predicted_label_for_cam = finalValPreds[sample_original_idx_in_fold_val_set]

                                logger.info(f"Fold {fold+1}: 生成 CAM, 图像: {os.path.basename(original_image_cv_path)}, 真实类别: {class_names[true_label_for_cam]}, 预测类别: {class_names[predicted_label_for_cam]}")
                                
                                # 为真实类别生成 CAM
                                generate_and_save_cam(model, img_tensor_for_cam, original_image_cv_path, true_label_for_cam, class_names, cam_save_directory_fold, device)
                                # 如果预测错误，也为预测类别生成 CAM (可选)
                                if true_label_for_cam != predicted_label_for_cam:
                                     generate_and_save_cam(model, img_tensor_for_cam, original_image_cv_path, predicted_label_for_cam, class_names, cam_save_directory_fold, device)

                            except Exception as e_cam:
                                logger.error(f"为图像 {original_image_cv_path} 生成 CAM 时出错: {e_cam}")
        
        # 记录超参数和此折的验证性能到 TensorBoard
        writer.add_hparams(
            {"lr": learningRate, "batch_size": batchSize, "fold": fold+1},
            {"hparam/val_accuracy": finalValAcc_fold, "hparam/val_loss": finalValLoss_fold}
        )
        writer.close() # 关闭当前折的 TensorBoard writer

    # --- 所有交叉验证折叠完成后的总结 ---
    logger.info("--- 所有交叉验证折叠已完成 --- ")
    if fold_results_val: # 确保 fold_results_val 不为空
        avg_val_accuracy = np.mean([res['accuracy'] for res in fold_results_val if 'accuracy' in res])
        avg_val_loss = np.mean([res['loss'] for res in fold_results_val if 'loss' in res])
        std_val_accuracy = np.std([res['accuracy'] for res in fold_results_val if 'accuracy' in res])
        
        logger.info(f"平均验证准确率 (K={k_folds} 折): {avg_val_accuracy:.4f} +/- {std_val_accuracy:.4f}")
        logger.info(f"平均验证损失 (K={k_folds} 折): {avg_val_loss:.4f}")

        # 保存所有折叠的性能摘要到 CSV 文件
        try:
            fold_results_df = pd.DataFrame([ {k:v for k,v in res.items() if k not in ['predictions', 'labels', 'probabilities']} for res in fold_results_val])
            fold_results_df.to_csv('results/cv_folds_validation_summary.csv', index=False)
            logger.info("交叉验证各折验证结果摘要已保存至: results/cv_folds_validation_summary.csv")
        except Exception as e_df:
            logger.error(f"保存交叉验证摘要到 CSV 时出错: {e_df}")
    else:
        logger.warning("交叉验证未能产生任何结果，无法进行总结。")
        
    logger.info("交叉验证训练和评估流程结束。")

if __name__ == '__main__':
    main() 
