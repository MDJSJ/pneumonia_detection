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

# 设置日志记录
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建结果目录
os.makedirs('results', exist_ok=True)

class ChestXRayDataset(Dataset):
    """
    用于加载胸部X光图像数据集的自定义数据集类。

    属性:
        dataDir (str): 数据集目录的路径。
        transform (callable, optional): 应用于样本的可选转换。
        images (list): 存储图像文件路径的列表。
        labels (list): 存储图像对应标签的列表。
    """
    def __init__(self, dataDir, transform=None):
        """
        初始化 ChestXRayDataset。

        参数:
            dataDir (str): 包含 'NORMAL' 和 'PNEUMONIA' 子目录的数据集根目录。
            transform (callable, optional): 应用于每个图像样本的可选转换。
        """
        self.dataDir = dataDir
        self.transform = transform
        self.images = []
        self.labels = []

        try:
            for labelName in ['NORMAL', 'PNEUMONIA']:
                labelDir = os.path.join(self.dataDir, labelName)
                if not os.path.isdir(labelDir):
                    logger.warning(f"目录 {labelDir} 不存在，跳过。")
                    continue
                
                imageNames = os.listdir(labelDir)
                self.images.extend([os.path.join(labelDir, imgName) for imgName in imageNames])
                self.labels.extend([1 if labelName == 'PNEUMONIA' else 0 for _ in imageNames])
        except OSError as e:
            logger.error(f"访问数据集目录 {self.dataDir} 时发生错误: {e}")
            # 可以选择抛出异常或使数据集为空
            # raise

    def __len__(self):
        """返回数据集中样本的总数。"""
        return len(self.images)

    def __getitem__(self, idx):
        """
        获取数据集中指定索引的样本。

        参数:
            idx (int): 样本的索引。

        返回:
            tuple: (image, label) 其中 image 是转换后的图像，label 是对应的标签。
                   如果图像加载失败，则可能返回 None 或根据错误处理策略采取其他行动。
        """
        imgPath = self.images[idx]
        try:
            image = Image.open(imgPath).convert('RGB')
        except FileNotFoundError:
            logger.error(f"图像文件未找到: {imgPath}")
            return None, None # 或者根据需要处理
        except IOError:
            logger.error(f"打开图像文件时发生IO错误: {imgPath}")
            return None, None # 或者根据需要处理
        
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

class PneumoniaNetPaper(nn.Module):
    """
    根据论文表格描述实现的肺炎检测CNN模型。
    输入图像尺寸假定为 256x256x3。
    """
    def __init__(self, numClasses=2):
        """
        初始化 PneumoniaNetPaper 模型。

        参数:
            numClasses (int): 输出类别的数量（例如，2表示NORMAL和PNEUMONIA）。
        """
        super(PneumoniaNetPaper, self).__init__()
        
        # 特征提取器
        self.features = nn.Sequential(
            # Block 1 (Input: 3x256x256)
            nn.Conv2d(in_channels=3, out_channels=48, kernel_size=3, stride=1, padding=1), # Conv_1
            nn.BatchNorm2d(48), # Batch_Norm_1
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # Maxpol_1 (Output: 48x128x128)

            # Block 2
            nn.Conv2d(in_channels=48, out_channels=128, kernel_size=1, stride=1, padding=0), # Conv_2
            nn.BatchNorm2d(128), # Batch_Norm_2
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=0), # Conv_3
            nn.BatchNorm2d(64), # Batch_Norm_3
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0), # Conv_4
            nn.BatchNorm2d(32), # Batch_Norm_4
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1), # Conv_5
            nn.BatchNorm2d(32), # Batch_Norm_5
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # Maxpol_2 (Output: 32x64x64)

            # Block 3
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, stride=1, padding=0), # Conv_6
            nn.BatchNorm2d(64), # Batch_Norm_6
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0), # Conv_7
            nn.BatchNorm2d(32), # Batch_Norm_7
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=1, stride=1, padding=0), # Conv_8
            nn.BatchNorm2d(16), # Batch_Norm_8
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1), # Conv_9
            nn.BatchNorm2d(16), # Batch_Norm_9
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # Maxpol_3 (Output: 16x32x32)

            # Block 4
            nn.Conv2d(in_channels=16, out_channels=128, kernel_size=1, stride=1, padding=0), # Conv_10
            nn.BatchNorm2d(128), # Batch_Norm_10
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=0), # Conv_11
            nn.BatchNorm2d(64), # Batch_Norm_11
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0), # Conv_12
            nn.BatchNorm2d(32), # Batch_Norm_12
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1), # Conv_13
            nn.BatchNorm2d(32), # Batch_Norm_13
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1), # Conv_14
            nn.BatchNorm2d(32), # Batch_Norm_14
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2) # Maxpol_4 (Output: 32x16x16)
        )
        
        # 分类器
        # 最终特征图尺寸为 32 * 16 * 16
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5), # 添加 Dropout 以增强泛化能力
            nn.Linear(512, numClasses)
        )

    def forward(self, x):
        """定义模型的前向传播路径。"""
        x = self.features(x)
        x = self.classifier(x)
        return x

def trainModel(model, trainLoader, criterion, optimizer, device, epoch, writer):
    """
    训练模型一个轮次。

    参数:
        model (nn.Module): 要训练的模型。
        trainLoader (DataLoader): 训练数据的 DataLoader。
        criterion (nn.Module): 损失函数。
        optimizer (Optimizer): 优化器。
        device (torch.device): 训练设备 (CPU 或 GPU)。
        epoch (int): 当前训练轮次。
        writer (SummaryWriter): TensorBoard 的 SummaryWriter。

    返回:
        float: 当前轮次的平均训练损失。
    """
    model.train()
    runningLoss = 0.0
    correctPredictions = 0
    totalSamples = 0
    
    for batchIdx, (images, labels) in enumerate(trainLoader):
        if images is None or labels is None:  # 处理 __getitem__ 可能返回 None 的情况
            logger.warning(f"跳过批次 {batchIdx} 因为数据加载失败。")
            continue

        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        runningLoss += loss.item()
        
        # 计算训练准确率
        _, predicted = torch.max(outputs.data, 1)
        totalSamples += labels.size(0)
        correctPredictions += (predicted == labels).sum().item()
        
        if batchIdx % 100 == 0:  # 每100个批次打印一次日志
             logger.info(f'训练轮次: {epoch} [{batchIdx * len(images)}/{len(trainLoader.dataset)} ({100. * batchIdx / len(trainLoader):.0f}%)]\t损失: {loss.item():.6f}')

    avgTrainLoss = runningLoss / len(trainLoader)
    trainAccuracy = correctPredictions / totalSamples
    
    # 记录每个轮次的损失和准确率
    writer.add_scalar('Loss/train_epoch', avgTrainLoss, epoch)
    writer.add_scalar('Accuracy/train_epoch', trainAccuracy, epoch)
    
    return avgTrainLoss, trainAccuracy

def evaluateModel(model, valLoader, criterion, device, epoch, writer, phase='Validation'):
    """
    评估模型在验证集或测试集上的性能。

    参数:
        model (nn.Module): 要评估的模型。
        valLoader (DataLoader): 验证/测试数据的 DataLoader。
        criterion (nn.Module): 损失函数。
        device (torch.device): 评估设备 (CPU 或 GPU)。
        epoch (int): 当前轮次或评估阶段的标识。
        writer (SummaryWriter): TensorBoard 的 SummaryWriter。
        phase (str): 'Validation' 或 'Test'，用于日志记录。

    返回:
        tuple: (平均损失, 准确率, 所有预测概率, 所有标签)
    """
    model.eval()
    runningLoss = 0.0
    correctPredictions = 0
    totalSamples = 0
    
    allProbabilities = []
    allPredictions = []
    allLabels = []
    
    with torch.no_grad():
        for images, labels in valLoader:
            if images is None or labels is None:
                continue
                
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            runningLoss += loss.item()
            
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            totalSamples += labels.size(0)
            correctPredictions += (predicted == labels).sum().item()
            
            # 收集所有预测和标签用于后续分析
            allProbabilities.extend(probabilities[:, 1].cpu().numpy())  # 肺炎的概率
            allPredictions.extend(predicted.cpu().numpy())
            allLabels.extend(labels.cpu().numpy())
            
    avgLoss = runningLoss / len(valLoader)
    accuracy = correctPredictions / totalSamples
    
    writer.add_scalar(f'Loss/{phase.lower()}_epoch', avgLoss, epoch)
    writer.add_scalar(f'Accuracy/{phase.lower()}_epoch', accuracy, epoch)
    
    logger.info(f'{phase} 集: 平均损失: {avgLoss:.4f}, 准确率: {correctPredictions}/{totalSamples} ({100. * accuracy:.2f}%)')
    
    return avgLoss, accuracy, allProbabilities, allPredictions, allLabels

def plotConfusionMatrix(confMatrix, classNames, title='混淆矩阵', savePath='results/confusion_matrix.png'):
    """
    绘制并保存混淆矩阵。
    
    参数:
        confMatrix (array): 混淆矩阵数组。
        classNames (list): 类别名称列表。
        title (str): 图表标题。
        savePath (str): 保存路径。
    """
    plt.figure(figsize=(10, 8))
    
    # 使用seaborn绘制更美观的混淆矩阵
    df_cm = pd.DataFrame(confMatrix, index=classNames, columns=classNames)
    heatmap = sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')
    
    plt.title(title, fontsize=15)
    plt.ylabel('真实标签', fontsize=12)
    plt.xlabel('预测标签', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(savePath, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"混淆矩阵已保存至: {savePath}")

def plotROCCurve(allLabels, allProbabilities, savePath='results/roc_curve.png'):
    """
    绘制并保存ROC曲线。
    
    参数:
        allLabels (list): 所有真实标签。
        allProbabilities (list): 所有预测概率。
        savePath (str): 保存路径。
    """
    fpr, tpr, _ = roc_curve(allLabels, allProbabilities)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 8))
    
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假正例率 (False Positive Rate)', fontsize=12)
    plt.ylabel('真正例率 (True Positive Rate)', fontsize=12)
    plt.title('受试者工作特征曲线 (ROC)', fontsize=15)
    plt.legend(loc="lower right", fontsize=12)
    
    plt.tight_layout()
    plt.savefig(savePath, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"ROC曲线已保存至: {savePath}")

def plotPrecisionRecallCurve(allLabels, allProbabilities, savePath='results/precision_recall_curve.png'):
    """
    绘制并保存精确率-召回率曲线。
    
    参数:
        allLabels (list): 所有真实标签。
        allProbabilities (list): 所有预测概率。
        savePath (str): 保存路径。
    """
    precision, recall, _ = precision_recall_curve(allLabels, allProbabilities)
    avg_precision = average_precision_score(allLabels, allProbabilities)
    
    plt.figure(figsize=(10, 8))
    
    plt.plot(recall, precision, color='blue', lw=2, label=f'精确率-召回率曲线 (AP = {avg_precision:.3f})')
    plt.axhline(y=sum(allLabels) / len(allLabels), color='red', linestyle='--', label='随机分类器基线')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('召回率 (Recall)', fontsize=12)
    plt.ylabel('精确率 (Precision)', fontsize=12)
    plt.title('精确率-召回率曲线', fontsize=15)
    plt.legend(loc="lower left", fontsize=12)
    
    plt.tight_layout()
    plt.savefig(savePath, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"精确率-召回率曲线已保存至: {savePath}")

def plotTrainingHistory(trainLosses, valLosses, trainAccs, valAccs, epochs, saveLossPath='results/loss_curve.png', saveAccPath='results/accuracy_curve.png'):
    """
    绘制并保存训练历史曲线。
    
    参数:
        trainLosses (list): 训练损失列表。
        valLosses (list): 验证损失列表。
        trainAccs (list): 训练准确率列表。
        valAccs (list): 验证准确率列表。
        epochs (int): 训练轮数。
        saveLossPath (str): 损失曲线保存路径。
        saveAccPath (str): 准确率曲线保存路径。
    """
    epochRange = range(1, epochs + 1)
    
    # 绘制损失曲线
    plt.figure(figsize=(12, 8))
    plt.plot(epochRange, trainLosses, 'b-', label='训练损失')
    plt.plot(epochRange, valLosses, 'r-', label='验证损失')
    
    plt.title('训练和验证损失', fontsize=15)
    plt.xlabel('轮次 (Epoch)', fontsize=12)
    plt.ylabel('损失 (Loss)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.xticks(epochRange)
    
    # 确保x轴刻度是整数
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    plt.tight_layout()
    plt.savefig(saveLossPath, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"损失曲线已保存至: {saveLossPath}")
    
    # 绘制准确率曲线
    plt.figure(figsize=(12, 8))
    plt.plot(epochRange, trainAccs, 'b-', label='训练准确率')
    plt.plot(epochRange, valAccs, 'r-', label='验证准确率')
    
    plt.title('训练和验证准确率', fontsize=15)
    plt.xlabel('轮次 (Epoch)', fontsize=12)
    plt.ylabel('准确率 (Accuracy)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.xticks(epochRange)
    
    # 确保x轴刻度是整数
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    plt.tight_layout()
    plt.savefig(saveAccPath, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"准确率曲线已保存至: {saveAccPath}")

def main():
    """主函数，执行模型训练和评估流程。"""
    # 设置参数
    dataDir = 'chest_xray_img'  # 根据你的数据集路径修改
    batchSize = 32  # 建议使用更大的批次大小，例如 16, 32, 64
    numEpochs = 50  # 增加训练轮数
    learningRate = 0.001  # 初始学习率
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    # TensorBoard Writer
    tensorboardLogDir = 'runs/pneumonia_paper_model_experiment'
    writer = SummaryWriter(tensorboardLogDir)
    logger.info(f"TensorBoard 日志将保存在: {tensorboardLogDir}")

    # 数据预处理与增强
    # 注意：输入尺寸调整为 256x256
    trainTransform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(), # 简单数据增强
        transforms.RandomRotation(10),     # 简单数据增强
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    valTestTransform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加载数据集
    try:
        trainDataset = ChestXRayDataset(os.path.join(dataDir, 'train'), transform=trainTransform)
        valDataset = ChestXRayDataset(os.path.join(dataDir, 'val'), transform=valTestTransform)
        testDataset = ChestXRayDataset(os.path.join(dataDir, 'test'), transform=valTestTransform)

        if not trainDataset or not valDataset or not testDataset:
            logger.error("一个或多个数据集为空，请检查数据集路径和内容。")
            return
    except Exception as e:
        logger.error(f"加载数据集时发生严重错误: {e}")
        return

    trainLoader = DataLoader(trainDataset, batch_size=batchSize, shuffle=True, num_workers=4, pin_memory=True)
    valLoader = DataLoader(valDataset, batch_size=batchSize, shuffle=False, num_workers=4, pin_memory=True)
    testLoader = DataLoader(testDataset, batch_size=batchSize, shuffle=False, num_workers=4, pin_memory=True)

    # 初始化模型
    model = PneumoniaNetPaper(numClasses=2).to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learningRate)
    
    # 学习率调度器 (可选, 但推荐)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.1, verbose=True)

    # 训练历史记录
    trainLosses = []
    valLosses = []
    trainAccs = []
    valAccs = []
    
    bestValAccuracy = 0.0
    modelSavePath = 'pneumonia_paper_model_best.pth'

    # 训练模型
    for epoch in range(1, numEpochs + 1):
        logger.info(f"开始轮次 {epoch}/{numEpochs}")
        
        # 训练阶段
        trainLoss, trainAcc = trainModel(model, trainLoader, criterion, optimizer, device, epoch, writer)
        trainLosses.append(trainLoss)
        trainAccs.append(trainAcc)
        
        # 验证阶段
        valLoss, valAcc, valProbs, valPreds, valLabels = evaluateModel(
            model, valLoader, criterion, device, epoch, writer, phase='Validation')
        valLosses.append(valLoss)
        valAccs.append(valAcc)
        
        logger.info(f'轮次 {epoch}/{numEpochs} 完成: 训练损失: {trainLoss:.4f}, 训练准确率: {trainAcc:.4f}, 验证损失: {valLoss:.4f}, 验证准确率: {valAcc:.4f}')
        
        # 更新学习率
        scheduler.step(valLoss)
        
        # 在每10个epoch或最后一个epoch绘制训练历史曲线
        if epoch % 10 == 0 or epoch == numEpochs:
            # 绘制并保存训练历史曲线
            plotTrainingHistory(
                trainLosses, valLosses, trainAccs, valAccs, epoch,
                saveLossPath=f'results/loss_curve_epoch_{epoch}.png', 
                saveAccPath=f'results/accuracy_curve_epoch_{epoch}.png'
            )
            
            # 如果是最后一个epoch，额外绘制混淆矩阵和ROC曲线
            if epoch == numEpochs:
                # 计算并绘制混淆矩阵
                cm = confusion_matrix(valLabels, valPreds)
                plotConfusionMatrix(
                    cm, ['正常', '肺炎'], 
                    title=f'验证集混淆矩阵 (Epoch {epoch})',
                    savePath=f'results/confusion_matrix_val_epoch_{epoch}.png'
                )
                
                # 绘制ROC曲线
                plotROCCurve(
                    valLabels, valProbs,
                    savePath=f'results/roc_curve_val_epoch_{epoch}.png'
                )
                
                # 绘制精确率-召回率曲线
                plotPrecisionRecallCurve(
                    valLabels, valProbs,
                    savePath=f'results/pr_curve_val_epoch_{epoch}.png'
                )

        # 保存最佳模型
        if valAcc > bestValAccuracy:
            bestValAccuracy = valAcc
            try:
                torch.save(model.state_dict(), modelSavePath)
                logger.info(f"验证准确率提升，模型已保存至: {modelSavePath}")
            except IOError as e:
                logger.error(f"保存模型时发生IO错误: {e}")

    # 测试模型 (使用最佳模型)
    logger.info("训练完成，加载最佳模型进行测试...")
    try:
        model.load_state_dict(torch.load(modelSavePath))
        logger.info(f"从 {modelSavePath} 加载模型成功。")
    except FileNotFoundError:
        logger.error(f"模型文件 {modelSavePath} 未找到，使用最后训练的模型进行测试。")
    except Exception as e:
        logger.error(f"加载模型 {modelSavePath} 时发生错误: {e}，使用最后训练的模型进行测试。")
    
    # 在测试集上评估
    testLoss, testAcc, testProbs, testPreds, testLabels = evaluateModel(
        model, testLoader, criterion, device, numEpochs, writer, phase='Test')
    logger.info(f'最终测试结果: 测试损失: {testLoss:.4f}, 测试准确率: {testAcc:.4f}')
    
    # 绘制测试集性能指标
    # 计算并绘制混淆矩阵
    cm = confusion_matrix(testLabels, testPreds)
    plotConfusionMatrix(
        cm, ['正常', '肺炎'], 
        title='测试集混淆矩阵',
        savePath='results/confusion_matrix_test.png'
    )
    
    # 绘制ROC曲线
    plotROCCurve(
        testLabels, testProbs,
        savePath='results/roc_curve_test.png'
    )
    
    # 绘制精确率-召回率曲线
    plotPrecisionRecallCurve(
        testLabels, testProbs,
        savePath='results/pr_curve_test.png'
    )
    
    # 生成分类报告
    classificationReportStr = classification_report(testLabels, testPreds, target_names=['正常', '肺炎'])
    logger.info(f"分类报告:\n{classificationReportStr}")
    
    # 保存分类报告到文件
    with open('results/classification_report.txt', 'w', encoding='utf-8') as f:
        f.write(classificationReportStr)
    
    writer.add_hparams(
        {"lr": learningRate, "batch_size": batchSize, "epochs": numEpochs},
        {"hparam/test_accuracy": testAcc, "hparam/test_loss": testLoss, "hparam/best_val_accuracy": bestValAccuracy}
    )

    writer.close()
    logger.info("训练和评估流程结束。")

if __name__ == '__main__':
    main() 