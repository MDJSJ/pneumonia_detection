# 导入必要的库
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter

# 定义数据集类
class ChestXRayDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.labels = []

        # 遍历数据集目录，加载图像和标签
        for label in ['NORMAL', 'PNEUMONIA']:
            label_dir = os.path.join(data_dir, label)
            for img_name in os.listdir(label_dir):
                img_path = os.path.join(label_dir, img_name)
                self.images.append(img_path)
                # 将PNEUMONIA标签设为1，NORMAL设为0
                self.labels.append(1 if label == 'PNEUMONIA' else 0)

    def __len__(self):
        # 返回数据集中的图像数量
        return len(self.images)

    def __getitem__(self, idx):
        # 获取指定索引的图像路径和标签
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        # 如果有预处理变换，则应用变换
        if self.transform:
            image = self.transform(image)

        return image, label

# 定义模型
class PneumoniaNet(nn.Module):
    def __init__(self):
        super(PneumoniaNet, self).__init__()
        # 定义卷积层
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # 定义池化层
        self.pool = nn.MaxPool2d(2, 2)
        # 定义全连接层
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, 2)
        # 定义激活函数
        self.relu = nn.ReLU()

    def forward(self, x):
        # 定义前向传播过程
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        # 展平特征图
        x = x.view(-1, 128 * 28 * 28)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 训练函数
def train_model(model, train_loader, criterion, optimizer, device):
    # 设置模型为训练模式
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        # 将数据移动到指定设备（CPU或GPU）
        images, labels = images.to(device), labels.to(device)
        # 清零梯度
        optimizer.zero_grad()
        # 前向传播
        outputs = model(images)
        # 计算损失
        loss = criterion(outputs, labels)
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()
        # 累积损失
        running_loss += loss.item()
    # 返回平均损失
    return running_loss / len(train_loader)

# 评估函数
def evaluate_model(model, val_loader, criterion, device):
    # 设置模型为评估模式
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            # 将数据移动到指定设备（CPU或GPU）
            images, labels = images.to(device), labels.to(device)
            # 前向传播
            outputs = model(images)
            # 计算损失
            loss = criterion(outputs, labels)
            # 累积损失
            running_loss += loss.item()
            # 获取预测结果
            _, predicted = torch.max(outputs.data, 1)
            # 累积总样本数和正确预测数
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    # 计算准确率
    accuracy = correct / total
    # 返回平均损失和准确率
    return running_loss / len(val_loader), accuracy

# 主函数
def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 初始化 TensorBoard Writer
    writer = SummaryWriter('runs/pneumonia_detection_experiment')

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加载数据集
    train_dataset = ChestXRayDataset('chest_xray_img/train', transform=transform)
    val_dataset = ChestXRayDataset('chest_xray_img/val', transform=transform)
    test_dataset = ChestXRayDataset('chest_xray_img/test', transform=transform)

    # 调整批次大小以确保输入和目标批次大小一致
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 初始化模型
    model = PneumoniaNet().to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # 训练模型
    num_epochs = 30

    for epoch in range(num_epochs):
        train_loss = train_model(model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy = evaluate_model(model, val_loader, criterion, device)
        print(f'第{epoch+1}/{num_epochs}轮，训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}, 验证准确率: {val_accuracy:.4f}')

        # 记录到 TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/validation', val_loss, epoch)
        writer.add_scalar('Accuracy/validation', val_accuracy, epoch)

    # 测试模型
    test_loss, test_accuracy = evaluate_model(model, test_loader, criterion, device)
    print(f'测试损失: {test_loss:.4f}, 测试准确率: {test_accuracy:.4f}')

    # 记录测试结果到 TensorBoard
    writer.add_scalar('Loss/test', test_loss, num_epochs)
    writer.add_scalar('Accuracy/test', test_accuracy, num_epochs)

    # 保存模型
    torch.save(model.state_dict(), 'pneumonia_model.pth')

    # 关闭 TensorBoard writer
    writer.close()

if __name__ == '__main__':
    main()
