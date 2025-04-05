# %%
from sklearn.datasets import fetch_olivetti_faces
import ssl 
import torch
import torch.nn as nn
from sklearn.datasets import fetch_olivetti_faces

ssl._create_default_https_context = ssl._create_unverified_context

# %%
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import fetch_olivetti_faces
from torch.utils.tensorboard import SummaryWriter

class RNN_Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=64,   # 修改为64，因为每行有64个像素
            hidden_size=256,
            num_layers=5,
            batch_first=True,
            dropout=0.3
        )
        self.fc = nn.Linear(256, 40)  # 输出40个类别

    def forward(self, x):
        outputs, _ = self.rnn(x)
        return self.fc(outputs[:, -1, :])

if __name__ == '__main__':
    writer = SummaryWriter()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. 加载Olivetti Faces数据集
    olivetti_faces = fetch_olivetti_faces(data_home='./', shuffle=True)
    
    # 2. 准备数据
    # 使用images而不是data，因为我们需要保持2D结构(64x64)
    images = olivetti_faces.images  # (400, 64, 64)
    labels = olivetti_faces.target  # (400,)
    
    # 3. 划分训练测试集
    train_data = torch.FloatTensor(images[:300])  # (300, 64, 64)
    train_label = torch.LongTensor(labels[:300])
    test_data = torch.FloatTensor(images[300:])  # (100, 64, 64)
    test_label = torch.LongTensor(labels[300:])
    
    # 4. 创建Dataset和DataLoader
    train_dataset = TensorDataset(train_data, train_label)
    test_dataset = TensorDataset(test_data, test_label)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # 5. 初始化模型
    model = RNN_Classifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 6. 训练循环
    num_epochs = 25
    for epoch in range(num_epochs):
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            # 注意：这里不需要squeeze()，因为数据已经是(batch, 64, 64)
            optimizer.zero_grad()
            outputs = model(images)  # 直接输入形状为(batch, 64, 64)
            loss = criterion(outputs, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            if i % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i}/{len(train_loader)}], Loss: {loss.item():.4f}')
                writer.add_scalar('training loss', loss.item(), epoch * len(train_loader) + i)
                # 保存全部
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images.squeeze())
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            accuracy = 100 * correct / total
            print(f'Epoch [{epoch+1}/{num_epochs}], Test Accuracy: {accuracy:.2f}%')
            writer.add_scalar('test accuracy', accuracy, epoch)

# %%
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import fetch_olivetti_faces
from torch.utils.tensorboard import SummaryWriter

class RNN_Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=64,   # 修改为64，因为每行有64个像素
            hidden_size=256,
            num_layers=5,
            batch_first=True,
            dropout=0.3
        )
        self.fc = nn.Linear(256, 40)  # 输出40个类别

    def forward(self, x):
        outputs, _ = self.rnn(x)
        return self.fc(outputs[:, -1, :])

if __name__ == '__main__':
    writer = SummaryWriter()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. 加载Olivetti Faces数据集
    olivetti_faces = fetch_olivetti_faces(data_home='./', shuffle=True)
    
    # 2. 准备数据
    # 使用images而不是data，因为我们需要保持2D结构(64x64)
    images = olivetti_faces.images  # (400, 64, 64)
    labels = olivetti_faces.target  # (400,)
    
    # 3. 划分训练测试集
    train_data = torch.FloatTensor(images[:300])  # (300, 64, 64)
    train_label = torch.LongTensor(labels[:300])
    test_data = torch.FloatTensor(images[300:])  # (100, 64, 64)
    test_label = torch.LongTensor(labels[300:])
    
    # 4. 创建Dataset和DataLoader
    train_dataset = TensorDataset(train_data, train_label)
    test_dataset = TensorDataset(test_data, test_label)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # 5. 初始化模型
    model = RNN_Classifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 6. 训练循环
    num_epochs = 25
    for epoch in range(num_epochs):
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            # 注意：这里不需要squeeze()，因为数据已经是(batch, 64, 64)
            optimizer.zero_grad()
            outputs = model(images)  # 直接输入形状为(batch, 64, 64)
            loss = criterion(outputs, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            if i % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i}/{len(train_loader)}], Loss: {loss.item():.4f}')
                writer.add_scalar('training loss', loss.item(), epoch * len(train_loader) + i)
                # 保存全部
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images.squeeze())
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            accuracy = 100 * correct / total
            print(f'Epoch [{epoch+1}/{num_epochs}], Test Accuracy: {accuracy:.2f}%')
            writer.add_scalar('test accuracy', accuracy, epoch)

# %%
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from datetime import datetime
import os

# 1. 数据预处理（仅提取日期和MaxTemp）
def load_data(data_path):
    df = pd.read_csv(data_path, parse_dates=['Date'], usecols=['Date', 'MaxTemp'])
    df = df.dropna(subset=['MaxTemp'])  # 删除MaxTemp为空的行
    
    # 将日期转换为数值特征（年、月、日、年积日）
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['DayOfYear'] = df['Date'].dt.dayofyear
    
    # 只保留需要的列
    data = df[['DayOfYear', 'MaxTemp']].values  # 使用年积日作为时间特征
    
    # 数据标准化
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler

# 2. 数据集类（时间序列处理）
class TemperatureDataset(Dataset):
    def __init__(self, data, seq_length=30, pred_length=5):
        self.data = data
        self.seq_length = seq_length
        self.pred_length = pred_length

    def __len__(self):
        return len(self.data) - self.seq_length - self.pred_length + 1

    def __getitem__(self, idx):
        # 输入：前seq_length天的[DayOfYear, MaxTemp]
        x = self.data[idx:idx+self.seq_length, :]  
        # 目标：未来pred_length天的MaxTemp
        y = self.data[idx+self.seq_length:idx+self.seq_length+self.pred_length, 1]  
        return torch.FloatTensor(x), torch.FloatTensor(y)
# 3. RNN模型（简化版）
class TempPredictor(nn.Module):
    def __init__(self, input_size=2, hidden_size=64):
        super().__init__()
        self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 5)  # 输出5天的预测

    def forward(self, x):
        # x形状: (batch, seq_len, input_size)
        _, hidden = self.rnn(x)  # hidden形状: (1, batch, hidden_size)
        outputs = self.fc(hidden.squeeze(0))  # 输出形状: (batch, 5)
        return outputs

# 4. 训练和可视化
def train_and_evaluate():
    # 初始化
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    writer = SummaryWriter(f'runs/temp_pred_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    
    # 加载数据
    data, scaler = load_data('test_data.csv')
    train_size = int(0.8 * len(data))
    train_data, test_data = data[:train_size], data[train_size:]
    
    # 创建数据集
    seq_length = 30  # 使用30天历史数据
    pred_length = 5   # 预测未来5天
    train_dataset = TemperatureDataset(train_data, seq_length, pred_length)
    test_dataset = TemperatureDataset(test_data, seq_length, pred_length)
    
    # 数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # 初始化模型
    model = TempPredictor().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 训练循环
    for epoch in range(50):
        model.train()
        for x, y in train_loader:
            print(f"Batch shapes - Input: {x.shape}, Target: {y.shape}")  # 应为 (32,30,2) 和 (32,5)
            x, y = x.to(device), y.to(device)
            
            # 确保模型输出形状匹配
            outputs = model(x)  # 应为 (32,5)
            assert outputs.shape == y.shape, f"Shape mismatch: {outputs.shape} vs {y.shape}"
            
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
    
        # 评估
        model.eval()
        with torch.no_grad():
            test_loss = sum(criterion(model(x.to(device)), y.to(device)) for x, y in test_loader) / len(test_loader)
        
        # 记录到TensorBoard
        writer.add_scalars('Loss', {'train': loss.item(), 'test': test_loss}, epoch)
        
        # 可视化样例预测
        if epoch % 10 == 0:
            plot_predictions(model, test_dataset, scaler, epoch, writer)
    
    writer.close()

# 5. 预测可视化
def plot_predictions(model, dataset, scaler, epoch, writer):
    model.eval()
    with torch.no_grad():
        # 获取测试样本
        x, y_true = dataset[0]  # x: (seq_len, 2), y_true: (5,)
        y_pred = model(x.unsqueeze(0).to(device)).cpu().numpy()[0]  # 形状 (5,)
        
        # 反标准化（处理5天气温预测）
        dummy_pred = np.zeros((5, 2))
        dummy_pred[:, 1] = y_pred  # 正确赋值整个预测序列
        y_pred = scaler.inverse_transform(dummy_pred)[:, 1]
        
        dummy_true = np.zeros((5, 2))
        dummy_true[:, 1] = y_true.numpy()
        y_true = scaler.inverse_transform(dummy_true)[:, 1]
        
        # 绘图
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, 6), y_true, label='True Temp', marker='o')
        plt.plot(range(1, 6), y_pred, label='Predicted Temp', marker='x')
        plt.title(f'Temperature Prediction (Epoch {epoch})')
        plt.xlabel('Day')
        plt.ylabel('Temperature (°F)')
        plt.xticks(range(1, 6))
        plt.legend()
        
        # 保存到TensorBoard
        writer.add_figure('predictions', plt.gcf(), epoch)
        plt.close()

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_and_evaluate()


