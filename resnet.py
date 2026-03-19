import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from scipy.io import loadmat
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import math
import random
torch.manual_seed(42)
# =====================
# 1. 参数配置
# =====================
DATA_ROOT = r"E:\yan\外辐射源数据集-雷达学报\RD"
DATA_ROOT = r"E:\yan\rd\种类识别"
BATCH_SIZE = 32
EPOCHS = 20
LR = 3e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================
# 2. 自定义 Dataset
# =====================
class RadarDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        for label in ["0", "1","2"]:
            class_dir = os.path.join(root_dir, label)
            for file in os.listdir(class_dir):
                if file.endswith(".mat"):
                    self.samples.append(
                        (os.path.join(class_dir, file), int(label))
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        mat = loadmat(path)
        # 取 mat 中第一个非系统变量
        data = None
        for k in mat.keys():
            if not k.startswith("__"):
                data = mat[k]
                break

        # data: (5, 200) 复数

        # mag_sq = abs(data)**2
        # mag_sq = np.log10(mag_sq)
        mag_sq = data
        mag_sq = math.e**(mag_sq/10)
        # 归一化
        # mag_sq = (mag_sq - mag_sq.min()) / (mag_sq.max() - mag_sq.min() + 1e-8)

        # (5,200) -> (3,5,200)
        # mag_sq = mag_sq[:, 175:225]
        mag_sq = np.expand_dims(mag_sq, axis=0)


        mag_sq = torch.tensor(mag_sq, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)

        return mag_sq, label


# =====================
# 3. 构建 ResNet34
# =====================
def build_resnet34(num_classes=3):
    model = models.resnet18(pretrained=False)

    # 修改第一层，适配小尺寸
    model.conv1 = nn.Conv2d(
        1, 64, kernel_size=3, stride=1, padding=1, bias=False
    )
    model.maxpool = nn.Identity()

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# =====================
# 4. 训练函数
# =====================
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = out.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    return total_loss / len(loader), correct / total


# =====================
# 5. 测试 + 混淆矩阵
# =====================
def test(model, loader):
    model.eval()
    preds = []
    labels = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            out = model(x)
            pred = out.argmax(dim=1).cpu().numpy()

            preds.extend(pred)
            labels.extend(y.numpy())

    acc = np.mean(np.array(preds) == np.array(labels))
    cm = confusion_matrix(labels, preds)

    return acc, cm


# =====================
# 6. 主流程
# =====================
def main():
    train_dataset = RadarDataset(os.path.join(DATA_ROOT, "test"))
    test_dataset = RadarDataset(os.path.join(DATA_ROOT, "train_92"))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = build_resnet34().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer
        )
        print(
            f"Epoch [{epoch+1}/{EPOCHS}] "
            f"Loss: {train_loss:.4f}  Acc: {train_acc:.4f}"
        )

    test_acc, cm = test(model, test_loader)
    print("\nTest Accuracy:", test_acc)
    print("Confusion Matrix:")
    print(cm)
    return  test_acc


if __name__ == "__main__":
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    acclist = []
    turn_list = []
    turn_num = 20
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    for i in range(turn_num):
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        acc_one = main()
        acclist.append(acc_one)
        turn_list.append(i+1)
    print(np.var(acclist),np.mean(acclist))
    plt.plot(turn_list,acclist)
    plt.show()