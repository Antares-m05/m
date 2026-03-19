import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from scipy.io import loadmat
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from easyfsl.methods import PrototypicalNetworks
from easyfsl.samplers import TaskSampler
import random

# =====================
# 0. 固定随机种子
# =====================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# =====================
# 1. 参数配置（few-shot）
# =====================
DATA_ROOT = r"E:\yan\新科楼顶数据集\切片16x16"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_WAY = 4
N_SHOT = 5
N_QUERY = 5
N_TRAIN_TASKS = 500
N_TEST_TASKS = 200
LR = 3e-4

# =====================
# 2. Dataset（保持你原始逻辑）
# =====================
class RadarDataset(Dataset):
    def __init__(self, root_dir):
        self.data = []
        self.labels = []

        for label in ["0", "1", "2","3"]:
            class_dir = os.path.join(root_dir, label)
            for file in os.listdir(class_dir):
                if file.endswith(".mat"):
                    path = os.path.join(class_dir, file)
                    mat = loadmat(path)

                    for k in mat.keys():
                        if not k.startswith("__"):
                            x = mat[k]
                            break
                    x = torch.tensor(np.exp(np.log(abs(x)))).float()
                    snr = x
                    cls = label
                    self.data.append(torch.tensor(x.unsqueeze(0), dtype=torch.float32))
                    self.labels.append(int(label))

                    self.data.append(snr.unsqueeze(0))
                    self.labels.append(int(cls))
                    snr2 = torch.flip(snr, [0])
                    self.data.append(snr2.unsqueeze(0))
                    self.labels.append(int(cls))
                    snr3 = torch.flip(snr, [1])
                    self.data.append(snr3.unsqueeze(0))
                    self.labels.append(int(cls))
                    snr4 = torch.flip(snr3, [0])
                    self.data.append(snr4.unsqueeze(0))

                    self.labels.append(int(cls))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def get_labels(self):
        return self.labels

# =====================
# 3. Backbone（轻量 CNN，适合 few-shot）
# =====================
class ResNetBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        model = models.resnet18(pretrained=False)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        model.maxpool = nn.Identity()
        self.encoder = nn.Sequential(*list(model.children())[:-1])

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)

# =====================
# 4. Few-shot DataLoader
# =====================
def make_fsl_loader(dataset, n_tasks):
    sampler = TaskSampler(
        dataset,
        n_way=N_WAY,
        n_shot=N_SHOT,
        n_query=N_QUERY,
        n_tasks=n_tasks,
    )
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=sampler.episodic_collate_fn,
    )

# =====================
# 5. 训练
# =====================
def train_fsl(model, loader):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    for sx, sy, qx, qy, _ in loader:
        sx, sy = sx.to(DEVICE), sy.to(DEVICE)
        qx, qy = qx.to(DEVICE), qy.to(DEVICE)

        model.process_support_set(sx, sy)
        logits = model(qx)
        loss = criterion(logits, qy)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# =====================
# 6. 测试 + 混淆矩阵
# =====================
@torch.no_grad()
def test_fsl(model, loader):
    model.eval()
    all_preds, all_labels = [], []

    for sx, sy, qx, qy, _ in loader:
        sx, sy = sx.to(DEVICE), sy.to(DEVICE)
        qx = qx.to(DEVICE)

        model.process_support_set(sx, sy)
        logits = model(qx)
        preds = logits.argmax(dim=1).cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(qy.numpy())

    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    cm = confusion_matrix(all_labels, all_preds)
    return acc, cm

# =====================
# 7. 主流程
# =====================
def main():
    set_seed(42)

    train_set = RadarDataset(os.path.join(DATA_ROOT, "train"))
    test_set  = RadarDataset(os.path.join(DATA_ROOT, "test"))

    train_loader = make_fsl_loader(train_set, N_TRAIN_TASKS)
    test_loader  = make_fsl_loader(test_set, N_TEST_TASKS)

    backbone = ResNetBackbone().to(DEVICE)
    model = PrototypicalNetworks(backbone).to(DEVICE)

    train_fsl(model, train_loader)
    acc, cm = test_fsl(model, test_loader)

    print(f"Test Accuracy: {acc:.4f}")
    print("Confusion Matrix:\n", cm)

    return acc

# =====================
# 8. 多次运行（论文标准）
# =====================
if __name__ == "__main__":
    accs = []
    for seed in range(5):
        set_seed(seed)
        accs.append(main())

    print(f"\nMean Acc: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
