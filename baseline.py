import os
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

# =====================
# Dataset
# =====================
class RadarMatDataset(Dataset):
    def __init__(self, root):
        self.samples = []
        self.labels = []

        for label in sorted(os.listdir(root)):
            class_dir = os.path.join(root, label)
            if not os.path.isdir(class_dir):
                continue
            for f in os.listdir(class_dir):
                if f.endswith(".mat"):
                    self.samples.append(os.path.join(class_dir, f))
                    self.labels.append(int(label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        mat = sio.loadmat(self.samples[idx])
        for v in mat.values():
            if isinstance(v, np.ndarray) and v.ndim == 2:
                x = v.astype(np.float32)
                break
        x = torch.from_numpy(x).unsqueeze(0)  # [1,H,W]
        return x, self.labels[idx]


# =====================
# Models
# =====================
class VGGSmall(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        return self.fc(x.view(x.size(0), -1))


def build_resnet18(num_classes=3):
    model = models.resnet18(pretrained=False)
    model.conv1 = nn.Conv2d(1, 64, 3, 1, 3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# =====================
# Train / Test
# =====================
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    loss_sum = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
    return loss_sum / len(loader)


def test_model(model, loader, device, num_classes):
    model.eval()
    y_true, y_pred, y_score = [], [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            out = model(x)
            prob = torch.softmax(out, dim=1)

            y_true.extend(y.numpy())
            y_pred.extend(prob.argmax(1).cpu().numpy())
            y_score.extend(prob.cpu().numpy())

    y_true = np.array(y_true)
    y_score = np.array(y_score)
    y_pred = np.array(y_pred)

    acc = accuracy_score(y_true, y_pred)

    auc_list = []
    y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        auc_list.append(auc(fpr, tpr))

    return acc, np.mean(auc_list)


# =====================
# Main Experiment Loop
# =====================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_root = r"E:\yan\rd\种类识别\train"
    test_root = r"E:\yan\rd\种类识别\test"

    train_set = RadarMatDataset(train_root)
    test_set = RadarMatDataset(test_root)

    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=16)

    num_runs = 5      # <<< 可修改
    num_epochs = 20
    num_classes = 3

    acc_list, auc_list = [], []

    for run in range(num_runs):
        print(f"\n========== Run {run+1}/{num_runs} ==========")

        # model = VGGSmall(num_classes)
        model = build_resnet18(num_classes)
        model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=3e-4)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(num_epochs):
            loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {loss:.4f}")

        acc, mean_auc = test_model(model, test_loader, device, num_classes)
        acc_list.append(acc)
        auc_list.append(mean_auc)

        print(f"Run {run+1} Accuracy: {acc*100:.2f}% | Mean AUC: {mean_auc:.3f}")

    print("\n========== Final Statistics ==========")
    print(f"Accuracy: mean={np.mean(acc_list)*100:.2f}%, std={np.std(acc_list)*100:.2f}%")
    print(f"AUC: mean={np.mean(auc_list):.3f}, std={np.std(auc_list):.3f}")


if __name__ == "__main__":
    main()
