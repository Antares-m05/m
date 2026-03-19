import os
import scipy.io as sio
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

class RDDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        self.labels = []

        for label in [0, 1]:
            folder = os.path.join(root_dir, str(label))
            for file in os.listdir(folder):
                if file.endswith('.mat'):
                    self.samples.append(os.path.join(folder, file))
                    self.labels.append(label)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        mat = sio.loadmat(self.samples[idx])
        rd = mat['sliceData']  # shape: (5, 200)

        rd = torch.tensor(rd, dtype=torch.float32)
        rd = rd.unsqueeze(0)  # (1, 5, 200)

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return rd, label

class StrongScatterBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(2, 2), padding=(0, 2)),
            nn.ReLU(),
            nn.MaxPool2d((1, 2))
        )

    def forward(self, x):
        return self.conv(x)

class StructuralBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(3, 7), padding=(1, 3)),
            nn.ReLU(),
            nn.MaxPool2d((1, 2))
        )

    def forward(self, x):
        return self.conv(x)

class SpatialBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(5, 15), padding=(2, 7)),
            nn.ReLU()
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 2))

    def forward(self, x):
        x = self.conv(x)
        return self.gap(x)

class RDPhysNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.b1 = StrongScatterBranch()
        self.b2 = StructuralBranch()
        self.b3 = SpatialBranch()

        self.fc = nn.Sequential(
            nn.Linear(14440, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        f1 = self.b1(x)
        f2 = self.b2(x)
        f3 = self.b3(x)

        f1f = f1.flatten(1)
        f2f = f2.flatten(1)
        f3f = f3.flatten(1)

        feat = torch.cat([f1f, f2f, f3f], dim=1)
        out = self.fc(feat)
        return out, f1, f2, f3


class BranchAttention(nn.Module):
    """
    Branch-level attention for physically meaningful branches
    """
    def __init__(self, in_channels, reduction=4):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, 1)
        )

    def forward(self, features):
        """
        features: list of feature maps [F1, F2, F3]
                  each shape: (B, C, H, W)
        """
        weights = []

        for f in features:
            # Global descriptor of each branch
            g = F.adaptive_avg_pool2d(f, 1).flatten(1)  # (B, C)
            w = self.fc(g)                              # (B, 1)
            weights.append(w)

        # Normalize weights across branches
        weights = torch.softmax(torch.cat(weights, dim=1), dim=1)  # (B, 3)

        # Weighted fusion
        fused = 0
        for i, f in enumerate(features):
            fused = fused + weights[:, i].view(-1, 1, 1, 1) * f

        return fused, weights


class RDPhysNet_Att(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        self.b1 = StrongScatterBranch()
        self.b2 = StructuralBranch()
        self.b3 = SpatialBranch()

        self.branch_att = BranchAttention(in_channels=8)

        self.classifier = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        # ----- branch feature extraction -----
        f1 = self.b1(x)   # (B,8,H,W)
        f2 = self.b2(x)   # (B,8,H,W)
        f3 = self.b3(x)   # (B,8,H,W)

        # ----- branch attention fusion -----
        fused, weights = self.branch_att([f1, f2, f3])

        # ----- classification -----
        g = F.adaptive_avg_pool2d(fused, 1).flatten(1)  # (B,8)
        out = self.classifier(g)

        return out, weights, f1, f2, f3

if __name__ == "__main__":

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset = RDDataset(r"E:\yan\外辐射源数据集-雷达学报\RD\train")
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    # model = RDPhysNet().to(device)
    model = RDPhysNet_Att().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 10
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        for rd, label in loader:
            rd, label = rd.to(device), label.to(device)

            optimizer.zero_grad()
            out, _, _, _ = model(rd)
            loss = criterion(out, label)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{epochs}] Loss: {total_loss / len(loader):.4f}")

    model.eval()
    rd, _ = dataset[0]
    rd = rd.unsqueeze(0).to(device)

    with torch.no_grad():
        _, f1, f2, f3 = model(rd)

    rd = rd.cpu().squeeze().numpy()
    f1 = f1.cpu().mean(1).squeeze().numpy()
    f2 = f2.cpu().mean(1).squeeze().numpy()
    f3 = f3.cpu().squeeze().numpy()



    # plt.figure(figsize=(16, 3))
    #
    # plt.subplot(1, 4, 1)
    # plt.title("Original RD")
    # plt.imshow(rd, aspect='auto')
    # plt.colorbar()
    #
    # plt.subplot(1, 4, 2)
    # plt.title("Branch 1: Strong Scatter")
    # plt.imshow(f1, aspect='auto')
    # plt.colorbar()
    #
    # plt.subplot(1, 4, 3)
    # plt.title("Branch 2: Structure")
    # plt.imshow(f2, aspect='auto')
    # plt.colorbar()
    #
    # plt.subplot(1, 4, 4)
    # plt.title("Branch 3: Global Spatial Feature")
    # plt.imshow(f3.repeat(5,0).reshape(8,5), aspect='auto')
    # plt.show()