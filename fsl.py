import os
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torchvision.models as models
import matplotlib.pyplot as plt
import math
from torch.utils.data import Dataset, DataLoader
from easyfsl.methods import PrototypicalNetworks
from easyfsl.samplers import TaskSampler
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from tqdm import tqdm

# =====================================================
# 1. MAT loader (auto detect variable)
# =====================================================
def load_mat_feature(path):
    mat = sio.loadmat(path)
    for v in mat.values():
        if isinstance(v, np.ndarray) and v.ndim == 2:
            return v
    raise RuntimeError(f"No 2D array found in {path}")

class PeakGuidedAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, feat, energy_map):
        """
        feat: [B, C, H, W]   backbone feature
        energy_map: [B, 1, H0, W0]  |X|^2 from input
        """
        # resize energy map to feature size
        att = torch.nn.functional.interpolate(
            energy_map,
            size=feat.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        # peak normalization
        att = att / (att.amax(dim=(2, 3), keepdim=True) + 1e-8)

        # apply attention
        return feat * att
# =====================================================
# 2. Dataset (EasyFSL required get_labels)
# =====================================================
class RadarDataset(Dataset):
    def __init__(self, root):
        self.samples = []
        self.labels = []

        for cls in sorted(os.listdir(root)):
            cls_dir = os.path.join(root, cls)
            for f in os.listdir(cls_dir):
                if f.endswith(".mat"):
                    self.samples.append((os.path.join(cls_dir, f), int(cls)))
                    self.labels.append(int(cls))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        x = load_mat_feature(path)
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        P = np.abs(x) ** 2
        snr = P / (P.mean())
        snr_db = 10 * np.log10(snr)
        # snr_db = snr
        snr_min = snr_db.min()
        snr_max = snr_db.max()
        #snr_norm = (snr_db - snr_min) / (snr_max - snr_min)
        return snr_db, label

    # EasyFSL REQUIRED
    def get_labels(self):
        return self.labels

class RadarDataset_lg(Dataset):
    ##给log化后的SNR图使用
    def __init__(self, root):
        self.samples = []
        self.labels = []

        for cls in sorted(os.listdir(root)):
            cls_dir = os.path.join(root, cls)
            for f in os.listdir(cls_dir):
                if f.endswith(".mat"):
                    self.samples.append((os.path.join(cls_dir, f), int(cls)))
                    self.labels.append(int(cls))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        x = load_mat_feature(path)
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        #x = 10**x
        snr_min = x.min()
        snr_max = x.max()
        snr_norm = (x - snr_min) / (snr_max - snr_min)
        return x, label

    # EasyFSL REQUIRED
    def get_labels(self):
        return self.labels

# =====================================================
# 3. ResNet18 Backbone (Embedding Network)
# =====================================================
class ResNetBackbone(nn.Module):
    def __init__(self, emb_dim=128):
        super().__init__()
        base = models.resnet18(pretrained=False)
        base.conv1 = nn.Conv2d(
            1, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.encoder = nn.Sequential(*list(base.children())[:-1])
        self.fc = nn.Linear(512, emb_dim)

    def forward(self, x):
        x = self.encoder(x)
        return self.fc(x.flatten(1))

class ResNetBackbone_attention(nn.Module):
    def __init__(self, emb_dim=128):
        super().__init__()
        base = models.resnet34(pretrained=False)
        base.conv1 = nn.Conv2d(
            1, 64, kernel_size=3, stride=1, padding=2, bias=False
        )

        # 拆 encoder，方便插注意力
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool

        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4

        self.att = PeakGuidedAttention()

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, emb_dim)

    def forward(self, x):
        # x: [B, 1, H, W]
        energy = x ** 2   # 峰值能量图（物理先验）

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)

        # ===== 在中层加入峰值引导注意力 =====
        x = self.att(x, energy)

        x = self.layer3(x)
        x = self.layer4(x)

        x = self.pool(x)
        return self.fc(x.flatten(1))


# =====================================================
# 4. EasyFSL Episodic Loader
# =====================================================
def make_fsl_loader(dataset, n_way, n_shot, n_query, n_tasks):
    sampler = TaskSampler(
        dataset,
        n_way=n_way,
        n_shot=n_shot,
        n_query=n_query,
        n_tasks=n_tasks,
    )
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=sampler.episodic_collate_fn,
    )


# =====================================================
# 5. Evaluation (Accuracy + CM + ROC)
# =====================================================
@torch.no_grad()
def evaluate(model, loader, device, n_classes):
    model.eval()
    all_preds, all_labels, all_scores = [], [], []

    for batch in tqdm(loader, desc="Testing"):
        support_x, support_y, query_x, query_y, _ = batch
        support_x, support_y = support_x.to(device), support_y.to(device)
        query_x, query_y = query_x.to(device), query_y.to(device)

        model.process_support_set(support_x, support_y)
        logits = model(query_x)

        preds = logits.argmax(1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(query_y.cpu().numpy())
        all_scores.extend(logits.softmax(dim=1).cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_scores = np.array(all_scores)

    acc = np.mean(all_preds == all_labels)
    print(f"\nFinal Accuracy: {acc*100:.2f}%")

    # -------- Confusion Matrix --------
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure()
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for i in range(n_classes):
        for j in range(n_classes):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.show()

    # -------- ROC --------
    y_bin = label_binarize(all_labels, classes=list(range(n_classes)))
    plt.figure()
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], all_scores[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"Class {i} (AUC={roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], "--")
    plt.legend()
    plt.title("ROC Curves")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.show()


# =====================================================
# 6. Main Pipeline
# =====================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()

    # ================= Stage 1: 2-class =================
    print("==== Stage 1: 2-class Training ====")
    train2 = RadarDataset(r"E:\yan\外辐射源数据集-雷达学报\RD\train")

    train_loader2 = make_fsl_loader(
        train2, n_way=2, n_shot=25, n_query=25, n_tasks=1000
    )

    backbone = ResNetBackbone_attention().to(device)
    model = PrototypicalNetworks(backbone).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    model.train()
    for batch in tqdm(train_loader2, desc="Stage1 Training"):

        support_x, support_y, query_x, query_y, _ = batch
        support_x, support_y = support_x.to(device), support_y.to(device)
        query_x, query_y = query_x.to(device), query_y.to(device)

        model.process_support_set(support_x, support_y)
        logits = model(query_x)
        loss = criterion(logits, query_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    torch.save(backbone.state_dict(), "stage1_backbone_easyfsl_0.pth")

    # ================= Stage 2: 3-class =================
    print("\n==== Stage 2: 3-class Transfer Training ====")
    train3 = RadarDataset_lg(r"E:\yan\rd\种类识别\test")
    test3 = RadarDataset_lg(r"E:\yan\rd\种类识别\train")

    train_loader3 = make_fsl_loader(
        train3, n_way=3, n_shot=3, n_query=3, n_tasks=1000
    )
    test_loader3 = make_fsl_loader(
        test3, n_way=3, n_shot=5, n_query=5, n_tasks=200
    )

    backbone.load_state_dict(torch.load("stage1_backbone_easyfsl_0.pth"))
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    model.train()
    for batch in tqdm(train_loader3, desc="Stage2 Training"):
        support_x, support_y, query_x, query_y, _ = batch
        support_x, support_y = support_x.to(device), support_y.to(device)
        query_x, query_y = query_x.to(device), query_y.to(device)

        model.process_support_set(support_x, support_y)
        logits = model(query_x)
        loss = criterion(logits, query_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # ================= Final Test =================
    print("\n==== Final 3-class Testing ====")
    evaluate(model, test_loader3, device, n_classes=3)


if __name__ == "__main__":
    main()
