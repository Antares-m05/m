import os
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from easyfsl.methods import PrototypicalNetworks
from easyfsl.samplers import TaskSampler
from sklearn.metrics import confusion_matrix, roc_curve, auc
from tqdm import tqdm


# =====================================================
# 1. MAT loader (auto-detect 2D RD matrix)
# =====================================================
def load_mat_feature(path):
    mat = sio.loadmat(path)
    for v in mat.values():
        if isinstance(v, np.ndarray) and v.ndim == 2:
            return v
    raise RuntimeError(f"No valid 2D matrix in {path}")


# =====================================================
# 2. Radar Dataset (SNR + Min-Max Norm)
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

        x = torch.tensor(x, dtype=torch.float32)
        P = x.abs()
        snr = P / (P.mean() + 1e-8)

        snr = (snr - snr.min()) / (snr.max() - snr.min() + 1e-8)
        snr = snr.unsqueeze(0)   # [1, 5, 400]

        return snr, label

    # EasyFSL requires this
    def get_labels(self):
        return self.labels


# =====================================================
# 3. Classic Spatial Attention (CBAM-SA)
# =====================================================
class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=3, padding=1)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.cat([avg_out, max_out], dim=1)
        attn = torch.sigmoid(self.conv(attn))
        self.last_attention = attn.detach()
        return x * attn


# =====================================================
# 4. ResNet18 + Attention Backbone
# =====================================================
class ResNetWithAttention(nn.Module):
    def __init__(self, emb_dim=128):
        super().__init__()
        base = models.resnet18(pretrained=False)
        base.conv1 = nn.Conv2d(
            1, 64, kernel_size=3, stride=1, padding=1, bias=False
        )

        self.encoder = nn.Sequential(*list(base.children())[:-2])
        self.attn = SpatialAttention()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, emb_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = self.attn(x)
        x = self.pool(x)
        return self.fc(x.flatten(1))


# =====================================================
# 5. Few-Shot Episodic DataLoader
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
# 6. Training (Few-Shot)
# =====================================================
def train_fsl(model, loader, device):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()

    for batch in tqdm(loader, desc="Training"):
        support_x, support_y, query_x, query_y, _ = batch
        support_x, support_y = support_x.to(device), support_y.to(device)
        query_x, query_y = query_x.to(device), query_y.to(device)

        model.process_support_set(support_x, support_y)
        logits = model(query_x)
        loss = criterion(logits, query_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# =====================================================
# 7. Testing + Metrics
# =====================================================
@torch.no_grad()
def test_fsl(model, loader, device):
    model.eval()
    all_preds, all_labels, all_scores = [], [], []

    for batch in tqdm(loader, desc="Testing"):
        support_x, support_y, query_x, query_y, _ = batch
        support_x, support_y = support_x.to(device), support_y.to(device)
        query_x = query_x.to(device)

        model.process_support_set(support_x, support_y)
        logits = model(query_x)
        probs = logits.softmax(dim=1)

        preds = probs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(query_y.numpy())
        all_scores.extend(probs[:, 1].cpu().numpy())

    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    print(f"\nFinal Accuracy: {acc * 100:.2f}%")

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure()
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, cm[i, j],
                ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black"
            )

    plt.tight_layout()
    plt.show()

    # ROC Curve
    fpr, tpr, _ = roc_curve(all_labels, all_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()


# =====================================================
# 8. Main
# =====================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_root = r"E:\yan\外辐射源数据集-雷达学报\RD\train"
    test_root  = r"E:\yan\外辐射源数据集-雷达学报\RD\test"

    train_set = RadarDataset(train_root)
    test_set  = RadarDataset(test_root)

    train_loader = make_fsl_loader(train_set, 2, 5, 5, 800)
    test_loader  = make_fsl_loader(test_set, 2, 5, 5, 200)

    backbone = ResNetWithAttention().to(device)
    model = PrototypicalNetworks(backbone).to(device)

    train_fsl(model, train_loader, device)
    test_fsl(model, test_loader, device)


if __name__ == "__main__":
    main()
