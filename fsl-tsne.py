import os
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torchvision.models as models
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from easyfsl.methods import PrototypicalNetworks
from easyfsl.samplers import TaskSampler
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE
from tqdm import tqdm

# =====================================================
# 1. MAT loader
# =====================================================
def load_mat_feature(path):
    mat = sio.loadmat(path)
    for v in mat.values():
        if isinstance(v, np.ndarray) and v.ndim == 2:
            return v
    raise RuntimeError(f"No 2D array found in {path}")


# =====================================================
# 2. Dataset
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
        P = torch.abs(x) ** 2
        snr = P / (P.mean() + 1e-8)
        return snr, label

    def get_labels(self):
        return self.labels


# =====================================================
# 3. ResNet Backbone
# =====================================================
class ResNetBackbone(nn.Module):
    def __init__(self, emb_dim=128):
        super().__init__()
        base = models.resnet18(pretrained=False)
        base.conv1 = nn.Conv2d(1, 64, 3, 1, 1, bias=False)
        self.encoder = nn.Sequential(*list(base.children())[:-1])
        self.fc = nn.Linear(512, emb_dim)

    def forward(self, x):
        x = self.encoder(x)
        return self.fc(x.flatten(1))


# =====================================================
# 4. FSL Loader
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
# 5. t-SNE Visualization
# =====================================================
@torch.no_grad()
def tsne_visualize(backbone, dataset, device, title, max_samples=300):
    backbone.eval()
    feats, labels = [], []

    for i in range(min(len(dataset), max_samples)):
        x, y = dataset[i]
        x = x.unsqueeze(0).to(device)
        z = backbone(x)
        feats.append(z.cpu().numpy()[0])
        labels.append(y)

    feats = np.array(feats)
    labels = np.array(labels)

    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate=200,
        #n_iter=1000,
        random_state=42,
    )
    z2d = tsne.fit_transform(feats)

    plt.figure(figsize=(6, 6))
    for c in np.unique(labels):
        idx = labels == c
        plt.scatter(z2d[idx, 0], z2d[idx, 1], label=f"Class {c}", s=20)
    plt.legend()
    plt.title(title)
    plt.grid(True)
    plt.show()


# =====================================================
# 6. Evaluation
# =====================================================
@torch.no_grad()
def evaluate(model, loader, device, n_classes):
    model.eval()
    all_preds, all_labels, all_scores = [], [], []

    for batch in tqdm(loader, desc="Testing"):
        sx, sy, qx, qy, _ = batch
        sx, sy = sx.to(device), sy.to(device)
        qx, qy = qx.to(device), qy.to(device)

        model.process_support_set(sx, sy)
        logits = model(qx)

        preds = logits.argmax(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(qy.cpu().numpy())
        all_scores.extend(logits.softmax(1).cpu().numpy())

    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    print(f"\nFinal Accuracy: {acc*100:.2f}%")

    cm = confusion_matrix(all_labels, all_preds)
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.show()

    y_bin = label_binarize(all_labels, classes=list(range(n_classes)))
    plt.figure()
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], np.array(all_scores)[:, i])
        plt.plot(fpr, tpr, label=f"Class {i}")
    plt.legend()
    plt.title("ROC Curves")
    plt.show()


# =====================================================
# 7. Main
# =====================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()

    # -------- Stage 1: 2-class --------
    print("==== Stage 1: 2-class Training ====")
    train2 = RadarDataset(r"E:\yan\外辐射源数据集-雷达学报\RD\train")
    loader2 = make_fsl_loader(train2, 2, 5, 5, 800)

    backbone = ResNetBackbone().to(device)
    model = PrototypicalNetworks(backbone).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    model.train()
    for batch in tqdm(loader2):
        sx, sy, qx, qy, _ = batch
        sx, sy = sx.to(device), sy.to(device)
        qx, qy = qx.to(device), qy.to(device)

        model.process_support_set(sx, sy)
        loss = criterion(model(qx), qy)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("\n=== t-SNE after 2-class training ===")
    tsne_visualize(backbone, train2, device, "2-Class Feature Space")

    # -------- Stage 2: 3-class --------
    print("\n==== Stage 2: 3-class Transfer ====")
    train3 = RadarDataset(r"E:\yan\rd\种类识别\train")
    test3 = RadarDataset(r"E:\yan\rd\种类识别\test")

    loader3 = make_fsl_loader(train3, 3, 5, 5, 500)
    test_loader3 = make_fsl_loader(test3, 3, 5, 5, 200)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    model.train()
    for batch in tqdm(loader3):
        sx, sy, qx, qy, _ = batch
        sx, sy = sx.to(device), sy.to(device)
        qx, qy = qx.to(device), qy.to(device)

        model.process_support_set(sx, sy)
        loss = criterion(model(qx), qy)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("\n=== t-SNE after 3-class transfer ===")
    tsne_visualize(backbone, test3, device, "3-Class Feature Space")

    print("\n==== Final Evaluation ====")
    evaluate(model, test_loader3, device, 3)


if __name__ == "__main__":
    main()
