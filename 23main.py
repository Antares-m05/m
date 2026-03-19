import os
import random
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

# =====================================================
# 1. Robust MAT Loader (auto variable name)
# =====================================================
def load_mat_feature(mat_path):
    mat = sio.loadmat(mat_path)
    for k, v in mat.items():
        if isinstance(v, np.ndarray) and v.ndim == 2:
            return v
    raise ValueError(f"No valid 2D feature in {mat_path}")


# =====================================================
# 2. Dataset
# =====================================================
class RadarMatDataset:
    def __init__(self, root):
        self.samples = []
        self.class_to_indices = {}

        for name in sorted(os.listdir(root)):
            label = int(name)
            self.class_to_indices[label] = []
            class_dir = os.path.join(root, name)

            for f in os.listdir(class_dir):
                if f.endswith(".mat"):
                    self.class_to_indices[label].append(len(self.samples))
                    self.samples.append((os.path.join(class_dir, f), label))

    def load(self, idx):
        path, label = self.samples[idx]
        x = load_mat_feature(path)
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        return x, label


# =====================================================
# 3. ResNet Backbone (Embedding Network)
# =====================================================
class ResNetBackbone(nn.Module):
    def __init__(self, emb_dim=128):
        super().__init__()
        base = models.resnet18(pretrained=False)
        base.conv1 = nn.Conv2d(
            1, 64, kernel_size=5, stride=2, padding=3, bias=False
        )
        self.encoder = nn.Sequential(*list(base.children())[:-1])
        self.fc = nn.Linear(512, emb_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = x.flatten(1)
        return self.fc(x)


# =====================================================
# 4. ProtoNet Core
# =====================================================
def euclidean_dist(a, b):
    return ((a[:, None] - b[None]) ** 2).sum(-1)


def prototypical_forward(model, sx, sy, qx):
    emb_s = model(sx)
    emb_q = model(qx)

    classes = torch.unique(sy)
    prototypes = torch.stack(
        [emb_s[sy == c].mean(0) for c in classes]
    )
    dists = euclidean_dist(emb_q, prototypes)
    probs = F.softmax(-dists, dim=1)

    return probs, classes


def prototypical_loss(model, sx, sy, qx, qy):
    probs, classes = prototypical_forward(model, sx, sy, qx)
    targets = torch.tensor(
        [torch.where(classes == y)[0].item() for y in qy],
        device=probs.device
    )
    loss = F.nll_loss(torch.log(probs), targets)
    acc = (probs.argmax(1) == targets).float().mean()
    return loss, acc


# =====================================================
# 5. Episode Sampler
# =====================================================
def sample_episode(dataset, k_shot, q_query):
    sx, sy, qx, qy = [], [], [], []

    for cls, idxs in dataset.class_to_indices.items():
        chosen = random.sample(idxs, k_shot + q_query)

        for i in chosen[:k_shot]:
            x, y = dataset.load(i)
            sx.append(x); sy.append(y)

        for i in chosen[k_shot:]:
            x, y = dataset.load(i)
            qx.append(x); qy.append(y)

    return (
        torch.stack(sx), torch.tensor(sy),
        torch.stack(qx), torch.tensor(qy)
    )


# =====================================================
# 6. Evaluation with progress
# =====================================================
@torch.no_grad()
def evaluate_full(model, dataset, device, episodes=200, k_shot=5, q_query=5):
    model.eval()
    all_preds, all_labels, all_scores = [], [], []

    for ep in range(episodes):
        sx, sy, qx, qy = sample_episode(dataset, k_shot, q_query)
        sx, sy = sx.to(device), sy.to(device)
        qx, qy = qx.to(device), qy.to(device)

        probs, classes = prototypical_forward(model, sx, sy, qx)
        preds = classes[probs.argmax(1)]

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(qy.cpu().numpy())
        all_scores.extend(probs.cpu().numpy())

        if (ep + 1) % 50 == 0 or ep == 0:
            print(f"[Test] Episode {ep+1}/{episodes}")

    return (
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_scores),
        classes.cpu().numpy()
    )


# =====================================================
# 7. Plotting
# =====================================================
def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xticks(range(len(classes)), classes)
    plt.yticks(range(len(classes)), classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")

    for i in range(len(classes)):
        for j in range(len(classes)):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.show()


def plot_roc(y_true, scores, classes):
    y_bin = label_binarize(y_true, classes=classes)

    for i, c in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], scores[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"Class {c} (AUC={roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    plt.show()


# =====================================================
# 8. Main
# =====================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------- Stage 1: 2-class --------
    print("==== Stage 1: 2-class Training ====")
    train2 = RadarMatDataset(
        r"E:\yan\外辐射源数据集-雷达学报\RD\train"
    )

    model = ResNetBackbone(128).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)

    total_ep1 = 800
    for ep in range(total_ep1):
        sx, sy, qx, qy = sample_episode(train2, 5, 5)
        sx, sy = sx.to(device), sy.to(device)
        qx, qy = qx.to(device), qy.to(device)

        loss, acc = prototypical_loss(model, sx, sy, qx, qy)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if (ep + 1) % 50 == 0 or ep == 0:
            print(
                f"[Stage1] Episode {ep+1}/{total_ep1} | "
                f"Loss: {loss.item():.4f} | Acc: {acc.item():.4f}"
            )

    torch.save(model.state_dict(), "stage1_resnet.pth")

    # -------- Stage 2: 3-class --------
    print("\n==== Stage 2: 3-class Transfer Training ====")
    train3 = RadarMatDataset(r"E:\yan\rd\种类识别\test")
    test3 = RadarMatDataset(r"E:\yan\rd\种类识别\train")

    model.load_state_dict(torch.load("stage1_resnet.pth"))
    opt = torch.optim.Adam(model.parameters(), lr=5e-5)

    total_ep2 = 500
    for ep in range(total_ep2):
        sx, sy, qx, qy = sample_episode(train3, 5, 5)
        sx, sy = sx.to(device), sy.to(device)
        qx, qy = qx.to(device), qy.to(device)

        loss, acc = prototypical_loss(model, sx, sy, qx, qy)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if (ep + 1) % 50 == 0 or ep == 0:
            print(
                f"[Stage2] Episode {ep+1}/{total_ep2} | "
                f"Loss: {loss.item():.4f} | Acc: {acc.item():.4f}"
            )

    # -------- Final Evaluation --------
    print("\n==== Final 3-class Testing ====")
    y_true, y_pred, scores, classes = evaluate_full(
        model, test3, device, episodes=200
    )

    acc = np.mean(y_true == y_pred)
    print(f"\nFinal Accuracy: {acc*100:.2f}%")

    plot_confusion_matrix(y_true, y_pred, classes)
    plot_roc(y_true, scores, classes)


if __name__ == "__main__":
    main()
