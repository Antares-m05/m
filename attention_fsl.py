import os
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import matplotlib.pyplot as plt
import random
from torch.utils.data import Dataset, DataLoader
from easyfsl.methods import PrototypicalNetworks
from easyfsl.samplers import TaskSampler
from sklearn.metrics import confusion_matrix, roc_curve, auc
from tqdm import tqdm
import math
def load_mat_feature(path):
    mat = sio.loadmat(path)
    for v in mat.values():
        if isinstance(v, np.ndarray) and v.ndim == 2:
            return v
    raise RuntimeError(f"No 2D RD matrix in {path}")

class RadarDataset(Dataset):
    def __init__(self, root):
        self.samples, self.labels = [], []
        self.Data = []
        for cls in sorted(os.listdir(root)):
            cls_dir = os.path.join(root, cls)
            for f in os.listdir(cls_dir):
                if f.endswith(".mat"):
                    self.samples.append((os.path.join(cls_dir, f), int(cls)))

                    x = load_mat_feature((os.path.join(cls_dir, f)))
                    x = torch.tensor(x, dtype=torch.float32)
                    x = x[:,175:225]
                    # x = x.repeat(10, 1)
                    P = x.abs() ** 2
                    snr = P
                    snr = (snr - snr.min()) / (snr.max() - snr.min())
                    self.Data.append(snr.unsqueeze(0))
                    self.labels.append(int(cls))
                    snr2 =torch.flip(snr, [0])
                    self.Data.append(snr2.unsqueeze(0))
                    self.labels.append(int(cls))
                    snr3 = torch.flip(snr, [1])
                    self.Data.append(snr3.unsqueeze(0))
                    self.labels.append(int(cls))
                    snr4= torch.flip(snr3, [0])
                    self.Data.append(snr4.unsqueeze(0))

                    self.labels.append(int(cls))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data = self.Data[idx]
        label = self.labels[idx]
        return data, label

    def get_labels(self):
        return self.labels


class RDGlobalAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        d_k = channels // reduction

        self.q_proj = nn.Conv1d(channels, d_k, kernel_size=1)
        self.k_proj = nn.Conv1d(channels, d_k, kernel_size=1)
        self.v_proj = nn.Conv1d(channels, channels, kernel_size=1)

        self.scale = math.sqrt(d_k)
        self.lambda_phy = nn.Parameter(torch.tensor(1.0))

        # 添加输出变换
        self.out_proj = nn.Conv1d(channels, channels, kernel_size=1)

        # 残差连接的权重
        self.gamma = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        """
        x: [B, C, R, D]  RD feature map
        """
        B, C, R, D = x.shape

        # 1. 距离维压缩
        x_d = x.mean(dim=2)  # [B, C, D]

        # 2. 计算 Q, K, V
        Q = self.q_proj(x_d)  # [B, d_k, D]
        K = self.k_proj(x_d)  # [B, d_k, D]
        V = self.v_proj(x_d)  # [B, C, D]

        # 3. 计算注意力权重 - 修复的einsum
        # 方法1：直接计算 [B, D, D]
        Q_t = Q.transpose(1, 2)  # [B, D, d_k]
        K_t = K.transpose(1, 2)  # [B, D, d_k]
        attn = torch.matmul(Q_t, K_t.transpose(1, 2)) / self.scale  # [B, D, D]

        # 或者用einsum（两种等价写法）：
        # attn = torch.einsum("bkd,bld->bdl", Q, K) / self.scale
        # attn = torch.einsum("bdk,blk->bdl", Q_t, K_t) / self.scale

        # 4. 物理引导（如果需要）
        # energy = x_d.pow(2).sum(dim=1)  # [B, D]
        # p = energy / (energy.sum(dim=1, keepdim=True) + 1e-6)
        # attn = attn + self.lambda_phy * torch.log(p.unsqueeze(1) + 1e-6)

        # 5. Softmax归一化
        attn = torch.softmax(attn, dim=-1)  # 在最后一个维度归一化

        # 6. 应用注意力到V
        V_t = V.transpose(1, 2)  # [B, D, C]
        out = torch.matmul(attn, V_t)  # [B, D, C]
        out = out.transpose(1, 2)  # [B, C, D]

        # 输出变换
        out = self.out_proj(out)  # [B, C, D]

        # 7. 广播回原始形状
        out = out.unsqueeze(2).expand(-1, -1, R, -1)  # [B, C, R, D]

        # 残差连接
        return x + self.gamma * out
class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        # 仍然只输出 1 个空间注意力图
        self.conv = nn.Conv2d(3, 1, kernel_size=2, padding=1, bias=False)

    def forward(self, x):
        """
        x: [B, C, H, W]
        return: [B, C, H, W]
        """

        # 1. 通道平均（整体能量）
        avg_out = torch.mean(x, dim=1, keepdim=True)   # [B,1,H,W]

        # 2. 通道最大（强散射点）
        max_out, _ = torch.max(x, dim=1, keepdim=True) # [B,1,H,W]

        # 3. 局部峰值对比（关键改进）
        # 表示：当前点相对邻域是否“突出”
        local_mean = nn.functional.avg_pool2d(
            avg_out, kernel_size=5, stride=1, padding=2
        )
        peak_contrast = avg_out - local_mean  # [B,1,H,W]

        # 4. 拼接注意力线索
        attn_feat = torch.cat(
            [avg_out, max_out, peak_contrast], dim=1
        )  # [B,3,H,W]

        # 5. 生成空间注意力
        attn = torch.sigmoid(self.conv(attn_feat))  # [B,1,H,W]

        # 6. 保存用于可视化（不参与梯度）
        self.last_attention = attn.detach()

        # 7. 空间加权（接口不变）
        return x * attn


class ResNetWithAttention(nn.Module):
    def __init__(self, emb_dim=128):
        super().__init__()
        base = models.resnet34(pretrained=False)
        base.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)

        self.encoder = nn.Sequential(*list(base.children())[:-2])
        self.attn = RDGlobalAttention(512)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, emb_dim)

    def forward(self, x):
        x = self.encoder(x)
        # print(x.size())
        x = self.attn(x)
        x = self.pool(x)
        return self.fc(x.flatten(1))

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

def train_fsl(model, loader, device):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    for batch in tqdm(loader, desc="Training"):
        sx, sy, qx, qy, _ = batch
        sx, sy = sx.to(device), sy.to(device)
        qx, qy = qx.to(device), qy.to(device)

        model.process_support_set(sx, sy)
        qx = F.dropout(qx,random.uniform(0.01,0.05))
        logits = model(qx)
        loss = criterion(logits, qy)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

@torch.no_grad()
def test_fsl(model, loader, device):
    model.eval()
    all_preds, all_labels, all_scores = [], [], []

    for batch in tqdm(loader, desc="Testing"):
        sx, sy, qx, qy, _ = batch
        sx, sy = sx.to(device), sy.to(device)
        qx = qx.to(device)

        model.process_support_set(sx, sy)
        logits = model(qx)
        probs = logits.softmax(dim=1)

        preds = probs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(qy.numpy())
        all_scores.extend(probs[:, 1].cpu().numpy())

    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    print(f"\nFinal Accuracy: {acc*100:.2f}%")

    # ---- Confusion Matrix ----
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure()
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("True")

    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max()/2 else "black")
    plt.show()

    # ---- ROC ----
    fpr, tpr, _ = roc_curve(all_labels, all_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.2f}")
    plt.plot([0,1],[0,1],'--')
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()

def visualize_attention(model, sample, title):
    model.eval()
    with torch.no_grad():
        _ = model(sample)

    rd = sample[0, 0].cpu().numpy()
    rd_norm = (rd - rd.min()) / (rd.max() - rd.min() + 1e-8)

    attn = model.attn.last_attention
    attn_up = F.interpolate(attn, size=rd.shape, mode="bilinear", align_corners=False)
    attn_up = attn_up[0, 0].cpu().numpy()

    overlay = rd_norm * attn_up

    plt.figure(figsize=(12,6))
    for i,(img,name) in enumerate([(rd,"RD"),(attn_up,"Attention"),(overlay,"RD×Attn")]):
        plt.subplot(1,3,i+1)
        plt.imshow(img, aspect=1, cmap="jet")
        plt.title(f"{title} {name}")
        plt.colorbar()
    plt.show()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_set = RadarDataset(r"E:\yan\外辐射源数据集-雷达学报\RD\train")
    test_set  = RadarDataset(r"E:\yan\外辐射源数据集-雷达学报\RD\test")

    train_loader = make_fsl_loader(train_set, 2, 25, 25, 500)
    test_loader  = make_fsl_loader(test_set, 2, 5, 5, 200)

    backbone = ResNetWithAttention().to(device)
    model = PrototypicalNetworks(backbone).to(device)

    # 固定样本用于 attention 对比
    sample, _ = test_set[0]
    sample = sample.unsqueeze(0).to(device)

    # print("=== Attention BEFORE Training ===")
    # visualize_attention(backbone, sample, "Before")
    #
    train_fsl(model, train_loader, device)
    #
    # print("=== Attention AFTER Training ===")
    # visualize_attention(backbone, sample, "After")

    test_fsl(model, test_loader, device)
if __name__ == "__main__":
    main()

