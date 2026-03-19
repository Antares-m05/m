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

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 3 #原来是7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out
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
                    # x = x.repeat(3, 1)
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
        return len(self.Data)

    def __getitem__(self, idx):
        data = self.Data[idx]
        label = self.labels[idx]
        return data, label

    def get_labels(self):
        return self.labels


class ResNetWithAttention(nn.Module):
    def __init__(self, emb_dim=128):
        super().__init__()
        base = models.resnet34(pretrained=False)
        base.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)

        self.encoder = nn.Sequential(*list(base.children())[:-2])
        # self.attn = ChannelAttention(512)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, emb_dim)
        self.cbam = CBAM(512,2)

    def forward(self, x):
        x = self.encoder(x)
        # print(x.size())
        x = self.cbam(x)
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

