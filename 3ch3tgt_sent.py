#添加语义模块
##在3ch2的基础上添加可视化
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
        kernel_size = 7#原来是7  97.65%
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
                    x = torch.tensor(x)
                    xm,ym = torch.where(x==torch.max(x))
                    # x = x[xm-2:xm+3,1:51]
                    # x = x.repeat(3, 1)
                    P = math.e**(x/10)
                    snr = P.float()
                    # snr = np.log10(snr)
                    # snr = (snr - snr.min()) / (snr.max() - snr.min())
                    self.Data.append(snr.unsqueeze(0))
                    # self.labels.append(int(cls))
                    # snr2 =torch.flip(snr, [0])
                    # self.Data.append(snr2.unsqueeze(0))
                    # self.labels.append(int(cls))
                    # snr3 = torch.flip(snr, [1])
                    # self.Data.append(snr3.unsqueeze(0))
                    # self.labels.append(int(cls))
                    # snr4= torch.flip(snr3, [0])
                    # self.Data.append(snr4.unsqueeze(0))

                    self.labels.append(int(cls))

    def __len__(self):
        return len(self.Data)

    def __getitem__(self, idx):
        data = self.Data[idx]
        label = self.labels[idx]
        return data, label

    def get_labels(self):
        return self.labels

class SemanticRelation(nn.Module):
    """
    Semantic relation modeling on fused physical features
    """
    def __init__(self, channels):
        super().__init__()
        self.query = nn.Conv2d(channels, channels // 2, kernel_size=1)
        self.key   = nn.Conv2d(channels, channels // 2, kernel_size=1)
        self.value = nn.Conv2d(channels, channels, kernel_size=1)

        self.gamma = nn.Parameter(torch.zeros(1))  # learnable importance

    def forward(self, x):
        """
        x: [B, C, H, W]
        """
        B, C, H, W = x.size()

        q = self.query(x).view(B, -1, H * W).permute(0, 2, 1)  # B, HW, C'
        k = self.key(x).view(B, -1, H * W)                    # B, C', HW
        v = self.value(x).view(B, -1, H * W)                  # B, C, HW

        attn = torch.softmax(torch.bmm(q, k), dim=-1)         # B, HW, HW
        out = torch.bmm(v, attn.permute(0, 2, 1))             # B, C, HW
        out = out.view(B, C, H, W)

        return self.gamma * out + x

class ResNetBackbone(nn.Module):
    def __init__(self):
        super(ResNetBackbone, self).__init__()
        resnet = models.resnet34(pretrained=False)  # 加载训练参数
        #self.conv1 = resnet.conv1
        #
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(4,2), stride=1, padding=1)  # 改了输入的
        #
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.Avg = nn.AdaptiveAvgPool2d((2,24))
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        self._feature_dim = resnet.fc.in_features
        # resnet18.fc = nn.Linear(num_ftrs, 11)#改了输出的
        #del resnet

    def forward(self, x):
        x = self.conv1(x)
        #print(x.size())
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)
        x = self.Avg(x)
        #print(x.size())
        x = self.layer1(x)
        #print(x.size())
        x = self.layer2(x)
        #print(x.size())
        x = self.layer3(x)
        # #print(x.size())
        # x = self.layer4(x)
        # #print(x.size())
        # x = self.avgpool(x)
        # #print(x.size())
        # x = x.view(x.size(0), -1)
        #input()
        return x

    def output_num(self):
        return self._feature_dim


class PartAttention(nn.Module):
    """
    Discover K discriminative local regions for fine-grained recognition
    """
    def __init__(self, in_channels, num_parts=4):
        super().__init__()
        self.num_parts = num_parts
        self.attn = nn.Conv2d(in_channels, num_parts, kernel_size=1)

    def forward(self, x):
        """
        x: [B, C, H, W]
        """
        B, C, H, W = x.size()

        attn_maps = torch.softmax(
            self.attn(x).view(B, self.num_parts, -1),
            dim=-1
        ).view(B, self.num_parts, H, W)

        parts = []
        for i in range(self.num_parts):
            ai = attn_maps[:, i:i+1]             # [B,1,H,W]
            pi = (x * ai).sum(dim=(2,3))         # [B,C]
            parts.append(pi)

        parts = torch.cat(parts, dim=1)          # [B, C*K]
        return parts, attn_maps

class ResNetWithAttention(nn.Module):
    def __init__(self, emb_dim=128):
        super().__init__()
        self.base1 = ResNetBackbone()
        self.base1.conv1 = nn.Conv2d(1, 64, kernel_size=(2,1), stride=1, padding=1, bias=False)

        self.base2 = ResNetBackbone()
        self.base2.conv1 = nn.Conv2d(1, 64, kernel_size=(5,2), stride=1, padding=1, bias=False)

        self.base3 = ResNetBackbone()
        self.base3.conv1 = nn.Conv2d(1, 64, kernel_size=(10,3), stride=1, padding=1, bias=False)

        #self.base4 = ResNetBackbone()


        # self.attn = ChannelAttention(512)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256*3, emb_dim)  #256为xc第二维
        self.cbam = CBAM(256*3,4)  #2/4 95.6 /8
        self.semantic = SemanticRelation(256 * 3)
    def forward(self, x):
        x1 = self.base1(x)
        x2 = self.base2(x)
        x3 = self.base3(x)


        xc = torch.cat([x1,x2,x3],dim=1)

        # x = self.encoder(x)
        # print(x.size())
        xc = self.cbam(xc)
        # xc = self.semantic(xc)
        xc = self.pool(xc)
        return self.fc(xc.flatten(1))

    def forward_for_vis(self, x):
        x1 = self.base1(x)  # [B,C,H,W]
        x2 = self.base2(x)
        x3 = self.base3(x)
        return x1, x2, x3


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
        qx = F.dropout(qx,random.uniform(0.01,0.03))
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

    for i in range(3):
        for j in range(3):
            plt.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max()/2 else "black")
    plt.show()

    # ---- ROC ----
    fpr, tpr, _ = roc_curve(all_labels, all_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.2f}")
    plt.plot([0,1,2],[0,1,2],'--')
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()

    return acc

def visualize_attention(model, sample, title):
    model.eval()
    with torch.no_grad():
        _ = model(sample)

    rd = sample[0, 0].cpu().numpy()
    rd_norm = rd

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

    train_set = RadarDataset(r"E:\yan\rd\种类识别\test")
    test_set  = RadarDataset(r"E:\yan\rd\种类识别\train_92")

    train_loader = make_fsl_loader(train_set, 3, 5, 5, 500)
    test_loader  = make_fsl_loader(test_set, 3, 5, 5, 200)

    backbone = ResNetWithAttention().to(device)
    model = PrototypicalNetworks(backbone).to(device)

    # 固定样本用于 attention 对比

    sample, _ = test_set[0]
    sample = sample.unsqueeze(0).to(device)

    print("=== Visualizing RD & Branch Outputs ===")
    visualize_rd_and_branches(backbone, sample)
    #
    train_fsl(model, train_loader, device)
    #
    # print("=== Attention AFTER Training ===")
    # visualize_attention(backbone, sample, "After")
    sample, _ = test_set[0]
    sample = sample.unsqueeze(0).to(device)

    print("=== Visualizing RD & Branch Outputs ===")
    visualize_rd_and_branches(backbone, sample)
    acc = test_fsl(model, test_loader, device)
    return acc

def visualize_rd_and_branches(model, sample):
    """
    sample: [1,1,H,W]
    """
    model.eval()
    with torch.no_grad():
        x1, x2, x3 = model.forward_for_vis(sample)

    rd = sample[0, 0].cpu().numpy()

    def feature_map(feat):
        # 对 channel 做平均，得到 2D 响应
        fmap = feat.mean(dim=1)[0].cpu().numpy()
        fmap = (fmap - fmap.min()) / (fmap.max() - fmap.min() + 1e-8)
        return fmap

    f1 = feature_map(x1)
    f2 = feature_map(x2)
    f3 = feature_map(x3)

    plt.figure(figsize=(16, 4))

    imgs = [rd, f1, f2, f3]
    titles = ["Original RD", "Branch-1 (1×5)", "Branch-2 (2×5)", "Branch-3 (5×10)"]

    for i in range(4):
        plt.subplot(1, 4, i+1)
        plt.imshow(imgs[i], aspect="auto", cmap="jet")
        plt.title(titles[i])
        plt.colorbar()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    acclist = []
    turn_list = []
    turn_num = 1
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    for i in range(turn_num):
        acc_one = main()
        acclist.append(acc_one)
        turn_list.append(i + 1)
    print(np.var(acclist), np.mean(acclist))
    plt.plot(turn_list, acclist)
    plt.show()
