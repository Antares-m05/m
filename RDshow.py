import random
import math
from matplotlib.font_manager import FontProperties
import torch
import torch.nn as nn
import torch.optim as optim
import scipy
import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import models
from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
"""
修改bottleneck，和lstm层的输入和输出
先训练两个RD
再训分类器
这个是训两个RD的
"""

if __name__ == "__main__":
    data = loadmat(r"E:\yan\新科楼顶数据集\切片16x16\3\45_y1147_x300.mat")
    rd = data['data']

    rd = abs(rd) # 看图直接用rd = rd 训练math.e**(rd/10)

    plt.imshow(rd)
    plt.show()

    # np.random.seed(42)  # 保证每次生成可复现，可改为 None 获得不同结果
    #
    # n_classes = 6
    # n_samples_per_class = 80
    #
    # # 定义大类
    # group1 = [0, 1, 2, 3]  # 前四类
    # group2 = [4, 5]  # 后两类
    #
    # cm = np.zeros((n_classes, n_classes), dtype=int)
    #
    # for i in range(n_classes):
    #     # 对角线正确率随机 82%-90%
    #     correct = np.random.randint(66, 73) if i < 4 else np.random.randint(70, 72)
    #     cm[i, i] = correct
    #
    #     # 分配剩余错误数
    #     remaining = n_samples_per_class - correct
    #
    #     if i in group1:
    #         # 错误主要分配到同组内
    #         other_group = [c for c in group1 if c != i]
    #         error_distribution = np.random.multinomial(remaining, [0.8 / 3] * 3 + [0.2 / 3] * 2)
    #         # 0.8/3 分配给大类内其他3类，0.2/3 分配给大类2
    #         cm[i, other_group] = error_distribution[:3]
    #         cm[i, group2] = error_distribution[3:]
    #     else:
    #         # 对于大类2，错误主要在组内
    #         other_group = [c for c in group2 if c != i]
    #         error_distribution = np.random.multinomial(remaining, [0.1, 0.9])
    #         cm[i, other_group] = error_distribution[0]
    #         cm[i, other_group[0] + 1 if other_group[0] + 1 < n_classes else other_group[0]] = error_distribution[1]
    #
    # # 转百分比
    # cm_percent = cm / n_samples_per_class * 100
    #
    # # 创建只显示对角线的标注
    # annot = np.full_like(cm_percent, "", dtype=object)
    # for i in range(n_classes):
    #     annot[i, i] = f"{cm_percent[i, i]:.1f}%"
    #
    # labels = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6']
    #
    # plt.figure(figsize=(8, 6))
    # sns.heatmap(cm_percent, annot=annot, fmt="", cmap="Blues", xticklabels=labels, yticklabels=labels,
    #             annot_kws={"size": 11})
    # plt.xlabel("Predicted Label")
    # plt.ylabel("True Label")
    # plt.title("Random Confusion Matrix (%)")
    # plt.tight_layout()
    # plt.show()
    #
    # # 打印矩阵
    # print("Raw Confusion Matrix (counts):")
    # print(cm)