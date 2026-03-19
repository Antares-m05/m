import os
import numpy as np
from scipy.io import loadmat
from PIL import Image
import matplotlib.pyplot as plt

# ========== 配置路径 ==========
source_dir = r"E:\yan\新科楼顶数据集\切片16x16\train"
rgb_out_dir = r"E:\yan\新科楼顶数据集\切片16x16\rgb_exp_train"
gray_out_dir = r"E:\yan\新科楼顶数据集\切片16x16\gray_exp_train"

# 创建输出根目录（如果不存在）
os.makedirs(rgb_out_dir, exist_ok=True)
os.makedirs(gray_out_dir, exist_ok=True)

# 定义使用的颜色映射（可根据喜好更换，如 'jet', 'plasma' 等）
cmap = plt.get_cmap('viridis')

# ========== 遍历源文件夹 ==========
for root, dirs, files in os.walk(source_dir):
    # 计算当前文件夹相对于源目录的相对路径
    rel_path = os.path.relpath(root, source_dir)
    if rel_path == '.':
        rel_path = ''

    for file in files:
        if file.lower().endswith('.mat'):
            mat_path = os.path.join(root, file)

            # ---------- 读取 .mat 文件 ----------
            try:
                mat_data = loadmat(mat_path)
            except Exception as e:
                print(f"读取失败 {mat_path}: {e}")
                continue

            # 提取复数矩阵（忽略 __header__, __version__, __globals__ 等元数据）
            var_names = [k for k in mat_data.keys() if not k.startswith('__')]
            if not var_names:
                print(f"文件中无有效变量: {mat_path}")
                continue
            # 取第一个变量作为目标矩阵
            matrix = mat_data[var_names[0]]
            # 确保矩阵是复数且形状为 (16,16)
            if not np.iscomplexobj(matrix) or matrix.shape != (16, 16):
                print(f"跳过非复数或非16x16矩阵: {mat_path}")
                continue

            # ---------- 计算幅度并归一化 ----------
            amplitude = np.exp(np.log10(np.abs(matrix)))                 # 幅度
            # 线性归一化到 [0, 255]（可改用百分比截断）
            amp_min, amp_max = amplitude.min(), amplitude.max()
            if amp_max - amp_min < 1e-12:                  # 避免全零情况
                amp_norm = np.zeros_like(amplitude)
            else:
                amp_norm = (amplitude - amp_min) / (amp_max - amp_min) * 255.0
            amp_uint8 = amp_norm.astype(np.uint8)

            # ---------- 准备输出路径 ----------
            # 将文件名后缀改为 .png
            base_name = os.path.splitext(file)[0] + '.png'

            # 灰度图输出路径
            gray_sub_dir = os.path.join(gray_out_dir, rel_path)
            os.makedirs(gray_sub_dir, exist_ok=True)
            gray_path = os.path.join(gray_sub_dir, base_name)

            # 彩色图输出路径
            rgb_sub_dir = os.path.join(rgb_out_dir, rel_path)
            os.makedirs(rgb_sub_dir, exist_ok=True)
            rgb_path = os.path.join(rgb_sub_dir, base_name)

            # ---------- 保存灰度图 ----------
            gray_img = Image.fromarray(amp_uint8, mode='L')  # 'L' 表示灰度
            gray_img.save(gray_path)

            # ---------- 生成伪彩色图并保存 ----------
            # 将灰度值转换到 [0,1] 作为颜色映射的输入
            amp_normalized = amp_uint8 / 255.0
            # 应用颜色映射，得到 RGBA 数组 (16,16,4)
            rgba = cmap(amp_normalized)
            # 转换为 RGB (丢弃 Alpha 通道) 并缩放至 0-255
            rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
            rgb_img = Image.fromarray(rgb, mode='RGB')
            rgb_img.save(rgb_path)

            print(f"已处理: {mat_path} -> {gray_path} & {rgb_path}")

print("全部处理完成！")