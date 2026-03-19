import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

src_root = r"E:\yan\rd\种类识别\test"
dst_root = r"E:\yan\rd\种类识别\png_29"

def process_matrix(mat):
    x = mat
    x = np.exp(x / 10)
    return x

for root, dirs, files in os.walk(src_root):
    for file in files:
        if file.endswith(".mat"):
            mat_path = os.path.join(root, file)

            rel_path = os.path.relpath(root, src_root)
            save_dir = os.path.join(dst_root, rel_path)
            os.makedirs(save_dir, exist_ok=True)

            data = sio.loadmat(mat_path)

            matrix = None
            for k, v in data.items():
                if not k.startswith("__") and isinstance(v, np.ndarray) and v.ndim == 2:
                    matrix = v
                    break

            if matrix is None:
                continue

            processed = process_matrix(matrix)

            # ================= 关键部分 =================
            h, w = processed.shape  # 应为 200, 51
            dpi = 100

            fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
            ax = plt.Axes(fig, [0., 0., 1., 1.])  # 填满整个画布
            ax.set_axis_off()
            fig.add_axes(ax)

            ax.imshow(processed, cmap="jet", aspect="auto")

            png_name = os.path.splitext(file)[0] + ".png"
            png_path = os.path.join(save_dir, png_name)

            fig.savefig(png_path, dpi=dpi)
            plt.close(fig)
            # ===========================================

            print(f"保存完成: {png_path}")
