import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button
plt.rcParams['font.sans-serif'] = ['SimHei']        # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False          # 解决负号 '-' 显示为方块的问题
class MatCropApp:
    def __init__(self, root_dir, output_dir, crop_size=50):
        """
        初始化裁剪应用
        :param root_dir: 包含 .mat 文件的根目录
        :param output_dir: 裁剪后保存的目录
        :param crop_size: 裁剪边长
        """
        self.root_dir = root_dir
        self.output_dir = output_dir
        self.crop_size = crop_size
        self.file_list = []          # 所有 .mat 文件路径
        self.current_idx = 0          # 当前文件索引
        self.data = None               # 当前复数矩阵 (numpy array)
        self.magnitude = None          # 幅度图像
        self.rows = self.cols = 0
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.im = None                 # imshow 对象
        self.rect = None                # 红色矩形框
        self.rect_visible = False       # 矩形是否有效（在边界内）
        self.status_text = None         # 状态显示文本

        # 收集所有 .mat 文件
        self._collect_files()
        if not self.file_list:
            print("错误：未找到任何 .mat 文件！")
            return

        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)

        # 加载第一个文件
        self._load_current_file()

        # 连接事件
        self._connect_events()

        # 显示图像
        plt.tight_layout()
        plt.show()

    def _collect_files(self):
        """遍历根目录，收集所有 .mat 文件（不递归子目录）"""
        for f in os.listdir(self.root_dir):
            if f.endswith('.mat'):
                self.file_list.append(os.path.join(self.root_dir, f))
        # 按文件名排序，便于顺序处理
        self.file_list.sort()

    def _extract_complex_matrix(self, mat_dict):
        """
        从 MATLAB 字典中提取第一个复数矩阵
        忽略 __header__, __version__, __globals__ 等元数据
        """
        for key, value in mat_dict.items():
            if key.startswith('__') and key.endswith('__'):
                continue
            if isinstance(value, np.ndarray) and np.iscomplexobj(value):
                return value
        # 如果没有找到复数矩阵，尝试返回第一个数值数组（可能是实数）
        for key, value in mat_dict.items():
            if key.startswith('__') and key.endswith('__'):
                continue
            if isinstance(value, np.ndarray) and value.dtype.kind in 'fc':
                return value
        raise ValueError("文件中未找到复数或浮点矩阵")

    def _load_current_file(self):
        file_path = self.file_list[self.current_idx]
        try:
            mat = sio.loadmat(file_path)
            self.full_data = self._extract_complex_matrix(mat)  # 保存完整矩阵
            self.rows, self.cols = self.full_data.shape

            # 定义要显示的区域
            self.row_start = 1250 - 150
            self.row_end = 1250 + 150
            self.col_start = 0
            self.col_end = 350

            # 确保不越界
            self.row_start = max(0, self.row_start)
            self.row_end = min(self.rows, self.row_end)
            self.col_start = max(0, self.col_start)
            self.col_end = min(self.cols, self.col_end)

            # 提取显示用的子矩阵
            self.display_data = self.full_data[self.row_start:self.row_end,
                                self.col_start:self.col_end]
            self.magnitude = np.abs(self.display_data)
            self.display_rows, self.display_cols = self.display_data.shape

        except Exception as e:
            print(f"加载文件 {file_path} 失败: {e}")
            self._next_file()
            return

        # 清除旧图像
        self.ax.clear()
        self.im = self.ax.imshow(self.magnitude, origin='upper',
                                 extent=[self.col_start, self.col_end,
                                         self.row_end, self.row_start])  # 关键：设置正确的坐标范围
        self.ax.set_title(
            f"文件: {os.path.basename(file_path)}  ({self.rows}x{self.cols})   [n:下一个, p:上一个, 单击保存]")

        # 矩形处理
        if self.rect is None:
            self.rect = patches.Rectangle((0, 0), self.crop_size, self.crop_size,
                                          linewidth=2, edgecolor='red', facecolor='none', visible=False)
            self.ax.add_patch(self.rect)
        else:
            self.ax.add_patch(self.rect)

        # 设置默认矩形位置
        center_x = (self.col_start + self.col_end) // 2
        center_y = (self.row_start + self.row_end) // 2
        x_left = center_x - self.crop_size // 2
        y_top = center_y - self.crop_size // 2

        if (x_left >= 0 and y_top >= 0 and
                x_left + self.crop_size <= self.cols and
                y_top + self.crop_size <= self.rows):
            self.rect.set_xy((x_left, y_top))
            self.rect.set_visible(True)
            self.rect_visible = True
        else:
            self.rect.set_visible(False)
            self.rect_visible = False

        self.fig.canvas.draw_idle()
    def _connect_events(self):
        """连接鼠标和键盘事件"""
        self.fig.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)


    def _on_mouse_move(self, event):
        """鼠标移动：更新红色矩形位置，并检查边界"""
        if event.inaxes != self.ax or self.full_data is None:  # 改为 full_data
            if self.rect is not None:
                self.rect.set_visible(False)
                self.fig.canvas.draw_idle()
            return

        # 由于设置了 extent，event.xdata 和 event.ydata 已经是完整矩阵的坐标
        x_center = int(round(event.xdata))
        y_center = int(round(event.ydata))
        x_left = x_center - self.crop_size // 2
        y_top = y_center - self.crop_size // 2

        # 边界检查（使用完整矩阵的尺寸）
        if (x_left >= 0 and y_top >= 0 and
                x_left + self.crop_size <= self.cols and
                y_top + self.crop_size <= self.rows):
            self.rect.set_xy((x_left, y_top))
            self.rect.set_visible(True)
            self.rect.set_edgecolor('red')
            self.rect_visible = True
        else:
            self.rect.set_visible(False)
            self.rect_visible = False

        self.fig.canvas.draw_idle()

    def _on_click(self, event):
        """鼠标单击：保存当前矩形对应的复数子矩阵"""
        if event.inaxes != self.ax or event.button != 1:
            return
        if not self.rect_visible or self.rect is None:
            print("错误：当前区域超出图像边界，无法保存！")
            return

        # 获取矩形左上角坐标（已经是完整矩阵的坐标）
        x_left, y_top = self.rect.get_xy()
        x_left = int(round(x_left))
        y_top = int(round(y_top))

        # 从完整矩阵提取子矩阵
        sub_matrix = self.full_data[y_top:y_top + self.crop_size,
                     x_left:x_left + self.crop_size]

        # 生成保存文件名
        base_name = os.path.basename(self.file_list[self.current_idx])
        name_without_ext = os.path.splitext(base_name)[0]
        save_name = f"{name_without_ext}_y{y_top}_x{x_left}.mat"
        save_path = os.path.join(self.output_dir, save_name)

        sio.savemat(save_path, {'data': sub_matrix})
        print(f"已保存: {save_path}")
    def _on_key_press(self, event):
        """键盘事件：n/p 切换文件"""
        if event.key == 'n':
            self._next_file()
        elif event.key == 'p':
            self._prev_file()

    def _next_file(self):
        """切换到下一个文件"""
        if self.current_idx + 1 < len(self.file_list):
            self.current_idx += 1
            self._load_current_file()
        else:
            print("已经是最后一个文件了。")

    def _prev_file(self):
        """切换到上一个文件"""
        if self.current_idx - 1 >= 0:
            self.current_idx -= 1
            self._load_current_file()
        else:
            print("已经是第一个文件了。")

# 使用示例
if __name__ == "__main__":
    root = r"E:\yan\新科楼顶数据集\phantom4"
    out = r"E:\yan\新科楼顶数据集\切片16x16\2"
    app = MatCropApp(root, out, crop_size=16)