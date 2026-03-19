import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import time

###适用于一个mat里多张图
class MatrixCropper:
    def __init__(self, mat_path, output_dir, crop_size=(200, 51)):
        """
        初始化矩阵裁切器

        参数:
            mat_path: mat文件路径
            output_dir: 输出目录
            crop_size: 裁剪框大小 (height, width)
        """
        self.mat_path = mat_path
        self.output_dir = Path(output_dir)
        self.crop_size = crop_size  # (height, width)

        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 加载数据
        self.load_data()

        # 初始化变量
        self.current_index = 0
        self.crop_rect = None
        self.fig = None
        self.ax = None
        self.rect_patch = None
        self.crop_count = 0  # 裁剪计数器

    def load_data(self):
        """加载mat文件数据"""
        print(f"正在加载数据: {self.mat_path}")
        data = sio.loadmat(self.mat_path)

        # 查找包含数据的主要变量
        # 通常mat文件会包含多个变量，我们寻找最大的那个
        data_keys = [key for key in data.keys() if not key.startswith('__')]

        if not data_keys:
            raise ValueError("未在mat文件中找到有效数据")

        # 选择第一个非元数据的变量
        main_key = data_keys[0]
        self.data = data[main_key]

        print(f"数据形状: {self.data.shape}")
        print(f"数据类型: {self.data.dtype}")

        # 确保数据是3D的 (n, m, b)
        if len(self.data.shape) != 3:
            raise ValueError(f"期望3D数据 (n, m, b)，但得到形状: {self.data.shape}")

        self.n, self.m, self.b = self.data.shape
        print(f"共 {self.b} 个矩阵，每个大小 {self.n}×{self.m}")

    def on_mouse_move(self, event):
        """鼠标移动事件处理"""
        if event.inaxes != self.ax:
            return

        # 获取鼠标位置
        x, y = event.xdata, event.ydata

        if x is not None and y is not None:
            # 计算裁剪框的边界
            # 确保裁剪框在图像范围内
            crop_width, crop_height = self.crop_size[1], self.crop_size[0]
            half_width = crop_width / 2
            half_height = crop_height / 2

            left = max(0, min(x - half_width, self.m - crop_width))
            right = left + crop_width
            bottom = max(0, min(y - half_height, self.n - crop_height))
            top = bottom + crop_height

            # 更新或创建红色矩形框
            if self.rect_patch is None:
                self.rect_patch = plt.Rectangle(
                    (left, bottom), crop_width, crop_height,
                    linewidth=2, edgecolor='r', facecolor='none'
                )
                self.ax.add_patch(self.rect_patch)
            else:
                self.rect_patch.set_xy((left, bottom))

            # 存储当前裁剪框位置
            self.crop_rect = {
                'left': int(round(left)),
                'right': int(round(right)),
                'bottom': int(round(bottom)),
                'top': int(round(top)),
                'x': int(round(left)),
                'y': int(round(bottom))
            }

            self.fig.canvas.draw_idle()

    def on_mouse_click(self, event):
        """鼠标点击事件处理"""
        if event.inaxes != self.ax:
            return

        if event.button == 1 and self.crop_rect is not None:  # 左键点击
            self.crop_and_save()

    def on_key_press(self, event):
        """键盘按键事件处理"""
        if event.key == 'n' or event.key == 'N':
            self.next_matrix()
        elif event.key == 'p' or event.key == 'P':
            self.prev_matrix()
        elif event.key == 'q' or event.key == 'Q':
            plt.close('all')
            print("程序已退出")
        elif event.key == 's' or event.key == 'S':
            # 保存当前视图为图像
            self.save_current_view()
        elif event.key == 'c' or event.key == 'C':
            # 手动调用裁剪保存
            if self.crop_rect is not None:
                self.crop_and_save()

    def save_current_view(self):
        """保存当前视图为图像"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = self.output_dir / f"matrix_{self.current_index + 1:03d}_view_{timestamp}.png"
        self.fig.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"已保存当前视图: {filename}")

    def crop_and_save(self):
        """执行裁剪并保存为MAT文件"""
        # 获取当前矩阵
        matrix = self.data[:, :, self.current_index]

        # 提取裁剪区域
        crop_region = matrix[
                      self.crop_rect['bottom']:self.crop_rect['top'],
                      self.crop_rect['left']:self.crop_rect['right']
                      ]

        # 确保裁剪区域大小正确
        if crop_region.shape != self.crop_size:
            print(f"警告: 裁剪区域大小 {crop_region.shape} 与预期 {self.crop_size} 不符")
            return

        # 增加裁剪计数器
        self.crop_count += 1

        # 创建文件名
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = self.output_dir / f"matrix_{self.current_index + 1:03d}_crop_{timestamp}.mat"

        # 准备保存的数据
        save_data = {
            'cropped_matrix': crop_region,
            'original_index': self.current_index + 1,  # 1-based索引
            'crop_position': self.crop_rect,
            'original_size': (self.n, self.m),
            'crop_size': self.crop_size,
            'timestamp': timestamp
        }

        # 保存为MAT文件
        sio.savemat(filename, save_data)

        print(f"已保存MAT文件: {filename}")
        print(f"  原始矩阵: {self.current_index + 1}/{self.b}")
        print(f"  裁剪位置: 左上角({self.crop_rect['x']}, {self.crop_rect['y']})")
        print(f"  裁剪大小: {crop_region.shape}")

        # 显示保存成功信息
        self.ax.set_title(f"Matrix {self.current_index + 1}/{self.b} - 已保存裁剪区域 (总计: {self.crop_count})",
                          color='green', fontsize=12)
        self.fig.canvas.draw_idle()

    def next_matrix(self):
        """显示下一个矩阵"""
        if self.current_index < self.b - 1:
            self.current_index += 1
            self.update_display()

    def prev_matrix(self):
        """显示上一个矩阵"""
        if self.current_index > 0:
            self.current_index -= 1
            self.update_display()

    def update_display(self):
        """更新显示"""
        self.ax.clear()
        matrix = self.data[:, :, self.current_index]

        # 显示当前矩阵
        im = self.ax.imshow(matrix, cmap='viridis', aspect='auto')
        #plt.colorbar(im, ax=self.ax)

        self.ax.set_title(f"Matrix {self.current_index + 1}/{self.b} (n={self.n}, m={self.m})", fontsize=14)
        self.ax.set_xlabel("列", fontsize=12)
        self.ax.set_ylabel("行", fontsize=12)

        # 显示网格线
        self.ax.grid(True, alpha=0.3, linestyle='--')

        # 重置裁剪框
        self.rect_patch = None
        self.crop_rect = None

        self.fig.canvas.draw_idle()

    def batch_process(self, x_positions=None, y_positions=None):
        """
        批量处理：在指定位置自动裁剪所有矩阵

        参数:
            x_positions: x坐标列表 (列位置)
            y_positions: y坐标列表 (行位置)
        """
        if x_positions is None or y_positions is None:
            print("未指定位置，请提供x_positions和y_positions参数")
            return

        # 确保位置列表长度一致
        if len(x_positions) != len(y_positions):
            print("错误: x_positions和y_positions长度不一致")
            return

        print(f"开始批量处理 {len(x_positions)} 个位置...")

        for pos_idx, (x, y) in enumerate(zip(x_positions, y_positions), 1):
            for matrix_idx in range(self.b):
                # 获取当前矩阵
                matrix = self.data[:, :, matrix_idx]

                # 计算裁剪框位置
                crop_width, crop_height = self.crop_size[1], self.crop_size[0]
                half_width = crop_width // 2
                half_height = crop_height // 2

                left = max(0, min(x - half_width, self.m - crop_width))
                bottom = max(0, min(y - half_height, self.n - crop_height))

                # 提取裁剪区域
                crop_region = matrix[
                              bottom:bottom + crop_height,
                              left:left + crop_width
                              ]

                # 确保裁剪区域大小正确
                if crop_region.shape == self.crop_size:
                    # 准备保存的数据
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = self.output_dir / f"batch_matrix_{matrix_idx + 1:03d}_pos{pos_idx}_{timestamp}.mat"

                    save_data = {
                        'cropped_matrix': crop_region,
                        'original_index': matrix_idx + 1,
                        'crop_position': {
                            'x': left,
                            'y': bottom,
                            'width': crop_width,
                            'height': crop_height
                        },
                        'original_size': (self.n, self.m),
                        'batch_position_index': pos_idx,
                        'timestamp': timestamp
                    }

                    # 保存为MAT文件
                    sio.savemat(filename, save_data)

                    print(f"  已保存: {filename}")

        print(f"批量处理完成! 共保存 {self.b * len(x_positions)} 个MAT文件")

    def run(self):
        """运行主程序"""
        print("=" * 60)
        print("矩阵裁剪工具 (保存为MAT文件)")
        print("=" * 60)
        print("操作说明:")
        print("  1. 移动鼠标 - 显示红色裁剪框 (200×51)")
        print("  2. 左键点击 - 裁剪并保存当前区域为MAT文件")
        print("  3. N键 - 下一个矩阵")
        print("  4. P键 - 上一个矩阵")
        print("  5. C键 - 手动触发裁剪保存")
        print("  6. S键 - 保存当前视图为PNG图像")
        print("  7. Q键 - 退出程序")
        print("=" * 60)

        # 创建图形窗口
        self.fig, self.ax = plt.subplots(figsize=(12, 8))

        # 显示第一个矩阵
        self.update_display()

        # 连接事件
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.fig.canvas.mpl_connect('button_press_event', self.on_mouse_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

        plt.tight_layout()
        plt.show()

        print(f"\n所有裁剪文件已保存到: {self.output_dir}")
        print(f"总计裁剪了 {self.crop_count} 个区域")


def main():
    # 设置路径
    mat_path = r"E:\yan\rd\种类识别\4tgts\无人机分类\吴家桥\rd-14.mat"  #14 good
    output_dir = r"E:\yan\rd\种类识别\4tgts\新吴家桥切片"

    # 设置裁剪大小 (200×51)
    crop_size = (200, 51)  # (height, width)

    try:
        # 创建并运行裁剪器
        cropper = MatrixCropper(mat_path, output_dir, crop_size)

        # 询问是否要批量处理
        response = input("是否要进行批量处理? (y/n): ").lower()
        if response == 'y':
            # 获取批量处理的位置
            print("请输入批量处理的位置 (格式: x1,y1 x2,y2 ...): ")
            positions_input = input("例如: 100,200 150,300 200,400: ")

            positions = positions_input.split()
            x_positions = []
            y_positions = []

            for pos in positions:
                try:
                    x, y = map(int, pos.split(','))
                    x_positions.append(x)
                    y_positions.append(y)
                except:
                    print(f"跳过无效位置: {pos}")

            if x_positions and y_positions:
                cropper.batch_process(x_positions, y_positions)
            else:
                print("未提供有效位置，进入交互模式")
                cropper.run()
        else:
            cropper.run()

    except FileNotFoundError:
        print(f"错误: 找不到文件 {mat_path}")
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()